# src/agent/strategies/statistical_arbitrage_strategies.py
import pandas as pd
import numpy as np
import ta
from ta.utils import dropna
from .base_strategy import BaseStrategy, StrategyConfig
from . import register_strategy
from typing import Dict, List, Optional, Any, Tuple 
import logging
import sys
import torch
import torch.nn.functional as F

# Attempt to import STRATEGY_REGISTRY
try:
    from ..strategies import STRATEGY_REGISTRY
except ImportError:
    # Fallback for cases where the script might be run in a context where relative import fails
    # This is less ideal and might indicate a structural issue if it happens during normal operation
    STRATEGY_REGISTRY = {} 
    logging.getLogger(__name__).warning("Failed to import STRATEGY_REGISTRY via relative import. StatisticalArbitrageStrategy may not find sub-strategies.")


@register_strategy("MeanReversionStrategy")
class MeanReversionStrategy(BaseStrategy):

    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="MeanReversionStrategy",
            description="Trades based on mean reversion principles using Bollinger Bands.",
            default_params={'bb_period': 20, 'bb_std_dev': 2.0, 'asset_list': [], 'close_idx': 0} 
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config=config, params=params, logger=logger)
        self.CLOSE_IDX = self.params.get('close_idx', 0)
        if self.config.input_dim is not None:
            self.logger.info(f"[{self.config.name}] Initialized with input_dim: {self.config.input_dim}. Close index used: {self.CLOSE_IDX}.")
        else:
            self.logger.info(f"[{self.config.name}] Initialized. Close index used: {self.CLOSE_IDX}. input_dim not specified in config.")

    def _rolling_mean(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """ Helper for rolling mean using conv1d. Assumes tensor is (batch, seq_len) or (batch, 1, seq_len). """
        if tensor.ndim == 2: # (batch, seq_len)
            tensor_unsqueezed = tensor.unsqueeze(1) # (batch, 1, seq_len)
        elif tensor.ndim == 3 and tensor.shape[1] == 1: # (batch, 1, seq_len)
            tensor_unsqueezed = tensor
        else:
            self.logger.error(f"[{self.config.name}] _rolling_mean expects 2D (batch, seq_len) or 3D (batch, 1, seq_len) tensor, got {tensor.shape}")
            # Return a tensor of zeros that matches the expected output shape if input was (batch, seq_len)
            return torch.zeros((tensor.shape[0], tensor.shape[-1]), device=tensor.device, dtype=tensor.dtype) if tensor.ndim == 2 else torch.zeros_like(tensor)


        if tensor_unsqueezed.shape[-1] < window_size: # sequence_length < window_size
             # Pad with first value to effectively compute mean over available data for initial points
            padding = window_size - tensor_unsqueezed.shape[-1]
            padded_tensor = F.pad(tensor_unsqueezed, (padding, 0), mode='replicate') # Pad at the beginning
        else:
            padding = window_size - 1
            padded_tensor = F.pad(tensor_unsqueezed, (padding, 0), mode='replicate')

        sma_weights = torch.full((1, 1, window_size), 1.0/window_size, device=tensor.device, dtype=tensor.dtype)
        pooled = F.conv1d(padded_tensor, sma_weights, stride=1)
        
        if tensor.ndim == 2:
            return pooled.squeeze(1) # (batch, seq_len)
        return pooled # (batch, 1, seq_len)

    def _rolling_std(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """ Helper for rolling std. Assumes tensor is (batch, seq_len) or (batch, 1, seq_len). """
        mean_x = self._rolling_mean(tensor, window_size)
        
        # Handle tensor**2 correctly based on original tensor shape
        if tensor.ndim == 2: # Input was (batch, seq_len)
            mean_x_sq = self._rolling_mean(tensor**2, window_size) # tensor**2 is (batch, seq_len)
        elif tensor.ndim == 3 and tensor.shape[1] == 1: # Input was (batch, 1, seq_len)
            mean_x_sq = self._rolling_mean(tensor**2, window_size) # tensor**2 is (batch, 1, seq_len)
        else: # Should not happen if _rolling_mean handled shapes correctly
            self.logger.error(f"[{self.config.name}] Unexpected tensor shape in _rolling_std after mean_x calculation. Tensor shape: {tensor.shape}, mean_x shape: {mean_x.shape}")
            return torch.zeros_like(tensor)

        # Ensure mean_x and mean_x_sq have compatible shapes for subtraction
        if mean_x.shape != mean_x_sq.shape:
            self.logger.error(f"[{self.config.name}] Shape mismatch between mean_x ({mean_x.shape}) and mean_x_sq ({mean_x_sq.shape}) in _rolling_std.")
            # Attempt to align if one is (B,S) and other is (B,1,S) due to intermediate steps, though ideally _rolling_mean is consistent
            if mean_x.ndim == 2 and mean_x_sq.ndim == 3 and mean_x_sq.shape[1] == 1:
                mean_x = mean_x.unsqueeze(1)
            elif mean_x_sq.ndim == 2 and mean_x.ndim == 3 and mean_x.shape[1] == 1:
                mean_x_sq = mean_x_sq.unsqueeze(1)
            else: # Fallback
                 return torch.zeros_like(tensor)


        variance = (mean_x_sq - mean_x**2).clamp(min=1e-9) # clamp for numerical stability
        return torch.sqrt(variance)

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        """
        Processes tensor features for a single asset to generate mean reversion signals.
        asset_features: (batch_size, sequence_length, num_features)
                        Expected feature order: self.CLOSE_IDX for close price
        Returns: (batch_size, 1, 1) signal tensor.
        """
        batch_size, sequence_length, num_features = asset_features.shape
        device = asset_features.device

        bb_period = self.params.get('bb_period', 20)
        bb_std_dev = self.params.get('bb_std_dev', 2.0)

        if self.CLOSE_IDX >= num_features:
            self.logger.error(f"[{self.config.name}] close_idx {self.CLOSE_IDX} is out of bounds for num_features {num_features}. Input dim from config: {self.config.input_dim}. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        if sequence_length < bb_period:
            self.logger.warning(f"[{self.config.name}] Sequence length ({sequence_length}) is less than bb_period ({bb_period}). Bollinger Bands may be unreliable. Proceeding, but consider longer sequence or shorter period.")
            # Fallback: if seq_len is too short for reliable BB, could return neutral or use simpler logic.
            # For now, we let _rolling_mean/_rolling_std handle it (they might pad or produce NaNs/zeros if not careful)
            # The current _rolling_mean pads, so it will produce values.

        close_prices = asset_features[:, :, self.CLOSE_IDX]  # (batch_size, sequence_length)

        sma_mid_band = self._rolling_mean(close_prices, bb_period) # (batch_size, sequence_length)
        rolling_std = self._rolling_std(close_prices, bb_period)   # (batch_size, sequence_length)
        
        upper_band = sma_mid_band + (rolling_std * bb_std_dev)
        lower_band = sma_mid_band - (rolling_std * bb_std_dev)

        last_close = close_prices[:, -1]    # (batch_size)
        last_upper_band = upper_band[:, -1] # (batch_size)
        last_lower_band = lower_band[:, -1] # (batch_size)
        
        signal = torch.zeros(batch_size, device=device)
        signal[last_close < last_lower_band] = 1.0
        signal[last_close > last_upper_band] = -1.0
        
        return signal.view(batch_size, 1, 1)

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        self.logger.info(f"[{self.config.name}] generate_signals (legacy) called.")
        output_index = None
        if processed_data_dict:
            for key, df_val in processed_data_dict.items():
                if isinstance(df_val, pd.DataFrame) and not df_val.empty:
                    output_index = df_val.index
                    self.logger.debug(f"Using index from processed_data_dict key '{key}' for generate_signals output.")
                    break
        if output_index is None:
            self.logger.warning(f"[{self.config.name}] No valid index found in processed_data_dict for generate_signals. Returning empty DataFrame with 'signal' column.")
            return pd.DataFrame(columns=['signal'])
        signals_df = pd.DataFrame(0.0, index=output_index, columns=['signal'])
        return signals_df

@register_strategy("CointegrationStrategy")
class CointegrationStrategy(BaseStrategy):

    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="CointegrationStrategy",
            description="Trades based on cointegration of an asset pair.",
            default_params={'asset_pair': [], 'window': 60, 'z_threshold': 2.0, 'asset1_close_idx': 0, 'asset2_close_idx': None} 
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config=config, params=params, logger=logger)
        self.asset_pair_names = self.params.get('asset_pair_names', self.params.get('asset_pair', [])) # Support both
        self.window = self.params.get('window', 60)
        self.z_threshold = self.params.get('z_threshold', 2.0)
        self.asset1_close_idx = self.params.get('asset1_close_idx', 0)
        self.asset2_close_idx = self.params.get('asset2_close_idx') 
        
        if self.config.input_dim is not None:
            self.logger.info(f"[{self.config.name}] Initialized with input_dim (total features for pair): {self.config.input_dim}. Asset1 Close Idx: {self.asset1_close_idx}, Asset2 Close Idx: {self.asset2_close_idx}.")
            if self.asset2_close_idx is None:
                 self.logger.warning(f"[{self.config.name}] asset2_close_idx is None. This strategy requires it to be configured correctly based on concatenated pair features.")
            elif self.asset1_close_idx >= self.config.input_dim or self.asset2_close_idx >= self.config.input_dim:
                 self.logger.warning(f"[{self.config.name}] One or both close indices ({self.asset1_close_idx}, {self.asset2_close_idx}) might be out of bounds for input_dim {self.config.input_dim}.")
        else:
            self.logger.info(f"[{self.config.name}] Initialized. Asset1 Close Idx: {self.asset1_close_idx}, Asset2 Close Idx: {self.asset2_close_idx}. input_dim not specified in config.")

        if len(self.asset_pair_names) != 2 or not all(isinstance(p, str) for p in self.asset_pair_names):
            self.logger.warning(f"[{self.config.name}] Requires 'asset_pair_names' to be a list of two asset name strings. Current: {self.asset_pair_names}.")
        # ... (rest of __init__ unchanged)

    # Using MeanReversionStrategy's rolling helpers by composition or re-implementation
    # For simplicity, copied here. Ideally, these would be in a shared utility module.
    def _rolling_mean(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        # (Identical to MeanReversionStrategy._rolling_mean)
        if tensor.ndim == 2: tensor_unsqueezed = tensor.unsqueeze(1)
        elif tensor.ndim == 3 and tensor.shape[1] == 1: tensor_unsqueezed = tensor
        else: self.logger.error(f"[{self.config.name}] _rolling_mean error"); return torch.zeros_like(tensor)
        if tensor_unsqueezed.shape[-1] < window_size:
            padding = window_size - tensor_unsqueezed.shape[-1]
            padded_tensor = F.pad(tensor_unsqueezed, (padding, 0), mode='replicate')
        else:
            padding = window_size - 1
            padded_tensor = F.pad(tensor_unsqueezed, (padding, 0), mode='replicate')
        sma_weights = torch.full((1, 1, window_size), 1.0/window_size, device=tensor.device, dtype=tensor.dtype)
        pooled = F.conv1d(padded_tensor, sma_weights, stride=1)
        return pooled.squeeze(1) if tensor.ndim == 2 else pooled

    def _rolling_std(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        # (Identical to MeanReversionStrategy._rolling_std, with safety for shape)
        mean_x = self._rolling_mean(tensor, window_size)
        if tensor.ndim == 2: mean_x_sq = self._rolling_mean(tensor**2, window_size)
        elif tensor.ndim == 3 and tensor.shape[1] == 1: mean_x_sq = self._rolling_mean(tensor**2, window_size)
        else: self.logger.error(f"[{self.config.name}] _rolling_std error"); return torch.zeros_like(tensor)
        if mean_x.shape != mean_x_sq.shape: # Ensure alignment
            if mean_x.ndim == 2 and mean_x_sq.ndim == 3 and mean_x_sq.shape[1] == 1: mean_x = mean_x.unsqueeze(1)
            elif mean_x_sq.ndim == 2 and mean_x.ndim == 3 and mean_x.shape[1] == 1: mean_x_sq = mean_x_sq.unsqueeze(1)
            else: self.logger.error(f"[{self.config.name}] Shape mismatch in _rolling_std"); return torch.zeros_like(tensor)
        variance = (mean_x_sq - mean_x**2).clamp(min=1e-9)
        return torch.sqrt(variance)

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, sequence_length, num_total_features = asset_features.shape
        device = asset_features.device

        if self.asset2_close_idx is None:
            self.logger.error(f"[{self.config.name}] asset2_close_idx is not configured. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        if not (0 <= self.asset1_close_idx < num_total_features and 0 <= self.asset2_close_idx < num_total_features):
            self.logger.error(f"[{self.config.name}] One or both close_idx ({self.asset1_close_idx}, {self.asset2_close_idx}) are out of bounds for num_total_features {num_total_features}. Config input_dim: {self.config.input_dim}. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)
        
        if sequence_length < self.window:
            self.logger.warning(f"[{self.config.name}] Sequence length ({sequence_length}) < window ({self.window}). Results may be unreliable. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        close_prices_asset1 = asset_features[:, :, self.asset1_close_idx]
        close_prices_asset2 = asset_features[:, :, self.asset2_close_idx]
        spread = close_prices_asset1 - close_prices_asset2

        spread_mean = self._rolling_mean(spread, self.window)
        spread_std = self._rolling_std(spread, self.window)
        
        # Ensure spread_std is not zero to avoid division by zero
        safe_spread_std = spread_std.clamp(min=1e-9)
        z_score = (spread - spread_mean) / safe_spread_std

        last_z_score = z_score[:, -1]
        
        signal = torch.zeros(batch_size, device=device)
        signal[last_z_score < -self.z_threshold] = 1.0
        signal[last_z_score > self.z_threshold] = -1.0
        
        return signal.view(batch_size, 1, 1)

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        self.logger.info(f"[{self.config.name}] generate_signals (legacy) called.")
        output_index = None
        if processed_data_dict:
            for key, df_val in processed_data_dict.items():
                if isinstance(df_val, pd.DataFrame) and not df_val.empty:
                    output_index = df_val.index
                    break
        if output_index is None: return pd.DataFrame(columns=['signal'])
        return pd.DataFrame(0.0, index=output_index, columns=['signal'])

@register_strategy("PairsTradeStrategy")
class PairsTradeStrategy(BaseStrategy):
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="PairsTradeStrategy",
            description="Trades based on the spread of an asset pair using z-scores with entry/exit thresholds.",
            default_params={
                'asset_pair': [], 'window': 60, 'entry_threshold': 2.0, 'exit_threshold': 0.5,
                'asset1_close_idx': 0, 'asset2_close_idx': None
            }
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config=config, params=params, logger=logger)
        self.asset_pair_names = self.params.get('asset_pair_names', self.params.get('asset_pair', [])) # Support both
        self.window = self.params.get('window', 60)
        self.entry_threshold = self.params.get('entry_threshold', 2.0)
        self.exit_threshold = self.params.get('exit_threshold', 0.5)
        self.asset1_close_idx = self.params.get('asset1_close_idx', 0)
        self.asset2_close_idx = self.params.get('asset2_close_idx')

        if self.config.input_dim is not None:
            self.logger.info(f"[{self.config.name}] Initialized with input_dim (total features for pair): {self.config.input_dim}. Asset1 Idx: {self.asset1_close_idx}, Asset2 Idx: {self.asset2_close_idx}.")
            if self.asset2_close_idx is None:
                 self.logger.warning(f"[{self.config.name}] asset2_close_idx is None.")
            elif self.asset1_close_idx >= self.config.input_dim or self.asset2_close_idx >= self.config.input_dim:
                 self.logger.warning(f"[{self.config.name}] Close indices ({self.asset1_close_idx}, {self.asset2_close_idx}) might be OOB for input_dim {self.config.input_dim}.")
        else:
            self.logger.info(f"[{self.config.name}] Initialized. Asset1 Idx: {self.asset1_close_idx}, Asset2 Idx: {self.asset2_close_idx}. input_dim not in config.")
        # ... (rest of __init__ unchanged)

    # Using MeanReversionStrategy's rolling helpers
    _rolling_mean = MeanReversionStrategy._rolling_mean 
    _rolling_std = MeanReversionStrategy._rolling_std
    # Note: This direct assignment works if MeanReversionStrategy is defined above.
    # If these were complex methods relying on MeanReversionStrategy's specific 'self', this could be an issue.
    # However, _rolling_mean and _rolling_std are quite self-contained or use self.logger.
    # To be fully robust, they should ideally be static methods or free functions if shared this way,
    # or defined within a common base or utility. For now, this is a common pattern.

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, sequence_length, num_total_features = asset_features.shape
        device = asset_features.device

        if self.asset2_close_idx is None:
            self.logger.error(f"[{self.config.name}] asset2_close_idx is not configured. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        if not (0 <= self.asset1_close_idx < num_total_features and 0 <= self.asset2_close_idx < num_total_features):
            self.logger.error(f"[{self.config.name}] Close indices OOB. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        if sequence_length < self.window:
            self.logger.warning(f"[{self.config.name}] Seq length ({sequence_length}) < window ({self.window}). Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        close_prices_asset1 = asset_features[:, :, self.asset1_close_idx]
        close_prices_asset2 = asset_features[:, :, self.asset2_close_idx]
        spread = close_prices_asset1 - close_prices_asset2

        spread_mean = self._rolling_mean(self, spread, self.window) # Pass self for logger context
        spread_std = self._rolling_std(self, spread, self.window)   # Pass self for logger context
        safe_spread_std = spread_std.clamp(min=1e-9)
        z_score = (spread - spread_mean) / safe_spread_std
        last_z_score = z_score[:, -1]
        
        current_pos_flat = torch.zeros(batch_size, device=device)
        if current_positions is not None:
            current_pos_flat = current_positions.view(batch_size).clone()
        
        target_position = current_pos_flat.clone()

        can_enter_long = (current_pos_flat == 0) & (last_z_score < -self.entry_threshold)
        target_position[can_enter_long] = 1.0
        can_enter_short = (current_pos_flat == 0) & (last_z_score > self.entry_threshold)
        target_position[can_enter_short] = -1.0
        should_exit_long = (current_pos_flat == 1.0) & (last_z_score >= -self.exit_threshold)
        target_position[should_exit_long] = 0.0
        should_exit_short = (current_pos_flat == -1.0) & (last_z_score <= self.exit_threshold)
        target_position[should_exit_short] = 0.0
        
        return target_position.view(batch_size, 1, 1)

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        self.logger.info(f"[{self.config.name}] generate_signals (legacy) called.")
        output_index = None
        if processed_data_dict:
            for key, df_val in processed_data_dict.items():
                if isinstance(df_val, pd.DataFrame) and not df_val.empty:
                    output_index = df_val.index
                    break
        if output_index is None: return pd.DataFrame(columns=['signal'])
        return pd.DataFrame(0.0, index=output_index, columns=['signal'])

class StatisticalArbitrageStrategy(BaseStrategy):

    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="StatisticalArbitrageStrategy",
            description="A composite strategy that combines multiple statistical arbitrage sub-strategies.",
            default_params={
                'base_strategies_config': [], 
                'combination_logic': 'sum', 
                # 'num_features_per_asset': None, # This will now come from self.config.input_dim
            },
            applicable_assets=[] 
        )
    
    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config=config, params=params, logger=logger)
        
        self.base_strategy_configs_from_params = self.params.get('base_strategies_config', [])
        self.combination_logic = self.params.get('combination_logic', 'sum')
        
        self.num_features_per_asset = self.config.input_dim # D_transformer_out
        if self.num_features_per_asset is None:
            self.logger.critical(f"[{self.config.name}] self.config.input_dim (num_features_per_asset) is NOT set. This is critical for sub-strategy configuration and feature slicing. Defaulting to 1, but this is likely incorrect.")
            self.num_features_per_asset = 1 # A potentially unsafe default, but prevents None errors later.

        # This map is crucial for slicing the composite strategy's input tensor in the forward pass.
        # self.config.applicable_assets should be populated by EnhancedQuantumStrategyLayer
        # with all unique assets this composite strategy instance will handle, in a defined order.
        self.asset_to_composite_idx_map: Dict[str, int] = {
            asset_name: i for i, asset_name in enumerate(self.config.applicable_assets or [])
        }
        if not self.asset_to_composite_idx_map and self.base_strategy_configs_from_params:
            self.logger.warning(f"[{self.config.name}] 'applicable_assets' for composite strategy is empty or None, but sub-strategies are defined. Feature slicing in forward pass will likely fail if not all sub-strategy assets are covered or if map is empty.")
        elif self.asset_to_composite_idx_map:
             self.logger.info(f"[{self.config.name}] Asset to composite index map: {self.asset_to_composite_idx_map}")


        self.sub_strategy_details: List[Dict[str, Any]] = []
        
        # Ensure STRATEGY_REGISTRY is populated (it should be at import time)
        if not STRATEGY_REGISTRY:
             self.logger.error(f"[{self.config.name}] STRATEGY_REGISTRY is empty. Cannot initialize sub-strategies.")
             return

        for sub_config_item in self.base_strategy_configs_from_params:
            strategy_class_name = sub_config_item.get('strategy_class_name')
            sub_strategy_instance_name = sub_config_item.get('name', strategy_class_name) # Default name
            sub_params = sub_config_item.get('params', {})

            if not strategy_class_name:
                self.logger.error(f"[{self.config.name}] Sub-strategy config missing 'strategy_class_name': {sub_config_item}")
                continue

            StrategyClass = STRATEGY_REGISTRY.get(strategy_class_name)
            if not StrategyClass:
                self.logger.error(f"[{self.config.name}] Sub-strategy class '{strategy_class_name}' not found in STRATEGY_REGISTRY.")
                continue

            # Determine assets and input_dim for the sub-strategy
            sub_asset_names: List[str] = []
            sub_strat_input_dim: Optional[int] = None
            final_sub_params = sub_params.copy() # Params to pass to sub-strategy constructor

            # Heuristic to identify pair strategies (could be made more robust, e.g. by checking class inheritance or a property)
            is_pair_strategy = "pair" in strategy_class_name.lower() or \
                               "cointegration" in strategy_class_name.lower() or \
                               hasattr(StrategyClass, 'asset_pair_names') or \
                               ('asset_pair_names' in sub_params or 'asset_pair' in sub_params)


            if is_pair_strategy:
                sub_asset_names = final_sub_params.get('asset_pair_names', final_sub_params.get('asset_pair', []))
                if len(sub_asset_names) != 2:
                    self.logger.error(f"[{self.config.name}] Pair strategy '{sub_strategy_instance_name}' ({strategy_class_name}) requires 'asset_pair_names' with 2 assets. Got: {sub_asset_names}. Skipping.")
                    continue
                if self.num_features_per_asset is not None:
                    sub_strat_input_dim = 2 * self.num_features_per_asset
                    # Adjust close indices for pair strategy:
                    # asset1_close_idx is relative to its own block (0-indexed within its D features)
                    # asset2_close_idx is relative to the start of the concatenated 2D block
                    # (i.e., D + relative_idx_in_second_block)
                    if 'asset1_price_feature_idx' in final_sub_params:
                        final_sub_params['asset1_close_idx'] = final_sub_params.pop('asset1_price_feature_idx')
                    if 'asset2_price_feature_idx' in final_sub_params and self.num_features_per_asset is not None:
                        final_sub_params['asset2_close_idx'] = self.num_features_per_asset + final_sub_params.pop('asset2_price_feature_idx')
                    # If asset1_close_idx or asset2_close_idx are already directly in final_sub_params, they are used as is.
                else: # num_features_per_asset is None
                    self.logger.error(f"[{self.config.name}] Cannot determine input_dim for pair sub-strategy '{sub_strategy_instance_name}' because num_features_per_asset is unknown.")
                    continue # Cannot proceed with this sub-strategy

            else: # Single asset strategy
                # Try to get asset name from common param names
                asset_name_param = final_sub_params.get('asset_name', final_sub_params.get('asset', final_sub_params.get('instrument')))
                if asset_name_param and isinstance(asset_name_param, str):
                    sub_asset_names = [asset_name_param]
                else:
                    self.logger.error(f"[{self.config.name}] Single-asset strategy '{sub_strategy_instance_name}' ({strategy_class_name}) missing 'asset_name' or similar in params: {final_sub_params}. Skipping.")
                    continue
                sub_strat_input_dim = self.num_features_per_asset
                if 'close_feature_idx' in final_sub_params: # Standardize param name
                    final_sub_params['close_idx'] = final_sub_params.pop('close_feature_idx')
            
            # Create StrategyConfig for the sub-strategy
            sub_strategy_config_obj = StrategyConfig(
                name=sub_strategy_instance_name,
                description=StrategyClass.default_config().description if hasattr(StrategyClass, 'default_config') else f"Instance of {strategy_class_name}",
                default_params={}, # Default params of sub-strategy class are handled by its own init
                applicable_assets=sub_asset_names, # Assets this specific instance works on
                input_dim=sub_strat_input_dim
            )
            
            try:
                strategy_instance = StrategyClass(
                    config=sub_strategy_config_obj, 
                    params=final_sub_params, 
                    logger=self.logger # Pass down the logger
                )
                self.sub_strategy_details.append({
                    'instance': strategy_instance,
                    'asset_names': sub_asset_names, # List of asset name(s)
                    'is_pair': is_pair_strategy
                })
                self.logger.info(f"[{self.config.name}] Successfully initialized sub-strategy: '{sub_strategy_instance_name}' ({strategy_class_name}) for assets: {sub_asset_names} with input_dim: {sub_strat_input_dim}")
            except Exception as e:
                self.logger.error(f"[{self.config.name}] Failed to initialize sub-strategy '{sub_strategy_instance_name}' ({strategy_class_name}): {e}", exc_info=True)

        if not self.sub_strategy_details:
            self.logger.warning(f"[{self.config.name}] No sub-strategies were successfully initialized.")


    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, sequence_length, total_composite_features = asset_features.shape
        device = asset_features.device
        
        if not self.sub_strategy_details:
            self.logger.warning(f"[{self.config.name}] No sub-strategies initialized. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        if self.num_features_per_asset is None or self.num_features_per_asset == 0:
            self.logger.error(f"[{self.config.name}] num_features_per_asset is invalid ({self.num_features_per_asset}). Cannot slice features. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        expected_total_features = len(self.asset_to_composite_idx_map) * self.num_features_per_asset
        if total_composite_features != expected_total_features:
            self.logger.error(f"[{self.config.name}] Input asset_features dimension mismatch. Expected {expected_total_features} features, got {total_composite_features}. (Num mapped assets: {len(self.asset_to_composite_idx_map)}, Features per asset: {self.num_features_per_asset}). Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        all_sub_signals = []

        for sub_detail in self.sub_strategy_details:
            sub_strat_instance = sub_detail['instance']
            sub_asset_names = sub_detail['asset_names']
            is_pair = sub_detail['is_pair']
            
            sub_input_features: Optional[torch.Tensor] = None
            D = self.num_features_per_asset

            try:
                if is_pair:
                    if len(sub_asset_names) != 2: # Should have been caught in init
                        self.logger.error(f"[{self.config.name}] Sub-strategy '{sub_strat_instance.config.name}' is pair type but has {len(sub_asset_names)} assets. Skipping.")
                        continue
                    
                    asset1_name, asset2_name = sub_asset_names[0], sub_asset_names[1]
                    if asset1_name not in self.asset_to_composite_idx_map or \
                       asset2_name not in self.asset_to_composite_idx_map:
                        self.logger.error(f"[{self.config.name}] Assets '{asset1_name}' or '{asset2_name}' for sub-strategy '{sub_strat_instance.config.name}' not found in composite's asset map: {list(self.asset_to_composite_idx_map.keys())}. Skipping.")
                        continue
                        
                    idx1 = self.asset_to_composite_idx_map[asset1_name]
                    idx2 = self.asset_to_composite_idx_map[asset2_name]

                    features1 = asset_features[:, :, idx1*D : (idx1+1)*D]
                    features2 = asset_features[:, :, idx2*D : (idx2+1)*D]
                    sub_input_features = torch.cat((features1, features2), dim=2)
                else: # Single asset
                    if len(sub_asset_names) != 1: # Should have been caught in init
                        self.logger.error(f"[{self.config.name}] Sub-strategy '{sub_strat_instance.config.name}' is single-asset type but has {len(sub_asset_names)} assets. Skipping.")
                        continue

                    asset_name = sub_asset_names[0]
                    if asset_name not in self.asset_to_composite_idx_map:
                        self.logger.error(f"[{self.config.name}] Asset '{asset_name}' for sub-strategy '{sub_strat_instance.config.name}' not found in composite's asset map: {list(self.asset_to_composite_idx_map.keys())}. Skipping.")
                        continue
                    
                    idx = self.asset_to_composite_idx_map[asset_name]
                    sub_input_features = asset_features[:, :, idx*D : (idx+1)*D]
                
                if sub_input_features is not None:
                    # Pass current_positions relevant to this sub-strategy if applicable (complex, for now pass None or global)
                    # This composite strategy itself might have a notion of position, or sub-strategies are stateless for this call.
                    signal = sub_strat_instance.forward(sub_input_features, current_positions=None, timestamp=timestamp)
                    all_sub_signals.append(signal)
                else:
                    self.logger.warning(f"[{self.config.name}] sub_input_features was None for sub-strategy '{sub_strat_instance.config.name}'. This should not happen.")

            except Exception as e:
                self.logger.error(f"[{self.config.name}] Error processing sub-strategy '{sub_strat_instance.config.name}': {e}", exc_info=True)
                # Add a neutral signal for this failing sub-strategy to maintain batch size consistency
                all_sub_signals.append(torch.zeros((batch_size, 1, 1), device=device))


        if not all_sub_signals:
            self.logger.warning(f"[{self.config.name}] No signals generated from sub-strategies. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        # Stack signals: (num_sub_strategies, batch_size, 1, 1)
        signals_tensor = torch.stack(all_sub_signals, dim=0)

        combined_signal: torch.Tensor
        if self.combination_logic == 'sum':
            combined_signal = torch.sum(signals_tensor, dim=0)
        elif self.combination_logic == 'average':
            combined_signal = torch.mean(signals_tensor, dim=0)
        elif self.combination_logic == 'majority_vote':
            # Sum signals and take the sign. If sum is 0, sign is 0.
            combined_signal = torch.sign(torch.sum(signals_tensor, dim=0))
        else: # Default to sum
            self.logger.warning(f"[{self.config.name}] Unknown combination_logic '{self.combination_logic}'. Defaulting to 'sum'.")
            combined_signal = torch.sum(signals_tensor, dim=0)
        
        return combined_signal.view(batch_size, 1, 1) # Ensure final shape

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        self.logger.info(f"[{self.config.name}] generate_signals (legacy) called. This composite strategy primarily uses the tensor-based forward() method.")
        output_index = None
        if processed_data_dict:
            for key, df_val in processed_data_dict.items():
                if isinstance(df_val, pd.DataFrame) and not df_val.empty:
                    output_index = df_val.index
                    break
        if output_index is None:
            self.logger.warning(f"[{self.config.name}] No valid index found in processed_data_dict for legacy generate_signals.")
            return pd.DataFrame(columns=['signal'])
        
        self.logger.warning(f"[{self.config.name}] Legacy generate_signals for composite strategy is not fully implemented to combine sub-signals. Returning neutral.")
        return pd.DataFrame(0.0, index=output_index, columns=['signal'])

@register_strategy("VolatilityBreakoutStrategy")
class VolatilityBreakoutStrategy(BaseStrategy):
    """
    A strategy that signals when price breaks out of a volatility-defined range (e.g., ATR bands).
    """
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="VolatilityBreakoutStrategy",
            description="Signals breakouts from ATR-based channels.",
            default_params={
                'atr_period': 14, 
                'atr_multiplier': 2.0, 
                'asset_list': [], # For orchestrator to know applicability, forward processes one asset's tensor
                'high_idx': 0,    # Index for high price in feature tensor
                'low_idx': 1,     # Index for low price in feature tensor
                'close_idx': 2    # Index for close price in feature tensor
            }
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config=config, params=params, logger=logger)
        self.atr_period = self.params.get('atr_period', 14)
        self.atr_multiplier = self.params.get('atr_multiplier', 2.0)
        self.HIGH_IDX = self.params.get('high_idx', 0)
        self.LOW_IDX = self.params.get('low_idx', 1)
        self.CLOSE_IDX = self.params.get('close_idx', 2)

    # Copied _rolling_mean helper method
    def _rolling_mean(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """ Helper for rolling mean using conv1d. Assumes tensor is (batch, seq_len). """
        if tensor.ndim == 2:
            tensor_unsqueezed = tensor.unsqueeze(1) # (batch, 1, seq_len)
        elif tensor.ndim == 3 and tensor.shape[1] == 1: # (batch, 1, seq_len)
            tensor_unsqueezed = tensor
        else:
            self.logger.error(f"{self.config.name} _rolling_mean expects 2D (batch, seq_len) or 3D (batch, 1, seq_len) tensor, got {tensor.shape}")
            return torch.zeros_like(tensor) # Fallback

        if window_size <= 0:
            self.logger.error(f"{self.config.name} _rolling_mean window_size must be positive, got {window_size}")
            return torch.zeros_like(tensor)
        if tensor.shape[-1] < window_size: # sequence_length < window_size
             # For SMA, often we want NaNs or partial sums. conv1d with padding replicates.
             # This behavior might differ from pandas rolling.mean(min_periods=1)
             # For now, proceed, but be aware if sequence is too short.
             # Or, return zeros/NaNs if seq_len < window_size strictly.
             # Let's allow it, F.pad will handle short sequences by replicating.
             pass


        padding = window_size - 1
        # Pad on the left to align rolling window calculation
        padded_tensor = F.pad(tensor_unsqueezed, (padding, 0), mode='replicate')
        
        # Define weights for simple moving average
        sma_weights = torch.full((1, 1, window_size), 1.0/window_size, device=tensor.device, dtype=tensor.dtype)
        
        pooled = F.conv1d(padded_tensor, sma_weights, stride=1)
        
        if tensor.ndim == 2:
            return pooled.squeeze(1) # (batch, seq_len)
        return pooled # (batch, 1, seq_len)

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        """
        Processes tensor features for a single asset to generate volatility breakout signals.
        asset_features: (batch_size, sequence_length, num_features)
                        Expected feature order: self.HIGH_IDX, self.LOW_IDX, self.CLOSE_IDX
        Returns: (batch_size, 1, 1) signal tensor.
        """
        batch_size, sequence_length, num_features = asset_features.shape
        device = asset_features.device

        if not (0 <= self.HIGH_IDX < num_features and \
                0 <= self.LOW_IDX < num_features and \
                0 <= self.CLOSE_IDX < num_features):
            self.logger.error(f"{self.config.name}: One or more feature indices (H:{self.HIGH_IDX}, L:{self.LOW_IDX}, C:{self.CLOSE_IDX}) are out of bounds for num_features {num_features}. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        if sequence_length < self.atr_period or sequence_length < 2: # Need at least 2 for TR, and atr_period for ATR
            self.logger.warning(f"{self.config.name}: Sequence length ({sequence_length}) is less than atr_period ({self.atr_period}) or too short for TR. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        high_prices = asset_features[:, :, self.HIGH_IDX]    # (batch_size, sequence_length)
        low_prices = asset_features[:, :, self.LOW_IDX]      # (batch_size, sequence_length)
        close_prices = asset_features[:, :, self.CLOSE_IDX]  # (batch_size, sequence_length)

        # Calculate True Range (TR)
        # prev_close needs to be handled carefully for the first element.
        prev_close = torch.zeros_like(close_prices)
        prev_close[:, 1:] = close_prices[:, :-1]
        prev_close[:, 0] = close_prices[:, 0] # Approximation for first element's prev_close

        tr1 = high_prices - low_prices
        tr2 = torch.abs(high_prices - prev_close)
        tr3 = torch.abs(low_prices - prev_close)
        
        true_range = torch.max(torch.max(tr1, tr2), tr3)
        # For the very first data point, TR is often defined as High - Low
        true_range[:, 0] = high_prices[:, 0] - low_prices[:, 0]

        # Calculate ATR (using SMA of True Range for simplicity)
        # Note: True ATR often uses Wilder's smoothing (an EMA). This is an SMA approximation.
        atr = self._rolling_mean(true_range, self.atr_period) # (batch_size, sequence_length)

        # Calculate center line for bands (SMA of close prices)
        # The original pandas code used mavg_close = df['close'].rolling(window=atr_period, min_periods=max(1, atr_period // 2)).mean()
        # Our _rolling_mean with conv1d and replicate padding behaves like min_periods=atr_period if seq_len >= atr_period
        # or effectively uses replicated values if shorter.
        # For consistency with how ATR is calculated (full window), use full window for SMA close too.
        center_line = self._rolling_mean(close_prices, self.atr_period) # (batch_size, sequence_length)
        
        # Calculate ATR Bands
        upper_band = center_line + (atr * self.atr_multiplier)
        lower_band = center_line - (atr * self.atr_multiplier)

        # Signals are based on the most recent time step
        last_close = close_prices[:, -1]    # (batch_size)
        last_upper_band = upper_band[:, -1] # (batch_size)
        last_lower_band = lower_band[:, -1] # (batch_size)
        
        signal = torch.zeros(batch_size, device=device)
        # Buy signal: close breaks above upper band
        signal[last_close > last_upper_band] = 1.0
        # Sell signal: close breaks below lower band
        signal[last_close < last_lower_band] = -1.0
        
        # Reshape to (batch_size, 1, 1)
        return signal.view(batch_size, 1, 1)

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        self.logger.info(f"{self.config.name}: generate_signals called. This method is part of the legacy interface. "
                         f"The primary signal generation is now in the tensor-based forward() method. "
                         f"This implementation will return a neutral signal DataFrame based on input dict's index, if possible.")

        output_index = None
        if processed_data_dict:
            for key, df_val in processed_data_dict.items():
                if isinstance(df_val, pd.DataFrame) and not df_val.empty:
                    output_index = df_val.index
                    self.logger.debug(f"Using index from processed_data_dict key '{key}' for generate_signals output.")
                    break

        if output_index is None:
            self.logger.warning(f"{self.config.name}: No valid index found in processed_data_dict for generate_signals. Returning empty DataFrame with 'signal' column.")
            return pd.DataFrame(columns=['signal'])

        signals_df = pd.DataFrame(0.0, index=output_index, columns=['signal'])
        return signals_df

class StatisticalArbitrageStrategy(BaseStrategy):

    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="StatisticalArbitrageStrategy",
            description="A composite strategy that combines multiple statistical arbitrage sub-strategies.",
            default_params={
                'base_strategies_config': [], 
                'combination_logic': 'sum', 
                # 'num_features_per_asset': None, # This will now come from self.config.input_dim
            },
            applicable_assets=[] 
        )
    
    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config=config, params=params, logger=logger)
        
        self.base_strategy_configs_from_params = self.params.get('base_strategies_config', [])
        self.combination_logic = self.params.get('combination_logic', 'sum')
        
        self.num_features_per_asset = self.config.input_dim # D_transformer_out
        if self.num_features_per_asset is None:
            self.logger.critical(f"[{self.config.name}] self.config.input_dim (num_features_per_asset) is NOT set. This is critical for sub-strategy configuration and feature slicing. Defaulting to 1, but this is likely incorrect.")
            self.num_features_per_asset = 1 # A potentially unsafe default, but prevents None errors later.

        # This map is crucial for slicing the composite strategy's input tensor in the forward pass.
        # self.config.applicable_assets should be populated by EnhancedQuantumStrategyLayer
        # with all unique assets this composite strategy instance will handle, in a defined order.
        self.asset_to_composite_idx_map: Dict[str, int] = {
            asset_name: i for i, asset_name in enumerate(self.config.applicable_assets or [])
        }
        if not self.asset_to_composite_idx_map and self.base_strategy_configs_from_params:
            self.logger.warning(f"[{self.config.name}] 'applicable_assets' for composite strategy is empty or None, but sub-strategies are defined. Feature slicing in forward pass will likely fail if not all sub-strategy assets are covered or if map is empty.")
        elif self.asset_to_composite_idx_map:
             self.logger.info(f"[{self.config.name}] Asset to composite index map: {self.asset_to_composite_idx_map}")


        self.sub_strategy_details: List[Dict[str, Any]] = []
        
        # Ensure STRATEGY_REGISTRY is populated (it should be at import time)
        if not STRATEGY_REGISTRY:
             self.logger.error(f"[{self.config.name}] STRATEGY_REGISTRY is empty. Cannot initialize sub-strategies.")
             return

        for sub_config_item in self.base_strategy_configs_from_params:
            strategy_class_name = sub_config_item.get('strategy_class_name')
            sub_strategy_instance_name = sub_config_item.get('name', strategy_class_name) # Default name
            sub_params = sub_config_item.get('params', {})

            if not strategy_class_name:
                self.logger.error(f"[{self.config.name}] Sub-strategy config missing 'strategy_class_name': {sub_config_item}")
                continue

            StrategyClass = STRATEGY_REGISTRY.get(strategy_class_name)
            if not StrategyClass:
                self.logger.error(f"[{self.config.name}] Sub-strategy class '{strategy_class_name}' not found in STRATEGY_REGISTRY.")
                continue

            # Determine assets and input_dim for the sub-strategy
            sub_asset_names: List[str] = []
            sub_strat_input_dim: Optional[int] = None
            final_sub_params = sub_params.copy() # Params to pass to sub-strategy constructor

            # Heuristic to identify pair strategies (could be made more robust, e.g. by checking class inheritance or a property)
            is_pair_strategy = "pair" in strategy_class_name.lower() or \
                               "cointegration" in strategy_class_name.lower() or \
                               hasattr(StrategyClass, 'asset_pair_names') or \
                               ('asset_pair_names' in sub_params or 'asset_pair' in sub_params)


            if is_pair_strategy:
                sub_asset_names = final_sub_params.get('asset_pair_names', final_sub_params.get('asset_pair', []))
                if len(sub_asset_names) != 2:
                    self.logger.error(f"[{self.config.name}] Pair strategy '{sub_strategy_instance_name}' ({strategy_class_name}) requires 'asset_pair_names' with 2 assets. Got: {sub_asset_names}. Skipping.")
                    continue
                if self.num_features_per_asset is not None:
                    sub_strat_input_dim = 2 * self.num_features_per_asset
                    # Adjust close indices for pair strategy:
                    # asset1_close_idx is relative to its own block (0-indexed within its D features)
                    # asset2_close_idx is relative to the start of the concatenated 2D block
                    # (i.e., D + relative_idx_in_second_block)
                    if 'asset1_price_feature_idx' in final_sub_params:
                        final_sub_params['asset1_close_idx'] = final_sub_params.pop('asset1_price_feature_idx')
                    if 'asset2_price_feature_idx' in final_sub_params and self.num_features_per_asset is not None:
                        final_sub_params['asset2_close_idx'] = self.num_features_per_asset + final_sub_params.pop('asset2_price_feature_idx')
                    # If asset1_close_idx or asset2_close_idx are already directly in final_sub_params, they are used as is.
                else: # num_features_per_asset is None
                    self.logger.error(f"[{self.config.name}] Cannot determine input_dim for pair sub-strategy '{sub_strategy_instance_name}' because num_features_per_asset is unknown.")
                    continue # Cannot proceed with this sub-strategy

            else: # Single asset strategy
                # Try to get asset name from common param names
                asset_name_param = final_sub_params.get('asset_name', final_sub_params.get('asset', final_sub_params.get('instrument')))
                if asset_name_param and isinstance(asset_name_param, str):
                    sub_asset_names = [asset_name_param]
                else:
                    self.logger.error(f"[{self.config.name}] Single-asset strategy '{sub_strategy_instance_name}' ({strategy_class_name}) missing 'asset_name' or similar in params: {final_sub_params}. Skipping.")
                    continue
                sub_strat_input_dim = self.num_features_per_asset
                if 'close_feature_idx' in final_sub_params: # Standardize param name
                    final_sub_params['close_idx'] = final_sub_params.pop('close_feature_idx')
            
            # Create StrategyConfig for the sub-strategy
            sub_strategy_config_obj = StrategyConfig(
                name=sub_strategy_instance_name,
                description=StrategyClass.default_config().description if hasattr(StrategyClass, 'default_config') else f"Instance of {strategy_class_name}",
                default_params={}, # Default params of sub-strategy class are handled by its own init
                applicable_assets=sub_asset_names, # Assets this specific instance works on
                input_dim=sub_strat_input_dim
            )
            
            try:
                strategy_instance = StrategyClass(
                    config=sub_strategy_config_obj, 
                    params=final_sub_params, 
                    logger=self.logger # Pass down the logger
                )
                self.sub_strategy_details.append({
                    'instance': strategy_instance,
                    'asset_names': sub_asset_names, # List of asset name(s)
                    'is_pair': is_pair_strategy
                })
                self.logger.info(f"[{self.config.name}] Successfully initialized sub-strategy: '{sub_strategy_instance_name}' ({strategy_class_name}) for assets: {sub_asset_names} with input_dim: {sub_strat_input_dim}")
            except Exception as e:
                self.logger.error(f"[{self.config.name}] Failed to initialize sub-strategy '{sub_strategy_instance_name}' ({strategy_class_name}): {e}", exc_info=True)

        if not self.sub_strategy_details:
            self.logger.warning(f"[{self.config.name}] No sub-strategies were successfully initialized.")


    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, sequence_length, total_composite_features = asset_features.shape
        device = asset_features.device
        
        if not self.sub_strategy_details:
            self.logger.warning(f"[{self.config.name}] No sub-strategies initialized. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        if self.num_features_per_asset is None or self.num_features_per_asset == 0:
            self.logger.error(f"[{self.config.name}] num_features_per_asset is invalid ({self.num_features_per_asset}). Cannot slice features. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        expected_total_features = len(self.asset_to_composite_idx_map) * self.num_features_per_asset
        if total_composite_features != expected_total_features:
            self.logger.error(f"[{self.config.name}] Input asset_features dimension mismatch. Expected {expected_total_features} features, got {total_composite_features}. (Num mapped assets: {len(self.asset_to_composite_idx_map)}, Features per asset: {self.num_features_per_asset}). Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        all_sub_signals = []

        for sub_detail in self.sub_strategy_details:
            sub_strat_instance = sub_detail['instance']
            sub_asset_names = sub_detail['asset_names']
            is_pair = sub_detail['is_pair']
            
            sub_input_features: Optional[torch.Tensor] = None
            D = self.num_features_per_asset

            try:
                if is_pair:
                    if len(sub_asset_names) != 2: # Should have been caught in init
                        self.logger.error(f"[{self.config.name}] Sub-strategy '{sub_strat_instance.config.name}' is pair type but has {len(sub_asset_names)} assets. Skipping.")
                        continue
                    
                    asset1_name, asset2_name = sub_asset_names[0], sub_asset_names[1]
                    if asset1_name not in self.asset_to_composite_idx_map or \
                       asset2_name not in self.asset_to_composite_idx_map:
                        self.logger.error(f"[{self.config.name}] Assets '{asset1_name}' or '{asset2_name}' for sub-strategy '{sub_strat_instance.config.name}' not found in composite's asset map: {list(self.asset_to_composite_idx_map.keys())}. Skipping.")
                        continue
                        
                    idx1 = self.asset_to_composite_idx_map[asset1_name]
                    idx2 = self.asset_to_composite_idx_map[asset2_name]

                    features1 = asset_features[:, :, idx1*D : (idx1+1)*D]
                    features2 = asset_features[:, :, idx2*D : (idx2+1)*D]
                    sub_input_features = torch.cat((features1, features2), dim=2)
                else: # Single asset
                    if len(sub_asset_names) != 1: # Should have been caught in init
                        self.logger.error(f"[{self.config.name}] Sub-strategy '{sub_strat_instance.config.name}' is single-asset type but has {len(sub_asset_names)} assets. Skipping.")
                        continue

                    asset_name = sub_asset_names[0]
                    if asset_name not in self.asset_to_composite_idx_map:
                        self.logger.error(f"[{self.config.name}] Asset '{asset_name}' for sub-strategy '{sub_strat_instance.config.name}' not found in composite's asset map: {list(self.asset_to_composite_idx_map.keys())}. Skipping.")
                        continue
                    
                    idx = self.asset_to_composite_idx_map[asset_name]
                    sub_input_features = asset_features[:, :, idx*D : (idx+1)*D]
                
                if sub_input_features is not None:
                    # Pass current_positions relevant to this sub-strategy if applicable (complex, for now pass None or global)
                    # This composite strategy itself might have a notion of position, or sub-strategies are stateless for this call.
                    signal = sub_strat_instance.forward(sub_input_features, current_positions=None, timestamp=timestamp)
                    all_sub_signals.append(signal)
                else:
                    self.logger.warning(f"[{self.config.name}] sub_input_features was None for sub-strategy '{sub_strat_instance.config.name}'. This should not happen.")

            except Exception as e:
                self.logger.error(f"[{self.config.name}] Error processing sub-strategy '{sub_strat_instance.config.name}': {e}", exc_info=True)
                # Add a neutral signal for this failing sub-strategy to maintain batch size consistency
                all_sub_signals.append(torch.zeros((batch_size, 1, 1), device=device))


        if not all_sub_signals:
            self.logger.warning(f"[{self.config.name}] No signals generated from sub-strategies. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        # Stack signals: (num_sub_strategies, batch_size, 1, 1)
        signals_tensor = torch.stack(all_sub_signals, dim=0)

        combined_signal: torch.Tensor
        if self.combination_logic == 'sum':
            combined_signal = torch.sum(signals_tensor, dim=0)
        elif self.combination_logic == 'average':
            combined_signal = torch.mean(signals_tensor, dim=0)
        elif self.combination_logic == 'majority_vote':
            # Sum signals and take the sign. If sum is 0, sign is 0.
            combined_signal = torch.sign(torch.sum(signals_tensor, dim=0))
        else: # Default to sum
            self.logger.warning(f"[{self.config.name}] Unknown combination_logic '{self.combination_logic}'. Defaulting to 'sum'.")
            combined_signal = torch.sum(signals_tensor, dim=0)
        
        return combined_signal.view(batch_size, 1, 1) # Ensure final shape

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        self.logger.info(f"[{self.config.name}] generate_signals (legacy) called. This composite strategy primarily uses the tensor-based forward() method.")
        output_index = None
        if processed_data_dict:
            for key, df_val in processed_data_dict.items():
                if isinstance(df_val, pd.DataFrame) and not df_val.empty:
                    output_index = df_val.index
                    break
        if output_index is None:
            self.logger.warning(f"[{self.config.name}] No valid index found in processed_data_dict for legacy generate_signals.")
            return pd.DataFrame(columns=['signal'])
        
        # This part is tricky: how to combine legacy signals?
        # For now, just return neutral. A full legacy implementation would be extensive.
        self.logger.warning(f"[{self.config.name}] Legacy generate_signals for composite strategy is not fully implemented to combine sub-signals. Returning neutral.")
        return pd.DataFrame(0.0, index=output_index, columns=['signal'])



