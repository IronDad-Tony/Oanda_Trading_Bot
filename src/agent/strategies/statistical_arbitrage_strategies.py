# src/agent/strategies/statistical_arbitrage_strategies.py
import pandas as pd
import numpy as np
import ta
from ta.utils import dropna
from .base_strategy import BaseStrategy, StrategyConfig
from typing import Dict, List, Optional, Any, Tuple # MODIFIED: Added Tuple
import logging
import sys
import torch # ADDED: Import torch
import torch.nn.functional as F # ADDED: Import F

class MeanReversionStrategy(BaseStrategy):

    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="MeanReversionStrategy",
            description="Trades based on mean reversion principles using Bollinger Bands.",
            default_params={'bb_period': 20, 'bb_std_dev': 2.0, 'asset_list': [], 'close_idx': 0} # Added close_idx
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config=config, params=params, logger=logger)
        self.CLOSE_IDX = self.params.get('close_idx', 0)


    def _rolling_mean(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """ Helper for rolling mean using avg_pool1d. Assumes tensor is (batch, seq_len). """
        if tensor.ndim == 2:
            tensor_unsqueezed = tensor.unsqueeze(1) # (batch, 1, seq_len)
        elif tensor.ndim == 3 and tensor.shape[1] == 1:
            tensor_unsqueezed = tensor
        else:
            self.logger.error(f"_rolling_mean expects 2D (batch, seq_len) or 3D (batch, 1, seq_len) tensor, got {tensor.shape}")
            return torch.zeros_like(tensor) # Fallback

        padding = window_size - 1
        padded_tensor = F.pad(tensor_unsqueezed, (padding, 0), mode='replicate')
        # Correct weights for conv1d as SMA: (out_channels, in_channels/groups, kernel_size)
        sma_weights = torch.full((1, 1, window_size), 1.0/window_size, device=tensor.device, dtype=tensor.dtype)
        pooled = F.conv1d(padded_tensor, sma_weights, stride=1)
        
        if tensor.ndim == 2:
            return pooled.squeeze(1) # (batch, seq_len)
        return pooled # (batch, 1, seq_len)

    def _rolling_std(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """ Helper for rolling std. Assumes tensor is (batch, seq_len). """
        mean_x = self._rolling_mean(tensor, window_size)
        # For (tensor**2), ensure it's handled correctly if tensor was unsqueezed in _rolling_mean
        if tensor.ndim == 2 and mean_x.ndim == 2: # if input was 2D and output of mean is 2D
             mean_x_sq = self._rolling_mean(tensor**2, window_size)
        elif tensor.ndim == 3 and mean_x.ndim == 3 : # if input was 3D (B,1,S) and output of mean is 3D
             mean_x_sq = self._rolling_mean(tensor**2, window_size)
        else: # Fallback or error
            self.logger.error(f"Shape mismatch in _rolling_std. Tensor shape: {tensor.shape}, mean_x shape: {mean_x.shape}")
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
            self.logger.error(f"{self.config.name}: close_idx {self.CLOSE_IDX} is out of bounds for num_features {num_features}. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        if sequence_length < bb_period:
            self.logger.warning(f"{self.config.name}: Sequence length ({sequence_length}) is less than bb_period ({bb_period}). Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        close_prices = asset_features[:, :, self.CLOSE_IDX]  # (batch_size, sequence_length)

        # Calculate Bollinger Bands
        sma_mid_band = self._rolling_mean(close_prices, bb_period) # (batch_size, sequence_length)
        rolling_std = self._rolling_std(close_prices, bb_period)   # (batch_size, sequence_length)
        
        upper_band = sma_mid_band + (rolling_std * bb_std_dev)
        lower_band = sma_mid_band - (rolling_std * bb_std_dev)

        # We are interested in the signals for the most recent time step
        last_close = close_prices[:, -1]    # (batch_size)
        last_upper_band = upper_band[:, -1] # (batch_size)
        last_lower_band = lower_band[:, -1] # (batch_size)
        
        signal = torch.zeros(batch_size, device=device)
        # Buy signal: close crosses below lower band (or is below)
        signal[last_close < last_lower_band] = 1.0
        # Sell signal: close crosses above upper band (or is above)
        signal[last_close > last_upper_band] = -1.0
        
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

class CointegrationStrategy(BaseStrategy):

    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="CointegrationStrategy",
            description="Trades based on cointegration of an asset pair.",
            # Ensure asset_pair is a list of two strings for this strategy to be valid
            # asset_features tensor should be structured such that features for asset1 come first, then for asset2
            # e.g., if each asset has F features, total features = 2*F.
            # asset1_close_idx would be the index for close price within asset1's features (e.g., 0)
            # asset2_close_idx would be the index for close price within asset2's features (e.g., F if asset1 has F features)
            default_params={'asset_pair': [], 'window': 60, 'z_threshold': 2.0, 'asset1_close_idx': 0, 'asset2_close_idx': None} # MODIFIED
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config=config, params=params, logger=logger)
        self.asset_pair = self.params.get('asset_pair', [])
        self.window = self.params.get('window', 60)
        self.z_threshold = self.params.get('z_threshold', 2.0)
        self.asset1_close_idx = self.params.get('asset1_close_idx', 0)
        # asset2_close_idx needs to be set based on the number of features per asset.
        # This is a bit tricky without knowing num_features_per_asset beforehand.
        # For now, we'll assume it's passed or can be inferred if asset_features is (batch, seq, num_total_features)
        # and num_total_features is known to be 2 * num_features_per_asset.
        self.asset2_close_idx = self.params.get('asset2_close_idx') 
        
        if len(self.asset_pair) != 2 or not all(isinstance(p, str) for p in self.asset_pair):
            self.logger.warning(f"{self.config.name} requires 'asset_pair' to be a list of two asset name strings. Current: {self.asset_pair}. Strategy may not function correctly with old methods.")
            self.valid_pair_for_old_methods = False # Renamed for clarity
        else:
            self.valid_pair_for_old_methods = True
            if not self.config.applicable_assets and self.valid_pair_for_old_methods:
                self.config.applicable_assets = list(self.asset_pair)
        
        if self.asset2_close_idx is None:
            self.logger.warning(f"{self.config.name}: 'asset2_close_idx' is not set. This is crucial for the new tensor-based forward method. It should be the starting index of the second asset's close price in the concatenated feature tensor.")


    # Helper methods _rolling_mean and _rolling_std can be used from MeanReversionStrategy if they are in the same file
    # or defined in a common utility, or copied here. For now, let's assume they are accessible if defined above.
    # If not, they would need to be copied/imported.
    # For simplicity, let's copy them here to make CointegrationStrategy self-contained with its tensor helpers.

    def _rolling_mean(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """ Helper for rolling mean. Assumes tensor is (batch, seq_len). """
        if tensor.ndim == 2:
            tensor_unsqueezed = tensor.unsqueeze(1) # (batch, 1, seq_len)
        elif tensor.ndim == 3 and tensor.shape[1] == 1: # (batch, 1, seq_len)
            tensor_unsqueezed = tensor
        else:
            self.logger.error(f"_rolling_mean expects 2D (batch, seq_len) or 3D (batch, 1, seq_len) tensor, got {tensor.shape}")
            return torch.zeros_like(tensor)

        padding = window_size - 1
        padded_tensor = F.pad(tensor_unsqueezed, (padding, 0), mode='replicate')
        sma_weights = torch.full((1, 1, window_size), 1.0/window_size, device=tensor.device, dtype=tensor.dtype)
        pooled = F.conv1d(padded_tensor, sma_weights, stride=1)
        
        if tensor.ndim == 2:
            return pooled.squeeze(1) # (batch, seq_len)
        return pooled # (batch, 1, seq_len)

    def _rolling_std(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """ Helper for rolling std. Assumes tensor is (batch, seq_len). """
        mean_x = self._rolling_mean(tensor, window_size)
        if tensor.ndim == 2 and mean_x.ndim == 2:
             mean_x_sq = self._rolling_mean(tensor**2, window_size)
        elif tensor.ndim == 3 and mean_x.ndim == 3 and tensor.shape[1] == 1 and mean_x.shape[1] == 1: # Ensure it's (B,1,S)
             mean_x_sq = self._rolling_mean(tensor**2, window_size) # tensor**2 will be (B,1,S)
        else:
            self.logger.error(f"Shape mismatch or unexpected dim in _rolling_std. Tensor shape: {tensor.shape}, mean_x shape: {mean_x.shape}")
            return torch.zeros_like(tensor)

        variance = (mean_x_sq - mean_x**2).clamp(min=1e-9)
        return torch.sqrt(variance)

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        """
        Processes tensor features for a cointegrated pair of assets.
        asset_features: (batch_size, sequence_length, num_total_features)
                        It's assumed that features for asset1 are followed by features for asset2.
                        e.g., if asset1 has F1 features and asset2 has F2 features, num_total_features = F1 + F2.
                        asset1_close_idx is the index for asset1's close price within its F1 features.
                        asset2_close_idx is the absolute index for asset2's close price in num_total_features.
        Returns: (batch_size, 1, 1) signal tensor.
        """
        batch_size, sequence_length, num_total_features = asset_features.shape
        device = asset_features.device

        if self.asset2_close_idx is None:
            self.logger.error(f"{self.config.name}: asset2_close_idx is not configured. Cannot determine second asset's close price. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        if self.asset1_close_idx >= num_total_features or self.asset2_close_idx >= num_total_features:
            self.logger.error(f"{self.config.name}: One or both close_idx ({self.asset1_close_idx}, {self.asset2_close_idx}) are out of bounds for num_total_features {num_total_features}. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)
        
        # It's crucial that asset1_close_idx refers to the column for the first asset's price
        # and asset2_close_idx refers to the column for the second asset's price in the *combined* asset_features tensor.
        # For example, if asset_features = [A1_feat1, A1_feat2, ..., A1_close, ..., A2_feat1, A2_feat2, ..., A2_close, ...]
        # then asset1_close_idx would be the index of A1_close, and asset2_close_idx the index of A2_close.

        if sequence_length < self.window:
            self.logger.warning(f"{self.config.name}: Sequence length ({sequence_length}) is less than window ({self.window}). Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        close_prices_asset1 = asset_features[:, :, self.asset1_close_idx]  # (batch_size, sequence_length)
        close_prices_asset2 = asset_features[:, :, self.asset2_close_idx]  # (batch_size, sequence_length)

        spread = close_prices_asset1 - close_prices_asset2 # (batch_size, sequence_length)

        spread_mean = self._rolling_mean(spread, self.window) # (batch_size, sequence_length)
        spread_std = self._rolling_std(spread, self.window)   # (batch_size, sequence_length)
        
        # Avoid division by zero or NaN if spread_std is zero (e.g., constant spread)
        # A small epsilon is added to std for numerical stability.
        z_score = (spread - spread_mean) / (spread_std + 1e-9) # (batch_size, sequence_length)

        # We are interested in the z-score for the most recent time step
        last_z_score = z_score[:, -1]    # (batch_size)
        
        signal = torch.zeros(batch_size, device=device)
        # Long spread (Buy asset1, Sell asset2) if z_score is too low
        signal[last_z_score < -self.z_threshold] = 1.0
        # Short spread (Sell asset1, Buy asset2) if z_score is too high
        signal[last_z_score > self.z_threshold] = -1.0
        
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

class PairsTradeStrategy(BaseStrategy):
    """Trades based on the z-score of a spread between two assets, with entry and exit thresholds.""" # MODIFIED

    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="PairsTradeStrategy",
            description="Trades based on the spread of an asset pair using z-scores with entry/exit thresholds.", # MODIFIED
            # Similar to CointegrationStrategy, asset_features tensor combines features for asset1 and asset2.
            default_params={
                'asset_pair': [], 
                'window': 60, 
                'entry_threshold': 2.0, 
                'exit_threshold': 0.5,
                'asset1_close_idx': 0, 
                'asset2_close_idx': None # Must be configured based on feature layout
            }
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config=config, params=params, logger=logger)
        self.asset_pair = self.params.get('asset_pair', [])
        self.window = self.params.get('window', 60)
        self.entry_threshold = self.params.get('entry_threshold', 2.0)
        self.exit_threshold = self.params.get('exit_threshold', 0.5)
        self.asset1_close_idx = self.params.get('asset1_close_idx', 0)
        self.asset2_close_idx = self.params.get('asset2_close_idx')

        if len(self.asset_pair) != 2 or not all(isinstance(p, str) for p in self.asset_pair):
            self.logger.warning(f"{self.config.name}: Requires 'asset_pair' of two asset name strings. Current: {self.asset_pair}. Strategy may not function correctly with old methods.")
            self.valid_pair_for_old_methods = False # Renamed for clarity
        else:
            self.valid_pair_for_old_methods = True
            if not self.config.applicable_assets and self.valid_pair_for_old_methods:
                 self.config.applicable_assets = list(self.asset_pair)
        
        if self.asset2_close_idx is None:
            self.logger.warning(f"{self.config.name}: 'asset2_close_idx' is not set. This is crucial for the new tensor-based forward method.")

        # Internal state for tracking current position based on z-score logic (for tensor method)
        # 0 = no position, 1 = long spread, -1 = short spread
        # This needs to be managed per batch item if we want to handle stateful exits correctly across calls for the same batch item.
        # For a stateless forward pass (typical in deep learning batch processing), this might be simplified or handled by `current_positions` input.
        # Let's assume for now `current_positions` gives the necessary state if needed, or the strategy is mostly stateless for entry.
        # The exit logic here will be based on current z-score vs exit_threshold, not requiring memory of *when* it entered.

    # Copying _rolling_mean and _rolling_std from CointegrationStrategy for self-containment
    def _rolling_mean(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """ Helper for rolling mean. Assumes tensor is (batch, seq_len). """
        if tensor.ndim == 2:
            tensor_unsqueezed = tensor.unsqueeze(1) # (batch, 1, seq_len)
        elif tensor.ndim == 3 and tensor.shape[1] == 1: # (batch, 1, seq_len)
            tensor_unsqueezed = tensor
        else:
            self.logger.error(f"_rolling_mean expects 2D (batch, seq_len) or 3D (batch, 1, seq_len) tensor, got {tensor.shape}")
            return torch.zeros_like(tensor)

        padding = window_size - 1
        padded_tensor = F.pad(tensor_unsqueezed, (padding, 0), mode='replicate')
        sma_weights = torch.full((1, 1, window_size), 1.0/window_size, device=tensor.device, dtype=tensor.dtype)
        pooled = F.conv1d(padded_tensor, sma_weights, stride=1)
        
        if tensor.ndim == 2:
            return pooled.squeeze(1) # (batch, seq_len)
        return pooled # (batch, 1, seq_len)

    def _rolling_std(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """ Helper for rolling std. Assumes tensor is (batch, seq_len). """
        mean_x = self._rolling_mean(tensor, window_size)
        if tensor.ndim == 2 and mean_x.ndim == 2:
             mean_x_sq = self._rolling_mean(tensor**2, window_size)
        elif tensor.ndim == 3 and mean_x.ndim == 3 and tensor.shape[1] == 1 and mean_x.shape[1] == 1:
             mean_x_sq = self._rolling_mean(tensor**2, window_size)
        else:
            self.logger.error(f"Shape mismatch or unexpected dim in _rolling_std. Tensor shape: {tensor.shape}, mean_x shape: {mean_x.shape}")
            return torch.zeros_like(tensor)

        variance = (mean_x_sq - mean_x**2).clamp(min=1e-9)
        return torch.sqrt(variance)

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        """
        Processes tensor features for a pair of assets to generate trading signals based on z-score thresholds.
        asset_features: (batch_size, sequence_length, num_total_features) - combined features for asset1 and asset2.
        current_positions: (batch_size, 1, 1) or (batch_size) - current position state for the pair (-1 short, 0 neutral, 1 long spread).
        Returns: (batch_size, 1, 1) signal tensor (target position: -1, 0, 1).
        """
        batch_size, sequence_length, num_total_features = asset_features.shape
        device = asset_features.device

        if self.asset2_close_idx is None:
            self.logger.error(f"{self.config.name}: asset2_close_idx is not configured. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        if self.asset1_close_idx >= num_total_features or self.asset2_close_idx >= num_total_features:
            self.logger.error(f"{self.config.name}: Close indices out of bounds. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        if sequence_length < self.window:
            self.logger.warning(f"{self.config.name}: Sequence length ({sequence_length}) < window ({self.window}). Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        close_prices_asset1 = asset_features[:, :, self.asset1_close_idx]
        close_prices_asset2 = asset_features[:, :, self.asset2_close_idx]
        spread = close_prices_asset1 - close_prices_asset2

        spread_mean = self._rolling_mean(spread, self.window)
        spread_std = self._rolling_std(spread, self.window)
        z_score = (spread - spread_mean) / (spread_std + 1e-9)

        last_z_score = z_score[:, -1] # (batch_size)
        
        # Initialize target_position based on current_positions if provided, else assume neutral
        # current_positions might be (batch_size, 1, 1) or (batch_size, 1) or (batch_size,)
        if current_positions is not None:
            current_pos_flat = current_positions.view(batch_size).clone() # Ensure it's (batch_size)
        else:
            current_pos_flat = torch.zeros(batch_size, device=device)
        
        target_position = current_pos_flat.clone()

        # Entry logic
        # Try to enter long spread if neutral and z_score < -entry_threshold
        can_enter_long = (current_pos_flat == 0) & (last_z_score < -self.entry_threshold)
        target_position[can_enter_long] = 1.0

        # Try to enter short spread if neutral and z_score > entry_threshold
        can_enter_short = (current_pos_flat == 0) & (last_z_score > self.entry_threshold)
        target_position[can_enter_short] = -1.0

        # Exit logic
        # Exit long spread if z_score >= -exit_threshold (or a more positive value like 0)
        # The exit_threshold is typically closer to 0 than the entry_threshold.
        should_exit_long = (current_pos_flat == 1.0) & (last_z_score >= -self.exit_threshold)
        target_position[should_exit_long] = 0.0

        # Exit short spread if z_score <= exit_threshold
        should_exit_short = (current_pos_flat == -1.0) & (last_z_score <= self.exit_threshold)
        target_position[should_exit_short] = 0.0
        
        return target_position.view(batch_size, 1, 1)


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

class StatisticalArbitrageStrategy(BaseStrategy): # This is a composite strategy

    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="StatisticalArbitrageStrategy", # Name of this composite strategy itself
            description="A composite strategy that combines multiple statistical arbitrage sub-strategies.",
            default_params={
                'base_strategies_config': [], # List of dicts, each defining a sub-strategy
                                            # Example for base_strategies_config item:
                                            # {
                                            #     'strategy_class_name': 'MeanReversionStrategy',
                                            #     'name': 'MR_EUR_USD', # Unique instance name
                                            #     'params': { # Params for MeanReversionStrategy
                                            #         'asset_name': 'EUR_USD', # Asset this instance applies to
                                            #         'close_feature_idx': 0, # Index of close price in EUR_USD's feature block
                                            #         'bb_period': 20, 
                                            #         'bb_std_dev': 2.0
                                            #     },
                                            # },
                                            # {
                                            #     'strategy_class_name': 'CointegrationStrategy',
                                            #     'name': 'Coint_GBPUSD_USDCHF', # Unique instance name
                                            #     'params': { # Params for CointegrationStrategy
                                            #         'asset_pair_names': ['GBP_USD', 'USD_CHF'], # Assets for the pair
                                            #         'asset1_price_feature_idx': 0, # Index of price in GBP_USD's feature block
                                            #         'asset2_price_feature_idx': 0, # Index of price in USD_CHF's feature block
                                            #         'window': 60,
                                            #         'z_threshold': 2.0
                                            #     },
                                            # }
                'combination_logic': 'sum', # 'sum', 'average', 'majority_vote' (implemented as sign of sum)
                'num_features_per_asset': None # Crucial: Number of features for each asset's data block
            },
            applicable_assets=[] # Dynamically determined from sub-strategies
        )
    
    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config=config, params=params, logger=logger)
        
        self.base_strategy_configs_from_params = self.params.get('base_strategies_config', [])
        self.combination_logic = self.params.get('combination_logic', 'sum')
        self.num_features_per_asset = self.params.get('num_features_per_asset')

        if self.num_features_per_asset is None:
            self.logger.critical(f"[{self.config.name}] 'num_features_per_asset' must be specified in params for the composite strategy. Sub-strategy tensor slicing will fail.")
            # Consider raising an error or setting a flag to prevent forward pass
            # For now, execution will likely fail later if this is not set.

        self.strategies: List[BaseStrategy] = [] # Stores instantiated sub-strategies
        self.sub_strategy_details = [] # Stores {instance, type, asset_names, asset_indices_in_composite}
        
        all_asset_names_discovered = set()

        current_package_module = None
        try:
            if __package__:
                current_package_module = sys.modules[__package__]
            elif __name__ != '__main__': # Try to get it via the class's module if not run as main
                 # Assuming structure like project_root.src.agent.strategies
                 # self.__class__.__module__ might be 'src.agent.strategies.statistical_arbitrage_strategies'
                 # We want 'src.agent.strategies'
                module_parts = self.__class__.__module__.split('.')
                if len(module_parts) > 1: # Ensure there's a package part
                    package_name_for_strategies = '.'.join(module_parts[:-1]) # e.g., 'src.agent.strategies'
                    if package_name_for_strategies in sys.modules:
                         current_package_module = sys.modules[package_name_for_strategies]
                    else: # Try importing it if not already loaded under that exact name
                        try:
                            import importlib
                            current_package_module = importlib.import_module(package_name_for_strategies)
                            self.logger.info(f"Dynamically imported module {package_name_for_strategies} for sub-strategy loading.")
                        except ImportError:
                            self.logger.warning(f"Could not dynamically import {package_name_for_strategies}.")

        except KeyError: # Should not happen if __package__ or __module__ is valid
            self.logger.error(f"[{self.config.name}] Error determining package for dynamic strategy loading.")
        
        if not current_package_module: # Fallback if running as a script or package context is tricky
            try:
                import src.agent.strategies as strategies_fallback_module
                current_package_module = strategies_fallback_module
                self.logger.info(f"[{self.config.name}] Fallback: Using 'src.agent.strategies' module directly for sub-strategy loading.")
            except ImportError:
                self.logger.error(f"[{self.config.name}] CRITICAL: Fallback import of 'src.agent.strategies' failed. Cannot load sub-strategies.")
                # self.strategies will remain empty

        if current_package_module:
            for sub_conf_dict in self.base_strategy_configs_from_params:
                if not isinstance(sub_conf_dict, dict):
                    self.logger.warning(f"Sub-strategy config item is not a dict: {sub_conf_dict}. Skipping.")
                    continue

                sub_strategy_class_name = sub_conf_dict.get('strategy_class_name')
                sub_strategy_instance_name = sub_conf_dict.get('name', sub_strategy_class_name) 
                sub_strategy_params_from_config = sub_conf_dict.get('params', {})
                
                # This was 'sub_strategy_applicable_assets' before, but now we derive from params
                # sub_strategy_explicit_applicable_assets = sub_conf_dict.get('applicable_assets') 

                if not sub_strategy_class_name:
                    self.logger.warning(f"Sub-strategy config missing 'strategy_class_name': {sub_conf_dict}. Skipping.")
                    continue

                self.logger.info(f"Attempting to load sub-strategy: Name='{sub_strategy_instance_name}', Class='{sub_strategy_class_name}'")
                
                if hasattr(current_package_module, sub_strategy_class_name):
                    SubStrategyClass = getattr(current_package_module, sub_strategy_class_name)
                    
                    # Prepare params and determine assets for this sub-strategy instance
                    current_sub_asset_names = []
                    current_strategy_type = None
                    params_for_sub_instance = sub_strategy_params_from_config.copy()

                    asset_name_param = params_for_sub_instance.get('asset_name')
                    asset_pair_names_param = params_for_sub_instance.get('asset_pair_names')

                    if asset_name_param and isinstance(asset_name_param, str): # Single asset strategy
                        current_strategy_type = 'single'
                        current_sub_asset_names = [asset_name_param]
                        all_asset_names_discovered.add(asset_name_param)
                        if 'close_feature_idx' in params_for_sub_instance:
                            params_for_sub_instance['close_idx'] = params_for_sub_instance.pop('close_feature_idx')
                        params_for_sub_instance.pop('asset_name', None) # Clean up

                    elif asset_pair_names_param and isinstance(asset_pair_names_param, list) and len(asset_pair_names_param) == 2: # Pair asset strategy
                        current_strategy_type = 'pair'
                        current_sub_asset_names = asset_pair_names_param
                        all_asset_names_discovered.update(current_sub_asset_names)
                        
                        params_for_sub_instance['asset_pair'] = asset_pair_names_param # Expected by pair strategies

                        if 'asset1_price_feature_idx' in params_for_sub_instance:
                            params_for_sub_instance['asset1_close_idx'] = params_for_sub_instance.pop('asset1_price_feature_idx')
                        else:
                            self.logger.warning(f"Missing 'asset1_price_feature_idx' for pair strategy {sub_strategy_instance_name}. Defaulting to 0 or sub-strategy default.")


                        if 'asset2_price_feature_idx' in params_for_sub_instance:
                            asset2_price_feature_idx_in_own_block = params_for_sub_instance.pop('asset2_price_feature_idx')
                            if self.num_features_per_asset is not None:
                                params_for_sub_instance['asset2_close_idx'] = self.num_features_per_asset + asset2_price_feature_idx_in_own_block
                            else:
                                self.logger.error(f"Cannot calculate 'asset2_close_idx' for {sub_strategy_instance_name} because 'num_features_per_asset' is not set for composite strategy. Sub-strategy might fail.")
                        else:
                             self.logger.warning(f"Missing 'asset2_price_feature_idx' for pair strategy {sub_strategy_instance_name}. Relying on sub-strategy default for 'asset2_close_idx' which might be incorrect for concatenated tensor.")
                        
                        params_for_sub_instance.pop('asset_pair_names', None) # Clean up
                    
                    else:
                        self.logger.warning(f"Sub-strategy {sub_strategy_instance_name} ({sub_strategy_class_name}) has unclear/missing asset configuration ('asset_name' or 'asset_pair_names'). Skipping.")
                        continue

                    try:
                        sub_default_config_obj: Optional[StrategyConfig] = None
                        if hasattr(SubStrategyClass, 'default_config') and callable(SubStrategyClass.default_config):
                            sub_default_config_obj = SubStrategyClass.default_config()
                            sub_default_config_obj.name = sub_strategy_instance_name 
                            sub_default_config_obj.applicable_assets = current_sub_asset_names # Set based on this instance
                        else: # Create basic config if default_config is missing
                             sub_default_config_obj = StrategyConfig(
                                name=sub_strategy_instance_name,
                                description=f"Sub-strategy of {self.config.name}",
                                applicable_assets=current_sub_asset_names
                            )
                        
                        strategy_instance = SubStrategyClass(
                            config=sub_default_config_obj, 
                            params=params_for_sub_instance, 
                            logger=self.logger
                        )
                        self.strategies.append(strategy_instance)
                        self.sub_strategy_details.append({
                            'instance': strategy_instance,
                            'type': current_strategy_type,
                            'asset_names': current_sub_asset_names,
                            'asset_indices_in_composite': [] # Will be filled after all assets are known
                        })
                        self.logger.info(f"Successfully instantiated sub-strategy: {sub_strategy_instance_name} ({sub_strategy_class_name}) for assets: {current_sub_asset_names}")
                    except Exception as e_inst:
                        self.logger.error(f"Error instantiating sub-strategy {sub_strategy_class_name} with name {sub_strategy_instance_name}: {e_inst}", exc_info=True)
                else:
                    self.logger.warning(f"Sub-strategy class '{sub_strategy_class_name}' not found in module '{current_package_module.__name__ if current_package_module else 'N/A'}'. Skipping.")
        
        # Finalize composite applicable assets and map
        self.composite_applicable_assets = sorted(list(all_asset_names_discovered))
        self.asset_name_to_tensor_idx_map = {name: i for i, name in enumerate(self.composite_applicable_assets)}
        
        # Update self.config.applicable_assets for the composite strategy itself
        self.config.applicable_assets = self.composite_applicable_assets
        self.logger.info(f"Composite strategy '{self.config.name}' will manage assets: {self.composite_applicable_assets}")

        # Populate asset_indices_in_composite for each sub_strategy_detail
        for detail in self.sub_strategy_details:
            try:
                detail['asset_indices_in_composite'] = [self.asset_name_to_tensor_idx_map[name] for name in detail['asset_names']]
            except KeyError as e:
                self.logger.error(f"Asset name {e} from sub-strategy {detail['instance'].config.name} not found in composite asset map. This should not happen.")
                # This indicates a logic error in asset collection or mapping.

        self.logger.info(f"StatisticalArbitrageStrategy '{self.config.name}' initialized with {len(self.strategies)} sub-strategies, covering {len(self.composite_applicable_assets)} unique assets.")


    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        """
        Processes tensor features for all applicable assets and combines signals from sub-strategies.
        Args:
            asset_features (torch.Tensor): Shape (batch_size, num_composite_assets, sequence_length, num_features_per_asset)
            current_positions (Optional[torch.Tensor]): Shape (batch_size, num_composite_assets, 1)
            timestamp (Optional[pd.Timestamp]): Current timestamp.
        Returns:
            torch.Tensor: Target positions for each composite asset. Shape (batch_size, num_composite_assets, 1)
        """
        batch_size = asset_features.shape[0]
        num_composite_assets_in_tensor = asset_features.shape[1]
        sequence_length = asset_features.shape[2]
        num_features_per_asset_in_tensor = asset_features.shape[3]
        device = asset_features.device

        expected_num_composite_assets = len(self.composite_applicable_assets)

        if self.num_features_per_asset is None:
            self.logger.error(f"[{self.config.name}] 'num_features_per_asset' is not configured for the composite strategy. Cannot proceed.")
            return torch.zeros((batch_size, expected_num_composite_assets, 1), device=device)
        
        if num_features_per_asset_in_tensor != self.num_features_per_asset:
            self.logger.error(f"[{self.config.name}] Mismatch in 'num_features_per_asset'. Expected {self.num_features_per_asset}, got {num_features_per_asset_in_tensor} from input tensor.")
            return torch.zeros((batch_size, expected_num_composite_assets, 1), device=device)

        if num_composite_assets_in_tensor != expected_num_composite_assets:
            self.logger.error(f"[{self.config.name}] Mismatch in number of assets in 'asset_features' tensor. Expected {expected_num_composite_assets} (for {self.composite_applicable_assets}), got {num_composite_assets_in_tensor}.")
            return torch.zeros((batch_size, expected_num_composite_assets, 1), device=device)

        if current_positions is not None and current_positions.shape[1] != expected_num_composite_assets:
            self.logger.error(f"[{self.config.name}] Mismatch in number of assets in 'current_positions' tensor. Expected {expected_num_composite_assets}, got {current_positions.shape[1]}. Discarding current_positions for this pass.")
            current_positions = None
        
        # Initialize tensor to store aggregated signals/target positions from sub-strategies
        sub_strategy_signal_contributions = torch.zeros((batch_size, expected_num_composite_assets, 1), device=device)

        for detail in self.sub_strategy_details:
            sub_strategy_instance = detail['instance']
            strategy_type = detail['type']
            # Indices of the sub-strategy's asset(s) in the composite strategy's tensor order
            asset_indices_in_composite_tensor = detail['asset_indices_in_composite']

            sub_asset_features_feed = None
            sub_current_positions_feed = None

            try:
                if strategy_type == 'single':
                    if not asset_indices_in_composite_tensor: continue # Should not happen if init is correct
                    asset_idx = asset_indices_in_composite_tensor[0]
                    
                    # Shape: (batch_size, sequence_length, num_features_per_asset)
                    sub_asset_features_feed = asset_features[:, asset_idx, :, :]
                    
                    if current_positions is not None:
                        # Shape: (batch_size, 1, 1) for sub-strategy
                        sub_current_positions_feed = current_positions[:, asset_idx, :].unsqueeze(1) 

                elif strategy_type == 'pair':
                    if len(asset_indices_in_composite_tensor) < 2: continue # Should not happen
                    asset1_idx_comp = asset_indices_in_composite_tensor[0]
                    asset2_idx_comp = asset_indices_in_composite_tensor[1]

                    features_asset1 = asset_features[:, asset1_idx_comp, :, :] # (B, S, F)
                    features_asset2 = asset_features[:, asset2_idx_comp, :, :] # (B, S, F)
                    
                    # Concatenate along the feature dimension for the pair strategy
                    # Shape: (batch_size, sequence_length, 2 * num_features_per_asset)
                    sub_asset_features_feed = torch.cat((features_asset1, features_asset2), dim=2)

                    # Derive pair's current position if sub-strategy is PairsTradeStrategy
                    if current_positions is not None and isinstance(sub_strategy_instance, PairsTradeStrategy):
                        pos_asset1 = current_positions[:, asset1_idx_comp, 0] # (batch_size)
                        pos_asset2 = current_positions[:, asset2_idx_comp, 0] # (batch_size)
                        
                        pair_pos_state = torch.zeros(batch_size, device=device)
                        # Long spread: asset1 long (>0), asset2 short (<0)
                        pair_pos_state[(pos_asset1 > 1e-6) & (pos_asset2 < -1e-6)] = 1.0
                        # Short spread: asset1 short (<0), asset2 long (>0)
                        pair_pos_state[(pos_asset1 < -1e-6) & (pos_asset2 > 1e-6)] = -1.0
                        
                        sub_current_positions_feed = pair_pos_state.view(batch_size, 1, 1)
                
                if sub_asset_features_feed is not None:
                    # Sub-strategy signal is expected to be (batch_size, 1, 1)
                    # representing target position for its entity (single asset or pair)
                    sub_target_position_signal = sub_strategy_instance.forward(
                        sub_asset_features_feed, 
                        sub_current_positions_feed, 
                        timestamp
                    )

                    if not (sub_target_position_signal.ndim == 3 and sub_target_position_signal.shape[0] == batch_size and sub_target_position_signal.shape[1:] == (1,1)):
                        self.logger.warning(f"Sub-strategy {sub_strategy_instance.config.name} returned signal with unexpected shape {sub_target_position_signal.shape}. Expected ({batch_size}, 1, 1). Skipping its contribution.")
                        continue

                    # Distribute this signal to the affected assets in the composite's contribution tensor
                    if strategy_type == 'single':
                        asset_idx = asset_indices_in_composite_tensor[0]
                        sub_strategy_signal_contributions[:, asset_idx, :] += sub_target_position_signal.squeeze(-1) # (B,1)
                    
                    elif strategy_type == 'pair':
                        asset1_idx_comp = asset_indices_in_composite_tensor[0]
                        asset2_idx_comp = asset_indices_in_composite_tensor[1]
                        
                        # If sub_target_position_signal is +1, it means long asset1 / short asset2
                        # If sub_target_position_signal is -1, it means short asset1 / long asset2
                        sub_strategy_signal_contributions[:, asset1_idx_comp, :] += sub_target_position_signal.squeeze(-1)
                        sub_strategy_signal_contributions[:, asset2_idx_comp, :] -= sub_target_position_signal.squeeze(-1)
            
            except Exception as e:
                self.logger.error(f"Error during forward pass of sub-strategy {sub_strategy_instance.config.name}: {e}", exc_info=True)

        # Apply final combination logic to the aggregated contributions
        final_target_positions = None
        if self.combination_logic == 'sum':
            final_target_positions = sub_strategy_signal_contributions
        elif self.combination_logic == 'average':
            # Proper averaging requires counting contributions per asset.
            # This is a simplified placeholder; true averaging needs more logic if strategies don't always contribute.
            # For now, if all strategies contribute to all their assets, this is like sum with a later global division.
            # A more robust average would divide each asset's sum of signals by the number of strategies targeting it.
            # This is non-trivial to implement here without more state/tracking during accumulation.
            self.logger.warning("Combination logic 'average' is complex for tensor aggregation with variable contributions; using 'sum' as a fallback for now. Consider implementing precise counting if 'average' is critical.")
            final_target_positions = sub_strategy_signal_contributions # Fallback to sum
        elif self.combination_logic == 'majority_vote': # Interpreted as taking the sign of the sum
            final_target_positions = torch.sign(sub_strategy_signal_contributions)
        else: 
            self.logger.warning(f"Unknown combination_logic '{self.combination_logic}', defaulting to 'sum'.")
            final_target_positions = sub_strategy_signal_contributions
            
        return final_target_positions.view(batch_size, expected_num_composite_assets, 1)


    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals based on the combined logic of its sub-strategies.
        This method should aggregate signals from MeanReversion, Cointegration, and PairsTrade.
        For now, it will return an empty DataFrame, assuming sub-strategies handle their own signal generation
        or that a more sophisticated aggregation logic will be implemented later.
        """
        # Placeholder: In a real scenario, this would involve complex aggregation.
        # For now, let's assume signals are generated and acted upon by sub-strategies
        # or a higher-level mechanism.
        # Returning an empty DataFrame with expected columns if necessary.
        # Example: return pd.DataFrame(columns=['signal', 'confidence'], index=market_data.index)
        if market_data.empty:
            return pd.DataFrame()
        
        signals_list = []
        for strategy_name, strategy_instance in self.sub_strategies.items():
            if hasattr(strategy_instance, 'generate_signals') and callable(strategy_instance.generate_signals):
                # Assuming sub-strategies' generate_signals take market_data and return a DataFrame
                # with a 'signal' column.
                # This part needs to be adapted based on actual sub-strategy signal generation.
                # For this placeholder, we'll just call it and expect it to work.
                # sub_signals = strategy_instance.generate_signals(market_data)
                # signals_list.append(sub_signals)
                pass # Actual aggregation logic would go here.

        # This is a simplified placeholder.
        # A real implementation would need to define how signals from different strategies are combined.
        # For example, averaging, voting, or a more complex model.
        # For now, returning a DataFrame with a neutral signal.
        neutral_signals = pd.DataFrame(index=market_data.index)
        neutral_signals['signal'] = 0 # Neutral signal
        if 'price' in market_data.columns: # Or a relevant column for signal alignment
            neutral_signals['price'] = market_data['price']

        # If sub-strategies directly produce actions or are managed by SAC,
        # this composite signal might be more for logging or high-level decision making.
        return neutral_signals

    def forward(self, market_data_tensor: torch.Tensor, observation: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass for the statistical arbitrage strategy composite.
        This aggregates the processed tensor features of all applicable assets and computes
        the target positions based on the combined signals of the sub-strategies.
        """
        # Placeholder for composite forward logic.
        # This would typically involve:
        # 1. Processing the market_data_tensor to extract features for each asset.
        # 2. Passing these features through each sub-strategy's forward method.
        # 3. Aggregating the signals/positions from each sub-strategy according to the combination_logic.
        # 4. Returning the final target positions for the portfolio.

        # For now, let's just log the call and return a tensor of zeros (neutral positions).
        self.logger.info(f"{self.config.name}: forward called for composite strategy.")
        batch_size = market_data_tensor.shape[0]
        expected_num_composite_assets = len(self.composite_applicable_assets)
        device = market_data_tensor.device

        # Returning neutral positions
        return torch.zeros((batch_size, expected_num_composite_assets, 1), device=device)

    # def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
    #     self.logger.debug(f"{self.config.name}: forward called.")
    #     processed_data = {}
    #     if len(self.asset_pair) == 2 and self.asset_pair[0] in market_data_dict and self.asset_pair[1] in market_data_dict: # MODIFIED: valid_pair_for_old_methods
    #         df1_orig = market_data_dict[self.asset_pair[0]]
    #         df2_orig = market_data_dict[self.asset_pair[1]]

    #         df1_close_col = 'Close' if 'Close' in df1_orig.columns else 'close'
    #         df2_close_col = 'Close' if 'Close' in df2_orig.columns else 'close'

    #         if df1_close_col not in df1_orig.columns or df2_close_col not in df2_orig.columns:
    #             self.logger.warning(f"{self.config.name}: Close column missing for one or both assets in pair {self.asset_pair}")
    #             return processed_data

    #         df1 = df1_orig[[df1_close_col]].copy().rename(columns={df1_close_col: self.asset_pair[0]})
    #         df2 = df2_orig[[df2_close_col]].copy().rename(columns={df2_close_col: self.asset_pair[1]})
            
    #         pair_key = "pair_" + "_".join(self.asset_pair) # Ensure self.asset_pair is used if valid_pair
    #         merged_df = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')

    #         if not merged_df.empty and len(merged_df) >= self.params['window']:
    #             merged_df['spread'] = merged_df[self.asset_pair[0]] - merged_df[self.asset_pair[1]]
    #             spread_mean = merged_df['spread'].rolling(window=self.params['window']).mean()
    #             spread_std = merged_df['spread'].rolling(window=self.params['window']).std()
    #             merged_df['z_score'] = (merged_df['spread'] - spread_mean) / spread_std.replace(0, np.nan)
    #             processed_data[pair_key] = merged_df # Keep NaNs
    #         elif not merged_df.empty:
    #              merged_df['spread'] = np.nan
    #              merged_df['z_score'] = np.nan
    #              processed_data[pair_key] = merged_df
    #         else:
    #             self.logger.debug(f"{self.config.name}: Merged DataFrame for pair {self.asset_pair} is empty.")
    #             processed_data[pair_key] = pd.DataFrame()
    #     else:
    #         self.logger.warning(f"{self.config.name}: Invalid pair or missing data for {self.asset_pair} (old method).") # MODIFIED
    #     return processed_data

    # def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    #     self.logger.debug(f"{self.config.name}: generate_signals called.")
    #     output_index = None
    #     pair_key = None

    #     if len(self.asset_pair) == 2: # MODIFIED: valid_pair_for_old_methods
    #         pair_key = "pair_" + "_".join(self.asset_pair)
    #         if pair_key in processed_data_dict and not processed_data_dict[pair_key].empty:
    #             output_index = processed_data_dict[pair_key].index
        
    #     if output_index is None: # Fallback
    #         if processed_data_dict:
    #             for key, df_val in processed_data_dict.items():
    #                 if not df_val.empty:
    #                     output_index = df_val.index
    #                     break
    #         if output_index is None:
    #             self.logger.warning(f"{self.config.name}: Could not determine output index for signals.")
    #             return pd.DataFrame(columns=['signal'])
        
    #     signals_df = pd.DataFrame(0.0, index=output_index, columns=['signal'])

    #     if len(self.asset_pair) == 2 and pair_key and pair_key in processed_data_dict: # MODIFIED: valid_pair_for_old_methods
    #         df = processed_data_dict[pair_key]
    #         if not df.empty and 'z_score' in df.columns and not df['z_score'].isnull().all():
    #             signals_df.loc[df['z_score'] < -self.params['entry_threshold'], 'signal'] = 1.0
    #             signals_df.loc[df['z_score'] > self.params['entry_threshold'], 'signal'] = -1.0
    #             # Exit condition (simplified) - This old logic was problematic and incomplete for stateful exits.
    #             # The new tensor forward method handles exits based on current_positions and z-score vs exit_threshold.
    #             # exiting_long = (df['z_score'] >= -self.params['exit_threshold']) & (df['z_score'].shift(1) < -self.params['exit_threshold']) 
    #             # exiting_short = (df['z_score'] <= self.params['exit_threshold']) & (df['z_score'].shift(1) > self.params['exit_threshold']) 
    #             # signals_df.loc[exiting_long | exiting_short, 'signal'] = 0.0 
    #         else:
    #             self.logger.debug(f"{self.config.name}: z_score not available or all NaN for pair {pair_key} (old method).") # MODIFIED
        
    #     return signals_df.fillna(0.0)

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
