# src/agent/strategies/trend_strategies.py
from .base_strategy import BaseStrategy, StrategyConfig
from . import register_strategy
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List, Tuple # MODIFIED: Added Tuple
import logging
import torch
import torch.nn.functional as F # Add F

# Technical Analysis library (optional, but useful)
# Ensure 'ta' is installed: pip install ta
try:
    import ta
except ImportError:
    print("Consider installing the 'ta' library for more technical indicators: pip install ta")
    ta = None

# --- Trend Strategies ---

@register_strategy("MomentumStrategy")
class MomentumStrategy(BaseStrategy):
    """å‹•é‡ç­–ç•¥ï¼šåŸºæ–¼åƒ¹æ ¼å‹•é‡é€²è¡Œäº¤æ˜“"""
    
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="MomentumStrategy",
            description="Trades based on price momentum.",
            default_params={'momentum_window': 20, 'momentum_period': 14, 'signal_threshold_value': 0.5, 'asset_list': []}
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config=config, params=params, logger=logger)
        self.CLOSE_IDX = self.params.get('close_idx', 0) # Make close index configurable

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, sequence_length, num_features = asset_features.shape
        device = asset_features.device

        momentum_window = self.params.get('momentum_window', 20)
        signal_threshold_value = self.params.get('signal_threshold_value', 0.5) # e.g., 0.5 for 0.5% ROC

        if self.CLOSE_IDX >= num_features:
            self.logger.error(f"{self.config.name}: CLOSE_IDX {self.CLOSE_IDX} is out of bounds for num_features {num_features}.")
            return torch.zeros((batch_size, 1, 1), device=device)

        # Need at least momentum_window + 1 data points: P(t) and P(t-momentum_window)
        if sequence_length <= momentum_window: 
            self.logger.warning(f"{self.config.name}: Sequence length ({sequence_length}) is not sufficient for momentum_window ({momentum_window}). Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        close_prices = asset_features[:, :, self.CLOSE_IDX]  # (batch_size, sequence_length)

        price_t = close_prices[:, -1]
        price_t_minus_window = close_prices[:, -1 - momentum_window]

        # Calculate ROC: ((P(t) - P(t-momentum_window)) / P(t-momentum_window)) * 100
        # Add epsilon to prevent division by zero and handle if price_t_minus_window is zero
        roc = torch.zeros_like(price_t) # Initialize roc tensor
        # Create a mask for valid denominators to avoid division by zero or very small numbers leading to huge ROC values
        valid_denominator_mask = torch.abs(price_t_minus_window) > 1e-9
        
        # Calculate ROC only for valid denominators
        roc[valid_denominator_mask] = (price_t[valid_denominator_mask] - price_t_minus_window[valid_denominator_mask]) / price_t_minus_window[valid_denominator_mask] * 100.0
        
        # For cases where price_t_minus_window is zero (or near zero):
        # If price_t is also zero, roc is 0 (already set).
        # If price_t is non-zero and price_t_minus_window is zero, it's an infinite percentage change.
        # We can cap it or set to a large number, or handle as per strategy definition.
        # For now, if price_t_minus_window was ~0 and price_t is positive, roc will be large positive.
        # If price_t_minus_window was ~0 and price_t is negative (not typical for prices), roc will be large negative.
        # The current calculation with 1e-9 in denominator (if not masked out) would lead to large values.
        # The mask `valid_denominator_mask` ensures we only divide where it's safe.
        # Where `price_t_minus_window` is ~0, `roc` remains 0 unless `price_t` is also ~0.
        # If `price_t_minus_window` is exactly 0:
        #   If `price_t` is also 0, ROC is 0.
        #   If `price_t` is > 0, ROC is effectively infinite positive.
        #   If `price_t` is < 0, ROC is effectively infinite negative.
        # Let's refine the handling for exactly zero `price_t_minus_window`
        zero_denominator_mask = ~valid_denominator_mask
        roc[zero_denominator_mask & (price_t > 1e-9)] = float('inf') # Positive infinity for positive price_t
        roc[zero_denominator_mask & (price_t < -1e-9)] = float('-inf') # Negative infinity for negative price_t (unlikely for prices)
        # Note: float('inf') might cause issues in comparisons if not handled by thresholding logic.
        # A large finite number might be safer if 'inf' is problematic downstream.
        # For now, standard ROC calculation is applied where denominator is valid.

        signal = torch.zeros(batch_size, device=device)
        
        signal[roc > signal_threshold_value] = 1.0
        signal[roc < -signal_threshold_value] = -1.0
        
        return signal.view(batch_size, 1, 1)
    
    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        primary_symbol = self._get_primary_symbol(processed_data_dict, portfolio_context)
        
        if not primary_symbol or primary_symbol not in processed_data_dict or processed_data_dict[primary_symbol].empty:
            # print(f"{self.get_strategy_name()}: No data for primary symbol '{primary_symbol}' or data is empty.")
            return pd.DataFrame(columns=['signal'])

        data_df = processed_data_dict[primary_symbol]
        signals_df = pd.DataFrame(index=data_df.index, columns=['signal'], dtype=float)
        signals_df['signal'] = 0.0 # Default to hold

        signal_threshold_pct = self.params.get('signal_threshold_pct', 0.005)

        if 'momentum' in data_df.columns and not data_df['momentum'].empty and 'close' in data_df.columns:
            # If using ROC, momentum is already a percentage. If using simple diff, calculate percentage.
            # Assuming 'momentum' column from forward() is the value to check.
            # If it was simple diff: momentum_pct = data_df['momentum'] / data_df['close'].shift(self.params.get('momentum_period', 14))
            # If it's ROC, it's already a percentage (e.g., 2 means 2% increase)
            # For ROC, the threshold should be adjusted (e.g. signal_threshold_abs = 0.5 for 0.5%)
            # Let's assume 'momentum' is the raw value (like ROC output, not simple diff for this example)
            # and signal_threshold_pct is for this raw value (e.g. if ROC is 0.5, it means 0.5%)
            
            # For ROC-like momentum (where value is already a percentage, e.g. 1 = 1%)
            # The threshold should be scaled if it's given as 0.005 for 0.5%
            # Let's assume signal_threshold_pct is the direct value to compare against ROC (e.g. 0.5 for 0.5%)
            # For clarity, let's rename param or adjust logic.
            # If momentum is ROC (e.g., 1.0 means 1% change), then threshold_pct = 0.5 means 0.5%
            threshold_value = self.params.get('signal_threshold_value', 0.5) # e.g. for ROC, 0.5 means 0.5%
            
            signals_df.loc[data_df['momentum'] > threshold_value, 'signal'] = 1.0  # Buy
            signals_df.loc[data_df['momentum'] < -threshold_value, 'signal'] = -1.0 # Sell
            signals_df.fillna(0.0, inplace=True) # Fill any NaNs from momentum calculation start
        else:
            # print(f"{self.get_strategy_name()}: 'momentum' or 'close' column not found or empty in processed data for {primary_symbol}.")
            pass # Already defaulted to hold

        return signals_df[['signal']]

@register_strategy("BreakoutStrategy")
class BreakoutStrategy(BaseStrategy):
    """çªç ´ç­–ç•¥ï¼šè­˜åˆ¥ä¸¦è·Ÿéš¨åƒ¹æ ¼çªç ´"""

    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="BreakoutStrategy",
            description="Identifies and follows price breakouts.",
            default_params={
                'breakout_window': 20, 
                'std_dev_multiplier': 2.0, 
                'min_breakout_volume_increase_pct': 0.5, 
                'asset_list': [],
                'close_idx': 0, # Default close index
                'high_idx': 1,  # Default high index
                'low_idx': 2,   # Default low index
                'volume_idx': 3 # Default volume index
            } 
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config=config, params=params, logger=logger)
        self.CLOSE_IDX = self.params.get('close_idx', 0)
        self.HIGH_IDX = self.params.get('high_idx', 1)
        self.LOW_IDX = self.params.get('low_idx', 2)
        self.VOLUME_IDX = self.params.get('volume_idx', 3)

    def _rolling_mean(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """ Helper for rolling mean using avg_pool1d. Assumes tensor is (batch, seq_len). """
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(1) # (batch, 1, seq_len)
        
        # Pad to ensure output has same seq_len and means are 'trailing'
        padding = window_size - 1
        padded_tensor = F.pad(tensor, (padding, 0), mode='replicate')
        pooled = F.avg_pool1d(padded_tensor, kernel_size=window_size, stride=1)
        return pooled.squeeze(1) # (batch, seq_len)

    def _rolling_std(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """ Helper for rolling std. Assumes tensor is (batch, seq_len). """
        mean_x = self._rolling_mean(tensor, window_size)
        mean_x_sq = self._rolling_mean(tensor**2, window_size)
        # Clamp to avoid sqrt of negative numbers due to precision issues
        variance = (mean_x_sq - mean_x**2).clamp(min=1e-6)
        return torch.sqrt(variance)

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        """
        Processes tensor features for a single asset to generate breakout signals.
        asset_features: (batch_size, sequence_length, num_features)
                        Expected feature order: uses self.CLOSE_IDX, self.HIGH_IDX, etc.
        Returns: (batch_size, 1, 1) signal tensor.
        """
        # Feature indices are now instance attributes initialized from params
        # CLOSE_IDX, HIGH_IDX, LOW_IDX, VOLUME_IDX = 0, 1, 2, 3 

        batch_size, sequence_length, num_features = asset_features.shape
        device = asset_features.device

        # Get parameters
        breakout_period = self.params.get('breakout_window', 20) # breakout_window from params
        std_dev_multiplier = self.params.get('std_dev_multiplier', 2.0)
        min_volume_increase_pct = self.params.get('min_breakout_volume_increase_pct', 0.5)

        if not (self.CLOSE_IDX < num_features and self.HIGH_IDX < num_features and 
                  self.LOW_IDX < num_features and self.VOLUME_IDX < num_features):
            self.logger.error(f"{self.config.name}: Feature indices out of bounds. Num_features: {num_features}, Required: C={self.CLOSE_IDX},H={self.HIGH_IDX},L={self.LOW_IDX},V={self.VOLUME_IDX}")
            return torch.zeros((batch_size, 1, 1), device=device)

        if sequence_length < breakout_period:
            self.logger.warning(f"{self.config.name}: Sequence length ({sequence_length}) is less than breakout_period ({breakout_period}). Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        close_prices = asset_features[:, :, self.CLOSE_IDX]  # (batch_size, sequence_length)
        high_prices = asset_features[:, :, self.HIGH_IDX]    # (batch_size, sequence_length)
        low_prices = asset_features[:, :, self.LOW_IDX]      # (batch_size, sequence_length)
        volumes = asset_features[:, :, self.VOLUME_IDX]      # (batch_size, sequence_length)

        # Calculate Bollinger Bands
        sma_mid_band = self._rolling_mean(close_prices, breakout_period) # (batch_size, sequence_length)
        rolling_std_dev = self._rolling_std(close_prices, breakout_period) # (batch_size, sequence_length)
        
        upper_band = sma_mid_band + (rolling_std_dev * std_dev_multiplier)
        lower_band = sma_mid_band - (rolling_std_dev * std_dev_multiplier)

        # Calculate average volume
        avg_volume = self._rolling_mean(volumes, breakout_period) # (batch_size, sequence_length)

        # We are interested in the signals for the most recent time step
        # So, we take the last element from sequence-dependent tensors
        last_high = high_prices[:, -1]
        last_close = close_prices[:, -1]
        last_low = low_prices[:, -1]
        last_volume = volumes[:, -1]
        
        last_upper_band = upper_band[:, -1]
        last_lower_band = lower_band[:, -1]
        last_avg_volume = avg_volume[:, -1]

        # Upward breakout conditions for the last time step
        upward_breakout = (
            (last_high > last_upper_band) &
            (last_close > last_upper_band) &
            (last_volume > last_avg_volume * (1 + min_volume_increase_pct))
        )

        # Downward breakout conditions for the last time step
        downward_breakout = (
            (last_low < last_lower_band) &
            (last_close < last_lower_band) &
            (last_volume > last_avg_volume * (1 + min_volume_increase_pct))
        )
        
        signal = torch.zeros(batch_size, device=device)
        signal[upward_breakout] = 1.0
        signal[downward_breakout] = -1.0
        
        # Reshape to (batch_size, 1, 1) as expected by EnhancedStrategySuperposition
        return signal.view(batch_size, 1, 1)

    # def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    #     primary_symbol = self._get_primary_symbol(processed_data_dict, portfolio_context)
    #     if not primary_symbol or primary_symbol not in processed_data_dict or processed_data_dict[primary_symbol].empty:
    #         return pd.DataFrame(columns=['signal'])
    #
    #     data_df = processed_data_dict[primary_symbol]
    #     signals_df = pd.DataFrame(index=data_df.index, columns=['signal'], dtype=float)
    #     signals_df['signal'] = 0.0
    #
    #     min_volume_increase_pct = self.params.get('min_breakout_volume_increase_pct', 0.5)
    #
    #     if not all(col in data_df.columns for col in ['close', 'high', 'low', 'volume', 'upper_band', 'lower_band', 'avg_volume']):
    #         # self.logger.debug(f"{self.get_strategy_name()}: Missing required columns in processed data for {primary_symbol}.")
    #         return signals_df[['signal']]
    #    
    #     # Check for upward breakout
    #     upward_breakout_conditions = (
    #         (data_df['high'] > data_df['upper_band']) & 
    #         (data_df['close'] > data_df['upper_band']) & 
    #         (data_df['volume'] > data_df['avg_volume'] * (1 + min_volume_increase_pct))
    #     )
    #     signals_df.loc[upward_breakout_conditions, 'signal'] = 1.0
    #
    #     # Check for downward breakout
    #     downward_breakout_conditions = (
    #         (data_df['low'] < data_df['lower_band']) & 
    #         (data_df['close'] < data_df['lower_band']) & 
    #         (data_df['volume'] > data_df['avg_volume'] * (1 + min_volume_increase_pct))
    #     )
    #     signals_df.loc[downward_breakout_conditions, 'signal'] = -1.0
    #    
    #     signals_df.fillna(0.0, inplace=True)
    #     return signals_df[['signal']]

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Placeholder for generate_signals. The primary logic is now in the tensor-based forward method.
        """
        self.logger.info(f"{self.config.name}: generate_signals called (pandas-based). Primary logic in tensor forward.")
        idx = None
        if processed_data_dict:
            for df in processed_data_dict.values():
                if not df.empty:
                    idx = df.index
                    break
        if idx is None:
            idx = pd.DatetimeIndex([])
        return pd.DataFrame(index=idx, columns=['signal'], dtype=float).fillna(0.0)

@register_strategy("TrendFollowingStrategy")
class TrendFollowingStrategy(BaseStrategy):
    """è¶¨å‹¢è·Ÿéš¨ç­–ç•¥ï¼šåŸºæ–¼ç§»å‹•å¹³å‡ç·šäº¤å‰"""

    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="TrendFollowingStrategy",
            description="Follows trends based on moving average crossovers.",
            default_params={
                'short_sma_period': 20, 
                'long_sma_period': 50, 
                'asset_list': [],
                'close_idx': 0 # Default close index
            } 
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config=config, params=params, logger=logger)
        self.CLOSE_IDX = self.params.get('close_idx', 0)

    def _rolling_mean(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """ Helper for rolling mean using avg_pool1d. Assumes tensor is (batch, seq_len). """
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(1) # (batch, 1, seq_len)
        
        # Pad to ensure output has same seq_len and means are 'trailing'
        padding = window_size - 1
        padded_tensor = F.pad(tensor, (padding, 0), mode='replicate')
        pooled = F.avg_pool1d(padded_tensor, kernel_size=window_size, stride=1)
        return pooled.squeeze(1) # (batch, seq_len)

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        """
        Processes tensor features for a single asset to generate trend-following signals.
        asset_features: (batch_size, sequence_length, num_features)
                        Expected feature order: uses self.CLOSE_IDX
        Returns: (batch_size, 1, 1) signal tensor.
        """
        # CLOSE_IDX = 0 # Assuming close price is the first feature # Now using self.CLOSE_IDX
        batch_size, sequence_length, num_features = asset_features.shape
        device = asset_features.device

        short_sma_period = self.params.get('short_sma_period', 20)
        long_sma_period = self.params.get('long_sma_period', 50)

        if self.CLOSE_IDX >= num_features:
            self.logger.error(f"{self.config.name}: CLOSE_IDX {self.CLOSE_IDX} is out of bounds for num_features {num_features}.")
            return torch.zeros((batch_size, 1, 1), device=device)

        if short_sma_period <= 0 or long_sma_period <= 0 or long_sma_period <= short_sma_period:
            self.logger.warning(f"{self.config.name}: Invalid SMA periods. Short: {short_sma_period}, Long: {long_sma_period}. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        if sequence_length < long_sma_period:
            self.logger.warning(f"{self.config.name}: Sequence length ({sequence_length}) is less than long_sma_period ({long_sma_period}). Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        close_prices = asset_features[:, :, self.CLOSE_IDX]  # (batch_size, sequence_length)

        short_sma = self._rolling_mean(close_prices, short_sma_period) # (batch_size, sequence_length)
        long_sma = self._rolling_mean(close_prices, long_sma_period)   # (batch_size, sequence_length)

        # Signals based on the most recent crossover
        # We need to look at the current and previous state of SMAs
        # current state: short_sma[:, -1], long_sma[:, -1]
        # previous state: short_sma[:, -2], long_sma[:, -2]
        
        signal = torch.zeros(batch_size, device=device)

        if sequence_length < 2: # Need at least 2 points to check for crossover
             return signal.view(batch_size, 1, 1)

        # Golden Cross: short_sma crosses above long_sma
        buy_condition = (short_sma[:, -1] > long_sma[:, -1]) & (short_sma[:, -2] <= long_sma[:, -2])
        signal[buy_condition] = 1.0

        # Death Cross: short_sma crosses below long_sma
        sell_condition = (short_sma[:, -1] < long_sma[:, -1]) & (short_sma[:, -2] >= long_sma[:, -2])
        signal[sell_condition] = -1.0
        
        return signal.view(batch_size, 1, 1)

    # def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    #     primary_symbol = self._get_primary_symbol(processed_data_dict, portfolio_context)
    #     if not primary_symbol or primary_symbol not in processed_data_dict or processed_data_dict[primary_symbol].empty:
    #         return pd.DataFrame(columns=['signal'])
    #
    #     data_df = processed_data_dict[primary_symbol]
    #     signals_df = pd.DataFrame(index=data_df.index, columns=['signal'], dtype=float)
    #     signals_df['signal'] = 0.0
    #
    #     if not all(col in data_df.columns for col in ['short_sma', 'long_sma']) or data_df[['short_sma', 'long_sma']].isnull().all().all():
    #         # self.logger.debug(f"{self.get_strategy_name()}: SMA columns not found or all NaN in processed data for {primary_symbol}.")
    #         return signals_df[['signal']]
    #
    #     # Golden Cross: short_sma crosses above long_sma
    #     buy_condition = (data_df['short_sma'] > data_df['long_sma']) & (data_df['short_sma'].shift(1) <= data_df['long_sma'].shift(1))
    #     signals_df.loc[buy_condition, 'signal'] = 1.0
    #
    #     # Death Cross: short_sma crosses below long_sma
    #     sell_condition = (data_df['short_sma'] < data_df['long_sma']) & (data_df['short_sma'].shift(1) >= data_df['long_sma'].shift(1))
    #     signals_df.loc[sell_condition, 'signal'] = -1.0
    #    
    #     signals_df.fillna(0.0, inplace=True)
    #     return signals_df[['signal']]

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Placeholder for generate_signals. The primary logic is now in the tensor-based forward method.
        """
        self.logger.info(f"{self.config.name}: generate_signals called (pandas-based). Primary logic in tensor forward.")
        idx = None
        if processed_data_dict:
            for df in processed_data_dict.values():
                if not df.empty:
                    idx = df.index
                    break
        if idx is None:
            idx = pd.DatetimeIndex([])
        return pd.DataFrame(index=idx, columns=['signal'], dtype=float).fillna(0.0)

@register_strategy("ReversalStrategy")
class ReversalStrategy(BaseStrategy):
    """åè½‰ç­–ç•¥ï¼šåŸºæ–¼RSIæŒ‡æ¨™çš„è¶…è²·è¶…è³£"""

    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="ReversalStrategy",
            description="Trades reversals based on RSI overbought/oversold levels.",
            default_params={
                'reversal_window': 14, 
                'rsi_oversold': 30.0, 
                'rsi_overbought': 70.0, 
                'rsi_period': 14, 
                'stoch_k_period':14, 
                'stoch_d_period':3, 
                'stoch_oversold': 20.0, 
                'stoch_overbought': 80.0, 
                'use_stochastic_confirmation': True, 
                'asset_list': []
            }
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config=config, params=params, logger=logger)
        self.CLOSE_IDX = self.params.get('close_idx', 0)
        self.HIGH_IDX = self.params.get('high_idx', 1)
        self.LOW_IDX = self.params.get('low_idx', 2)

    def _rolling_mean_for_indicators(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """ 
        Helper for rolling mean using conv1d, suitable for indicators like RSI's Wilder's smoothing (approx) or SMA for Stochastic %D.
        Assumes tensor is (batch, seq_len) or (batch, 1, seq_len).
        Returns tensor of same input shape (batch, seq_len) or (batch, 1, seq_len).
        """
        input_ndim = tensor.ndim
        if input_ndim == 2:
            tensor_unsqueezed = tensor.unsqueeze(1) # (batch, 1, seq_len)
        elif input_ndim == 3 and tensor.shape[1] == 1:
            tensor_unsqueezed = tensor # Already (batch, 1, seq_len)
        else:
            self.logger.error(f"_rolling_mean_for_indicators expects 2D (batch, seq_len) or 3D (batch, 1, seq_len) tensor, got {tensor.shape}")
            # Fallback or raise error, returning zeros for now to avoid crash
            return torch.zeros_like(tensor)

        # Pad to ensure output has same seq_len and means are 'trailing' (causal)
        padding = window_size - 1
        # mode='replicate' pads with the edge values, good for financial series
        padded_tensor = F.pad(tensor_unsqueezed, (padding, 0), mode='replicate') 
        
        # Define weights for conv1d to act as SMA
        # conv1d weights: (out_channels, in_channels/groups, kernel_size)
        # Here, out_channels=1, in_channels=1 (from unsqueezed tensor), kernel_size=window_size
        sma_weights = torch.full((1, 1, window_size), 1.0/window_size, device=tensor.device, dtype=tensor.dtype)
        
        pooled = F.conv1d(padded_tensor, sma_weights, stride=1)
        
        if input_ndim == 2:
            return pooled.squeeze(1) # (batch, seq_len)
        return pooled # (batch, 1, seq_len)

    def _tensor_rsi(self, close_prices: torch.Tensor, period: int) -> torch.Tensor:
        """
        Calculates RSI using tensor operations.
        close_prices: (batch_size, sequence_length)
        period: RSI period
        Returns: (batch_size, sequence_length) RSI tensor
        """
        current_sequence_length = close_prices.shape[1] # MODIFIED: Get sequence_length from input tensor

        if current_sequence_length < period:
            self.logger.warning(f"RSI calculation: sequence length {current_sequence_length} < period {period}. Returning 50s.")
            return torch.full_like(close_prices, 50.0)
            
        delta = torch.diff(close_prices, dim=1, prepend=close_prices[:, :1]) # Prepend first element to keep seq_len
        gain = F.relu(delta)
        loss = F.relu(-delta)

        # Approximate Wilder's smoothing with SMA for gain/loss.
        # For true Wilder's (EMA with alpha=1/period), a different approach is needed.
        avg_gain = self._rolling_mean_for_indicators(gain, period)
        avg_loss = self._rolling_mean_for_indicators(loss, period)
        
        rs = avg_gain / (avg_loss + 1e-9) # Add epsilon to prevent div by zero
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        rsi[avg_loss < 1e-9] = 100.0 # If avg_loss is near zero, RSI is 100
        rsi[(avg_gain < 1e-9) & (avg_loss >= 1e-9)] = 0.0 # If avg_gain is near zero but avg_loss is not, RSI is 0

        # Set initial unstable period to a neutral value like 50
        # The first 'period-1' values of SMA are less reliable, and diff makes it 'period'
        # Corrected unstable period for RSI: first `period` values are affected by SMA and diff.
        # SMA of window `p` makes first `p-1` values less reliable.
        # `torch.diff` uses one less point, so `delta` has `seq_len-1` if no prepend, or `seq_len` with prepend.
        # If `delta` is calculated from `close_prices` (length `S`), and then `avg_gain/loss` (length `S`) are calculated using SMA of `period`,
        # then the first `period-1` values of `avg_gain/loss` are the ones affected by padding/initialization of SMA.
        # So, `rsi` values from index 0 to `period-2` should be marked as unstable.
        if current_sequence_length >= period:
             rsi[:, :period-1] = 50.0
        else:
             rsi[:, :] = 50.0
        return rsi

    def _tensor_stochastic_oscillator(self, high_prices: torch.Tensor, low_prices: torch.Tensor, close_prices: torch.Tensor, k_period: int, d_period: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates Stochastic Oscillator (%K and %D) using tensor operations.
        high_prices, low_prices, close_prices: (batch_size, sequence_length)
        k_period: %K period
        d_period: %D period (smoothing for %K)
        Returns: Tuple of (%K, %D) tensors, each (batch_size, sequence_length)
        """
        batch_size, current_sequence_length = close_prices.shape # MODIFIED: Get sequence_length from input tensor
        device = close_prices.device

        if current_sequence_length < k_period:
            self.logger.warning(f"Stochastic %K: seq_len {current_sequence_length} < k_period {k_period}. Returning 50s.")
            percent_k = torch.full_like(close_prices, 50.0)
        else:
            # Rolling min/max for L_k and H_k
            # Pad to ensure causal calculation and output same sequence_length
            padding = k_period - 1
            
            padded_high = F.pad(high_prices.unsqueeze(1), (padding, 0), mode='replicate') # (B, 1, S+P)
            h_k = F.max_pool1d(padded_high, kernel_size=k_period, stride=1).squeeze(1) # (B,S)

            padded_low = F.pad(low_prices.unsqueeze(1), (padding, 0), mode='replicate') # (B, 1, S+P)
            l_k = -F.max_pool1d(-padded_low, kernel_size=k_period, stride=1).squeeze(1) # (B,S)

            percent_k = (close_prices - l_k) / (h_k - l_k + 1e-9) * 100.0
            percent_k[h_k == l_k] = 50.0 # Avoid NaN; if H_k=L_k, range is 0, set %K to neutral 50
            percent_k.clamp_(0, 100) # Ensure %K is within [0, 100]
            # Corrected unstable period for %K: first `k_period-1` values are affected by rolling min/max.
            if current_sequence_length >= k_period:
                percent_k[:, :k_period-1] = 50.0
            else:
                percent_k[:,:] = 50.0

        min_len_for_stable_d = k_period + d_period - 2

        if current_sequence_length < d_period:
            self.logger.warning(f"Stochastic %D: seq_len {current_sequence_length} for %K input is less than d_period {d_period}. Returning 50s for %D.")
            percent_d = torch.full_like(close_prices, 50.0)
        else:
            percent_d = self._rolling_mean_for_indicators(percent_k, d_period)
            if current_sequence_length > min_len_for_stable_d:
                 percent_d[:, :min_len_for_stable_d] = 50.0
            else:
                 percent_d[:, :] = 50.0

        return percent_k, percent_d

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        """
        Processes tensor features for a single asset to generate reversal signals.
        asset_features: (batch_size, sequence_length, num_features)
                        Expected feature order: self.CLOSE_IDX, self.HIGH_IDX, self.LOW_IDX
        Returns: (batch_size, 1, 1) signal tensor.
        """
        batch_size, sequence_length, num_features = asset_features.shape
        device = asset_features.device

        # Get parameters from config first
        rsi_period = self.params.get('rsi_period', 14)
        rsi_oversold = self.params.get('rsi_oversold', 30.0)
        rsi_overbought = self.params.get('rsi_overbought', 70.0)
        
        use_stochastic = self.params.get('use_stochastic_confirmation', True)
        stoch_k_period = self.params.get('stoch_k_period', 14)
        stoch_d_period = self.params.get('stoch_d_period', 3)
        stoch_oversold = self.params.get('stoch_oversold', 20.0)
        stoch_overbought = self.params.get('stoch_overbought', 80.0)

        # Check for minimum sequence length required for calculations
        min_len_rsi = rsi_period 
        min_len_stoch_k = stoch_k_period
        min_len_stoch_d = (stoch_k_period - 1) + stoch_d_period if stoch_k_period > 0 else stoch_d_period
        
        required_len = min_len_rsi
        if use_stochastic:
            required_len = max(required_len, min_len_stoch_k, min_len_stoch_d)
        
        if use_stochastic and sequence_length < 2:
             self.logger.warning(f"{self.config.name}: Stochastic confirmation needs sequence_length >= 2 for crossover. Got {sequence_length}. Returning zero signal.")
             return torch.zeros((batch_size, 1, 1), device=device)

        if sequence_length < required_len:
            self.logger.warning(f"{self.config.name}: Sequence length ({sequence_length}) is less than required ({required_len}) for configured periods. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)
        
        # Ensure feature indices are within bounds
        if not (self.CLOSE_IDX < num_features and self.HIGH_IDX < num_features and self.LOW_IDX < num_features):
            self.logger.error(f"{self.config.name}: Feature indices out of bounds. Num_features: {num_features}, Required indices: C={self.CLOSE_IDX},H={self.HIGH_IDX},L={self.LOW_IDX}")
            return torch.zeros((batch_size, 1, 1), device=device)

        close_prices = asset_features[:, :, self.CLOSE_IDX]  # (batch_size, sequence_length)
        high_prices = asset_features[:, :, self.HIGH_IDX]    # (batch_size, sequence_length)
        low_prices = asset_features[:, :, self.LOW_IDX]      # (batch_size, sequence_length)

        # Calculate RSI
        rsi = self._tensor_rsi(close_prices, rsi_period) # (batch_size, sequence_length)
        last_rsi = rsi[:, -1] # (batch_size)

        signal = torch.zeros(batch_size, device=device)

        # Base RSI conditions for the last time step
        rsi_buy_condition = last_rsi < rsi_oversold
        rsi_sell_condition = last_rsi > rsi_overbought

        if use_stochastic:
            percent_k, percent_d = self._tensor_stochastic_oscillator(high_prices, low_prices, close_prices, stoch_k_period, stoch_d_period)
            
            last_k = percent_k[:, -1] # (batch_size)
            last_d = percent_d[:, -1] # (batch_size)
            # Need previous values for crossover detection
            prev_k = percent_k[:, -2] if sequence_length > 1 else last_k # Handle seq_len=1 for prev
            prev_d = percent_d[:, -2] if sequence_length > 1 else last_d

            # Stochastic conditions for the last time step
            stoch_is_oversold_area = (last_k < stoch_oversold) & (last_d < stoch_oversold)
            stoch_is_overbought_area = (last_k > stoch_overbought) & (last_d > stoch_overbought)
            
            # Crossover conditions for Stochastic
            # Buy: %K crosses above %D in oversold zone
            stoch_buy_crossover = (last_k > last_d) & (prev_k <= prev_d) & stoch_is_oversold_area
            # Sell: %K crosses below %D in overbought zone
            stoch_sell_crossover = (last_k < last_d) & (prev_k >= prev_d) & stoch_is_overbought_area
            
            # Combined signals
            final_buy_condition = rsi_buy_condition & stoch_buy_crossover 
            final_sell_condition = rsi_sell_condition & stoch_sell_crossover
            
            signal[final_buy_condition] = 1.0
            signal[final_sell_condition] = -1.0
        else:
            # RSI only signals
            signal[rsi_buy_condition] = 1.0
            signal[rsi_sell_condition] = -1.0
            
        return signal.view(batch_size, 1, 1)

    # def _calculate_rsi(self, series: pd.Series, period: int) -> Optional[pd.Series]:
    #     # ... (old pandas RSI commented out) ...

    # def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int, d_period: int) -> Optional[pd.DataFrame]:
    #     # ... (old pandas Stochastic commented out) ...

    # def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]: # OLD PANDAS FORWARD
    #     # ... (old pandas forward commented out) ...

    # def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    #     # ... (old pandas generate_signals commented out) ...

class MeanReversionStrategy(BaseStrategy):
    """å‡å€¼åè½‰ç­–ç•¥ï¼šåŸºæ–¼åƒ¹æ ¼å’ŒæŒ‡æ¨™çš„è¶…è²·è¶…è³£"""

    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="MeanReversionStrategy",
            description="Exploits deviations from mean price or indicator levels.",
            default_params={
                'lookback_window': 20, 
                'entry_threshold': 2.0, 
                'exit_threshold': 0.5, 
                'rsi_period': 14, 
                'rsi_oversold': 30.0, 
                'rsi_overbought': 70.0, 
                'asset_list': []
            }
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config=config, params=params, logger=logger)
        self.CLOSE_IDX = self.params.get('close_idx', 0) # Make close index configurable

    def _rolling_mean(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """ Helper for rolling mean using avg_pool1d. Assumes tensor is (batch, seq_len). """
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(1) # (batch, 1, seq_len)
        
        # Pad to ensure output has same seq_len and means are 'trailing'
        padding = window_size - 1
        padded_tensor = F.pad(tensor, (padding, 0), mode='replicate')
        pooled = F.avg_pool1d(padded_tensor, kernel_size=window_size, stride=1)
        return pooled.squeeze(1) # (batch, seq_len)

    def _rolling_std(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """ Helper for rolling std. Assumes tensor is (batch, seq_len). """
        # This version is copied from BreakoutStrategy and assumes tensor is 2D.
        # self._rolling_mean is expected to handle 2D input (batch, seq_len)
        mean_x = self._rolling_mean(tensor, window_size) 
        mean_x_sq = self._rolling_mean(tensor**2, window_size)
        # Clamp to avoid sqrt of negative numbers due to precision issues
        variance = (mean_x_sq - mean_x**2).clamp(min=1e-6) # Ensure variance is non-negative
        return torch.sqrt(variance)

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        """
        Processes tensor features for a single asset to generate mean reversion signals.
        asset_features: (batch_size, sequence_length, num_features)
                        Expected feature order: self.CLOSE_IDX (configurable)
        Returns: (batch_size, 1, 1) signal tensor.
        """
        # CLOSE_IDX = 0 # Assuming close price is the first feature # Now using self.CLOSE_IDX
        batch_size, sequence_length, num_features = asset_features.shape
        device = asset_features.device

        lookback_window = self.params.get('lookback_window', 20)
        entry_threshold = self.params.get('entry_threshold', 2.0)
        exit_threshold = self.params.get('exit_threshold', 0.5)

        if sequence_length < lookback_window:
            self.logger.warning(f"{self.config.name}: Sequence length ({sequence_length}) is less than lookback_window ({lookback_window}). Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        if self.CLOSE_IDX >= num_features:
            self.logger.error(f"{self.config.name}: CLOSE_IDX {self.CLOSE_IDX} is out of bounds for num_features {num_features}.")
            return torch.zeros((batch_size, 1, 1), device=device)

        close_prices = asset_features[:, :, self.CLOSE_IDX]  # (batch_size, sequence_length)

        # Calculate mean and std dev over the lookback window
        mean_price = self._rolling_mean(close_prices, lookback_window) # (batch_size, sequence_length)
        std_dev_price = self._rolling_std(close_prices, lookback_window) # (batch_size, sequence_length)

        # We are interested in the signals for the most recent time step
        last_close = close_prices[:, -1]
        last_mean = mean_price[:, -1]
        last_std_dev = std_dev_price[:, -1]

        # Entry conditions: Price is beyond entry_threshold std devs from the mean
        long_entry = (last_close < last_mean - entry_threshold * last_std_dev)
        short_entry = (last_close > last_mean + entry_threshold * last_std_dev)

        # Exit conditions: Price has reverted back within exit_threshold std devs from the mean
        long_exit = (last_close > last_mean - exit_threshold * last_std_dev)
        short_exit = (last_close < last_mean + exit_threshold * last_std_dev)
        
        signal = torch.zeros(batch_size, device=device)
        signal[long_entry] = 1.0
        signal[short_entry] = -1.0

        # Implementing a basic position exit logic
        # This is a placeholder and should be replaced with proper position management
        if current_positions is not None:
            # Exit long positions if price has reverted
            exit_long = (current_positions[:, 0] > 0) & long_exit
            signal[exit_long] = 0.0
            
            # Exit short positions if price has reverted
            exit_short = (current_positions[:, 0] < 0) & short_exit
            signal[exit_short] = 0.0

        return signal.view(batch_size, 1, 1)



