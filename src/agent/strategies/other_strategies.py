# src/agent/strategies/other_strategies.py
from .base_strategy import BaseStrategy, StrategyConfig
import pandas as pd # Keep for type hints if any legacy methods remain, but avoid in tensor forward paths
import numpy as np # Keep for type hints or rare cases, avoid in tensor forward paths
import ta # Try to replace with tensor equivalents
from typing import Dict, Any, Optional, Tuple
import logging
import torch
import torch.nn.functional as F

# Tensor-based helper functions
# These could be part of a shared utility module in a larger system

def _rolling_mean_tensor(data: torch.Tensor, window: int) -> torch.Tensor:
    """Calculates rolling mean on a 2D tensor (batch_size, seq_len) or 3D (batch_size, 1, seq_len)."""
    if data.ndim == 2:
        data_unsqueezed = data.unsqueeze(1)  # (B, 1, S)
    elif data.ndim == 3 and data.shape[1] == 1:
        data_unsqueezed = data
    else:
        raise ValueError(f"_rolling_mean_tensor expects 2D (B,S) or 3D (B,1,S) tensor, got {data.shape}")

    if data_unsqueezed.shape[-1] == 0: # Handle empty sequence
        return torch.zeros_like(data_unsqueezed).squeeze(1) if data.ndim == 2 else torch.zeros_like(data_unsqueezed)
    
    padding_mode = 'replicate' # Pad with the first/last value
    # Calculate padding: if seq_len < window, pad to effectively use available data.
    # If seq_len >= window, standard padding is window - 1.
    if data_unsqueezed.shape[-1] < window:
        # Pad to length `window -1` at the beginning so conv1d can run. 
        # The result will be based on fewer than `window` points for initial part of sequence.
        # This is a choice; another might be to return NaNs/zeros or require seq_len >= window.
        # F.avg_pool1d handles this by effectively reducing window size for initial elements.
        # Using conv1d with replicate padding and full window weight is one way to approximate.
        padding_val = window - 1 # Standard padding for conv1d to produce output of same length as input
    else:
        padding_val = window - 1

    padded_data = F.pad(data_unsqueezed, (padding_val, 0), mode=padding_mode)
    weights = torch.full((1, 1, window), 1.0/window, device=data.device, dtype=data.dtype)
    rolled_mean = F.conv1d(padded_data, weights, stride=1)
    
    return rolled_mean.squeeze(1) if data.ndim == 2 else rolled_mean

def _rolling_std_tensor(data: torch.Tensor, window: int) -> torch.Tensor:
    """Calculates rolling std dev on a 2D tensor (batch_size, seq_len)."""
    mean_x = _rolling_mean_tensor(data, window)
    mean_x_sq = _rolling_mean_tensor(data**2, window)
    variance = (mean_x_sq - mean_x**2).clamp(min=1e-9) # Add clamp for numerical stability
    return torch.sqrt(variance)

def _atr_tensor(high: torch.Tensor, low: torch.Tensor, close: torch.Tensor, window: int) -> torch.Tensor:
    """Calculates Average True Range (ATR) using PyTorch. high, low, close are (batch_size, seq_len)."""
    if high.shape[1] == 0: # Handle empty sequence
        return torch.zeros_like(high)

    # True Range calculation
    tr1 = high - low
    # close_prev: prepend first close to close_prices[:, :-1]
    close_prev = torch.cat((close[:, :1], close[:, :-1]), dim=1)
    tr2 = torch.abs(high - close_prev)
    tr3 = torch.abs(low - close_prev)
    
    true_range = torch.max(torch.max(tr1, tr2), tr3) # (batch_size, seq_len)
    
    # ATR is typically a Wilder's Smoothing of True Range. 
    # For simplicity, using SMA here via _rolling_mean_tensor.
    # A more accurate Wilder's MA would use an EMA-like calculation.
    atr = _rolling_mean_tensor(true_range, window)
    return atr

def _roc_tensor(close: torch.Tensor, window: int) -> torch.Tensor:
    """Calculates Rate of Change (ROC) using PyTorch. close is (batch_size, seq_len)."""
    if close.shape[1] <= window: # Not enough data for ROC over this window
        return torch.zeros_like(close)
    
    close_n_periods_ago = close[:, :-(window)]
    roc = (close[:, window:] - close_n_periods_ago) / close_n_periods_ago.clamp(min=1e-9)
    # Pad with zeros at the beginning to match original shape
    roc_padded = F.pad(roc, (window, 0), mode='constant', value=0)
    return roc_padded

def _sma_tensor(data: torch.Tensor, window: int) -> torch.Tensor:
    """Alias for _rolling_mean_tensor for clarity when used as SMA."""
    return _rolling_mean_tensor(data, window)

# Simplified RSI (using SMAs for gains/losses instead of Wilder's MA)
def _rsi_tensor_simplified(close: torch.Tensor, window: int) -> torch.Tensor:
    if close.shape[1] <= 1: return torch.full_like(close, 50.0) # Neutral RSI for short sequences
    delta = close[:, 1:] - close[:, :-1]
    gain = delta.clamp(min=0)
    loss = -delta.clamp(max=0)

    # Pad gain and loss to be same length as original sequence for rolling mean
    gain_padded = F.pad(gain, (1,0), 'constant', 0)
    loss_padded = F.pad(loss, (1,0), 'constant', 0)

    avg_gain = _rolling_mean_tensor(gain_padded, window)
    avg_loss = _rolling_mean_tensor(loss_padded, window)
    
    rs = avg_gain / avg_loss.clamp(min=1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[avg_loss == 0] = 100.0 # If avg_loss is 0, RSI is 100
    rsi[avg_gain == 0] = 0.0 # If avg_gain is 0 (and avg_loss is not), RSI is 0
    # Handle case where both are zero (e.g. flat price for window) -> RSI is often 50 or undefined, let's use 50
    rsi[(avg_gain == 0) & (avg_loss == 0)] = 50.0
    return rsi

# Simplified Stochastic Oscillator
def _stochastic_oscillator_tensor(high: torch.Tensor, low: torch.Tensor, close: torch.Tensor, k_window: int, d_window: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if high.shape[1] < k_window or high.shape[1] == 0:
        return torch.full_like(close, 50.0), torch.full_like(close, 50.0) # Neutral if not enough data

    # Rolling min/max using unfold (more complex but true rolling)
    # For simplicity, if unfold is too much, this could be an area for future improvement or use a simpler proxy.
    # Let's use a simpler approach for now: calculate over the full window for the last point, not a true rolling series for all points.
    # This is a significant simplification for a true stochastic over sequence.
    # A full rolling min/max is needed for a proper %K series.
    # For now, let's calculate for the last point only, which is what the signal is based on.
    
    # To get a series for %K, we need rolling min_low and rolling max_high.
    # This requires unfold or a loop. Let's try with unfold for a more correct %K series.
    # high_unfolded = high.unfold(dimension=1, size=k_window, step=1)
    # low_unfolded = low.unfold(dimension=1, size=k_window, step=1)
    # rolling_max_high = high_unfolded.max(dim=2).values
    # rolling_min_low = low_unfolded.min(dim=2).values
    # percent_k_series = 100 * (close[:, k_window-1:] - rolling_min_low) / (rolling_max_high - rolling_min_low).clamp(min=1e-9)
    # percent_k_padded = F.pad(percent_k_series, (k_window-1, 0), 'constant', 50.0) # Pad to original length

    # Simpler approach for now: calculate for each point using available window (less accurate for initial points)
    percent_k_list = []
    for t in range(high.shape[1]):
        start_idx = max(0, t - k_window + 1)
        window_high = high[:, start_idx : t+1]
        window_low = low[:, start_idx : t+1]
        current_close = close[:, t]
        
        highest_high = window_high.max(dim=1).values
        lowest_low = window_low.min(dim=1).values
        
        numerator = current_close - lowest_low
        denominator = (highest_high - lowest_low).clamp(min=1e-9)
        
        pk = 100 * numerator / denominator
        pk[denominator < 1e-8] = 50.0 # If range is zero, stoch is often 50 or previous value
        percent_k_list.append(pk.unsqueeze(1))
    
    percent_k_padded = torch.cat(percent_k_list, dim=1)
    percent_d = _sma_tensor(percent_k_padded, d_window)
    return percent_k_padded, percent_d

def _bollinger_bands_tensor(close: torch.Tensor, window: int, num_std_dev: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sma = _sma_tensor(close, window)
    rolling_std = _rolling_std_tensor(close, window)
    upper_band = sma + (rolling_std * num_std_dev)
    lower_band = sma - (rolling_std * num_std_dev)
    return upper_band, lower_band, sma

class OptionFlowStrategy(BaseStrategy):
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="OptionFlowStrategy",
            description="Simplified: Uses high volume and volatility as a proxy for significant option activity.",
            default_params={
                'volume_period': 20, 
                'volatility_period': 14, 
                'volume_z_threshold': 2.0, 
                'atr_threshold_multiplier': 1.5,
                'high_idx': 1, 'low_idx': 2, 'close_idx': 3, 'volume_idx': 4
            },
            applicable_assets=[]
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.volume_period = self.params.get('volume_period', 20)
        self.volatility_period = self.params.get('volatility_period', 14)
        self.volume_z_threshold = self.params.get('volume_z_threshold', 2.0)
        self.atr_threshold_multiplier = self.params.get('atr_threshold_multiplier', 1.5)
        self.high_idx = self.params.get('high_idx', 1)
        self.low_idx = self.params.get('low_idx', 2)
        self.close_idx = self.params.get('close_idx', 3)
        self.volume_idx = self.params.get('volume_idx', 4)

        if self.config.input_dim is not None:
            max_idx = max(self.high_idx, self.low_idx, self.close_idx, self.volume_idx)
            if max_idx >= self.config.input_dim:
                self.logger.error(f"[{self.config.name}] Feature indices OOB for input_dim {self.config.input_dim}.")
        else:
            self.logger.warning(f"[{self.config.name}] input_dim not specified. Index validation skipped.")

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, num_features = asset_features.shape
        device = asset_features.device

        if self.config.input_dim and max(self.high_idx, self.low_idx, self.close_idx, self.volume_idx) >= num_features:
            self.logger.error(f"[{self.config.name}] Feature indices OOB for actual features {num_features}. Zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)
        
        # Placeholder: Actual logic for OptionFlowStrategy
        # For now, returning a neutral signal
        return torch.zeros((batch_size, 1, 1), device=device) # (batch_size, num_assets, signal_dim)

class MicrostructureStrategy(BaseStrategy):
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="MicrostructureStrategy",
            description="Analyzes order book imbalances, bid-ask spreads, and trade frequency.",
            default_params={
                'spread_threshold': 0.0005, 
                'imbalance_threshold': 1.5,
                'depth_levels': 5, # Number of order book levels to consider
                'close_idx': 3, # Assuming close price is at index 3
                # Indices for bid/ask prices and volumes if available in features
                # 'bid_price_idx_start': 5, 'ask_price_idx_start': 10, 
                # 'bid_volume_idx_start': 15, 'ask_volume_idx_start': 20
            },
            applicable_assets=[]
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.spread_threshold = self.params.get('spread_threshold', 0.0005)
        self.imbalance_threshold = self.params.get('imbalance_threshold', 1.5)
        self.depth_levels = self.params.get('depth_levels', 5)
        self.close_idx = self.params.get('close_idx', 3)
        # Example: self.bid_price_idx_start = self.params.get('bid_price_idx_start')

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, num_features = asset_features.shape
        device = asset_features.device
        # Placeholder: Actual logic for MicrostructureStrategy
        # This would require order book data, which is not typically in OHLCV features.
        # For now, returning a neutral signal.
        # self.logger.warning(f"[{self.config.name}] Requires order book data, not implemented with current features. Neutral signal.")
        return torch.zeros((batch_size, 1, 1), device=device)

class CarryTradeStrategy(BaseStrategy):
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="CarryTradeStrategy",
            description="Exploits interest rate differentials between currencies (simplified for general assets).",
            default_params={
                'interest_rate_diff_threshold': 0.01, # Example threshold
                'funding_rate_idx': -1, # Placeholder: index for funding/interest rate data if available
                'volatility_period': 20,
                'volatility_cap': 0.02, # Max daily volatility to consider trade
                'close_idx': 3, 'high_idx': 1, 'low_idx': 2
            },
            applicable_assets=[] # Typically FX pairs
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.interest_rate_diff_threshold = self.params.get('interest_rate_diff_threshold', 0.01)
        self.funding_rate_idx = self.params.get('funding_rate_idx', -1)
        self.volatility_period = self.params.get('volatility_period', 20)
        self.volatility_cap = self.params.get('volatility_cap', 0.02)
        self.close_idx = self.params.get('close_idx', 3)
        self.high_idx = self.params.get('high_idx', 1)
        self.low_idx = self.params.get('low_idx', 2)

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, num_features = asset_features.shape
        device = asset_features.device

        if self.funding_rate_idx < 0 or self.funding_rate_idx >= num_features:
            # self.logger.warning(f"[{self.config.name}] Funding rate index not valid or data unavailable. Neutral signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        # Placeholder: Actual logic for CarryTradeStrategy
        # funding_rate = asset_features[:, -1, self.funding_rate_idx]
        # close_prices = asset_features[:, :, self.close_idx]
        # high_prices = asset_features[:, :, self.high_idx]
        # low_prices = asset_features[:, :, self.low_idx]
        # atr = _atr_tensor(high_prices, low_prices, close_prices, self.volatility_period)[:, -1]
        # daily_volatility_proxy = atr / close_prices[:, -1].clamp(min=1e-9)
        # signal = torch.zeros((batch_size, 1, 1), device=device)
        # signal[(funding_rate > self.interest_rate_diff_threshold) & (daily_volatility_proxy < self.volatility_cap)] = 1.0 # Buy signal
        # signal[(funding_rate < -self.interest_rate_diff_threshold) & (daily_volatility_proxy < self.volatility_cap)] = -1.0 # Sell signal
        return torch.zeros((batch_size, 1, 1), device=device)

class MacroEconomicStrategy(BaseStrategy):
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="MacroEconomicStrategy",
            description="Trades based on macroeconomic indicators (e.g., inflation, GDP growth, unemployment).",
            default_params={
                'indicator_idx': -1, # Placeholder: index for a relevant macro indicator
                'threshold_high': 0.5, 
                'threshold_low': -0.5
            },
            applicable_assets=[] # Broad market indices, currencies
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.indicator_idx = self.params.get('indicator_idx', -1)
        self.threshold_high = self.params.get('threshold_high', 0.5)
        self.threshold_low = self.params.get('threshold_low', -0.5)

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, num_features = asset_features.shape
        device = asset_features.device

        if self.indicator_idx < 0 or self.indicator_idx >= num_features:
            # self.logger.warning(f"[{self.config.name}] Macro indicator index not valid or data unavailable. Neutral signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        # Placeholder: Actual logic for MacroEconomicStrategy
        # indicator_value = asset_features[:, -1, self.indicator_idx]
        # signal = torch.zeros((batch_size, 1, 1), device=device)
        # signal[indicator_value > self.threshold_high] = 1.0
        # signal[indicator_value < self.threshold_low] = -1.0
        return torch.zeros((batch_size, 1, 1), device=device)

class EventDrivenStrategy(BaseStrategy):
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="EventDrivenStrategy",
            description="Trades on specific events like earnings announcements, mergers, etc.",
            default_params={
                'event_impact_threshold': 0.7, # Arbitrary scale of event impact
                'event_indicator_idx': -1 # Index for feature indicating event proximity/impact
            },
            applicable_assets=[] # Specific stocks usually
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.event_impact_threshold = self.params.get('event_impact_threshold', 0.7)
        self.event_indicator_idx = self.params.get('event_indicator_idx', -1)

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, num_features = asset_features.shape
        device = asset_features.device

        if self.event_indicator_idx < 0 or self.event_indicator_idx >= num_features:
            # self.logger.warning(f"[{self.config.name}] Event indicator index not valid or data unavailable. Neutral signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        # Placeholder: Actual logic for EventDrivenStrategy
        # event_signal = asset_features[:, -1, self.event_indicator_idx] # Assuming higher is more impactful
        # signal = torch.zeros((batch_size, 1, 1), device=device)
        # signal[event_signal > self.event_impact_threshold] = 1.0 # Example: Go long on positive event
        # signal[event_signal < -self.event_impact_threshold] = -1.0 # Example: Go short on negative event
        return torch.zeros((batch_size, 1, 1), device=device)

class SentimentStrategy(BaseStrategy):
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="SentimentStrategy",
            description="Uses sentiment data (e.g., from news, social media) to make trading decisions.",
            default_params={
                'sentiment_idx': -1, # Index for sentiment score feature
                'positive_threshold': 0.6,
                'negative_threshold': 0.2
            },
            applicable_assets=[]
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.sentiment_idx = self.params.get('sentiment_idx', -1)
        self.positive_threshold = self.params.get('positive_threshold', 0.6)
        self.negative_threshold = self.params.get('negative_threshold', 0.2)

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, num_features = asset_features.shape
        device = asset_features.device

        if self.sentiment_idx < 0 or self.sentiment_idx >= num_features:
            # self.logger.warning(f"[{self.config.name}] Sentiment index not valid or data unavailable. Neutral signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        # Placeholder: Actual logic for SentimentStrategy
        # sentiment_score = asset_features[:, -1, self.sentiment_idx]
        # signal = torch.zeros((batch_size, 1, 1), device=device)
        # signal[sentiment_score > self.positive_threshold] = 1.0
        # signal[sentiment_score < self.negative_threshold] = -1.0
        return torch.zeros((batch_size, 1, 1), device=device)

class QuantitativeStrategy(BaseStrategy): # Generic Quantitative
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="QuantitativeStrategy",
            description="A generic quantitative strategy based on a combination of factors.",
            default_params={
                'factor1_idx': 0, 'factor2_idx': 1, # Example factor indices
                'weight1': 0.5, 'weight2': 0.5
            },
            applicable_assets=[]
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.factor1_idx = self.params.get('factor1_idx', 0)
        self.factor2_idx = self.params.get('factor2_idx', 1)
        self.weight1 = self.params.get('weight1', 0.5)
        self.weight2 = self.params.get('weight2', 0.5)

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, num_features = asset_features.shape
        device = asset_features.device

        if self.factor1_idx >= num_features or self.factor2_idx >= num_features:
            # self.logger.error(f"[{self.config.name}] Factor indices out of bounds. Neutral signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        # Placeholder: Actual logic for QuantitativeStrategy
        # factor1_value = asset_features[:, -1, self.factor1_idx]
        # factor2_value = asset_features[:, -1, self.factor2_idx]
        # combined_signal = self.weight1 * factor1_value + self.weight2 * factor2_value
        # For simplicity, let's assume combined_signal is already scaled between -1 and 1
        # signal = combined_signal.unsqueeze(-1).unsqueeze(-1) # (batch_size, 1, 1)
        return torch.zeros((batch_size, 1, 1), device=device)

class MarketMakingStrategy(BaseStrategy):
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="MarketMakingStrategy",
            description="Provides liquidity by placing bid and ask orders around the perceived fair value.",
            default_params={
                'spread_factor': 0.001, 
                'order_size': 100,
                'fair_value_period': 20, # Period for calculating fair value (e.g., SMA of mid-price)
                'close_idx': 3 # Assuming close price is a proxy for mid-price if order book not available
            },
            applicable_assets=[]
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.spread_factor = self.params.get('spread_factor', 0.001)
        self.order_size = self.params.get('order_size', 100)
        self.fair_value_period = self.params.get('fair_value_period', 20)
        self.close_idx = self.params.get('close_idx', 3)

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, num_features = asset_features.shape
        device = asset_features.device

        # Placeholder: Actual logic for MarketMakingStrategy
        # This strategy is complex and typically requires real-time order book data and execution capabilities.
        # For a simulation with OHLCV, it's highly simplified.
        # self.logger.warning(f"[{self.config.name}] MarketMaking is highly simplified. Neutral signal.")
        # fair_value_proxy = _sma_tensor(asset_features[:, :, self.close_idx], self.fair_value_period)[:, -1]
        # bid_price = fair_value_proxy * (1 - self.spread_factor)
        # ask_price = fair_value_proxy * (1 + self.spread_factor)
        # For SAC, we need a target position or action. This strategy doesn't directly map to that well.
        # Let's return a neutral signal as it's more about order placement than directional bets.
        return torch.zeros((batch_size, 1, 1), device=device)

class HighFrequencyStrategy(BaseStrategy):
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="HighFrequencyStrategy",
            description="Exploits very short-term market inefficiencies, often latency-sensitive.",
            default_params={
                'latency_proxy_idx': -1, # Index for a feature that might proxy for latency arbitrage opps
                'threshold': 0.0001
            },
            applicable_assets=[]
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.latency_proxy_idx = self.params.get('latency_proxy_idx', -1)
        self.threshold = self.params.get('threshold', 0.0001)

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, num_features = asset_features.shape
        device = asset_features.device
        # Placeholder: Actual logic for HighFrequencyStrategy
        # self.logger.warning(f"[{self.config.name}] HFT is highly simplified and data-dependent. Neutral signal.")
        return torch.zeros((batch_size, 1, 1), device=device)

class AlgorithmicStrategy(BaseStrategy): # Generic Algorithmic
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="AlgorithmicStrategy",
            description="A general algorithmic strategy placeholder.",
            default_params={},
            applicable_assets=[]
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, num_features = asset_features.shape
        device = asset_features.device
        # Placeholder: Actual logic for AlgorithmicStrategy
        return torch.zeros((batch_size, 1, 1), device=device)
