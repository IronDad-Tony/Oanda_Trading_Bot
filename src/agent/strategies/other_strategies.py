# src/agent/strategies/other_strategies.py
from .base_strategy import BaseStrategy, StrategyConfig
import pandas as pd
import numpy as np
import ta
from typing import Dict, Any, Optional # Updated typing import
import logging # Added logging import
import torch # Added for PyTorch tensors

# Helper function for tensor-based rolling operations if needed (or stick to pd for ta)
# For now, we will convert to pd.DataFrame to use `ta` and then convert back.

class OptionFlowStrategy(BaseStrategy):
    """ 
    Simplified: Uses high volume and volatility as a proxy for significant option activity.
    Ideally, this strategy would use actual option chain data (volume, open interest, greeks).
    """
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="OptionFlowStrategy",
            description="Simplified: Uses high volume and volatility as a proxy for significant option activity.",
            default_params={
                'instrument_key': None, 
                'volume_period': 20, 
                'volatility_period': 14, 
                'volume_z_threshold': 2.0, 
                'atr_threshold_multiplier': 1.5,
                'feature_indices': {'High': 1, 'Low': 2, 'Close': 3, 'Volume': 4} # Example indices
            },
            applicable_assets=[]
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.instrument_key = self.params.get('instrument_key')

        if self.instrument_key is None and self.config.applicable_assets:
            self.instrument_key = self.config.applicable_assets[0]
            self.logger.info(f"{self.config.name}: 'instrument_key' not in params, using first from config.applicable_assets: {self.instrument_key}")

        if self.instrument_key is None:
            self.logger.error(f"{self.config.name} requires 'instrument_key' in params or config.applicable_assets to be non-empty.")
        elif not self.config.applicable_assets: # If instrument_key is set, ensure it's in applicable_assets
            self.config.applicable_assets = [self.instrument_key]
        
        # Use self.effective_params which is set by BaseStrategy
        self.volume_period = self.effective_params.get('volume_period', 20)
        self.volatility_period = self.effective_params.get('volatility_period', 14)
        self.volume_z_threshold = self.effective_params.get('volume_z_threshold', 2.0)
        self.atr_threshold_multiplier = self.effective_params.get('atr_threshold_multiplier', 1.5)
        self.feature_indices = self.effective_params.get('feature_indices', {'High': 1, 'Low': 2, 'Close': 3, 'Volume': 4})

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, _ = asset_features.shape
        signals_batch = torch.zeros(batch_size, seq_len, 1, device=asset_features.device)

        high_idx = self.feature_indices['High']
        low_idx = self.feature_indices['Low']
        close_idx = self.feature_indices['Close']
        vol_idx = self.feature_indices['Volume']

        for i in range(batch_size):
            df = pd.DataFrame({
                'High': asset_features[i, :, high_idx].cpu().numpy(),
                'Low': asset_features[i, :, low_idx].cpu().numpy(),
                'Close': asset_features[i, :, close_idx].cpu().numpy(),
                'Volume': asset_features[i, :, vol_idx].cpu().numpy()
            })

            if df.empty or not all(c in df.columns for c in ['High', 'Low', 'Close', 'Volume']):
                continue

            df['avg_volume'] = df['Volume'].rolling(window=self.volume_period, min_periods=max(1,self.volume_period//2)).mean()
            df['std_volume'] = df['Volume'].rolling(window=self.volume_period, min_periods=max(1,self.volume_period//2)).std()
            df['volume_z_score'] = (df['Volume'] - df['avg_volume']) / df['std_volume'].replace(0, np.nan)
            
            df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=self.volatility_period, fillna=True)
            df['price_range'] = df['High'] - df['Low']
            df.fillna({'volume_z_score': 0, 'atr': 0, 'price_range': 0}, inplace=True)

            # Original generate_signals logic
            current_signals_np = np.zeros(seq_len)
            high_volume_activity = df['volume_z_score'] > self.volume_z_threshold
            high_volatility_activity = df['price_range'] > (df['atr'] * self.atr_threshold_multiplier)
            
            bullish_condition = high_volume_activity & high_volatility_activity & (df['Close'] > df['Close'].shift(1))
            bearish_condition = high_volume_activity & high_volatility_activity & (df['Close'] < df['Close'].shift(1))

            current_signals_np[bullish_condition.fillna(False)] = 1.0
            current_signals_np[bearish_condition.fillna(False)] = -1.0
            signals_batch[i, :, 0] = torch.tensor(current_signals_np, device=asset_features.device, dtype=asset_features.dtype)
            
        return signals_batch

class MicrostructureStrategy(BaseStrategy):
    """
    Simplified: Uses price spread (High-Low) and volume spikes as proxies for microstructure imbalances.
    Ideally, this strategy would use L1/L2 order book data (bid/ask prices, sizes, order flow).
    """
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="MicrostructureStrategy",
            description="Simplified: Uses price spread and volume spikes as proxies for microstructure imbalances.",
            default_params={
                'instrument_key': None,
                'spread_window': 10,
                'volume_window': 20,
                'spread_quantile_threshold': 0.90,
                'volume_spike_multiplier': 3.0,
                'feature_indices': {'High': 1, 'Low': 2, 'Close': 3, 'Volume': 4} # Example indices
            },
            applicable_assets=[]
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.instrument_key = self.params.get('instrument_key')

        if self.instrument_key is None and self.config.applicable_assets:
            self.instrument_key = self.config.applicable_assets[0]
            self.logger.info(f"{self.config.name}: 'instrument_key' not in params, using first from config.applicable_assets: {self.instrument_key}")

        if self.instrument_key is None:
            self.logger.error(f"{self.config.name} requires 'instrument_key' in params or config.applicable_assets to be non-empty.")
        elif not self.config.applicable_assets:
            self.config.applicable_assets = [self.instrument_key]

        self.spread_window = self.params.get('spread_window', 10)
        self.volume_window = self.params.get('volume_window', 20)
        self.spread_quantile_threshold = self.params.get('spread_quantile_threshold', 0.90)
        self.volume_spike_multiplier = self.params.get('volume_spike_multiplier', 3.0)
        self.feature_indices = self.params.get('feature_indices', {'High': 1, 'Low': 2, 'Close': 3, 'Volume': 4})

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, _ = asset_features.shape
        signals_batch = torch.zeros(batch_size, seq_len, 1, device=asset_features.device)

        high_idx = self.feature_indices['High']
        low_idx = self.feature_indices['Low']
        close_idx = self.feature_indices['Close']
        vol_idx = self.feature_indices['Volume']

        for i in range(batch_size):
            df = pd.DataFrame({
                'High': asset_features[i, :, high_idx].cpu().numpy(),
                'Low': asset_features[i, :, low_idx].cpu().numpy(),
                'Close': asset_features[i, :, close_idx].cpu().numpy(),
                'Volume': asset_features[i, :, vol_idx].cpu().numpy()
            })

            if df.empty or not all(c in df.columns for c in ['High', 'Low', 'Close', 'Volume']):
                continue

            df['spread'] = df['High'] - df['Low']
            df['rolling_median_spread'] = df['spread'].rolling(window=self.spread_window, min_periods=max(1,self.spread_window//2)).median()
            df['avg_volume'] = df['Volume'].rolling(window=self.volume_window, min_periods=max(1,self.volume_window//2)).mean()
            df.fillna({'rolling_median_spread': 0, 'avg_volume': 0, 'spread':0}, inplace=True)

            # Original generate_signals logic
            current_signals_np = np.zeros(seq_len)
            wide_spread = df['spread'] > df['spread'].rolling(window=self.spread_window*2, min_periods=max(1,self.spread_window)).quantile(self.spread_quantile_threshold)
            volume_spike = df['Volume'] > (df['avg_volume'] * self.volume_spike_multiplier)
            
            price_up_sharply = df['Close'] > df['Close'].shift(1) + df['rolling_median_spread']
            price_down_sharply = df['Close'] < df['Close'].shift(1) - df['rolling_median_spread']

            current_signals_np[(wide_spread & volume_spike & price_up_sharply).fillna(False)] = -1.0 # Sell
            current_signals_np[(wide_spread & volume_spike & price_down_sharply).fillna(False)] = 1.0  # Buy
            signals_batch[i, :, 0] = torch.tensor(current_signals_np, device=asset_features.device, dtype=asset_features.dtype)
            
        return signals_batch

class CarryTradeStrategy(BaseStrategy):
    """
    Simplified: Uses momentum as a proxy for chasing yield/currency strength.
    Ideally, this strategy would use actual interest rate differentials between currency pairs.
    For a single instrument, this translates to trend following if we assume higher yield attracts capital.
    """
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="CarryTradeStrategy",
            description="Simplified: Uses momentum as a proxy for chasing yield/currency strength.",
            default_params={'short_ma_period': 10, 'long_ma_period': 30,
                            'feature_indices': {'Close': 3}}, # Example index
            applicable_assets=[] # This strategy expects applicable_assets to be set in config or instance
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.short_ma_period = self.params.get('short_ma_period', 10)
        self.long_ma_period = self.params.get('long_ma_period', 30)
        self.feature_indices = self.params.get('feature_indices', {'Close': 3})

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, _ = asset_features.shape
        signals_batch = torch.zeros(batch_size, seq_len, 1, device=asset_features.device)
        close_idx = self.feature_indices['Close']

        for i in range(batch_size):
            # This strategy can apply to multiple assets if asset_features is structured accordingly.
            # For now, assuming asset_features is for a single asset type per call, or needs selection.
            # If this strategy instance is for ONE asset, this is fine.
            # If it's meant to process multiple assets from a single asset_features tensor, logic would need asset indexing.
            df = pd.DataFrame({'Close': asset_features[i, :, close_idx].cpu().numpy()})

            if df.empty or 'Close' not in df.columns:
                continue

            df['short_ma'] = ta.trend.sma_indicator(df['Close'], window=self.short_ma_period, fillna=True)
            df['long_ma'] = ta.trend.sma_indicator(df['Close'], window=self.long_ma_period, fillna=True)

            # Original generate_signals logic
            current_signals_np = np.zeros(seq_len)
            buy_condition = (df['short_ma'] > df['long_ma']) & (df['short_ma'].shift(1) <= df['long_ma'].shift(1))
            sell_condition = (df['short_ma'] < df['long_ma']) & (df['short_ma'].shift(1) >= df['long_ma'].shift(1))

            current_signals_np[buy_condition.fillna(False)] = 1.0
            current_signals_np[sell_condition.fillna(False)] = -1.0
            signals_batch[i, :, 0] = torch.tensor(current_signals_np, device=asset_features.device, dtype=asset_features.dtype)
            
        return signals_batch

class MacroEconomicStrategy(BaseStrategy):
    """
    Simplified: Uses long-term moving averages to proxy economic cycles/regimes.
    Ideally, this strategy would use actual macroeconomic indicators (GDP, inflation, unemployment etc.).
    """
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="MacroEconomicStrategy",
            description="Simplified: Uses long-term moving averages to proxy economic cycles/regimes.",
            default_params={'instrument_key': None, 'ma_period': 200, 'roc_period': 60,
                            'feature_indices': {'Close': 3}}, # Example index
            applicable_assets=[]
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.ma_period = self.params.get('ma_period', 200)
        self.roc_period = self.params.get('roc_period', 60)
        self.feature_indices = self.params.get('feature_indices', {'Close': 3})

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None, external_data: Optional[pd.DataFrame] = None) -> torch.Tensor:
        # Note: original forward had `external_data` which is not in BaseStrategy.forward signature.
        # This will be ignored for now. If needed, BaseStrategy signature must change.
        batch_size, seq_len, _ = asset_features.shape
        signals_batch = torch.zeros(batch_size, seq_len, 1, device=asset_features.device)
        close_idx = self.feature_indices['Close']

        for i in range(batch_size):
            df = pd.DataFrame({'Close': asset_features[i, :, close_idx].cpu().numpy()})

            if df.empty or 'Close' not in df.columns:
                continue

            df['long_ma'] = ta.trend.sma_indicator(df['Close'], window=self.ma_period, fillna=True)
            df['roc'] = ta.momentum.roc(df['Close'], window=self.roc_period, fillna=True)

            # Original generate_signals logic
            current_signals_np = np.zeros(seq_len)
            bullish_condition = (df['Close'] > df['long_ma']) & (df['roc'] > 0)
            bearish_condition = (df['Close'] < df['long_ma']) & (df['roc'] < 0)

            current_signals_np[bullish_condition.fillna(False)] = 1.0
            current_signals_np[bearish_condition.fillna(False)] = -1.0
            signals_batch[i, :, 0] = torch.tensor(current_signals_np, device=asset_features.device, dtype=asset_features.dtype)
            
        return signals_batch

class EventDrivenStrategy(BaseStrategy):
    """
    Simplified: Uses unusual price changes (gaps) and volume spikes to proxy for market-moving events.
    Ideally, this strategy would use news feeds, sentiment analysis on news, and event calendars.
    """
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="EventDrivenStrategy",
            description="Simplified: Uses unusual price changes (gaps) and volume spikes to proxy for market-moving events.",
            default_params={
                'instrument_key': None,
                'gap_threshold_atr_multiplier': 1.5,
                'volume_spike_multiplier': 3.0,
                'atr_period': 14,
                'volume_period': 20,
                'feature_indices': {'Open':0, 'High': 1, 'Low': 2, 'Close': 3, 'Volume': 4} # Example indices
            },
            applicable_assets=[]
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.gap_threshold_atr_multiplier = self.params.get('gap_threshold_atr_multiplier', 1.5)
        self.volume_spike_multiplier = self.params.get('volume_spike_multiplier', 3.0)
        self.atr_period = self.params.get('atr_period', 14)
        self.volume_period = self.params.get('volume_period', 20)
        self.feature_indices = self.params.get('feature_indices', {'Open':0, 'High': 1, 'Low': 2, 'Close': 3, 'Volume': 4})

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, _ = asset_features.shape
        signals_batch = torch.zeros(batch_size, seq_len, 1, device=asset_features.device)

        open_idx = self.feature_indices['Open']
        high_idx = self.feature_indices['High']
        low_idx = self.feature_indices['Low']
        close_idx = self.feature_indices['Close']
        vol_idx = self.feature_indices['Volume']

        for i in range(batch_size):
            df = pd.DataFrame({
                'Open': asset_features[i, :, open_idx].cpu().numpy(),
                'High': asset_features[i, :, high_idx].cpu().numpy(),
                'Low': asset_features[i, :, low_idx].cpu().numpy(),
                'Close': asset_features[i, :, close_idx].cpu().numpy(),
                'Volume': asset_features[i, :, vol_idx].cpu().numpy()
            })

            if df.empty or not all(c in df.columns for c in ['Open', 'High', 'Low', 'Close', 'Volume']):
                continue

            df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=self.atr_period, fillna=True)
            df['avg_volume'] = df['Volume'].rolling(window=self.volume_period, min_periods=max(1,self.volume_period//2)).mean()
            
            df['gap_up'] = df['Open'] > (df['Close'].shift(1) + df['atr'].shift(1) * self.gap_threshold_atr_multiplier)
            df['gap_down'] = df['Open'] < (df['Close'].shift(1) - df['atr'].shift(1) * self.gap_threshold_atr_multiplier)
            df['volume_spike'] = df['Volume'] > (df['avg_volume'] * self.volume_spike_multiplier)
            df.fillna({'atr':0, 'avg_volume':0, 'gap_up':False, 'gap_down':False, 'volume_spike':False}, inplace=True)

            # Original generate_signals logic
            current_signals_np = np.zeros(seq_len)
            bullish_event = df['gap_up'] & df['volume_spike'] & (df['Close'] > df['Open'])
            bearish_event = df['gap_down'] & df['volume_spike'] & (df['Close'] < df['Open'])

            current_signals_np[bullish_event.fillna(False)] = 1.0
            current_signals_np[bearish_event.fillna(False)] = -1.0
            signals_batch[i, :, 0] = torch.tensor(current_signals_np, device=asset_features.device, dtype=asset_features.dtype)
            
        return signals_batch

class SentimentStrategy(BaseStrategy):
    """
    Simplified: Uses RSI and Stochastics to gauge overbought/oversold conditions as a proxy for extreme sentiment.
    Ideally, this strategy would use actual sentiment scores from news, social media, surveys.
    """
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="SentimentStrategy",
            description="Simplified: Uses RSI and Stochastics to gauge overbought/oversold conditions as a proxy for extreme sentiment.",
            default_params={
                'instrument_key': None,
                'rsi_period': 14,
                'stoch_k_period': 14,
                'stoch_d_period': 3,
                'rsi_ob_threshold': 70.0,
                'rsi_os_threshold': 30.0,
                'stoch_ob_threshold': 80.0,
                'stoch_os_threshold': 20.0,
                'feature_indices': {'High': 1, 'Low': 2, 'Close': 3} # Example indices
            },
            applicable_assets=[]
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.rsi_period = self.params.get('rsi_period', 14)
        self.stoch_k_period = self.params.get('stoch_k_period', 14)
        self.stoch_d_period = self.params.get('stoch_d_period', 3)
        self.rsi_ob_threshold = self.params.get('rsi_ob_threshold', 70.0)
        self.rsi_os_threshold = self.params.get('rsi_os_threshold', 30.0)
        self.stoch_ob_threshold = self.params.get('stoch_ob_threshold', 80.0)
        self.stoch_os_threshold = self.params.get('stoch_os_threshold', 20.0)
        self.feature_indices = self.params.get('feature_indices', {'High': 1, 'Low': 2, 'Close': 3})
            
    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, _ = asset_features.shape
        signals_batch = torch.zeros(batch_size, seq_len, 1, device=asset_features.device)

        high_idx = self.feature_indices['High']
        low_idx = self.feature_indices['Low']
        close_idx = self.feature_indices['Close']

        for i in range(batch_size):
            df = pd.DataFrame({
                'High': asset_features[i, :, high_idx].cpu().numpy(),
                'Low': asset_features[i, :, low_idx].cpu().numpy(),
                'Close': asset_features[i, :, close_idx].cpu().numpy()
            })

            if df.empty or not all(c in df.columns for c in ['High', 'Low', 'Close']):
                continue

            df['rsi'] = ta.momentum.rsi(df['Close'], window=self.rsi_period, fillna=True)
            stoch = ta.momentum.StochasticOscillator(
                high=df['High'], low=df['Low'], close=df['Close'], 
                window=self.stoch_k_period, smooth_window=self.stoch_d_period, fillna=True
            )
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()

            # Original generate_signals logic
            current_signals_np = np.zeros(seq_len)
            overbought_condition = (df['rsi'] > self.rsi_ob_threshold) & (df['stoch_k'] > self.stoch_ob_threshold) & (df['stoch_k'] < df['stoch_k'].shift(1))
            oversold_condition = (df['rsi'] < self.rsi_os_threshold) & (df['stoch_k'] < self.stoch_os_threshold) & (df['stoch_k'] > df['stoch_k'].shift(1))

            current_signals_np[overbought_condition.fillna(False)] = -1.0
            current_signals_np[oversold_condition.fillna(False)] = 1.0
            signals_batch[i, :, 0] = torch.tensor(current_signals_np, device=asset_features.device, dtype=asset_features.dtype)
            
        return signals_batch

class QuantitativeStrategy(BaseStrategy):
    """
    A simple multi-factor model using common technical indicators.
    """
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="QuantitativeStrategy",
            description="A generic quantitative strategy placeholder. Relies on 'expression' and 'feature_dict' params.",
            default_params={
                'expression': None, 
                'feature_dict': {}, 
                'asset_list': [],
                'feature_indices': {'Close': 3, 'ma_short': -1, 'ma_long': -1, 'rsi': -1, 'bb_lower': -1, 'bb_upper': -1} # Placeholder indices
            }
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.expression = self.params.get('expression')
        self.feature_dict = self.params.get('feature_dict', {})
        self.feature_indices = self.params.get('feature_indices', {})
        if not self.expression:
            self.logger.warning(f"{self.config.name}: 'expression' parameter is not set. Strategy may not produce signals.")

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, num_features_tensor = asset_features.shape
        signals_batch = torch.zeros(batch_size, seq_len, 1, device=asset_features.device)

        # This strategy is complex due to dynamic feature calculation via 'eval' and 'feature_dict'.
        # A full tensor-based refactor would require a tensor-native way to compute these dynamic features.
        # For now, we'll process per batch item and use pandas, similar to other strategies.
        # The `feature_dict` logic with `eval` is particularly tricky and unsafe for production.
        # The `generate_signals` part uses hardcoded features like 'ma_short', 'rsi', etc.
        # We need to ensure these are present or calculated.

        # Assuming standard features are available at known indices if not dynamically calculated.
        close_idx = self.feature_indices.get('Close', -1) # Default to -1 if not found
        # The following features are used in the original generate_signals, but not calculated in original forward.
        # This implies they are expected to be pre-calculated or part of a richer asset_features tensor.
        # For this refactor, we'll assume they need to be calculated if not present.
        # This part is highly dependent on how `asset_features` is constructed upstream.

        for i in range(batch_size):
            # Create a base DataFrame from essential features
            df_data = {}
            if close_idx != -1 and close_idx < num_features_tensor:
                 df_data['Close'] = asset_features[i, :, close_idx].cpu().numpy()
            else:
                self.logger.warning(f"{self.config.name}: 'Close' feature index not valid. Cannot proceed for batch item {i}.")
                continue # Skip this batch item
            
            df = pd.DataFrame(df_data)
            if df.empty:
                continue

            # Attempt to calculate features defined in feature_dict (original forward logic)
            # This is still problematic and not very tensor-friendly.
            # For a real scenario, features should be tensor operations or pre-calculated.
            # For now, retain the eval-based approach for compatibility, but it's a major refactoring point.
            temp_market_data_for_eval = pd.DataFrame(asset_features[i,:,:].cpu().numpy()) # Provide all raw features to eval
            # We need to map column names for eval if feature_dict uses names like 'close'
            # This is a simplification; a robust solution needs careful name mapping.
            # Assuming feature_dict refers to columns by standard names like 'Close', 'High' etc.
            # and that asset_features has these at appropriate indices.
            # This part is very fragile.

            # For generate_signals, we need: ma_short, ma_long, rsi, bb_lower, bb_upper
            # Let's calculate them using `ta` for simplicity, assuming `Close` is available.
            df['ma_short'] = ta.trend.sma_indicator(df['Close'], window=10, fillna=True) # Example window
            df['ma_long'] = ta.trend.sma_indicator(df['Close'], window=20, fillna=True)  # Example window
            df['rsi'] = ta.momentum.rsi(df['Close'], window=14, fillna=True) # Example window
            bbands = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2, fillna=True) # Example params
            df['bb_lower'] = bbands.bollinger_lband()
            df['bb_upper'] = bbands.bollinger_hband()
            df.fillna(0, inplace=True) # Fill NaNs that might result from indicator calculations

            # Original generate_signals logic
            current_signals_np = np.zeros(seq_len)
            trend_factor = (df['ma_short'] > df['ma_long']).astype(int) * 2 - 1
            momentum_factor_bull = (df['rsi'] > 55).astype(int)
            momentum_factor_bear = (df['rsi'] < 45).astype(int) * -1
            momentum_factor = momentum_factor_bull + momentum_factor_bear
            
            volatility_factor_buy = (df['Close'] < df['bb_lower']).astype(int)
            volatility_factor_sell = (df['Close'] > df['bb_upper']).astype(int) * -1
            volatility_factor = volatility_factor_buy + volatility_factor_sell

            combined_score = trend_factor + momentum_factor + volatility_factor

            current_signals_np[combined_score >= 2] = 1.0
            current_signals_np[combined_score <= -2] = -1.0
            signals_batch[i, :, 0] = torch.tensor(current_signals_np, device=asset_features.device, dtype=asset_features.dtype)
            
        return signals_batch

class MarketMakingStrategy(BaseStrategy):
    """
    Simplified: Simulates placing bid/ask signals around a central moving average.
    Signals are +1 for simulated bid, -1 for simulated ask. No actual order placement logic.
    Ideally, this strategy uses L1/L2 order book data, inventory management, and sophisticated fair value estimation.
    """
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="MarketMakingStrategy",
            description="Simplified market making: places bid/ask around a reference price if spread is wide enough.",
            default_params={
                'instrument_key': None,
                'reference_price_ma_period': 10, 
                'min_spread_threshold_abs': 0.0005, 
                'quote_offset_abs': 0.0002,
                # 'ma_period' and 'spread_percentage' were used in old forward/generate_signals, ensure they are covered or updated.
                # Assuming 'reference_price_ma_period' replaces 'ma_period'.
                # 'spread_percentage' is not directly used in the new logic, using fixed offsets.
                'feature_indices': {'Close': 3, 'Low': 2, 'High':1} # Example indices
            },
            applicable_assets=[]
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.ma_period = self.params.get('reference_price_ma_period', 10) # Renamed from 'ma_period' for clarity
        self.quote_offset_abs = self.params.get('quote_offset_abs', 0.0002)
        # self.spread_percentage = self.params.get('spread_percentage', 0.001) # Not used in current simplified signal logic
        self.feature_indices = self.params.get('feature_indices', {'Close': 3, 'Low': 2, 'High':1})

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, _ = asset_features.shape
        # This strategy is more about providing target bid/ask, not direct buy/sell signals.
        # The output tensor will be (batch_size, seq_len, 3) for [signal, target_bid, target_ask]
        # However, BaseStrategy.forward expects (batch_size, seq_len, 1) for signals.
        # For now, we will return a signal of 0, and log/ignore target_bid/ask for compatibility.
        # A proper solution would involve a different output structure or strategy type.
        signals_batch = torch.zeros(batch_size, seq_len, 1, device=asset_features.device)
        
        close_idx = self.feature_indices['Close']
        # low_idx = self.feature_indices['Low'] # Not used in current simplified signal logic
        # high_idx = self.feature_indices['High'] # Not used in current simplified signal logic

        for i in range(batch_size):
            df = pd.DataFrame({'Close': asset_features[i, :, close_idx].cpu().numpy()})
            # df_low = pd.Series(asset_features[i, :, low_idx].cpu().numpy())
            # df_high = pd.Series(asset_features[i, :, high_idx].cpu().numpy())

            if df.empty or 'Close' not in df.columns:
                continue

            df['central_ma'] = ta.trend.sma_indicator(df['Close'], window=self.ma_period, fillna=True)
            # The original forward used spread_percentage. The default_params now has quote_offset_abs.
            # Let's use quote_offset_abs for sim_bid/ask.
            df['sim_bid_price'] = df['central_ma'] - self.quote_offset_abs
            df['sim_ask_price'] = df['central_ma'] + self.quote_offset_abs
            df.fillna(0, inplace=True)

            # Original generate_signals logic was mostly about setting target_bid/target_ask.
            # It had commented out logic for actual signals if Low/High hit these targets.
            # For now, signal is 0 as per original dominant logic.
            current_signals_np = np.zeros(seq_len)
            # Example: (Uncomment and adapt if actual signals are desired)
            # if 'Low' in df.columns and 'High' in df.columns: # Requires Low/High features
            #    current_signals_np[df_low.to_numpy() <= df['sim_bid_price'].to_numpy()] = 1.0
            #    current_signals_np[df_high.to_numpy() >= df['sim_ask_price'].to_numpy()] = -1.0
            signals_batch[i, :, 0] = torch.tensor(current_signals_np, device=asset_features.device, dtype=asset_features.dtype)
            
        return signals_batch

class HighFrequencyStrategy(BaseStrategy):
    """
    Simplified: Uses very short-term RSI and MA to capture quick fluctuations.
    Assumes input data is high-frequency (e.g., 1-minute bars).
    Ideally, this strategy uses tick data, order book dynamics, and latency-sensitive execution.
    """
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="HighFrequencyStrategy",
            description="Placeholder for HFT. Simplified: reacts to very short-term price changes (tick-level momentum).",
            default_params={
                'instrument_key': None,
                'tick_momentum_threshold': 0.0001, # Not directly used in current RSI/MA logic
                'rsi_period': 5, # Short period for HFT proxy
                'ma_period': 10, # Short period for HFT proxy
                'rsi_os': 30,
                'rsi_ob': 70,
                'feature_indices': {'Close': 3} # Example index
            },
            applicable_assets=[]
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.rsi_period = self.params.get('rsi_period', 5)
        self.ma_period = self.params.get('ma_period', 10)
        self.rsi_os = self.params.get('rsi_os', 30)
        self.rsi_ob = self.params.get('rsi_ob', 70)
        self.feature_indices = self.params.get('feature_indices', {'Close': 3})

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, _ = asset_features.shape
        signals_batch = torch.zeros(batch_size, seq_len, 1, device=asset_features.device)
        close_idx = self.feature_indices['Close']

        for i in range(batch_size):
            df = pd.DataFrame({'Close': asset_features[i, :, close_idx].cpu().numpy()})

            if df.empty or 'Close' not in df.columns:
                continue

            df['rsi'] = ta.momentum.rsi(df['Close'], window=self.rsi_period, fillna=True)
            df['ma'] = ta.trend.sma_indicator(df['Close'], window=self.ma_period, fillna=True)
            df.fillna(0, inplace=True)

            # Original generate_signals logic
            current_signals_np = np.zeros(seq_len)
            buy_condition = (df['rsi'] < self.rsi_os) & (df['Close'] > df['ma'])
            sell_condition = (df['rsi'] > self.rsi_ob) & (df['Close'] < df['ma'])

            current_signals_np[buy_condition.fillna(False)] = 1.0
            current_signals_np[sell_condition.fillna(False)] = -1.0
            signals_batch[i, :, 0] = torch.tensor(current_signals_np, device=asset_features.device, dtype=asset_features.dtype)
            
        return signals_batch

class AlgorithmicStrategy(BaseStrategy):
    """
    A generic algorithmic strategy, here implemented as a VWAP cross strategy.
    """
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="AlgorithmicStrategy",
            description="A generic algorithmic strategy that executes trades based on predefined rules (placeholder).",
            default_params={
                'rule_buy_condition': None, 
                'rule_sell_condition': None, 
                'asset_list': [],
                'ma_window_algo': 20, # Added for the example logic
                'feature_indices': {'Close': 3} # Example index
            }
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.rule_buy_condition = self.params.get('rule_buy_condition') # Not directly used in refactored example
        self.rule_sell_condition = self.params.get('rule_sell_condition') # Not directly used
        self.ma_window_algo = self.params.get('ma_window_algo', 20)
        self.feature_indices = self.params.get('feature_indices', {'Close': 3})
        if not self.rule_buy_condition or not self.rule_sell_condition:
            self.logger.warning(f"{self.config.name}: Buy/Sell rule conditions are not fully set. Strategy may not produce signals if relying on them.")

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, _ = asset_features.shape
        signals_batch = torch.zeros(batch_size, seq_len, 1, device=asset_features.device)
        close_idx = self.feature_indices['Close']

        # The original forward was a pass-through. The generate_signals had example MA logic.
        # We will implement the example MA logic here.
        # Proper rule evaluation based on 'rule_buy_condition' strings is complex and out of scope for this direct refactor.

        for i in range(batch_size):
            df = pd.DataFrame({'Close': asset_features[i, :, close_idx].cpu().numpy()})

            if df.empty or 'Close' not in df.columns:
                continue
            
            # Example logic from original generate_signals
            df['ma_algo'] = df['Close'].rolling(window=self.ma_window_algo).mean().fillna(0)
            df.fillna(0, inplace=True)

            current_signals_np = np.zeros(seq_len)
            buy_condition = df['Close'] > df['ma_algo']
            sell_condition = df['Close'] < df['ma_algo']

            # Avoid setting signal if MA is zero (e.g. at the start of series due to not enough data for rolling mean)
            valid_ma = df['ma_algo'] != 0
            current_signals_np[(buy_condition & valid_ma).fillna(False)] = 1.0
            current_signals_np[(sell_condition & valid_ma).fillna(False)] = -1.0
            signals_batch[i, :, 0] = torch.tensor(current_signals_np, device=asset_features.device, dtype=asset_features.dtype)
            
        return signals_batch
