# src/agent/strategies/other_strategies.py
from .base_strategy import BaseStrategy, StrategyConfig
import pandas as pd
import numpy as np
import ta
from typing import Dict, Optional

class OptionFlowStrategy(BaseStrategy):
    """ 
    Simplified: Uses high volume and volatility as a proxy for significant option activity.
    Ideally, this strategy would use actual option chain data (volume, open interest, greeks).
    """
    def __init__(self, config: StrategyConfig, params: dict = None): # Corrected signature
        super().__init__(config, params) # Corrected call
        self.instrument_key = self.config.applicable_assets[0] if self.config.applicable_assets else None
        if self.instrument_key is None:
            raise ValueError("OptionFlowStrategy requires at least one instrument key in config.applicable_assets.")
        
        # Use self.effective_params which is set by BaseStrategy
        self.volume_period = self.effective_params.get('volume_period', 20)
        self.volatility_period = self.effective_params.get('volatility_period', 14) # ATR period
        self.volume_z_threshold = self.effective_params.get('volume_z_threshold', 2.0)
        self.atr_threshold_multiplier = self.effective_params.get('atr_threshold_multiplier', 1.5)

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        market_data = market_data_dict.get(self.instrument_key)
        if market_data is None or market_data.empty or not all(c in market_data.columns for c in ['High', 'Low', 'Close', 'Volume']):
            return {self.instrument_key: pd.DataFrame(index=market_data.index if market_data is not None else None)}

        processed_data = market_data.copy()
        processed_data['avg_volume'] = processed_data['Volume'].rolling(window=self.volume_period, min_periods=max(1,self.volume_period//2)).mean()
        processed_data['std_volume'] = processed_data['Volume'].rolling(window=self.volume_period, min_periods=max(1,self.volume_period//2)).std()
        processed_data['volume_z_score'] = (processed_data['Volume'] - processed_data['avg_volume']) / processed_data['std_volume'].replace(0, np.nan)
        
        processed_data['atr'] = ta.volatility.average_true_range(processed_data['High'], processed_data['Low'], processed_data['Close'], window=self.volatility_period, fillna=True)
        processed_data['price_range'] = processed_data['High'] - processed_data['Low']
        processed_data.fillna({'volume_z_score': 0, 'atr': 0, 'price_range': 0}, inplace=True)
        return {self.instrument_key: processed_data}

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> pd.DataFrame:
        processed_data = processed_data_dict.get(self.instrument_key)
        if processed_data is None or processed_data.empty or not all(c in processed_data.columns for c in ['volume_z_score', 'atr', 'price_range', 'Close']):
            idx = processed_data.index if processed_data is not None else None
            signals = pd.DataFrame(index=idx)
            signals['signal'] = 0
            signals['reason'] = "Missing data for signal generation"
            return signals

        signals = pd.DataFrame(index=processed_data.index)
        
        high_volume_activity = processed_data['volume_z_score'] > self.volume_z_threshold
        high_volatility_activity = processed_data['price_range'] > (processed_data['atr'] * self.atr_threshold_multiplier)
        
        # Bullish if high volume/volatility and price is increasing
        bullish_condition = high_volume_activity & high_volatility_activity & (processed_data['Close'] > processed_data['Close'].shift(1))
        # Bearish if high volume/volatility and price is decreasing
        bearish_condition = high_volume_activity & high_volatility_activity & (processed_data['Close'] < processed_data['Close'].shift(1))

        signals['signal'] = 0
        signals.loc[bullish_condition, 'signal'] = 1
        signals.loc[bearish_condition, 'signal'] = -1
        
        signals['reason'] = "Neutral"
        signals.loc[bullish_condition, 'reason'] = f"High Vol/Volum Z>{self.volume_z_threshold}, ATR Mult>{self.atr_threshold_multiplier}, Price Up"
        signals.loc[bearish_condition, 'reason'] = f"High Vol/Volum Z>{self.volume_z_threshold}, ATR Mult>{self.atr_threshold_multiplier}, Price Down"
        return signals

class MicrostructureStrategy(BaseStrategy):
    """
    Simplified: Uses price spread (High-Low) and volume spikes as proxies for microstructure imbalances.
    Ideally, this strategy would use L1/L2 order book data (bid/ask prices, sizes, order flow).
    """
    def __init__(self, config: StrategyConfig, params: dict = None): # Corrected signature
        super().__init__(config, params) # Corrected call
        self.instrument_key = self.config.applicable_assets[0] if self.config.applicable_assets else None
        if self.instrument_key is None:
            raise ValueError("MicrostructureStrategy requires at least one instrument key in config.applicable_assets.")
        
        self.spread_window = self.effective_params.get('spread_window', 10)
        self.volume_window = self.effective_params.get('volume_window', 20)
        self.spread_quantile_threshold = self.effective_params.get('spread_quantile_threshold', 0.90) # Top 10% spread
        self.volume_spike_multiplier = self.effective_params.get('volume_spike_multiplier', 3.0) # Volume > 3x average

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        market_data = market_data_dict.get(self.instrument_key)
        if market_data is None or market_data.empty or not all(c in market_data.columns for c in ['High', 'Low', 'Close', 'Volume']):
            return {self.instrument_key: pd.DataFrame(index=market_data.index if market_data is not None else None)}

        processed_data = market_data.copy()
        processed_data['spread'] = processed_data['High'] - processed_data['Low']
        processed_data['rolling_median_spread'] = processed_data['spread'].rolling(window=self.spread_window, min_periods=max(1,self.spread_window//2)).median()
        processed_data['avg_volume'] = processed_data['Volume'].rolling(window=self.volume_window, min_periods=max(1,self.volume_window//2)).mean()
        processed_data.fillna({'rolling_median_spread': 0, 'avg_volume': 0, 'spread':0}, inplace=True)
        return {self.instrument_key: processed_data}

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> pd.DataFrame:
        processed_data = processed_data_dict.get(self.instrument_key)
        if processed_data is None or processed_data.empty or not all(c in processed_data.columns for c in ['spread', 'rolling_median_spread', 'Volume', 'avg_volume', 'Close']):
            idx = processed_data.index if processed_data is not None else None
            signals = pd.DataFrame(index=idx)
            signals['signal'] = 0
            signals['reason'] = "Missing data for signal generation"
            return signals

        signals = pd.DataFrame(index=processed_data.index)
        
        wide_spread = processed_data['spread'] > processed_data['spread'].rolling(window=self.spread_window*2, min_periods=max(1,self.spread_window)).quantile(self.spread_quantile_threshold) # Dynamic threshold
        volume_spike = processed_data['Volume'] > (processed_data['avg_volume'] * self.volume_spike_multiplier)
        
        # Example: Fade wide spreads on volume spikes if price moved sharply (mean reversion idea)
        price_up_sharply = processed_data['Close'] > processed_data['Close'].shift(1) + processed_data['rolling_median_spread']
        price_down_sharply = processed_data['Close'] < processed_data['Close'].shift(1) - processed_data['rolling_median_spread']

        signals['signal'] = 0
        signals.loc[wide_spread & volume_spike & price_up_sharply, 'signal'] = -1 # Sell, expecting reversion
        signals.loc[wide_spread & volume_spike & price_down_sharply, 'signal'] = 1  # Buy, expecting reversion

        signals['reason'] = "Neutral"
        signals.loc[wide_spread & volume_spike & price_up_sharply, 'reason'] = f"Wide Spread & Vol Spike ({self.volume_spike_multiplier}x) & Price Up Sharply, Fade"
        signals.loc[wide_spread & volume_spike & price_down_sharply, 'reason'] = f"Wide Spread & Vol Spike ({self.volume_spike_multiplier}x) & Price Down Sharply, Fade"
        return signals

class CarryTradeStrategy(BaseStrategy):
    """
    Simplified: Uses momentum as a proxy for chasing yield/currency strength.
    Ideally, this strategy would use actual interest rate differentials between currency pairs.
    For a single instrument, this translates to trend following if we assume higher yield attracts capital.
    """
    def __init__(self, config: StrategyConfig, params: dict = None): # Corrected signature
        super().__init__(config, params) # Corrected call
        # self.instrument_key = self.config.applicable_assets[0] if self.config.applicable_assets else None # Removed, forward will handle multiple
        if not self.config.applicable_assets:
            raise ValueError("CarryTradeStrategy requires at least one instrument key in config.applicable_assets.")
        
        self.short_ma_period = self.effective_params.get('short_ma_period', 10)
        self.long_ma_period = self.effective_params.get('long_ma_period', 30)

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        output_data_dict = {}
        for asset_key in self.config.applicable_assets:
            market_data = market_data_dict.get(asset_key)
            if market_data is None or market_data.empty or 'Close' not in market_data.columns:
                output_data_dict[asset_key] = pd.DataFrame(index=market_data.index if market_data is not None else None)
                continue

            processed_data = market_data.copy()
            processed_data['short_ma'] = ta.trend.sma_indicator(processed_data['Close'], window=self.short_ma_period, fillna=True)
            processed_data['long_ma'] = ta.trend.sma_indicator(processed_data['Close'], window=self.long_ma_period, fillna=True)
            # Note: The test expects 'interest_rate_differential', which is not calculated here.
            output_data_dict[asset_key] = processed_data
        return output_data_dict

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> pd.DataFrame:
        if not self.config.applicable_assets:
            # Should have been caught in __init__, but as a safeguard
            return pd.DataFrame(columns=['signal', 'reason'])

        primary_asset_key = self.config.applicable_assets[0]
        processed_data = processed_data_dict.get(primary_asset_key)

        if processed_data is None or processed_data.empty or not all(c in processed_data.columns for c in ['short_ma', 'long_ma']):
            idx = processed_data.index if processed_data is not None else None
            signals = pd.DataFrame(index=idx)
            signals['signal'] = 0
            signals['reason'] = "Missing data for signal generation"
            return signals

        signals = pd.DataFrame(index=processed_data.index)
        signals['signal'] = 0
        
        buy_condition = (processed_data['short_ma'] > processed_data['long_ma']) & (processed_data['short_ma'].shift(1) <= processed_data['long_ma'].shift(1))
        sell_condition = (processed_data['short_ma'] < processed_data['long_ma']) & (processed_data['short_ma'].shift(1) >= processed_data['long_ma'].shift(1))

        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        signals['reason'] = "Neutral"
        signals.loc[buy_condition, 'reason'] = f"MA Crossover ({self.short_ma_period}/{self.long_ma_period}) - Bullish (Carry Proxy)"
        signals.loc[sell_condition, 'reason'] = f"MA Crossover ({self.short_ma_period}/{self.long_ma_period}) - Bearish (Carry Proxy)"
        return signals

class MacroEconomicStrategy(BaseStrategy):
    """
    Simplified: Uses long-term moving averages to proxy economic cycles/regimes.
    Ideally, this strategy would use actual macroeconomic indicators (GDP, inflation, unemployment etc.).
    """
    def __init__(self, config: StrategyConfig, params: dict = None): # Corrected signature
        super().__init__(config, params) # Corrected call
        self.instrument_key = self.config.applicable_assets[0] if self.config.applicable_assets else None
        if self.instrument_key is None:
            raise ValueError("MacroEconomicStrategy requires at least one instrument key in config.applicable_assets.")
        
        self.ma_period = self.effective_params.get('ma_period', 200) # Long-term MA for cycle proxy
        self.roc_period = self.effective_params.get('roc_period', 60) # Rate of change over ~3 months

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None, external_data: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        market_data = market_data_dict.get(self.instrument_key)
        if market_data is None or market_data.empty or 'Close' not in market_data.columns:
            return {self.instrument_key: pd.DataFrame(index=market_data.index if market_data is not None else None)}

        processed_data = market_data.copy()
        processed_data['long_ma'] = ta.trend.sma_indicator(processed_data['Close'], window=self.ma_period, fillna=True)
        processed_data['roc'] = ta.momentum.roc(processed_data['Close'], window=self.roc_period, fillna=True)
        return {self.instrument_key: processed_data}

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> pd.DataFrame:
        processed_data = processed_data_dict.get(self.instrument_key)
        if processed_data is None or processed_data.empty or not all(c in processed_data.columns for c in ['Close', 'long_ma', 'roc']):
            idx = processed_data.index if processed_data is not None else None
            signals = pd.DataFrame(index=idx)
            signals['signal'] = 0
            signals['reason'] = "Missing data for signal generation"
            return signals

        signals = pd.DataFrame(index=processed_data.index)
        signals['signal'] = 0
        
        # Bullish if above long MA and momentum is positive
        bullish_condition = (processed_data['Close'] > processed_data['long_ma']) & (processed_data['roc'] > 0)
        # Bearish if below long MA and momentum is negative
        bearish_condition = (processed_data['Close'] < processed_data['long_ma']) & (processed_data['roc'] < 0)

        signals.loc[bullish_condition, 'signal'] = 1
        signals.loc[bearish_condition, 'signal'] = -1
        
        signals['reason'] = "Neutral (Macro Proxy)"
        signals.loc[bullish_condition, 'reason'] = f"Price > {self.ma_period}MA & ROC({self.roc_period}) > 0 (Bullish Macro Proxy)"
        signals.loc[bearish_condition, 'reason'] = f"Price < {self.ma_period}MA & ROC({self.roc_period}) < 0 (Bearish Macro Proxy)"
        return signals

class EventDrivenStrategy(BaseStrategy):
    """
    Simplified: Uses unusual price changes (gaps) and volume spikes to proxy for market-moving events.
    Ideally, this strategy would use news feeds, sentiment analysis on news, and event calendars.
    """
    def __init__(self, config: StrategyConfig, params: dict = None): # Corrected signature
        super().__init__(config, params) # Corrected call
        self.instrument_key = self.config.applicable_assets[0] if self.config.applicable_assets else None
        if self.instrument_key is None:
            raise ValueError("EventDrivenStrategy requires at least one instrument key in config.applicable_assets.")
        
        self.gap_threshold_atr_multiplier = self.effective_params.get('gap_threshold_atr_multiplier', 1.5)
        self.volume_spike_multiplier = self.effective_params.get('volume_spike_multiplier', 3.0)
        self.atr_period = self.effective_params.get('atr_period', 14)
        self.volume_period = self.effective_params.get('volume_period', 20)

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        market_data = market_data_dict.get(self.instrument_key)
        if market_data is None or market_data.empty or not all(c in market_data.columns for c in ['Open', 'High', 'Low', 'Close', 'Volume']):
            return {self.instrument_key: pd.DataFrame(index=market_data.index if market_data is not None else None)}

        processed_data = market_data.copy()
        processed_data['atr'] = ta.volatility.average_true_range(processed_data['High'], processed_data['Low'], processed_data['Close'], window=self.atr_period, fillna=True)
        processed_data['avg_volume'] = processed_data['Volume'].rolling(window=self.volume_period, min_periods=max(1,self.volume_period//2)).mean()
        
        processed_data['gap_up'] = processed_data['Open'] > (processed_data['Close'].shift(1) + processed_data['atr'].shift(1) * self.gap_threshold_atr_multiplier)
        processed_data['gap_down'] = processed_data['Open'] < (processed_data['Close'].shift(1) - processed_data['atr'].shift(1) * self.gap_threshold_atr_multiplier)
        processed_data['volume_spike'] = processed_data['Volume'] > (processed_data['avg_volume'] * self.volume_spike_multiplier)
        processed_data.fillna({'atr':0, 'avg_volume':0, 'gap_up':False, 'gap_down':False, 'volume_spike':False}, inplace=True)
        return {self.instrument_key: processed_data}

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> pd.DataFrame:
        processed_data = processed_data_dict.get(self.instrument_key)
        if processed_data is None or processed_data.empty or not all(c in processed_data.columns for c in ['gap_up', 'gap_down', 'volume_spike', 'Close', 'Open']):
            idx = processed_data.index if processed_data is not None else None
            signals = pd.DataFrame(index=idx)
            signals['signal'] = 0
            signals['reason'] = "Missing data for signal generation"
            return signals

        signals = pd.DataFrame(index=processed_data.index)
        signals['signal'] = 0
        
        # Trade in direction of gap if confirmed by volume and intra-bar momentum
        bullish_event = processed_data['gap_up'] & processed_data['volume_spike'] & (processed_data['Close'] > processed_data['Open'])
        bearish_event = processed_data['gap_down'] & processed_data['volume_spike'] & (processed_data['Close'] < processed_data['Open'])

        signals.loc[bullish_event, 'signal'] = 1
        signals.loc[bearish_event, 'signal'] = -1
        
        signals['reason'] = "Neutral (Event Proxy)"
        signals.loc[bullish_event, 'reason'] = f"Gap Up ({self.gap_threshold_atr_multiplier}xATR) & Vol Spike ({self.volume_spike_multiplier}x) & Bullish Bar (Event Proxy)"
        signals.loc[bearish_event, 'reason'] = f"Gap Down ({self.gap_threshold_atr_multiplier}xATR) & Vol Spike ({self.volume_spike_multiplier}x) & Bearish Bar (Event Proxy)"
        return signals

class SentimentStrategy(BaseStrategy):
    """
    Simplified: Uses RSI and Stochastics to gauge overbought/oversold conditions as a proxy for extreme sentiment.
    Ideally, this strategy would use actual sentiment scores from news, social media, surveys.
    """
    def __init__(self, config: StrategyConfig, params: dict = None): # Corrected signature
        super().__init__(config, params) # Corrected call
        self.instrument_key = self.config.applicable_assets[0] if self.config.applicable_assets else None
        if self.instrument_key is None:
            raise ValueError("SentimentStrategy requires at least one instrument key in config.applicable_assets.")
        
        self.rsi_period = self.effective_params.get('rsi_period', 14)
        self.stoch_k_period = self.effective_params.get('stoch_k_period', 14)
        self.stoch_d_period = self.effective_params.get('stoch_d_period', 3)
        self.rsi_ob_threshold = self.effective_params.get('rsi_ob_threshold', 70)
        self.rsi_os_threshold = self.effective_params.get('rsi_os_threshold', 30)
        self.stoch_ob_threshold = self.effective_params.get('stoch_ob_threshold', 80)
        self.stoch_os_threshold = self.effective_params.get('stoch_os_threshold', 20)

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None, sentiment_data: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        market_data = market_data_dict.get(self.instrument_key)
        if market_data is None or market_data.empty or not all(c in market_data.columns for c in ['High', 'Low', 'Close']):
            return {self.instrument_key: pd.DataFrame(index=market_data.index if market_data is not None else None)}

        processed_data = market_data.copy()
        processed_data['rsi'] = ta.momentum.rsi(processed_data['Close'], window=self.rsi_period, fillna=True)
        stoch = ta.momentum.StochasticOscillator(
            high=processed_data['High'], low=processed_data['Low'], close=processed_data['Close'], 
            window=self.stoch_k_period, smooth_window=self.stoch_d_period, fillna=True
        )
        processed_data['stoch_k'] = stoch.stoch()
        processed_data['stoch_d'] = stoch.stoch_signal()
        return {self.instrument_key: processed_data}

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> pd.DataFrame:
        processed_data = processed_data_dict.get(self.instrument_key)
        if processed_data is None or processed_data.empty or not all(c in processed_data.columns for c in ['rsi', 'stoch_k', 'stoch_d']):
            idx = processed_data.index if processed_data is not None else None
            signals = pd.DataFrame(index=idx)
            signals['signal'] = 0
            signals['reason'] = "Missing data for signal generation"
            return signals

        signals = pd.DataFrame(index=processed_data.index)
        signals['signal'] = 0
        
        # Extreme bullish sentiment (overbought) -> potential reversal (sell)
        overbought_condition = (processed_data['rsi'] > self.rsi_ob_threshold) & (processed_data['stoch_k'] > self.stoch_ob_threshold) & (processed_data['stoch_k'] < processed_data['stoch_k'].shift(1)) # Stoch K turning down
        # Extreme bearish sentiment (oversold) -> potential reversal (buy)
        oversold_condition = (processed_data['rsi'] < self.rsi_os_threshold) & (processed_data['stoch_k'] < self.stoch_os_threshold) & (processed_data['stoch_k'] > processed_data['stoch_k'].shift(1)) # Stoch K turning up

        signals.loc[overbought_condition, 'signal'] = -1
        signals.loc[oversold_condition, 'signal'] = 1
        
        signals['reason'] = "Neutral (Sentiment Proxy)"
        signals.loc[overbought_condition, 'reason'] = f"RSI>{self.rsi_ob_threshold} & StochK>{self.stoch_ob_threshold}, StochK down (Overbought Proxy)"
        signals.loc[oversold_condition, 'reason'] = f"RSI<{self.rsi_os_threshold} & StochK<{self.stoch_os_threshold}, StochK up (Oversold Proxy)"
        return signals

class QuantitativeStrategy(BaseStrategy):
    """
    A simple multi-factor model using common technical indicators.
    """
    def __init__(self, config: StrategyConfig, params: dict = None): # Corrected signature
        super().__init__(config, params) # Corrected call
        self.instrument_key = self.config.applicable_assets[0] if self.config.applicable_assets else None
        if self.instrument_key is None:
            raise ValueError("QuantitativeStrategy requires at least one instrument key in config.applicable_assets.")
        
        self.ma_short_period = self.effective_params.get('ma_short_period', 10)
        self.ma_long_period = self.effective_params.get('ma_long_period', 50)
        self.rsi_period = self.effective_params.get('rsi_period', 14)
        self.bb_period = self.effective_params.get('bb_period', 20)
        self.bb_std_dev = self.effective_params.get('bb_std_dev', 2)

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        market_data = market_data_dict.get(self.instrument_key)
        if market_data is None or market_data.empty or not all(c in market_data.columns for c in ['High', 'Low', 'Close']):
            return {self.instrument_key: pd.DataFrame(index=market_data.index if market_data is not None else None)}

        processed_data = market_data.copy()
        processed_data['ma_short'] = ta.trend.sma_indicator(processed_data['Close'], window=self.ma_short_period, fillna=True)
        processed_data['ma_long'] = ta.trend.sma_indicator(processed_data['Close'], window=self.ma_long_period, fillna=True)
        processed_data['rsi'] = ta.momentum.rsi(processed_data['Close'], window=self.rsi_period, fillna=True)
        bb = ta.volatility.BollingerBands(close=processed_data['Close'], window=self.bb_period, window_dev=self.bb_std_dev, fillna=True)
        processed_data['bb_upper'] = bb.bollinger_hband()
        processed_data['bb_lower'] = bb.bollinger_lband()
        processed_data['bb_mavg'] = bb.bollinger_mavg()
        return {self.instrument_key: processed_data}

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> pd.DataFrame:
        processed_data = processed_data_dict.get(self.instrument_key)
        if processed_data is None or processed_data.empty or not all(c in processed_data.columns for c in ['Close', 'ma_short', 'ma_long', 'rsi', 'bb_upper', 'bb_lower']):
            idx = processed_data.index if processed_data is not None else None
            signals = pd.DataFrame(index=idx)
            signals['signal'] = 0
            signals['reason'] = "Missing data for signal generation"
            return signals

        signals = pd.DataFrame(index=processed_data.index)
        signals['signal'] = 0

        # Factor scores (simple binary)
        trend_factor = (processed_data['ma_short'] > processed_data['ma_long']).astype(int) * 2 - 1 # +1 for uptrend, -1 for downtrend
        momentum_factor_bull = (processed_data['rsi'] > 55).astype(int)
        momentum_factor_bear = (processed_data['rsi'] < 45).astype(int) * -1
        momentum_factor = momentum_factor_bull + momentum_factor_bear
        
        volatility_factor_buy = (processed_data['Close'] < processed_data['bb_lower']).astype(int)
        volatility_factor_sell = (processed_data['Close'] > processed_data['bb_upper']).astype(int) * -1 # Mean reversion assumption
        volatility_factor = volatility_factor_buy + volatility_factor_sell

        # Combine factors (example: weighted or simple sum)
        combined_score = trend_factor + momentum_factor + volatility_factor

        signals.loc[combined_score >= 2, 'signal'] = 1
        signals.loc[combined_score <= -2, 'signal'] = -1
        
        signals['reason'] = "Neutral (Quant Multi-Factor)"
        signals.loc[combined_score >= 2, 'reason'] = f"Quant Score {combined_score} >= 2 (Bullish)"
        signals.loc[combined_score <= -2, 'reason'] = f"Quant Score {combined_score} <= -2 (Bearish)"
        return signals

class MarketMakingStrategy(BaseStrategy):
    """
    Simplified: Simulates placing bid/ask signals around a central moving average.
    Signals are +1 for simulated bid, -1 for simulated ask. No actual order placement logic.
    Ideally, this strategy uses L1/L2 order book data, inventory management, and sophisticated fair value estimation.
    """
    def __init__(self, config: StrategyConfig, params: dict = None): # Corrected signature
        super().__init__(config, params) # Corrected call
        self.instrument_key = self.config.applicable_assets[0] if self.config.applicable_assets else None
        if self.instrument_key is None:
            raise ValueError("MarketMakingStrategy requires at least one instrument key in config.applicable_assets.")
        
        self.ma_period = self.effective_params.get('ma_period', 20)
        self.spread_percentage = self.effective_params.get('spread_percentage', 0.001) # 0.1% spread around MA

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        market_data = market_data_dict.get(self.instrument_key)
        if market_data is None or market_data.empty or 'Close' not in market_data.columns:
            return {self.instrument_key: pd.DataFrame(index=market_data.index if market_data is not None else None)}

        processed_data = market_data.copy()
        processed_data['central_ma'] = ta.trend.sma_indicator(processed_data['Close'], window=self.ma_period, fillna=True)
        processed_data['sim_bid_price'] = processed_data['central_ma'] * (1 - self.spread_percentage / 2)
        processed_data['sim_ask_price'] = processed_data['central_ma'] * (1 + self.spread_percentage / 2)
        return {self.instrument_key: processed_data}

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> pd.DataFrame:
        processed_data = processed_data_dict.get(self.instrument_key)
        if processed_data is None or processed_data.empty or not all(c in processed_data.columns for c in ['Close', 'sim_bid_price', 'sim_ask_price']):
            idx = processed_data.index if processed_data is not None else None
            signals = pd.DataFrame(index=idx)
            signals['signal'] = 0
            signals['reason'] = "Missing data for signal generation"
            return signals

        signals = pd.DataFrame(index=processed_data.index)
        # This strategy doesn't generate simple buy/sell signals in the traditional sense.
        # It would generate quotes. We simulate this by providing target bid/ask levels.
        # A signal of 0 means hold/do nothing in terms of market orders.
        # The 'sim_bid_price' and 'sim_ask_price' are for information or a more complex execution model.
        signals['signal'] = 0 
        signals['target_bid'] = processed_data['sim_bid_price']
        signals['target_ask'] = processed_data['sim_ask_price']
        signals['reason'] = f"Market Making: Simulating quotes around MA({self.ma_period}) with {self.spread_percentage*100}% spread"
        
        # Example of generating a directional signal if price touches simulated quotes (for compatibility)
        # signals.loc[processed_data['Low'] <= processed_data['sim_bid_price'], 'signal'] = 1 # Hit our bid, simulate buy
        # signals.loc[processed_data['High'] >= processed_data['sim_ask_price'], 'signal'] = -1 # Hit our ask, simulate sell
        return signals

class HighFrequencyStrategy(BaseStrategy):
    """
    Simplified: Uses very short-term RSI and MA to capture quick fluctuations.
    Assumes input data is high-frequency (e.g., 1-minute bars).
    Ideally, this strategy uses tick data, order book dynamics, and latency-sensitive execution.
    """
    def __init__(self, config: StrategyConfig, params: dict = None): # Corrected signature
        super().__init__(config, params) # Corrected call
        self.instrument_key = self.config.applicable_assets[0] if self.config.applicable_assets else None
        if self.instrument_key is None:
            raise ValueError("HighFrequencyStrategy requires at least one instrument key in config.applicable_assets.")
        
        self.rsi_period = self.effective_params.get('rsi_period', 5) # Very short RSI
        self.ma_period = self.effective_params.get('ma_period', 3)   # Very short MA
        self.rsi_ob = self.effective_params.get('rsi_ob', 80)
        self.rsi_os = self.effective_params.get('rsi_os', 20)

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        market_data = market_data_dict.get(self.instrument_key)
        if market_data is None or market_data.empty or 'Close' not in market_data.columns:
            return {self.instrument_key: pd.DataFrame(index=market_data.index if market_data is not None else None)}

        processed_data = market_data.copy()
        processed_data['rsi'] = ta.momentum.rsi(processed_data['Close'], window=self.rsi_period, fillna=True)
        processed_data['ma'] = ta.trend.sma_indicator(processed_data['Close'], window=self.ma_period, fillna=True)
        return {self.instrument_key: processed_data}

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> pd.DataFrame:
        processed_data = processed_data_dict.get(self.instrument_key)
        if processed_data is None or processed_data.empty or not all(c in processed_data.columns for c in ['Close', 'rsi', 'ma']):
            idx = processed_data.index if processed_data is not None else None
            signals = pd.DataFrame(index=idx)
            signals['signal'] = 0
            signals['reason'] = "Missing data for signal generation"
            return signals

        signals = pd.DataFrame(index=processed_data.index)
        signals['signal'] = 0
        
        # Mean reversion on RSI extremes, confirmed by MA direction
        buy_condition = (processed_data['rsi'] < self.rsi_os) & (processed_data['Close'] > processed_data['ma']) # Oversold, price starts recovering above short MA
        sell_condition = (processed_data['rsi'] > self.rsi_ob) & (processed_data['Close'] < processed_data['ma']) # Overbought, price starts falling below short MA

        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        signals['reason'] = "Neutral (HFT Proxy)"
        signals.loc[buy_condition, 'reason'] = f"RSI({self.rsi_period})<{self.rsi_os} & Close > MA({self.ma_period}) (HFT Buy Proxy)"
        signals.loc[sell_condition, 'reason'] = f"RSI({self.rsi_period})>{self.rsi_ob} & Close < MA({self.ma_period}) (HFT Sell Proxy)"
        return signals

class AlgorithmicStrategy(BaseStrategy):
    """
    A generic algorithmic strategy, here implemented as a VWAP cross strategy.
    """
    def __init__(self, config: StrategyConfig, params: dict = None):
        super().__init__(config, params)
        self.instrument_key = self.config.applicable_assets[0] if self.config.applicable_assets else None
        if self.instrument_key is None:
            raise ValueError("AlgorithmicStrategy requires at least one instrument key in config.applicable_assets.")
        p = params if params else {}
        self.vwap_period = p.get('vwap_period', 20)

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        market_data = market_data_dict.get(self.instrument_key)
        if market_data is None or market_data.empty or not all(c in market_data.columns for c in ['High', 'Low', 'Close', 'Volume']):
            return {self.instrument_key: pd.DataFrame(index=market_data.index if market_data is not None else None)}

        processed_data = market_data.copy()
        processed_data['vwap'] = ta.volume.volume_weighted_average_price(
            high=processed_data['High'], low=processed_data['Low'], close=processed_data['Close'], volume=processed_data['Volume'], 
            window=self.vwap_period, fillna=True
        )
        return {self.instrument_key: processed_data}

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> pd.DataFrame:
        processed_data = processed_data_dict.get(self.instrument_key)
        if processed_data is None or processed_data.empty or not all(c in processed_data.columns for c in ['Close', 'vwap']):
            idx = processed_data.index if processed_data is not None else None
            signals = pd.DataFrame(index=idx)
            signals['signal'] = 0
            signals['reason'] = "Missing data for signal generation"
            return signals

        signals = pd.DataFrame(index=processed_data.index)
        signals['signal'] = 0
        
        buy_condition = (processed_data['Close'] > processed_data['vwap']) & (processed_data['Close'].shift(1) <= processed_data['vwap'].shift(1))
        sell_condition = (processed_data['Close'] < processed_data['vwap']) & (processed_data['Close'].shift(1) >= processed_data['vwap'].shift(1))

        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        signals['reason'] = "Neutral (VWAP Cross)"
        signals.loc[buy_condition, 'reason'] = f"Close crossed above VWAP({self.vwap_period})"
        signals.loc[sell_condition, 'reason'] = f"Close crossed below VWAP({self.vwap_period})"
        return signals
