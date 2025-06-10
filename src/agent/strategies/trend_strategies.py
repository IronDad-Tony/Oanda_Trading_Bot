# src/agent/strategies/trend_strategies.py
from .base_strategy import BaseStrategy, StrategyConfig
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List

# Technical Analysis library (optional, but useful)
# Ensure 'ta' is installed: pip install ta
try:
    import ta
except ImportError:
    print("Consider installing the 'ta' library for more technical indicators: pip install ta")
    ta = None

# --- Trend Strategies ---

class MomentumStrategy(BaseStrategy):
    """動量策略：基於價格動量進行交易"""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.momentum_window = int(self.config.default_params.get('momentum_window', 20))
        self.asset_list = self.config.default_params.get('asset_list', [])

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
        processed_output = {}
        momentum_period = self.params.get('momentum_period', 14)

        for asset, df in market_data_dict.items():
            if self.config.applicable_assets and asset not in self.config.applicable_assets:
                continue
            if 'close' not in df.columns or len(df) < momentum_period:
                # print(f"{self.get_strategy_name()}: Insufficient data for {asset} (need {momentum_period}, got {len(df)}).")
                processed_output[asset] = df.copy() # Pass through original if not enough data
                processed_output[asset]['momentum'] = np.nan
                continue
            
            df_copy = df.copy()
            # Simple momentum: Price change over the period
            # df_copy['momentum'] = df_copy['close'].diff(momentum_period)
            # Or, using ROC (Rate of Change) from 'ta' library if available
            if ta:
                try:
                    df_copy['momentum_roc'] = ta.momentum.ROCIndicator(close=df_copy['close'], window=momentum_period).roc()
                    df_copy['momentum'] = df_copy['momentum_roc'] # Use ROC as the primary momentum feature
                except Exception as e:
                    # print(f"Error calculating ROC for {asset} with ta library: {e}. Falling back to simple diff.")
                    df_copy['momentum'] = df_copy['close'].diff(momentum_period)
            else:
                df_copy['momentum'] = df_copy['close'].diff(momentum_period)
            
            processed_output[asset] = df_copy
        return processed_output
    
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

class BreakoutStrategy(BaseStrategy):
    """突破策略：識別並跟隨價格突破"""
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.breakout_window = int(self.config.default_params.get('breakout_window', 20))
        self.std_dev_multiplier = float(self.config.default_params.get('std_dev_multiplier', 2.0))
        self.asset_list = self.config.default_params.get('asset_list', [])

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
        processed_output = {}
        breakout_period = self.params.get('breakout_period', 20)
        std_dev_multiplier = self.params.get('std_dev_multiplier', 2.0)

        for asset, df in market_data_dict.items():
            if self.config.applicable_assets and asset not in self.config.applicable_assets:
                continue
            if 'close' not in df.columns or 'high' not in df.columns or 'low' not in df.columns or 'volume' not in df.columns or len(df) < breakout_period:
                # print(f"{self.get_strategy_name()}: Insufficient data for {asset}.")
                processed_output[asset] = df.copy()
                processed_output[asset]['upper_band'] = np.nan
                processed_output[asset]['lower_band'] = np.nan
                processed_output[asset]['avg_volume'] = np.nan
                continue

            df_copy = df.copy()
            # Using Bollinger Bands for breakout levels
            if ta:
                try:
                    bollinger = ta.volatility.BollingerBands(close=df_copy['close'], window=breakout_period, window_dev=std_dev_multiplier)
                    df_copy['upper_band'] = bollinger.bollinger_hband()
                    df_copy['lower_band'] = bollinger.bollinger_lband()
                    df_copy['sma_channel_mid'] = bollinger.bollinger_mavg()
                except Exception as e:
                    # print(f"Error calculating Bollinger Bands for {asset} with ta library: {e}. Using manual calculation.")
                    df_copy['sma_channel_mid'] = df_copy['close'].rolling(window=breakout_period).mean()
                    rolling_std = df_copy['close'].rolling(window=breakout_period).std()
                    df_copy['upper_band'] = df_copy['sma_channel_mid'] + (rolling_std * std_dev_multiplier)
                    df_copy['lower_band'] = df_copy['sma_channel_mid'] - (rolling_std * std_dev_multiplier)
            else: # Manual calculation if 'ta' is not available
                df_copy['sma_channel_mid'] = df_copy['close'].rolling(window=breakout_period).mean()
                rolling_std = df_copy['close'].rolling(window=breakout_period).std()
                df_copy['upper_band'] = df_copy['sma_channel_mid'] + (rolling_std * std_dev_multiplier)
                df_copy['lower_band'] = df_copy['sma_channel_mid'] - (rolling_std * std_dev_multiplier)
            
            df_copy['avg_volume'] = df_copy['volume'].rolling(window=breakout_period).mean()
            processed_output[asset] = df_copy
        return processed_output

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        primary_symbol = self._get_primary_symbol(processed_data_dict, portfolio_context)
        if not primary_symbol or primary_symbol not in processed_data_dict or processed_data_dict[primary_symbol].empty:
            return pd.DataFrame(columns=['signal'])

        data_df = processed_data_dict[primary_symbol]
        signals_df = pd.DataFrame(index=data_df.index, columns=['signal'], dtype=float)
        signals_df['signal'] = 0.0

        min_volume_increase_pct = self.params.get('min_breakout_volume_increase_pct', 0.5)

        if not all(col in data_df.columns for col in ['close', 'high', 'low', 'volume', 'upper_band', 'lower_band', 'avg_volume']):
            # print(f"{self.get_strategy_name()}: Missing required columns in processed data for {primary_symbol}.")
            return signals_df[['signal']]
        
        # Check for upward breakout
        # Condition: Current high breaks above upper_band, and close is also above upper_band.
        # Volume on breakout day should be significantly higher than average volume.
        upward_breakout_conditions = (
            (data_df['high'] > data_df['upper_band']) & 
            (data_df['close'] > data_df['upper_band']) & 
            (data_df['volume'] > data_df['avg_volume'] * (1 + min_volume_increase_pct))
        )
        signals_df.loc[upward_breakout_conditions, 'signal'] = 1.0

        # Check for downward breakout
        # Condition: Current low breaks below lower_band, and close is also below lower_band.
        # Volume on breakout day should be significantly higher than average volume.
        downward_breakout_conditions = (
            (data_df['low'] < data_df['lower_band']) & 
            (data_df['close'] < data_df['lower_band']) & 
            (data_df['volume'] > data_df['avg_volume'] * (1 + min_volume_increase_pct))
        )
        signals_df.loc[downward_breakout_conditions, 'signal'] = -1.0
        
        # Ensure that a position is not immediately reversed if conditions flicker
        # This might require holding state or looking at previous signal
        # For stateless signals, this is fine. If stateful, add more logic.
        # Example: if previous signal was BUY, don't immediately SELL unless strong counter-signal

        signals_df.fillna(0.0, inplace=True)
        return signals_df[['signal']]

class TrendFollowingStrategy(BaseStrategy):
    """趨勢跟隨策略：基於移動平均線交叉"""
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.short_ma_window = int(self.config.default_params.get('short_ma_window', 10))
        self.long_ma_window = int(self.config.default_params.get('long_ma_window', 50))
        self.asset_list = self.config.default_params.get('asset_list', [])

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
        processed_output = {}
        short_sma_period = self.params.get('short_sma_period', 20)
        long_sma_period = self.params.get('long_sma_period', 50)

        if short_sma_period <= 0 or long_sma_period <= 0 or long_sma_period <= short_sma_period:
            # print(f"{self.get_strategy_name()}: Invalid SMA periods. Short: {short_sma_period}, Long: {long_sma_period}")
            # Pass through original data if params are invalid
            for asset, df_orig in market_data_dict.items():
                df_copy = df_orig.copy()
                df_copy['short_sma'] = np.nan
                df_copy['long_sma'] = np.nan
                processed_output[asset] = df_copy
            return processed_output

        for asset, df in market_data_dict.items():
            if self.config.applicable_assets and asset not in self.config.applicable_assets:
                continue
            if 'close' not in df.columns or len(df) < long_sma_period:
                # print(f"{self.get_strategy_name()}: Insufficient data for {asset}.")
                df_copy = df.copy()
                df_copy['short_sma'] = np.nan
                df_copy['long_sma'] = np.nan
                processed_output[asset] = df_copy
                continue
            
            df_copy = df.copy()
            df_copy['short_sma'] = df_copy['close'].rolling(window=short_sma_period, min_periods=short_sma_period).mean()
            df_copy['long_sma'] = df_copy['close'].rolling(window=long_sma_period, min_periods=long_sma_period).mean()
            processed_output[asset] = df_copy
        return processed_output

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        primary_symbol = self._get_primary_symbol(processed_data_dict, portfolio_context)
        if not primary_symbol or primary_symbol not in processed_data_dict or processed_data_dict[primary_symbol].empty:
            return pd.DataFrame(columns=['signal'])

        data_df = processed_data_dict[primary_symbol]
        signals_df = pd.DataFrame(index=data_df.index, columns=['signal'], dtype=float)
        signals_df['signal'] = 0.0

        if not all(col in data_df.columns for col in ['short_sma', 'long_sma']) or data_df[['short_sma', 'long_sma']].isnull().all().all():
            # print(f"{self.get_strategy_name()}: SMA columns not found or all NaN in processed data for {primary_symbol}.")
            return signals_df[['signal']]

        # Golden Cross: short_sma crosses above long_sma
        buy_condition = (data_df['short_sma'] > data_df['long_sma']) & (data_df['short_sma'].shift(1) <= data_df['long_sma'].shift(1))
        signals_df.loc[buy_condition, 'signal'] = 1.0

        # Death Cross: short_sma crosses below long_sma
        sell_condition = (data_df['short_sma'] < data_df['long_sma']) & (data_df['short_sma'].shift(1) >= data_df['long_sma'].shift(1))
        signals_df.loc[sell_condition, 'signal'] = -1.0
        
        # Hold signal if already in a trend (optional, simple version just signals on cross)
        # To implement holding: forward fill signals after a cross until the next cross
        # signals_df['signal'] = signals_df['signal'].replace(0.0, np.nan).ffill().fillna(0.0)

        signals_df.fillna(0.0, inplace=True)
        return signals_df[['signal']]

class ReversalStrategy(BaseStrategy):
    """反轉策略：基於RSI指標的超買超賣"""
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.reversal_window = int(self.config.default_params.get('reversal_window', 14))
        self.rsi_oversold = float(self.config.default_params.get('rsi_oversold', 30))
        self.rsi_overbought = float(self.config.default_params.get('rsi_overbought', 70))
        self.asset_list = self.config.default_params.get('asset_list', [])

    def _calculate_rsi(self, series: pd.Series, period: int) -> Optional[pd.Series]:
        if not isinstance(series, pd.Series) or series.empty or period <= 0 or len(series) < period:
            return None
        if ta:
            try:
                return ta.momentum.RSIIndicator(close=series, window=period).rsi()
            except Exception as e:
                # print(f"Error calculating RSI with ta library: {e}. Using manual calculation.")
                pass # Fall through to manual
        
        # Manual RSI calculation
        delta = series.diff()
        gain = (delta.where(delta > 0, 0.0))
        loss = (-delta.where(delta < 0, 0.0))
        # Use exponential moving average for gain/loss for standard RSI
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int, d_period: int) -> Optional[pd.DataFrame]:
        if not all(isinstance(s, pd.Series) for s in [high, low, close]) or k_period <= 0 or d_period <=0 or not all(len(s) >= k_period for s in [high, low, close]):
            return None
        if ta:
            try:
                stoch_indicator = ta.momentum.StochasticOscillator(
                    high=high, low=low, close=close, 
                    window=k_period, smooth_window=d_period, fillna=False
                )
                return pd.DataFrame({'stoch_k': stoch_indicator.stoch(), 'stoch_d': stoch_indicator.stoch_signal()})
            except Exception as e:
                # print(f"Error calculating Stochastic with ta library: {e}. Manual calculation not implemented for Stochastic.")
                return None # Or implement manual if 'ta' fails and is critical
        return None # If 'ta' is not available and no manual implementation

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
        processed_output = {}
        rsi_period = self.params.get('rsi_period', 14)
        stoch_k_period = self.params.get('stoch_k_period', 14)
        stoch_d_period = self.params.get('stoch_d_period', 3)
        use_stochastic = self.params.get('use_stochastic_confirmation', True)

        for asset, df in market_data_dict.items():
            if self.config.applicable_assets and asset not in self.config.applicable_assets:
                continue
            
            df_copy = df.copy()
            if 'close' not in df_copy.columns:
                # print(f"{self.get_strategy_name()}: 'close' column missing for {asset}.")
                df_copy['rsi'] = np.nan
                if use_stochastic: df_copy['stoch_k'] = np.nan; df_copy['stoch_d'] = np.nan
                processed_output[asset] = df_copy
                continue

            rsi_series = self._calculate_rsi(df_copy['close'], rsi_period)
            df_copy['rsi'] = rsi_series if rsi_series is not None else np.nan

            if use_stochastic:
                if 'high' in df_copy.columns and 'low' in df_copy.columns:
                    stoch_df = self._calculate_stochastic(df_copy['high'], df_copy['low'], df_copy['close'], stoch_k_period, stoch_d_period)
                    if stoch_df is not None:
                        df_copy['stoch_k'] = stoch_df['stoch_k']
                        df_copy['stoch_d'] = stoch_df['stoch_d']
                    else:
                        df_copy['stoch_k'] = np.nan
                        df_copy['stoch_d'] = np.nan
                else:
                    # print(f"{self.get_strategy_name()}: 'high' or 'low' columns missing for stochastic calculation in {asset}.")
                    df_copy['stoch_k'] = np.nan
                    df_copy['stoch_d'] = np.nan
            
            processed_output[asset] = df_copy
        return processed_output

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        primary_symbol = self._get_primary_symbol(processed_data_dict, portfolio_context)
        if not primary_symbol or primary_symbol not in processed_data_dict or processed_data_dict[primary_symbol].empty:
            return pd.DataFrame(columns=['signal'])

        data_df = processed_data_dict[primary_symbol]
        signals_df = pd.DataFrame(index=data_df.index, columns=['signal'], dtype=float)
        signals_df['signal'] = 0.0

        rsi_oversold = self.params.get('rsi_oversold', 30)
        rsi_overbought = self.params.get('rsi_overbought', 70)
        use_stochastic = self.params.get('use_stochastic_confirmation', True)
        stoch_oversold = self.params.get('stoch_oversold', 20)
        stoch_overbought = self.params.get('stoch_overbought', 80)

        if 'rsi' not in data_df.columns or data_df['rsi'].isnull().all():
            # print(f"{self.get_strategy_name()}: RSI not available for {primary_symbol}.")
            return signals_df[['signal']]

        # Base RSI signals
        rsi_buy_signal = data_df['rsi'] < rsi_oversold
        rsi_sell_signal = data_df['rsi'] > rsi_overbought

        if use_stochastic:
            if 'stoch_k' in data_df.columns and 'stoch_d' in data_df.columns and \
               not data_df['stoch_k'].isnull().all() and not data_df['stoch_d'].isnull().all():
                
                stoch_buy_condition = (data_df['stoch_k'] < stoch_oversold) & \
                                      (data_df['stoch_d'] < stoch_oversold) & \
                                      (data_df['stoch_k'] > data_df['stoch_d']) & \
                                      (data_df['stoch_k'].shift(1) <= data_df['stoch_d'].shift(1)) # K crosses above D in oversold
                
                stoch_sell_condition = (data_df['stoch_k'] > stoch_overbought) & \
                                       (data_df['stoch_d'] > stoch_overbought) & \
                                       (data_df['stoch_k'] < data_df['stoch_d']) & \
                                       (data_df['stoch_k'].shift(1) >= data_df['stoch_d'].shift(1)) # K crosses below D in overbought

                # Combined signals
                signals_df.loc[rsi_buy_signal & stoch_buy_condition, 'signal'] = 1.0
                signals_df.loc[rsi_sell_signal & stoch_sell_condition, 'signal'] = -1.0
            else:
                # print(f"{self.get_strategy_name()}: Stochastic confirmation enabled but indicators not available for {primary_symbol}. Using RSI only.")
                signals_df.loc[rsi_buy_signal, 'signal'] = 1.0
                signals_df.loc[rsi_sell_signal, 'signal'] = -1.0
        else:
            # RSI only signals
            signals_df.loc[rsi_buy_signal, 'signal'] = 1.0
            signals_df.loc[rsi_sell_signal, 'signal'] = -1.0

        signals_df.fillna(0.0, inplace=True)
        return signals_df[['signal']]
