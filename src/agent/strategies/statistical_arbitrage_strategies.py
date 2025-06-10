# src/agent/strategies/statistical_arbitrage_strategies.py
import pandas as pd
import numpy as np
import ta
from ta.utils import dropna
from .base_strategy import BaseStrategy, StrategyConfig
from typing import Dict, List, Optional, Any

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig, params: Dict[str, Any] = None):
        super().__init__(config, params)
        self.bb_period = int(self.effective_params.get('bb_period', 20))
        self.bb_std_dev = float(self.effective_params.get('bb_std_dev', 2.0))
        # Ensure asset_list is correctly fetched from effective_params or defaults to applicable_assets
        self.asset_list = self.effective_params.get('asset_list', self.config.applicable_assets)

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context=None) -> Dict[str, pd.DataFrame]:
        processed_data = {}
        # Use self.asset_list which is now correctly initialized
        assets_to_process = self.asset_list if self.asset_list else market_data_dict.keys()

        for asset in assets_to_process:
            if asset not in market_data_dict:
                # print(f"{self.config.name}: Market data for {asset} not found. Skipping.")
                continue
            
            df = market_data_dict[asset].copy()
            # Ensure 'close' is derived from 'Close' if not present, common in OANDA data
            if 'close' not in df.columns and 'Close' in df.columns:
                df['close'] = df['Close']
            elif 'close' not in df.columns:
                # print(f"{self.config.name}: 'close' column missing for {asset}. Skipping.")
                continue

            df = dropna(df)

            if len(df) < self.bb_period:
                # print(f"{self.config.name}: Insufficient data for {asset} ({len(df)} points) for BB period {self.bb_period}. Skipping.")
                continue

            try:
                bollinger = ta.volatility.BollingerBands(close=df['close'], window=self.bb_period, window_dev=self.bb_std_dev)
                df['mavg'] = bollinger.bollinger_mavg()
                df['hband'] = bollinger.bollinger_hband()
                df['lband'] = bollinger.bollinger_lband()
                
                processed_data[asset] = df.dropna(subset=['mavg', 'hband', 'lband'])
            except Exception as e:
                # print(f"{self.config.name}: Error calculating Bollinger Bands for {asset}: {e}. Skipping.")
                continue
        
        return processed_data

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context=None) -> pd.DataFrame:
        all_asset_signals_dfs = []

        for asset, df_processed in processed_data_dict.items():
            # Ensure 'close' is derived from 'Close' if not present
            if 'close' not in df_processed.columns and 'Close' in df_processed.columns:
                df_processed['close'] = df_processed['Close']
            elif 'close' not in df_processed.columns:
                # print(f"{self.config.name}: Processed data for {asset} is missing 'close' column. Skipping.")
                continue

            if df_processed.empty or not all(col in df_processed.columns for col in ['close', 'hband', 'lband']):
                # print(f"{self.config.name}: Processed data for {asset} is empty or missing BB columns. Skipping.")
                continue

            if not isinstance(df_processed.index, pd.DatetimeIndex):
                 df_processed.index = pd.to_datetime(df_processed.index)

            signals_df = pd.DataFrame(index=df_processed.index)
            signals_df[asset] = 0 

            signals_df.loc[df_processed['close'] < df_processed['lband'], asset] = 1
            signals_df.loc[df_processed['close'] > df_processed['hband'], asset] = -1
            
            all_asset_signals_dfs.append(signals_df[[asset]])

        if not all_asset_signals_dfs:
            return pd.DataFrame() 
            
        final_signals_df = pd.concat(all_asset_signals_dfs, axis=1)
        final_signals_df = final_signals_df.fillna(method='ffill').fillna(0) 
        final_signals_df = final_signals_df.astype(int)
        
        return final_signals_df

class CointegrationStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig, params: Dict[str, Any] = None):
        super().__init__(config, params)
        self.asset_pair = self.effective_params.get('asset_pair', [])
        self.window = int(self.effective_params.get('window', 60))
        self.z_threshold = float(self.effective_params.get('z_threshold', 2.0))
        
        if len(self.asset_pair) != 2:
            print(f"Warning: {self.config.name} CointegrationStrategy requires 'asset_pair' to be a list of two asset names. Strategy may not function.")
            self.valid_pair = False
        else:
            self.valid_pair = True


    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context=None) -> Dict[str, pd.DataFrame]:
        if not self.valid_pair:
            return {}
            
        processed_data = {}
        asset1_name, asset2_name = self.asset_pair

        if asset1_name not in market_data_dict or asset2_name not in market_data_dict:
            # print(f"{self.config.name}: Data for one or both assets in pair {self.asset_pair} not found. Skipping.")
            return processed_data

        # Ensure 'close' is derived from 'Close' if not present
        df1_original = market_data_dict[asset1_name].copy()
        if 'close' not in df1_original.columns and 'Close' in df1_original.columns:
            df1_original['close'] = df1_original['Close']
        df1 = df1_original[['close']].rename(columns={'close': 'price1'})
        
        df2_original = market_data_dict[asset2_name].copy()
        if 'close' not in df2_original.columns and 'Close' in df2_original.columns:
            df2_original['close'] = df2_original['Close']
        df2 = df2_original[['close']].rename(columns={'close': 'price2'})

        data = pd.concat([df1, df2], axis=1).dropna()
        
        if len(data) < self.window:
            # print(f"{self.config.name}: Insufficient aligned data for pair {self.asset_pair} (need {self.window}, got {len(data)}). Skipping.")
            return processed_data

        try:
            from statsmodels.regression.rolling import RollingOLS
            import statsmodels.api as sm

            data['price2_float'] = data['price2'].astype(float)
            exog = sm.add_constant(data['price2_float'])
            
            rols = RollingOLS(endog=data['price1'], exog=exog, window=self.window, min_nobs=self.window)
            results = rols.fit()
            hedge_ratios = results.params['price2_float'] 
            intercepts = results.params['const']

            data['hedge_ratio'] = hedge_ratios
            data['intercept'] = intercepts
            data = data.dropna(subset=['hedge_ratio', 'intercept'])

            if data.empty:
                # print(f"{self.config.name}: Data empty after rolling OLS for {self.asset_pair}. Skipping.")
                return processed_data

            data['spread'] = data['price1'] - data['hedge_ratio'] * data['price2'] - data['intercept']
            
            spread_mean = data['spread'].rolling(window=self.window, min_periods=self.window).mean()
            spread_std = data['spread'].rolling(window=self.window, min_periods=self.window).std()
            
            data['z_score'] = (data['spread'] - spread_mean) / spread_std.replace(0, np.nan) 
            
            pair_name = f"{asset1_name}_{asset2_name}_coint_spread"
            processed_data[pair_name] = data.dropna(subset=['z_score'])

        except ImportError:
            print(f"CRITICAL: {self.config.name} requires 'statsmodels'. Please install it (pip install statsmodels). Skipping pair.")
            self.valid_pair = False
            return processed_data
        except Exception as e:
            # print(f"{self.config.name}: Error calculating spread/z-score for {self.asset_pair}: {e}. Skipping.")
            return processed_data
            
        return processed_data

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context=None) -> pd.DataFrame:
        if not self.valid_pair:
            return pd.DataFrame()

        all_asset_signals_dfs = []
        asset1_name, asset2_name = self.asset_pair
        pair_name = f"{asset1_name}_{asset2_name}_coint_spread"

        if pair_name not in processed_data_dict or processed_data_dict[pair_name].empty:
            # print(f"{self.config.name}: Processed data for pair {pair_name} not found or empty. Skipping.")
            return pd.DataFrame()

        df_spread = processed_data_dict[pair_name]
        
        if 'z_score' not in df_spread.columns:
            # print(f"{self.config.name}: 'z_score' missing in processed data for {pair_name}. Skipping.")
            return pd.DataFrame()

        if not isinstance(df_spread.index, pd.DatetimeIndex):
            df_spread.index = pd.to_datetime(df_spread.index)

        signals1 = pd.DataFrame(index=df_spread.index)
        signals1[asset1_name] = 0
        signals2 = pd.DataFrame(index=df_spread.index)
        signals2[asset2_name] = 0

        signals1.loc[df_spread['z_score'] < -self.z_threshold, asset1_name] = 1
        signals2.loc[df_spread['z_score'] < -self.z_threshold, asset2_name] = -1

        signals1.loc[df_spread['z_score'] > self.z_threshold, asset1_name] = -1
        signals2.loc[df_spread['z_score'] > self.z_threshold, asset2_name] = 1
        
        all_asset_signals_dfs.append(signals1[[asset1_name]])
        all_asset_signals_dfs.append(signals2[[asset2_name]])
        
        if not all_asset_signals_dfs:
            return pd.DataFrame()
            
        final_signals_df = pd.concat(all_asset_signals_dfs, axis=1)
        final_signals_df = final_signals_df.fillna(method='ffill').fillna(0)
        final_signals_df = final_signals_df.astype(int)
        
        return final_signals_df

class StatisticalArbitrageStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig, params: Dict[str, Any] = None):
        super().__init__(config, params) # Call to super().__init__ added
        self.base_strategies_config = self.effective_params.get('base_strategies_config', [])
        self.strategies = []
        for strategy_conf_dict in self.base_strategies_config:
            strategy_name = strategy_conf_dict.get('name')
            strategy_params = strategy_conf_dict.get('params')
            strategy_applicable_assets = strategy_conf_dict.get('applicable_assets', self.config.applicable_assets)

            if strategy_name == "MeanReversionStrategy":
                conf = StrategyConfig(name=strategy_name, default_params=strategy_params, applicable_assets=strategy_applicable_assets)
                self.strategies.append(MeanReversionStrategy(config=conf, params=strategy_params))
            elif strategy_name == "CointegrationStrategy":
                conf = StrategyConfig(name=strategy_name, default_params=strategy_params, applicable_assets=strategy_applicable_assets)
                self.strategies.append(CointegrationStrategy(config=conf, params=strategy_params))
            # Add other strategies as needed

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context=None) -> Dict[str, pd.DataFrame]:
        all_processed_data = {}
        for strategy in self.strategies:
            processed_data = strategy.forward(market_data_dict, portfolio_context)
            for asset, data in processed_data.items():
                if asset not in all_processed_data:
                    all_processed_data[asset] = data.copy()
                else:
                    # Simple merge, consider more sophisticated merging if column names clash or data needs alignment
                    all_processed_data[asset] = pd.merge(all_processed_data[asset], data, left_index=True, right_index=True, how='outer', suffixes=('', f'_{strategy.config.name}'))
        return all_processed_data

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context=None) -> pd.DataFrame:
        final_signals = pd.DataFrame()
        for strategy in self.strategies:
            # Each strategy needs its own relevant slice of processed_data_dict
            # This might need refinement based on how strategies name their output columns in forward
            strategy_specific_processed_data = {}
            for asset_key in strategy.config.applicable_assets:
                if asset_key in processed_data_dict:
                    strategy_specific_processed_data[asset_key] = processed_data_dict[asset_key]
                # For CointegrationStrategy, the key might be a pair name
                elif hasattr(strategy, 'asset_pair') and strategy.valid_pair:
                    pair_name = f"{strategy.asset_pair[0]}_{strategy.asset_pair[1]}_coint_spread"
                    if pair_name in processed_data_dict:
                         strategy_specific_processed_data[pair_name] = processed_data_dict[pair_name]

            if not strategy_specific_processed_data and not isinstance(strategy, CointegrationStrategy): # Cointegration can have empty if pair name not in dict
                 # print(f"No specific processed data for strategy {strategy.config.name}, assets: {strategy.config.applicable_assets}")
                 # print(f"Available keys in processed_data_dict: {list(processed_data_dict.keys())}")
                 continue # Skip if no relevant data for this strategy

            signals = strategy.generate_signals(strategy_specific_processed_data, portfolio_context)
            if not signals.empty:
                if final_signals.empty:
                    final_signals = signals
                else:
                    # Combine signals: sum for now, could be more complex (e.g., voting, weighted average)
                    final_signals = final_signals.add(signals, fill_value=0).astype(int)
        return final_signals

class VolatilityBreakoutStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig, params: Dict[str, Any] = None):
        super().__init__(config, params)
        self.atr_period = int(self.effective_params.get('atr_period', 14))
        self.donchian_period = int(self.effective_params.get('donchian_period', self.effective_params.get('atr_period', 14))) # Default Donchian to ATR period if not specified
        self.instrument_key = self.effective_params.get('instrument_key', self.config.applicable_assets[0] if self.config.applicable_assets else None)
        if self.instrument_key is None:
            raise ValueError("VolatilityBreakoutStrategy requires an instrument_key in params or applicable_assets in config.")

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        market_data = market_data_dict.get(self.instrument_key)
        if market_data is None or market_data.empty:
            return {self.instrument_key: pd.DataFrame(index=market_data.index if market_data is not None else None)}

        # Ensure required columns are present, deriving 'high', 'low', 'close' from 'High', 'Low', 'Close'
        df = market_data.copy()
        required_ohlc = ['High', 'Low', 'Close']
        if not all(col in df.columns for col in required_ohlc):
            # print(f"{self.config.name}: Market data for {self.instrument_key} missing one or more of {required_ohlc}. Skipping.")
            return {self.instrument_key: pd.DataFrame(index=df.index)}
        
        df['high'] = df['High']
        df['low'] = df['Low']
        df['close'] = df['Close']

        df = dropna(df)

        if len(df) < max(self.atr_period, self.donchian_period):
            # print(f"{self.config.name}: Insufficient data for {self.instrument_key} for ATR/Donchian. Skipping.")
            return {self.instrument_key: pd.DataFrame(index=df.index)}

        try:
            # ATR
            atr_indicator = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=self.atr_period, fillna=True)
            df['atr'] = atr_indicator.average_true_range()

            # Donchian Channel
            donchian_indicator = ta.volatility.DonchianChannel(high=df['high'], low=df['low'], close=df['close'], window=self.donchian_period, offset=0, fillna=True)
            df['donchian_hband'] = donchian_indicator.donchian_channel_hband()
            df['donchian_lband'] = donchian_indicator.donchian_channel_lband()
            df['donchian_mband'] = donchian_indicator.donchian_channel_mband()

            # Ensure columns exist before trying to dropna on them
            df.dropna(subset=['atr', 'donchian_hband', 'donchian_lband'], inplace=True)

        except Exception as e:
            # print(f"{self.config.name}: Error calculating indicators for {self.instrument_key}: {e}. Skipping.")
            return {self.instrument_key: pd.DataFrame(index=df.index)}

        return {self.instrument_key: df}

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> pd.DataFrame:
        processed_data = processed_data_dict.get(self.instrument_key)
        
        if processed_data is None or processed_data.empty or not all(col in processed_data.columns for col in ['close', 'donchian_hband', 'donchian_lband']):
            idx = processed_data.index if processed_data is not None else None
            signals = pd.DataFrame(index=idx)
            signals['signal'] = 0
            signals['reason'] = "No data or required columns missing"
            return signals

        signals = pd.DataFrame(index=processed_data.index)
        signals['signal'] = 0

        # Breakout above Donchian H-Band
        signals.loc[processed_data['close'] > processed_data['donchian_hband'], 'signal'] = 1
        # Breakout below Donchian L-Band
        signals.loc[processed_data['close'] < processed_data['donchian_lband'], 'signal'] = -1
        
        # Add reasons for signals
        signals['reason'] = "No breakout"
        signals.loc[signals['signal'] == 1, 'reason'] = f"Breakout above {self.donchian_period}-period Donchian High"
        signals.loc[signals['signal'] == -1, 'reason'] = f"Breakout below {self.donchian_period}-period Donchian Low"
        
        return signals

class VolatilityArbitrageStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig, params: Dict[str, Any] = None):
        super().__init__(config, params)
        self.instrument_key = self.effective_params.get('instrument_key', self.config.applicable_assets[0] if self.config.applicable_assets else None)
        if self.instrument_key is None:
            raise ValueError("VolatilityArbitrageStrategy requires an instrument_key in params or applicable_assets in config.")
        self.vol_window_short = int(self.effective_params.get('vol_window_short', 10))
        self.vol_window_long = int(self.effective_params.get('vol_window_long', 50))
        self.vol_threshold_ratio = float(self.effective_params.get('vol_threshold_ratio', 1.5))

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        market_data = market_data_dict.get(self.instrument_key)
        if market_data is None or market_data.empty:
            return {self.instrument_key: pd.DataFrame(index=market_data.index if market_data is not None else None)}

        df = market_data.copy()
        if 'Close' not in df.columns:
            # print(f"{self.config.name}: 'Close' column missing for {self.instrument_key}. Skipping.")
            return {self.instrument_key: pd.DataFrame(index=df.index)}
        
        df['returns'] = df['Close'].pct_change()
        df = dropna(df)

        if len(df) < self.vol_window_long:
            # print(f"{self.config.name}: Insufficient data for {self.instrument_key} for volatility calculation. Skipping.")
            return {self.instrument_key: pd.DataFrame(index=df.index)}

        try:
            df['vol_short'] = df['returns'].rolling(window=self.vol_window_short, min_periods=max(1, self.vol_window_short//2)).std() * np.sqrt(252) # Annualized
            df['vol_long'] = df['returns'].rolling(window=self.vol_window_long, min_periods=max(1, self.vol_window_long//2)).std() * np.sqrt(252) # Annualized
            df.dropna(subset=['vol_short', 'vol_long'], inplace=True)
            df['vol_ratio'] = df['vol_short'] / df['vol_long']
            df['vol_ratio'].fillna(1, inplace=True) # Avoid NaN if long vol is zero

        except Exception as e:
            # print(f"{self.config.name}: Error calculating volatility for {self.instrument_key}: {e}. Skipping.")
            return {self.instrument_key: pd.DataFrame(index=df.index)}

        return {self.instrument_key: df}

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> pd.DataFrame:
        processed_data = processed_data_dict.get(self.instrument_key)

        if processed_data is None or processed_data.empty or 'vol_ratio' not in processed_data.columns:
            idx = processed_data.index if processed_data is not None else None
            signals = pd.DataFrame(index=idx)
            signals['signal'] = 0
            signals['reason'] = "No data or vol_ratio missing"
            return signals

        signals = pd.DataFrame(index=processed_data.index)
        signals['signal'] = 0

        # Example: Go long volatility (e.g., buy straddle/strangle, or trade vol ETP) if short-term vol is much higher than long-term
        signals.loc[processed_data['vol_ratio'] > self.vol_threshold_ratio, 'signal'] = 1 
        # Example: Go short volatility if short-term vol is much lower (or mean-reverting)
        signals.loc[processed_data['vol_ratio'] < (1/self.vol_threshold_ratio), 'signal'] = -1

        signals['reason'] = "Vol ratio within threshold"
        signals.loc[signals['signal'] == 1, 'reason'] = f"Short vol > {self.vol_threshold_ratio} * Long vol"
        signals.loc[signals['signal'] == -1, 'reason'] = f"Short vol < {1/self.vol_threshold_ratio:.2f} * Long vol"

        return signals
