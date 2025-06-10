# src/agent/strategies/risk_management_strategies.py
from .base_strategy import BaseStrategy, StrategyConfig
import pandas as pd
import numpy as np
import ta # For DynamicHedgingStrategy
from scipy.stats import norm # For VaRControlStrategy
from typing import Dict, Optional, List, Any

class DynamicHedgingStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig, params: Dict[str, Any] = None):
        super().__init__(config, params)
        # Use effective_params to get instrument_key, falling back to the first applicable_asset
        self.instrument_key = self.effective_params.get('instrument_key', self.config.applicable_assets[0] if self.config.applicable_assets else None)
        if self.instrument_key is None:
            raise ValueError("DynamicHedgingStrategy requires at least one instrument key in config or params.")

        self.atr_period = self.effective_params.get('atr_period', 14)
        self.atr_multiplier_threshold = self.effective_params.get('atr_multiplier_threshold', 2.0) # Price drop > 2*ATR

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        market_data = market_data_dict.get(self.instrument_key)
        if market_data is None or market_data.empty or not all(col in market_data.columns for col in ['High', 'Low', 'Close']):
            return {self.instrument_key: pd.DataFrame(index=market_data.index if market_data is not None else None)}

        processed_data = market_data.copy()
        processed_data['atr'] = ta.volatility.average_true_range(
            high=processed_data['High'], 
            low=processed_data['Low'], 
            close=processed_data['Close'], 
            window=self.atr_period, 
            fillna=True
        )
        # Price change: positive if price drops (Close.shift(1) - Close)
        processed_data['price_change'] = processed_data['Close'].shift(1) - processed_data['Close']
        processed_data['price_change_vs_atr'] = (processed_data['price_change'] / processed_data['atr']).fillna(0)
        
        return {self.instrument_key: processed_data}

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> pd.DataFrame:
        processed_data = processed_data_dict.get(self.instrument_key)
        
        if processed_data is None or processed_data.empty or 'price_change_vs_atr' not in processed_data.columns:
            idx = processed_data.index if processed_data is not None else None
            signals = pd.DataFrame(index=idx)
            signals['signal'] = 0
            signals['reason'] = "No data for signal generation"
            return signals

        signals = pd.DataFrame(index=processed_data.index)
        signals['signal'] = np.where(processed_data['price_change_vs_atr'] > self.atr_multiplier_threshold, -1, 0) # -1 to hedge/reduce risk
        signals['reason'] = np.where(
            processed_data['price_change_vs_atr'] > self.atr_multiplier_threshold, 
            f"Price drop > {self.atr_multiplier_threshold}*ATR", 
            "No hedge signal"
        )
        return signals

class RiskParityStrategy(BaseStrategy): # Simplified for single asset: volatility-based risk adjustment
    def __init__(self, config: StrategyConfig, params: Dict[str, Any] = None):
        super().__init__(config, params)
        self.instrument_key = self.effective_params.get('instrument_key', self.config.applicable_assets[0] if self.config.applicable_assets else None)
        if self.instrument_key is None:
            raise ValueError("RiskParityStrategy requires at least one instrument key in config or params.")

        self.vol_window = self.effective_params.get('vol_window', 20)
        self.high_vol_threshold_pct = self.effective_params.get('high_vol_threshold_pct', 0.02) # e.g. 2% daily vol
        self.low_vol_threshold_pct = self.effective_params.get('low_vol_threshold_pct', 0.005)  # e.g. 0.5% daily vol

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        market_data = market_data_dict.get(self.instrument_key)
        if market_data is None or market_data.empty or 'Close' not in market_data.columns:
            return {self.instrument_key: pd.DataFrame(index=market_data.index if market_data is not None else None)}

        processed_data = market_data.copy()
        processed_data['returns'] = processed_data['Close'].pct_change()
        processed_data['volatility'] = processed_data['returns'].rolling(window=self.vol_window, min_periods=max(1, self.vol_window//2)).std().fillna(0)
        return {self.instrument_key: processed_data}

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> pd.DataFrame:
        processed_data = processed_data_dict.get(self.instrument_key)

        if processed_data is None or processed_data.empty or 'volatility' not in processed_data.columns:
            idx = processed_data.index if processed_data is not None else None
            signals = pd.DataFrame(index=idx)
            signals['signal'] = 0
            signals['reason'] = "No data for signal generation"
            return signals
            
        signals = pd.DataFrame(index=processed_data.index)
        signals['signal'] = 0 # Default to hold
        
        high_vol_condition = processed_data['volatility'] > self.high_vol_threshold_pct
        low_vol_condition = processed_data['volatility'] < self.low_vol_threshold_pct
        
        signals.loc[high_vol_condition, 'signal'] = -1 # Reduce risk
        signals.loc[high_vol_condition, 'reason'] = f"Volatility > {self.high_vol_threshold_pct*100}%"
        
        signals.loc[low_vol_condition, 'signal'] = 1  # Increase risk appetite (or enable other strategies)
        signals.loc[low_vol_condition, 'reason'] = f"Volatility < {self.low_vol_threshold_pct*100}%"
        
        signals['reason'].fillna("Volatility within thresholds", inplace=True)
        return signals

class VaRControlStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig, params: Dict[str, Any] = None):
        super().__init__(config, params)
        self.instrument_key = self.effective_params.get('instrument_key', self.config.applicable_assets[0] if self.config.applicable_assets else None)
        if self.instrument_key is None:
            raise ValueError("VaRControlStrategy requires at least one instrument key in config or params.")

        self.var_window = self.effective_params.get('var_window', 20)
        self.var_confidence = self.effective_params.get('var_confidence', 0.99) # 99% VaR
        self.var_limit = self.effective_params.get('var_limit', 0.02) # Max 2% VaR (positive number)
        self.z_score = norm.ppf(self.var_confidence)

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        market_data = market_data_dict.get(self.instrument_key)
        if market_data is None or market_data.empty or 'Close' not in market_data.columns:
            return {self.instrument_key: pd.DataFrame(index=market_data.index if market_data is not None else None)}

        processed_data = market_data.copy()
        processed_data['returns'] = processed_data['Close'].pct_change()
        processed_data['rolling_std'] = processed_data['returns'].rolling(window=self.var_window, min_periods=max(1,self.var_window//2)).std().fillna(0)
        # Parametric VaR (positive value for loss): std_dev * z_score (assuming mean return is 0 for short horizon)
        processed_data['estimated_var'] = processed_data['rolling_std'] * self.z_score
        return {self.instrument_key: processed_data}

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> pd.DataFrame:
        processed_data = processed_data_dict.get(self.instrument_key)

        if processed_data is None or processed_data.empty or 'estimated_var' not in processed_data.columns:
            idx = processed_data.index if processed_data is not None else None
            signals = pd.DataFrame(index=idx)
            signals['signal'] = 0
            signals['reason'] = "No data for signal generation"
            return signals

        signals = pd.DataFrame(index=processed_data.index)
        condition = processed_data['estimated_var'] > self.var_limit
        signals['signal'] = np.where(condition, -1, 0) # -1 to reduce risk
        signals['reason'] = np.where(condition, f"Estimated VaR > {self.var_limit*100}%", "VaR within limit")
        return signals

class MaxDrawdownControlStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig, params: Dict[str, Any] = None):
        super().__init__(config, params)
        self.instrument_key = self.effective_params.get('instrument_key', self.config.applicable_assets[0] if self.config.applicable_assets else None)
        if self.instrument_key is None:
            raise ValueError("MaxDrawdownControlStrategy requires at least one instrument key in config or params.")
            
        self.max_drawdown_limit = self.effective_params.get('max_drawdown_limit', 0.10) # 10%
        self.hwm_scalar = -float('inf') # Persistent High Water Mark scalar

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        market_data = market_data_dict.get(self.instrument_key)
        if market_data is None or market_data.empty or 'Close' not in market_data.columns:
            return {self.instrument_key: pd.DataFrame(index=market_data.index if market_data is not None else None)}

        processed_data = market_data.copy()
        
        if 'Close' in processed_data and not processed_data.empty:
            hwm_series = []
            # Use a temporary variable for HWM calculation within this batch, starting from the persistent HWM
            current_batch_hwm = self.hwm_scalar 
            for price in processed_data['Close']:
                if pd.notna(price):
                    current_batch_hwm = max(current_batch_hwm, price)
                hwm_series.append(current_batch_hwm if current_batch_hwm != -float('inf') else np.nan) # Store NaN if HWM not established
            
            processed_data['high_water_mark'] = hwm_series
            
            # Update the persistent scalar HWM with the latest peak from this batch
            if hwm_series and pd.notna(hwm_series[-1]):
                self.hwm_scalar = hwm_series[-1]

            # Calculate drawdown, ensure HWM is not zero or NaN to avoid division issues
            processed_data['drawdown'] = np.where(
                (processed_data['high_water_mark'].notna()) & (processed_data['high_water_mark'] > 0),
                (processed_data['high_water_mark'] - processed_data['Close']) / processed_data['high_water_mark'],
                0.0 # Default drawdown to 0 if HWM is not valid
            )
            processed_data['drawdown'].fillna(0.0, inplace=True) # Ensure no NaNs in drawdown
        else:
            processed_data['high_water_mark'] = np.nan
            processed_data['drawdown'] = 0.0
            
        return {self.instrument_key: processed_data}

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict] = None) -> pd.DataFrame:
        processed_data = processed_data_dict.get(self.instrument_key)

        if processed_data is None or processed_data.empty or 'drawdown' not in processed_data.columns:
            idx = processed_data.index if processed_data is not None else None
            signals = pd.DataFrame(index=idx)
            signals['signal'] = 0
            signals['reason'] = "No data for signal generation"
            return signals

        signals = pd.DataFrame(index=processed_data.index)
        condition = processed_data['drawdown'] > self.max_drawdown_limit
        signals['signal'] = np.where(condition, -1, 0) # -1 to reduce risk
        signals['reason'] = np.where(condition, f"Drawdown > {self.max_drawdown_limit*100}%", "Drawdown within limit")
        return signals
