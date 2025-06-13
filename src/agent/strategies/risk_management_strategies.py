# src/agent/strategies/risk_management_strategies.py
from .base_strategy import BaseStrategy, StrategyConfig
import pandas as pd
import numpy as np
import ta # For DynamicHedgingStrategy
from scipy.stats import norm # For VaRControlStrategy
from typing import Dict, Optional, List, Any
import logging # Added for logger
import torch # Added for PyTorch tensors

class DynamicHedgingStrategy(BaseStrategy):

    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="DynamicHedgingStrategy",
            description="Dynamically hedges based on price changes relative to ATR.",
            default_params={'instrument_key': None, 'atr_period': 14, 'atr_multiplier_threshold': 2.0, 
                            'feature_indices': {'High': 1, 'Low': 2, 'Close': 3}}, # Assuming H,L,C are at these indices
            applicable_assets=[] # Should be set by 'instrument_key' or instance config
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.instrument_key = self.params.get('instrument_key') # Get from merged params
        
        if self.instrument_key is None and self.config.applicable_assets:
            self.instrument_key = self.config.applicable_assets[0]
            self.logger.info(f"{self.config.name}: 'instrument_key' not in params, using first from config.applicable_assets: {self.instrument_key}")
        
        if self.instrument_key is None:
            self.logger.error(f"{self.config.name} requires 'instrument_key' in params or config.applicable_assets to be non-empty.")
            # Strategy might not function correctly, but allow init to complete to avoid hard crash during layer setup.
            # Downstream methods (forward/generate_signals) should handle self.instrument_key being None.
            # raise ValueError(f"{self.config.name} requires 'instrument_key' in params or config.applicable_assets.")
        elif not self.config.applicable_assets: # If instrument_key is set, ensure it's in applicable_assets
            self.config.applicable_assets = [self.instrument_key]

        self.atr_period = self.params.get('atr_period', 14)
        self.atr_multiplier_threshold = self.params.get('atr_multiplier_threshold', 2.0)
        self.feature_indices = self.params.get('feature_indices', {'High': 1, 'Low': 2, 'Close': 3})


    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        # asset_features shape: (batch_size, sequence_length, num_features)
        # Assuming num_features are: High, Low, Close at specified indices

        batch_size, seq_len, _ = asset_features.shape
        signals_batch = torch.zeros(batch_size, seq_len, 1, device=asset_features.device)

        if seq_len == 0:
            return signals_batch

        high_idx = self.feature_indices['High']
        low_idx = self.feature_indices['Low']
        close_idx = self.feature_indices['Close']

        for i in range(batch_size):
            # Convert tensor slice to DataFrame for ta compatibility
            df = pd.DataFrame({
                'High': asset_features[i, :, high_idx].cpu().numpy(),
                'Low': asset_features[i, :, low_idx].cpu().numpy(),
                'Close': asset_features[i, :, close_idx].cpu().numpy()
            })

            if df.empty or not all(col in df.columns for col in ['High', 'Low', 'Close']):
                # If data is insufficient, signals remain 0 (neutral)
                continue
            
            # Handle cases where sequence length is less than ATR period
            if seq_len < self.atr_period:
                # Not enough data to calculate ATR, return neutral signals
                # signals_batch[i, :, 0] is already zeros
                continue

            df['atr'] = ta.volatility.average_true_range(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                window=self.atr_period,
                fillna=True # fillna=True in ta.average_true_range handles initial NaNs by backfilling
            )
            # Price change: positive if price drops (Close.shift(1) - Close)
            df['price_change'] = df['Close'].shift(1) - df['Close']
            
            # Ensure ATR is not zero to avoid division by zero; if ATR is zero, price_change_vs_atr is zero.
            df['price_change_vs_atr'] = np.where(df['atr'] != 0, (df['price_change'] / df['atr']), 0)
            df['price_change_vs_atr'] = df['price_change_vs_atr'].fillna(0) # Fill NaNs from price_change or initial ATR
            
            # Generate signals based on processed data
            # Original generate_signals logic:
            # signals['signal'] = np.where(processed_data['price_change_vs_atr'] > self.atr_multiplier_threshold, -1, 0)
            current_signals = np.where(df['price_change_vs_atr'] > self.atr_multiplier_threshold, -1.0, 0.0)
            signals_batch[i, :, 0] = torch.tensor(current_signals, device=asset_features.device, dtype=asset_features.dtype)
            
        return signals_batch

class RiskParityStrategy(BaseStrategy): # Simplified for single asset: volatility-based risk adjustment

    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="RiskParityStrategy",
            description="Adjusts risk based on asset volatility (simplified).",
            default_params={'instrument_key': None, 'vol_window': 20, 
                            'high_vol_threshold_pct': 0.02, 'low_vol_threshold_pct': 0.005,
                            'feature_indices': {'Close': 3}}, # Assuming Close is at index 3
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

        self.vol_window = self.params.get('vol_window', 20)
        self.high_vol_threshold_pct = self.params.get('high_vol_threshold_pct', 0.02)
        self.low_vol_threshold_pct = self.params.get('low_vol_threshold_pct', 0.005)
        self.feature_indices = self.params.get('feature_indices', {'Close': 3})

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        # asset_features shape: (batch_size, sequence_length, num_features)
        batch_size, seq_len, _ = asset_features.shape
        signals_batch = torch.zeros(batch_size, seq_len, 1, device=asset_features.device)
        
        if seq_len == 0:
            return signals_batch
            
        close_idx = self.feature_indices['Close']

        for i in range(batch_size):
            df = pd.DataFrame({'Close': asset_features[i, :, close_idx].cpu().numpy()})

            if df.empty or 'Close' not in df.columns:
                continue

            df['returns'] = df['Close'].pct_change()
            # Calculate volatility only when enough data is available (min_periods = window)
            # NaN will be present for initial periods where window is not full.
            df['volatility'] = df['returns'].rolling(window=self.vol_window, min_periods=self.vol_window).std()
            
            current_signals_np = np.zeros(seq_len)
            # Conditions should only apply where volatility is not NaN
            high_vol_condition = pd.notna(df['volatility']) & (df['volatility'] > self.high_vol_threshold_pct)
            low_vol_condition = pd.notna(df['volatility']) & (df['volatility'] < self.low_vol_threshold_pct) 
            # Ensure that low_vol_condition also checks that volatility is not zero, if zero is not a valid low vol signal.
            # However, if std can be legitimately zero (e.g. flat price), it might be considered very low volatility.
            # The current thresholds should handle this. If low_vol_threshold_pct is > 0, then 0 vol will trigger it.
            
            current_signals_np[high_vol_condition] = -1.0 # Reduce risk
            current_signals_np[low_vol_condition] = 1.0  # Increase risk appetite
            
            signals_batch[i, :, 0] = torch.tensor(current_signals_np, device=asset_features.device, dtype=asset_features.dtype)
            
        return signals_batch

class VaRControlStrategy(BaseStrategy):

    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="VaRControlStrategy",
            description="Controls risk based on Value at Risk (VaR) estimates.",
            default_params={'instrument_key': None, 'var_window': 20, 
                            'var_confidence': 0.99, 'var_limit': 0.02,
                            'feature_indices': {'Close': 3}}, # Assuming Close is at index 3
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
            
        self.var_window = self.params.get('var_window', 20)
        self.var_limit = self.params.get('var_limit', 0.02)
        self.feature_indices = self.params.get('feature_indices', {'Close': 3})
        self.z_score = norm.ppf(self.params.get('var_confidence', 0.99))


    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, _ = asset_features.shape
        signals_batch = torch.zeros(batch_size, seq_len, 1, device=asset_features.device)

        if seq_len == 0:
            return signals_batch
            
        close_idx = self.feature_indices['Close']

        for i in range(batch_size):
            df = pd.DataFrame({'Close': asset_features[i, :, close_idx].cpu().numpy()})

            if df.empty or 'Close' not in df.columns:
                continue

            df['returns'] = df['Close'].pct_change()
            # Calculate rolling_std only when enough data is available (min_periods = window)
            df['rolling_std'] = df['returns'].rolling(window=self.var_window, min_periods=self.var_window).std()
            df['estimated_var'] = df['rolling_std'] * self.z_score # Parametric VaR
            
            current_signals_np = np.zeros(seq_len)
            # Condition should only apply where estimated_var is not NaN
            condition = pd.notna(df['estimated_var']) & (df['estimated_var'] > self.var_limit)
            current_signals_np[condition] = -1.0 # -1 to reduce risk
            signals_batch[i, :, 0] = torch.tensor(current_signals_np, device=asset_features.device, dtype=asset_features.dtype)
            
        return signals_batch

class MaxDrawdownControlStrategy(BaseStrategy):

    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="MaxDrawdownControlStrategy",
            description="Controls risk by monitoring and reacting to maximum drawdown.",
            default_params={'instrument_key': None, 'max_drawdown_limit': 0.10,
                            'feature_indices': {'Close': 3}}, # Assuming Close is at index 3
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
            
        self.max_drawdown_limit = self.params.get('max_drawdown_limit', 0.10)
        self.feature_indices = self.params.get('feature_indices', {'Close': 3})
        # self.hwm_scalar is instance specific and needs careful handling with batched tensor inputs.
        # For simplicity in this refactor, HWM will be calculated per batch item independently.
        # A truly persistent HWM across batches for a learnable module would require state management
        # outside the forward pass or as a non-differentiable buffer.
        # For now, this strategy is likely not part of gradient-based learning.
        self.hwm_per_batch_item = {} # Stores hwm for each item in batch if we need persistence across calls for same item

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, _ = asset_features.shape
        signals_batch = torch.zeros(batch_size, seq_len, 1, device=asset_features.device)
        close_idx = self.feature_indices['Close']

        for i in range(batch_size):
            # For this strategy, HWM should ideally be persistent.
            # The self.hwm_scalar was instance-level.
            # With batches, we'd need one HWM per item in the batch if sequences are independent.
            # Or a single HWM if the batch represents a single time series processed in segments.
            # Assuming independent sequences in a batch for now.
            # This means hwm is reset for each call to forward for each batch item, which is not ideal for true drawdown.
            # A proper fix would involve managing state more carefully.
            # For now, we calculate HWM within each sequence in the batch.
            
            current_batch_hwm_val = -float('inf') # Reset for each item in batch for this simplified version

            close_prices_np = asset_features[i, :, close_idx].cpu().numpy()
            df = pd.DataFrame({'Close': close_prices_np})

            if df.empty or 'Close' not in df.columns:
                continue
            
            hwm_series = []
            for price in df['Close']:
                if pd.notna(price):
                    current_batch_hwm_val = max(current_batch_hwm_val, price)
                hwm_series.append(current_batch_hwm_val if current_batch_hwm_val != -float('inf') else np.nan)
            
            df['high_water_mark'] = hwm_series
            
            df['drawdown'] = np.where(
                (pd.notna(df['high_water_mark'])) & (df['high_water_mark'] > 0),
                (df['high_water_mark'] - df['Close']) / df['high_water_mark'],
                0.0 
            )
            df['drawdown'] = df['drawdown'].fillna(0.0) # Modified line to address FutureWarning
            
            # Original generate_signals logic:
            condition = df['drawdown'] > self.max_drawdown_limit
            current_signals_np = np.where(condition, -1.0, 0.0) # -1 to reduce risk
            signals_batch[i, :, 0] = torch.tensor(current_signals_np, device=asset_features.device, dtype=asset_features.dtype)
            
        return signals_batch
