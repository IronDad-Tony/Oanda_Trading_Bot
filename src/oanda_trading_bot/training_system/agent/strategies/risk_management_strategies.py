# src/agent/strategies/risk_management_strategies.py
from .base_strategy import BaseStrategy, StrategyConfig
import pandas as pd
import numpy as np
import ta
from scipy.stats import norm
from typing import Dict, Optional, List, Any
import logging
import torch
import torch.nn.functional as F # For rolling operations if needed

class DynamicHedgingStrategy(BaseStrategy):

    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="DynamicHedgingStrategy",
            description="Dynamically hedges based on price changes relative to ATR.",
            default_params={
                'atr_period': 14, 
                'atr_multiplier_threshold': 2.0, 
                'high_idx': 1, # Default index for High prices in input_dim features
                'low_idx': 2,  # Default index for Low prices
                'close_idx': 3 # Default index for Close prices
            },
            applicable_assets=[] # This strategy is typically single-asset
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.atr_period = self.params.get('atr_period', 14)
        self.atr_multiplier_threshold = self.params.get('atr_multiplier_threshold', 2.0)
        self.high_idx = self.params.get('high_idx', 1)
        self.low_idx = self.params.get('low_idx', 2)
        self.close_idx = self.params.get('close_idx', 3)

        if self.config.input_dim is not None:
            max_idx = max(self.high_idx, self.low_idx, self.close_idx)
            if max_idx >= self.config.input_dim:
                self.logger.error(f"[{self.config.name}] Feature indices ({self.high_idx}, {self.low_idx}, {self.close_idx}) are out of bounds for input_dim {self.config.input_dim}. Strategy may fail.")
        else:
            self.logger.warning(f"[{self.config.name}] input_dim not specified in config. Feature index validation skipped.")

    def _calculate_atr(self, high: torch.Tensor, low: torch.Tensor, close: torch.Tensor, window: int) -> torch.Tensor:
        # high, low, close are (batch_size, seq_len)
        # True Range calculation
        tr1 = high - low
        tr2 = torch.abs(high - torch.cat((close[:, :1], close[:, :-1]), dim=1)) # high - close_prev
        tr3 = torch.abs(low - torch.cat((close[:, :1], close[:, :-1]), dim=1))  # low - close_prev
        
        true_range = torch.max(torch.max(tr1, tr2), tr3) # (batch_size, seq_len)
        
        # ATR is Wilder's Smoothing of True Range
        # Using simple moving average for ATR here for simplicity with conv1d
        # A more accurate Wilder's MA would use EMA-like calculation.
        if true_range.shape[1] < window:
            padding = window - true_range.shape[1]
            padded_tr = F.pad(true_range.unsqueeze(1), (padding,0), mode='replicate')
        else:
            padding = window -1
            padded_tr = F.pad(true_range.unsqueeze(1), (padding,0), mode='replicate') # (B, 1, S_padded)

        sma_weights = torch.full((1, 1, window), 1.0/window, device=true_range.device, dtype=true_range.dtype)
        atr = F.conv1d(padded_tr, sma_weights, stride=1).squeeze(1) # (B, S)
        return atr

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, num_features = asset_features.shape
        device = asset_features.device

        if self.config.input_dim and max(self.high_idx, self.low_idx, self.close_idx) >= num_features:
            self.logger.error(f"[{self.config.name}] Feature indices out of bounds for actual features {num_features}. Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)
        
        if seq_len < self.atr_period or seq_len == 0:
            self.logger.warning(f"[{self.config.name}] Sequence length ({seq_len}) insufficient for ATR period ({self.atr_period}). Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        high_prices = asset_features[:, :, self.high_idx]
        low_prices = asset_features[:, :, self.low_idx]
        close_prices = asset_features[:, :, self.close_idx]

        atr = self._calculate_atr(high_prices, low_prices, close_prices, self.atr_period)

        # Price change: positive if price drops (Close_prev - Close_current)
        price_change = torch.cat((close_prices[:, :1], close_prices[:, :-1]), dim=1) - close_prices
        price_change[:, 0] = 0 # No change for the first element

        # Avoid division by zero for ATR
        safe_atr = atr.clamp(min=1e-9)
        price_change_vs_atr = price_change / safe_atr
        
        # Signal based on the last time step
        last_price_change_vs_atr = price_change_vs_atr[:, -1]
        
        signal = torch.zeros(batch_size, device=device)
        # Original logic: signal = -1 if price_change_vs_atr > threshold (i.e., large drop)
        signal[last_price_change_vs_atr > self.atr_multiplier_threshold] = -1.0
            
        return signal.view(batch_size, 1, 1)

class RiskParityStrategy(BaseStrategy):
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="RiskParityStrategy",
            description="Adjusts risk based on asset volatility (simplified).",
            default_params={
                'vol_window': 20, 
                'high_vol_threshold_pct': 0.02, 
                'low_vol_threshold_pct': 0.005,
                'close_idx': 3 # Default index for Close prices
            },
            applicable_assets=[]
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.vol_window = self.params.get('vol_window', 20)
        self.high_vol_threshold_pct = self.params.get('high_vol_threshold_pct', 0.02)
        self.low_vol_threshold_pct = self.params.get('low_vol_threshold_pct', 0.005)
        self.close_idx = self.params.get('close_idx', 3)

        if self.config.input_dim is not None and self.close_idx >= self.config.input_dim:
            self.logger.error(f"[{self.config.name}] Close index {self.close_idx} is out of bounds for input_dim {self.config.input_dim}.")
        elif self.config.input_dim is None:
            self.logger.warning(f"[{self.config.name}] input_dim not specified in config. Feature index validation skipped.")

    def _rolling_std(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        # tensor is (batch, seq_len)
        if tensor.shape[1] < window_size:
            padding = window_size - tensor.shape[1]
            tensor_padded_for_mean = F.pad(tensor.unsqueeze(1), (padding,0), mode='replicate')
        else:
            padding = window_size -1
            tensor_padded_for_mean = F.pad(tensor.unsqueeze(1), (padding,0), mode='replicate')
        
        sma_weights = torch.full((1, 1, window_size), 1.0/window_size, device=tensor.device, dtype=tensor.dtype)
        mean_x = F.conv1d(tensor_padded_for_mean, sma_weights, stride=1).squeeze(1)
        
        # For mean_x_sq, input tensor is tensor**2
        tensor_sq = tensor**2
        if tensor_sq.shape[1] < window_size:
            padding_sq = window_size - tensor_sq.shape[1]
            tensor_sq_padded = F.pad(tensor_sq.unsqueeze(1), (padding_sq,0), mode='replicate')
        else:
            padding_sq = window_size -1
            tensor_sq_padded = F.pad(tensor_sq.unsqueeze(1), (padding_sq,0), mode='replicate')

        mean_x_sq = F.conv1d(tensor_sq_padded, sma_weights, stride=1).squeeze(1)
        
        variance = (mean_x_sq - mean_x**2).clamp(min=1e-9)
        return torch.sqrt(variance)

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, num_features = asset_features.shape
        device = asset_features.device

        if self.config.input_dim and self.close_idx >= num_features:
            self.logger.error(f"[{self.config.name}] Close index {self.close_idx} OOB for actual features {num_features}. Zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        if seq_len < self.vol_window or seq_len == 0:
            self.logger.warning(f"[{self.config.name}] Seq length ({seq_len}) < vol_window ({self.vol_window}). Zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        close_prices = asset_features[:, :, self.close_idx]
        returns = torch.zeros_like(close_prices)
        # pct_change: (current - previous) / previous
        returns[:, 1:] = (close_prices[:, 1:] - close_prices[:, :-1]) / close_prices[:, :-1].clamp(min=1e-9)
        returns[:, 0] = 0 # No return for the first element

        volatility = self._rolling_std(returns, self.vol_window)
        last_volatility = volatility[:, -1]
        
        signal = torch.zeros(batch_size, device=device)
        signal[last_volatility > self.high_vol_threshold_pct] = -1.0 # Reduce risk
        signal[last_volatility < self.low_vol_threshold_pct] = 1.0  # Increase risk appetite
            
        return signal.view(batch_size, 1, 1)

class VaRControlStrategy(BaseStrategy):
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="VaRControlStrategy",
            description="Controls risk based on Value at Risk (VaR) estimates.",
            default_params={
                'var_window': 20, 
                'var_confidence': 0.99, 
                'var_limit': 0.02,
                'close_idx': 3 # Default index for Close prices
            },
            applicable_assets=[]
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.var_window = self.params.get('var_window', 20)
        self.var_limit = self.params.get('var_limit', 0.02)
        self.close_idx = self.params.get('close_idx', 3)
        self.z_score_val = norm.ppf(self.params.get('var_confidence', 0.99))

        if self.config.input_dim is not None and self.close_idx >= self.config.input_dim:
            self.logger.error(f"[{self.config.name}] Close index {self.close_idx} OOB for input_dim {self.config.input_dim}.")
        elif self.config.input_dim is None:
             self.logger.warning(f"[{self.config.name}] input_dim not specified. Index validation skipped.")

    _rolling_std = RiskParityStrategy._rolling_std # Reuse from RiskParityStrategy

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, num_features = asset_features.shape
        device = asset_features.device

        if self.config.input_dim and self.close_idx >= num_features:
            self.logger.error(f"[{self.config.name}] Close index {self.close_idx} OOB for actual features {num_features}. Zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        if seq_len < self.var_window or seq_len == 0:
            self.logger.warning(f"[{self.config.name}] Seq length ({seq_len}) < var_window ({self.var_window}). Zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        close_prices = asset_features[:, :, self.close_idx]
        returns = torch.zeros_like(close_prices)
        returns[:, 1:] = (close_prices[:, 1:] - close_prices[:, :-1]) / close_prices[:, :-1].clamp(min=1e-9)
        returns[:, 0] = 0

        rolling_std_dev = self._rolling_std(self, returns, self.var_window) # Pass self for logger context
        estimated_var = rolling_std_dev[:, -1] * self.z_score_val
        
        signal = torch.zeros(batch_size, device=device)
        signal[estimated_var > self.var_limit] = -1.0 # Reduce risk
            
        return signal.view(batch_size, 1, 1)

class MaxDrawdownControlStrategy(BaseStrategy):
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="MaxDrawdownControlStrategy",
            description="Controls risk by monitoring and reacting to maximum drawdown.",
            default_params={
                'max_drawdown_limit': 0.10,
                'close_idx': 3 # Default index for Close prices
            },
            applicable_assets=[]
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.max_drawdown_limit = self.params.get('max_drawdown_limit', 0.10)
        self.close_idx = self.params.get('close_idx', 3)
        # HWM state is tricky with stateless forward passes. 
        # This strategy, if stateful HWM is needed across calls, requires external state management or becomes non-batchable easily.
        # For a batch, HWM is computed per item over its sequence length.
        if self.config.input_dim is not None and self.close_idx >= self.config.input_dim:
            self.logger.error(f"[{self.config.name}] Close index {self.close_idx} OOB for input_dim {self.config.input_dim}.")
        elif self.config.input_dim is None:
             self.logger.warning(f"[{self.config.name}] input_dim not specified. Index validation skipped.")

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, seq_len, num_features = asset_features.shape
        device = asset_features.device

        if self.config.input_dim and self.close_idx >= num_features:
            self.logger.error(f"[{self.config.name}] Close index {self.close_idx} OOB for actual features {num_features}. Zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        if seq_len == 0:
            return torch.zeros((batch_size, 1, 1), device=device)

        close_prices = asset_features[:, :, self.close_idx] # (batch_size, seq_len)

        # Calculate High Water Mark (HWM) and drawdown for each item in the batch
        # HWM is computed as cumulative max over the sequence for each batch item.
        hwm = torch.cummax(close_prices, dim=1).values # (batch_size, seq_len)
        
        # Drawdown = (HWM - Close) / HWM
        # Clamp HWM to avoid division by zero if HWM is 0 or negative (though prices are usually positive)
        drawdown = (hwm - close_prices) / hwm.clamp(min=1e-9)
        drawdown[hwm <= 1e-9] = 0 # If HWM is effectively zero, drawdown is zero
        
        last_drawdown = drawdown[:, -1] # (batch_size)
        
        signal = torch.zeros(batch_size, device=device)
        signal[last_drawdown > self.max_drawdown_limit] = -1.0 # Reduce risk
            
        return signal.view(batch_size, 1, 1)

# Ensure all strategies are registered if __init__.py uses a discovery mechanism
# For explicit registration (if needed, though __init__.py should handle it):
# from ..strategies import STRATEGY_REGISTRY
# STRATEGY_REGISTRY[DynamicHedgingStrategy.default_config().name] = DynamicHedgingStrategy
# STRATEGY_REGISTRY[RiskParityStrategy.default_config().name] = RiskParityStrategy
# STRATEGY_REGISTRY[VaRControlStrategy.default_config().name] = VaRControlStrategy
# STRATEGY_REGISTRY[MaxDrawdownControlStrategy.default_config().name] = MaxDrawdownControlStrategy
