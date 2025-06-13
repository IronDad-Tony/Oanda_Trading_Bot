# filepath: c:\Users\tonyh\Oanda_Trading_Bot\tests\unit_tests\test_risk_management_strategies.py
import pytest
import torch
import pandas as pd
import numpy as np
import logging
from src.agent.strategies.base_strategy import StrategyConfig
from src.agent.strategies.risk_management_strategies import (
    MaxDrawdownControlStrategy, # Corrected class name
    DynamicHedgingStrategy,
    RiskParityStrategy,
    VaRControlStrategy
)

# Mock logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_mock_config(name: str, params: dict = None, applicable_assets: list = None) -> StrategyConfig:
    default_params = {
        'instrument_key': 'EUR_USD', # Default instrument for testing
        'feature_indices': {'Open': 0, 'High': 1, 'Low': 2, 'Close': 3, 'Volume': 4} # Example indices
    }
    if params:
        default_params.update(params)
    
    return StrategyConfig(
        name=name,
        description=f"Test config for {name}",
        default_params=default_params,
        applicable_assets=applicable_assets if applicable_assets is not None else ['EUR_USD']
    )

class TestMaximumDrawdownControlStrategy:
    def test_initialization(self):
        config = MaxDrawdownControlStrategy.default_config() # Corrected class name
        strategy = MaxDrawdownControlStrategy(config, logger=logger) # Corrected class name
        assert strategy.config.name == "MaxDrawdownControlStrategy"
        assert strategy.params['max_drawdown_limit'] == 0.10
        assert strategy.instrument_key is None # Default config has no instrument_key

        custom_params = {'max_drawdown_limit': 0.05, 'instrument_key': 'USD_JPY', 'feature_indices': {'Close': 0}}
        config_custom = get_mock_config("MaxDrawdownControlStrategyCustom", params=custom_params)
        strategy_custom = MaxDrawdownControlStrategy(config_custom, params=custom_params, logger=logger) # Corrected class name
        assert strategy_custom.params['max_drawdown_limit'] == 0.05
        assert strategy_custom.instrument_key == 'USD_JPY'
        assert strategy_custom.feature_indices == {'Close': 0}

    def test_forward_no_drawdown(self):
        params = {'max_drawdown_limit': 0.1, 'instrument_key': 'EUR_USD', 'feature_indices': {'Close': 0}}
        config = get_mock_config("MaxDrawdownControlStrategy", params=params)
        strategy = MaxDrawdownControlStrategy(config, params=params, logger=logger) # Corrected class name
        
        # Prices: 100, 101, 102, 103, 104 (no drawdown)
        asset_features = torch.tensor([[[100.0], [101.0], [102.0], [103.0], [104.0]]], dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features)
        
        assert signals.shape == (1, 5, 1)
        assert torch.all(signals == 0.0).item() # No risk reduction signal

    def test_forward_with_drawdown(self):
        params = {'max_drawdown_limit': 0.1, 'instrument_key': 'EUR_USD', 'feature_indices': {'Close': 0}}
        config = get_mock_config("MaxDrawdownControlStrategy", params=params)
        strategy = MaxDrawdownControlStrategy(config, params=params, logger=logger) # Corrected class name
        
        # Prices: 100, 105, 90 (drawdown > 10% from 105), 92, 85 (drawdown > 10% from 105)
        # HWM:    100, 105, 105, 105, 105
        # Drawdown: 0,   0, (105-90)/105=0.14, (105-92)/105=0.12, (105-85)/105=0.19
        asset_features = torch.tensor([[[100.0], [105.0], [90.0], [92.0], [85.0]]], dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features)
        
        expected_signals = torch.tensor([[[0.0], [0.0], [-1.0], [-1.0], [-1.0]]], dtype=torch.float32).to(DEVICE)
        assert signals.shape == (1, 5, 1)
        assert torch.allclose(signals, expected_signals)

    def test_forward_increasing_prices(self):
        params = {'max_drawdown_limit': 0.05, 'instrument_key': 'EUR_USD', 'feature_indices': {'Close': 0}}
        config = get_mock_config("MaxDrawdownControlStrategy", params=params)
        strategy = MaxDrawdownControlStrategy(config, params=params, logger=logger) # Corrected class name
        
        asset_features = torch.tensor([[[10.0], [11.0], [12.0], [13.0], [14.0]]], dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features)
        assert torch.all(signals == 0.0).item()

    def test_forward_with_nan_prices(self):
        params = {'max_drawdown_limit': 0.1, 'instrument_key': 'EUR_USD', 'feature_indices': {'Close': 0}}
        config = get_mock_config("MaxDrawdownControlStrategy", params=params)
        strategy = MaxDrawdownControlStrategy(config, params=params, logger=logger) # Corrected class name
        
        # Prices: 100, NaN, 90 (HWM should be 100, drawdown (100-90)/100 = 0.1, not > 0.1)
        # HWM: 100, 100 (from last valid), 100
        # Drawdown: 0, 0 (fillna), (100-90)/100 = 0.1
        asset_features_nan = torch.tensor([[[100.0], [float('nan')], [90.0]]], dtype=torch.float32).to(DEVICE)
        signals_nan = strategy.forward(asset_features_nan)
        expected_signals_nan = torch.tensor([[[0.0], [0.0], [0.0]]], dtype=torch.float32).to(DEVICE) # Drawdown is 0.1, not > 0.1
        assert torch.allclose(signals_nan, expected_signals_nan)

        # Prices: 100, 105, NaN, 80 (HWM 105, drawdown (105-80)/105 = 0.23 > 0.1)
        # HWM: 100, 105, 105, 105
        # Drawdown: 0, 0, 0 (fillna), (105-80)/105 = 0.238
        asset_features_nan_2 = torch.tensor([[[100.0], [105.0], [float('nan')], [80.0]]], dtype=torch.float32).to(DEVICE)
        signals_nan_2 = strategy.forward(asset_features_nan_2)
        expected_signals_nan_2 = torch.tensor([[[0.0], [0.0], [0.0], [-1.0]]], dtype=torch.float32).to(DEVICE)
        assert torch.allclose(signals_nan_2, expected_signals_nan_2)

    def test_forward_empty_input(self):
        params = {'max_drawdown_limit': 0.1, 'instrument_key': 'EUR_USD', 'feature_indices': {'Close': 0}}
        config = get_mock_config("MaxDrawdownControlStrategy", params=params)
        strategy = MaxDrawdownControlStrategy(config, params=params, logger=logger) # Corrected class name
        asset_features_empty = torch.empty((1, 0, 1), dtype=torch.float32).to(DEVICE) # seq_len is 0
        signals = strategy.forward(asset_features_empty)
        assert signals.shape == (1, 0, 1)

        asset_features_empty_batch = torch.empty((0, 5, 1), dtype=torch.float32).to(DEVICE) # batch_size is 0
        signals_batch = strategy.forward(asset_features_empty_batch)
        assert signals_batch.shape == (0, 5, 1)

class TestDynamicHedgingStrategy:
    def test_initialization(self):
        config = DynamicHedgingStrategy.default_config()
        strategy = DynamicHedgingStrategy(config, logger=logger)
        assert strategy.config.name == "DynamicHedgingStrategy"
        assert strategy.params['atr_period'] == 14
        assert strategy.instrument_key is None

        custom_params = {'atr_period': 10, 'atr_multiplier_threshold': 1.5, 'instrument_key': 'GBP_USD', 
                         'feature_indices': {'High': 0, 'Low': 1, 'Close': 2}}
        config_custom = get_mock_config("DynamicHedgingStrategyCustom", params=custom_params)
        strategy_custom = DynamicHedgingStrategy(config_custom, params=custom_params, logger=logger)
        assert strategy_custom.params['atr_period'] == 10
        assert strategy_custom.params['atr_multiplier_threshold'] == 1.5
        assert strategy_custom.instrument_key == 'GBP_USD'
        assert strategy_custom.feature_indices == {'High': 0, 'Low': 1, 'Close': 2}

    def test_forward_no_hedge(self):
        params = {'atr_period': 3, 'atr_multiplier_threshold': 2.0, 'instrument_key': 'EUR_USD', 
                  'feature_indices': {'High': 0, 'Low': 1, 'Close': 2}}
        config = get_mock_config("DynamicHedgingStrategy", params=params)
        strategy = DynamicHedgingStrategy(config, params=params, logger=logger)
        
        # Data: High, Low, Close
        # Price change small relative to ATR
        asset_features = torch.tensor([[
            [1.10, 1.00, 1.05],  # ATR will be NaN/0 initially
            [1.12, 1.02, 1.08],  
            [1.15, 1.05, 1.10],  # ATR calculated from here
            [1.16, 1.06, 1.11],  # Close.shift(1)-Close = 1.10-1.11 = -0.01. ATR ~0.08. Ratio small.
            [1.18, 1.08, 1.12]   # Close.shift(1)-Close = 1.11-1.12 = -0.01. Ratio small.
        ]], dtype=torch.float32).to(DEVICE)
        
        signals = strategy.forward(asset_features)
        assert signals.shape == (1, 5, 1)
        # ATR calculation needs a few periods, first signals might be 0 due to NaN ATR or small changes
        # Price change: positive if price drops (Close.shift(1) - Close)
        # df['price_change'] = df['Close'].shift(1) - df['Close']
        # signals['signal'] = np.where(processed_data['price_change_vs_atr'] > self.atr_multiplier_threshold, -1, 0)
        # Expected: 0, 0, 0, 0, 0 (assuming price_change_vs_atr does not exceed threshold)
        assert torch.all(signals == 0.0).item()


    def test_forward_hedge_triggered(self):
        params = {'atr_period': 3, 'atr_multiplier_threshold': 1.0, 'instrument_key': 'EUR_USD', 
                  'feature_indices': {'High': 0, 'Low': 1, 'Close': 2}}
        config = get_mock_config("DynamicHedgingStrategy", params=params)
        strategy = DynamicHedgingStrategy(config, params=params, logger=logger)

        # Data: High, Low, Close
        # Large price drop relative to ATR
        asset_features = torch.tensor([[
            [1.10, 1.08, 1.09], # t0
            [1.10, 1.08, 1.09], # t1
            [1.10, 1.08, 1.09], # t2, ATR will be small, e.g. ~0.01
            [1.10, 1.00, 1.00], # t3, Price drop: 1.09 - 1.00 = 0.09. ATR ~0.03 ((0.02+0.02+0.09)/3). 0.09/0.03 = 3 > 1.0
            [1.02, 0.98, 1.01]  # t4, Price rise: 1.00 - 1.01 = -0.01. No signal
        ]], dtype=torch.float32).to(DEVICE)
        
        signals = strategy.forward(asset_features)
        # Expected signals: 0, 0, 0, -1 (hedge), 0
        # ATR for t0,t1,t2 will be small or NaN initially.
        # For t3: H=1.1, L=1.0, C=1.0. PrevC=1.09. TRs: (1.1-1.08)=0.02, (1.1-1.08)=0.02, (1.1-1.08)=0.02. ATR(3) for C[2] is ~0.02
        # Price change at t3: C[2]-C[3] = 1.09 - 1.00 = 0.09.
        # ATR at t3 (using C[0,1,2] for ATR calc affecting C[2]'s ATR, then C[0,1,2,3] for C[3]'s ATR):
        # df for ATR: H[1.1,1.1,1.1,1.1], L[1.08,1.08,1.08,1.0], C[1.09,1.09,1.09,1.0]
        # ATRs: nan, nan, 0.02, (0.02*2 + (1.1-1.0))/3 = (0.04+0.1)/3 = 0.14/3 = 0.0466
        # price_change_vs_atr for t3: 0.09 / 0.0466 = 1.93 > 1.0. So signal -1.
        expected_signals = torch.tensor([[[0.0], [0.0], [0.0], [-1.0], [0.0]]], dtype=torch.float32).to(DEVICE)
        assert signals.shape == (1, 5, 1)
        assert torch.allclose(signals, expected_signals)

    def test_forward_empty_input(self):
        params = {'instrument_key': 'EUR_USD', 'feature_indices': {'High':0,'Low':1,'Close':2}}
        config = get_mock_config("DynamicHedgingStrategy", params=params)
        strategy = DynamicHedgingStrategy(config, params=params, logger=logger)
        asset_features_empty = torch.empty((1, 0, 3), dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features_empty)
        assert signals.shape == (1, 0, 1)

    def test_forward_insufficient_data_for_atr(self):
        params = {'atr_period': 5, 'instrument_key': 'EUR_USD', 'feature_indices': {'High':0,'Low':1,'Close':2}}
        config = get_mock_config("DynamicHedgingStrategy", params=params)
        strategy = DynamicHedgingStrategy(config, params=params, logger=logger)
        # Only 2 data points, less than atr_period
        asset_features = torch.tensor([[[1.1, 1.0, 1.05], [1.12, 1.02, 1.08]]], dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features)
        assert signals.shape == (1, 2, 1)
        assert torch.all(signals == 0.0).item() # ATR will be NaN, so price_change_vs_atr will be 0

class TestRiskParityStrategy:
    def test_initialization(self):
        config = RiskParityStrategy.default_config()
        strategy = RiskParityStrategy(config, logger=logger)
        assert strategy.config.name == "RiskParityStrategy"
        assert strategy.params['vol_window'] == 20
        assert strategy.instrument_key is None

        custom_params = {'vol_window': 10, 'high_vol_threshold_pct': 0.03, 'low_vol_threshold_pct': 0.008, 
                         'instrument_key': 'AUD_USD', 'feature_indices': {'Close': 0}}
        config_custom = get_mock_config("RiskParityStrategyCustom", params=custom_params)
        strategy_custom = RiskParityStrategy(config_custom, params=custom_params, logger=logger)
        assert strategy_custom.params['vol_window'] == 10
        assert strategy_custom.params['high_vol_threshold_pct'] == 0.03
        assert strategy_custom.instrument_key == 'AUD_USD'

    def test_forward_low_volatility(self):
        params = {'vol_window': 3, 'high_vol_threshold_pct': 0.02, 'low_vol_threshold_pct': 0.005, 
                  'instrument_key': 'EUR_USD', 'feature_indices': {'Close': 0}}
        config = get_mock_config("RiskParityStrategy", params=params)
        strategy = RiskParityStrategy(config, params=params, logger=logger)
        
        # Prices with low volatility
        # Returns: NaN, 0.001, 0.001, 0.001. StdDev of returns will be small.
        asset_features = torch.tensor([[[100.0], [100.1], [100.2], [100.3], [100.4]]], dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features)
        # Volatility calculation needs vol_window=3 periods for returns, then std of those.
        # df['returns'] = [NaN, 0.000999, 0.000998, 0.000997, 0.000996]
        # df['volatility'] (window=3, min_periods=3 for returns):
        #   - index 0, 1: NaN (not enough data for full window of returns for std calc)
        #   - index 2: std(returns[0,1,2]) = std(NaN, 0.000999, 0.000998) -> NaN (due to initial NaN in returns)
        #   - index 2 (corrected logic): std(returns[0..2]) -> df['volatility'][2] is based on returns[0], returns[1], returns[2]. Since returns[0] is NaN, vol[2] is NaN.
        #   - index 3: std(returns[1,2,3]) = std(0.000999, 0.000998, 0.000997) approx 0.000001. This is < 0.005. Signal = 1.
        #   - index 4: std(returns[2,3,4]) = std(0.000998, 0.000997, 0.000996) approx 0.000001. This is < 0.005. Signal = 1.
        # Expected signals: 0 (for NaN vol), 0 (for NaN vol), 0 (for NaN vol), 1, 1
        expected_signals = torch.tensor([[[0.0], [0.0], [0.0], [1.0], [1.0]]], dtype=torch.float32).to(DEVICE)
        assert signals.shape == (1, 5, 1)
        assert torch.allclose(signals, expected_signals)

    def test_forward_high_volatility(self):
        params = {'vol_window': 3, 'high_vol_threshold_pct': 0.02, 'low_vol_threshold_pct': 0.005, 
                  'instrument_key': 'EUR_USD', 'feature_indices': {'Close': 0}}
        config = get_mock_config("RiskParityStrategy", params=params)
        strategy = RiskParityStrategy(config, params=params, logger=logger)
        
        asset_features = torch.tensor([[[100.0], [105.0], [100.0], [105.0], [100.0]]], dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features)
        # df['returns'] = [NaN, 0.05, -0.0476, 0.05, -0.0476]
        # df['volatility'] (window=3, min_periods=3):
        #   - index 0, 1, 2: NaN
        #   - index 3: std(returns[1,2,3]) = std(0.05, -0.0476, 0.05) approx 0.056. This is > 0.02. Signal = -1.
        #   - index 4: std(returns[2,3,4]) = std(-0.0476, 0.05, -0.0476) approx 0.056. This is > 0.02. Signal = -1.
        # Expected: 0, 0, 0, -1, -1
        expected_signals = torch.tensor([[[0.0], [0.0], [0.0], [-1.0], [-1.0]]], dtype=torch.float32).to(DEVICE)
        assert signals.shape == (1, 5, 1)
        assert torch.allclose(signals, expected_signals)

    def test_forward_neutral_volatility(self):
        params = {'vol_window': 3, 'high_vol_threshold_pct': 0.02, 'low_vol_threshold_pct': 0.01, 
                  'instrument_key': 'EUR_USD', 'feature_indices': {'Close': 0}}
        config = get_mock_config("RiskParityStrategy", params=params)
        strategy = RiskParityStrategy(config, params=params, logger=logger)
        
        # Prices with volatility between low and high thresholds
        # Returns: NaN, 0.015, 0.0147, 0.0145
        asset_features = torch.tensor([[[100.0], [101.5], [103.0], [104.5], [106.0]]], dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features)
        # Returns: nan, 0.015, 0.01477, 0.01456, 0.01435
        # Rolling std(3): nan, nan, std([0.015, 0.01477, 0.01456]) ~0.0002. This is < low_vol_threshold (0.01)
        # Let's adjust prices for neutral:
        # Prices: 100, 101, 102, 100, 101. Returns: nan, 0.01, 0.0099, -0.0196, 0.01
        # std([0.01, 0.0099, -0.0196]) ~ 0.017. This is between 0.01 and 0.02.
        asset_features_neutral = torch.tensor([[[100.0], [101.0], [102.0], [100.0], [101.0]]], dtype=torch.float32).to(DEVICE)
        signals_neutral = strategy.forward(asset_features_neutral)
        expected_signals_neutral = torch.tensor([[[0.0], [0.0], [0.0], [0.0], [0.0]]], dtype=torch.float32).to(DEVICE)
        assert signals_neutral.shape == (1, 5, 1)
        assert torch.allclose(signals_neutral, expected_signals_neutral)


    def test_forward_empty_input(self):
        params = {'instrument_key': 'EUR_USD', 'feature_indices': {'Close':0}}
        config = get_mock_config("RiskParityStrategy", params=params)
        strategy = RiskParityStrategy(config, params=params, logger=logger)
        asset_features_empty = torch.empty((1, 0, 1), dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features_empty)
        assert signals.shape == (1, 0, 1)

class TestVaRControlStrategy:
    def test_initialization(self):
        config = VaRControlStrategy.default_config()
        strategy = VaRControlStrategy(config, logger=logger)
        assert strategy.config.name == "VaRControlStrategy"
        assert strategy.params['var_window'] == 20
        assert strategy.params['var_limit'] == 0.02
        assert strategy.instrument_key is None
        assert strategy.z_score == pytest.approx(2.3263, abs=1e-4) # for 0.99 confidence

        custom_params = {'var_window': 10, 'var_limit': 0.03, 'var_confidence': 0.95, 
                         'instrument_key': 'USD_CAD', 'feature_indices': {'Close': 0}}
        config_custom = get_mock_config("VaRControlStrategyCustom", params=custom_params)
        strategy_custom = VaRControlStrategy(config_custom, params=custom_params, logger=logger)
        assert strategy_custom.params['var_window'] == 10
        assert strategy_custom.params['var_limit'] == 0.03
        assert strategy_custom.instrument_key == 'USD_CAD'
        assert strategy_custom.z_score == pytest.approx(1.64485, abs=1e-4) # for 0.95 confidence

    def test_forward_var_below_limit(self):
        params = {'var_window': 3, 'var_limit': 0.05, 'var_confidence': 0.99, 
                  'instrument_key': 'EUR_USD', 'feature_indices': {'Close': 0}}
        config = get_mock_config("VaRControlStrategy", params=params)
        strategy = VaRControlStrategy(config, params=params, logger=logger)
        z_score = strategy.z_score # approx 2.3263
        
        # Prices with low volatility, so VaR should be low
        # Returns: NaN, 0.001, 0.001, 0.001
        asset_features = torch.tensor([[[100.0], [100.1], [100.2], [100.3], [100.4]]], dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features)
        # Returns: nan, 0.000999, 0.000998, 0.000997, 0.000996
        # Rolling std(3): nan, nan, std([0.000999, 0.000998, 0.000997]) ~0.000001
        # Estimated VaR = 0.000001 * 2.3263 = 0.0000023 < 0.05
        # Expected: 0, 0, 0, 0, 0
        expected_signals = torch.tensor([[[0.0], [0.0], [0.0], [0.0], [0.0]]], dtype=torch.float32).to(DEVICE)
        assert signals.shape == (1, 5, 1)
        assert torch.allclose(signals, expected_signals)

    def test_forward_var_exceeds_limit(self):
        params = {'var_window': 3, 'var_limit': 0.05, 'var_confidence': 0.99, 
                  'instrument_key': 'EUR_USD', 'feature_indices': {'Close': 0}}
        config = get_mock_config("VaRControlStrategy", params=params)
        strategy = VaRControlStrategy(config, params=params, logger=logger)
        z_score = strategy.z_score # approx 2.3263

        asset_features = torch.tensor([[[100.0], [103.0], [100.0], [103.0], [100.0]]], dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features)
        # df['returns'] = [NaN, 0.03, -0.02912, 0.03, -0.02912]
        # df['rolling_std'] (window=3, min_periods=3):
        #   - index 0, 1, 2: NaN
        #   - index 3: std(returns[1,2,3]) = std(0.03, -0.02912, 0.03) approx 0.034.
        #     Estimated VaR = 0.034 * 2.3263 = 0.079 > 0.05. Signal = -1.
        #   - index 4: std(returns[2,3,4]) = std(-0.02912, 0.03, -0.02912) approx 0.034.
        #     Estimated VaR = 0.034 * 2.3263 = 0.079 > 0.05. Signal = -1.
        # Expected: 0, 0, 0, -1, -1
        expected_signals = torch.tensor([[[0.0], [0.0], [0.0], [-1.0], [-1.0]]], dtype=torch.float32).to(DEVICE)
        assert signals.shape == (1, 5, 1)
        assert torch.allclose(signals, expected_signals)

    def test_forward_empty_input(self):
        params = {'instrument_key': 'EUR_USD', 'feature_indices': {'Close':0}}
        config = get_mock_config("VaRControlStrategy", params=params)
        strategy = VaRControlStrategy(config, params=params, logger=logger)
        asset_features_empty = torch.empty((1, 0, 1), dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features_empty)
        assert signals.shape == (1, 0, 1)

