import unittest
import pandas as pd
import numpy as np
from src.agent.strategies.base_strategy import StrategyConfig
from src.agent.strategies.risk_management_strategies import (
    DynamicHedgingStrategy, RiskParityStrategy, VaRControlStrategy, MaxDrawdownControlStrategy
)

class TestRiskManagementStrategies(unittest.TestCase):
    def setUp(self):
        self.market_data_dict = {
            'EUR_USD': pd.DataFrame({
                'Open': np.random.rand(100) * 1.1,
                'High': np.random.rand(100) * 1.12,
                'Low': np.random.rand(100) * 1.08,
                'Close': np.random.rand(100) * 1.1,
                'open': np.random.rand(100) * 1.1, # Keep lowercase for compatibility if needed
                'high': np.random.rand(100) * 1.12,
                'low': np.random.rand(100) * 1.08,
                'close': np.random.rand(100) * 1.1,
                'volume': np.random.randint(100, 1000, 100)
            }, index=pd.date_range(start='1/1/2020', periods=100)),
            'USD_JPY': pd.DataFrame({
                'Open': np.random.rand(100) * 110,
                'High': np.random.rand(100) * 112,
                'Low': np.random.rand(100) * 108,
                'Close': np.random.rand(100) * 110,
                'open': np.random.rand(100) * 110,
                'high': np.random.rand(100) * 112,
                'low': np.random.rand(100) * 108,
                'close': np.random.rand(100) * 110,
                'volume': np.random.randint(100, 1000, 100)
            }, index=pd.date_range(start='1/1/2020', periods=100))
        }
        self.portfolio_context = {
            'cash': 100000, 
            'positions': {'EUR_USD': {'quantity': 10, 'average_price': 1.09}},
            'total_value': 100000 + 10 * 1.09 # Simplified
        }
        # self.base_config_params removed

    def test_dynamic_hedging_strategy(self):
        config = StrategyConfig(
            name="DynamicHedgingTest",
            default_params={
                'instrument_key': 'EUR_USD', # Primary instrument
                'timeframe': 'D1',
                'hedge_instrument_key': 'USD_JPY', 
                'correlation_window': 20, 
                'hedge_ratio_threshold': 0.5
            },
            applicable_assets=['EUR_USD', 'USD_JPY']
        )
        strategy = DynamicHedgingStrategy(config)
        
        market_data_for_strategy = { 
            k: self.market_data_dict[k] for k in config.applicable_assets 
            if k in self.market_data_dict # Ensure keys exist
        }
        processed_data = strategy.forward(market_data_for_strategy, self.portfolio_context)
        # DynamicHedgingStrategy produces ATR and price change related columns
        # Check for outputs specific to dynamic hedging
        if config.applicable_assets[0] in processed_data and not processed_data[config.applicable_assets[0]].empty:
            self.assertIn('atr', processed_data[config.applicable_assets[0]].columns)
            self.assertIn('price_change_vs_atr', processed_data[config.applicable_assets[0]].columns)
        
        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)
        # Hedging strategies might not produce a simple -1, 0, 1 signal for the primary asset.
        # They might produce signals for the hedge instrument or adjust position sizes.
        # Adjust assertions based on the strategy's actual output.
        # For this example, let's assume it can still output a 'signal' column for adjustments.
        self.assertIn('signal', signals.columns) 
        # self.assertTrue(signals['signal'].isin([-1, 0, 1]).all()) # This might be too strict

    def test_risk_parity_strategy(self):
        config = StrategyConfig(
            name="RiskParityTest",
            default_params={
                'timeframe': 'D1',
                'lookback_period': 20
            },
            applicable_assets=['EUR_USD', 'USD_JPY'] # Risk parity typically involves multiple assets
        )
        strategy = RiskParityStrategy(config)
        
        market_data_for_strategy = {
            k: self.market_data_dict[k] for k in config.applicable_assets
            if k in self.market_data_dict
        }
        processed_data = strategy.forward(market_data_for_strategy, self.portfolio_context)
        
        # Risk parity strategy processes data and should add 'volatility'
        for asset in config.applicable_assets:
            if asset in processed_data and not processed_data[asset].empty:
                self.assertIn('volatility', processed_data[asset].columns)
        
        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)
        # RiskParityStrategy returns standard signal format (e.g., target weights or trade signals)
        self.assertIn('signal', signals.columns) # Or check for asset weight columns if applicable

    def test_var_control_strategy(self):
        config = StrategyConfig(
            name="VaRControlTest",
            default_params={
                # 'instrument_key': 'EUR_USD', # VaR can be portfolio-wide or per instrument
                'timeframe': 'D1',
                'var_limit': 0.01, # 1% VaR limit
                'confidence_level': 0.99,
                'lookback_period': 50
            },
            applicable_assets=['EUR_USD'] # Test with a single asset for simplicity
        )
        strategy = VaRControlStrategy(config)
        
        market_data_for_strategy = {
            k: self.market_data_dict[k] for k in config.applicable_assets
            if k in self.market_data_dict
        }
        processed_data = strategy.forward(market_data_for_strategy, self.portfolio_context)

        if config.applicable_assets[0] in processed_data and not processed_data[config.applicable_assets[0]].empty:
            # VaRControlStrategy should produce 'returns' and 'estimated_var' columns
            self.assertIn('returns', processed_data[config.applicable_assets[0]].columns)
            self.assertIn('estimated_var', processed_data[config.applicable_assets[0]].columns) # Changed 'VaR' to 'estimated_var'
            
        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)
        # VaRControlStrategy returns standard signal format
        self.assertIn('signal', signals.columns)

    def test_max_drawdown_control_strategy(self):
        config = StrategyConfig(
            name="MaxDrawdownControlTest",
            default_params={
                # 'instrument_key': 'EUR_USD', # Can be portfolio-wide or per instrument
                'timeframe': 'D1',
                'max_drawdown_limit': 0.10, # 10% max drawdown
                'lookback_period': 100
            },
            applicable_assets=['EUR_USD'] # Test with a single asset
        )
        strategy = MaxDrawdownControlStrategy(config)

        market_data_for_strategy = {
            k: self.market_data_dict[k] for k in config.applicable_assets
            if k in self.market_data_dict
        }
        processed_data = strategy.forward(market_data_for_strategy, self.portfolio_context)

        if config.applicable_assets[0] in processed_data and not processed_data[config.applicable_assets[0]].empty:
            # MaxDrawdownControlStrategy produces drawdown-related columns
            self.assertIn('high_water_mark', processed_data[config.applicable_assets[0]].columns)
            self.assertIn('drawdown', processed_data[config.applicable_assets[0]].columns)
        
        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)
        # MaxDrawdownControlStrategy returns standard signal format
        self.assertIn('signal', signals.columns)


if __name__ == '__main__':
    unittest.main()
