import unittest
import pandas as pd
import numpy as np
from src.agent.strategies.base_strategy import StrategyConfig
from src.agent.strategies.trend_strategies import MomentumStrategy, BreakoutStrategy, TrendFollowingStrategy, ReversalStrategy

class TestTrendStrategies(unittest.TestCase):

    def setUp(self):
        # Sample market data for testing
        self.market_data_dict = {
            'EUR_USD': pd.DataFrame({
                'open': np.random.rand(100) * 1.1,
                'high': np.random.rand(100) * 1.12,
                'low': np.random.rand(100) * 1.08,
                'close': np.random.rand(100) * 1.1,
                'volume': np.random.randint(100, 1000, 100)
            }, index=pd.date_range(start='1/1/2020', periods=100))
        }
        self.portfolio_context = {'cash': 100000, 'positions': {}}

    def test_momentum_strategy(self):
        config = StrategyConfig(
            name="MomentumTest",
            default_params={'momentum_window': 10, 'instrument_key': 'EUR_USD', 'timeframe': 'D1'},
            applicable_assets=['EUR_USD']
        )
        strategy = MomentumStrategy(config)
        
        processed_data = strategy.forward(self.market_data_dict, self.portfolio_context)
        self.assertIn('EUR_USD', processed_data)
        self.assertIn('momentum', processed_data['EUR_USD'].columns)
        
        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertTrue(signals['signal'].isin([-1, 0, 1]).all())

    def test_breakout_strategy(self):
        config = StrategyConfig(
            name="BreakoutTest",
            default_params={'breakout_window': 20, 'std_dev_multiplier': 2, 'instrument_key': 'EUR_USD', 'timeframe': 'D1'},
            applicable_assets=['EUR_USD']
        )
        strategy = BreakoutStrategy(config)
        
        processed_data = strategy.forward(self.market_data_dict, self.portfolio_context)
        self.assertIn('EUR_USD', processed_data)
        self.assertIn('upper_band', processed_data['EUR_USD'].columns)
        self.assertIn('lower_band', processed_data['EUR_USD'].columns)
        
        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertTrue(signals['signal'].isin([-1, 0, 1]).all())

    def test_trend_following_strategy(self):
        config = StrategyConfig(
            name="TrendFollowingTest", 
            default_params={'short_ma_window': 10, 'long_ma_window': 30, 'instrument_key': 'EUR_USD', 'timeframe': 'D1'},
            applicable_assets=['EUR_USD']
        )
        strategy = TrendFollowingStrategy(config)
        
        processed_data = strategy.forward(self.market_data_dict, self.portfolio_context)
        self.assertIn('EUR_USD', processed_data)
        # Check for actual column names that TrendFollowingStrategy produces
        # Based on trend_strategies.py, it should produce 'short_sma' and 'long_sma'
        self.assertIn('short_sma', processed_data['EUR_USD'].columns)
        self.assertIn('long_sma', processed_data['EUR_USD'].columns)
        
        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertTrue(signals['signal'].isin([-1, 0, 1]).all())

    def test_reversal_strategy(self):
        config = StrategyConfig(
            name="ReversalTest",
            default_params={'reversal_window': 14, 'rsi_oversold': 30, 'rsi_overbought': 70, 'instrument_key': 'EUR_USD', 'timeframe': 'D1'},
            applicable_assets=['EUR_USD']
        )
        strategy = ReversalStrategy(config)
        
        processed_data = strategy.forward(self.market_data_dict, self.portfolio_context)
        self.assertIn('EUR_USD', processed_data)
        self.assertIn('rsi', processed_data['EUR_USD'].columns)
        
        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertTrue(signals['signal'].isin([-1, 0, 1]).all())

if __name__ == '__main__':
    unittest.main()
