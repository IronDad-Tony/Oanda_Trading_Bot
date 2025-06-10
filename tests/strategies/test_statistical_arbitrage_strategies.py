import unittest
import pandas as pd
import numpy as np
from src.agent.strategies.base_strategy import StrategyConfig
from src.agent.strategies.statistical_arbitrage_strategies import (
    MeanReversionStrategy, CointegrationStrategy, VolatilityBreakoutStrategy, 
    StatisticalArbitrageStrategy, VolatilityArbitrageStrategy
)

class TestStatisticalArbitrageStrategies(unittest.TestCase):

    def setUp(self):
        # Create market data with both uppercase and lowercase column names
        self.market_data_dict = {
            'EUR_USD': pd.DataFrame({
                'Open': np.random.rand(100) * 1.1,
                'High': np.random.rand(100) * 1.12,
                'Low': np.random.rand(100) * 1.08,
                'Close': np.random.rand(100) * 1.1,
                'open': np.random.rand(100) * 1.1,
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
        self.portfolio_context = {'cash': 100000, 'positions': {}}

    def test_mean_reversion_strategy(self):
        config = StrategyConfig(
            name="MeanReversionTest",
            default_params={'bb_period': 20, 'bb_std_dev': 2, 'instrument_key': 'EUR_USD', 'timeframe': 'D1'},
            applicable_assets=['EUR_USD']
        )
        strategy = MeanReversionStrategy(config)
        
        processed_data = strategy.forward(self.market_data_dict, self.portfolio_context)
        self.assertIn('EUR_USD', processed_data)
        # MeanReversionStrategy produces 'mavg', 'hband', 'lband' columns based on Bollinger Bands
        self.assertIn('mavg', processed_data['EUR_USD'].columns)
        self.assertIn('hband', processed_data['EUR_USD'].columns)
        self.assertIn('lband', processed_data['EUR_USD'].columns)
        
        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)
        # MeanReversionStrategy returns a DataFrame with asset names as columns
        if not signals.empty:
            self.assertTrue(all(col in ['EUR_USD'] for col in signals.columns))

    def test_cointegration_strategy(self):
        config = StrategyConfig(
            name="CointegrationTest",
            default_params={'window': 20, 'entry_threshold': 1.5, 'exit_threshold': 0.5, 'asset_pair': ['EUR_USD', 'USD_JPY'], 'timeframe': 'D1'},
            applicable_assets=['EUR_USD', 'USD_JPY']
        )
        strategy = CointegrationStrategy(config)
        
        market_data_for_strategy = {k: self.market_data_dict[k] for k in config.applicable_assets}
        processed_data = strategy.forward(market_data_for_strategy, self.portfolio_context)
        
        # CointegrationStrategy should return a processed data dict
        # The spread should be calculated and stored
        if processed_data:
            # Check if any data was processed
            first_key = list(processed_data.keys())[0]
            self.assertIsInstance(processed_data[first_key], pd.DataFrame)
        
        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)

    def test_volatility_breakout_strategy(self):
        config = StrategyConfig(
            name="VolatilityBreakoutTest",
            default_params={'atr_period': 14, 'donchian_period': 20, 'multiplier': 1.5, 'instrument_key': 'EUR_USD', 'timeframe': 'D1'},
            applicable_assets=['EUR_USD']
        )
        strategy = VolatilityBreakoutStrategy(config)
        
        processed_data = strategy.forward(self.market_data_dict, self.portfolio_context)
        self.assertIn('EUR_USD', processed_data)
        # VolatilityBreakoutStrategy produces 'atr', 'donchian_hband', 'donchian_lband' columns
        if not processed_data['EUR_USD'].empty:
            self.assertIn('atr', processed_data['EUR_USD'].columns)
            self.assertIn('donchian_hband', processed_data['EUR_USD'].columns)
            self.assertIn('donchian_lband', processed_data['EUR_USD'].columns)
        
        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)

    def test_statistical_arbitrage_strategy(self):
        config = StrategyConfig(
            name="StatisticalArbitrageTest",
            default_params={'window': 20, 'threshold': 1.0, 'instrument_key': ['EUR_USD', 'USD_JPY'], 'timeframe': 'D1'},
            applicable_assets=['EUR_USD', 'USD_JPY']
        )
        strategy = StatisticalArbitrageStrategy(config)
        market_data_for_strategy = {k: self.market_data_dict[k] for k in config.applicable_assets}

        processed_data = strategy.forward(market_data_for_strategy, self.portfolio_context)
        # StatisticalArbitrageStrategy is a composite strategy
        self.assertIsInstance(processed_data, dict)
        
        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)

    def test_volatility_arbitrage_strategy(self):
        config = StrategyConfig(
            name="VolatilityArbitrageTest",
            default_params={'vol_window_short': 10, 'vol_window_long': 50, 'vol_threshold_ratio': 1.5, 'instrument_key': 'EUR_USD', 'timeframe': 'D1'},
            applicable_assets=['EUR_USD']
        )
        strategy = VolatilityArbitrageStrategy(config)
        
        processed_data = strategy.forward(self.market_data_dict, self.portfolio_context)
        self.assertIn('EUR_USD', processed_data)
        # VolatilityArbitrageStrategy produces volatility-related columns
        if not processed_data['EUR_USD'].empty:
            self.assertIn('vol_ratio', processed_data['EUR_USD'].columns)
        
        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)

if __name__ == '__main__':
    unittest.main()
