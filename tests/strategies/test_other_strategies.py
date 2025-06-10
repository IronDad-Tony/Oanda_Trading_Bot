import unittest
import pandas as pd
import numpy as np
from src.agent.strategies.base_strategy import StrategyConfig
from src.agent.strategies.other_strategies import (
    OptionFlowStrategy, MicrostructureStrategy, CarryTradeStrategy, MacroEconomicStrategy,
    EventDrivenStrategy, SentimentStrategy, QuantitativeStrategy, MarketMakingStrategy,
    HighFrequencyStrategy, AlgorithmicStrategy
)

class TestOtherStrategies(unittest.TestCase):

    def setUp(self):
        self.market_data_dict = {
            'EUR_USD': pd.DataFrame({
                'open': np.random.rand(100) * 1.1,
                'high': np.random.rand(100) * 1.12,
                'low': np.random.rand(100) * 1.08,
                'close': np.random.rand(100) * 1.1, 
                'volume': np.random.randint(100, 1000, 100),
                'bid_price': np.random.rand(100) * 1.09,
                'ask_price': np.random.rand(100) * 1.11,
                'bid_volume': np.random.randint(50, 500, 100),
                'ask_volume': np.random.randint(50, 500, 100),
                'option_volume': np.random.randint(100, 1000, 100), 
                'put_call_ratio': np.random.rand(100) * 2,
                'interest_rate': np.random.rand(100) * 0.01 # Added for EUR_USD
            }, index=pd.date_range(start='1/1/2020', periods=100)),
            'USD_CAD': pd.DataFrame({ 
                'open': np.random.rand(100) * 1.25,
                'high': np.random.rand(100) * 1.26,
                'low': np.random.rand(100) * 1.24,
                'close': np.random.rand(100) * 1.25,
                'volume': np.random.randint(100, 1000, 100),
                'interest_rate': np.random.rand(100) * 0.02 
            }, index=pd.date_range(start='1/1/2020', periods=100))
        }
        self.portfolio_context = {'cash': 100000, 'positions': {}}
        self.macro_economic_data = pd.DataFrame({
            'gdp_growth': np.random.rand(100) * 0.03,
            'inflation_rate': np.random.rand(100) * 0.02,
            'unemployment_rate': np.random.rand(100) * 0.05
        }, index=pd.date_range(start='1/1/2020', periods=100))
        self.news_sentiment_data = pd.DataFrame({
            'sentiment_score': np.random.rand(100) * 2 - 1, # Scores from -1 to 1
            'news_volume': np.random.randint(10, 100, 100)
        }, index=pd.date_range(start='1/1/2020', periods=100))

    def _get_market_data_for_strategy(self, applicable_assets):
        return { 
            k: self.market_data_dict[k] for k in applicable_assets 
            if k in self.market_data_dict
        } 

    def test_option_flow_strategy(self):
        config = StrategyConfig(
            name="OptionFlowTest",
            default_params={
                'instrument_key': 'EUR_USD',
                'timeframe': 'D1',
                'flow_threshold': 1000,
                'put_call_ratio_threshold': 0.7
            },
            applicable_assets=['EUR_USD']
        )
        strategy = OptionFlowStrategy(config)
        market_data = self._get_market_data_for_strategy(config.applicable_assets)
        processed_data = strategy.forward(market_data, self.portfolio_context)
        self.assertIn('EUR_USD', processed_data)
        if not processed_data['EUR_USD'].empty:
            self.assertIn('option_signal_strength', processed_data['EUR_USD'].columns)

        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertTrue(signals['signal'].isin([-1, 0, 1]).all())

    def test_microstructure_strategy(self):
        config = StrategyConfig(
            name="MicrostructureTest",
            default_params={
                'instrument_key': 'EUR_USD',
                'timeframe': 'M1',
                'order_imbalance_threshold': 0.6
            },
            applicable_assets=['EUR_USD']
        )
        strategy = MicrostructureStrategy(config)
        market_data = self._get_market_data_for_strategy(config.applicable_assets)
        processed_data = strategy.forward(market_data, self.portfolio_context)
        self.assertIn('EUR_USD', processed_data)
        if not processed_data['EUR_USD'].empty:
            self.assertIn('order_imbalance', processed_data['EUR_USD'].columns)

        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertTrue(signals['signal'].isin([-1, 0, 1]).all())

    def test_carry_trade_strategy(self):
        config = StrategyConfig(
            name="CarryTradeTest",
            default_params={
                'timeframe': 'D1',
                'short_ma_period': 10, # Added for clarity, matches strategy default
                'long_ma_period': 30   # Added for clarity, matches strategy default
            },
            applicable_assets=['EUR_USD', 'USD_CAD']
        )
        strategy = CarryTradeStrategy(config)
        market_data = self._get_market_data_for_strategy(config.applicable_assets)
        processed_data = strategy.forward(market_data, self.portfolio_context)
        for asset in config.applicable_assets:
            self.assertIn(asset, processed_data)
            self.assertIsInstance(processed_data[asset], pd.DataFrame) # Ensure it's a DataFrame
            if not processed_data[asset].empty:
                 # The simplified strategy uses MAs, not interest rate differentials directly
                 self.assertIn('short_ma', processed_data[asset].columns)
                 self.assertIn('long_ma', processed_data[asset].columns)

        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertTrue(signals['signal'].isin([-1, 0, 1]).all())

    def test_macro_economic_strategy(self):
        config = StrategyConfig(
            name="MacroEconomicTest",
            default_params={
                'timeframe': 'D1',
                'ma_period': 200, # Added for clarity
                'roc_period': 60  # Added for clarity
            },
            applicable_assets=['EUR_USD']
        )
        strategy = MacroEconomicStrategy(config)
        market_data = self._get_market_data_for_strategy(config.applicable_assets)
        # The test was passing external_data, but the simplified strategy doesn't use it.
        # The strategy's forward signature was updated to accept it to prevent TypeError.
        # No specific assertion for external_data processing is made as it's not used.
        processed_data = strategy.forward(market_data, self.portfolio_context, external_data=self.macro_economic_data)
        self.assertIn('EUR_USD', processed_data)
        if not processed_data['EUR_USD'].empty:
            # Asserting columns from the simplified strategy
            self.assertIn('long_ma', processed_data['EUR_USD'].columns)
            self.assertIn('roc', processed_data['EUR_USD'].columns)

        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertTrue(signals['signal'].isin([-1, 0, 1]).all())

    def test_event_driven_strategy(self):
        config = StrategyConfig(
            name="EventDrivenTest",
            default_params={
                'instrument_key': 'EUR_USD',
                'timeframe': 'H1',
                'event_impact_threshold': 0.5
            },
            applicable_assets=['EUR_USD']
        )
        strategy = EventDrivenStrategy(config)
        simulated_event = {'type': 'NFP', 'impact_score': 0.8, 'currency': 'USD'}
        self.portfolio_context['current_event'] = simulated_event
        market_data = self._get_market_data_for_strategy(config.applicable_assets)
        processed_data = strategy.forward(market_data, self.portfolio_context)
        self.assertIn('EUR_USD', processed_data)
        if not processed_data['EUR_USD'].empty:
            self.assertIn('event_driven_signal', processed_data['EUR_USD'].columns)

        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertTrue(signals['signal'].isin([-1, 0, 1]).all())
        del self.portfolio_context['current_event']

    def test_sentiment_strategy(self):
        config = StrategyConfig(
            name="SentimentTest",
            default_params={
                'timeframe': 'D1',
                'rsi_period': 14, # Added for clarity
                'stoch_k_period': 14, # Added for clarity
                'stoch_d_period': 3 # Added for clarity
            },
            applicable_assets=['EUR_USD']
        )
        strategy = SentimentStrategy(config)
        market_data = self._get_market_data_for_strategy(config.applicable_assets)
        # The test was passing sentiment_data, but the simplified strategy doesn't use it.
        # The strategy's forward signature was updated to accept it to prevent TypeError.
        # No specific assertion for sentiment_data processing is made as it's not used.
        processed_data = strategy.forward(market_data, self.portfolio_context, sentiment_data=self.news_sentiment_data)
        self.assertIn('EUR_USD', processed_data)
        if not processed_data['EUR_USD'].empty:
            # Asserting columns from the simplified strategy
            self.assertIn('rsi', processed_data['EUR_USD'].columns)
            self.assertIn('stoch_k', processed_data['EUR_USD'].columns)
            self.assertIn('stoch_d', processed_data['EUR_USD'].columns)

        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertTrue(signals['signal'].isin([-1, 0, 1]).all())

    def test_quantitative_strategy(self):
        config = StrategyConfig(
            name="QuantitativeTest",
            default_params={
                'instrument_key': 'EUR_USD',
                'timeframe': 'D1',
                'model_type': 'linear_regression' # Example parameter
            },
            applicable_assets=['EUR_USD']
        )
        strategy = QuantitativeStrategy(config)
        market_data = self._get_market_data_for_strategy(config.applicable_assets)
        processed_data = strategy.forward(market_data, self.portfolio_context)
        self.assertIn('EUR_USD', processed_data)
        if not processed_data['EUR_USD'].empty:
            self.assertIn('quantitative_prediction', processed_data['EUR_USD'].columns)

        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertTrue(signals['signal'].isin([-1, 0, 1]).all())

    def test_market_making_strategy(self):
        config = StrategyConfig(
            name="MarketMakingTest",
            default_params={
                'instrument_key': 'EUR_USD',
                'timeframe': 'S1', # Market making is very high frequency
                'spread_target': 0.0002, # Target 2 pips
                'inventory_limit': 10000 # Max inventory
            },
            applicable_assets=['EUR_USD']
        )
        strategy = MarketMakingStrategy(config)
        market_data = self._get_market_data_for_strategy(config.applicable_assets)
        processed_data = strategy.forward(market_data, self.portfolio_context)
        self.assertIn('EUR_USD', processed_data)
        if not processed_data['EUR_USD'].empty:
            self.assertIn('target_bid', processed_data['EUR_USD'].columns)
            self.assertIn('target_ask', processed_data['EUR_USD'].columns)

        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns) 

    def test_high_frequency_strategy(self):
        config = StrategyConfig(
            name="HighFrequencyTest",
            default_params={
                'instrument_key': 'EUR_USD',
                'timeframe': 'TICK', # HFT operates on tick data
                'latency_threshold_ms': 1 # Millisecond latency
            },
            applicable_assets=['EUR_USD']
        )
        strategy = HighFrequencyStrategy(config)
        market_data = self._get_market_data_for_strategy(config.applicable_assets)
        processed_data = strategy.forward(market_data, self.portfolio_context)
        self.assertIn('EUR_USD', processed_data)
        if not processed_data['EUR_USD'].empty:
            self.assertIn('hft_feature', processed_data['EUR_USD'].columns) # Example output

        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)

    def test_algorithmic_strategy(self):
        config = StrategyConfig(
            name="AlgorithmicTest",
            default_params={
                'instrument_key': 'EUR_USD',
                'timeframe': 'M5',
                'custom_param': 'value'
            },
            applicable_assets=['EUR_USD']
        )
        strategy = AlgorithmicStrategy(config)
        market_data = self._get_market_data_for_strategy(config.applicable_assets)
        processed_data = strategy.forward(market_data, self.portfolio_context)
        self.assertIn('EUR_USD', processed_data)
        if not processed_data['EUR_USD'].empty:
            self.assertIn('algo_output', processed_data['EUR_USD'].columns) 

        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertTrue(signals['signal'].isin([-1, 0, 1]).all())

if __name__ == '__main__':
    unittest.main()
