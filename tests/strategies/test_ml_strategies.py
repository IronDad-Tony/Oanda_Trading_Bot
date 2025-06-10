import unittest
import pandas as pd
import numpy as np
from src.agent.strategies.base_strategy import StrategyConfig
from src.agent.strategies.ml_strategies import (
    ReinforcementLearningStrategy, DeepLearningPredictionStrategy, 
    EnsembleLearningStrategy, TransferLearningStrategy
)

class TestMLStrategies(unittest.TestCase):

    def setUp(self):
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

    def test_reinforcement_learning_strategy(self):
        config = StrategyConfig(
            name="RLTest",
            default_params={
                'instrument_key': 'EUR_USD',
                'timeframe': 'D1',
                'model_path': 'dummy_rl_model.pth'
            },
            applicable_assets=['EUR_USD']
        )
        strategy = ReinforcementLearningStrategy(config)
        
        processed_data = strategy.forward(self.market_data_dict, self.portfolio_context)
        self.assertIn('EUR_USD', processed_data)
        # ReinforcementLearningStrategy produces 'feature_X' columns
        if not processed_data['EUR_USD'].empty:
            feature_columns = [col for col in processed_data['EUR_USD'].columns if col.startswith('feature_')]
            self.assertTrue(len(feature_columns) > 0)
        
        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)

    def test_deep_learning_prediction_strategy(self):
        config = StrategyConfig(
            name="DLModelTest",
            default_params={
                'instrument_key': 'EUR_USD',
                'timeframe': 'D1',
                'model_path': 'dummy_dl_model.pth',
                'sequence_length': 10
            },
            applicable_assets=['EUR_USD']
        )
        strategy = DeepLearningPredictionStrategy(config)
        
        processed_data = strategy.forward(self.market_data_dict, self.portfolio_context)
        self.assertIn('EUR_USD', processed_data)
        # DeepLearningPredictionStrategy produces 'lag_X' columns
        if not processed_data['EUR_USD'].empty:
            lag_columns = [col for col in processed_data['EUR_USD'].columns if col.startswith('lag_')]
            self.assertTrue(len(lag_columns) > 0)

        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)

    def test_ensemble_learning_strategy(self):
        # Mock base strategies for the ensemble - using dict format expected by the strategy
        base_strategy_configs = [
            {
                'name': 'BaseTrend',
                'class': 'TrendFollowingStrategy',
                'params': {'short_window': 5, 'long_window': 10, 'instrument_key': 'EUR_USD', 'timeframe': 'D1'}
            },
            {
                'name': 'BaseMeanReversion',
                'class': 'MeanReversionStrategy', 
                'params': {'window': 7, 'std_dev_multiplier': 1.5, 'instrument_key': 'EUR_USD', 'timeframe': 'D1'}
            }
        ]
        
        ensemble_config = StrategyConfig(
            name="EnsembleTest",
            default_params={
                'instrument_key': 'EUR_USD',
                'timeframe': 'D1',
                'base_strategy_configs': base_strategy_configs,
                'ensemble_method': 'majority_vote'
            },
            applicable_assets=['EUR_USD']
        )
        strategy = EnsembleLearningStrategy(ensemble_config)

        processed_data = strategy.forward(self.market_data_dict, self.portfolio_context)
        self.assertIn('EUR_USD', processed_data)
        # EnsembleLearningStrategy passes through market data when no base strategies are properly configured
        
        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)

    def test_transfer_learning_strategy(self):
        config = StrategyConfig(
            name="TransferLearnTest",
            default_params={
                'instrument_key': 'EUR_USD',
                'timeframe': 'D1',
                'base_model_path': 'dummy_base_model.pth',
                'num_fine_tune_layers': 2
            },
            applicable_assets=['EUR_USD']
        )
        strategy = TransferLearningStrategy(config)
        
        processed_data = strategy.forward(self.market_data_dict, self.portfolio_context)
        self.assertIn('EUR_USD', processed_data)
        # TransferLearningStrategy produces 'tl_lag_X' columns
        if not processed_data['EUR_USD'].empty:
            tl_lag_columns = [col for col in processed_data['EUR_USD'].columns if col.startswith('tl_lag_')]
            self.assertTrue(len(tl_lag_columns) > 0)

        signals = strategy.generate_signals(processed_data, self.portfolio_context)
        self.assertIsInstance(signals, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()