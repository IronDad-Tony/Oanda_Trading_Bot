import unittest
from unittest.mock import MagicMock, patch, ANY, call # Added call
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Type, Callable # Ensure all necessary typing imports are here
from sklearn.model_selection import TimeSeriesSplit # Added this import

from src.agent.enhanced_quantum_strategy_layer import DynamicStrategyGenerator
from src.agent.strategies.base_strategy import BaseStrategy, StrategyConfig # Corrected import for StrategyConfig
from src.agent.optimizers.genetic_optimizer import GeneticOptimizer

# Try to import mock_fitness_function from test_optimizers
try:
    from .test_optimizers import mock_fitness_function
except ImportError:
    try:
        from test_optimizers import mock_fitness_function
    except ImportError:
        def mock_fitness_function(strategy_instance: BaseStrategy, current_context: Optional[Dict]) -> float: # Updated signature
            # This is a placeholder if import fails.
            # Real mock_fitness_function should be compatible with DSG's GA wrapper.
            params = strategy_instance.get_params()
            fitness = 0.0
            if 'param_A' in params: fitness += params['param_A']
            if 'param_B' in params: fitness += params['param_B'] * 10
            return fitness

class MockStrategy(BaseStrategy):
    identifier = "MockStrategy"
    version = "1.0"
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(name="MockStrategy", default_params={'param_A': 10, 'param_B': 0.5, 'param_C': 'default_str'}) # Added param_C for more tests
    
    @staticmethod
    def param_definitions() -> Dict[str, Dict[str, Any]]:
        return {
            'param_A': {'type': int, 'default': 10, 'min': 1, 'max': 100, 'step': 1},
            'param_B': {'type': float, 'default': 0.5, 'min': 0.1, 'max': 1.0, 'step': 0.01},
            'param_C': {'type': str, 'default': 'default_str', 'choices': ['default_str', 'option1', 'option2']},
        }

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, market_data_source=None, risk_manager=None, portfolio_manager=None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params, logger=logger)
        self.logger_instance = logger # Store logger if needed for assertions
    def _calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not data.empty and 'close' in data.columns and len(data) > 1:
            signals = np.sign(data['close'].diff().fillna(0)).astype(int)
            return pd.Series(signals, index=data.index)
        return pd.Series(np.zeros(len(data)), index=data.index)

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]: # Updated signature
        # Simplified: operates on the first available DataFrame in market_data_dict
        # and uses 'default_asset' if provided in portfolio_context.
        asset_key = next(iter(market_data_dict)) if market_data_dict else None
        if portfolio_context and portfolio_context.get("default_asset") in market_data_dict:
            asset_key = portfolio_context.get("default_asset")

        if not asset_key or market_data_dict[asset_key].empty:
            return {asset_key: pd.DataFrame()} if asset_key else {}
            
        data = market_data_dict[asset_key]
        signals = self._calculate_signals(data)
        output_df = pd.DataFrame({'signal': signals, 'value': data['close'] if 'close' in data else np.random.rand(len(data))}, index=data.index)
        return {asset_key: output_df}
    
    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame: # Updated signature
        # Simplified: returns signals for the first asset or 'default_asset'
        asset_key = next(iter(processed_data_dict)) if processed_data_dict else None
        if portfolio_context and portfolio_context.get("default_asset") in processed_data_dict:
            asset_key = portfolio_context.get("default_asset")

        if not asset_key or asset_key not in processed_data_dict or processed_data_dict[asset_key].empty:
            return pd.DataFrame(columns=['signal'])
            
        return processed_data_dict[asset_key][['signal']]


class AnotherMockStrategy(BaseStrategy):
    identifier = "AnotherMockStrategy"
    version = "1.0"
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(name="AnotherMockStrategy",
                              description="Another mock strategy for testing different param types.",
                              default_params={'param_X': 100, 'param_Y': True})

    @staticmethod
    def param_definitions() -> Dict[str, Dict[str, Any]]:
        return {
            'param_X': {'type': int, 'default': 100, 'min': 50, 'max': 200, 'step': 1},
            'param_Y': {'type': bool, 'default': True}, # BaseStrategy.get_parameter_space handles bool
        }

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, market_data_source=None, risk_manager=None, portfolio_manager=None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params, logger=logger)
        self.logger_instance = logger # Store logger if needed for assertions
    def _calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(np.random.randint(0, 2, len(data)), index=data.index)
    
    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]: # Updated signature
        asset_key = next(iter(market_data_dict)) if market_data_dict else None
        if not asset_key or market_data_dict[asset_key].empty:
             return {asset_key: pd.DataFrame()} if asset_key else {}
        data = market_data_dict[asset_key]
        signals = self._calculate_signals(data)
        param_x_value = self.params.get('param_X', 1)
        output_df = pd.DataFrame({'signal': signals, 'value_X': np.random.rand(len(data)) * param_x_value}, index=data.index)
        return {asset_key: output_df}

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame: # Updated signature
        asset_key = next(iter(processed_data_dict)) if processed_data_dict else None
        if not asset_key or asset_key not in processed_data_dict or processed_data_dict[asset_key].empty:
            return pd.DataFrame(columns=['signal'])
        return processed_data_dict[asset_key][['signal']]


class TestDynamicStrategyGenerator(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger('TestDSG') # Use a real logger for tests
        # self.logger_name = 'TradingSystem' # No longer needed for DSG init

        # MockStrategy and AnotherMockStrategy are defined in this file
        # Their param_definitions and get_parameter_space methods will be used directly

        self.optimizer_config_for_dsg = {
            "name": "GeneticOptimizer",
            "settings": {
                'population_size': 5, 
                'n_generations': 3,  
                'mutation_rate': 0.1,
                'crossover_rate': 0.7,
                'tournament_size': 2,
            }
        }
        
        self.historical_data_dict = { 
            'asset1': pd.DataFrame({'close': np.random.rand(50) * 100}, index=pd.date_range(start='2023-01-01', periods=50))
        }

        # Initialize DSG with the new signature
        self.dsg = DynamicStrategyGenerator(
            logger=self.logger,
            optimizer_config=self.optimizer_config_for_dsg
        )

        self.mock_market_data_dict = {
            'asset1': pd.DataFrame({'close': np.random.rand(50) * 100}, index=pd.date_range(start='2023-01-01', periods=50))
        }
        # current_context_for_test is now defined where needed, or as a common attribute if used across many tests
        self.current_context_for_test = {'balance': 10000, 'market_data': self.mock_market_data_dict}

        # Patch GeneticOptimizer where it's instantiated (inside enhanced_quantum_strategy_layer)
        patcher = patch('src.agent.enhanced_quantum_strategy_layer.GeneticOptimizer', autospec=True)
        self.mock_genetic_optimizer_class = patcher.start()
        self.addCleanup(patcher.stop)
        
        self.mock_genetic_optimizer_instance = self.mock_genetic_optimizer_class.return_value
        # Default return for run_optimizer: best_params (dict), best_fitness (float)
        self.mock_genetic_optimizer_instance.run_optimizer.return_value = ({}, -float('inf')) # Default GA failure (no params found)

        self.mock_fitness_function_for_dsg = MagicMock(return_value=1.0) # Mock fitness function for DSG
        
        # This logger is for the DSG itself, not for BaseStrategy or other components unless passed explicitly.
        # For asserting logs from BaseStrategy, use the logger name like f"{MockStrategy.__module__}.{MockStrategy.__name__}"
        self.dsg_logger_name = 'src.agent.enhanced_quantum_strategy_layer' # Or whatever logger DSG uses internally if different from self.logger passed to it.
                                                                          # Based on current DSG, it uses the logger passed to it.

    def tearDown(self):
        # Stop any patchers started in individual tests if not using addCleanup
        pass

    def test_initialization(self):
        """Test DynamicStrategyGenerator initialization."""
        self.assertIsNotNone(self.dsg.logger)
        self.assertEqual(self.dsg.optimizer_config, self.optimizer_config_for_dsg)
        self.assertIsNone(self.dsg.genetic_optimizer) # GA is initialized in generate_new_strategy

    # test_default_fitness_function_integration removed as _default_fitness_function is no longer part of DSG

    def test_generate_new_strategy_no_optimizer(self):
        current_context = {"some_key": "some_value", "market_data": self.mock_market_data_dict}
        
        strategy_no_fitness = self.dsg.generate_new_strategy(
            strategy_class=MockStrategy,
            fitness_function=None, 
            current_context=current_context
        )
        self.assertIsNotNone(strategy_no_fitness)
        self.assertIsInstance(strategy_no_fitness, MockStrategy)
        self.mock_genetic_optimizer_class.assert_not_called() 

        dsg_no_opt_config = DynamicStrategyGenerator(logger=self.logger, optimizer_config=None)
        # Reset the global mock before this call if it could have been called by self.dsg
        self.mock_genetic_optimizer_class.reset_mock()
        strategy_dsg_no_opt = dsg_no_opt_config.generate_new_strategy(
            strategy_class=MockStrategy,
            fitness_function=self.mock_fitness_function_for_dsg, 
            current_context=current_context
        )
        self.assertIsNotNone(strategy_dsg_no_opt)
        self.assertIsInstance(strategy_dsg_no_opt, MockStrategy)
        self.mock_genetic_optimizer_class.assert_not_called() # GeneticOptimizer class should not be instantiated by dsg_no_opt_config
        self.assertIsNone(dsg_no_opt_config.genetic_optimizer)


    def test_generate_new_strategy_unknown_strategy(self):
        # DSG's generate_new_strategy now requires a valid strategy_class.
        current_context = {"some_key": "some_value", "market_data": self.mock_market_data_dict}
        with self.assertLogs(logger=self.logger.name, level='ERROR') as log:
            strategy_instance = self.dsg.generate_new_strategy(
                strategy_class=None, # type: ignore # Pass None as an invalid class
                fitness_function=self.mock_fitness_function_for_dsg,
                current_context=current_context
            )
        self.assertIsNone(strategy_instance)
        self.assertIn("Invalid strategy_class provided: None", "".join(log.output))

        # Test with a class that is not a BaseStrategy subclass
        class NotAStrategy: pass
        with self.assertLogs(logger=self.logger.name, level='ERROR') as log:
            strategy_instance_invalid_type = self.dsg.generate_new_strategy(
                strategy_class=NotAStrategy, # type: ignore 
                fitness_function=self.mock_fitness_function_for_dsg,
                current_context=current_context
            )
        self.assertIsNone(strategy_instance_invalid_type)
        self.assertIn(f"Invalid strategy_class provided: {NotAStrategy}", "".join(log.output))


    def test_generate_new_strategy_with_optimizer(self):
        optimized_params_dict = {'param_A': 75, 'param_B': 0.75, 'param_C': 'option1'}
        
        mock_strategy_param_space = MockStrategy.get_parameter_space(optimizer_type="genetic")
        self.assertIsNotNone(mock_strategy_param_space)
        param_keys = list(mock_strategy_param_space.keys())
        
        self.assertEqual(set(optimized_params_dict.keys()), set(param_keys))

        optimized_params_tuple = tuple(optimized_params_dict[k] for k in param_keys)

        self.mock_genetic_optimizer_instance.run_optimizer.return_value = (optimized_params_tuple, 100.0)
        # Reset mocks for self.dsg related optimizer calls for this specific test
        self.mock_genetic_optimizer_class.reset_mock()
        self.mock_genetic_optimizer_instance.reset_mock()
        self.mock_genetic_optimizer_instance.run_optimizer.return_value = (optimized_params_tuple, 100.0) # Re-assign after reset
        # Link the instance back to the class mock if it was reset
        self.mock_genetic_optimizer_class.return_value = self.mock_genetic_optimizer_instance

        current_context = {"market_data": self.historical_data_dict} # This is the portfolio_context
        market_data_for_ga_arg = self.historical_data_dict # This is the market_data for fitness
        
        strategy_instance = self.dsg.generate_new_strategy(
            strategy_class=MockStrategy,
            fitness_function=self.mock_fitness_function_for_dsg, 
            current_context=current_context,
            market_data_for_ga=market_data_for_ga_arg
        )
        self.assertIsNotNone(self.dsg.genetic_optimizer, "Genetic optimizer instance should be created on DSG")
        self.assertIsInstance(strategy_instance, MockStrategy)
        
        self.mock_genetic_optimizer_class.assert_called_once()
        args, kwargs = self.mock_genetic_optimizer_class.call_args
        self.assertEqual(kwargs.get('param_space'), mock_strategy_param_space)
        self.mock_genetic_optimizer_instance.run_optimizer.assert_called_once()

        self.assertEqual(strategy_instance.params['param_A'], optimized_params_dict['param_A'])
        self.assertEqual(strategy_instance.params['param_B'], optimized_params_dict['param_B'])
        self.assertEqual(strategy_instance.params['param_C'], optimized_params_dict['param_C'])


    @patch(f'{MockStrategy.__module__}.MockStrategy.get_parameter_space')
    def test_generate_new_strategy_optimizer_missing_param_space(self, mock_get_param_space):
        mock_get_param_space.return_value = None 
        
        current_context = {"market_data": self.historical_data_dict}
        self.mock_genetic_optimizer_class.reset_mock()
        self.mock_genetic_optimizer_instance.reset_mock()

        with self.assertLogs(logger=self.logger.name, level='INFO') as log: 
            strategy_instance = self.dsg.generate_new_strategy(
                strategy_class=MockStrategy,
                fitness_function=self.mock_fitness_function_for_dsg,
                current_context=current_context
            )
        self.assertIsInstance(strategy_instance, MockStrategy) 
        self.assertIn("param_space_for_ga present and not empty: False", "".join(log.output)) # Check log reason
        default_conf = MockStrategy.default_config()
        self.assertEqual(strategy_instance.params['param_A'], default_conf.default_params['param_A'])
        
        self.mock_genetic_optimizer_class.assert_not_called() # Class should not be instantiated if param space is missing
        self.mock_genetic_optimizer_instance.run_optimizer.assert_not_called()
        self.assertIsNone(self.dsg.genetic_optimizer)


    def test_generate_new_strategy_optimizer_missing_ga_config(self):
        dsg_no_ga_config = DynamicStrategyGenerator(logger=self.logger, optimizer_config=None)
        
        current_context = {"market_data": self.historical_data_dict}
        # The global mock self.mock_genetic_optimizer_class is active.
        # We need to ensure *this specific call* for dsg_no_ga_config does not use it.
        self.mock_genetic_optimizer_class.reset_mock()

        with self.assertLogs(logger=self.logger.name, level='INFO') as log: 
            strategy_instance = dsg_no_ga_config.generate_new_strategy(
                strategy_class=MockStrategy,
                fitness_function=self.mock_fitness_function_for_dsg, 
                current_context=current_context
            )
        self.assertIsInstance(strategy_instance, MockStrategy) 
        self.assertIn("optimizer_config present and named 'GeneticOptimizer': False", "".join(log.output))
        default_conf = MockStrategy.default_config()
        self.assertEqual(strategy_instance.params['param_A'], default_conf.default_params['param_A'])
        self.mock_genetic_optimizer_class.assert_not_called() 
        self.assertIsNone(dsg_no_ga_config.genetic_optimizer)


    def test_generate_new_strategy_with_optimizer_param_type_handling(self):
        # GA returns raw values. BaseStrategy.__init__ does coercion.
        # param_X: int, param_Y: bool
        
        # AnotherMockStrategy.param_definitions: param_X (int), param_Y (bool)
        # AnotherMockStrategy.get_parameter_space will define the space for these.
        
        # GA would return a tuple based on the order from get_parameter_space
        # Let's assume get_parameter_space for AnotherMockStrategy returns keys in order: param_X, param_Y
        ga_returned_params_tuple = ("77", "false") 
        # The mock optimizer will return this tuple. DSG will then convert it to a dict.
        # The keys for this dict come from param_space_for_ga.keys()
        # So, we need to ensure the mock_genetic_optimizer_instance.run_optimizer returns a tuple
        # that matches the order of keys in the param_space used by DSG when it calls the optimizer.

        # Get the expected param_space to ensure keys are ordered correctly for the mock tuple
        param_space_for_another_mock = AnotherMockStrategy.get_parameter_space(optimizer_type="genetic")
        self.assertIsNotNone(param_space_for_another_mock)
        param_keys_ordered = list(param_space_for_another_mock.keys()) # e.g., ['param_X', 'param_Y']
        
        # Ensure our ga_returned_params_tuple matches this order and content
        # If param_keys_ordered is ['param_X', 'param_Y'], then ga_returned_params_tuple = ("77", "false") is correct.
        # If param_keys_ordered is ['param_Y', 'param_X'], then it should be ("false", "77").
        # For AnotherMockStrategy, param_definitions are X then Y, so space should be X then Y.
        self.assertEqual(param_keys_ordered, ['param_X', 'param_Y'])

        self.mock_genetic_optimizer_instance.run_optimizer.return_value = (ga_returned_params_tuple, 90.0)
        self.mock_genetic_optimizer_class.reset_mock() 
        self.mock_genetic_optimizer_instance.reset_mock() 
        self.mock_genetic_optimizer_instance.run_optimizer.return_value = (ga_returned_params_tuple, 90.0) 
        self.mock_genetic_optimizer_class.return_value = self.mock_genetic_optimizer_instance 

        current_context_for_test = {'market_data': self.historical_data_dict}
        market_data_for_ga_arg = self.historical_data_dict
        
        with self.assertLogs(logger=self.logger.name, level='DEBUG') as basestrategy_log: 
            strategy_instance = self.dsg.generate_new_strategy(
                strategy_class=AnotherMockStrategy,
                fitness_function=self.mock_fitness_function_for_dsg,
                current_context=current_context_for_test,
                market_data_for_ga=market_data_for_ga_arg
            )
        
        self.assertIsNotNone(strategy_instance, "Strategy instance should not be None")
        self.assertIsInstance(strategy_instance, AnotherMockStrategy)
        
        log_output = "".join(basestrategy_log.output)
        
        self.assertIn("Strategy AnotherMockStrategy: Parameter 'param_X' ('77') converted from str to int ('77')", log_output)
        self.assertIn("Strategy AnotherMockStrategy: Parameter 'param_Y' ('false') converted from str to bool ('False')", log_output)

        self.assertIsInstance(strategy_instance.params.get('param_X'), int)
        self.assertEqual(strategy_instance.params.get('param_X'), 77)
        self.assertIsInstance(strategy_instance.params.get('param_Y'), bool)
        self.assertEqual(strategy_instance.params.get('param_Y'), False) 
        
        # Test with GA returning already correct types
        # Ensure the tuple matches the order of param_keys_ordered
        ga_returned_params_tuple_correct_types = (88, True) # (param_X, param_Y)
        self.mock_genetic_optimizer_instance.run_optimizer.return_value = (ga_returned_params_tuple_correct_types, 95.0)
        self.mock_genetic_optimizer_instance.run_optimizer.reset_mock()
        self.mock_genetic_optimizer_instance.run_optimizer.return_value = (ga_returned_params_tuple_correct_types, 95.0) 

        strategy_instance_2 = self.dsg.generate_new_strategy(
            strategy_class=AnotherMockStrategy,
            fitness_function=self.mock_fitness_function_for_dsg,
            current_context=current_context_for_test,
            market_data_for_ga=market_data_for_ga_arg
        )
        self.assertEqual(strategy_instance_2.params.get('param_X'), 88)
        self.assertEqual(strategy_instance_2.params.get('param_Y'), True)
        self.mock_genetic_optimizer_instance.run_optimizer.assert_called_once() 


    def test_generate_new_strategy_uses_default_when_optimizer_fails(self):
        self.mock_genetic_optimizer_instance.run_optimizer.return_value = ([], -float('inf'))
        self.mock_genetic_optimizer_class.reset_mock()
        self.mock_genetic_optimizer_instance.reset_mock()
        self.mock_genetic_optimizer_instance.run_optimizer.return_value = ([], -float('inf'))
        self.mock_genetic_optimizer_class.return_value = self.mock_genetic_optimizer_instance

        current_context = {"market_data": self.mock_market_data_dict}
        market_data_for_ga_arg = self.mock_market_data_dict
        
        strategy = self.dsg.generate_new_strategy(
            strategy_class=MockStrategy,
            fitness_function=self.mock_fitness_function_for_dsg,
            current_context=current_context,
            market_data_for_ga=market_data_for_ga_arg
        )
        self.assertIsNotNone(strategy)
        self.mock_genetic_optimizer_instance.run_optimizer.assert_called_once() # It was called
        
        default_mock_params = MockStrategy.default_config().default_params
        self.assertEqual(strategy.params['param_A'], default_mock_params['param_A'])
        self.assertEqual(strategy.params['param_B'], default_mock_params['param_B'])
        self.assertIsInstance(strategy, MockStrategy)

    def test_generate_new_strategy_with_initial_params_override(self):
        initial_params_payload = {'param_A': 123, 'param_C': 'overridden'} # Renamed variable
        current_context = {"market_data": self.mock_market_data_dict}
        
        dsg_no_opt_config = DynamicStrategyGenerator(logger=self.logger, optimizer_config=None)
        self.mock_genetic_optimizer_class.reset_mock() # Reset global mock
        
        strategy_no_opt = dsg_no_opt_config.generate_new_strategy(
            strategy_class=MockStrategy,
            fitness_function=self.mock_fitness_function_for_dsg, 
            current_context=current_context,
            initial_parameters=initial_params_payload.copy() # Use new arg name
        )
        self.mock_genetic_optimizer_class.assert_not_called()

        self.assertIsNotNone(strategy_no_opt)
        self.assertEqual(strategy_no_opt.params['param_A'], 123)
        self.assertEqual(strategy_no_opt.params['param_B'], MockStrategy.default_config().default_params['param_B'])
        self.assertEqual(strategy_no_opt.params['param_C'], 'overridden')

        self.mock_genetic_optimizer_instance.run_optimizer.return_value = ([], -float('inf'))
        self.mock_genetic_optimizer_class.reset_mock()
        self.mock_genetic_optimizer_instance.reset_mock()
        self.mock_genetic_optimizer_instance.run_optimizer.return_value = ([], -float('inf'))
        self.mock_genetic_optimizer_class.return_value = self.mock_genetic_optimizer_instance

        market_data_for_ga_arg = self.mock_market_data_dict
        strategy_opt_fail = self.dsg.generate_new_strategy(
            strategy_class=MockStrategy,
            fitness_function=self.mock_fitness_function_for_dsg,
            current_context=current_context,
            initial_parameters=initial_params_payload.copy(), # Use new arg name
            market_data_for_ga=market_data_for_ga_arg
        )
        self.assertIsNotNone(strategy_opt_fail)
        self.mock_genetic_optimizer_instance.run_optimizer.assert_called_once()
        self.assertEqual(strategy_opt_fail.params['param_A'], 123)
        self.assertEqual(strategy_opt_fail.params['param_B'], MockStrategy.default_config().default_params['param_B'])
        self.assertEqual(strategy_opt_fail.params['param_C'], 'overridden')

    def test_generate_new_strategy_with_base_config_override(self):
        overridden_default_params = {'param_A': 99, 'param_B': 0.99, 'param_C': 'config_override_C'}
        base_override_config_payload = StrategyConfig( # Renamed variable
            name="MockStrategyOverridden", 
            description="Overridden config",
            default_params=overridden_default_params
        )
        current_context = {"market_data": self.mock_market_data_dict}
        
        dsg_no_opt_config = DynamicStrategyGenerator(logger=self.logger, optimizer_config=None)
        self.mock_genetic_optimizer_class.reset_mock()

        strategy = dsg_no_opt_config.generate_new_strategy(
            strategy_class=MockStrategy, 
            fitness_function=self.mock_fitness_function_for_dsg, 
            current_context=current_context,
            strategy_config_override=base_override_config_payload # Use new arg name
        )
        self.mock_genetic_optimizer_class.assert_not_called()


    def test_generate_new_strategy_with_complex_context(self):
        complex_context = {
            "market_regime": "oscillating", "risk_aversion": "high",
            "investment_horizon": "long_term", "market_data": self.mock_market_data_dict
        }
        optimized_params_dict = {'param_A': 75, 'param_B': 0.75, 'param_C': 'option1'}
        
        mock_strategy_param_space = MockStrategy.get_parameter_space(optimizer_type="genetic")
        param_keys = list(mock_strategy_param_space.keys())
        self.assertEqual(set(optimized_params_dict.keys()), set(param_keys))
        optimized_params_tuple = tuple(optimized_params_dict[k] for k in param_keys)
        
        # Simulate optimizer finding these params
        self.mock_genetic_optimizer_instance.run_optimizer.return_value = (optimized_params_tuple, 100.0)
        
        # Reset mocks for this specific test path
        self.mock_genetic_optimizer_class.reset_mock()
        self.mock_genetic_optimizer_instance.reset_mock()
        self.mock_genetic_optimizer_instance.run_optimizer.return_value = (optimized_params_tuple, 100.0)
        self.mock_genetic_optimizer_class.return_value = self.mock_genetic_optimizer_instance
        self.mock_fitness_function_for_dsg.reset_mock()

        market_data_for_ga_arg = complex_context['market_data']

        strategy = self.dsg.generate_new_strategy(
            strategy_class=MockStrategy,
            fitness_function=self.mock_fitness_function_for_dsg, # This is the function ga_fitness_wrapper should call
            current_context=complex_context,
            market_data_for_ga=market_data_for_ga_arg
        )
        self.assertIsNotNone(strategy)
        self.assertIsInstance(strategy, MockStrategy)

        # 1. Assert GeneticOptimizer was called and capture its arguments
        self.mock_genetic_optimizer_class.assert_called_once()
        constructor_args, constructor_kwargs = self.mock_genetic_optimizer_class.call_args
        
        # 2. Extract the ga_fitness_wrapper passed to the GeneticOptimizer
        ga_fitness_wrapper_passed_to_optimizer = constructor_kwargs.get('fitness_function_callback')
        self.assertIsNotNone(ga_fitness_wrapper_passed_to_optimizer)

        # 3. Call the captured ga_fitness_wrapper with mock data to simulate GA's internal call
        #    The strategy instance for GA would be created by GA itself using the trial params.
        #    We need a mock strategy instance configured with the trial parameters.
        #    The `optimized_params_dict` are what the GA *would have found*.
        #    Let's simulate the GA trying out these `optimized_params_dict`.
        
        # Create a dummy strategy instance as GA would, using the params GA is testing.
        # The config for this instance would be the one DSG prepared for the optimizer.
        default_config_for_mock_strategy = MockStrategy.default_config()
        # The logger passed to this instance inside GA would be the GA's logger.
        # For this test, we can use self.logger or a new mock logger.
        mock_ga_logger = MagicMock(spec=logging.Logger)
        
        # Simulate the strategy instance that GA's _calculate_fitness would create and pass
        # It would use the `base_config` given to GA and the `individual_params` (optimized_params_dict here)
        strategy_instance_for_ga_evaluation = MockStrategy(
            config=constructor_kwargs.get('base_config', default_config_for_mock_strategy),
            params=optimized_params_dict, 
            logger=mock_ga_logger
        )

        # Call the wrapper with this instance and other relevant data
        ga_fitness_wrapper_passed_to_optimizer(
            strategy_instance_for_ga_evaluation, 
            market_data_for_ga_arg, # market_data_for_fitness_eval
            complex_context,        # portfolio_context_for_fitness_eval
            optimized_params_dict   # raw_params_from_ga
        )

        # 4. Assert that self.mock_fitness_function_for_dsg was called by the wrapper
        self.mock_fitness_function_for_dsg.assert_called_once()
        args_to_actual_fitness_func, _ = self.mock_fitness_function_for_dsg.call_args
        
        called_strategy_instance, called_portfolio_context, called_market_data, called_raw_params = args_to_actual_fitness_func
        
        self.assertIsInstance(called_strategy_instance, MockStrategy)
        self.assertEqual(called_strategy_instance.params, optimized_params_dict)
        self.assertEqual(called_portfolio_context, complex_context)
        self.assertEqual(called_market_data, market_data_for_ga_arg)
        self.assertEqual(called_raw_params, optimized_params_dict)

        # Check that the final strategy instance has the optimized params
        self.assertEqual(strategy.params['param_A'], optimized_params_dict['param_A'])


    def test_generate_new_strategy_with_invalid_context_type(self):
        with self.assertLogs(logger=self.logger.name, level='ERROR') as log:
            strategy = self.dsg.generate_new_strategy(
                strategy_class=MockStrategy,
                fitness_function=self.mock_fitness_function_for_dsg,
                current_context="invalid_context_type" # type: ignore
            )
        self.assertIsNone(strategy)
        self.assertIn("Invalid current_context type: <class 'str'>. Expected Dict or None.", "".join(log.output))

    def test_generate_new_strategy_with_extra_strategy_params_from_optimizer(self):
        initial_params_with_extra = {'param_A': 130, 'extra_param': 999}
        current_context = {"market_data": self.mock_market_data_dict}

        dsg_no_opt_config = DynamicStrategyGenerator(logger=self.logger, optimizer_config=None)
        
        # Logs will be emitted by the logger passed from DSG to BaseStrategy, which is self.logger (TestDSG)
        with self.assertLogs(logger=self.logger.name, level='WARNING') as log: 
            strategy = dsg_no_opt_config.generate_new_strategy(
                strategy_class=MockStrategy,
                fitness_function=self.mock_fitness_function_for_dsg, 
                current_context=current_context,
                initial_parameters=initial_params_with_extra 
            )
        
        self.assertIsNotNone(strategy)
        # BaseStrategy logs: "Strategy {self.config.name}: Received unexpected parameter '{key}' with value '{value}'. It will be included."
        self.assertIn("Strategy MockStrategy: Received unexpected parameter 'extra_param' with value '999'. It will be included.", "".join(log.output))
        self.assertIn('extra_param', strategy.params) 
        self.assertEqual(strategy.params['extra_param'], 999)
        self.assertEqual(strategy.params['param_A'], 130)

    # def test_generate_new_strategy_with_conflicting_param_spaces_ga_behavior(self):
    # This test's original premise (DSG managing multiple, potentially conflicting param spaces)
    # is no longer valid as DSG gets param_space directly from the strategy_class.
    # Commenting out as it needs a complete rethink if a similar concern exists.
    #     pass


    def test_generate_new_strategy_with_no_market_data_in_context(self):
        current_context_no_market = {} # No market_data key
        
        default_params = MockStrategy.default_config().default_params
        mock_strategy_param_space = MockStrategy.get_parameter_space(optimizer_type="genetic")
        param_keys = list(mock_strategy_param_space.keys())
        
        default_values_list = []
        for k in param_keys:
            if k in default_params:
                default_values_list.append(default_params[k])
            else: 
                if mock_strategy_param_space[k].get('type') == bool: default_values_list.append(False)
                elif mock_strategy_param_space[k].get('type') == int: default_values_list.append(0)
                elif mock_strategy_param_space[k].get('type') == float: default_values_list.append(0.0)
                else: default_values_list.append(None)

        default_values_tuple = tuple(default_values_list)
        
        self.mock_genetic_optimizer_instance.run_optimizer.return_value = (default_values_tuple, 5.0)
        self.mock_genetic_optimizer_class.reset_mock()
        self.mock_genetic_optimizer_instance.reset_mock()
        self.mock_genetic_optimizer_instance.run_optimizer.return_value = (default_values_tuple, 5.0)
        self.mock_genetic_optimizer_class.return_value = self.mock_genetic_optimizer_instance
        self.mock_fitness_function_for_dsg.reset_mock()

        market_data_for_ga_arg = None 

        strategy = self.dsg.generate_new_strategy(
            strategy_class=MockStrategy,
            fitness_function=self.mock_fitness_function_for_dsg, 
            current_context=current_context_no_market,
            market_data_for_ga=market_data_for_ga_arg 
        )
        self.assertIsNotNone(strategy)
        self.assertIsInstance(strategy, MockStrategy)
        self.mock_genetic_optimizer_instance.run_optimizer.assert_called_once()

        # Test the fitness function call chain
        self.mock_genetic_optimizer_class.assert_called_once() 
        constructor_args, constructor_kwargs = self.mock_genetic_optimizer_class.call_args
        
        ga_fitness_wrapper_passed_to_optimizer = constructor_kwargs.get('fitness_function_callback')
        self.assertIsNotNone(ga_fitness_wrapper_passed_to_optimizer)

        params_dict_for_ga_instance = dict(zip(param_keys, default_values_tuple))
        
        base_config_for_ga = constructor_kwargs.get('base_config', MockStrategy.default_config())
        mock_ga_logger = MagicMock(spec=logging.Logger)

        strategy_instance_for_ga_evaluation = MockStrategy(
            config=base_config_for_ga,
            params=params_dict_for_ga_instance, 
            logger=mock_ga_logger
        )

        ga_fitness_wrapper_passed_to_optimizer(
            strategy_instance_for_ga_evaluation, 
            market_data_for_ga_arg,       
            current_context_no_market,    
            default_values_tuple          
        )
        
        self.mock_fitness_function_for_dsg.assert_called_once_with(
            ANY, # strategy_instance_for_ga_evaluation (use ANY for robustness)
            current_context_no_market,
            market_data_for_ga_arg,
            ANY  # default_values_tuple (use ANY for robustness)
        )


    def test_generate_new_strategy_with_time_series_split_preprocessing(self):
        raw_data = pd.DataFrame({'close': np.random.rand(100) * 100}, index=pd.date_range(start='2023-01-01', periods=100))
        tscv = TimeSeriesSplit(n_splits=4, test_size=20, gap=0)
        train_index, _ = list(tscv.split(raw_data))[-1]
        train_data_df = raw_data.iloc[train_index]
        
        current_context_with_split_data = {"market_data": {'asset1': train_data_df}}
        market_data_for_ga_arg = {'asset1': train_data_df} 
        
        optimized_params_dict = MockStrategy.default_config().default_params
        mock_strategy_param_space = MockStrategy.get_parameter_space(optimizer_type="genetic")
        param_keys = list(mock_strategy_param_space.keys())
        # Ensure all keys from param_keys are in optimized_params_dict
        # This should hold if param_definitions and default_config are consistent
        for k in param_keys:
            self.assertIn(k, optimized_params_dict, f"Key {k} from param_space not in default_params.")
        optimized_params_tuple = tuple(optimized_params_dict[k] for k in param_keys)
        
        self.mock_genetic_optimizer_instance.run_optimizer.return_value = (optimized_params_tuple, 10.0)
        self.mock_genetic_optimizer_class.reset_mock()
        self.mock_genetic_optimizer_instance.reset_mock()
        self.mock_genetic_optimizer_instance.run_optimizer.return_value = (optimized_params_tuple, 10.0)
        self.mock_genetic_optimizer_class.return_value = self.mock_genetic_optimizer_instance
        self.mock_fitness_function_for_dsg.reset_mock()
        
        strategy = self.dsg.generate_new_strategy(
            strategy_class=MockStrategy,
            fitness_function=self.mock_fitness_function_for_dsg,
            current_context=current_context_with_split_data,
            market_data_for_ga=market_data_for_ga_arg
        )
        self.assertIsNotNone(strategy)
        self.assertIsInstance(strategy, MockStrategy)
        self.mock_genetic_optimizer_instance.run_optimizer.assert_called_once()

        # Test the fitness function call chain
        self.mock_genetic_optimizer_class.assert_called_once()
        constructor_args, constructor_kwargs = self.mock_genetic_optimizer_class.call_args

        ga_fitness_wrapper_passed_to_optimizer = constructor_kwargs.get('fitness_function_callback')
        self.assertIsNotNone(ga_fitness_wrapper_passed_to_optimizer)

        # Params GA would be testing (optimized_params_tuple in this case)
        # optimized_params_dict is already available and corresponds to optimized_params_tuple
        
        base_config_for_ga = constructor_kwargs.get('base_config', MockStrategy.default_config())
        mock_ga_logger = MagicMock(spec=logging.Logger)

        strategy_instance_for_ga_evaluation = MockStrategy(
            config=base_config_for_ga,
            params=optimized_params_dict, # Use the dict form for strategy instantiation
            logger=mock_ga_logger
        )

        ga_fitness_wrapper_passed_to_optimizer(
            strategy_instance_for_ga_evaluation,
            market_data_for_ga_arg,
            current_context_with_split_data,
            optimized_params_tuple # raw_params_from_ga must be tuple
        )
        
        self.mock_fitness_function_for_dsg.assert_called_once_with(
            ANY, # strategy_instance_for_ga_evaluation
            current_context_with_split_data, 
            market_data_for_ga_arg, 
            ANY # optimized_params_tuple
        )

    def test_generate_new_strategy_with_strategy_that_raises_in_init(self):
        class FaultyInitStrategy(BaseStrategy):
            identifier = "FaultyInitStrategy" # Add identifier
            version = "1.0" # Add version
            @staticmethod
            def default_config(): return StrategyConfig(name="FaultyInit", default_params={})
            # No param_definitions, so get_parameter_space would be empty. Opto won't run.
            # Or add param_definitions if we want to test optimizer path before init failure.
            # For this test, let's assume no optimizer, focusing on init error.
            @staticmethod
            def param_definitions() -> Dict[str, Dict[str, Any]]: # Add to avoid issues if optimizer path is taken
                return {}

            def __init__(self, config, params=None, logger: Optional[logging.Logger] = None): # Adjusted signature to match BaseStrategy and accept logger
                # super().__init__(config, params, logger=logger) # Don't call super if we want to ensure our error
                # Ensure this logger is used if super() is not called, or pass to super if it is.
                # For this test, we are not calling super to isolate the init error.
                self.logger = logger if logger else logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
                raise ValueError("Intentional init error")

            # Add dummy implementations for abstract methods to make the class non-abstract
            def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
                pass

            def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
                pass

        # No need to modify self.dsg.available_strategies_classes or strategy_param_spaces

        with self.assertLogs(logger=self.logger.name, level='ERROR') as log:
            strategy = self.dsg.generate_new_strategy(
                strategy_class=FaultyInitStrategy,
                fitness_function=None, # No fitness function, so no optimization, direct instantiation
                current_context={}
            )
        self.assertIsNone(strategy) 
        log_output_joined = "".join(log.output)
        self.assertIn("Failed to instantiate strategy FaultyInitStrategy", log_output_joined)
        self.assertIn("Intentional init error", log_output_joined)
        
        # Test with optimizer path, where instantiation happens inside GA wrapper or after GA
        # If GA runs, it will try to instantiate.
        # Let's assume FaultyInitStrategy.get_parameter_space() returns something to trigger GA
        # For simplicity, the above test with fitness_function=None is clearer for init failure.