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
            # Real mock_fitness_function should be compatible with DSG\'s GA wrapper.
            params = strategy_instance.get_params()
            fitness = 0.0
            if 'param_A' in params: fitness += params['param_A']
            if 'param_B' in params: fitness += params['param_B'] * 10
            return fitness

# Import MockStrategyWithParams from test_dynamic_strategy_generator
# try:
#     from .test_dynamic_strategy_generator import MockStrategyWithParams
# except ImportError:
#     from test_dynamic_strategy_generator import MockStrategyWithParams


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

    @classmethod # ADDED
    def get_parameter_space(cls, optimizer_type: str = "genetic") -> Optional[Dict[str, Any]]: # ADDED
        if optimizer_type == "genetic": # ADDED
            space = {} # ADDED
            param_defs = cls.param_definitions() # Use the static method # ADDED
            for name, definition in param_defs.items(): # ADDED
                if definition['type'] == int: # ADDED
                    space[name] = {'type': 'int', 'low': definition.get('min', 1), 'high': definition.get('max', 100)} # ADDED
                elif definition['type'] == float: # ADDED
                    space[name] = {'type': 'float', 'low': definition.get('min', 0.0), 'high': definition.get('max', 1.0)} # ADDED
                elif definition['type'] == bool: # ADDED
                    space[name] = {'type': 'categorical', 'choices': [True, False]} # ADDED
                elif definition['type'] == str and 'choices' in definition: # ADDED
                    space[name] = {'type': 'categorical', 'choices': definition['choices']} # ADDED
            return space # ADDED
        elif optimizer_type == "nas": # ADDED
            return None # Placeholder for NAS space # ADDED
        return None # ADDED

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

    @classmethod # ADDED
    def get_parameter_space(cls, optimizer_type: str = "genetic") -> Optional[Dict[str, Any]]: # ADDED
        if optimizer_type == "genetic": # ADDED
            space = {} # ADDED
            param_defs = cls.param_definitions() # Use the static method # ADDED
            for name, definition in param_defs.items(): # ADDED
                if definition['type'] == int: # ADDED
                    space[name] = {'type': 'int', 'low': definition.get('min', 1), 'high': definition.get('max', 100)} # ADDED
                elif definition['type'] == bool: # ADDED
                    space[name] = {'type': 'categorical', 'choices': [True, False]} # ADDED
            return space # ADDED
        elif optimizer_type == "nas": # ADDED
            return None # Placeholder for NAS space # ADDED
        return None # ADDED

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


class TestDynamicStrategyGenerator(unittest.TestCase): # This is for testing DSG from enhanced_quantum_strategy_layer.py

    def setUp(self):
        self.logger = logging.getLogger('TestDSGFromEQSL') 
        self.logger.setLevel(logging.WARNING) # MODIFIED: Changed level to WARNING to capture warning logs

        self.optimizer_config_for_dsg = {
            "name": "GeneticOptimizer",
            "settings": {
                'population_size': 5, 
                'n_generations': 3,  
                'mutation_rate': 0.1, # Will be multiplied by 100 for PyGAD
                'crossover_rate': 0.7,
                'tournament_size': 2,
            }
        }
        
        # MODIFIED: Initialize DSG with the new signature from enhanced_quantum_strategy_layer.py
        self.dsg = DynamicStrategyGenerator(
            logger=self.logger,
            optimizer_config=self.optimizer_config_for_dsg # Pass optimizer_config here
        )

        self.mock_market_data_dict = {
            'asset1': pd.DataFrame({'close': np.random.rand(50) * 100}, index=pd.date_range(start='2023-01-01', periods=50))
        }
        self.current_context_for_test = {'balance': 10000, 'market_data_dict': self.mock_market_data_dict} # Changed key

        # Patch GeneticOptimizer where it's instantiated (inside enhanced_quantum_strategy_layer)
        self.ga_patcher = patch('src.agent.enhanced_quantum_strategy_layer.GeneticOptimizer', autospec=True)
        self.MockGeneticOptimizer = self.ga_patcher.start()
        self.addCleanup(self.ga_patcher.stop)
        
        self.mock_genetic_optimizer_instance = self.MockGeneticOptimizer.return_value
        self.mock_genetic_optimizer_instance.run_optimizer.return_value = ({}, -float('inf')) 

        self.mock_fitness_function_for_dsg = MagicMock(return_value=1.0)
    
    def test_initialization(self):
        """Test DynamicStrategyGenerator initialization."""
        self.assertIsNotNone(self.dsg.logger)
        self.assertEqual(self.dsg.optimizer_config, self.optimizer_config_for_dsg)
        self.assertIsNone(self.dsg.genetic_optimizer) 

    def test_generate_new_strategy_no_optimizer(self):
        """Test generation when no optimizer should run."""
        original_opt_config = self.dsg.optimizer_config
        self.dsg.optimizer_config = {} # Disable optimizer for this test

        strategy_no_fitness = self.dsg.generate_new_strategy(
            strategy_class=MockStrategy, # Use the one defined in this file
            fitness_function=None, 
            current_context=self.current_context_for_test,
            market_data_for_ga=self.mock_market_data_dict 
        )
        self.assertIsNotNone(strategy_no_fitness)
        self.assertIsInstance(strategy_no_fitness, MockStrategy)
        self.MockGeneticOptimizer.assert_not_called() 
        self.dsg.optimizer_config = original_opt_config # Restore

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
        """Test generation with GeneticOptimizer."""
        # Ensure optimizer_config is set for GA
        self.dsg.optimizer_config = self.optimizer_config_for_dsg 
        
        # Simulate GA finding good parameters
        optimized_params = {'param_A': 100, 'param_B': 'optimized_val'}
        self.mock_genetic_optimizer_instance.run_optimizer.return_value = (optimized_params, 5.0)

        strategy_instance = self.dsg.generate_new_strategy(
            strategy_class=MockStrategy, 
            fitness_function=self.mock_fitness_function_for_dsg,
            current_context=self.current_context_for_test,
            market_data_for_ga=self.mock_market_data_dict
        )

        self.assertIsNotNone(strategy_instance)
        self.assertIsInstance(strategy_instance, MockStrategy)
        self.MockGeneticOptimizer.assert_called_once() # GA class should be instantiated
        
        # Check args passed to GeneticOptimizer constructor
        ga_constructor_args = self.MockGeneticOptimizer.call_args[1] # kwargs
        self.assertIsNotNone(ga_constructor_args.get('fitness_function')) # The wrapper
        self.assertEqual(ga_constructor_args.get('param_space'), MockStrategy.get_parameter_space("genetic"))
        expected_ga_settings = self.optimizer_config_for_dsg.get("settings", {})
        self.assertEqual(ga_constructor_args.get('ga_settings'), expected_ga_settings)
        self.assertEqual(ga_constructor_args.get('logger'), self.logger)
        
        # Check context passed to GA for its fitness function
        expected_base_config_for_ga = {
            'market_data': self.mock_market_data_dict,
            'portfolio_context': self.current_context_for_test
        }
        self.assertEqual(ga_constructor_args.get('base_config'), expected_base_config_for_ga)

        # self.mock_genetic_optimizer_instance.run_optimizer.assert_called_once_with() # No args to run_optimizer
        self.mock_genetic_optimizer_instance.run_optimizer.assert_called_once()
        run_optimizer_call_args = self.mock_genetic_optimizer_instance.run_optimizer.call_args[1] # kwargs
        self.assertIn('current_context', run_optimizer_call_args)
        self.assertEqual(run_optimizer_call_args['current_context'], expected_base_config_for_ga)
        
        # Check if optimized params are applied
        self.assertEqual(strategy_instance.params['param_A'], 100)
        self.assertEqual(strategy_instance.params['param_B'], 'optimized_val')
        # Check if default params not in optimized_params are still there
        self.assertEqual(strategy_instance.params['param_C'], 'default_str')


    def test_generate_new_strategy_optimizer_missing_ga_config(self):
        """Test when optimizer_config is for GA but 'settings' is missing."""
        original_opt_config = self.dsg.optimizer_config
        self.dsg.optimizer_config = {"name": "GeneticOptimizer"} # Missing "settings"

        with patch.object(self.logger, 'warning') as mock_log_warning: # Or error, depending on implementation
            strategy_instance = self.dsg.generate_new_strategy(
                strategy_class=MockStrategy,
                fitness_function=self.mock_fitness_function_for_dsg,
                current_context=self.current_context_for_test,
                market_data_for_ga=self.mock_market_data_dict
            )
            # Strategy should still be created with defaults if optimizer fails to init/run
            self.assertIsNotNone(strategy_instance)
            self.assertEqual(strategy_instance.params, MockStrategy.default_config().default_params)
            # Check that GA was attempted but likely failed gracefully or used empty settings
            self.MockGeneticOptimizer.assert_called_once() 
            # Check that a warning/error was logged about settings
            # The exact log message depends on GeneticOptimizer's robustness to empty settings.
            # For DSG, it passes the settings dict. If it's empty, GA might use defaults.
            # If GeneticOptimizer itself requires specific settings, it might log.
            # Here, we assume DSG passes what it has.
        self.dsg.optimizer_config = original_opt_config # Restore

    @patch('src.agent.enhanced_quantum_strategy_layer.GeneticOptimizer')
    def test_generate_new_strategy_with_initial_params_override(self, mock_genetic_optimizer_class):
        """Optimizer is configured, but initial_parameters are also provided."""
        mock_optimizer_instance = MagicMock(spec=GeneticOptimizer)
        mock_optimizer_instance.run_optimizer.return_value = ({'param_A': 10}, 0.9) # Optimizer returns 10 for param_A
        mock_genetic_optimizer_class.return_value = mock_optimizer_instance

        opt_config_settings = {'generations': 5} # Dummy optimizer settings

        # Strategy defaults param_A to 10 (from MockStrategy.default_config)
        # Initial params set param_A to 50
        # Optimizer returns param_A as 10
        # Expected: Optimizer wins, param_A should be 10
        strategy2 = self.dsg.generate_new_strategy(
            MockStrategy,
            initial_parameters={'param_A': 50},
            optimizer_type_override="genetic", # Corrected argument
            optimizer_settings_override=opt_config_settings, # Corrected argument
            current_context=self.current_context_for_test, # Corrected argument
            market_data_for_ga=self.mock_market_data_dict # Added market data for GA
        )
        self.assertIsInstance(strategy2, MockStrategy)
        self.assertEqual(strategy2.params['param_A'], 10) # Optimizer result (10) should take precedence
        self.assertEqual(strategy2.params['param_B'], 0.5) # Corrected: Unchanged by optimizer or initial_params, should be default 0.5

        mock_genetic_optimizer_class.assert_called_once()
        mock_optimizer_instance.run_optimizer.assert_called_once()
        # Check that initial_parameters were logged as being present alongside optimizer
        # This depends on specific logging implemented in DSG for this scenario.
        # For now, primary check is the parameter value.

    def test_generate_new_strategy_with_base_config_override(self):
        overridden_default_params = {'param_A': 99, 'param_B': 0.99, 'param_C': 'config_override_C'}
        base_override_config_payload = StrategyConfig( # Renamed variable
            name="MockStrategyOverridden", 
            description="Overridden config",
            default_params=overridden_default_params
        )
        current_context = {"market_data": self.mock_market_data_dict}
        
        dsg_no_opt_config = DynamicStrategyGenerator(logger=self.logger, optimizer_config=None)
        self.MockGeneticOptimizer.reset_mock() # MODIFIED: MockGeneticOptimizer instead of mock_genetic_optimizer_class

        strategy = dsg_no_opt_config.generate_new_strategy(
            strategy_class=MockStrategy, 
            fitness_function=self.mock_fitness_function_for_dsg, 
            current_context=current_context,
            strategy_config_override=base_override_config_payload # Use new arg name
        )
        self.MockGeneticOptimizer.assert_not_called() # MODIFIED: MockGeneticOptimizer instead of mock_genetic_optimizer_class


    def test_generate_new_strategy_with_complex_context(self):
        complex_context = {
            "market_regime": "oscillating", "risk_aversion": "high",
            "investment_horizon": "long_term", "market_data": self.mock_market_data_dict
        }
        optimized_params_dict = {'param_A': 75, 'param_B': 0.75, 'param_C': 'option1'}
        
        mock_strategy_param_space = MockStrategy.get_parameter_space(optimizer_type="genetic")
        param_keys = list(mock_strategy_param_space.keys())
        # self.assertEqual(set(optimized_params_dict.keys()), set(param_keys)) # This might be too strict if param_space has more keys than optimized_params
        
        # Ensure optimized_params_dict keys are a subset of param_keys
        for k_opt in optimized_params_dict.keys():
            self.assertIn(k_opt, param_keys, f"Optimized param key '{k_opt}' not in strategy's param space.")

        # Create tuple based on the order in param_keys, using defaults from strategy if not in optimized_params_dict
        # This is what GA would return: a tuple matching the order of param_names derived from param_space
        # For this test, we assume optimized_params_dict contains all and only the params for param_keys
        # If param_keys from get_parameter_space is ['param_A', 'param_B', 'param_C']
        # and optimized_params_dict is {'param_A': 75, 'param_B': 0.75, 'param_C': 'option1'}
        # then optimized_params_tuple should be (75, 0.75, 'option1')
        
        # Reconstruct the tuple in the order of param_keys from the strategy's param_space
        # This ensures the mock GA output matches what the real GA would produce based on param_names
        ordered_param_names_from_strategy = list(MockStrategy.get_parameter_space(optimizer_type="genetic").keys())
        optimized_params_tuple = tuple(optimized_params_dict[k] for k in ordered_param_names_from_strategy if k in optimized_params_dict)
        # This assumes optimized_params_dict has all keys in ordered_param_names_from_strategy.
        # A more robust way if optimized_params_dict might be incomplete:
        # default_vals_for_strategy = MockStrategy.default_config().default_params
        # optimized_params_tuple = tuple(optimized_params_dict.get(k, default_vals_for_strategy.get(k)) for k in ordered_param_names_from_strategy)


        # Simulate optimizer finding these params
        self.mock_genetic_optimizer_instance.run_optimizer.return_value = (optimized_params_dict, 100.0) # GA returns dict
        
        # Reset mocks for this specific test path
        self.MockGeneticOptimizer.reset_mock() # MODIFIED
        self.mock_genetic_optimizer_instance.reset_mock()
        self.mock_genetic_optimizer_instance.run_optimizer.return_value = (optimized_params_dict, 100.0) # GA returns dict
        self.MockGeneticOptimizer.return_value = self.mock_genetic_optimizer_instance # MODIFIED
        self.mock_fitness_function_for_dsg.reset_mock()

        market_data_for_ga_arg = complex_context['market_data']

        strategy = self.dsg.generate_new_strategy(
            strategy_class=MockStrategy,
            fitness_function=self.mock_fitness_function_for_dsg, 
            current_context=complex_context,
            market_data_for_ga=market_data_for_ga_arg
        )
        self.assertIsNotNone(strategy)
        self.assertIsInstance(strategy, MockStrategy)

        # 1. Assert GeneticOptimizer was called and capture its arguments
        self.MockGeneticOptimizer.assert_called_once() # MODIFIED
        constructor_args, constructor_kwargs = self.MockGeneticOptimizer.call_args # MODIFIED
        
        # 2. Extract the ga_fitness_wrapper passed to the GeneticOptimizer
        ga_fitness_wrapper_passed_to_optimizer = constructor_kwargs.get('fitness_function') # MODIFIED: Corrected key
        self.assertIsNotNone(ga_fitness_wrapper_passed_to_optimizer)

        # 3. Call the captured ga_fitness_wrapper with mock data to simulate GA's internal call
        #    The strategy instance for GA would be created by GA itself using the trial params.
        #    We need a mock strategy instance configured with the trial parameters.
        #    The `optimized_params_dict` are what the GA *would have found*.
        #    Let's simulate the GA trying out these `optimized_params_dict`.
        
        # Create a dummy strategy instance as GA would, using the params GA is testing.
        # The config for this instance would be the one DSG prepared for the optimizer.
        # MODIFIED: Define mock_ga_logger before use
        mock_ga_logger = MagicMock(spec=logging.Logger)
        default_config_for_mock_strategy = MockStrategy.default_config()
        # The logger passed to this instance inside GA would be the GA's logger.
        # For this test, we can use self.logger or a new mock logger.
        
        # Simulate the strategy instance that GA's _calculate_fitness would create and pass
        # It would use the `base_config` given to GA and the `individual_params` (optimized_params_dict here)
        strategy_instance_for_ga_evaluation = MockStrategy(
            config=default_config_for_mock_strategy, # Use a proper StrategyConfig object
            params=optimized_params_dict, 
            logger=mock_ga_logger
        )

        # This is the context that DSG's wrapper will receive from the GA
        # It's the 'base_config' that was passed to the GA constructor
        expected_base_config_for_ga = constructor_kwargs.get('base_config')

        # Call the captured ga_fitness_wrapper with mock data to simulate GA's internal call
        # The GA calls this wrapper with: params_to_evaluate, context_for_optimizer_fitness
        ga_fitness_wrapper_passed_to_optimizer(
            optimized_params_dict,      # These are the params the GA is currently evaluating
            expected_base_config_for_ga # This is the context (market_data, portfolio_context) for the fitness function
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
        with self.assertLogs(logger=self.dsg.logger.name, level='ERROR') as log: # Use dsg.logger.name
            strategy = self.dsg.generate_new_strategy(
                strategy_class=MockStrategy,
                fitness_function=self.mock_fitness_function_for_dsg,
                current_context="invalid_context_type" # type: ignore
            )
        self.assertIsNone(strategy)
        self.assertIn("Invalid current_context type: <class \'str\'>. Expected Dict or None.", "".join(log.output))

    def test_generate_new_strategy_with_extra_strategy_params_from_optimizer(self):
        initial_params_with_extra = {'param_A': 130, 'extra_param': 999}
        current_context = {"market_data": self.mock_market_data_dict}

        dsg_no_opt_config = DynamicStrategyGenerator(logger=self.logger, optimizer_config=None)
        
        # Logs will be emitted by the logger passed from DSG to BaseStrategy, which is self.logger (TestDSG)
        # BaseStrategy itself logs warnings for unknown params during its __init__.
        # The logger in BaseStrategy is derived from the one passed to DSG, or a default one.
        # We need to ensure the logger being asserted is the one used by BaseStrategy.
        # If DSG passes its own logger to BaseStrategy, then self.logger.name is correct.
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
        
        default_params_dict = MockStrategy.default_config().default_params
        # Optimized params (mocked GA output) should be a dict
        # For this test, let's say GA returns the default params
        optimized_ga_output_params = default_params_dict.copy()
        self.mock_genetic_optimizer_instance.run_optimizer.return_value = (optimized_ga_output_params, 1.0) # GA returns defaults

        strategy_no_market_data = self.dsg.generate_new_strategy(
            strategy_class=MockStrategy,
            fitness_function=self.mock_fitness_function_for_dsg,
            current_context=current_context_no_market,
            market_data_for_ga=self.mock_market_data_dict # market data provided here
        )

        self.assertIsNotNone(strategy_no_market_data)
        self.assertIsInstance(strategy_no_market_data, MockStrategy)
        self.assertEqual(strategy_no_market_data.params, default_params_dict) # Should be set to defaults if no market data in context

        # Check that GeneticOptimizer was called with the correct context
        self.MockGeneticOptimizer.assert_called_once()
        ga_constructor_args = self.MockGeneticOptimizer.call_args[1] # kwargs
        self.assertIn('base_config', ga_constructor_args)
        self.assertEqual(ga_constructor_args['base_config']['market_data'], self.mock_market_data_dict) # market data passed here
        self.assertEqual(ga_constructor_args['base_config']['portfolio_context'], current_context_no_market) # Empty context passed
