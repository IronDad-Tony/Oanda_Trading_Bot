import unittest
from unittest.mock import MagicMock, patch # Ensure MagicMock is imported
import numpy as np
import pandas as pd # GeneticOptimizer uses pd.DataFrame for historical_data
import sys # Added for path manipulation
import os # Added for path manipulation
from typing import Dict, Any, Optional # Added Optional, Dict, Any
import logging # Added for logger in tests

# Add src directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Assuming GeneticOptimizer is in src.agent.optimizers.genetic_optimizer
# Adjust the import path based on your project structure and how you run tests
# For example, if tests/ is a top-level directory and src/ is a sibling:
from src.agent.optimizers.genetic_optimizer import GeneticOptimizer
from src.agent.strategies.base_strategy import BaseStrategy, StrategyConfig # For MockStrategy

# Mock Strategy for testing
class MockStrategy(BaseStrategy):
    identifier = "MockStrategy"
    version = "1.0"

    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="MockStrategy", 
            description="A mock strategy for testing.",
            default_params={'param_A': 10, 'param_B': 0.5} # Ensure these defaults exist
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, market_data_source=None, risk_manager=None, portfolio_manager=None):
        super().__init__(config, params)

    def _calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        idx = data.index if data is not None else pd.DatetimeIndex([])
        if data is None or data.empty:
            return pd.Series(np.zeros(len(idx)), index=idx, dtype=float)

        # Use params for deterministic output
        param_a_val = self.params.get('param_A', 0.0) # Default to float if not found
        param_b_val = self.params.get('param_B', 0.0) # Default to float if not found
        
        signal_val = float(param_a_val + param_b_val) # Ensure float
        return pd.Series([signal_val] * len(data), index=data.index)

    def forward(self, data: pd.DataFrame, context: dict = None) -> pd.DataFrame:
        idx = data.index if data is not None and not data.empty else pd.DatetimeIndex([])
        
        signals = self._calculate_signals(data) # This will handle None/empty data

        if data is None or data.empty:
            values = pd.Series(np.zeros(len(idx)), index=idx, dtype=float)
        else:
            # Ensure 'value' is also somewhat deterministic or at least consistently shaped for tests
            values = pd.Series(np.arange(len(data), dtype=float) / len(data) if len(data) > 0 else [], index=idx) 
        
        output = pd.DataFrame({'signal': signals, 'value': values}, index=idx)
        return output

    def generate_signals(self, data: pd.DataFrame, context: dict = None) -> pd.DataFrame:
        # In BaseStrategy, generate_signals takes processed_data_dict.
        # For this mock, assume data is the processed DataFrame for the primary asset.
        return self.forward(data, context)

# Mock fitness function for testing
def mock_fitness_function(strategy_instance: BaseStrategy, market_data: pd.DataFrame, portfolio_context: dict, individual_params: dict) -> float:
    """
    Mock fitness function that simulates fitness calculation.
    Uses the strategy's default config and applies individual_params to it.
    """
    try:
        # We expect strategy_instance to be already configured with individual_params
        # So, we directly use it.
        signals_df = strategy_instance.forward(market_data, portfolio_context) # Corrected variable name
        # print(f"mock_fitness_function: Strategy: {type(strategy_instance).__name__}, Params: {strategy_instance.params}, Signals: {signals_df.shape if signals_df is not None else 'None'}")
        
        # If market_data was None or empty, signals_df might be an empty DataFrame.
        # In this case, fitness should be 0.0, not -inf.
        if signals_df is None: # Should not happen if forward always returns a DF
            return 0.0 
        
        if signals_df.empty: # This handles cases where no data led to an empty (0-row) signals DataFrame
            # print(f"mock_fitness_function: Empty signals DataFrame for {type(strategy_instance).__name__} with params {strategy_instance.params}. Returning 0.0 fitness.")
            return 0.0

        if 'signal' in signals_df.columns:
            fitness = signals_df['signal'].sum()
            # print(f"Calculated fitness: {fitness}")
            return float(fitness)
        else:
            # print("Warning: 'signal' column not in signals DataFrame from strategy.")
            return -float('inf') # Penalize if 'signal' column is missing and DF is not empty
    except Exception as e:
        # print(f"Error during mock_fitness_function execution with {type(strategy_instance).__name__} and params {strategy_instance.params}: {e}")
        # import traceback
        # traceback.print_exc()
        return -float('inf')


class TestGeneticOptimizer(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("TestGeneticOptimizer")
        self.logger.setLevel(logging.DEBUG) # Or CRITICAL to suppress

        self.strategy_class = MockStrategy # Defined above test class
        
        # MODIFIED: param_space uses actual types, not strings, and matches test expectations
        self.param_space = {
            'param_int': (int, (1, 10)),
            'param_float': (float, (0.0, 1.0)),
            'param_cat_str': (str, ['A', 'B', 'C']),
            'param_bool': (bool, [True, False]) # PyGAD uses 0/1 for bool if not specified otherwise
        }
        self.param_types_expected = {
            'param_int': int,
            'param_float': float,
            'param_cat_str': str,
            'param_bool': bool
        }
        self.historical_data = pd.DataFrame({
            'open': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100 + 100,
            'low': np.random.rand(100) * 100 - 50,
            'close': np.random.rand(100) * 100,
            'volume': np.random.rand(100) * 1000
        }, index=pd.date_range(start='2023-01-01', periods=100))

        self.mock_portfolio_context = {'balance': 100000, 'instrument': 'EUR_USD'}

        # Define self.optimizer_config before it's used in ga_settings_for_test
        self.optimizer_config = {
            'population_size': 20,
            'n_generations': 10,
            'mutation_rate': 0.1,  # For mutation_percent_genes (0-100)
            'crossover_rate': 0.7, # For crossover_probability (0-1)
            'tournament_size': 3,
            'early_stopping_generations': 5
        }

        # These are the settings directly passed to PyGAD, derived from optimizer_config
        self.ga_settings_for_test = {
            'sol_per_pop': self.optimizer_config['population_size'],
            'num_generations': self.optimizer_config['n_generations'],
            'mutation_type': 'random', # Default or from config
            'mutation_percent_genes': self.optimizer_config['mutation_rate'] * 100,
            'crossover_type': 'single_point', # Default or from config
            'crossover_probability': self.optimizer_config['crossover_rate'],
            'parent_selection_type': 'sss', # Default or from config
            'K_tournament': self.optimizer_config['tournament_size'],
            'stop_criteria': f"saturate_{self.optimizer_config.get('early_stopping_generations', 5)}",
            'random_seed': self.optimizer_config.get('random_seed'),
            'allow_duplicate_genes': True # A common PyGAD default/recommendation
        }
        
        self.mock_fitness_function_for_optimizer_test = MagicMock(return_value=10.0)

        # Context to be passed to the fitness function by the optimizer
        self.base_context_for_fitness = {
            'market_data': self.historical_data, 
            'portfolio_context': self.mock_portfolio_context,
            'strategy_class_ref': self.strategy_class 
        }

        self.optimizer = GeneticOptimizer(
            fitness_function=self.mock_fitness_function_for_optimizer_test,
            param_space=self.param_space,
            base_config=self.base_context_for_fitness, # This context is for the fitness_function
            logger=self.logger,
            ga_settings=self.ga_settings_for_test
        )
        # self.optimizer.strategy_class = MockStrategyForOptimizer # No longer needed, strategy class is not part of GA init
        self.mock_market_data = pd.DataFrame({
            'open': np.random.rand(10), 'high': np.random.rand(10),
            'low': np.random.rand(10), 'close': np.random.rand(10),
            'volume': np.random.rand(10) * 100
        })
        self.empty_mock_market_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    def test_optimizer_initialization(self):
        self.assertIsNotNone(self.optimizer)
        self.assertEqual(self.optimizer.ga_settings['sol_per_pop'], self.ga_settings_for_test['sol_per_pop'])
        self.assertEqual(self.optimizer.ga_settings['num_generations'], self.ga_settings_for_test['num_generations'])
        self.assertEqual(self.optimizer.ga_settings['mutation_percent_genes'], self.ga_settings_for_test['mutation_percent_genes'])
        self.assertEqual(self.optimizer.ga_settings['crossover_probability'], self.ga_settings_for_test['crossover_probability'])
        self.assertEqual(self.optimizer.ga_settings['K_tournament'], self.ga_settings_for_test['K_tournament'])
        self.assertEqual(self.optimizer.ga_settings['stop_criteria'], self.ga_settings_for_test['stop_criteria'])
        self.assertEqual(self.optimizer.ga_settings['random_seed'], self.ga_settings_for_test['random_seed'])
        
        # Check derived gene_space and gene_type (simplified check)
        self.assertEqual(len(self.optimizer.gene_space), len(self.param_space))
        self.assertEqual(len(self.optimizer.gene_type), len(self.param_space))
        self.assertIn('param_int', self.optimizer.param_names)
        self.assertIn('param_float', self.optimizer.param_names)
        self.assertIn('param_cat_str', self.optimizer.param_names)
        self.assertIn('param_bool', self.optimizer.param_names)

        # Check types in gene_type (order should match param_names)
        idx_p_int = self.optimizer.param_names.index('param_int')
        idx_p_float = self.optimizer.param_names.index('param_float')
        idx_p_cat_str = self.optimizer.param_names.index('param_cat_str')
        idx_p_bool = self.optimizer.param_names.index('param_bool')

        self.assertEqual(self.optimizer.gene_type[idx_p_int], int)
        self.assertEqual(self.optimizer.gene_type[idx_p_float], float)
        self.assertEqual(self.optimizer.gene_type[idx_p_cat_str], str)
        self.assertEqual(self.optimizer.gene_type[idx_p_bool], int) # bool is handled as int (0/1)
    
    def test_param_space_preparation(self):
        # Check that the param_space is prepared correctly for the optimizer
        expected_param_space = {
            'param_int': (int, (1, 10)),
            'param_float': (float, (0.0, 1.0)),
            'param_cat_str': (str, ['A', 'B', 'C']),
            'param_bool': (bool, [True, False])
        }
        self.assertEqual(self.optimizer.param_space, expected_param_space)

    def test_calculate_fitness_wrapper_logic(self):
        # This tests the _calculate_fitness wrapper inside GeneticOptimizer
        # It should correctly call the user-provided fitness function with decoded params and context
        
        # Example solution from GA (raw gene values)
        # param_space = {
        #     'param_int': (int, (1, 10)),
        #     'param_float': (float, (0.0, 1.0)),
        #     'param_cat_str': (str, ['A', 'B', 'C']),
        #     'param_bool': (bool, [True, False]) # PyGAD uses 0/1
        # }
        # param_names order: ['param_int', 'param_float', 'param_cat_str', 'param_bool'] (depends on dict iteration order in Python 3.7+)
        # To make it deterministic for test, let's assume an order or use the optimizer's internal param_names
        
        # Use the optimizer's actual param_names order
        # Ensure the solution matches the param_space defined in setUp
        solution_map = {'param_int': 3, 'param_float': 0.75, 'param_cat_str': 'B', 'param_bool': 0} # param_bool as 0 for False
        solution = []
        for p_name in self.optimizer.param_names:
            if p_name not in solution_map:
                # This case should ideally not happen if solution_map covers all param_names
                # For robustness, add a default or handle missing key, though test setup should ensure it.
                if self.optimizer.param_space[p_name][0] == bool: # type is bool
                    solution.append(0) # Default to False (0)
                elif self.optimizer.param_space[p_name][0] == str: # type is str
                    solution.append(self.optimizer.param_space[p_name][1][0]) # Default to first categorical option
                else: # int, float
                    solution.append(self.optimizer.param_space[p_name][1][0]) # Default to lower bound
            else:
                solution.append(solution_map[p_name])
        
        solution_idx = 0 
        
        # The context passed to _calculate_fitness is the one given to run_optimizer
        # For this test, let's simulate that context.
        test_run_context = {'some_runtime_data': 'value'}
        
        fitness_value = self.optimizer._calculate_fitness(solution, solution_idx, context=test_run_context)

        self.mock_fitness_function_for_optimizer_test.assert_called_once()
        call_args = self.mock_fitness_function_for_optimizer_test.call_args[0]
        
        params_dict_called = call_args[0]
        context_called = call_args[1]
        
        self.assertEqual(params_dict_called['param_int'], 3)
        self.assertEqual(params_dict_called['param_float'], 0.75)
        self.assertEqual(params_dict_called['param_cat_str'], 'B')
        self.assertEqual(params_dict_called['param_bool'], False) # 0 converted to False
        
        self.assertEqual(context_called, test_run_context) # Context passed to _calculate_fitness is used
        self.assertEqual(fitness_value, 10.0) # Return value of mock

    @patch('pygad.GA') # Mock the pygad.GA class
    def test_evolve_successful_run(self, MockPyGADGA):
        mock_ga_instance = MockPyGADGA.return_value
        
        # Simulate PyGAD's on_generation callback setting best_params_ and best_fitness_
        # This happens internally in the GeneticOptimizer's on_generation method
        # So, we need to ensure that the call to ga_instance.run() leads to on_generation being called
        # and that on_generation correctly decodes and stores the best solution.

        # To simulate this, we can mock the on_generation callback or check its effects.
        # Let's assume on_generation works as intended and PyGAD provides a best solution.
        
        # The best solution from PyGAD (genes)
        # Order based on self.optimizer.param_names
        best_solution_genes_map = {'param_int': 5, 'param_float': 0.5, 'param_cat_str': 'A', 'param_bool': 1} # param_bool as 1 for True
        best_solution_genes = [best_solution_genes_map[p_name] for p_name in self.optimizer.param_names]
        
        mock_ga_instance.best_solution.return_value = (best_solution_genes, 100.0, 0) # solution_genes, fitness, index

        # We also need to ensure that self.optimizer.on_generation is called by PyGAD
        # and correctly updates self.optimizer.best_params_ and self.optimizer.best_fitness_
        # One way is to allow the actual on_generation to run.
        # PyGAD calls on_generation with the ga_instance.
        
        # To make this test simpler and focus on run_optimizer's orchestration:
        # We can directly set the outcome of the GA run as if on_generation populated it.
        # This is because on_generation is tightly coupled with PyGAD's execution flow.
        
        # Let's refine: test that run_optimizer calls pygad.GA, then ga_instance.run(),
        # and returns the best_params_ and best_fitness_ that on_generation (called by PyGAD) would have set.

        # To test on_generation's effect, we can make a dummy call to it after run()
        # or trust that PyGAD calls it. For now, let's assume PyGAD calls it.
        # The GeneticOptimizer's run_optimizer will return self.best_params_ and self.best_fitness_
        # which are set by the on_generation callback.

        # So, if ga_instance.run() completes, and on_generation was correctly configured,
        # self.best_params_ should be populated.

        # Simulate that on_generation has been called and updated the optimizer's state
        # This is a bit of a shortcut for testing run_optimizer's return values
        # A more integrated test would involve PyGAD actually calling on_generation.
        
        # Let's make the mock_ga_instance.run() call the on_generation callback
        # This is tricky because on_generation is a method of self.optimizer.
        def mock_run(*args, **kwargs):
            # Simulate PyGAD calling on_generation at least once with the best solution
            self.optimizer.on_generation(mock_ga_instance)
            return

        mock_ga_instance.run.side_effect = mock_run
        
        test_run_context = {'runtime_key': 'runtime_value'}
        best_params, best_fitness = self.optimizer.run_optimizer(current_context=test_run_context)

        MockPyGADGA.assert_called_once() 
        pygad_call_args = MockPyGADGA.call_args[1] 
        
        self.assertEqual(pygad_call_args['num_generations'], self.ga_settings_for_test['num_generations'])
        self.assertEqual(pygad_call_args['sol_per_pop'], self.ga_settings_for_test['sol_per_pop'])
        self.assertIsNotNone(pygad_call_args['fitness_func']) 
        self.assertEqual(pygad_call_args['gene_space'], self.optimizer.gene_space)
        # gene_type can be a list of types or a single type if all are same.
        # self.optimizer.gene_type is always a list.
        self.assertListEqual(pygad_call_args['gene_type'], self.optimizer.gene_type)
        self.assertEqual(pygad_call_args['on_generation'], self.optimizer.on_generation)


        mock_ga_instance.run.assert_called_once() 
        
        self.assertIsNotNone(best_params)
        self.assertEqual(best_fitness, 100.0)
        self.assertEqual(best_params['param_int'], 5)
        self.assertEqual(best_params['param_float'], 0.5)
        self.assertEqual(best_params['param_cat_str'], 'A')
        self.assertEqual(best_params['param_bool'], True) 

    def test_calculate_fitness_exception_in_user_func(self):
        self.mock_fitness_function_for_optimizer_test.side_effect = ValueError("User fitness error")
        
        solution_map = {'param_int': 1, 'param_float': 0.1, 'param_cat_str': 'A', 'param_bool': 1} # param_bool as 1 for True
        solution = [solution_map[p_name] for p_name in self.optimizer.param_names]
        
        test_run_context = {'ctx_key': 'ctx_val'}

        with patch.object(self.logger, 'error') as mock_log_error:
            fitness = self.optimizer._calculate_fitness(solution, 0, context=test_run_context)
            self.assertEqual(fitness, -float('inf'))
            
            # Construct the expected params dict based on the solution and param_names
            expected_params_dict = {
                'param_int': 1, 
                'param_float': 0.1, 
                'param_cat_str': 'A', 
                'param_bool': True # Decoded from 1
            }
            mock_log_error.assert_called_with(
                f"Error in user-provided fitness_function for params {expected_params_dict}: User fitness error",
                exc_info=True
            )
            
    @patch('pygad.GA')
    def test_evolve_no_context_provided_to_run_optimizer(self, MockPyGADGA):
        # Test that run_optimizer raises an error or handles if current_context is missing,
        # as it's now a required argument.
        with self.assertRaises(TypeError): # run_optimizer() missing 1 required positional argument: 'current_context'
            self.optimizer.run_optimizer()

    @patch('pygad.GA')
    def test_run_optimizer_ga_exception(self, MockPyGADGA):
        MockPyGADGA.side_effect = Exception("PyGAD init failed")
        
        test_run_context = {'data': 'some_data'}
        with patch.object(self.logger, 'error') as mock_log_error:
            params, fitness = self.optimizer.run_optimizer(current_context=test_run_context)
            self.assertEqual(params, {})
            self.assertEqual(fitness, -float('inf'))
            mock_log_error.assert_called_with(
                "Exception during PyGAD GA instantiation or run: PyGAD init failed",
                exc_info=True
            )

    @patch('pygad.GA')
    def test_run_optimizer_no_solution_found(self, MockPyGADGA):
        mock_ga_instance = MockPyGADGA.return_value
        
        # Simulate on_generation never finding a solution better than -inf
        # or PyGAD itself not finding one.
        # If on_generation is called but best_fitness remains -inf, best_params_ remains None.
        def mock_run_no_solution(*args, **kwargs):
            # Simulate on_generation being called, but it doesn't improve best_fitness_ from -inf
            # or best_solution() from PyGAD returns a very bad fitness.
            mock_ga_instance.best_solution.return_value = ([], -float('inf'), 0) # No good solution
            self.optimizer.on_generation(mock_ga_instance) # Manually call to simulate PyGAD's behavior
            return

        mock_ga_instance.run.side_effect = mock_run_no_solution
        
        test_run_context = {'data': 'some_data'}
        with patch.object(self.logger, 'warning') as mock_log_warning:
            params, fitness = self.optimizer.run_optimizer(current_context=test_run_context)
            self.assertEqual(params, {})
            self.assertEqual(fitness, -float('inf'))
            mock_log_warning.assert_any_call( # Updated to assert_any_call due to other potential warnings
                "Genetic optimizer did not find a valid solution during the run. Returning empty params and -inf fitness."
            )
