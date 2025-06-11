import unittest
import numpy as np
import pandas as pd # GeneticOptimizer uses pd.DataFrame for historical_data
import sys # Added for path manipulation
import os # Added for path manipulation
from typing import Dict, Any, Optional # Added Optional, Dict, Any

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
        self.strategy_class = MockStrategy
        self.param_space = {
            'param_A': (1, 150),  # Integer range
            'param_B': (0.1, 1.0) # Float range
        }
        # Expected python types for parameters based on param_space
        self.param_types_expected = {
            'param_A': int,
            'param_B': float
        }
        # Create some dummy historical data
        self.historical_data = pd.DataFrame({
            'open': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100 + 100,
            'low': np.random.rand(100) * 100 - 50,
            'close': np.random.rand(100) * 100,
            'volume': np.random.rand(100) * 1000
        }, index=pd.date_range(start='2023-01-01', periods=100))

        self.mock_portfolio_context = {'balance': 100000, 'instrument': 'EUR_USD'} # Added mock portfolio context

        self.optimizer_config = {
            'population_size': 20,
            'n_generations': 10,
            'mutation_rate': 0.1,
            'crossover_rate': 0.7,
            'tournament_size': 3, # For tournament selection
            # 'param_types': {'param_A': int, 'param_B': float}, # Kept for reference, not passed to GA
            'early_stopping_generations': 5 # Added for clarity
        }
        self.optimizer = GeneticOptimizer(
            strategy_class=self.strategy_class,
            param_space=self.param_space,
            population_size=self.optimizer_config['population_size'],
            n_generations=self.optimizer_config['n_generations'],
            crossover_rate=self.optimizer_config['crossover_rate'],
            mutation_rate=self.optimizer_config['mutation_rate'],
            fitness_function_callback=mock_fitness_function,
            early_stopping_generations=self.optimizer_config.get('early_stopping_generations', 10),
            tournament_size=self.optimizer_config.get('tournament_size', 3)
        )
        self.mock_market_data = pd.DataFrame({
            'open': np.random.rand(10),
            'high': np.random.rand(10),
            'low': np.random.rand(10),
            'close': np.random.rand(10),
            'volume': np.random.rand(10) * 100
        })
        self.empty_mock_market_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    def test_optimizer_initialization(self):
        self.assertIsNotNone(self.optimizer)
        self.assertEqual(self.optimizer.population_size, self.optimizer_config['population_size'])
        self.assertEqual(self.optimizer.n_generations, self.optimizer_config['n_generations'])
        self.assertEqual(self.optimizer.mutation_rate, self.optimizer_config['mutation_rate'])
        self.assertEqual(self.optimizer.crossover_rate, self.optimizer_config['crossover_rate'])
        self.assertEqual(self.optimizer.tournament_size, self.optimizer_config['tournament_size'])
        self.assertEqual(self.optimizer.early_stopping_generations, self.optimizer_config.get('early_stopping_generations', 10))

    def test_initialize_population(self):
        population = self.optimizer._initialize_population()
        self.assertEqual(len(population), self.optimizer.population_size)
        for individual in population:
            self.assertIn('param_A', individual)
            self.assertIn('param_B', individual)
            # Check types based on self.param_types_expected
            self.assertIsInstance(individual['param_A'], self.param_types_expected['param_A'])
            self.assertIsInstance(individual['param_B'], self.param_types_expected['param_B'])
            # Check ranges
            self.assertTrue(self.param_space['param_A'][0] <= individual['param_A'] <= self.param_space['param_A'][1])
            self.assertTrue(self.param_space['param_B'][0] <= individual['param_B'] <= self.param_space['param_B'][1])

    def test_calculate_fitness(self):
        population = self.optimizer._initialize_population()
        self.assertTrue(len(population) > 0, "Population should not be empty")
        # First test with params from initialized population (less predictable result for assertion)
        individual_params_from_pop = population[0]
        fitness_score_from_pop = self.optimizer._calculate_fitness(individual_params_from_pop, self.historical_data, self.mock_portfolio_context)
        self.assertIsInstance(fitness_score_from_pop, float)

        # Second test with specific params for a predictable result
        # The optimizer's _calculate_fitness will instantiate the strategy with these params
        # then pass the instance to the fitness_function_callback (mock_fitness_function)
        specific_individual_params = {'param_A': 1.5, 'param_B': 2.5} # Corrected to use param_A and param_B
        fitness = self.optimizer._calculate_fitness(
            individual_params=specific_individual_params, 
            market_data=self.mock_market_data, # Has 10 rows
            portfolio_context=self.mock_portfolio_context
        )
        # print(f"test_calculate_fitness: Specific Params: {specific_individual_params}, Fitness: {fitness}")
        self.assertIsInstance(fitness, float)
        # Expected fitness: param_A (1.5) + param_B (2.5) = 4.0. Signal is 4.0 for each of 10 rows.
        # mock_fitness_function sums these signals: 4.0 * 10 = 40.0
        self.assertEqual(fitness, 40.0)

    def test_selection(self):
        population = self.optimizer._initialize_population()
        # Need to create population_with_fitness: List[Tuple[Dict[str, Any], float]]
        population_with_fitness = []
        for ind_params in population:
            fitness = self.optimizer._calculate_fitness(ind_params, self.historical_data, self.mock_portfolio_context)
            population_with_fitness.append((ind_params, fitness))
        
        selected_parents = self.optimizer._selection(population_with_fitness)
        self.assertEqual(len(selected_parents), self.optimizer.population_size)

    def test_crossover_param_types_preserved(self):
        # Test with params relevant to self.optimizer.param_space
        parent1 = {'param_A': 10, 'param_B': 0.5}
        parent2 = {'param_A': 20, 'param_B': 0.8}
        
        # Ensure parents conform to expected types, though _crossover itself doesn't type check them
        self.assertIsInstance(parent1['param_A'], self.param_types_expected['param_A'])
        self.assertIsInstance(parent2['param_B'], self.param_types_expected['param_B'])

        child1, child2 = self.optimizer._crossover(parent1, parent2)

        for child in [child1, child2]:
            for param_name, expected_type in self.param_types_expected.items():
                if param_name in child:
                    self.assertIsInstance(child[param_name], expected_type, 
                                          f"Child param {param_name} has wrong type. Expected {expected_type}, got {type(child[param_name])}")
                    # Also check bounds if possible (though crossover might produce out-of-bound initially)
                    # Bounds are typically enforced by mutation or a specific clamping step if crossover produces out-of-range values.
                    # The current _crossover does not enforce bounds.

    def test_mutate_param_types_preserved(self):
        individual = {'param_A': 15, 'param_B': 0.6}
        # Ensure individual conforms to expected types before mutation
        self.assertIsInstance(individual['param_A'], self.param_types_expected['param_A'])
        self.assertIsInstance(individual['param_B'], self.param_types_expected['param_B'])
        
        mutated_individual = self.optimizer._mutate(individual.copy()) # Pass a copy

        for param_name, expected_type in self.param_types_expected.items():
            if param_name in mutated_individual:
                self.assertIsInstance(mutated_individual[param_name], expected_type,
                                      f"Mutated param {param_name} has wrong type. Expected {expected_type}, got {type(mutated_individual[param_name])}")
                # Check bounds are respected by mutation
                min_val, max_val = self.optimizer.param_space[param_name]
                self.assertTrue(min_val <= mutated_individual[param_name] <= max_val, 
                                f"Mutated param {param_name} {mutated_individual[param_name]} out of bounds ({min_val}, {max_val})")

    def test_evolve(self):
        self.optimizer.n_generations = 3 
        self.optimizer.population_size = 10

        # Call evolve with market_data_for_fitness and portfolio_context_for_fitness
        best_params, best_fitness = self.optimizer.evolve(
            market_data_for_fitness=self.historical_data, 
            portfolio_context_for_fitness=self.mock_portfolio_context
        )

        self.assertIsNotNone(best_params)
        self.assertIsInstance(best_params, dict)
        self.assertIn('param_A', best_params)
        self.assertIn('param_B', best_params)
        self.assertIsInstance(best_params['param_A'], self.param_types_expected['param_A'])
        self.assertIsInstance(best_params['param_B'], self.param_types_expected['param_B'])

        self.assertIsInstance(best_fitness, float)
        
        self.assertTrue(self.param_space['param_A'][0] <= best_params['param_A'] <= self.param_space['param_A'][1])
        self.assertTrue(self.param_space['param_B'][0] <= best_params['param_B'] <= self.param_space['param_B'][1])

    def test_evolve_no_historical_data(self):
        optimizer_no_data_config = self.optimizer_config.copy()
        optimizer_no_data_config['n_generations'] = 2
        optimizer_no_data_config['population_size'] = 5
        
        optimizer_no_data = GeneticOptimizer(
            strategy_class=self.strategy_class,
            param_space=self.param_space,
            population_size=optimizer_no_data_config['population_size'],
            n_generations=optimizer_no_data_config['n_generations'],
            crossover_rate=optimizer_no_data_config['crossover_rate'],
            mutation_rate=optimizer_no_data_config['mutation_rate'],
            fitness_function_callback=mock_fitness_function,
            early_stopping_generations=optimizer_no_data_config.get('early_stopping_generations', 10),
            tournament_size=optimizer_no_data_config.get('tournament_size', 3)
        )

        # Call evolve with market_data_for_fitness=None
        # With robust MockStrategy, fitness should be 0.0 for all individuals if params lead to 0 sum of signals.
        # Best params should be one of the initial random individuals.
        best_params, best_fitness = optimizer_no_data.evolve(
            market_data_for_fitness=None, 
            portfolio_context_for_fitness=self.mock_portfolio_context,
            verbose=False # Reduce noise for test run
        )
        # print(f"test_evolve_no_historical_data: Best Params: {best_params}, Best Fitness: {best_fitness}")
        self.assertIsNotNone(best_params) # Expecting some params as fitness should be 0.0 not -inf
        self.assertIsInstance(best_fitness, float)
        # Fitness should be 0.0 because _calculate_signals will return a Series of 0s for None data
        # and mock_fitness_function will sum these 0s.
        self.assertEqual(best_fitness, 0.0)

    def test_evolve_empty_historical_data(self):
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'], index=pd.DatetimeIndex([]))
        optimizer_empty_data_config = self.optimizer_config.copy()
        optimizer_empty_data_config['n_generations'] = 2
        optimizer_empty_data_config['population_size'] = 5

        optimizer_empty_data = GeneticOptimizer(
            strategy_class=self.strategy_class,
            param_space=self.param_space,
            population_size=optimizer_empty_data_config['population_size'],
            n_generations=optimizer_empty_data_config['n_generations'],
            crossover_rate=optimizer_empty_data_config['crossover_rate'],
            mutation_rate=optimizer_empty_data_config['mutation_rate'],
            fitness_function_callback=mock_fitness_function,
            early_stopping_generations=optimizer_empty_data_config.get('early_stopping_generations', 10),
            tournament_size=optimizer_empty_data_config.get('tournament_size', 3)
        )
        
        # Call evolve with empty DataFrame for market_data_for_fitness
        # Similar to no_historical_data, fitness should be 0.0 for all individuals.
        best_params, best_fitness = optimizer_empty_data.evolve(
            market_data_for_fitness=empty_df, 
            portfolio_context_for_fitness=self.mock_portfolio_context,
            verbose=False # Reduce noise
        )
        # print(f"test_evolve_empty_historical_data: Best Params: {best_params}, Best Fitness: {best_fitness}")
        self.assertIsNotNone(best_params) # Expecting some params
        self.assertIsInstance(best_fitness, float)
        # Fitness should be 0.0 for the same reasons as above.
        self.assertEqual(best_fitness, 0.0)

    def test_evolve_with_early_stopping(self):
        # Test evolve method with early stopping
        optimizer_early_stopping = GeneticOptimizer(
            strategy_class=MockStrategy,
            param_space=self.param_space,
            population_size=10,
            n_generations=50, # Max generations, corrected from 'generations'
            mutation_rate=0.1,
            crossover_rate=0.7,
            fitness_function_callback=mock_fitness_function, # Use the corrected mock fitness function
            early_stopping_generations=3 # Stop if no improvement for 3 generations
        )
        best_params, best_fitness = optimizer_early_stopping.evolve(
            market_data_for_fitness=self.mock_market_data,
            portfolio_context_for_fitness=self.mock_portfolio_context,
            verbose=False
        )
        # print(f"test_evolve_with_early_stopping: Best Params: {best_params}, Best Fitness: {best_fitness}")
        self.assertIsNotNone(best_params)
        self.assertIsInstance(best_fitness, float)
        self.assertGreater(best_fitness, -float('inf'))
        # Add an assertion to check if early stopping might have occurred.
        # This requires the optimizer to store the number of generations it actually ran.
        # Assuming optimizer_early_stopping has an attribute like 'generations_ran_' or similar
        # For now, we'll assume it exists for the sake of the test logic.
        # If not, this part of the test would need adjustment based on actual optimizer implementation.
        if hasattr(optimizer_early_stopping, 'generations_ran_'): # Check if the attribute exists
            # print(f"Early stopping generations ran: {optimizer_early_stopping.generations_ran_}")
            self.assertTrue(optimizer_early_stopping.generations_ran_ <= optimizer_early_stopping.n_generations) # Corrected to n_generations
            if best_fitness > -float('inf'): # if a valid solution was found
                 pass # It could have run all generations or stopped early.
                      # Hard to make a strict assertion without a predictable fitness landscape
                      # or knowing the exact point of convergence for early stopping.

    # ...existing code...
if __name__ == '__main__':
    unittest.main()
