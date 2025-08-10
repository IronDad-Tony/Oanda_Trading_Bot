import unittest
from unittest.mock import Mock, patch, call
import logging
# import pygad # Not strictly needed for import if pygad.GA is patched, but good for clarity

from src.agent.optimizers.genetic_optimizer import GeneticOptimizer, DEFAULT_GA_SETTINGS

class TestGeneticOptimizer(unittest.TestCase):
    def setUp(self):
        self.logger = Mock(spec=logging.Logger)
        self.mock_external_fitness_function = Mock(return_value=0.5)

        self.param_space_simple = {
            'param_float': (float, (0.1, 1.0)),
            'param_int': (int, (1, 10))
        }
        # Order of param_names must match iteration order of param_space_simple keys
        self.param_names_simple = list(self.param_space_simple.keys())


        self.param_space_complex = {
            'p_float': (float, (0.0, 1.0)),
            'p_int': (int, (1, 100)),
            'p_cat_str': (str, ['A', 'B', 'C']),
            'p_bool': (bool, [True, False]),
            'p_cat_int': (int, [10, 20, 30])
        }
        self.param_names_complex = list(self.param_space_complex.keys())

        self.base_config = {'name': 'TestStrategyFromGO', 'version': '0.1'}
        self.ga_settings_custom = {
            'num_generations': 5,
            # Use 'sol_per_pop' directly as this is what PyGAD expects and what GeneticOptimizer will store
            'sol_per_pop': 10, 
            'num_parents_mating': 2,
            # Other settings will take from DEFAULT_GA_SETTINGS
        }

        # Optimizer for most tests, using simple param space
        self.optimizer = GeneticOptimizer(
            fitness_function=self.mock_external_fitness_function,
            param_space=self.param_space_simple,
            base_config=self.base_config,
            logger=self.logger,
            ga_settings=self.ga_settings_custom
        )

    def test_initialization(self):
        self.assertEqual(self.optimizer.fitness_function, self.mock_external_fitness_function)
        self.assertEqual(self.optimizer.param_space, self.param_space_simple)
        self.assertEqual(self.optimizer.base_config, self.base_config)
        self.assertEqual(self.optimizer.logger, self.logger)
        
        expected_ga_settings = DEFAULT_GA_SETTINGS.copy()
        # Simulate the mapping logic from GeneticOptimizer.__init__ for the test's expectation
        custom_settings_for_expected = self.ga_settings_custom.copy()
        if 'population_size' in custom_settings_for_expected: # Though we changed it to sol_per_pop above, this makes it robust
            custom_settings_for_expected['sol_per_pop'] = custom_settings_for_expected.pop('population_size')
        
        expected_ga_settings.update(custom_settings_for_expected)
        self.assertEqual(self.optimizer.ga_settings, expected_ga_settings)

        self.assertIsNone(self.optimizer.best_params_)
        self.assertEqual(self.optimizer.best_fitness_, -float('inf'))
        # Check that _prepare_gene_space was effectively called for param_names
        self.assertEqual(self.optimizer.param_names, self.param_names_simple)

    def test_prepare_gene_space_simple(self):
        # This optimizer was created with param_space_simple in setUp
        gene_space, gene_type, param_names = self.optimizer._prepare_gene_space()

        self.assertEqual(param_names, self.param_names_simple)
        self.assertEqual(gene_space, [{'low': 0.1, 'high': 1.0}, {'low': 1, 'high': 10}])
        self.assertEqual(gene_type, [float, int])

    def test_prepare_gene_space_complex(self):
        optimizer_complex = GeneticOptimizer(
            fitness_function=self.mock_external_fitness_function,
            param_space=self.param_space_complex,
            base_config=self.base_config,
            logger=self.logger,
            ga_settings=self.ga_settings_custom
        )
        gene_space, gene_type, param_names = optimizer_complex._prepare_gene_space()

        self.assertEqual(param_names, self.param_names_complex)
        expected_gene_space = [
            {'low': 0.0, 'high': 1.0},      # p_float
            {'low': 1, 'high': 100},        # p_int
            ['A', 'B', 'C'],                # p_cat_str
            {'low': 0, 'high': 1},          # p_bool (mapped to int 0/1)
            [10, 20, 30]                    # p_cat_int
        ]
        expected_gene_type = [float, int, str, int, int] # bool is int for pygad
        
        self.assertEqual(gene_space, expected_gene_space)
        self.assertEqual(gene_type, expected_gene_type)

    def test_calculate_fitness_calls_external_fitness_function(self):
        # _calculate_fitness is the internal wrapper for PyGAD's fitness_func
        solution_genes = [0.6, 6] # Corresponds to param_float, param_int from param_space_simple
        solution_idx = 0
        current_context = {'data': 'test_data'}
        
        expected_params_dict = dict(zip(self.param_names_simple, solution_genes))
        self.mock_external_fitness_function.return_value = 0.789
        
        fitness = self.optimizer._calculate_fitness(solution_genes, solution_idx, current_context)
        
        self.assertEqual(fitness, 0.789)
        self.mock_external_fitness_function.assert_called_once_with(expected_params_dict, current_context)

    def test_calculate_fitness_handles_bool_and_categorical_decoding(self):
        optimizer_complex = GeneticOptimizer(
            fitness_function=self.mock_external_fitness_function,
            param_space=self.param_space_complex,
            base_config=self.base_config,
            logger=self.logger
        )
        # Genes: p_float, p_int, p_cat_str, p_bool (0/1), p_cat_int
        solution_genes = [0.5, 50, 'B', 1, 20] # p_bool=1 means True
        solution_idx = 0
        current_context = {'complex_data': True}

        expected_params_dict = {
            'p_float': 0.5,
            'p_int': 50,
            'p_cat_str': 'B',
            'p_bool': True, # Decoded from 1
            'p_cat_int': 20
        }
        self.mock_external_fitness_function.return_value = 0.99
        
        fitness = optimizer_complex._calculate_fitness(solution_genes, solution_idx, current_context)
        
        self.assertEqual(fitness, 0.99)
        self.mock_external_fitness_function.assert_called_once_with(expected_params_dict, current_context)

    def test_calculate_fitness_handles_invalid_return_from_external_fitness(self):
        solution_genes = [0.6, 6]
        self.mock_external_fitness_function.return_value = None # Invalid fitness
        
        fitness = self.optimizer._calculate_fitness(solution_genes, 0, {})
        
        self.assertEqual(fitness, -float('inf'))
        self.logger.warning.assert_called_once()
        # Check log message content if necessary

    def test_calculate_fitness_handles_exception_from_external_fitness(self):
        solution_genes = [0.6, 6]
        self.mock_external_fitness_function.side_effect = ValueError("Fitness calculation failed")
        
        fitness = self.optimizer._calculate_fitness(solution_genes, 0, {})
        
        self.assertEqual(fitness, -float('inf'))
        self.logger.error.assert_called_once()
        # Check log message content if necessary

    def test_on_generation_updates_best_solution_if_better(self):
        mock_ga_instance = Mock()
        # Genes for param_float, param_int
        best_solution_genes = [0.77, 7] 
        new_best_fitness = 0.95
        
        mock_ga_instance.best_solution.return_value = (best_solution_genes, new_best_fitness, 0) # solution, fitness, idx
        
        self.optimizer.best_fitness_ = 0.90 # Previous best fitness

        self.optimizer.on_generation(mock_ga_instance)

        self.assertEqual(self.optimizer.best_fitness_, new_best_fitness)
        expected_best_params = dict(zip(self.param_names_simple, best_solution_genes))
        self.assertEqual(self.optimizer.best_params_, expected_best_params)
        self.logger.info.assert_called() # Check for logging

    def test_on_generation_does_not_update_if_not_better(self):
        mock_ga_instance = Mock()
        initial_best_params_dict = {'param_float': 0.88, 'param_int': 8}
        initial_best_fitness = 0.98
        
        self.optimizer.best_fitness_ = initial_best_fitness
        self.optimizer.best_params_ = initial_best_params_dict
        
        # New solution from GA, but it's worse
        worse_solution_genes = [0.11, 1] # Example, assuming param_names_simple order
        worse_fitness = 0.5
        mock_ga_instance.best_solution.return_value = (worse_solution_genes, worse_fitness, 0)

        self.optimizer.on_generation(mock_ga_instance)

        self.assertEqual(self.optimizer.best_fitness_, initial_best_fitness) 
        self.assertEqual(self.optimizer.best_params_, initial_best_params_dict)
        self.logger.info.assert_called() 

    @patch('src.agent.optimizers.genetic_optimizer.pygad.GA')
    def test_run_optimizer_full_flow_returns_best(self, MockPyGAD_GA):
        mock_ga_instance = Mock() # This is the mock for the GA *instance*
        MockPyGAD_GA.return_value = mock_ga_instance

        # These are the values we expect run_optimizer to set via on_generation
        final_best_params_dict = {'param_float': 0.75, 'param_int': 8} 
        final_best_fitness = 0.99

        # Simulate that on_generation (called during mock_ga_instance.run())
        # has found and set the best solution on the optimizer instance.
        def mock_run_sets_best_solution_on_optimizer():
            self.optimizer.best_params_ = final_best_params_dict
            self.optimizer.best_fitness_ = final_best_fitness
        
        # Assign the function to the 'run' attribute of the *mock_ga_instance*
        mock_ga_instance.run = Mock(side_effect=mock_run_sets_best_solution_on_optimizer) 
        
        current_context = {'market_condition': 'bullish'}
        best_params, best_fitness = self.optimizer.run_optimizer(current_context)

        MockPyGAD_GA.assert_called_once()
        args, kwargs = MockPyGAD_GA.call_args
        
        self.assertEqual(kwargs['num_generations'], self.ga_settings_custom['num_generations'])
        # self.ga_settings_custom now directly uses 'sol_per_pop'
        self.assertEqual(kwargs['sol_per_pop'], self.ga_settings_custom['sol_per_pop']) 
        self.assertEqual(kwargs['num_parents_mating'], self.ga_settings_custom['num_parents_mating'])
        self.assertEqual(kwargs['gene_space'], self.optimizer.gene_space)
        self.assertEqual(kwargs['gene_type'], self.optimizer.gene_type)
        self.assertTrue(callable(kwargs['fitness_func']))
        self.assertEqual(kwargs['on_generation'], self.optimizer.on_generation)
        # Check other GA params from DEFAULT_GA_SETTINGS are passed
        self.assertEqual(kwargs['mutation_type'], DEFAULT_GA_SETTINGS['mutation_type'])

        mock_ga_instance.run.assert_called_once()
        self.assertEqual(best_params, final_best_params_dict)
        self.assertEqual(best_fitness, final_best_fitness)
        self.logger.info.assert_any_call(f"Genetic optimizer run completed. Best fitness: {final_best_fitness}")

    @patch('src.agent.optimizers.genetic_optimizer.pygad.GA')
    def test_run_optimizer_no_solution_found_logs_warning(self, MockPyGAD_GA):
        mock_ga_instance = Mock() # This is the mock for the GA *instance*
        MockPyGAD_GA.return_value = mock_ga_instance

        def mock_run_no_solution():
            # Simulate that best_params_ remains None after GA run
            self.optimizer.best_params_ = None 
            self.optimizer.best_fitness_ = -float('inf') 
        
        # Assign the function to the 'run' attribute of the *mock_ga_instance*
        mock_ga_instance.run = Mock(side_effect=mock_run_no_solution)
        
        # Ensure initial state before call
        self.optimizer.best_params_ = None
        self.optimizer.best_fitness_ = -float('inf')

        current_context = {'market_condition': 'choppy'}
        best_params, best_fitness = self.optimizer.run_optimizer(current_context)

        self.assertEqual(best_params, {}) # Expected return for no solution
        self.assertEqual(best_fitness, -float('inf'))
        # Update expected log message to match the actual implementation
        self.logger.warning.assert_called_with("Genetic optimizer did not find a valid solution during the run. Returning empty params and -inf fitness.")
        self.logger.info.assert_any_call("Genetic optimizer run completed. Best fitness: -inf")

if __name__ == '__main__':
    unittest.main()
