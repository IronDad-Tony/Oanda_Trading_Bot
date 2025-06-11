import pygad # Added for GA functionality
import logging
import random # Added for random seed
import numpy as np # Added for random seed
from typing import Callable, Dict, List, Any, Tuple, Optional

DEFAULT_GA_SETTINGS = {
    'num_generations': 100,
    'sol_per_pop': 50,  # PyGAD uses sol_per_pop for population_size
    'num_parents_mating': 2,
    'keep_parents': -1, # Keep all parents in the next population.
    'parent_selection_type': "sss",  # steady-state selection
    'crossover_type': "single_point",
    'mutation_type': "random", # Can be 'adaptive' or custom
    'mutation_percent_genes': "default", # PyGAD default, often 10%. Can be a float like 0.1 for 10%
    'random_seed': None,
    'stop_criteria': None, # e.g., "saturate_10" to stop if fitness doesn't improve for 10 gens
    'allow_duplicate_genes': True,
}

class GeneticOptimizer:
    def __init__(self,
                 fitness_function: Callable[[Dict[str, Any], Dict[str, Any]], float],
                 param_space: Dict[str, Tuple[type, Any]], # Type hint for param_space value updated
                 base_config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None,
                 ga_settings: Optional[Dict[str, Any]] = None):
        """
        Initializes the GeneticOptimizer as a wrapper around PyGAD.

        Args:
            fitness_function: A function that takes a dictionary of parameters 
                              and a context dictionary, and returns a fitness score.
            param_space: A dictionary defining the parameter space.
                         Keys are parameter names. Values are tuples: (type, definition).
                         - type: float, int, bool, str (for categorical)
                         - definition:
                           - For float/int: (min_val, max_val) tuple.
                           - For bool: [True, False] or ignored (PyGAD uses 0/1).
                           - For str (categorical): List of possible string values.
                           - For int (categorical): List of possible int values.
            base_config: Base configuration dictionary, stored but not directly used by PyGAD.
            logger: Logger instance.
            ga_settings: Dictionary of GA settings to override PyGAD defaults or provide custom ones.
        """
        self.fitness_function = fitness_function
        self.param_space = param_space
        self.base_config = base_config if base_config is not None else {}

        self.logger = logger or logging.getLogger(__name__ + ".GeneticOptimizer")
        if not logger: # If a logger was not provided, and we just created one
            self.logger.info("GeneticOptimizer initialized without a specific logger. Using default.")
            if not self.logger.handlers: # Configure basic logging if no handlers are present for the new logger
                logging.basicConfig(level=logging.INFO) # Or your preferred default logging setup

        # Prepare GA settings by starting with defaults and updating with user-provided ones
        current_ga_settings = DEFAULT_GA_SETTINGS.copy()
        if ga_settings:
            # Map conceptual settings to PyGAD's specific parameter names if they exist
            if 'population_size' in ga_settings:
                current_ga_settings['sol_per_pop'] = ga_settings.pop('population_size')
            if 'n_generations' in ga_settings:
                current_ga_settings['num_generations'] = ga_settings.pop('n_generations')
            # For other parameters, directly update
            current_ga_settings.update(ga_settings)
        
        self.ga_settings = current_ga_settings
        self.logger.info("GeneticOptimizer initialized.")
        self.logger.debug(f"Effective GA Settings: {self.ga_settings}")

        # Seed for reproducibility if provided in the final settings
        random_seed_value = self.ga_settings.get('random_seed')
        if random_seed_value is not None:
            random.seed(random_seed_value)
            np.random.seed(random_seed_value)

        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_fitness_: float = -float('inf')
        
        # Prepare gene_space and gene_type for PyGAD from param_space
        self.gene_space, self.gene_type, self.param_names = self._prepare_gene_space()

        if not self.fitness_function: # Should be caught by type hinting, but good runtime check
            self.logger.error("A fitness_function must be provided.")
            raise ValueError("A fitness_function must be provided.")

        if not self.param_space:
            self.logger.error("param_space cannot be None or empty for GeneticOptimizer.")
            raise ValueError("param_space cannot be None or empty for GeneticOptimizer.")

    def _prepare_gene_space(self) -> Tuple[List[Any], List[Any], List[str]]:
        gene_space_list = []
        gene_type_list = []
        param_names_list = list(self.param_space.keys())

        for param_name in param_names_list:
            space_def = self.param_space[param_name]
            
            if not isinstance(space_def, tuple) or len(space_def) != 2:
                raise ValueError(f"Parameter '{param_name}' in param_space has invalid format. Expected (type, definition). Got: {space_def}")

            param_type, definition = space_def

            if param_type == float:
                if not (isinstance(definition, tuple) and len(definition) == 2 and all(isinstance(v, (int, float)) for v in definition)):
                    raise ValueError(f"Float param '{param_name}' expects (min, max) tuple. Got: {definition}")
                gene_space_list.append({'low': float(definition[0]), 'high': float(definition[1])})
                gene_type_list.append(float)
            elif param_type == int:
                if isinstance(definition, tuple) and len(definition) == 2 and all(isinstance(v, int) for v in definition): # Range
                    gene_space_list.append({'low': definition[0], 'high': definition[1]})
                    gene_type_list.append(int)
                elif isinstance(definition, list) and all(isinstance(v, int) for v in definition): # Categorical int
                    gene_space_list.append(definition)
                    gene_type_list.append(int) # PyGAD handles categorical ints directly
                else:
                    raise ValueError(f"Int param '{param_name}' expects (min, max) tuple or list of ints. Got: {definition}")
            elif param_type == bool:
                # PyGAD handles bools as int 0/1. Definition like [True, False] is mostly for user clarity.
                gene_space_list.append({'low': 0, 'high': 1})
                gene_type_list.append(int) 
            elif param_type == str:
                if not (isinstance(definition, list) and all(isinstance(v, str) for v in definition)):
                    raise ValueError(f"String (categorical) param '{param_name}' expects a list of strings. Got: {definition}")
                gene_space_list.append(definition)
                gene_type_list.append(str)
            else:
                raise ValueError(f"Unsupported parameter type '{param_type.__name__}' for param '{param_name}'.")
        
        return gene_space_list, gene_type_list, param_names_list

    def _calculate_fitness(self, solution: List[Any], solution_idx: int, context: Dict[str, Any]) -> float:
        params_dict = {}
        for i, param_name in enumerate(self.param_names):
            param_type, _ = self.param_space[param_name] # Get type from original param_space
            value = solution[i]
            if param_type == bool:
                params_dict[param_name] = bool(value) # Convert 0/1 back to bool
            else:
                params_dict[param_name] = value
        
        try:
            fitness = self.fitness_function(params_dict, context) # Pass context
            if fitness is None:
                self.logger.warning(f"Fitness function returned None for params: {params_dict}. Assigning -inf fitness.")
                return -float('inf') # PyGAD expects a float
            return float(fitness)
        except Exception as e:
            self.logger.error(f"Error in user-provided fitness_function for params {params_dict}: {e}", exc_info=True)
            return -float('inf') # Penalize solutions that cause errors

    def on_generation(self, ga_instance: pygad.GA):
        # This callback is executed after each generation.
        # We can use it to update our internal tracking of the best solution.
        current_ga_best_solution_genes, current_ga_best_fitness, _ = ga_instance.best_solution()
        
        # PyGAD fitness can sometimes be numpy.float64, ensure it's a Python float
        current_ga_best_fitness = float(current_ga_best_fitness)

        if current_ga_best_fitness > self.best_fitness_:
            self.best_fitness_ = current_ga_best_fitness
            
            # Decode genes to params
            decoded_params = {}
            for i, param_name in enumerate(self.param_names):
                param_type, _ = self.param_space[param_name]
                value = current_ga_best_solution_genes[i]
                if param_type == bool:
                    decoded_params[param_name] = bool(value)
                else:
                    # Ensure correct type for int if PyGAD returns float for int gene_type with step
                    if param_type == int and isinstance(value, float):
                        decoded_params[param_name] = int(round(value))
                    else:
                        decoded_params[param_name] = value
            self.best_params_ = decoded_params
            self.logger.info(f"New best solution found in generation {ga_instance.generations_completed}: Fitness = {self.best_fitness_}, Params = {self.best_params_}")
        # Optional: Log generation progress
        # self.logger.debug(f"Generation {ga_instance.generations_completed} finished. Current best fitness in GA: {current_ga_best_fitness}")


    def run_optimizer(self, current_context: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], float]:
        self.best_params_ = None # Reset before a new run
        self.best_fitness_ = -float('inf')

        # Wrapper for PyGAD\'s fitness_func to include current_context
        # PyGAD\'s fitness_func expects (solution, solution_idx)
        def fitness_wrapper_for_pygad(solution, solution_idx):
            return self._calculate_fitness(solution, solution_idx, current_context)

        # Prepare GA arguments from self.ga_settings, ensuring all required ones for PyGAD are present
        pygad_ga_args = {
            'num_generations': self.ga_settings['num_generations'],
            'sol_per_pop': self.ga_settings['sol_per_pop'],
            'num_parents_mating': self.ga_settings['num_parents_mating'],
            'fitness_func': fitness_wrapper_for_pygad,
            'gene_space': self.gene_space,
            'gene_type': self.gene_type, # PyGAD uses this for type consistency
            'parent_selection_type': self.ga_settings.get('parent_selection_type', 'sss'),
            'keep_parents': self.ga_settings.get('keep_parents', -1),
            'crossover_type': self.ga_settings.get('crossover_type', 'single_point'),
            'mutation_type': self.ga_settings.get('mutation_type', 'random'),
            'mutation_percent_genes': self.ga_settings.get('mutation_percent_genes', 'default'), # PyGAD handles 'default'
            'random_seed': self.ga_settings.get('random_seed'),
            'on_generation': self.on_generation,
            'allow_duplicate_genes': self.ga_settings.get('allow_duplicate_genes', True),
        }
        
        stop_criteria_val = self.ga_settings.get('stop_criteria')
        if stop_criteria_val:
            pygad_ga_args['stop_criteria'] = stop_criteria_val
            
        # Handle gene_type specification for PyGAD if types are mixed
        # If all gene_type are same, PyGAD allows single value. If mixed, must be a list.
        # self.gene_type is already a list.

        # For mutation_by_replacement, PyGAD needs it if you want to ensure new values are from gene_space for categorical
        if self.ga_settings.get('mutation_by_replacement') is not None: # Explicitly check if user set it
            pygad_ga_args['mutation_by_replacement'] = self.ga_settings['mutation_by_replacement']


        self.logger.info(f"Starting PyGAD optimizer run with args: { {k:v for k,v in pygad_ga_args.items() if k != 'fitness_func'} }") # Don't log the function object itself
        
        try:
            ga_instance = pygad.GA(**pygad_ga_args)
            ga_instance.run()
            self.logger.info(f"Genetic optimizer run completed. Best fitness: {self.best_fitness_}")
        except Exception as e:
            self.logger.error(f"Exception during PyGAD GA instantiation or run: {e}", exc_info=True)
            # Fallback: return current best, which might be None/-inf if error was early
            if self.best_params_ is None: # If no solution was ever found
                 return {}, -float('inf')
            return self.best_params_, self.best_fitness_


        if self.best_params_ is None:
            # This log message was updated to match the test expectation.
            self.logger.warning("Genetic optimizer did not find a valid solution during the run. Returning empty params and -inf fitness.")
            return {}, -float('inf') # Return empty dict and -inf if no solution was found

        return self.best_params_, self.best_fitness_

# Example of how this GeneticOptimizer might be used (for illustration, not part of the class):
# def my_fitness_function(params: Dict[str, Any], context: Dict[str, Any]) -> float:
#     x = params['x']
#     y = params['y']
#     # Use context if needed, e.g., context['target_value']
#     return -(x**2 + y**2) # Minimize x^2 + y^2 (so maximize -(x^2+y^2))

# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)

#     parameter_space = {
#         'x': (float, (-5.0, 5.0)),
#         'y': (int, (-10, 10)),
#         'option': (str, ['A', 'B', 'C']),
#         'active': (bool, [True, False]) # Definition for bool is mostly for clarity
#     }
    
#     optimizer_settings = {
#         'num_generations': 50,
#         'sol_per_pop': 20,
#         'random_seed': 42,
#         'mutation_percent_genes': 0.1 # 10%
#     }

#     optimizer = GeneticOptimizer(
#         fitness_function=my_fitness_function,
#         param_space=parameter_space,
#         logger=logger,
#         ga_settings=optimizer_settings
#     )

#     run_context = {'target_value': 100} # Example context
#     best_parameters, best_score = optimizer.run_optimizer(current_context=run_context)

#     logger.info(f"Optimization finished.")
#     logger.info(f"Best parameters found: {best_parameters}")
#     logger.info(f"Best fitness score: {best_score}")
