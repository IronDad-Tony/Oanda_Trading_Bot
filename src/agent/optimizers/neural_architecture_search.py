import logging
from typing import Dict, Any, Callable, Tuple, Optional, List
import random
import copy

class NeuralArchitectureSearch:
    def __init__(self,
                 fitness_function: Callable[[Dict[str, Any], Dict[str, Any]], float],
                 search_space: Dict[str, Any],
                 logger: Optional[logging.Logger] = None,
                 nas_settings: Optional[Dict[str, Any]] = None):
        """
        Initializes the NeuralArchitectureSearch optimizer using an Evolutionary Algorithm.

        Args:
            fitness_function: A function that takes a candidate architecture (represented as a list of layer dicts)
                              and a context dictionary, evaluates it, and returns a fitness score.
            search_space: A dictionary defining the search space for neural architectures.
                          Example:
                          {
                              "min_layers": 2, "max_layers": 5,
                              "layer_types": {
                                  "dense": {"units": {"type": "int_range", "min": 32, "max": 512, "step": 32},
                                            "activation": {"type": "choice", "values": ["relu", "tanh"]}},
                                  "dropout": {"rate": {"type": "float_range", "min": 0.1, "max": 0.5, "step": 0.1}}
                              },
                              "output_layer_activation": {"type": "choice", "values": ["sigmoid", "linear"]} # Optional
                          }
            logger: Logger instance.
            nas_settings: Dictionary of EA-specific settings.
                          Default values are used if not provided.
                          {
                              "population_size": 20,
                              "num_generations": 10,
                              "mutation_rate": 0.2,       # Prob of mutating an individual
                              "gene_mutation_probability": 0.1, # Prob of mutating a gene within an individual
                              "crossover_rate": 0.7,
                              "tournament_size": 3,
                              "elitism_count": 2
                          }
        """
        self.fitness_function = fitness_function
        self.search_space = search_space
        self.logger = logger or logging.getLogger(__name__ + ".NeuralArchitectureSearch")
        
        default_settings = {
            "population_size": 20,
            "num_generations": 10,
            "mutation_rate": 0.2,
            "gene_mutation_probability": 0.1,
            "crossover_rate": 0.7,
            "tournament_size": 3,
            "elitism_count": 2
        }
        self.nas_settings = {**default_settings, **(nas_settings if nas_settings is not None else {})}

        if not self.fitness_function:
            self.logger.error("A fitness_function must be provided for NeuralArchitectureSearch.")
            raise ValueError("A fitness_function must be provided for NeuralArchitectureSearch.")
        if not self.search_space:
            self.logger.error("search_space cannot be None or empty for NeuralArchitectureSearch.")
            raise ValueError("search_space cannot be None or empty for NeuralArchitectureSearch.")
        if not self.search_space.get("layer_types") or not isinstance(self.search_space["layer_types"], dict):
            self.logger.error("search_space must contain a 'layer_types' dictionary.")
            raise ValueError("search_space must contain a 'layer_types' dictionary.")
        if "min_layers" not in self.search_space or "max_layers" not in self.search_space:
            self.logger.warning("search_space should define 'min_layers' and 'max_layers'. Using defaults 1 and 5.")
            self.search_space.setdefault("min_layers", 1)
            self.search_space.setdefault("max_layers", 5)
        if self.search_space["min_layers"] > self.search_space["max_layers"]:
            raise ValueError("min_layers cannot be greater than max_layers in search_space.")


        self.logger.info("NeuralArchitectureSearch (Evolutionary Algorithm) initialized.")
        self.logger.info(f"NAS Settings: {self.nas_settings}")
        self.logger.debug(f"Search Space: {self.search_space}")

        self.best_architecture_: Optional[List[Dict[str, Any]]] = None
        self.best_fitness_: float = -float('inf')

    def _get_random_value(self, param_config: Dict[str, Any]) -> Any:
        param_type = param_config["type"]
        if param_type == "choice":
            return random.choice(param_config["values"])
        elif param_type == "int_range":
            return random.randrange(param_config["min"], param_config["max"] + 1, param_config.get("step", 1))
        elif param_type == "float_range":
            # Note: random.uniform doesn't support step directly.
            # If step is crucial, need to implement discrete steps for floats.
            if "step" in param_config:
                 num_steps = int((param_config["max"] - param_config["min"]) / param_config["step"])
                 return param_config["min"] + random.randint(0, num_steps) * param_config["step"]
            return random.uniform(param_config["min"], param_config["max"])
        else:
            self.logger.error(f"Unsupported parameter type in search_space: {param_type}")
            raise ValueError(f"Unsupported parameter type: {param_type}")

    def _initialize_individual(self) -> List[Dict[str, Any]]:
        architecture = []
        num_layers = random.randint(self.search_space["min_layers"], self.search_space["max_layers"])
        available_layer_types = list(self.search_space["layer_types"].keys())

        for _ in range(num_layers):
            layer_type = random.choice(available_layer_types)
            layer_config = {"type": layer_type}
            for param_name, param_spec in self.search_space["layer_types"][layer_type].items():
                layer_config[param_name] = self._get_random_value(param_spec)
            architecture.append(layer_config)
        
        # Handle output layer if specified separately
        if "output_layer_activation" in self.search_space and architecture:
            # This is a simple way; could be more complex (e.g., ensuring last layer is dense)
            # For now, we assume the fitness function/model builder handles the final output layer structure
            # based on the task, but we can provide a hint for its activation.
            # If the last layer is suitable (e.g. dense), we can try to set its activation.
            # This part might need more sophisticated logic based on how models are constructed.
            pass

        return architecture

    def _initialize_population(self) -> List[List[Dict[str, Any]]]:
        return [self._initialize_individual() for _ in range(self.nas_settings["population_size"])]

    def _select_parents(self, population_with_fitness: List[Tuple[List[Dict[str, Any]], float]]) -> List[List[Dict[str, Any]]]:
        parents = []
        for _ in range(self.nas_settings["population_size"]): # Need enough parents to generate next population
            tournament = random.sample(population_with_fitness, self.nas_settings["tournament_size"])
            tournament.sort(key=lambda x: x[1], reverse=True) # Higher fitness is better
            parents.append(tournament[0][0]) # Add the architecture of the winner
        return parents

    def _crossover(self, parent1: List[Dict[str, Any]], parent2: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # Single-point crossover for list of layers
        p1 = copy.deepcopy(parent1)
        p2 = copy.deepcopy(parent2)
        
        min_len = min(len(p1), len(p2))
        if min_len <= 1: # Not enough length to crossover
            return p1, p2

        crossover_point = random.randint(1, min_len -1) # Ensure point is not at the ends
        
        child1 = p1[:crossover_point] + p2[crossover_point:]
        child2 = p2[:crossover_point] + p1[crossover_point:]
        
        # Ensure children respect min/max layer constraints
        child1 = child1[:self.search_space["max_layers"]]
        child2 = child2[:self.search_space["max_layers"]]
        # If too short, this needs a strategy (e.g. append random, or handled by mutation/fitness)
        # For now, we accept potentially shorter children if crossover results in it.
        # Fitness function should penalize invalid architectures (e.g. too few layers if that's an issue)

        return child1, child2

    def _mutate(self, architecture: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        mutated_arch = copy.deepcopy(architecture)
        gene_mutation_prob = self.nas_settings["gene_mutation_probability"]

        # 1. Mutate layer parameters
        for layer in mutated_arch:
            if random.random() < gene_mutation_prob: # Mutate this layer's params
                layer_type_spec = self.search_space["layer_types"].get(layer["type"])
                if layer_type_spec:
                    param_to_mutate = random.choice(list(layer_type_spec.keys()))
                    layer[param_to_mutate] = self._get_random_value(layer_type_spec[param_to_mutate])
        
        # 2. Add a layer
        if len(mutated_arch) < self.search_space["max_layers"] and random.random() < gene_mutation_prob:
            available_layer_types = list(self.search_space["layer_types"].keys())
            new_layer_type = random.choice(available_layer_types)
            new_layer_config = {"type": new_layer_type}
            for param_name, param_spec in self.search_space["layer_types"][new_layer_type].items():
                new_layer_config[param_name] = self._get_random_value(param_spec)
            
            insert_idx = random.randint(0, len(mutated_arch))
            mutated_arch.insert(insert_idx, new_layer_config)

        # 3. Remove a layer
        if len(mutated_arch) > self.search_space["min_layers"] and random.random() < gene_mutation_prob:
            if mutated_arch: # Ensure not empty
                remove_idx = random.randint(0, len(mutated_arch) - 1)
                del mutated_arch[remove_idx]
        
        # 4. Change layer type (less common, can be disruptive, but adds diversity)
        if mutated_arch and random.random() < gene_mutation_prob / 2: # Lower probability
            idx_to_change = random.randint(0, len(mutated_arch) - 1)
            current_layer_type = mutated_arch[idx_to_change]["type"]
            available_layer_types = list(self.search_space["layer_types"].keys())
            new_type = random.choice([lt for lt in available_layer_types if lt != current_layer_type] or [current_layer_type]) # Avoid no choice
            
            if new_type != current_layer_type:
                new_layer_config = {"type": new_type}
                for param_name, param_spec in self.search_space["layer_types"][new_type].items():
                    new_layer_config[param_name] = self._get_random_value(param_spec)
                mutated_arch[idx_to_change] = new_layer_config
        
        # Ensure final architecture is within min/max layer bounds after mutations
        while len(mutated_arch) < self.search_space["min_layers"] and len(mutated_arch) < self.search_space["max_layers"]:
             # Add random layers if too short
            available_layer_types = list(self.search_space["layer_types"].keys())
            new_layer_type = random.choice(available_layer_types)
            new_layer_config = {"type": new_layer_type}
            for param_name, param_spec in self.search_space["layer_types"][new_layer_type].items():
                new_layer_config[param_name] = self._get_random_value(param_spec)
            mutated_arch.append(new_layer_config)

        if len(mutated_arch) > self.search_space["max_layers"]:
            mutated_arch = mutated_arch[:self.search_space["max_layers"]]

        return mutated_arch

    def run_optimizer(self, current_context: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], float]:
        self.logger.info("Starting Neural Architecture Search (Evolutionary Algorithm)...")
        
        population = self._initialize_population()
        self.best_architecture_ = None
        self.best_fitness_ = -float('inf')

        for gen in range(self.nas_settings["num_generations"]):
            self.logger.info(f"Generation {gen + 1}/{self.nas_settings['num_generations']}")
            
            population_with_fitness: List[Tuple[List[Dict[str, Any]], float]] = []
            for i, arch_candidate in enumerate(population):
                if not arch_candidate: # Skip empty architectures that might result from aggressive mutation/crossover
                    self.logger.warning(f"Skipping empty architecture candidate in generation {gen+1}, individual {i}.")
                    population_with_fitness.append((arch_candidate, -float('inf'))) # Penalize heavily
                    continue
                try:
                    fitness = self._evaluate_architecture(arch_candidate, current_context)
                    population_with_fitness.append((arch_candidate, fitness))
                except Exception as e:
                    self.logger.error(f"Error evaluating architecture {arch_candidate} in generation {gen+1}: {e}", exc_info=True)
                    population_with_fitness.append((arch_candidate, -float('inf'))) # Penalize heavily on error


            population_with_fitness.sort(key=lambda x: x[1], reverse=True)

            if population_with_fitness[0][1] > self.best_fitness_:
                self.best_fitness_ = population_with_fitness[0][1]
                self.best_architecture_ = copy.deepcopy(population_with_fitness[0][0])
                self.logger.info(f"New best fitness: {self.best_fitness_:.4f} (Gen {gen + 1})")
                self.logger.debug(f"Best architecture: {self.best_architecture_}")

            self.logger.info(f"Generation {gen + 1} - Best Fitness: {population_with_fitness[0][1]:.4f}, Avg Fitness: {sum(f for _, f in population_with_fitness)/len(population_with_fitness):.4f}")

            next_population = []
            
            # Elitism
            elitism_count = min(self.nas_settings["elitism_count"], len(population_with_fitness))
            for i in range(elitism_count):
                next_population.append(copy.deepcopy(population_with_fitness[i][0]))

            # Selection and Reproduction
            parents = self._select_parents(population_with_fitness)
            
            num_offspring_needed = self.nas_settings["population_size"] - elitism_count
            current_offspring_count = 0
            parent_idx = 0

            while current_offspring_count < num_offspring_needed:
                p1 = parents[parent_idx % len(parents)]
                p2 = parents[(parent_idx + 1) % len(parents)] # Ensure different parent for crossover
                parent_idx += 2


                if random.random() < self.nas_settings["crossover_rate"] and len(p1)>0 and len(p2)>0 : # Ensure parents are not empty
                    child1, child2 = self._crossover(p1, p2)
                else:
                    child1, child2 = copy.deepcopy(p1), copy.deepcopy(p2)
                
                if random.random() < self.nas_settings["mutation_rate"] and child1: # Mutate if not empty
                    child1 = self._mutate(child1)
                if random.random() < self.nas_settings["mutation_rate"] and child2: # Mutate if not empty
                    child2 = self._mutate(child2)

                if child1: # Add if valid
                    next_population.append(child1)
                    current_offspring_count +=1
                if current_offspring_count < num_offspring_needed and child2: # Add if valid and space permits
                     next_population.append(child2)
                     current_offspring_count +=1
                
                if not parents: # Safety break if parents list somehow becomes empty
                    self.logger.warning("Parent pool is empty, cannot generate more offspring.")
                    break
            
            population = next_population[:self.nas_settings["population_size"]] # Ensure population size

        if self.best_architecture_ is None:
            self.logger.warning("Neural Architecture Search did not find a valid architecture. Returning empty and -inf fitness.")
            return [], -float('inf') # Return empty list for architecture

        self.logger.info(f"Neural Architecture Search completed. Best fitness: {self.best_fitness_:.4f}")
        self.logger.info(f"Best Architecture: {self.best_architecture_}")
        return self.best_architecture_, self.best_fitness_

    def _evaluate_architecture(self, architecture: List[Dict[str, Any]], context: Dict[str, Any]) -> float:
        """
        Internal helper to call the provided fitness function.
        """
        if not architecture: # Handle empty architecture list
            self.logger.warning("Attempted to evaluate an empty architecture. Returning -inf fitness.")
            return -float('inf')
        try:
            fitness = self.fitness_function(architecture, context)
            if fitness is None: # Ensure fitness function returns a float
                self.logger.warning(f"Fitness function returned None for architecture: {architecture}. Assigning -inf fitness.")
                return -float('inf')
            return float(fitness)
        except Exception as e:
            self.logger.error(f"Error in user-provided fitness_function for architecture {architecture}: {e}", exc_info=True)
            return -float('inf')

# Example Usage (conceptual)
if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("NAS_Example_Quant")

    # --- More Realistic Fitness Function for Quant Models ---
    def example_nas_fitness_function(architecture_params: List[Dict[str, Any]], context: Dict[str, Any]) -> float:
        logger.info(f"Evaluating architecture: {architecture_params}")
        
        # Context might contain: input_features, output_features, sequence_length, device
        input_features = context.get("input_features", 10) # e.g., 10 input features
        sequence_length = context.get("sequence_length", 50) # e.g., sequence length of 50
        output_size = context.get("output_size", 1) # e.g., predict 1 value
        device = context.get("device", "cpu")

        model_layers = []
        current_features = input_features
        current_seq_len = sequence_length # For Conv1D/LSTM, track sequence length changes if pooling/etc.

        try:
            if not architecture_params:
                logger.warning("Empty architecture_params received.")
                return -float('inf')

            for i, layer_def in enumerate(architecture_params):
                layer_type = layer_def["type"]
                
                if layer_type == "dense":
                    units = layer_def["units"]
                    activation_str = layer_def["activation"]
                    # Dense layer expects flattened input if previous was Conv/LSTM without flatten
                    # For simplicity, assume input is already [batch, features] or will be handled
                    if i > 0 and isinstance(model_layers[-1], (nn.Conv1d, nn.LSTM, nn.GRU)):
                         # Need to know output features of Conv/LSTM to flatten correctly
                         # This part can get complex; for a simple example, we might need a Flatten layer
                         # or ensure the previous layer outputs a suitable shape.
                         # Let's assume for now that if a Dense follows a Conv/LSTM,
                         # the model construction needs to be aware of flattening.
                         # A dedicated Flatten layer in search space would be better.
                         pass # Requires careful handling of dimensions in a real model builder

                    model_layers.append(nn.Linear(current_features, units))
                    current_features = units
                    if activation_str == "relu":
                        model_layers.append(nn.ReLU())
                    elif activation_str == "tanh":
                        model_layers.append(nn.Tanh())
                    elif activation_str == "elu":
                        model_layers.append(nn.ELU())
                
                elif layer_type == "conv1d":
                    out_channels = layer_def["filters"]
                    kernel_size = layer_def["kernel_size"]
                    # Conv1d expects (batch, channels, seq_len)
                    # If first layer, current_features is input_channels.
                    # If not first, previous layer must output compatible shape.
                    # This example assumes input_features is channels for the first conv layer.
                    # For subsequent conv layers, current_features would be out_channels of prev conv.
                    
                    # Simple check for input shape:
                    # If previous layer was not Conv1D, assume we need to reshape or it's the first layer.
                    # This is a simplification. A real model builder needs robust shape inference.
                    in_channels = current_features if i == 0 or not isinstance(model_layers[-1], (nn.LSTM, nn.GRU)) else model_layers[-1].out_channels # Placeholder
                    
                    model_layers.append(nn.Conv1d(in_channels=current_features, 
                                                  out_channels=out_channels, 
                                                  kernel_size=kernel_size,
                                                  padding=(kernel_size // 2))) # Preserve length
                    current_features = out_channels # Now features are the out_channels
                    # Sequence length might change if padding is not 'same' or if striding > 1
                    # current_seq_len = (current_seq_len - kernel_size + 2 * (kernel_size//2)) + 1 # Example for 'valid' padding

                    if layer_def.get("activation") == "relu": model_layers.append(nn.ReLU())
                    elif layer_def.get("activation") == "tanh": model_layers.append(nn.Tanh())

                elif layer_type == "lstm" or layer_type == "gru":
                    hidden_size = layer_def["hidden_size"]
                    num_layers_recurrent = layer_def["num_layers"]
                    # LSTM/GRU expects (seq_len, batch, features) or (batch, seq_len, features) if batch_first=True
                    # For simplicity, let's assume batch_first=True
                    # current_features is input_size for LSTM/GRU
                    RecurrentLayer = nn.LSTM if layer_type == "lstm" else nn.GRU
                    model_layers.append(RecurrentLayer(input_size=current_features, 
                                                       hidden_size=hidden_size, 
                                                       num_layers=num_layers_recurrent, 
                                                       batch_first=True))
                    current_features = hidden_size # Output features are hidden_size

                elif layer_type == "batchnorm1d":
                     # BatchNorm1d can be applied to Conv1D output (expects num_features = channels)
                     # or Dense output (expects num_features = output units of dense)
                    model_layers.append(nn.BatchNorm1d(num_features=current_features))

                elif layer_type == "dropout":
                    model_layers.append(nn.Dropout(layer_def["rate"]))
                
                else:
                    logger.warning(f"Unsupported layer type in architecture: {layer_type}")
                    return -float('inf')

            # Add a final layer to match output_size, assuming current_features is the input to this final layer
            # This is a common requirement for supervised learning.
            # If the last layer from search space is not dense, or doesn't match output_size, add one.
            if not model_layers or not isinstance(model_layers[-1], nn.Linear) or current_features != output_size:
                 # If last layer was recurrent, its output is (batch, seq_len, hidden_size).
                 # We typically take the last time step's output: output[:, -1, :]
                 # This part needs to be handled carefully. For simplicity, assume current_features is correct.
                 # A more robust solution would be to have a specific "output_layer" in search space or
                 # a more intelligent model builder.
                model_layers.append(nn.Linear(current_features, output_size))
            
            if context.get("final_activation") == "sigmoid": model_layers.append(nn.Sigmoid())
            elif context.get("final_activation") == "softmax": model_layers.append(nn.Softmax(dim=-1))
            # No final activation for regression usually, or handled by loss function (e.g. BCEWithLogitsLoss)

            model = nn.Sequential(*model_layers).to(device)
            logger.debug(f"Constructed model: {model}")

        except Exception as e:
            logger.error(f"Error constructing model from architecture {architecture_params}: {e}", exc_info=True)
            return -float('inf') # Penalize architectures that can't be built

        # --- Simulate Training/Evaluation ---
        try:
            # Dummy data
            # For Conv1D/LSTM: (batch_size, channels/features, sequence_length) or (batch_size, sequence_length, features)
            # For Dense: (batch_size, features)
            # This needs to be consistent with how the first layer expects input.
            # Assuming first layer handles input_features correctly.
            # If first layer is Conv1D: (batch, input_features, seq_len)
            # If first layer is LSTM/GRU: (batch, seq_len, input_features)
            # If first layer is Dense: (batch, input_features) - seq_len might be implicitly part of features or handled by preprocessing
            
            batch_size = context.get("batch_size", 4)
            
            # Adjust input shape based on the first layer type
            first_layer_type = architecture_params[0]["type"] if architecture_params else None
            if first_layer_type in ["lstm", "gru"]:
                dummy_input = torch.randn(batch_size, sequence_length, input_features).to(device)
            elif first_layer_type == "conv1d":
                 dummy_input = torch.randn(batch_size, input_features, sequence_length).to(device)
            else: # Default to Dense-like input or let the model handle it
                dummy_input = torch.randn(batch_size, input_features).to(device) # This might need adjustment if seq_len is used by Dense

            dummy_target = torch.randn(batch_size, output_size).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=context.get("learning_rate", 1e-3))
            loss_fn = nn.MSELoss()

            num_epochs = context.get("num_epochs_eval", 3) # Small number of epochs for quick eval
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                # Handle potential mismatch between LSTM/Conv output and Dense input within Sequential
                # This is where a proper model builder that handles flattening or reshaping is crucial.
                # For this example, we assume the sequence of layers is valid or errors out.
                
                # If the model contains recurrent layers, its output might be (output, (hn, cn))
                # We need to extract the actual output tensor.
                output = model(dummy_input)
                if isinstance(output, tuple): # e.g. LSTM/GRU output
                    output = output[0] # Get the sequence of hidden states

                # If output is (batch, seq_len, features) from LSTM/Conv1D and target is (batch, features)
                # we might need to select the last time step or average over time.
                if output.dim() == 3 and dummy_target.dim() == 2:
                    output = output[:, -1, :] # Common practice: use last time step output for sequence tasks

                # Ensure output and target shapes match for loss calculation
                if output.shape != dummy_target.shape:
                    logger.warning(f"Shape mismatch! Output: {output.shape}, Target: {dummy_target.shape}. Adjusting target or output for loss.")
                    # Attempt a simple fix if possible, e.g. if output has an extra dim of 1
                    if output.shape == (dummy_target.shape[0], dummy_target.shape[1], 1) and dummy_target.dim() == 2 :
                        output = output.squeeze(-1)
                    elif output.shape[0] == dummy_target.shape[0] and output.numel() // output.shape[0] == dummy_target.numel() // dummy_target.shape[0]:
                         # If batch size matches and total features per batch item match, try reshaping output
                        logger.info(f"Attempting to reshape output from {output.shape} to {dummy_target.shape}")
                        try:
                            output = output.view(dummy_target.shape)
                        except RuntimeError as reshape_err:
                            logger.error(f"Could not reshape output to target shape. Error: {reshape_err}")
                            return -float('inf')
                    else:
                        logger.error(f"Cannot resolve shape mismatch between output {output.shape} and target {dummy_target.shape}")
                        return -float('inf')


                loss = loss_fn(output, dummy_target)
                loss.backward()
                optimizer.step()
                logger.debug(f"Epoch {epoch+1}, Loss: {loss.item()}")

            final_loss = loss.item()
            # Fitness: higher is better, so use negative loss. Add penalty for complexity if desired.
            # For example: fitness = -final_loss - 0.01 * sum(p.numel() for p in model.parameters())
            fitness = -final_loss 
            logger.info(f"Architecture evaluated. Fitness (neg_loss): {fitness:.4f}")

        except Exception as e:
            logger.error(f"Error during training/evaluation of model: {e}", exc_info=True)
            return -float('inf') # Penalize architectures that cause errors during training

        return fitness

    # --- Define a Search Space for Quant Models ---
    example_search_space_quant = {
        "min_layers": 2, 
        "max_layers": 6, 
        "layer_types": {
            "dense": {
                "units": {"type": "int_range", "min": 32, "max": 256, "step": 32},
                "activation": {"type": "choice", "values": ["relu", "tanh", "elu"]}
            },
            "conv1d": { # For sequence data if features are channels
                "filters": {"type": "int_range", "min": 16, "max": 64, "step": 16},
                "kernel_size": {"type": "choice", "values": [3, 5, 7]},
                "activation": {"type": "choice", "values": ["relu", "tanh", None]} # Activation can be separate
            },
            "lstm": {
                "hidden_size": {"type": "int_range", "min": 32, "max": 128, "step": 32},
                "num_layers": {"type": "int_range", "min": 1, "max": 2, "step": 1} # Stacked LSTMs
            },
            "gru": {
                "hidden_size": {"type": "int_range", "min": 32, "max": 128, "step": 32},
                "num_layers": {"type": "int_range", "min": 1, "max": 2, "step": 1}
            },
            "batchnorm1d": { 
                # No specific params here, applies to previous layer's features
            },
            "dropout": {
                "rate": {"type": "float_range", "min": 0.1, "max": 0.5, "step": 0.05}
            }
            # Could add "attention", "flatten", "pooling" layers etc.
        },
        # "output_layer_activation": {"type": "choice", "values": ["sigmoid"]} # If needed for classification
    }

    nas_optimizer_quant = NeuralArchitectureSearch(
        fitness_function=example_nas_fitness_function,
        search_space=example_search_space_quant,
        logger=logger,
        nas_settings={ 
            "population_size": 10, # Smaller for quicker example run
            "num_generations": 5,  # Fewer generations
            "mutation_rate": 0.3,
            "gene_mutation_probability": 0.2,
            "crossover_rate": 0.7,
            "elitism_count": 2,
            "tournament_size": 3
        }
    )

    # --- Context for the Fitness Function ---
    run_context_quant = {
        "input_features": 20,   # Number of input features per time step
        "sequence_length": 60,  # Number of time steps in a sequence
        "output_size": 1,       # Predicting a single value (e.g., future price, signal)
        "batch_size": 8,
        "learning_rate": 0.005,
        "num_epochs_eval": 5,   # Number of epochs to train each candidate for evaluation
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # "final_activation": "sigmoid" # if doing binary classification
    }
    logger.info(f"Using device: {run_context_quant['device']}")

    logger.info("\\nRunning full NAS optimizer example for Quant Model...")
    best_arch, best_score = nas_optimizer_quant.run_optimizer(run_context_quant)
    
    if best_arch:
        logger.info(f"NAS run finished. Best Fitness Score: {best_score:.4f}")
        logger.info(f"Best Architecture Found:")
        for i, layer in enumerate(best_arch):
            logger.info(f"  Layer {i+1}: {layer}")
        
        # You could now take 'best_arch' and build/train a full model with it.
        # Example:
        # final_model = build_model_from_architecture(best_arch, run_context_quant)
        # train_final_model(final_model, full_dataset)
    else:
        logger.warning("NAS did not find a suitable architecture.")

