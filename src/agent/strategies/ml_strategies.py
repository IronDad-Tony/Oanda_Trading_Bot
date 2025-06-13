# src/agent/strategies/ml_strategies.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F # Added F
import sys # Added for dynamic module loading
import logging # Added for logger
from .base_strategy import BaseStrategy, StrategyConfig
from typing import Dict, List, Any, Optional, Tuple # Added Optional, Tuple
import importlib # Added for dynamic import of strategy classes
import joblib # Added for joblib import
import os # ADDED for os.path.exists

# Activation functions that might be used by internal PyTorch models
class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor: # Added type hints
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor: # Added type hints
        return x * torch.tanh(F.softplus(x)) # Use F.softplus

class ReinforcementLearningStrategy(BaseStrategy):
    """
    Reinforcement Learning Strategy.
    The forward method now processes a tensor and directly outputs a signal tensor.
    """
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="ReinforcementLearningStrategy",
            description="Uses a pre-trained RL policy model for signals.",
            default_params={'input_dim': 10, 'action_dim': 3, 'close_idx': 3} # Example dimensions, added close_idx
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.input_dim = int(self.params.get('input_dim', 10)) # Default from original simple model
        self.action_dim = int(self.params.get('action_dim', 3)) # Default from original simple model
        self.feature_indices = self.params.get('feature_indices', []) # Use specific feature indices

        # Define a simple model if no model_path is provided or loading fails
        self.model = nn.Sequential(
            nn.Linear(len(self.feature_indices) if self.feature_indices else self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
        
        model_path = self.params.get('model_path')
        if model_path:
            try:
                # Attempt to load a pre-trained model (e.g., from stable-baselines3 or joblib)
                # This part is highly dependent on the RL library used for training
                loaded_model = joblib.load(model_path)
                # Assuming loaded_model is compatible or provides a policy network
                if hasattr(loaded_model, 'policy') and isinstance(loaded_model.policy, nn.Module):
                    self.model = loaded_model.policy
                elif isinstance(loaded_model, nn.Module):
                    self.model = loaded_model
                else:
                    self.logger.warning(f"{self.config.name}: Loaded model from {model_path} is not an nn.Module or has no policy attribute. Using default model.")
                self.model.to(self.params.get('device', 'cpu')) # Ensure model is on correct device
            except Exception as e:
                self.logger.error(f"{self.config.name}: Failed to load model from {model_path}: {e}. Using default model.")
        
        self.logger.info(f"{self.config.name}: Initialized. Model input based on {len(self.feature_indices) if self.feature_indices else self.input_dim} features. Action dim: {self.action_dim}.")

    def forward(self, price_data_batch: torch.Tensor, 
                feature_data_batch: torch.Tensor, 
                portfolio_composition_batch: Optional[torch.Tensor] = None, 
                market_state_batch: Optional[torch.Tensor] = None, 
                current_positions_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Processes feature_data_batch to generate signals for each asset in the batch.
        Args:
            price_data_batch: Tensor of shape (batch_size, num_assets, seq_len, num_ohlcv_features)
            feature_data_batch: Tensor of shape (batch_size, num_assets, seq_len, num_input_features)
            portfolio_composition_batch: Optional tensor of shape (batch_size, num_assets)
            market_state_batch: Optional tensor of shape (batch_size,)
            current_positions_batch: Optional tensor of shape (batch_size, num_assets)

        Returns:
            A tensor of signals, shape (batch_size, num_assets).
        """
        batch_size, num_assets, seq_len, num_input_features = feature_data_batch.shape
        
        # Use the latest features from the sequence for each asset
        # If specific feature_indices are provided, select them. Otherwise, use all features up to self.input_dim.
        if self.feature_indices:
            if not all(idx < num_input_features for idx in self.feature_indices):
                self.logger.warning(f"{self.config.name}: One or more feature_indices are out of bounds. Max index: {num_input_features-1}. Using all features up to input_dim or num_input_features.")
                selected_features = feature_data_batch[:, :, -1, :min(self.input_dim, num_input_features)] # (batch_size, num_assets, selected_features)
            else:
                selected_features = feature_data_batch[:, :, -1, self.feature_indices] # (batch_size, num_assets, len(feature_indices))
        else:
            # Use the first self.input_dim features if no specific indices are given
            selected_features = feature_data_batch[:, :, -1, :min(self.input_dim, num_input_features)] # (batch_size, num_assets, selected_features)

        if selected_features.shape[-1] == 0:
            self.logger.warning(f"{self.config.name}: No features selected. Returning zero signals.")
            return torch.zeros(batch_size, num_assets, device=feature_data_batch.device)

        # Reshape for model: (batch_size * num_assets, num_selected_features)
        model_input = selected_features.reshape(batch_size * num_assets, -1)

        if model_input.shape[-1] != self.model[0].in_features:
             self.logger.warning(f"{self.config.name}: Model expected {self.model[0].in_features} input features, but got {model_input.shape[-1]}. Re-initializing a new default model or returning zeros.")
             # Option 1: Re-initialize model (if allowed and makes sense)
             # self.model[0] = nn.Linear(model_input.shape[-1], self.model[0].out_features).to(model_input.device)
             # self.model.to(model_input.device) # Ensure whole model is on device
             # Option 2: Return zeros
             return torch.zeros(batch_size, num_assets, device=feature_data_batch.device)


        signals_flat = torch.zeros(batch_size * num_assets, device=feature_data_batch.device)
        try:
            with torch.no_grad():
                action_logits = self.model(model_input) # (batch_size * num_assets, action_dim)
                action_indices = torch.argmax(action_logits, dim=1) # (batch_size * num_assets,)

                if self.action_dim == 3: # Sell, Hold, Buy
                    signals_flat = action_indices.float() - 1.0
                elif self.action_dim == 2: # Sell, Buy
                    signals_flat = (action_indices.float() * 2.0) - 1.0
                else: # Default to hold
                    signals_flat = torch.zeros_like(action_indices, dtype=torch.float)
        except Exception as e:
            self.logger.error(f"{self.config.name}: Error during model forward pass: {e}. Returning zero signals.")
            return torch.zeros(batch_size, num_assets, device=feature_data_batch.device)
            
        return signals_flat.reshape(batch_size, num_assets)

class DeepLearningPredictionStrategy(BaseStrategy):
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="DeepLearningPredictionStrategy",
            description="Uses a deep learning model for prediction and signal generation from features.",
            default_params={
                'lookback_window': 20, 
                'output_dim': 1, # Can be 1 for regression (e.g. price change) or N for classification (e.g. N action classes)
                'model_class_str': None, # e.g., 'mymodule.MyCustomModel'
                'model_path': None, # Path to pre-trained model state_dict
                'feature_indices': [] # List of feature indices to use from feature_data_batch
            }
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.lookback_window = int(self.params.get('lookback_window', 20))
        self.output_dim = int(self.params.get('output_dim', 1))
        self.model_class_str = self.params.get('model_class_str')
        self.model_path = self.params.get('model_path')
        self.feature_indices = self.params.get('feature_indices', [])
        
        num_selected_features = len(self.feature_indices) if self.feature_indices else self.params.get('input_features_dim', 10) # Fallback if not using feature_indices

        self.model: nn.Module
        if self.model_class_str and self.model_path:
            try:
                module_path, class_name = self.model_class_str.rsplit('.', 1)
                module = importlib.import_module(module_path)
                ModelClass = getattr(module, class_name)
                # Instantiate model - assumes ModelClass constructor takes relevant dims
                # This part needs to be flexible based on how ModelClass is defined
                # Example: self.model = ModelClass(input_dim=num_selected_features * self.lookback_window, output_dim=self.output_dim)
                self.model = ModelClass(input_dim=num_selected_features * self.lookback_window, output_dim=self.output_dim) # Adjust as per actual model constructor
                
                state_dict = torch.load(self.model_path, map_location=self.params.get('device', 'cpu'))
                if 'state_dict' in state_dict: # Common practice to save optimizer state too
                    self.model.load_state_dict(state_dict['state_dict'])
                else:
                    self.model.load_state_dict(state_dict)
                self.model.to(self.params.get('device', 'cpu'))
                self.model.eval()
                self.logger.info(f"{self.config.name}: Loaded model {self.model_class_str} from {self.model_path}.")
            except Exception as e:
                self.logger.error(f"{self.config.name}: Error loading model {self.model_class_str} from {self.model_path}: {e}. Using default model.")
                self._create_default_model(num_selected_features)
        else:
            self._create_default_model(num_selected_features)
            
        self.logger.info(f"{self.config.name}: Initialized. Lookback: {self.lookback_window}, Output Dim: {self.output_dim}, Num Selected Features: {num_selected_features}")

    def _create_default_model(self, num_input_features_per_step: int):
        # Default model: Simple MLP operating on flattened lookback window of selected features
        # Input to MLP will be (num_input_features_per_step * self.lookback_window)
        mlp_input_dim = num_input_features_per_step * self.lookback_window
        self.model = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            Mish(),
            nn.Linear(128, 64),
            Swish(),
            nn.Linear(64, self.output_dim)
        )
        self.model.to(self.params.get('device', 'cpu'))
        self.model.eval()
        self.logger.info(f"{self.config.name}: Created default MLP model with input_dim={mlp_input_dim}.")

    def forward(self, price_data_batch: torch.Tensor, 
                feature_data_batch: torch.Tensor, 
                portfolio_composition_batch: Optional[torch.Tensor] = None, 
                market_state_batch: Optional[torch.Tensor] = None, 
                current_positions_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Processes feature_data_batch to generate signals.
        Args:
            price_data_batch: (batch_size, num_assets, seq_len, num_ohlcv_features)
            feature_data_batch: (batch_size, num_assets, seq_len, num_input_features)
            ... other args
        Returns:
            Signals tensor of shape (batch_size, num_assets).
        """
        batch_size, num_assets, seq_len, num_input_features = feature_data_batch.shape
        device = feature_data_batch.device

        if seq_len < self.lookback_window:
            self.logger.debug(f"{self.config.name}: Insufficient data ({seq_len} points) for lookback {self.lookback_window}. Returning zero signals.")
            return torch.zeros(batch_size, num_assets, device=device)

        # Select features based on self.feature_indices or use all if empty
        if self.feature_indices:
            if not all(idx < num_input_features for idx in self.feature_indices):
                self.logger.warning(f"{self.config.name}: Feature indices out of bounds. Returning zero signals.")
                return torch.zeros(batch_size, num_assets, device=device)
            current_features = feature_data_batch[:, :, :, self.feature_indices] # (B, A, S, F_selected)
        else:
            # If no indices, use all features. The model's input_dim should match this.
            current_features = feature_data_batch # (B, A, S, F_all)
        
        num_selected_features = current_features.shape[-1]
        
        # We need the last 'lookback_window' steps for each asset
        # Input to model: (batch_size * num_assets, lookback_window * num_selected_features)
        # current_features is (B, A, S, F_selected)
        # We take last lookback_window steps: current_features[:, :, -self.lookback_window:, :]
        # This gives (B, A, L, F_selected)
        lookback_features = current_features[:, :, -self.lookback_window:, :]
        
        # Flatten the lookback_window and num_selected_features dimensions
        # (B, A, L * F_selected)
        model_input_flat_per_asset = lookback_features.reshape(batch_size, num_assets, -1)
        
        # Reshape for batch processing by the MLP: (B * A, L * F_selected)
        model_input_batched = model_input_flat_per_asset.reshape(batch_size * num_assets, -1)

        if model_input_batched.shape[-1] != self.model[0].in_features:
            self.logger.warning(f"{self.config.name}: Model input dim mismatch. Expected {self.model[0].in_features}, got {model_input_batched.shape[-1]}. Re-creating default model or returning zeros.")
            # Potentially re-create model or return zeros
            # self._create_default_model(num_selected_features) # This would re-init the model
            # if model_input_batched.shape[-1] != self.model[0].in_features: # Check again after potential re-init
            return torch.zeros(batch_size, num_assets, device=device)


        predictions_flat = torch.zeros(batch_size * num_assets, self.output_dim, device=device)
        try:
            with torch.no_grad():
                predictions_flat = self.model(model_input_batched) # (B * A, output_dim)
        except Exception as e:
            self.logger.error(f"{self.config.name}: Error during model forward pass: {e}. Returning zero signals.")
            return torch.zeros(batch_size, num_assets, device=device)

        signals_flat: torch.Tensor
        if self.output_dim == 1: # Regression
            signals_flat = torch.sign(predictions_flat.squeeze(-1)) # (B * A,)
        elif self.output_dim == 3: # Classification: [sell, hold, buy]
            action_indices = torch.argmax(predictions_flat, dim=1) # (B * A,)
            signals_flat = action_indices.float() - 1.0
        else:
            signals_flat = torch.zeros(predictions_flat.shape[0], device=device)
        
        return signals_flat.reshape(batch_size, num_assets)


class EnsembleLearningStrategy(BaseStrategy):
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="EnsembleLearningStrategy",
            description="Combines signals from multiple base strategies using PyTorch.",
            default_params={
                'base_strategy_configs': [], 
                'combination_logic': 'sum', # 'sum', 'average', 'majority_vote'
                'close_idx': 3 # Default for sub-strategies if not specified by them
            },
            applicable_assets=[]
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        
        self.base_strategy_configs_from_params: List[Dict[str, Any]] = self.params.get('base_strategy_configs', [])
        self.combination_logic = self.params.get('combination_logic', 'average') # Changed default to 'average'
        self.weights: Optional[torch.Tensor] = None # Store weights as a tensor if applicable
        
        # Extract weights if provided in config, matching the order of base_strategies_modules
        # This assumes weights are provided in the 'params' of the EnsembleLearningStrategy config
        strategy_weights_list = self.params.get('strategy_weights', [])

        self.base_strategies_modules = nn.ModuleList()
        
        self.logger.info(f"[{self.config.name}] Initializing. Found {len(self.base_strategy_configs_from_params)} base strategy configurations. Combination: {self.combination_logic}")

        # Simplified sub-strategy loading logic (assuming strategy classes are accessible)
        # For a robust solution, a strategy registry or more sophisticated dynamic import is needed.
        import importlib # Ensure importlib is available

        loaded_strategies_count = 0
        temp_weights = []

        for i, bs_conf_item in enumerate(self.base_strategy_configs_from_params):
            if not isinstance(bs_conf_item, dict):
                self.logger.warning(f"Base strategy config item #{i} is not a dict: {bs_conf_item}. Skipping.")
                continue

            strategy_class_path = bs_conf_item.get('class_path') # Expecting 'module.submodule.ClassName'
            strategy_instance_name = bs_conf_item.get('name', f"SubStrategy_{i}")
            sub_strategy_init_params = bs_conf_item.get('params', {})
            
            # Get weight for this strategy
            current_strategy_weight = 1.0 # Default weight if not specified
            if strategy_weights_list and i < len(strategy_weights_list):
                current_strategy_weight = strategy_weights_list[i]
            elif 'weight' in bs_conf_item: # Fallback to weight in sub_strategy_config item itself
                current_strategy_weight = bs_conf_item['weight']


            if not strategy_class_path:
                self.logger.warning(f"Base strategy config item #{i} missing 'class_path': {bs_conf_item}. Skipping.")
                continue

            try:
                module_path, class_name = strategy_class_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                SubStrategyClass = getattr(module, class_name)

                # Create config for sub-strategy
                # Sub-strategy's default_config provides the base
                sub_default_config: StrategyConfig = SubStrategyClass.default_config()
                # Override name
                sub_default_config.name = strategy_instance_name 
                # Sub-strategies inherit applicable_assets from ensemble if not specified by them
                if not sub_default_config.applicable_assets and self.config.applicable_assets:
                    sub_default_config.applicable_assets = self.config.applicable_assets
                
                # The sub_strategy_init_params will be passed to the SubStrategyClass constructor,
                # which via BaseStrategy.__init__ will merge them with sub_default_config.default_params.

                sub_strategy_instance = SubStrategyClass(
                    config=sub_default_config,
                    params=sub_strategy_init_params, # These are the overrides for the sub-strategy
                    logger=self.logger
                )
                self.base_strategies_modules.append(sub_strategy_instance)
                temp_weights.append(current_strategy_weight)
                loaded_strategies_count += 1
                self.logger.info(f"Successfully loaded and added sub-strategy: {strategy_instance_name} ({strategy_class_path}) with weight {current_strategy_weight}")

            except Exception as e:
                self.logger.error(f"Error loading sub-strategy {strategy_instance_name} ({strategy_class_path}): {e}", exc_info=True)

        if loaded_strategies_count > 0 and temp_weights:
            self.weights = torch.tensor(temp_weights, dtype=torch.float, device=self.params.get('device', 'cpu'))
            if self.combination_logic == 'average' or self.combination_logic == 'weighted_average':
                self.weights = self.weights / self.weights.sum() # Normalize for average
            self.logger.info(f"Finalized strategy weights: {self.weights.tolist()}")
        elif loaded_strategies_count == 0:
             self.logger.warning(f"{self.config.name}: No sub-strategies were loaded.")


    def forward(self, price_data_batch: torch.Tensor, 
                feature_data_batch: torch.Tensor, 
                portfolio_composition_batch: Optional[torch.Tensor] = None, 
                market_state_batch: Optional[torch.Tensor] = None, 
                current_positions_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Combines signals from base strategies.
        All input tensors are (batch_size, num_assets, ...).
        Output signals should be (batch_size, num_assets).
        """
        batch_size, num_assets, _, _ = feature_data_batch.shape # Assuming feature_data_batch is always provided
        device = feature_data_batch.device

        if not self.base_strategies_modules:
            self.logger.warning(f"{self.config.name}: No base strategies loaded. Returning zero signals.")
            return torch.zeros(batch_size, num_assets, device=device)

        all_signals = []
        for strategy_module in self.base_strategies_modules:
            try:
                # Each sub-strategy should also conform to the 5-argument forward pass
                sub_signal = strategy_module.forward(
                    price_data_batch, 
                    feature_data_batch, 
                    portfolio_composition_batch, 
                    market_state_batch, 
                    current_positions_batch
                )
                # Ensure sub_signal is (batch_size, num_assets)
                if sub_signal.shape != (batch_size, num_assets):
                    self.logger.warning(f"Sub-strategy {strategy_module.config.name} produced signal of shape {sub_signal.shape}, expected {(batch_size, num_assets)}. Attempting to adapt or skipping.")
                    # Simple adaptation: if it's (B*A,), try to reshape. This is risky.
                    if sub_signal.ndim == 1 and sub_signal.numel() == batch_size * num_assets:
                        sub_signal = sub_signal.reshape(batch_size, num_assets)
                    else: # If not adaptable, use zeros for this sub-strategy
                         sub_signal = torch.zeros(batch_size, num_assets, device=device)
                all_signals.append(sub_signal)
            except Exception as e:
                self.logger.error(f"Error in sub-strategy {strategy_module.config.name} forward pass: {e}. Using zero signals for this sub-strategy.", exc_info=True)
                all_signals.append(torch.zeros(batch_size, num_assets, device=device))
        
        if not all_signals:
            return torch.zeros(batch_size, num_assets, device=device)

        # Stack signals: (num_strategies, batch_size, num_assets)
        stacked_signals = torch.stack(all_signals, dim=0)

        if self.combination_logic == 'sum':
            combined_signals = torch.sum(stacked_signals, dim=0)
        elif self.combination_logic == 'average':
            combined_signals = torch.mean(stacked_signals, dim=0)
        elif self.combination_logic == 'weighted_average' or self.combination_logic == 'weighted_sum':
            if self.weights is not None and self.weights.shape[0] == stacked_signals.shape[0]:
                # Reshape weights for broadcasting: (num_strategies, 1, 1)
                w = self.weights.reshape(-1, 1, 1).to(device)
                combined_signals = torch.sum(stacked_signals * w, dim=0)
                if self.combination_logic == 'weighted_average': # Already normalized if sum of weights is 1
                    pass # Weights should be pre-normalized if true weighted average is desired.
                         # If weights don't sum to 1, this is a weighted sum.
            else:
                self.logger.warning(f"{self.config.name}: Weights not properly configured for weighted combination. Falling back to simple average.")
                combined_signals = torch.mean(stacked_signals, dim=0)
        elif self.combination_logic == 'majority_vote':
            # Sign of sum: +1 if sum > 0, -1 if sum < 0, 0 if sum == 0
            combined_signals = torch.sign(torch.sum(torch.sign(stacked_signals), dim=0))
        else: # Default to average
            combined_signals = torch.mean(stacked_signals, dim=0)
            
        return combined_signals.clamp_(-1, 1) # Ensure signals are within [-1, 1]

class TransferLearningStrategy(BaseStrategy):
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="TransferLearningStrategy",
            description="Uses a pre-trained model and fine-tunes it or uses its features.",
            default_params={
                'base_model_path': None,
                'model_class_str': None, # e.g., 'mymodule.MyCustomModel'
                # 'num_ft_layers_to_unfreeze': 0, # Original name
                'n_layers_to_freeze': 0, # Matches test usage, interpreted as layers to unfreeze from end
                'new_output_dim': None, # Matches test usage for the final output dimension
                'output_dim': 1, # Original output_dim if new_output_dim not specified
                'lookback_window': 20,
                'feature_indices': [],
                'model_input_dim': 10 # Added to align with test params for num_selected_features fallback
            }
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.base_model_path = self.params.get('base_model_path')
        self.model_class_str = self.params.get('model_class_str')
        
        self.device = torch.device(self.params.get('device', 'cpu'))

        # Number of final layers to unfreeze/modify. Test uses 'n_layers_to_freeze'.
        # Default config had 'num_ft_layers_to_unfreeze'. Prioritize 'n_layers_to_freeze'.
        self.layers_to_unfreeze_count = int(self.params.get('n_layers_to_freeze', self.params.get('num_ft_layers_to_unfreeze', 0)))
        
        # Final output dimension for the modified model. Test uses 'new_output_dim'.
        # Fallback to 'output_dim' if 'new_output_dim' is not present or None.
        _new_output_dim_param = self.params.get('new_output_dim')
        if _new_output_dim_param is not None:
            self.final_output_dim = int(_new_output_dim_param)
        else:
            self.final_output_dim = int(self.params.get('output_dim', 1))

        self.lookback_window = int(self.params.get('lookback_window', 20))
        self.feature_indices = self.params.get('feature_indices', [])
        
        # Determine num_selected_features for model constructor if applicable
        if self.feature_indices:
            num_selected_features = len(self.feature_indices)
        else:
            # Fallback: 'input_features_dim' or 'model_input_dim' (from test) or default
            num_selected_features = self.params.get('input_features_dim', self.params.get('model_input_dim', 10))

        self.model: nn.Module
        if self.model_class_str and self.base_model_path:
            try:
                module_path, class_name = self.model_class_str.rsplit('.', 1)
                module = importlib.import_module(module_path)
                ModelClass = getattr(module, class_name)
                
                # The ModelClass constructor signature might vary.
                # For transfer learning, often the base model structure is loaded, then adapted.
                # The input_dim for ModelClass here might be for the original model structure.
                # Or, if ModelClass is a wrapper, it might take new head params.
                # The test mock for ModelClass returns mock_dl_model, bypassing these constructor args' direct effect on mock_dl_model.
                constructor_input_dim = num_selected_features * self.lookback_window
                
                # This call in the test will be mocked to return mock_dl_model
                self.model = ModelClass(input_dim=constructor_input_dim, output_dim=self.final_output_dim) 
                
                state_dict_path = self.base_model_path
                if not os.path.exists(state_dict_path): # Check if path exists
                    raise FileNotFoundError(f"Model state_dict not found at {state_dict_path}")

                # Ensure map_location is a string 'cpu' or 'cuda:x' as expected by torch.load sometimes
                map_location_str = self.device.type 
                if self.device.type == 'cuda' and self.device.index is not None:
                    map_location_str = f'cuda:{self.device.index}'

                state_dict = torch.load(state_dict_path, map_location=map_location_str) # MODIFIED
                if 'state_dict' in state_dict:
                    self.model.load_state_dict(state_dict['state_dict'], strict=False)
                else:
                    self.model.load_state_dict(state_dict, strict=False)
                
                self._modify_model_for_transfer() # Called without arguments
                
                self.model.to(self.device)
                if self.layers_to_unfreeze_count > 0: # If any layers are unfrozen, model is for training
                    self.model.train() 
                else: # Otherwise, feature extraction mode
                    self.model.eval()
                self.logger.info(f"{self.config.name}: Loaded and modified base model {self.model_class_str} from {self.base_model_path}.")

            except FileNotFoundError as e: # Specific handling for file not found
                self.logger.error(f"{self.config.name}: {e}. Using default model.")
                self._create_default_transfer_model(num_selected_features)
            except Exception as e:
                self.logger.error(f"{self.config.name}: Error loading/modifying base model: {e}. Using default model.", exc_info=True)
                self._create_default_transfer_model(num_selected_features)
        else:
            self.logger.warning(f"{self.config.name}: model_class_str or base_model_path not provided. Using default model.")
            self._create_default_transfer_model(num_selected_features)

    def _create_default_transfer_model(self, num_input_features_per_step: int):
        # Default model: Simple MLP
        mlp_input_dim = num_input_features_per_step * self.lookback_window
        self.model = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            Mish(), # Assuming Mish is defined
            nn.Linear(128, 64),
            Swish(), # Assuming Swish is defined
            nn.Linear(64, self.final_output_dim) # Use final_output_dim
        )
        self.model.to(self.device)
        self.model.eval() # Default to eval mode
        self.logger.info(f"{self.config.name}: Created default MLP model with input_dim={mlp_input_dim}, output_dim={self.final_output_dim}.")

    def _modify_model_for_transfer(self):
        model_to_modify = self.model 
        # layers_to_unfreeze_count is self.layers_to_unfreeze_count
        # new_output_dim is self.final_output_dim
        # device is self.device

        self.logger.info(f"Modifying model for transfer. Layers to unfreeze (from end): {self.layers_to_unfreeze_count}, New output dim: {self.final_output_dim}, Device: {self.device}")

        if not (hasattr(model_to_modify, 'children') and callable(model_to_modify.children)):
            self.logger.warning(f"Model {type(model_to_modify)} does not have a 'children' method or is not an nn.Module. Cannot modify layers.")
            # Handle case where model_to_modify might be a single nn.Linear layer itself
            if isinstance(model_to_modify, nn.Linear):
                if model_to_modify.out_features != self.final_output_dim:
                    self.logger.info(f"Model is a single Linear layer. Replacing it.")
                    in_features = model_to_modify.in_features
                    self.model = nn.Linear(in_features, self.final_output_dim) # Replace self.model
                # Parameters of a new nn.Linear are requires_grad=True by default
            else:
                return # Cannot modify if not Linear and no children

        children_modules = []
        if hasattr(model_to_modify, 'children') and callable(model_to_modify.children):
            children_modules = list(model_to_modify.children())
        
        num_children = len(children_modules)

        if num_children == 0 and not isinstance(model_to_modify, nn.Linear): # Already handled single Linear above
             self.logger.warning(f"Model {type(model_to_modify)} has no children modules and is not Linear. Modification might be limited.")
             return


        # Freeze/unfreeze layers if there are children_modules
        if num_children > 0:
            for i, child_module in enumerate(children_modules):
                # Freeze layers that are NOT among the final 'self.layers_to_unfreeze_count'
                if i < (num_children - self.layers_to_unfreeze_count):
                    for param in child_module.parameters():
                        param.requires_grad = False
                    self.logger.debug(f"Froze layer {i}: {child_module}")
                else: # Unfreeze the final 'self.layers_to_unfreeze_count' layers
                    for param in child_module.parameters():
                        param.requires_grad = True
                    self.logger.debug(f"Unfroze layer {i}: {child_module}")
            
            # Replace the last layer if applicable
            last_layer_module = children_modules[-1]
            if isinstance(last_layer_module, nn.Linear):
                if last_layer_module.out_features != self.final_output_dim:
                    in_features = last_layer_module.in_features
                    new_final_layer = nn.Linear(in_features, self.final_output_dim)
                    
                    if isinstance(model_to_modify, nn.Sequential):
                        model_to_modify[-1] = new_final_layer # This uses __setitem__
                    else:
                        # Attempt to find by attribute name if not Sequential (more complex)
                        # For simplicity, this example assumes nn.Sequential or direct attribute replacement if known
                        # This part might need to be more robust for general nn.Module containers
                        replaced_by_attr = False
                        for name, module_item in model_to_modify.named_children():
                            if module_item is last_layer_module:
                                setattr(model_to_modify, name, new_final_layer)
                                replaced_by_attr = True
                                break
                        if not replaced_by_attr:
                            self.logger.warning("Could not replace last layer by attribute: model is not Sequential and direct attribute not found/set.")
                    self.logger.info(f"Replaced last layer. Old out: {last_layer_module.out_features}, New out: {self.final_output_dim}")
            else:
                self.logger.warning(f"Last layer is not nn.Linear (type: {type(last_layer_module)}). Cannot automatically change output dim.")
        
        self.model.to(self.device) # Ensure final model is on device

    def forward(self, price_data_batch: torch.Tensor, 
                feature_data_batch: torch.Tensor, 
                portfolio_composition_batch: Optional[torch.Tensor] = None, 
                market_state_batch: Optional[torch.Tensor] = None, 
                current_positions_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Processes feature_data_batch using the (potentially modified) model to generate signals.
        Args:
            price_data_batch: (batch_size, num_assets, seq_len, num_ohlcv_features)
            feature_data_batch: (batch_size, num_assets, seq_len, num_input_features)
            ... other args
        Returns:
            Signals tensor of shape (batch_size, num_assets).
        """
        batch_size, num_assets, seq_len, num_input_features = feature_data_batch.shape
        device = feature_data_batch.device # Use device of input data for consistency

        if seq_len < self.lookback_window:
            self.logger.debug(f"{self.config.name}: Insufficient data ({seq_len} points) for lookback {self.lookback_window}. Returning zero signals.")
            return torch.zeros(batch_size, num_assets, device=device)

        # Select features based on self.feature_indices or use all if empty
        if self.feature_indices:
            if not all(idx < num_input_features for idx in self.feature_indices):
                self.logger.warning(f"{self.config.name}: Feature indices out of bounds (max index: {num_input_features-1}). Returning zero signals.")
                return torch.zeros(batch_size, num_assets, device=device)
            current_features = feature_data_batch[:, :, :, self.feature_indices] # (B, A, S, F_selected)
        else:
            # If no indices, use all features. The model's input_dim should match this.
            current_features = feature_data_batch # (B, A, S, F_all)
        
        num_selected_features = current_features.shape[-1]
        
        # We need the last 'lookback_window' steps for each asset
        # Input to model: (batch_size * num_assets, lookback_window * num_selected_features)
        lookback_features = current_features[:, :, -self.lookback_window:, :]
        
        # Flatten the lookback_window and num_selected_features dimensions
        model_input_flat_per_asset = lookback_features.reshape(batch_size, num_assets, -1)
        
        # Reshape for batch processing by the MLP: (B * A, L * F_selected)
        model_input_batched = model_input_flat_per_asset.reshape(batch_size * num_assets, -1)

        # Check if the model is a Sequential model to access in_features of the first layer
        first_layer_in_features = -1
        if isinstance(self.model, nn.Sequential) and len(self.model) > 0 and hasattr(self.model[0], 'in_features'):
            first_layer_in_features = self.model[0].in_features
        elif hasattr(self.model, 'in_features'): # For single layer models like nn.Linear directly
            first_layer_in_features = self.model.in_features

        if first_layer_in_features != -1 and model_input_batched.shape[-1] != first_layer_in_features:
            self.logger.warning(f"{self.config.name}: Model input dim mismatch. Expected {first_layer_in_features}, got {model_input_batched.shape[-1]}. Returning zeros.")
            return torch.zeros(batch_size, num_assets, device=device)
        elif first_layer_in_features == -1:
            self.logger.warning(f"{self.config.name}: Could not determine model's expected input features. Proceeding with caution.")

        predictions_flat = torch.zeros(batch_size * num_assets, self.final_output_dim, device=device)
        try:
            with torch.no_grad() if not (self.model.training and self.layers_to_unfreeze_count > 0) else torch.enable_grad():
                # If model is in training mode (fine-tuning), grads should be enabled for the forward pass
                # otherwise, no_grad context for inference
                predictions_flat = self.model(model_input_batched) # (B * A, final_output_dim)
        except Exception as e:
            self.logger.error(f"{self.config.name}: Error during model forward pass: {e}. Returning zero signals.", exc_info=True)
            return torch.zeros(batch_size, num_assets, device=device)

        signals_flat: torch.Tensor
        if self.final_output_dim == 1: # Regression: signal is sign of prediction
            signals_flat = torch.sign(predictions_flat.squeeze(-1)) # (B * A,)
        elif self.final_output_dim == 3: # Classification: [sell, hold, buy]
            action_indices = torch.argmax(predictions_flat, dim=1) # (B * A,)
            signals_flat = action_indices.float() - 1.0 # Map 0,1,2 to -1,0,1
        else: # Default to zero signals if output_dim is not 1 or 3
            self.logger.warning(f"{self.config.name}: Unsupported final_output_dim {self.final_output_dim} for signal generation. Returning zero signals.")
            signals_flat = torch.zeros(predictions_flat.shape[0], device=device)
        
        return signals_flat.reshape(batch_size, num_assets)
