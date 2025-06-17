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
    Conforms to BaseStrategy.forward signature.
    """
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="ReinforcementLearningStrategy",
            description="Uses a pre-trained RL policy model for signals.",
            # input_dim will be set by EnhancedStrategySuperposition based on its config.
            # This default_params['input_dim'] is a fallback if not set at a higher level,
            # but self.config.input_dim should be the source of truth.
            default_params={'action_dim': 3} 
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        
        # self.config.input_dim is the number of features per time step for this strategy.
        # This is provided by EnhancedStrategySuperposition.
        if self.config.input_dim is None:
            self.logger.error(f"{self.config.name}: self.config.input_dim is None. Cannot initialize model. Defaulting to a dummy input_dim of 10.")
            model_feature_input_dim = 10
        else:
            model_feature_input_dim = self.config.input_dim

        self.action_dim = int(self.params.get('action_dim', 3))

        # Define a simple model if no model_path is provided or loading fails
        # This model operates on the last time step of features.
        self.model = nn.Sequential(
            nn.Linear(model_feature_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
        
        model_path = self.params.get('model_path')
        if model_path:
            try:
                loaded_model = joblib.load(model_path)
                if hasattr(loaded_model, 'policy') and isinstance(loaded_model.policy, nn.Module):
                    # Check if the loaded policy's input layer matches model_feature_input_dim
                    first_layer = next(loaded_model.policy.modules())
                    while not isinstance(first_layer, nn.Linear) and list(first_layer.children()):
                        first_layer = next(first_layer.children().__iter__(), None) # Basic way to find first linear
                    
                    if isinstance(first_layer, nn.Linear) and first_layer.in_features == model_feature_input_dim:
                        self.model = loaded_model.policy
                        self.logger.info(f"{self.config.name}: Successfully loaded RL policy model from {model_path} with matching input_dim {model_feature_input_dim}.")
                    else:
                        self.logger.warning(f"{self.config.name}: Loaded RL policy model from {model_path} has input_dim {first_layer.in_features if isinstance(first_layer, nn.Linear) else 'Unknown'}, expected {model_feature_input_dim}. Using default model.")
                elif isinstance(loaded_model, nn.Module):
                    # Similar check for a raw nn.Module
                    first_layer = next(loaded_model.modules())
                    while not isinstance(first_layer, nn.Linear) and list(first_layer.children()):
                         first_layer = next(first_layer.children().__iter__(), None)

                    if isinstance(first_layer, nn.Linear) and first_layer.in_features == model_feature_input_dim:
                        self.model = loaded_model
                        self.logger.info(f"{self.config.name}: Successfully loaded nn.Module model from {model_path} with matching input_dim {model_feature_input_dim}.")
                    else:
                        self.logger.warning(f"{self.config.name}: Loaded nn.Module model from {model_path} has input_dim {first_layer.in_features if isinstance(first_layer, nn.Linear) else 'Unknown'}, expected {model_feature_input_dim}. Using default model.")
                else:
                    self.logger.warning(f"{self.config.name}: Loaded model from {model_path} is not an nn.Module or has no policy attribute. Using default model.")
                
            except Exception as e:
                self.logger.error(f"{self.config.name}: Failed to load model from {model_path}: {e}. Using default model.")
        
        self.model.to(self.params.get('device', 'cpu'))
        self.logger.info(f"{self.config.name}: Initialized. Model input dim: {model_feature_input_dim}. Action dim: {self.action_dim}.")

    def forward(self, asset_features: torch.Tensor, 
                current_positions: Optional[torch.Tensor] = None, 
                timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        """
        Processes asset_features to generate signals.
        asset_features: Tensor of shape (batch_size, sequence_length, num_input_features)
                        where num_input_features == self.config.input_dim.
        current_positions: Optional tensor of shape (batch_size, 1) or (batch_size,)
        Returns:
            A tensor of signals, shape (batch_size, 1, 1) representing action (-1, 0, or 1).
        """
        batch_size, sequence_length, num_input_features = asset_features.shape
        device = asset_features.device

        if num_input_features != self.model[0].in_features:
            self.logger.error(f"{self.config.name}: Mismatch between asset_features dim ({num_input_features}) and model input dim ({self.model[0].in_features}). Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        # Use the latest features from the sequence
        latest_features = asset_features[:, -1, :]  # Shape: (batch_size, num_input_features)

        signals_flat = torch.zeros(batch_size, device=device)
        try:
            with torch.no_grad(): # Typically, strategies don't train themselves during inference
                self.model.eval() # Ensure model is in eval mode
                action_logits = self.model(latest_features) # (batch_size, action_dim)
                action_indices = torch.argmax(action_logits, dim=1) # (batch_size,)

                if self.action_dim == 3: # Sell (-1), Hold (0), Buy (1)
                    signals_flat = action_indices.float() - 1.0
                elif self.action_dim == 2: # Sell (-1), Buy (1)
                    signals_flat = (action_indices.float() * 2.0) - 1.0
                else: # Default to hold if action_dim is not 2 or 3
                    self.logger.warning(f"{self.config.name}: action_dim is {self.action_dim}, which is not 2 or 3. Defaulting to hold signal.")
                    signals_flat = torch.zeros_like(action_indices, dtype=torch.float)
        except Exception as e:
            self.logger.error(f"{self.config.name}: Error during model forward pass: {e}. Returning zero signals.", exc_info=True)
            return torch.zeros((batch_size, 1, 1), device=device)
            
        return signals_flat.reshape(batch_size, 1, 1) # Reshape to (batch_size, num_assets=1, signal_dim=1)

class DeepLearningPredictionStrategy(BaseStrategy):
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="DeepLearningPredictionStrategy",
            description="Uses a deep learning model for prediction and signal generation from features.",
            default_params={
                'lookback_window': 20, 
                'output_dim': 1, # Can be 1 for regression or N for classification
                'model_class_str': None, 
                'model_path': None,
                'threshold_buy': 0.01, # Example threshold for regression output
                'threshold_sell': -0.01 # Example threshold for regression output
            }
            # self.config.input_dim will be set by EnhancedStrategySuperposition
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.lookback_window = int(self.params.get('lookback_window', 20))
        self.output_dim = int(self.params.get('output_dim', 1))
        self.model_class_str = self.params.get('model_class_str')
        self.model_path = self.params.get('model_path')
        
        if self.config.input_dim is None:
            self.logger.error(f"{self.config.name}: self.config.input_dim is None. Cannot initialize model. Defaulting to dummy input_dim_per_step of 10.")
            self.input_dim_per_step = 10
        else:
            self.input_dim_per_step = self.config.input_dim # Features per time step

        self.model_internal_input_dim = self.input_dim_per_step * self.lookback_window

        self.model: nn.Module
        loaded_successfully = False
        if self.model_class_str and self.model_path and os.path.exists(self.model_path): # Check path exists
            try:
                module_path, class_name = self.model_class_str.rsplit('.', 1)
                module = importlib.import_module(module_path)
                ModelClass = getattr(module, class_name)
                
                # Instantiate model - This part needs to be flexible
                # Assuming ModelClass constructor takes input_dim and output_dim
                self.model = ModelClass(input_dim=self.model_internal_input_dim, output_dim=self.output_dim)
                
                state_dict = torch.load(self.model_path, map_location=self.params.get('device', 'cpu'))
                if 'state_dict' in state_dict: 
                    self.model.load_state_dict(state_dict['state_dict'])
                else:
                    self.model.load_state_dict(state_dict)
                
                self.model.to(self.params.get('device', 'cpu'))
                self.model.eval()
                self.logger.info(f"{self.config.name}: Loaded model {self.model_class_str} from {self.model_path} with model_internal_input_dim {self.model_internal_input_dim}.")
                loaded_successfully = True
            except Exception as e:
                self.logger.error(f"{self.config.name}: Error loading model {self.model_class_str} from {self.model_path}: {e}. Using default model.", exc_info=True)
        
        if not loaded_successfully:
            self._create_default_model()
            
        self.logger.info(f"{self.config.name}: Initialized. Lookback: {self.lookback_window}, Output Dim: {self.output_dim}, Features per step: {self.input_dim_per_step}, Model internal input dim: {self.model_internal_input_dim}")

    def _create_default_model(self):
        self.model = nn.Sequential(
            nn.Linear(self.model_internal_input_dim, 128),
            Mish(),
            nn.Linear(128, 64),
            Swish(),
            nn.Linear(64, self.output_dim)
        )
        self.model.to(self.params.get('device', 'cpu'))
        self.model.eval()
        self.logger.info(f"{self.config.name}: Created default MLP model with input_dim={self.model_internal_input_dim}.")

    def forward(self, asset_features: torch.Tensor, 
                current_positions: Optional[torch.Tensor] = None, 
                timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        """
        asset_features: (batch_size, sequence_length, self.input_dim_per_step)
        Returns: (batch_size, 1, 1) signal tensor
        """
        batch_size, sequence_length, features_per_step = asset_features.shape
        device = asset_features.device

        if features_per_step != self.input_dim_per_step:
            self.logger.error(f"{self.config.name}: Mismatch asset_features dim ({features_per_step}) and expected features_per_step ({self.input_dim_per_step}). Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        if sequence_length < self.lookback_window:
            self.logger.warning(f"{self.config.name}: Sequence length ({sequence_length}) is less than lookback window ({self.lookback_window}). Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        # Take the most recent 'lookback_window' timesteps
        model_input_sequence = asset_features[:, -self.lookback_window:, :]
        # Flatten: (batch_size, self.lookback_window * self.input_dim_per_step)
        model_input_flat = model_input_sequence.reshape(batch_size, -1)

        if model_input_flat.shape[1] != self.model_internal_input_dim:
             self.logger.error(f"{self.config.name}: Mismatch between flattened input dim ({model_input_flat.shape[1]}) and model internal input dim ({self.model_internal_input_dim}). Returning zero signal.")
             return torch.zeros((batch_size, 1, 1), device=device)

        signal = torch.zeros(batch_size, device=device)
        try:
            with torch.no_grad():
                self.model.eval()
                prediction = self.model(model_input_flat) # (batch_size, output_dim)

            if self.output_dim == 1: # Regression: predict price change or similar
                threshold_buy = self.params.get('threshold_buy', 0.01)
                threshold_sell = self.params.get('threshold_sell', -0.01)
                signal[prediction[:, 0] > threshold_buy] = 1.0
                signal[prediction[:, 0] < threshold_sell] = -1.0
            elif self.output_dim == 3: # Classification: Sell, Hold, Buy
                action_indices = torch.argmax(prediction, dim=1)
                signal = action_indices.float() - 1.0
            elif self.output_dim == 2: # Classification: Sell, Buy
                 action_indices = torch.argmax(prediction, dim=1)
                 signal = (action_indices.float() * 2.0) - 1.0
            else:
                self.logger.warning(f"{self.config.name}: output_dim is {self.output_dim}. Signal generation logic not defined for this. Defaulting to hold.")
        
        except Exception as e:
            self.logger.error(f"{self.config.name}: Error during model forward pass or signal generation: {e}. Returning zero signals.", exc_info=True)
            return torch.zeros((batch_size, 1, 1), device=device)
            
        return signal.reshape(batch_size, 1, 1)

class EnsembleLearningStrategy(BaseStrategy):
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="EnsembleLearningStrategy",
            description="Combines predictions from multiple models.",
            default_params={
                'model_configs': [], # List of dicts, each with 'model_path', 'model_class_str', 'input_dim_model' (optional, if different from strategy's input_dim)
                'aggregation_method': 'majority_vote', # or 'average' for regression
                # 'strategy_input_dim' is now self.config.input_dim for the features this ensemble strategy receives.
                # Individual models within the ensemble can have their own input_dim specified in model_configs.
            }
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        self.models = nn.ModuleList()
        self.model_configs_params = self.params.get('model_configs', []) # Renamed to avoid clash with self.config
        self.aggregation_method = self.params.get('aggregation_method', 'majority_vote')

        if self.config.input_dim is None:
            self.logger.warning(f"{self.config.name}: self.config.input_dim is None. This is the expected input dim for the ensemble strategy itself. Sub-models might use this or their own configured input_dim.")
            # Defaulting strategy_feature_input_dim for sub-models if self.config.input_dim is None AND sub-model doesn't specify its own.
            default_sub_model_input_dim_fallback = 10 
        else:
            default_sub_model_input_dim_fallback = self.config.input_dim

        for i, model_conf_dict in enumerate(self.model_configs_params):
            model_class_str = model_conf_dict.get('model_class_str')
            model_path = model_conf_dict.get('model_path')
            
            # Sub-model input_dim: use model_conf_dict['input_dim_model'], fallback to strategy's input_dim (self.config.input_dim), then to default_sub_model_input_dim_fallback.
            sub_model_input_dim = model_conf_dict.get('input_dim_model', default_sub_model_input_dim_fallback)
            sub_model_output_dim = model_conf_dict.get('output_dim_model', 3 if self.aggregation_method == 'majority_vote' else 1)

            model_instance: Optional[nn.Module] = None
            if model_class_str and model_path and os.path.exists(model_path):
                try:
                    module_path, class_name = model_class_str.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    ModelClass = getattr(module, class_name)
                    model_instance = ModelClass(input_dim=sub_model_input_dim, output_dim=sub_model_output_dim)
                    
                    state_dict = torch.load(model_path, map_location=self.params.get('device', 'cpu'))
                    model_instance.load_state_dict(state_dict.get('state_dict', state_dict))
                    model_instance.to(self.params.get('device', 'cpu'))
                    model_instance.eval()
                    self.models.append(model_instance)
                    self.logger.info(f"{self.config.name}: Loaded sub-model {i+1} ({model_class_str}) from {model_path} with input_dim {sub_model_input_dim}.")
                except Exception as e:
                    self.logger.error(f"{self.config.name}: Error loading sub-model {i+1} ({model_class_str}) from {model_path}: {e}. Skipping.", exc_info=True)
            elif model_class_str: # Create default if path not found but class is specified
                try:
                    module_path, class_name = model_class_str.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    ModelClass = getattr(module, class_name)
                    model_instance = ModelClass(input_dim=sub_model_input_dim, output_dim=sub_model_output_dim)
                    model_instance.to(self.params.get('device', 'cpu'))
                    model_instance.eval()
                    self.models.append(model_instance)
                    self.logger.warning(f"{self.config.name}: Path for sub-model {i+1} ({model_class_str}) not found or invalid. Created default instance with input_dim {sub_model_input_dim}.")
                except Exception as e:
                    self.logger.error(f"{self.config.name}: Error creating default instance for sub-model {i+1} ({model_class_str}): {e}. Skipping.", exc_info=True)
            else:
                self.logger.warning(f"{self.config.name}: Sub-model config {i+1} is missing model_class_str. Skipping.")
        
        if not self.models:
            self.logger.warning(f"{self.config.name}: No sub-models were loaded. This strategy will produce zero signals.")
        else:
            self.logger.info(f"{self.config.name}: Initialized with {len(self.models)} sub-models.")

    def forward(self, asset_features: torch.Tensor, 
                current_positions: Optional[torch.Tensor] = None, 
                timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, sequence_length, num_strategy_features = asset_features.shape
        device = asset_features.device

        if not self.models:
            return torch.zeros((batch_size, 1, 1), device=device)

        # The ensemble strategy receives asset_features with num_strategy_features (== self.config.input_dim).
        # Each sub-model might expect a different input_dim, as configured in self.model_configs_params.
        # If a sub-model expects a different dim, it's an issue unless feature selection/transformation is done here.
        # For now, assume sub-models are compatible with num_strategy_features or their config handles it.
        
        latest_features = asset_features[:, -1, :] # (batch_size, num_strategy_features)
        
        all_predictions = []
        for i, model in enumerate(self.models):
            sub_model_conf = self.model_configs_params[i]
            sub_model_expected_input_dim = sub_model_conf.get('input_dim_model', self.config.input_dim if self.config.input_dim is not None else 10)

            if num_strategy_features != sub_model_expected_input_dim:
                self.logger.warning(f"{self.config.name}, sub-model {i} ({sub_model_conf.get('model_class_str')}): Feature mismatch. Strategy provides {num_strategy_features}, model expects {sub_model_expected_input_dim}. Skipping this model.")
                # Create a dummy prediction that won't skew aggregation if possible
                output_dim_model = sub_model_conf.get('output_dim_model', 3 if self.aggregation_method == 'majority_vote' else 1)
                dummy_pred = torch.zeros((batch_size, output_dim_model), device=device)
                if self.aggregation_method == 'majority_vote' and output_dim_model == 3: # sell, hold, buy
                    dummy_pred[:, 1] = 1 # Vote for hold
                all_predictions.append(dummy_pred)
                continue
            
            try:
                with torch.no_grad():
                    model.eval()
                    prediction = model(latest_features) # (batch_size, sub_model_output_dim)
                    all_predictions.append(prediction)
            except Exception as e:
                self.logger.error(f"{self.config.name}: Error in sub-model {i} forward pass: {e}. Using neutral prediction.", exc_info=True)
                output_dim_model = sub_model_conf.get('output_dim_model', 3 if self.aggregation_method == 'majority_vote' else 1)
                error_pred = torch.zeros((batch_size, output_dim_model), device=device)
                if self.aggregation_method == 'majority_vote' and output_dim_model == 3:
                    error_pred[:, 1] = 1 # Vote for hold on error
                all_predictions.append(error_pred)

        if not all_predictions:
            return torch.zeros((batch_size, 1, 1), device=device)

        stacked_predictions = torch.stack(all_predictions, dim=0) # (num_models, batch_size, sub_model_output_dim)
        final_signal = torch.zeros(batch_size, device=device)

        if self.aggregation_method == 'majority_vote':
            # Assumes sub_model_output_dim is 3 (sell, hold, buy) for each model
            if stacked_predictions.shape[-1] != 3:
                self.logger.warning(f"{self.config.name}: Majority vote expects output_dim=3 from sub-models, but got {stacked_predictions.shape[-1]}. Returning zero signal.")
                return torch.zeros((batch_size, 1, 1), device=device)
            action_indices = torch.argmax(stacked_predictions, dim=2) # (num_models, batch_size)
            voted_actions, _ = torch.mode(action_indices, dim=0) # (batch_size,)
            final_signal = voted_actions.float() - 1.0 # Convert 0,1,2 to -1,0,1
        elif self.aggregation_method == 'average':
            if stacked_predictions.shape[-1] != 1:
                self.logger.warning(f"{self.config.name}: Average aggregation expects output_dim=1 (regression) from sub-models, but got {stacked_predictions.shape[-1]}. Returning zero signal.")
                return torch.zeros((batch_size, 1, 1), device=device)
            averaged_predictions = torch.mean(stacked_predictions, dim=0) # (batch_size, 1)
            threshold_buy = self.params.get('threshold_buy', 0.01)
            threshold_sell = self.params.get('threshold_sell', -0.01)
            final_signal[averaged_predictions[:, 0] > threshold_buy] = 1.0
            final_signal[averaged_predictions[:, 0] < threshold_sell] = -1.0
        else:
            self.logger.warning(f"{self.config.name}: Unknown aggregation method '{self.aggregation_method}'. Defaulting to hold.")

        return final_signal.reshape(batch_size, 1, 1)

class TransferLearningStrategy(BaseStrategy):
    @staticmethod
    def default_config() -> StrategyConfig:
        return StrategyConfig(
            name="TransferLearningStrategy",
            description="Uses a pre-trained base model and fine-tunes a new head.",
            default_params={
                'base_model_path': None,
                'base_model_class_str': None, 
                'base_model_output_features': 128, 
                'new_head_output_dim': 3, 
                'freeze_base_model': True,
                'new_head_hidden_layers': [64],
                'lookback_window': 1 # Default to 1, meaning use last time step features for base model
            }
            # self.config.input_dim (features per step) will be used for the base model's input layer.
        )

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params=params, logger=logger)
        
        self.base_model_path = self.params.get('base_model_path')
        self.base_model_class_str = self.params.get('base_model_class_str')
        self.base_model_output_features = int(self.params.get('base_model_output_features', 128))
        self.new_head_output_dim = int(self.params.get('new_head_output_dim', 3))
        self.freeze_base_model = self.params.get('freeze_base_model', True)
        self.new_head_hidden_layers_config = self.params.get('new_head_hidden_layers', [64])
        self.lookback_window = int(self.params.get('lookback_window', 1))

        if self.config.input_dim is None:
            self.logger.error(f"{self.config.name}: self.config.input_dim is None. Cannot initialize base model. Defaulting to dummy input_dim_per_step of 10.")
            self.input_dim_per_step = 10
        else:
            self.input_dim_per_step = self.config.input_dim

        # Input to the base_model will be (self.input_dim_per_step * self.lookback_window)
        self.base_model_internal_input_dim = self.input_dim_per_step * self.lookback_window

        self.base_model_instance: Optional[nn.Module] = None # Renamed to avoid clash with self.model
        self.new_head_instance: Optional[nn.Module] = None # Renamed
        self.model: Optional[nn.Module] = None # This will be the combined model

        loaded_successfully = False
        if self.base_model_class_str and self.base_model_path and os.path.exists(self.base_model_path):
            try:
                module_path, class_name = self.base_model_class_str.rsplit('.', 1)
                module = importlib.import_module(module_path)
                BaseModelClass = getattr(module, class_name)
                # Base model constructor should take its expected input_dim
                self.base_model_instance = BaseModelClass(input_dim=self.base_model_internal_input_dim) 
                
                state_dict = torch.load(self.base_model_path, map_location=self.params.get('device', 'cpu'))
                self.base_model_instance.load_state_dict(state_dict.get('state_dict', state_dict))
                self.logger.info(f"{self.config.name}: Loaded base model {self.base_model_class_str} from {self.base_model_path} with input_dim {self.base_model_internal_input_dim}.")
                loaded_successfully = True
            except Exception as e:
                self.logger.error(f"{self.config.name}: Error loading base model {self.base_model_class_str}: {e}. No model created.", exc_info=True)
        
        if not loaded_successfully and self.base_model_class_str: # Try creating default if path failed but class given
             try:
                module_path, class_name = self.base_model_class_str.rsplit('.', 1)
                module = importlib.import_module(module_path)
                BaseModelClass = getattr(module, class_name)
                self.base_model_instance = BaseModelClass(input_dim=self.base_model_internal_input_dim)
                self.logger.warning(f"{self.config.name}: Path for base model {self.base_model_class_str} failed or not provided. Created default instance with input_dim {self.base_model_internal_input_dim}.")
                loaded_successfully = True
             except Exception as e:
                self.logger.error(f"{self.config.name}: Error creating default base model {self.base_model_class_str}: {e}. No model created.", exc_info=True)

        if self.base_model_instance:
            if self.freeze_base_model:
                for param_item in self.base_model_instance.parameters(): # param_item to avoid clash
                    param_item.requires_grad = False
                self.logger.info(f"{self.config.name}: Froze base model parameters.")
            
            head_layers_list = []
            current_head_dim = self.base_model_output_features
            for hidden_dim_item in self.new_head_hidden_layers_config: # hidden_dim_item
                head_layers_list.append(nn.Linear(current_head_dim, hidden_dim_item))
                head_layers_list.append(nn.ReLU())
                current_head_dim = hidden_dim_item
            head_layers_list.append(nn.Linear(current_head_dim, self.new_head_output_dim))
            self.new_head_instance = nn.Sequential(*head_layers_list)
            
            # Combine base model and new head
            # This assumes base_model_instance is a feature extractor and new_head_instance is a classifier/regressor.
            # If base_model_instance itself has multiple stages, this might need adjustment.
            self.model = nn.Sequential(self.base_model_instance, self.new_head_instance)
            self.model.to(self.params.get('device', 'cpu'))
            self.logger.info(f"{self.config.name}: Initialized with base model and new head (output_dim={self.new_head_output_dim}). Base model input dim: {self.base_model_internal_input_dim}")
        else:
            self.logger.error(f"{self.config.name}: Base model could not be loaded or created. Transfer learning strategy will not function correctly.")
            # Create a dummy model to prevent runtime errors if self.model is accessed
            dummy_input_dim = self.input_dim_per_step * self.lookback_window
            self.model = nn.Linear(dummy_input_dim, self.new_head_output_dim) 
            self.model.to(self.params.get('device', 'cpu'))

    def forward(self, asset_features: torch.Tensor, 
                current_positions: Optional[torch.Tensor] = None, 
                timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor:
        batch_size, sequence_length, features_per_step = asset_features.shape
        device = asset_features.device

        if not self.model:
             self.logger.error(f"{self.config.name}: Model not initialized. Returning zero signal.")
             return torch.zeros((batch_size, 1, 1), device=device)
        
        if features_per_step != self.input_dim_per_step:
            self.logger.error(f"{self.config.name}: Mismatch asset_features dim_per_step ({features_per_step}) and expected input_dim_per_step ({self.input_dim_per_step}). Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        if sequence_length < self.lookback_window:
            self.logger.warning(f"{self.config.name}: Sequence length ({sequence_length}) is less than lookback window ({self.lookback_window}). Returning zero signal.")
            return torch.zeros((batch_size, 1, 1), device=device)

        model_input_sequence = asset_features[:, -self.lookback_window:, :]
        model_input_flat = model_input_sequence.reshape(batch_size, -1) # (batch_size, self.lookback_window * self.input_dim_per_step)

        if model_input_flat.shape[1] != self.base_model_internal_input_dim:
             self.logger.error(f"{self.config.name}: Mismatch between flattened input dim ({model_input_flat.shape[1]}) and base model internal input dim ({self.base_model_internal_input_dim}). Returning zero signal.")
             return torch.zeros((batch_size, 1, 1), device=device)

        signal = torch.zeros(batch_size, device=device)
        try:
            # Set grad enabled based on whether the base model is frozen and if the head is being trained.
            # For pure inference, no_grad() is usually appropriate.
            with torch.no_grad(): 
                self.model.eval()
                logits = self.model(model_input_flat) # (batch_size, new_head_output_dim)

            action_indices = torch.argmax(logits, dim=1)
            if self.new_head_output_dim == 3: # Sell, Hold, Buy
                signal = action_indices.float() - 1.0
            elif self.new_head_output_dim == 2: # Sell, Buy
                signal = (action_indices.float() * 2.0) - 1.0
            else: # Regression or other classification
                # If regression (output_dim=1), might compare to thresholds
                if self.new_head_output_dim == 1:
                    threshold_buy = self.params.get('threshold_buy', 0.01)
                    threshold_sell = self.params.get('threshold_sell', -0.01)
                    signal[logits[:, 0] > threshold_buy] = 1.0
                    signal[logits[:, 0] < threshold_sell] = -1.0
                else:
                    self.logger.warning(f"{self.config.name}: new_head_output_dim is {self.new_head_output_dim}. Signal logic might need adjustment. Defaulting to hold for unhandled cases.")
        except Exception as e:
            self.logger.error(f"{self.config.name}: Error during model forward pass: {e}. Returning zero signal.", exc_info=True)
            return torch.zeros((batch_size, 1, 1), device=device)

        return signal.reshape(batch_size, 1, 1)
