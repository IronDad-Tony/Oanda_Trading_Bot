# src/agent/strategies/base_strategy.py
import joblib # ADDED
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field, asdict # MODIFIED: Import asdict
import logging # Import logging
import torch # ADDED: Import torch
import copy # ADDED

# Attempt to import the project-specific logger
try:
    from src.common.logger_setup import logger
except ImportError:
    # Fallback to standard logging if src.common.logger_setup is not found
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO) # Basic config for fallback

@dataclass
class StrategyConfig:
    name: str
    description: str = ""
    risk_level: float = 0.5  # 0.0 to 1.0
    market_regime: str = "all"  # e.g., "trending", "ranging", "all"
    complexity: int = 3  # 1 to 5
    base_performance: float = 0.5  # Expected baseline performance
    # Parameters specific to the strategy type, can be overridden by instance params
    default_params: Dict[str, Any] = field(default_factory=dict)
    # List of assets this strategy is applicable to, if empty, applies to all provided.
    applicable_assets: List[str] = field(default_factory=list)
    # ADDED: To hold strategy-specific parameter overrides at the layer configuration level
    strategy_specific_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    input_dim: Optional[int] = None # ADDED: Input dimension for the strategy

    def copy(self) -> 'StrategyConfig': # ADDED
        return copy.deepcopy(self)

    @staticmethod
    def merge_configs(base: 'StrategyConfig', override: 'StrategyConfig') -> 'StrategyConfig':
        """Merges two StrategyConfig objects. Values from override_config take precedence."""
        
        merged_input_dim = override.input_dim if override.input_dim is not None else base.input_dim

        # Merge default_params
        merged_default_params = base.default_params.copy()
        merged_default_params.update(override.default_params)

        # Merge applicable_assets (override list replaces base list if override is non-empty)
        merged_applicable_assets = override.applicable_assets if override.applicable_assets else base.applicable_assets

        # Merge strategy_specific_params (deep merge for inner dicts)
        merged_strategy_specific_params = base.strategy_specific_params.copy()
        for key, value_override in override.strategy_specific_params.items():
            if key in merged_strategy_specific_params and \
               isinstance(merged_strategy_specific_params[key], dict) and \
               isinstance(value_override, dict):
                inner_merged = merged_strategy_specific_params[key].copy()
                inner_merged.update(value_override)
                merged_strategy_specific_params[key] = inner_merged
            else:
                merged_strategy_specific_params[key] = value_override
            
        return StrategyConfig(
            name=override.name, # Name from override config is preferred
            description=override.description, # Always take override's, even if ""
            risk_level=override.risk_level,
            market_regime=override.market_regime,
            complexity=override.complexity,
            base_performance=override.base_performance,
            default_params=merged_default_params,
            applicable_assets=merged_applicable_assets,
            strategy_specific_params=merged_strategy_specific_params,
            input_dim=merged_input_dim
        )

class BaseStrategy(ABC, torch.nn.Module): # MODIFIED: Inherit from torch.nn.Module
    param_definitions: Dict[str, Dict[str, Any]] = {}

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__() # MODIFIED: Call super().__init__() for torch.nn.Module
        self.config = config
        
        if logger is None:
            # Fallback to a generic logger if none is provided
            self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
            if not self.logger.hasHandlers(): # Avoid adding multiple handlers if already configured
                logging.basicConfig(level=logging.INFO) # Or your preferred default config
                self.logger.info(f"No logger provided to {self.config.name}, using default basicConfig logger.")
        else:
            self.logger = logger

        # Initialize effective_params: Start with config's default_params
        # Ensure default_params from config is a dictionary
        effective_params_temp = {} 
        if hasattr(self.config, 'default_params') and isinstance(self.config.default_params, dict):
            effective_params_temp = self.config.default_params.copy() # Use a copy
        else:
            self.logger.debug(f"Strategy {self.config.name}: config.default_params is not a dict or missing. Starting with empty default_params.")

        # Override/add with instance-specific params, performing type coercion if possible
        if params is not None and isinstance(params, dict):
            for key, value in params.items():
                if key in effective_params_temp: # Key is known from default_params
                    default_value_from_config = effective_params_temp[key]
                    if type(default_value_from_config) != type(value) and value is not None:
                        try:
                            # Attempt to coerce to the type of the default value
                            coerced_value = type(default_value_from_config)(value)
                            effective_params_temp[key] = coerced_value
                            self.logger.debug(f"Coerced param '{key}' from type {type(value)} to {type(default_value_from_config)}.")
                        except (ValueError, TypeError) as e:
                            self.logger.warning(f"Could not coerce param '{key}' (value: {value}) to type {type(default_value_from_config)}. Using original value. Error: {e}")
                            effective_params_temp[key] = value # Use original value if coercion fails
                    else:
                        effective_params_temp[key] = value # Update with new value (same type or None)
                else: # Key is new / not in default_params
                    effective_params_temp[key] = value # Add new param
                    self.logger.warning(f"Strategy {self.config.name}: Received unexpected parameter '{key}' with value '{value}'. It will be included.")

        self.params = effective_params_temp # Store the final effective parameters

        # Load feature_config if feature_config_path is provided
        self.feature_config = None
        feature_config_path = self.params.get('feature_config_path')
        if feature_config_path:
            try:
                self.feature_config = joblib.load(feature_config_path)
                self.logger.info(f"Successfully loaded feature configuration from {feature_config_path}")
            except Exception as e:
                self.logger.error(f"Failed to load feature configuration from {feature_config_path}: {e}")
        # MODIFIED: Ensure self.device is initialized, defaulting to 'cpu' if not specified
        # This should ideally be handled by subclasses or through params,
        # but providing a default here for broader compatibility.
        # However, specific ML strategies should manage their device more explicitly.
        # For BaseStrategy, it might not always need a device, so this is a placeholder.
        # Consider if all strategies truly need a 'device' attribute at this base level.
        # If a strategy uses PyTorch, it should handle its own device management.
        # self.device = torch.device(self.params.get('device', 'cpu')) # Example, might be too specific for BaseStrategy

    @abstractmethod
    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor: # MODIFIED SIGNATURE
        """
        Processes features for a single asset to generate strategy-specific signals or processed features.
        
        Input: 
            asset_features: torch.Tensor 
                Features for a specific asset. 
                Shape: (batch_size, num_features) 
                         or (batch_size, sequence_length, num_features) if handling sequences.
            current_positions: Optional[torch.Tensor] 
                Current positions for the asset. 
                Shape: (batch_size, 1) or similar.
            timestamp: Optional[pd.Timestamp] 
                The current timestamp for this forward pass (can be a single timestamp representative of the batch,
                or if strategies need per-item timestamps, this design might need List[pd.Timestamp] 
                and strategies to handle it).

        Output: 
            torch.Tensor 
                Output signals or processed features for the strategy for the given asset.
                Expected Shape: (batch_size, 1, 1) or (batch_size, 1) for a single signal value per batch item.
        """
        pass

    # @abstractmethod # MODIFIED: Removed abstractmethod decorator
    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame: # Kept as is for now
        """
        Generates trading signals based on processed data from self.forward().
        Input: Dict of DataFrames as returned by self.forward().
        Output: A single Pandas DataFrame with at least a 'signal' column and a DatetimeIndex.
                The 'signal' column should contain numerical values (e.g., 1 for buy, -1 for sell, 0 for hold).
                The DataFrame should contain signals for a primary asset or an aggregated signal.
                The last row of this DataFrame is typically used by the superposition layer.
                Ensure the index aligns with the input data's timestamps.
        """
        # MODIFIED: Added default implementation
        self.logger.warning(
            f"Strategy {self.config.name} ({self.get_class_name()}) called generate_signals. "
            f"This method is intended for older strategy patterns or specific use cases. "
            f"Ensure this is the intended behavior for strategies refactored to use torch.nn.Module and forward()."
        )
        # Return an empty DataFrame with expected columns if this method is called unexpectedly.
        # Adjust columns as necessary based on typical downstream expectations.
        return pd.DataFrame(columns=['signal'])

    @property
    def effective_params(self) -> Dict[str, Any]:
        """Alias for params to maintain backward compatibility."""
        return self.params

    @classmethod
    def get_strategy_name(cls) -> str:
        # This method is intended to return a general name for the strategy class,
        # not tied to a specific instance's config.
        # It's often the class name itself or a predefined static name.
        # If a dynamic name based on a default config is needed, that logic would be here.
        # For now, let's return the class name as a sensible default for a class method.
        return cls.__name__

    def get_params(self) -> Dict[str, Any]:
        return self.params

    def update_params(self, new_params: Dict[str, Any]):
        # This will overwrite existing keys and add new ones.
        self.params.update(new_params)
        # print(f"Parameters for {self.get_strategy_name()} updated to: {self.params}")

    @classmethod
    def get_class_name(cls) -> str:
        """Returns the class name, useful for mapping in generators/superposition."""
        return cls.__name__

    @classmethod
    def get_param_definitions(cls) -> Dict[str, Dict[str, Any]]:
        return cls.param_definitions

    @classmethod
    def get_parameter_space(cls) -> Dict[str, Any]:
        """
        Returns the parameter space definition for the strategy.
        This should be a dictionary where keys are parameter names and values are
        descriptions or specifications (e.g., ranges, types, choices).
        This is used by optimization and dynamic generation processes.
        """
        # Corrected: Access param_definitions as a class attribute
        definitions = cls.param_definitions 
        
        # Ensure the definitions are suitable for direct use or further processing
        # For example, if using a library like Optuna, this might return
        # a dictionary of parameter types or distributions.
        # For now, we assume it's a dictionary ready for use.
        if not isinstance(definitions, dict):
            logger.error(f"param_definitions for {cls.__name__} is not a dictionary. Found: {type(definitions)}")
            return {}
        return definitions

    def _get_primary_symbol(self, data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Helper to determine the primary symbol to operate on.
        Strategies can override this if they have specific logic.
        """
        if self.config.applicable_assets:
            for asset in self.config.applicable_assets:
                if asset in data_dict:
                    return asset
            # If none of the applicable_assets are in market_data_dict, maybe return None or log warning
            # print(f"Warning: None of the applicable_assets for {self.get_strategy_name()} found in market data.")
            return None
