# src/agent/strategies/base_strategy.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
import logging # Import logging

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


class BaseStrategy(ABC):
    param_definitions: Dict[str, Dict[str, Any]] = {}

    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        self.config = config
        
        # Initialize logger first so it can be used during parameter processing
        if not hasattr(self, 'logger') or self.logger is None:
            self.logger = logger if logger else logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        final_params: Dict[str, Any] = {}
        default_params_from_config = self.config.default_params

        # 1. Start with all defaults from config
        for key, default_value in default_params_from_config.items():
            final_params[key] = default_value

        # 2. Override with instance params (params passed during instantiation), performing type coercion
        if params:
            for key, value in params.items():
                if key in default_params_from_config:
                    default_value = default_params_from_config[key]
                    
                    if default_value is None: # If default is None, accept any type for override
                         final_params[key] = value
                         self.logger.debug(f"Strategy {self.config.name}: Parameter '{key}' uses provided value '{value}' (type {type(value).__name__}) as default was None.")
                         continue 

                    default_type = type(default_value)
                    if not isinstance(value, default_type):
                        try:
                            converted_value = None
                            if default_type == bool and isinstance(value, str):
                                if value.lower() in ['true', '1', 'yes']:
                                    converted_value = True
                                elif value.lower() in ['false', '0', 'no']:
                                    converted_value = False
                                else:
                                    # If string is not a recognized boolean, fall back to standard bool conversion or raise error
                                    # For now, let's try standard bool conversion which might be True for non-empty unrecognized strings
                                    # Or, better, raise a warning and use default or keep original if that's desired.
                                    # Let's log a warning and use the original value if it's an unrecognized string for bool.
                                    self.logger.warning(f"Strategy {self.config.name}: Parameter '{key}' value '{value}' (string) is not a recognized boolean string. Using original value.")
                                    converted_value = value # Or perhaps default_value, or raise error
                            elif default_type == bool and isinstance(value, (int, float)):
                                converted_value = bool(value) # Standard conversion for numbers to bool (0 is False, others True)
                            else:
                                converted_value = default_type(value) # Standard conversion for other types
                            
                            final_params[key] = converted_value
                            self.logger.debug(f"Strategy {self.config.name}: Parameter '{key}' ('{value}') converted from {type(value).__name__} to {default_type.__name__} ('{converted_value}').")
                        except (ValueError, TypeError) as e:
                            self.logger.warning(f"Strategy {self.config.name}: Could not convert parameter '{key}' value '{value}' (type {type(value).__name__}) to expected type {default_type.__name__}. Using original value. Error: {e}")
                            final_params[key] = value # Keep original value if conversion fails
                    else:
                        final_params[key] = value # Types match, use provided value
                else:
                    # Parameter not in defaults, it's an extra parameter
                    self.logger.warning(f"Strategy {self.config.name}: Received unexpected parameter '{key}' with value '{value}'. It will be included.")
                    final_params[key] = value
        
        self.params = final_params
        # self.strategy_id = config.name # strategy_id is now self.config.name

    @abstractmethod
    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
        """
        Processes raw market data to generate features or processed data.
        Input: Dict of DataFrames, one per asset. Each DF should have 'open', 'high', 'low', 'close', 'volume'.
               Index should be DatetimeIndex.
        Output: Dict of DataFrames (can be a subset of input keys if strategy is asset-specific),
                with added columns for indicators/features. Index should be preserved.
        """
        pass

    @abstractmethod
    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Generates trading signals based on processed data from self.forward().
        Input: Dict of DataFrames as returned by self.forward().
        Output: A single Pandas DataFrame with at least a 'signal' column and a DatetimeIndex.
                The 'signal' column should contain numerical values (e.g., 1 for buy, -1 for sell, 0 for hold).
                The DataFrame should contain signals for a primary asset or an aggregated signal.
                The last row of this DataFrame is typically used by the superposition layer.
                Ensure the index aligns with the input data's timestamps.
        """
        pass

    @property
    def effective_params(self) -> Dict[str, Any]:
        """Alias for params to maintain backward compatibility."""
        return self.params

    def get_strategy_name(self) -> str:
        return self.config.name

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
    def get_parameter_space(cls, optimizer_type: str) -> Optional[Dict[str, Any]]:
        """
        Returns the parameter space for the given optimizer type.
        For "genetic", it expects a dictionary where keys are parameter names and
        values are tuples (min, max) for continuous/integer params, or lists for categorical params.
        """
        if optimizer_type == "genetic":
            space = {}
            definitions = cls.param_definitions() # Changed from cls.get_param_definitions()
            if not definitions:
                # self.logger.debug(f"No param_definitions found for strategy {cls.__name__} to build genetic parameter space.") # Cannot use self.logger in classmethod directly without instance
                # Consider logging this at the point of call if needed, or use a class-level logger if available.
                pass # No definitions, so no space

            for name, definition in definitions.items():
                param_type = definition.get('type')
                # For numerical types with min and max
                if 'min' in definition and 'max' in definition and \
                   (param_type is None or param_type in (int, float)): # Check type if available or assume numeric if min/max present
                    # Ensure min is less than max to avoid issues with optimizer
                    if definition['min'] < definition['max']:
                        space[name] = (definition['min'], definition['max'])
                    # else: log warning or skip if min >= max? For now, skip.
                # For categorical types with choices
                elif 'choices' in definition and isinstance(definition['choices'], list) and definition['choices']:
                    space[name] = definition['choices']
                elif param_type == bool: # ADDED: Handle boolean type explicitly
                    space[name] = [True, False] # Define search space as actual boolean values
            
            # Return the constructed space. If empty, it means no optimizable parameters were found or defined correctly.
            return space if space else None # Let's make it return None if empty, as per original thought.
        return None

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
