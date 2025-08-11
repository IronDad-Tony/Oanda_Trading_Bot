# src/agent/strategies/__init__.py
import os
import importlib
import logging
from typing import Dict, Type, Any

# Explicitly import BaseStrategy and StrategyConfig to make them available for package-level imports
from .base_strategy import BaseStrategy, StrategyConfig

# Configure logger for this module
logger = logging.getLogger(__name__)

# Global registry for all discovered strategy classes
STRATEGY_REGISTRY: Dict[str, Type[Any]] = {}

def register_strategy(name: str):
    """
    A decorator to register a strategy class in the STRATEGY_REGISTRY.
    
    Args:
        name (str): The name to register the strategy under.
    """
    def decorator(cls):
        if name in STRATEGY_REGISTRY:
            logger.warning(f"Strategy '{name}' is already registered. Overwriting.")
        STRATEGY_REGISTRY[name] = cls
        logger.info(f"Successfully registered strategy: '{name}' -> {cls.__name__}")
        return cls
    return decorator

def get_strategy(name: str) -> Type[Any]:
    """
    Retrieves a strategy class from the registry.

    Args:
        name (str): The name of the strategy to retrieve.

    Returns:
        The strategy class if found.

    Raises:
        ValueError: If the strategy is not found in the registry.
    """
    strategy = STRATEGY_REGISTRY.get(name)
    if not strategy:
        logger.error(f"Strategy '{name}' not found in registry. Available strategies: {list(STRATEGY_REGISTRY.keys())}")
        raise ValueError(f"Strategy '{name}' not found in registry.")
    return strategy

def _discover_strategies():
    """
    Dynamically discovers and imports all strategies from the current directory.
    This function assumes that strategy classes are decorated with @register_strategy.
    """
    current_dir = os.path.dirname(__file__)
    # Correctly determine the base package path for imports
    # Assumes the file is at src/agent/strategies/__init__.py
    # and we want to import src.agent.strategies.*
    
    # To make this more robust, we could try to find the 'src' directory 
    # and build the path from there, but for now, this is a common structure.
    module_name_prefix = 'oanda_trading_bot.training_system.agent.strategies.'

    logger.info(f"Starting strategy discovery in: {current_dir}")
    
    for filename in os.listdir(current_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            module_name = filename[:-3]
            full_module_path = module_name_prefix + module_name
            try:
                # Import the module to trigger registration decorators
                importlib.import_module(full_module_path)
            except ImportError as e:
                logger.error(f"Failed to import strategy module '{full_module_path}': {e}", exc_info=True)
            except Exception as e:
                logger.error(f"An unexpected error occurred while importing '{full_module_path}': {e}", exc_info=True)

# --- Run Discovery on Initial Import ---
# When this __init__.py is first imported, automatically discover all strategies.
logger.info("Initializing strategy registry and discovering strategies...")
_discover_strategies()
logger.info(f"Strategy discovery complete. Registered strategies: {list(STRATEGY_REGISTRY.keys())}")

# Optionally, define __all__ to control `from oanda_trading_bot.training_system.agent.strategies import *`
__all__ = ['STRATEGY_REGISTRY', 'register_strategy', 'get_strategy', 'BaseStrategy', 'StrategyConfig']
