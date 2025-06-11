# src/agent/enhanced_quantum_strategy_layer.py
"""
增強版量子策略層實現
擴展原有3種策略至15+種策略，實現階段一核心架構增強

主要增強：
1. 15種專業交易策略實現
2. 動態策略生成器
3. 量子策略組合優化
4. 自適應權重極習機制
5. 策略創新引擎（基礎版）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Type, Callable # Added Type, Callable, Optional, Dict, Any
import logging
from abc import ABC, abstractmethod
import math
import random # Ensure random is imported at the top level

try:
    from src.common.logger_setup import logger
    from src.common.config import DEVICE, MAX_SYMBOLS_ALLOWED
    from .strategies import BaseStrategy, StrategyConfig
    from .optimizers.genetic_optimizer import GeneticOptimizer # NEW IMPORT
    from .optimizers.neural_architecture_search import NeuralArchitectureSearch # ADDED IMPORT
    from .strategies import (
        MomentumStrategy,
        BreakoutStrategy,
        TrendFollowingStrategy,
        ReversalStrategy,
        StatisticalArbitrageStrategy,
        VolatilityBreakoutStrategy, # NEW - Correct import from statistical_arbitrage_strategies
        PairsTradeStrategy,
        MeanReversionStrategy,
        CointegrationStrategy,
        VolatilityArbitrageStrategy,
        ReinforcementLearningStrategy,
        DeepLearningPredictionStrategy,
        EnsembleLearningStrategy,
        TransferLearningStrategy,
        DynamicHedgingStrategy,
        RiskParityStrategy,
        VaRControlStrategy,
        MaxDrawdownControlStrategy,
        OptionFlowStrategy,
        MicrostructureStrategy,
        CarryTradeStrategy,
        MacroEconomicStrategy,
        EventDrivenStrategy,
        SentimentStrategy,
        QuantitativeStrategy,
        MarketMakingStrategy,
        HighFrequencyStrategy,
        AlgorithmicStrategy
    )

except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Import error in enhanced_quantum_strategy_layer: {e}")
    DEVICE = "cpu"
    MAX_SYMBOLS_ALLOWED = 5
    class BaseStrategy(nn.Module, ABC):
        def __init__(self, params: dict = None, config: Optional[Any] = None):
            super().__init__()
            self.params = params if params is not None else {}
            self.config = config
            self.strategy_id = self.__class__.__name__ if config is None else config.name

        @abstractmethod
        def forward(self, market_data: pd.DataFrame, portfolio_context: dict = None) -> pd.DataFrame:
            pass

        @abstractmethod
        def generate_signals(self, processed_data: pd.DataFrame, portfolio_context: dict = None) -> pd.DataFrame:
            pass

        def get_strategy_name(self) -> str:
            return self.strategy_id
    
    from dataclasses import dataclass
    @dataclass
    class StrategyConfig:
        name: str
        description: str
        risk_level: float
        market_regime: str
        complexity: int
        base_performance: float = 0.5

# 自定義激活函數實現
class Swish(nn.Module):
    """Swish激活函數實現"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    """Mish激活函數實現"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# ===============================
# 15種專業交易策略實現
# ===============================

# All original strategy class definitions (MomentumStrategy, BreakoutStrategy, ..., AlgorithmicStrategy)
# have been moved to their respective files under src/agent/strategies/
# and are now imported above.

# Placeholder classes like CointegrationStrategy, VolatilityArbitrageStrategy, 
# ReinforcementLearningStrategy, etc., are also imported from their new locations.
# Ensure these placeholder classes are properly defined in their respective files 
# (e.g., statistical_arbitrage_strategies.py, ml_strategies.py)
# and inherit from the new BaseStrategy.

# ===============================
# 動態策略生成器
# ===============================

class DynamicStrategyGenerator:
    """
    Generates and manages trading strategies dynamically based on market conditions
    and other factors.
    """
    def __init__(self, logger: logging.Logger, optimizer_config: Optional[Dict] = None):
        self.logger = logger
        self.optimizer_config = optimizer_config
        self.genetic_optimizer: Optional[GeneticOptimizer] = None
        self.nas_optimizer: Optional[NeuralArchitectureSearch] = None # ADDED
        
        if self.optimizer_config:
            optimizer_name = self.optimizer_config.get("name")
            if optimizer_name == "GeneticOptimizer":
                self.logger.info(f"GeneticOptimizer configuration stored. It will be initialized/used in generate_new_strategy.")
                # self.integrate_genetic_optimizer(self.optimizer_config) # Initialization moved to generate_new_strategy
            elif optimizer_name == "NeuralArchitectureSearch":
                self.logger.info(f"NeuralArchitectureSearch configuration stored. It will be initialized/used in generate_new_strategy.")
                # self.integrate_nas(self.optimizer_config) # Initialization moved to generate_new_strategy
            elif optimizer_name:
                self.logger.warning(f"Unknown optimizer specified in config: {optimizer_name}")
        else:
            self.logger.info("No optimizer_config provided to DynamicStrategyGenerator.")

    def generate_new_strategy(
        self,
        strategy_class: Type[BaseStrategy],
        fitness_function: Optional[Callable[[BaseStrategy, Optional[Dict], Optional[pd.DataFrame], Optional[Dict]], float]] = None, # Adjusted signature for ga_fitness_wrapper
        market_data_for_ga: Optional[Any] = None, 
        current_context: Optional[Dict] = None,
        strategy_config_override: Optional[StrategyConfig] = None, 
        initial_parameters: Optional[Dict[str, Any]] = None,    
    ) -> Optional[BaseStrategy]:
        if not strategy_class:
            self.logger.error("Invalid strategy_class provided: None")
            return None
        if not isinstance(strategy_class, type) or not issubclass(strategy_class, BaseStrategy): 
             self.logger.error(f"Invalid strategy_class provided: {strategy_class}. It must be a subclass of BaseStrategy.")
             return None
        
        # ADDED: Validate current_context type
        if current_context is not None and not isinstance(current_context, Dict):
            self.logger.error(f"Invalid current_context type: {type(current_context)}. Expected Dict or None.")
            return None
            
        self.logger.info(f"Attempting to generate new strategy of type: {strategy_class.__name__}")

        config_for_instance: StrategyConfig
        if strategy_config_override:
            config_for_instance = strategy_config_override
            self.logger.info(f"Using provided strategy_config_override for {strategy_class.__name__} (Name: {config_for_instance.name}).")
        else:
            config_for_instance = strategy_class.default_config()
            self.logger.info(f"Using default_config() from {strategy_class.__name__} (Name: {config_for_instance.name}).")

        if not config_for_instance: 
            self.logger.error(f"Could not determine a valid configuration for {strategy_class.__name__}. default_config() might have returned None.")
            return None

        params_for_instance: Dict[str, Any] = {}
        if config_for_instance.default_params: 
            params_for_instance.update(config_for_instance.default_params)
        
        if initial_parameters: 
            self.logger.info(f"Applying initial_parameters: {initial_parameters}")
            params_for_instance.update(initial_parameters)
        
        param_space_for_ga = strategy_class.get_parameter_space(optimizer_type="genetic")
        
        self.logger.debug(f"DSG Internal State: strategy_class={strategy_class.__name__}")
        self.logger.debug(f"DSG Internal State: self.optimizer_config={self.optimizer_config}")
        self.logger.debug(f"DSG Internal State: param_space_for_ga={param_space_for_ga}")
        self.logger.debug(f"DSG Internal State: fitness_function provided? {'Yes' if fitness_function else 'No'}")

        use_genetic_optimizer = (
            self.optimizer_config is not None and
            self.optimizer_config.get("name") == "GeneticOptimizer" and
            param_space_for_ga and 
            fitness_function is not None
        )
        self.logger.debug(f"DSG Internal State: Decision to use Genetic Optimizer: {use_genetic_optimizer}")

        # Placeholder for NAS specific parameter space and decision logic
        param_space_for_nas = getattr(strategy_class, 'get_parameter_space_for_nas', lambda: None)() # Example
        use_nas_optimizer = (
            self.optimizer_config is not None and
            self.optimizer_config.get("name") == "NeuralArchitectureSearch" and
            param_space_for_nas and # Strategy must define a NAS search space
            fitness_function is not None and # NAS also needs a fitness function
            issubclass(strategy_class, nn.Module) # NAS typically applies to nn.Module based strategies
        )
        self.logger.debug(f"DSG Internal State: Decision to use NAS Optimizer: {use_nas_optimizer}")


        if use_genetic_optimizer:
            self.logger.info(f"Optimizing parameters for {strategy_class.__name__} using GeneticOptimizer. Param space: {param_space_for_ga}")

            def ga_fitness_wrapper(strategy_instance_for_ga: BaseStrategy, 
                                   market_data_for_fitness_eval: Optional[pd.DataFrame], 
                                   portfolio_context_for_fitness_eval: Optional[Dict],   
                                   raw_params_from_ga: Dict[str, Any]):
                self.logger.debug(f"ga_fitness_wrapper: Evaluating {strategy_class.__name__} (instance: {type(strategy_instance_for_ga)}) with GA params: {raw_params_from_ga}")
                if not fitness_function: 
                    self.logger.error("ga_fitness_wrapper: fitness_function is None, this should not happen here.")
                    return -float('inf') 
                return fitness_function(strategy_instance_for_ga, portfolio_context_for_fitness_eval, market_data_for_fitness_eval, raw_params_from_ga)

            try:
                ga_settings = self.optimizer_config.get('settings', {}) if self.optimizer_config else {}
                # Ensure param_space_for_ga is not None before passing to GeneticOptimizer
                if not param_space_for_ga: # Should be caught by use_optimizer, but defensive check
                    self.logger.error(f"param_space_for_ga is None or empty for {strategy_class.__name__} even when optimizer was to be used. This is unexpected.")
                    raise ValueError("param_space_for_ga cannot be None or empty for GeneticOptimizer")

                self.genetic_optimizer = GeneticOptimizer(
                    strategy_class=strategy_class,
                    base_config=config_for_instance, 
                    param_space=param_space_for_ga,
                    fitness_function_callback=ga_fitness_wrapper,
                    logger=self.logger, # Pass DSG's logger to GeneticOptimizer
                    ga_settings=ga_settings
                )
                self.logger.debug(f"DSG Internal State: GeneticOptimizer instance created: {type(self.genetic_optimizer)}")
                
                optimized_param_values_tuple, best_fitness = self.genetic_optimizer.run_optimizer(
                    market_data_for_fitness=market_data_for_ga, 
                    portfolio_context_for_fitness=current_context 
                )
                
                if optimized_param_values_tuple and param_space_for_ga: # param_space_for_ga check is redundant if it was required for GA init
                    optimized_params_dict = dict(zip(param_space_for_ga.keys(), optimized_param_values_tuple))
                    self.logger.info(f"GeneticOptimizer found optimized_params: {optimized_params_dict} with fitness: {best_fitness}")
                    params_for_instance.update(optimized_params_dict) 
                elif not optimized_param_values_tuple and best_fitness == -float('inf'):
                     self.logger.warning(f"GeneticOptimizer did not find a better set of parameters for {strategy_class.__name__} (returned empty/None tuple and -inf fitness). Using non-optimized params: {params_for_instance}")
                else: 
                    self.logger.warning(f"GeneticOptimizer did not return a valid parameter set for {strategy_class.__name__} (Result: {optimized_param_values_tuple}, Fitness: {best_fitness}). Using non-optimized params: {params_for_instance}")

            except Exception as e:
                self.logger.error(f"Error during genetic optimization for {strategy_class.__name__}: {e}", exc_info=True)
                self.logger.warning(f"Falling back to non-optimized params for {strategy_class.__name__} due to optimizer error. Params: {params_for_instance}")
            self.genetic_optimizer = None # Clear after use
        
        elif use_nas_optimizer:
            self.logger.info(f"Optimizing architecture/parameters for {strategy_class.__name__} using NeuralArchitectureSearch. Search space: {param_space_for_nas}")
            
            # Wrapper for NAS fitness function (similar to GA)
            def nas_fitness_wrapper(strategy_instance_for_nas: BaseStrategy, # Or potentially a model definition
                                   market_data_for_fitness_eval: Optional[pd.DataFrame], 
                                   portfolio_context_for_fitness_eval: Optional[Dict],   
                                   # NAS might pass architecture definition or specific params
                                   nas_trial_info: Dict[str, Any]): 
                self.logger.debug(f"nas_fitness_wrapper: Evaluating {strategy_class.__name__} (instance: {type(strategy_instance_for_nas)}) with NAS trial: {nas_trial_info}")
                if not fitness_function: 
                    self.logger.error("nas_fitness_wrapper: fitness_function is None, this should not happen here.")
                    return -float('inf') 
                # The fitness_function needs to be compatible with what NAS provides.
                # This might involve instantiating a strategy with the NAS-proposed architecture/params.
                return fitness_function(strategy_instance_for_nas, portfolio_context_for_fitness_eval, market_data_for_fitness_eval, nas_trial_info)

            try:
                nas_settings = self.optimizer_config.get('settings', {}) if self.optimizer_config else {}
                if not param_space_for_nas:
                    self.logger.error(f"param_space_for_nas is None or empty for {strategy_class.__name__} even when NAS optimizer was to be used.")
                    raise ValueError("param_space_for_nas cannot be None or empty for NeuralArchitectureSearch")

                self.nas_optimizer = NeuralArchitectureSearch(
                    strategy_class=strategy_class, # The base class to search architectures for/within
                    search_space=param_space_for_nas,
                    fitness_function_callback=nas_fitness_wrapper, # This callback will be called by NAS
                    logger=self.logger,
                    nas_settings=nas_settings # Pass specific NAS settings
                )
                self.logger.debug(f"DSG Internal State: NeuralArchitectureSearch instance created: {type(self.nas_optimizer)}")
                
                # NAS run_optimizer might return an optimized strategy instance directly,
                # or optimized parameters/architecture definition.
                # This is a placeholder and depends on NeuralArchitectureSearch implementation.
                optimization_result = self.nas_optimizer.run_optimizer(
                    market_data_for_fitness=market_data_for_ga, # Assuming same data is used
                    portfolio_context_for_fitness=current_context,
                    # Potentially other args like num_trials, epochs_per_trial etc.
                    # would be part of nas_settings passed to constructor.
                )
                
                # Placeholder: Process NAS results
                # If NAS returns parameters:
                # if isinstance(optimization_result, tuple) and len(optimization_result) == 2:
                #     optimized_nas_params, best_nas_fitness = optimization_result
                #     if optimized_nas_params:
                #         self.logger.info(f"NeuralArchitectureSearch found optimized_params: {optimized_nas_params} with fitness: {best_nas_fitness}")
                #         params_for_instance.update(optimized_nas_params)
                #     else:
                #         self.logger.warning(f"NeuralArchitectureSearch did not return valid parameters. Using non-optimized params.")
                # elif isinstance(optimization_result, BaseStrategy): # If NAS returns a fully formed strategy
                #     self.logger.info(f"NeuralArchitectureSearch returned an optimized strategy instance: {type(optimization_result)}")
                #     # In this case, we might return this instance directly, skipping the final instantiation below.
                #     # For now, let's assume it modifies params_for_instance or we handle it later.
                #     # This part needs careful design based on NAS output.
                #     # For simplicity, let's assume it updates params_for_instance for now.
                #     # This is a BIG assumption.
                #     if hasattr(optimization_result, 'params'):
                #          params_for_instance.update(optimization_result.params)
                #     else:
                #          self.logger.warning("NAS returned a strategy but could not extract params to update current flow.")

                # For now, let's assume run_optimizer returns a dict of parameters to update
                if isinstance(optimization_result, dict):
                    self.logger.info(f"NeuralArchitectureSearch returned optimized parameters: {optimization_result}")
                    params_for_instance.update(optimization_result)
                elif optimization_result is None: # Or some other failure indication
                     self.logger.warning(f"NeuralArchitectureSearch did not find a better architecture/parameters for {strategy_class.__name__}. Using non-optimized params: {params_for_instance}")
                else:
                     self.logger.warning(f"NeuralArchitectureSearch returned an unexpected result type: {type(optimization_result)}. Using non-optimized params: {params_for_instance}")


            except Exception as e:
                self.logger.error(f"Error during Neural Architecture Search for {strategy_class.__name__}: {e}", exc_info=True)
                self.logger.warning(f"Falling back to non-optimized params for {strategy_class.__name__} due to NAS error. Params: {params_for_instance}")
            self.nas_optimizer = None # Clear after use

        else:
            log_msg = f"No optimizer used for {strategy_class.__name__}. Conditions: "
            log_msg += f"GeneticOptimizer eligible: {use_genetic_optimizer}, "
            log_msg += f"NASOptimizer eligible: {use_nas_optimizer}. "
            log_msg += f"Using default/initial params: {params_for_instance}"
            self.logger.info(log_msg)
            self.genetic_optimizer = None
            self.nas_optimizer = None


        try:
            final_strategy_instance = strategy_class(
                config=config_for_instance,
                params=params_for_instance,
                logger=self.logger # Pass DSG's logger to the final strategy instance
            )
            self.logger.info(f"Successfully generated and instantiated strategy: {strategy_class.__name__} with config '{config_for_instance.name}' and final params: {final_strategy_instance.params}")
            return final_strategy_instance
        except Exception as e:
            self.logger.error(f"Failed to instantiate strategy {strategy_class.__name__} with config '{config_for_instance.name}' and params {params_for_instance}: {e}", exc_info=True)
            return None

    def integrate_genetic_optimizer(self, optimizer_config: Dict):
        """
        Integrates and configures the GeneticOptimizer.
        The strategy_class and param_space will be set per optimization task.
        """
        # Store the base config. Actual instantiation might happen in generate_new_strategy
        # or here if a default strategy_class and param_space are always used.
        # For flexibility, we'll allow generate_new_strategy to specify them.
        
        self.genetic_optimizer_default_config = optimizer_config # Store for later use
        
        # Example: If we want to pre-initialize for a default strategy (e.g. Momentum)
        # This is optional and depends on how it's intended to be used.
        # strategy_to_optimize_name = optimizer_config.get("default_strategy_to_optimize", "MomentumStrategy")
        # default_param_space = optimizer_config.get("default_param_space")
        # StrategyClass = self.available_strategies_classes.get(strategy_to_optimize_name)

        # if StrategyClass and default_param_space:
        #     try:
        #         self.genetic_optimizer = GeneticOptimizer(
        #             strategy_class=StrategyClass,
        #             param_space=default_param_space,
        #             fitness_function_callback=self._default_fitness_function,
        #             population_size=optimizer_config.get('population_size', 50),
        #             n_generations=optimizer_config.get('n_generations', 20), # Reduced for quicker tests
        #             crossover_rate=optimizer_config.get('crossover_rate', 0.8),
        #             mutation_rate=optimizer_config.get('mutation_rate', 0.2),
        #             tournament_size=optimizer_config.get('tournament_size', 5),
        #             early_stopping_generations=optimizer_config.get('early_stopping_generations', 5),
        #             random_seed=optimizer_config.get('random_seed', None)
        #         )
        #         logger.info(f"GeneticOptimizer pre-initialized for {strategy_to_optimize_name} with provided config.")
        #     except Exception as e:
        #         logger.error(f"Failed to pre-initialize GeneticOptimizer: {e}")
        # else:
        #     logger.info("GeneticOptimizer config stored. It will be fully initialized on demand in generate_new_strategy.")
        self.logger.info("GeneticOptimizer configuration stored. It will be initialized/used in generate_new_strategy.") # Changed from logger to self.logger

    def integrate_nas(self, nas_config: Dict):
        """
        Integrates and configures Neural Architecture Search.
        The actual NAS optimizer instance is created on demand in generate_new_strategy.
        This method primarily stores the configuration.
        """
        # self.nas_optimizer = NeuralArchitectureSearch(**nas_config) # To be implemented
        # logger.info("NeuralArchitectureSearch integration placeholder.")
        if not isinstance(nas_config, dict):
            self.logger.error("NAS configuration must be a dictionary.")
            return

        # Store the NAS config. Actual instantiation will happen in generate_new_strategy
        # This makes it consistent with how GeneticOptimizer config is handled.
        # self.optimizer_config should already hold this if 'name' was 'NeuralArchitectureSearch'
        # This method might be redundant if __init__ handles storing the config.
        # However, it can be used for explicit re-configuration if needed.
        
        # For now, let's assume __init__ sets self.optimizer_config correctly.
        # This method can be used to validate or pre-process nas_config if necessary.
        
        # Example: Validate required NAS settings
        # required_nas_keys = ['search_space_definition_method', 'max_trials'] # Example keys
        # if not all(key in nas_config.get('settings', {}) for key in required_nas_keys):
        #     self.logger.warning(f"NAS config might be missing some recommended settings: {required_nas_keys}")

        self.logger.info(f"NeuralArchitectureSearch configuration stored/updated: {nas_config.get('settings', {})}. NAS will be initialized on demand in generate_new_strategy.")
        # If we want to ensure this config is specifically for NAS:
        if self.optimizer_config and self.optimizer_config.get("name") == "NeuralArchitectureSearch":
            self.optimizer_config.update(nas_config) # Or merge more carefully
        elif not self.optimizer_config:
             self.optimizer_config = nas_config # If no config was present before
        else: # A different optimizer was configured
            self.logger.warning("integrate_nas called, but a different optimizer is already configured. NAS config will be stored but might not be used unless optimizer_config is updated.")
            # Potentially overwrite or store separately:
            # self.nas_specific_config = nas_config


# ===============================
# 增強版策略疊加系統
# ===============================

class EnhancedStrategySuperposition(nn.Module):
    """
    增強版策略疊加系統
    管理15+種策略的量子疊加，包含動態策略生成
    支持完全動態自適應維度配置
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 enable_dynamic_generation: bool = True,
                 initial_strategy_params: Optional[Dict[str, Dict]] = None,
                 strategy_configs_list: Optional[List[StrategyConfig]] = None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim # Note: action_dim for PyTorch layers, but strategies now use pandas
        self.enable_dynamic_generation = enable_dynamic_generation
        self.initial_strategy_params = initial_strategy_params if initial_strategy_params else {}
        self.strategy_configs_list = strategy_configs_list if strategy_configs_list else []
        self.strategy_config_map = {cfg.name: cfg for cfg in self.strategy_configs_list}

        # Initialize all strategies based on the new structure
        self.base_strategies_instances = nn.ModuleList()
        
        strategy_classes_to_instantiate = [
            MomentumStrategy, BreakoutStrategy, TrendFollowingStrategy, ReversalStrategy,
            StatisticalArbitrageStrategy, VolatilityBreakoutStrategy, PairsTradeStrategy, MeanReversionStrategy,
            CointegrationStrategy, VolatilityArbitrageStrategy, 
            ReinforcementLearningStrategy, DeepLearningPredictionStrategy, EnsembleLearningStrategy, TransferLearningStrategy,
            DynamicHedgingStrategy, RiskParityStrategy, VaRControlStrategy, MaxDrawdownControlStrategy,
            OptionFlowStrategy, MicrostructureStrategy, CarryTradeStrategy, MacroEconomicStrategy,
            EventDrivenStrategy, SentimentStrategy, QuantitativeStrategy, MarketMakingStrategy,
            HighFrequencyStrategy, AlgorithmicStrategy
        ]

        for SClass in strategy_classes_to_instantiate:
            strategy_name = SClass.get_strategy_name(SClass) if hasattr(SClass, 'get_strategy_name') else SClass.__name__
            params = self.initial_strategy_params.get(strategy_name, {})
            config = self.strategy_config_map.get(strategy_name)
            if not config:
                 # Create a default config if not provided - this might need more robust handling
                logger.warning(f"StrategyConfig not found for {strategy_name}, using default placeholder.")
                config = StrategyConfig(name=strategy_name, description="Default", risk_level=0.5, market_regime="all", complexity=3)

            try:
                self.base_strategies_instances.append(SClass(params=params, config=config))
            except Exception as e:
                logger.error(f"Error instantiating {strategy_name} in Superposition: {e}")
        
        self.num_base_strategies = len(self.base_strategies_instances)
        
        # 動態策略生成器
        # TODO: The use of self.config.get below is problematic as self.config is not defined.
        # This should be reviewed. For now, defaulting max_generated_strategies.
        _max_generated_strategies_default = 5
        if enable_dynamic_generation:
            # Pass available strategy classes and their configs to the generator
            generator_config = {
                'strategy_configs_list': self.strategy_configs_list,
                # Potentially other configs for GA/NAS if they are part of the generator directly
            }
            self.dynamic_generator = DynamicStrategyGenerator(
                config=generator_config, 
                strategies=None # Generator will use its available_strategies_classes
            )
            # Max generated strategies needs to be managed by the generator itself or via config
            # self.max_generated_strategies = self.config.get('max_generated_strategies', 5) # Problematic line
            self.max_generated_strategies = self.initial_strategy_params.get('enhanced_superposition_config', {}).get('max_generated_strategies', _max_generated_strategies_default)

            self.total_strategies = self.num_base_strategies + self.max_generated_strategies
        else:
            self.dynamic_generator = None
            self.max_generated_strategies = 0
            self.total_strategies = self.num_base_strategies
        
        # 動態計算權重網絡的隱藏層維度 - 自適應縮放
        # These nn.Linear layers expect PyTorch tensors, but strategies now output pandas DataFrames.
        # The forward pass will need significant changes to reconcile this.
        # For now, keeping the layer definitions, but their inputs will be problematic.
        # This part is highly dependent on the definition of `state_dim`
        self.vol_hidden_dim = max(16, min(64, self.total_strategies if self.total_strategies > 0 else 1)) # Ensure > 0
        self.regime_hidden_dim = max(32, min(128, state_dim // 4 if state_dim > 0 else 32)) # Ensure > 0
        self.corr_hidden_dim = max(32, min(128, (state_dim + 1) // 4 if state_dim > 0 else 32)) # Ensure > 0
        
        # 量子振幅參數（可學習）
        self.quantum_amplitudes = nn.Parameter(
            torch.ones(self.total_strategies if self.total_strategies > 0 else 1) / math.sqrt(self.total_strategies if self.total_strategies > 0 else 1)
        )
        
        # 多層次權重調整網絡 - 動態維度配置
        # Ensure total_strategies is at least 1 for nn.Linear output features
        _out_features_for_nets = max(1, self.total_strategies)
        self.weight_networks = nn.ModuleDict({
            'volatility_net': nn.Sequential(
                nn.Linear(1, self.vol_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.vol_hidden_dim, _out_features_for_nets),
                nn.Sigmoid()
            ),
            'regime_net': nn.Sequential(
                nn.Linear(max(1, state_dim), self.regime_hidden_dim), # Ensure in_features > 0
                nn.GELU(),
                nn.Linear(self.regime_hidden_dim, _out_features_for_nets),
                nn.Softmax(dim=-1)
            ),
            'correlation_net': nn.Sequential(
                nn.Linear(max(1, state_dim + (1 if state_dim > 0 else 0)), self.corr_hidden_dim), # Ensure in_features > 0. If state_dim=0, input is 1 (volatility)
                nn.GELU(),
                nn.Linear(self.corr_hidden_dim, _out_features_for_nets),
                nn.Softmax(dim=-1)
            )
        })
        
        # 策略相互作用矩陣
        self.interaction_matrix = nn.Parameter(
            torch.eye(_out_features_for_nets) + torch.randn(_out_features_for_nets, _out_features_for_nets) * 0.05
        )
        
        # 策略性能追蹤
        self.register_buffer('strategy_performance_history', 
                           torch.zeros(_out_features_for_nets, 100))  # 記錄最近100次性能
        self.register_buffer('performance_index', torch.tensor(0))
        
        # 量子糾纏效應模擬
        self.entanglement_strength = nn.Parameter(torch.tensor(0.1))

        # For final action projection
        self.final_feature_dim = 1 # Assuming combined strategy signal is a single feature
        self.output_projection_layer = nn.Linear(self.final_feature_dim, self.action_dim if self.action_dim > 0 else 1) # Ensure action_dim > 0 for Linear layer

        logger.info(f"🌟 初始化增強版策略疊加系統: {self.total_strategies}種策略")
        logger.info(f"📐 動態維度配置 - State: {state_dim}, Action: {action_dim}")
        logger.info(f"🔧 權重網絡隱藏層 - Vol: {self.vol_hidden_dim}, Regime: {self.regime_hidden_dim}, Corr: {self.corr_hidden_dim}")
    
    def get_dynamic_dimensions(self) -> Dict[str, int]:
        """獲取當前動態維度配置信息"""
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'total_strategies': self.total_strategies,
            'vol_hidden_dim': self.vol_hidden_dim,
            'regime_hidden_dim': self.regime_hidden_dim,
            'corr_hidden_dim': self.corr_hidden_dim,
            'num_base_strategies': self.num_base_strategies,
            'dynamic_generation_enabled': self.enable_dynamic_generation
        }
    
    def _adaptive_dimension_handler(self, tensor: torch.Tensor, 
                                   expected_dim: int, 
                                   operation_name: str = "unknown") -> torch.Tensor:
        """
        動態維度適配處理器
        自動調整輸入張量以匹配期望維度
        
        Args:
            tensor: 輸入張量
            expected_dim: 期望的最後一個維度
            operation_name: 操作名稱，用於日誌
            
        Returns:
            適配後的張量
        """
        current_shape = tensor.shape
        current_last_dim = current_shape[-1]
        
        if current_last_dim == expected_dim:
            return tensor
        
        batch_dims = current_shape[:-1]
        
        if current_last_dim > expected_dim:
            # 維度過大：使用線性投影降維
            if not hasattr(self, f'_adaptive_projector_{operation_name}_{current_last_dim}_{expected_dim}'):
                projector = nn.Linear(current_last_dim, expected_dim).to(tensor.dev极)
                setattr(self, f'_adaptive_projector_{operation_name}_{current_last_dim}_{expected_dim}', projector)
                logger.info(f"🔧 創建動態投影器: {operation_name} {current_last_dim}→{expected_dim}")
            
            projector = getattr(self, f'_adaptive_projector_{operation_name}_{current_last_dim}_{expected_dim}')
            adapted_tensor = projector(tensor)
            
        elif current_last_dim < expected_dim:
            # 維度過小：使用零填充擴展
            pad_size = expected_dim - current_last_dim
            padding = torch.zeros(*batch_dims, pad_size, device=tensor.device, dtype=tensor.dtype)
            adapted_tensor = torch.cat([tensor, padding], dim=-1)
            logger.info(f"🔧 動態零填充: {operation_name} {current_last_dim}→{expected_dim}")
        
        return adapted_tensor
    
    def _validate_and_adapt_inputs(self, state: torch.Tensor, 
                                  volatility: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        驗證並適配輸入維度
        
        Args:
            state: 市場狀態張量
            volatility: 波動率張量
            
        Returns:
            適配後的狀態和波動率張量
        """
        # 適配狀態張量
        adapted_state = self._adaptive_dimension_handler(
            state, self.state_dim, "state_input"
        )
        
        # 確保波動率是一維的
        if volatility.dim() > 1 and volatility.shape[-1] != 1:
            volatility = volatility.mean(dim=-1, keepdim=True)
        
        if volatility.dim() == 1:
            volatility = volatility.unsqueeze(-1)
        
        return adapted_state, volatility.squeeze(-1)
    
    def forward(self, market_data_dict: Dict[str, pd.DataFrame], 
                portfolio_context: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        核心前向傳播邏輯，整合所有策略並計算最終行動建議
        
        Args:
            market_data_dict: 包含各個交易對市場數據的字典，鍵為交易對名稱，值為Pandas DataFrame。
                              DataFrame應包含 'close', 'volume', 'high', 'low' 等列。
            portfolio_context: 包含投資組合當前狀態和上下文信息的字典。
                               例如: {'cash': 10000, 'positions': {'EUR_USD': 100}, 
                                     'market_regime': 'trending', 'overall_volatility': 0.02,
                                     'risk_appetite': 'high'}

        Returns:
            Tuple containing:
            - final_action_distribution (torch.Tensor): 最終的行動分佈 (e.g., [batch_size, num_assets, num_actions])
                                                        or a more abstract representation.
            - strategy_weights (torch.Tensor): 各策略的計算權重 (e.g., [batch_size, total_strategies])
            - diagnostic_info (Dict[str, Any]): 包含診斷信息的字典
        """
        batch_size = 1 # Assuming batch_size of 1 for now, as market_data_dict is not batched.
                       # This needs to be generalized if batch processing is intended.

        # 0. Prepare data for strategies
        # Assuming all strategies operate on a common primary market data (e.g., a specific symbol or aggregated data)
        # This is a simplification. In reality, each strategy might need different parts of market_data_dict
        # or specific pre-processing.
        # For now, let's assume strategies can handle the raw dict or we pick one primary symbol.
        # Let's assume 'EUR_USD' is a primary symbol if available, otherwise the first one.
        primary_symbol = next(iter(market_data_dict)) if market_data_dict else None
        if not primary_symbol:
            logger.error("Market data dictionary is empty. Cannot proceed.")
            # Return dummy tensors of appropriate (though possibly incorrect) shape
            # This part needs careful consideration for robust error handling
            _action_dim_for_dummy = self.action_dim if self.action_dim > 0 else 1
            _total_strategies_for_dummy = self.total_strategies if self.total_strategies > 0 else 1
            dummy_action = torch.zeros(batch_size, _action_dim_for_dummy, device=DEVICE) 
            dummy_weights = torch.zeros(batch_size, _total_strategies_for_dummy, device=DEVICE)
            return dummy_action, dummy_weights, {}

        primary_market_data_df = market_data_dict[primary_symbol]

        # 1. 執行所有基礎策略和動態生成策略
        strategy_outputs_dfs: List[pd.DataFrame] = []
        active_strategies_for_forward = list(self.base_strategies_instances)

        # Handle dynamically generated strategies if enabled
        if self.enable_dynamic_generation and self.dynamic_generator:
            # This part is conceptual. Dynamic generation might happen less frequently
            # or be triggered by specific conditions, not necessarily every forward pass.
            # For now, let's assume we might have some pre-generated dynamic strategies.
            # Or, if we want to generate one on the fly (less likely for every forward pass):
            # new_dynamic_strategy = self.dynamic_generator.generate_new_strategy(primary_market_data_df, portfolio_context)
            # if new_dynamic_strategy:
            #     active_strategies_for_forward.append(new_dynamic_strategy)
            # For simplicity, let's assume dynamic_generator.active_strategies_instances holds them
            active_strategies_for_forward.extend(self.dynamic_generator.active_strategies_instances)
        
        current_num_strategies = len(active_strategies_for_forward)
        # Ensure total_strategies is at least 1 for dummy outputs if no active strategies
        _total_strategies_for_dummy_active = self.total_strategies if self.total_strategies > 0 else 1
        _action_dim_for_dummy_active = self.action_dim if self.action_dim > 0 else 1

        if current_num_strategies == 0 and self.total_strategies == 0: # Only error if total_strategies is also 0
            logger.error("No active strategies available and total_strategies is 0. Cannot proceed.")
            dummy_action = torch.zeros(batch_size, _action_dim_for_dummy_active, device=DEVICE)
            dummy_weights = torch.zeros(batch_size, _total_strategies_for_dummy_active, device=DEVICE) 
            return dummy_action, dummy_weights, {}

        for i, strategy_instance in enumerate(active_strategies_for_forward):
            try:
                # Each strategy's forward method processes market_data and returns processed_data (pd.DataFrame)
                # Then generate_signals uses this processed_data to output signals (pd.DataFrame)
                # The signals DataFrame should ideally have a consistent format, e.g., a 'signal' column.
                # For multi-asset strategies, it might have signals per asset.
                
                # Pass the full market_data_dict and portfolio_context to each strategy
                processed_data = strategy_instance.forward(market_data_dict, portfolio_context)
                signals_df = strategy_instance.generate_signals(processed_data, portfolio_context)
                
                # Ensure signals_df is not empty and has expected structure
                if signals_df is None or signals_df.empty:
                    logger.warning(f"Strategy {strategy_instance.get_strategy_name()} produced empty signals. Using zeros.")
                    # Assuming signals are for the primary_symbol and a single 'signal' column
                    # This needs to be robust: define expected output structure for strategies
                    num_timesteps = len(primary_market_data_df) if primary_market_data_df is not None else 1 
                    index_for_df = primary_market_data_df.index if primary_market_data_df is not None and not primary_market_data_df.empty else pd.RangeIndex(num_timesteps)
                    signals_df = pd.DataFrame({'signal': np.zeros(num_timesteps)}, index=index_for_df)

                strategy_outputs_dfs.append(signals_df)

            except Exception as e:
                logger.error(f"Error executing strategy {strategy_instance.get_strategy_name()}: {e}", exc_info=True)
                # Append a DataFrame of zeros or handle error appropriately
                num_timesteps = len(primary_market_data_df) if primary_market_data_df is not None else 1
                index_for_df = primary_market_data_df.index if primary_market_data_df is not None and not primary_market_data_df.empty else pd.RangeIndex(num_timesteps)
                error_signals_df = pd.DataFrame({'signal': np.zeros(num_timesteps)}, index=index_for_df)
                strategy_outputs_dfs.append(error_signals_df)
        
        # Pad with zero signals if current_num_strategies < self.total_strategies (due to dynamic part)
        # Ensure self.total_strategies is at least current_num_strategies
        effective_total_strategies = max(current_num_strategies, self.total_strategies if self.total_strategies > 0 else current_num_strategies)
        if effective_total_strategies == 0 and current_num_strategies == 0 : # If all are zero, make it 1 to avoid division by zero later
             effective_total_strategies = 1

        num_padding_strategies = effective_total_strategies - current_num_strategies
            
        for _ in range(num_padding_strategies):
            num_timesteps = len(primary_market_data_df) if primary_market_data_df is not None else 1
            index_for_df = primary_market_data_df.index if primary_market_data_df is not None and not primary_market_data_df.empty else pd.RangeIndex(num_timesteps)
            padding_signals_df = pd.DataFrame({'signal': np.zeros(num_timesteps)}, index=index_for_df)
            strategy_outputs_dfs.append(padding_signals_df)

        # 2. Convert strategy outputs (List[pd.DataFrame]) to a PyTorch tensor `strategy_tensor`
        # Assumption: Each DataFrame in strategy_outputs_dfs contains a 'signal' column 
        # for the primary_symbol, representing the strategy's output (e.g., -1, 0, 1).
        # We'll take the latest signal from each strategy.
        # Shape: [batch_size, total_strategies, num_features_per_strategy]
        # For now, num_features_per_strategy = 1 (the signal itself)
        
        latest_signals = []
        for df in strategy_outputs_dfs:
            if not df.empty and 'signal' in df.columns:
                latest_signals.append(df['signal'].iloc[-1] if not df['signal'].empty else 0.0)
            else:
                logger.warning(f"Signal DataFrame is empty or missing 'signal' column. Appending 0.0.")
                latest_signals.append(0.0)
        
        # Ensure latest_signals has effective_total_strategies elements
        if len(latest_signals) != effective_total_strategies:
            logger.error(f"Mismatch in number of signals ({len(latest_signals)}) and effective_total_strategies ({effective_total_strategies}). This indicates an issue in padding or strategy execution.")
            # Fallback: pad or truncate latest_signals to match effective_total_strategies
            if len(latest_signals) < effective_total_strategies:
                latest_signals.extend([0.0] * (effective_total_strategies - len(latest_signals)))
            else:
                latest_signals = latest_signals[:effective_total_strategies]
        
        if not latest_signals: # Ensure latest_signals is not empty before converting to tensor
            latest_signals = [0.0] * effective_total_strategies # Should be at least 1 element

        strategy_tensor = torch.tensor(latest_signals, dtype=torch.float32, device=DEVICE).unsqueeze(0) # [1, effective_total_strategies]
        strategy_tensor = strategy_tensor.unsqueeze(-1) # [1, effective_total_strategies, 1] (feature_dim=1)

        # 3. Derive `state_features_for_weighting` from market_data_dict and portfolio_context
        # Shape: [batch_size, self.state_dim]
        # This is a placeholder. Actual feature engineering will be more complex.
        # Example: use 'close' price of primary_symbol, and some portfolio context.
        if self.state_dim > 0:
            lookback_for_state = self.state_dim 
            if primary_market_data_df is not None and 'close' in primary_market_data_df.columns and not primary_market_data_df.empty:
                close_prices = primary_market_data_df['close'].values
                if len(close_prices) >= lookback_for_state:
                    state_features_raw = close_prices[-lookback_for_state:]
                else: 
                    state_features_raw = np.pad(close_prices, (lookback_for_state - len(close_prices), 0), 'edge')
            else: 
                state_features_raw = np.zeros(lookback_for_state)

            state_features_for_weighting = torch.tensor(state_features_raw, dtype=torch.float32, device=DEVICE).unsqueeze(0) # [1, state_dim]
            
            if state_features_for_weighting.shape[1] != self.state_dim:
                logger.warning(f"Constructed state_features_for_weighting shape {state_features_for_weighting.shape} does not match self.state_dim {self.state_dim}. Adjusting/Padding.")
                if state_features_for_weighting.shape[1] < self.state_dim:
                    padding = torch.zeros(batch_size, self.state_dim - state_features_for_weighting.shape[1], device=DEVICE)
                    state_features_for_weighting = torch.cat((state_features_for_weighting, padding), dim=1)
                else:
                    state_features_for_weighting = state_features_for_weighting[:, :self.state_dim]
        else: 
             state_features_for_weighting = torch.empty(batch_size, 0, device=DEVICE)


        # 4. Derive `volatility_for_weighting` (e.g., from portfolio_context or market_data)
        # Shape: [batch_size, 1]
        if 'overall_volatility' in portfolio_context:
            volatility_value = float(portfolio_context['overall_volatility'])
        elif primary_market_data_df is not None and 'close' in primary_market_data_df.columns and len(primary_market_data_df['close']) > 1:
            log_returns = np.log(primary_market_data_df['close'] / primary_market_data_df['close'].shift(1)).dropna()
            if len(log_returns) > 1:
                volatility_value = float(log_returns.std())
            else:
                volatility_value = 0.01 
        else:
            volatility_value = 0.01 
            
        volatility_for_weighting = torch.tensor([[volatility_value]], dtype=torch.float32, device=DEVICE) # [1, 1]

        # 5. 計算策略權重
        # Ensure weight network outputs match effective_total_strategies
        _out_features_for_nets = max(1, effective_total_strategies)
        if self.weight_networks['volatility_net'][-2].out_features != _out_features_for_nets:
            logger.warning("Volatility net output features mismatch. This might indicate dynamic changes not fully propagated to network structure at init.")
            # This would ideally reinitialize or adapt the layer, but for now, we proceed.
            # The layer was initialized with _out_features_for_nets based on self.total_strategies.
            # If effective_total_strategies is different, this is a mismatch.
            # For safety, let's assume the initialized _target_weight_dim is the target.
            _target_weight_dim = self.weight_networks['volatility_net'][-2].out_features
        else:
            _target_weight_dim = _out_features_for_nets


        weights_vol = self.weight_networks['volatility_net'](volatility_for_weighting) 
        
        if self.state_dim > 0 :
            weights_regime = self.weight_networks['regime_net'](state_features_for_weighting)
        else: 
            # If state_dim is 0, regime_net input is (batch_size, 1) if its Linear layer was set to 1.
            # The init logic sets nn.Linear(max(1, state_dim), ...). So if state_dim=0, in_features=1.
            # We need to pass a tensor of shape [batch_size, 1]
            dummy_state_for_regime_net = torch.zeros(batch_size, 1, device=DEVICE) # Or ones, or rand
            weights_regime = self.weight_networks['regime_net'](dummy_state_for_regime_net)
            # weights_regime = torch.ones(batch_size, _target_weight_dim, device=DEVICE) / _target_weight_dim

        if self.state_dim > 0:
            corr_net_input = torch.cat([state_features_for_weighting, volatility_for_weighting], dim=1) 
        else: 
            # If state_dim is 0, corr_net input is nn.Linear(1, ...)
            corr_net_input = volatility_for_weighting 

        weights_corr = self.weight_networks['correlation_net'](corr_net_input)
        
        # Ensure all weight tensors have the same dimension (_target_weight_dim) before combining
        # This is a safeguard if dynamic changes led to mismatches not caught by init.
        # A more robust solution involves re-initializing layers or using adaptive layers.
        if weights_vol.shape[1] != _target_weight_dim: weights_vol = self._adaptive_dimension_handler(weights_vol.unsqueeze(-1), _target_weight_dim, "weights_vol_final").squeeze(-1)
        if weights_regime.shape[1] != _target_weight_dim: weights_regime = self._adaptive_dimension_handler(weights_regime.unsqueeze(-1), _target_weight_dim, "weights_regime_final").squeeze(-1)
        if weights_corr.shape[1] != _target_weight_dim: weights_corr = self._adaptive_dimension_handler(weights_corr.unsqueeze(-1), _target_weight_dim, "weights_corr_final").squeeze(-1)
        
        base_weights = (weights_vol + weights_regime + weights_corr) / 3.0
        
        # Ensure quantum_amplitudes match _target_weight_dim
        if self.quantum_amplitudes.shape[0] != _target_weight_dim:
            # This is a critical mismatch, log error and use a placeholder
            logger.error(f"Quantum amplitudes dim {self.quantum_amplitudes.shape[0]} mismatch with target_weight_dim {_target_weight_dim}")
            # Fallback: create amplitudes of the target dimension
            _amplitudes = torch.ones(_target_weight_dim, device=DEVICE) / math.sqrt(_target_weight_dim)
        else:
            _amplitudes = self.quantum_amplitudes

        normalized_amplitudes = F.softmax(_amplitudes, dim=0) 
        weighted_amplitudes = base_weights * normalized_amplitudes.unsqueeze(0) 
        
        # Ensure interaction_matrix matches _target_weight_dim
        if self.interaction_matrix.shape[0] != _target_weight_dim or self.interaction_matrix.shape[1] != _target_weight_dim:
            logger.error(f"Interaction matrix dim {self.interaction_matrix.shape} mismatch with target_weight_dim {_target_weight_dim}")
            _interaction_matrix = torch.eye(_target_weight_dim, device=DEVICE)
        else:
            _interaction_matrix = self.interaction_matrix.to(DEVICE)
            
        interacted_weights = torch.matmul(weighted_amplitudes, _interaction_matrix)

        # --- Define missing variables ---
        strategy_weights = F.softmax(interacted_weights, dim=-1) # [batch_size, _target_weight_dim]

        # strategy_tensor is [batch_size, effective_total_strategies, 1]
        # strategy_weights is [batch_size, _target_weight_dim]
        # If effective_total_strategies != _target_weight_dim, this is an issue.
        # Assuming they should match due to _target_weight_dim logic.
        if strategy_tensor.shape[1] != strategy_weights.shape[1]:
            logger.error(f"Mismatch between strategy_tensor dim1 ({strategy_tensor.shape[1]}) and strategy_weights dim1 ({strategy_weights.shape[1]})")
            # Fallback: Adjust strategy_tensor to match strategy_weights dimension for the multiplication
            # This is a patch; the underlying cause of mismatch should be fixed.
            st_squeezed = strategy_tensor.squeeze(-1) # [batch_size, effective_total_strategies]
            if st_squeezed.shape[1] > strategy_weights.shape[1]: # truncate
                st_squeezed = st_squeezed[:, :strategy_weights.shape[1]]
            elif st_squeezed.shape[1] < strategy_weights.shape[1]: # pad
                padding = torch.zeros(batch_size, strategy_weights.shape[1] - st_squeezed.shape[1], device=DEVICE)
                st_squeezed = torch.cat([st_squeezed, padding], dim=1)
            final_output_features = torch.sum(st_squeezed * strategy_weights, dim=1, keepdim=True)
        else:
            final_output_features = torch.sum(strategy_tensor.squeeze(-1) * strategy_weights, dim=1, keepdim=True)
        # final_output_features is [batch_size, 1]

        projected_actions = self.output_projection_layer(final_output_features) # [batch_size, action_dim]
        
        _effective_action_dim = self.action_dim if self.action_dim > 0 else 1
        if _effective_action_dim > 1:
            final_action_distribution = F.softmax(projected_actions, dim=-1)
        else:
            final_action_distribution = torch.tanh(projected_actions) # For single continuous action
        # --- End of defining missing variables ---

        diagnostic_info = {
            "raw_strategy_signals": strategy_tensor.squeeze(0).squeeze(-1).detach().cpu().numpy().tolist() if strategy_tensor is not None else [],
            "weights_volatility": weights_vol.squeeze(0).detach().cpu().numpy().tolist(),
            "weights_regime": weights_regime.squeeze(0).detach().cpu().numpy().tolist(),
            "weights_correlation": weights_corr.squeeze(0).detach().cpu().numpy().tolist(),
            "base_combined_weights": base_weights.squeeze(0).detach().cpu().numpy().tolist(),
            "quantum_amplitudes_normalized": normalized_amplitudes.detach().cpu().numpy().tolist(),
            "interacted_weights": interacted_weights.squeeze(0).detach().cpu().numpy().tolist(),
            "final_strategy_weights": strategy_weights.squeeze(0).detach().cpu().numpy().tolist(),
            "state_features_input": state_features_for_weighting.squeeze(0).detach().cpu().numpy().tolist() if self.state_dim > 0 else "N/A",
            "volatility_input": volatility_for_weighting.item(),
            "final_output_features_before_action_mapping": final_output_features.squeeze(0).detach().cpu().numpy().tolist()
        }
        
        return final_action_distribution, strategy_weights, diagnostic_info

    def update_strategy_performance(self, performance_scores: torch.Tensor):
        """更新策略性能歷史"""
        current_idx = self.performance_index.item() % 100
        self.strategy_performance_history[:, current_idx] = performance_scores.mean(dim=0)
        self.performance_index += 1
        
        # 更新動態策略生成器
        if self.enable_dynamic_generation:
            dynamic_performance = performance_scores[:, self.num_base_strategies:]
            if dynamic_performance.numel() > 0:
                self.dynamic_generator.evolve_strategies(dynamic_performance.mean(dim=0))
    
    def get_strategy_analysis(self) -> Dict[str, Any]:
        """獲取策略分析信息"""
        recent_performance = self.strategy_performance_history[:, :min(100, self.performance_index.item())]
        
        return {
            'num_strategies': len(self.base_strategies_instances) + (5 if self.enable_dynamic_generation else 0),
            'avg_performance': recent_performance.mean(dim=-1),
            'performance_std': recent_performance.std(dim=-1),
            'best_strategy_idx': recent_performance.mean(dim=-1).argmax().item(),
            'worst_strategy_idx': recent_performance.mean(dim=-1).argmin().item(),
            'strategy_names': [s.get_strategy_name() for s in self.base_strategies_instances] + 
                            ([f"Dynamic_{i}" for i in range(5)] if self.enable_dynamic_generation else []),
            'quantum_amplitude_distribution': F.softmax(self.quantum_amplitudes, dim=0),
            'entanglement_strength': self.entanglement_strength.item(),
        }


if __name__ == "__main__":
    # 測試增強版量子策略層
    logger.info("開始測試增強版量子策略層...")
    
    # 測試參數
    batch_size = 8
    state_dim = 64
    action_dim = 10
    
    # 創建測試數據
    test_state = torch.randn(batch_size, state_dim)
    test_volatility = torch.rand(batch_size) * 0.5
    
    # 初始化增強版策略疊加系統
    enhanced_strategy_layer = EnhancedStrategySuperposition(
        state_dim=state_dim,
        action_dim=action_dim,
        enable_dynamic_generation=True
    )
    
    try:
        # 前向傳播測試
        with torch.no_grad():
            output, info = enhanced_strategy_layer(test_state, test_volatility)
            
        logger.info(f"測試成功！")
        logger.info(f"輸入狀態形狀: {test_state.shape}")
        logger.info(f"輸出動作形狀: {output.shape}")
        logger.info(f"策略權重形狀: {info['strategy_weights'].shape}")
        logger.info(f"活躍策略數量: {info['num_active_strategies'].mean():.2f}")
        
        # 獲取策略分析
        analysis = enhanced_strategy_layer.get_strategy_analysis()
        logger.info(f"策略總數: {analysis['num_strategies']}")
        logger.info(f"策略名稱: {analysis['strategy_names']}")
        
        # 測試梯度計算
        enhanced_strategy_layer.train()
        output, info = enhanced_strategy_layer(test_state, test_volatility)
        loss = output.abs().mean()
        loss.backward()
        
        logger.info("梯度計算測試通過")
        logger.info(f"總參數量: {sum(p.numel() for p in enhanced_strategy_layer.parameters()):,}")
        
    except Exception as e:
        logger.error(f"測試失敗: {e}")
        raise e

    # ==============================================
    # 測試動態策略生成器的遺傳算法功能
    # ==============================================
    logger.info("\n開始測試動態策略生成器的遺傳算法功能...")
    
    # 加載真實數據
    try:
        import pandas as pd
        # 加載EUR/USD 5秒數據
        data_path = "data/EUR_USD_5S_20250601.csv"
        df = pd.read_csv(data_path)
        logger.info(f"成功加載數據: {data_path}, 形狀: {df.shape}")
        
        # 數據預處理
        # 使用收盤價作為狀態特徵
        prices = df['close'].values
        # 計算波動率
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        # 轉換為PyTorch張量
        state_tensor = torch.tensor(prices[-state_dim:], dtype=torch.float32).unsqueeze(0)
        volatility_tensor = torch.tensor([volatility], dtype=torch.float32)
        
        # 初始化動態策略生成器
        generator = DynamicStrategyGenerator(
            state_dim=state_dim,
            action_dim=action_dim,
            base_strategies=enhanced_strategy_layer.base_strategies,
            max_generated_strategies=5
        )
        
        # 測試策略生成
        strategies, fitness_scores = generator.generate_strategies(state_tensor)
        logger.info(f"生成策略數量: {len(strategies)}")
        logger.info(f"策略輸出形狀: {strategies[0].shape}")
        logger.info(f"適應度分數形狀: {fitness_scores.shape}")
        
        # 測試遺傳算法操作
        logger.info("\n測試遺傳算法操作:")
        logger.info("1. 測試輪盤賭選擇...")
        selected_idx = generator.roulette_wheel_selection(fitness_scores)
        logger.info(f"選擇的索引: {selected_idx.item()}")
        
        logger.info("2. 測試單點交叉...")
        parent1 = torch.randn(1, generator.gene_latent_dim)
        parent2 = torch.randn(1, generator.gene_latent_dim)
        child1, child2 = generator.single_point_crossover(parent1, parent2)
        logger.info(f"父代1形狀: {parent1.shape}, 父代2形狀: {parent2.shape}")
        logger.info(f"子代1形狀: {child1.shape}, 子代2形狀: {child2.shape}")
        
        logger.info("3. 測試高斯變異...")
        genes = torch.randn(1, generator.gene_latent_dim)
        mutated_genes = generator.gaussian_mutation(genes)
        logger.info(f"變異前: {genes.mean().item():.4f}, 變異後: {mutated_genes.mean().item():.4f}")
        
        logger.info("✅ 動態策略生成器測試通過")
        
    except Exception as e:
        logger.error(f"動態策略生成器測試失敗: {e}")
        raise e

# Placeholder for GeneticOptimizer and NeuralArchitectureSearch if they are to be defined in this file
# class GeneticOptimizer:
#     def __init__(self, ...):
#         pass
#     def evolve_strategy(self, ...):
#         pass

# class NeuralArchitectureSearch:
#     def __init__(self, ...):
#         pass
#     def search_architecture(self, ...):
#         pass
