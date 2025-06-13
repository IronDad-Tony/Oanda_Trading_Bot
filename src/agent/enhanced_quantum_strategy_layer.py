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
from torch.nn.functional import gumbel_softmax
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Type, Callable
import logging # Use standard logging
from abc import ABC, abstractmethod
import random # For DynamicStrategyGenerator
from dataclasses import asdict # ADDED: Import asdict

from .strategies import STRATEGY_REGISTRY # From src/agent/strategies/__init__.py
from .strategies.base_strategy import StrategyConfig, BaseStrategy

class EnhancedStrategySuperposition(nn.Module):
    def __init__(self, input_dim: int, num_strategies: int, # num_strategies is the target/max number
                 strategy_configs: Optional[List[Union[Dict[str, Any], StrategyConfig]]] = None, 
                 explicit_strategies: Optional[List[Type[BaseStrategy]]] = None, 
                 dropout_rate: float = 0.1, 
                 initial_temperature: float = 1.0, 
                 use_gumbel_softmax: bool = True, 
                 strategy_input_dim: int = 64, # Default input_dim for individual strategies
                 dynamic_loading_enabled: bool = True):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.input_dim = input_dim
        self.target_num_strategies = num_strategies # Renaming for clarity from the input arg
        self.strategy_configs = strategy_configs
        self.explicit_strategies = explicit_strategies
        self.dropout_rate = dropout_rate
        self.initial_temperature = initial_temperature
        self.temperature = nn.Parameter(torch.tensor(initial_temperature), requires_grad=False)
        self.use_gumbel_softmax = use_gumbel_softmax
        self.strategy_input_dim = strategy_input_dim # Default for strategies if not in their config
        self.dynamic_loading_enabled = dynamic_loading_enabled

        self.strategies = nn.ModuleList()
        self.strategy_names: List[str] = []
        self.num_actual_strategies = 0

        # Dynamic Strategy Generator (optional)
        # self.dynamic_generator = DynamicStrategyGenerator(...) # Potentially initialize here or pass in

        self._initialize_strategies(input_dim, strategy_configs, explicit_strategies, strategy_input_dim, dynamic_loading_enabled)

        if self.num_actual_strategies > 0:
            # Attention mechanism to learn strategy weights
            self.attention_network = nn.Linear(input_dim, self.num_actual_strategies)
            if self.dropout_rate > 0:
                self.dropout = nn.Dropout(self.dropout_rate)
            else:
                self.dropout = nn.Identity() # No dropout
        else:
            self.logger.warning("No strategies loaded, attention network and dropout layer not created.")
            self.attention_network = None
            self.dropout = nn.Identity()
        
        self.gumbel_selector = None # For Gumbel-Softmax based selection if used

    def _initialize_strategies(self, layer_input_dim: int, 
                               strategy_configs: Optional[List[Union[Dict[str, Any], StrategyConfig]]], 
                               explicit_strategies: Optional[List[Type[BaseStrategy]]], 
                               default_strategy_input_dim: int, 
                               dynamic_loading_enabled: bool):
        self.strategies = nn.ModuleList()
        self.strategy_names = []

        processed_strategy_names = set() # To avoid duplicates

        # Process strategy_configs first
        if strategy_configs:
            for cfg_item_idx, cfg_item in enumerate(strategy_configs):
                strategy_class: Optional[Type[BaseStrategy]] = None
                final_config: Optional[StrategyConfig] = None

                if isinstance(cfg_item, type) and issubclass(cfg_item, BaseStrategy):
                    strategy_class = cfg_item
                    # Check if a config for this class is explicitly provided later in the list
                    # This is a bit complex; simpler if configs are always separate or always dicts/StrategyConfig objects
                    explicit_config_for_class = None
                    for next_item in strategy_configs[cfg_item_idx+1:]: # Look ahead
                        if isinstance(next_item, StrategyConfig) and next_item.name == strategy_class.__name__:
                            explicit_config_for_class = next_item
                            break
                        if isinstance(next_item, dict) and next_item.get('name') == strategy_class.__name__:
                            explicit_config_for_class = StrategyConfig(**next_item)
                            break
                    
                    default_cfg = strategy_class.default_config()
                    if not isinstance(default_cfg, StrategyConfig):
                        self.logger.warning(f"{strategy_class.__name__}.default_config() did not return StrategyConfig (got {type(default_cfg)}). Using minimal.")
                        default_cfg = StrategyConfig(name=strategy_class.__name__)
                    
                    if explicit_config_for_class:
                        final_config = StrategyConfig.merge_configs(default_cfg, explicit_config_for_class)
                    else:
                        final_config = default_cfg

                elif isinstance(cfg_item, StrategyConfig):
                    if cfg_item.name in STRATEGY_REGISTRY:
                        strategy_class = STRATEGY_REGISTRY[cfg_item.name]
                        default_cfg = strategy_class.default_config()
                        if not isinstance(default_cfg, StrategyConfig):
                            self.logger.warning(f"{strategy_class.__name__}.default_config() did not return StrategyConfig (got {type(default_cfg)}). Using minimal.")
                            default_cfg = StrategyConfig(name=strategy_class.__name__)
                        final_config = StrategyConfig.merge_configs(default_cfg, cfg_item)
                    else:
                        self.logger.warning(f"Strategy name '{cfg_item.name}' from StrategyConfig not in STRATEGY_REGISTRY. Skipping.")
                        continue
                
                elif isinstance(cfg_item, dict):
                    name = cfg_item.get("name")
                    if name and name in STRATEGY_REGISTRY:
                        strategy_class = STRATEGY_REGISTRY[name]
                        try:
                            explicit_cfg_from_dict = StrategyConfig(**cfg_item)
                            default_cfg = strategy_class.default_config()
                            if not isinstance(default_cfg, StrategyConfig):
                                self.logger.warning(f"{strategy_class.__name__}.default_config() did not return StrategyConfig (got {type(default_cfg)}). Using minimal.")
                                default_cfg = StrategyConfig(name=strategy_class.__name__)
                            final_config = StrategyConfig.merge_configs(default_cfg, explicit_cfg_from_dict)
                        except Exception as e:
                            self.logger.error(f"Error creating StrategyConfig from dict for {name}: {e}. Skipping.")
                            continue
                    else:
                        self.logger.warning(f"Strategy name '{name}' from dict not in STRATEGY_REGISTRY or name missing. Skipping.")
                        continue
                else:
                    self.logger.warning(f"Unsupported configuration item type: {type(cfg_item)}. Skipping.")
                    continue

                if strategy_class and final_config:
                    if strategy_class.__name__ in processed_strategy_names:
                        self.logger.info(f"Strategy {strategy_class.__name__} already processed. Skipping duplicate.")
                        continue

                    # Determine input_dim for the strategy
                    # Priority: final_config.input_dim > default_strategy_input_dim > layer_input_dim
                    strat_input_dim = final_config.input_dim
                    if strat_input_dim is None:
                        strat_input_dim = default_strategy_input_dim
                    if strat_input_dim is None: # Should not happen if default_strategy_input_dim has a value
                         strat_input_dim = layer_input_dim 
                         self.logger.debug(f"Strategy {strategy_class.__name__} input_dim fell back to layer_input_dim: {layer_input_dim}")
                    
                    final_config.input_dim = strat_input_dim # Ensure it's set in the config object

                    self.logger.debug(f"Initializing strategy {strategy_class.__name__} with config: {asdict(final_config)} and input_dim: {strat_input_dim}") # Use asdict for logging
                    try:
                        # MODIFIED: Remove explicit input_dim from constructor call
                        self.strategies.append(strategy_class(config=final_config))
                        self.strategy_names.append(strategy_class.__name__)
                        processed_strategy_names.add(strategy_class.__name__)
                    except Exception as e:
                        self.logger.error(f"Error instantiating strategy {strategy_class.__name__} with config {asdict(final_config)}: {e}", exc_info=True) # Use asdict
        
        # Process explicit_strategies (if they weren\'t covered by name in strategy_configs)
        if explicit_strategies:
            for strat_class in explicit_strategies:
                if strat_class.__name__ not in processed_strategy_names: # Avoid duplicates
                    default_cfg = strat_class.default_config()
                    if not isinstance(default_cfg, StrategyConfig):
                        self.logger.warning(f"{strat_class.__name__}.default_config() did not return StrategyConfig (got {type(default_cfg)}). Using minimal.")
                        default_cfg = StrategyConfig(name=strat_class.__name__)
                    
                    strat_input_dim = default_cfg.input_dim
                    if strat_input_dim is None:
                        strat_input_dim = default_strategy_input_dim
                    if strat_input_dim is None: # Should not happen if default_strategy_input_dim has a value
                         strat_input_dim = layer_input_dim
                    default_cfg.input_dim = strat_input_dim

                    self.logger.debug(f"Initializing explicit strategy {strat_class.__name__} with default config: {asdict(default_cfg)} and input_dim: {strat_input_dim}") # Use asdict
                    try:
                        # MODIFIED: Remove explicit input_dim from constructor call
                        self.strategies.append(strat_class(config=default_cfg))
                        self.strategy_names.append(strat_class.__name__)
                        processed_strategy_names.add(strat_class.__name__)
                    except Exception as e:
                        self.logger.error(f"Error instantiating explicit strategy {strat_class.__name__} with default config: {e}", exc_info=True)

        # Dynamic loading if enabled and not enough strategies
        # This part needs self.dynamic_generator to be initialized
        # For now, this is a placeholder for where dynamic loading logic would go.
        # if dynamic_loading_enabled and len(self.strategies) < self.target_num_strategies and hasattr(self, 'dynamic_generator') and self.dynamic_generator:
        #     self.logger.info("Attempting dynamic strategy generation...")
            # Call dynamic_generator to create more strategies up to target_num_strategies

        self.num_actual_strategies = len(self.strategies)
        if self.num_actual_strategies == 0:
            self.logger.warning("EnhancedStrategySuperposition: No strategies were successfully initialized.")
        else:
            self.logger.info(f"EnhancedStrategySuperposition initialized with {self.num_actual_strategies} strategies: {self.strategy_names}")

    def forward(self, 
                asset_features_batch: torch.Tensor, 
                market_state_features: Optional[torch.Tensor] = None,
                current_positions_batch: Optional[torch.Tensor] = None,
                timestamps: Optional[List[pd.Timestamp]] = None, # Assuming a list of timestamps, one per batch item
                external_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Main forward pass for the EnhancedStrategySuperposition layer.

        Args:
            asset_features_batch (torch.Tensor): Batch of asset features. 
                                                 Shape: (batch_size, num_assets, sequence_length, feature_dim)
            market_state_features (Optional[torch.Tensor]): Market state features for attention.
                                                            Shape: (batch_size, market_feature_dim)
            current_positions_batch (Optional[torch.Tensor]): Current positions for each asset.
                                                              Shape: (batch_size, num_assets, 1)
            timestamps (Optional[List[pd.Timestamp]]): List of timestamps for each item in the batch.
            external_weights (Optional[torch.Tensor]): Externally provided strategy weights.
                                                       Shape: (batch_size, num_actual_strategies)

        Returns:
            torch.Tensor: Combined action signals from all strategies.
                          Shape: (batch_size, num_assets, 1) 
        """
        if self.num_actual_strategies == 0:
            self.logger.warning("No strategies loaded. Returning zero actions.")
            batch_size = asset_features_batch.shape[0]
            num_assets = asset_features_batch.shape[1]
            return torch.zeros(batch_size, num_assets, 1, device=asset_features_batch.device)

        batch_size, num_assets, seq_len, feature_dim = asset_features_batch.shape
        
        # 1. Determine Strategy Weights
        strategy_weights: torch.Tensor
        if external_weights is not None and external_weights.shape == (batch_size, self.num_actual_strategies):
            strategy_weights = F.softmax(external_weights / self.temperature, dim=1)
            self.logger.debug("Using external strategy weights.")
        elif self.attention_network is not None and market_state_features is not None:
            if market_state_features.shape[0] != batch_size or market_state_features.shape[1] != self.input_dim:
                self.logger.warning(
                    f"market_state_features shape mismatch. Expected ({batch_size}, {self.input_dim}), "
                    f"got {market_state_features.shape}. Falling back to uniform weights."
                )
                strategy_weights = torch.ones(batch_size, self.num_actual_strategies, device=asset_features_batch.device) / self.num_actual_strategies
            else:
                # Ensure attention_network is on the same device as market_state_features
                target_device = market_state_features.device
                try:
                    # Check device of the first parameter of attention_network
                    current_attn_device = next(self.attention_network.parameters()).device
                    if current_attn_device != target_device:
                        self.logger.info(f"Moving attention_network from {current_attn_device} to {target_device} to match market_state_features.")
                        self.attention_network.to(target_device) # Move the module
                except StopIteration:
                    # This case (module with no parameters) should not happen for nn.Linear.
                    self.logger.warning("attention_network appears to have no parameters. Cannot verify or move device. This might lead to errors if it's not on the correct device.")
                
                attention_logits = self.attention_network(market_state_features) # (batch_size, num_actual_strategies)
                if self.use_gumbel_softmax:
                    if self.training: # Gumbel-Softmax during training
                        strategy_weights = gumbel_softmax(attention_logits, tau=self.temperature, hard=False, dim=1)
                    else: # Argmax (hard selection) during inference
                        # strategy_weights = F.one_hot(torch.argmax(attention_logits, dim=1), num_classes=self.num_actual_strategies).float()
                        # Softmax for inference is generally preferred over hard selection unless specifically required
                        strategy_weights = F.softmax(attention_logits / self.temperature, dim=1) 
                else: # Standard Softmax
                    strategy_weights = F.softmax(attention_logits / self.temperature, dim=1)
                self.logger.debug("Using attention-based internal strategy weights.")
        else:
            self.logger.warning("No valid external weights or attention mechanism for market_state_features. Using uniform weights.")
            strategy_weights = torch.ones(batch_size, self.num_actual_strategies, device=asset_features_batch.device) / self.num_actual_strategies

        strategy_weights = self.dropout(strategy_weights) # Apply dropout to weights

        # 2. Get signals from each strategy for each asset
        # Initialize a tensor to store all strategy signals: (batch_size, num_assets, num_strategies)
        all_strategy_signals = torch.zeros(batch_size, num_assets, self.num_actual_strategies, device=asset_features_batch.device)

        for asset_idx in range(num_assets):
            # Extract features for the current asset: (batch_size, seq_len, feature_dim)
            single_asset_features = asset_features_batch[:, asset_idx, :, :]
            
            current_pos_for_asset = None
            if current_positions_batch is not None:
                # Ensure current_positions_batch has the right shape for slicing
                if current_positions_batch.ndim == 3 and current_positions_batch.shape[1] == num_assets:
                    current_pos_for_asset = current_positions_batch[:, asset_idx, :] # (batch_size, 1)
                elif current_positions_batch.ndim == 2 and current_positions_batch.shape[0] == batch_size and num_assets == 1: # Special case for single asset
                     current_pos_for_asset = current_positions_batch # (batch_size, 1) assuming it's already for the single asset
                else:
                    self.logger.warning(f"current_positions_batch shape {current_positions_batch.shape} not as expected for asset slicing. Skipping for asset {asset_idx}.")


            # Collect signals from each strategy for this asset
            # strategy_outputs_for_asset = [] # List to hold (batch_size, 1, 1) from each strategy
            for i, strategy_module in enumerate(self.strategies):
                # The strategy's forward method expects (batch_size, seq_len, num_features)
                # or (batch_size, num_features) if it doesn't handle sequences internally.
                # Most of our strategies now expect (batch_size, sequence_length, num_features)
                # and return (batch_size, 1, 1)
                
                # Pass timestamp if the strategy can use it.
                # For now, assuming strategies take a single timestamp if any.
                # If strategies need per-batch-item timestamps, their forward methods would need to handle a list.
                current_timestamp_for_strategy = timestamps[0] if timestamps and len(timestamps) > 0 else None
                
                try:
                    # Ensure strategy_input_dim matches what the strategy was configured for
                    # This check might be redundant if initialization ensures this.
                    # if single_asset_features.shape[2] != strategy_module.config.input_dim:
                    #    self.logger.error(f"Strategy {strategy_module.config.name} expects input_dim {strategy_module.config.input_dim}, got {single_asset_features.shape[2]}")
                    #    # Handle mismatch, e.g., skip strategy or use zeros
                    #    signal_tensor = torch.zeros(batch_size, 1, 1, device=asset_features_batch.device)
                    # else:

                    signal_tensor = strategy_module.forward(
                        single_asset_features, 
                        current_positions=current_pos_for_asset,
                        timestamp=current_timestamp_for_strategy # Pass single timestamp
                    ) # Expected: (batch_size, 1, 1)
                    
                    if signal_tensor.shape != (batch_size, 1, 1):
                        self.logger.warning(f"Strategy {strategy_module.config.name} for asset {asset_idx} produced signal of shape {signal_tensor.shape}, expected ({batch_size}, 1, 1). Reshaping/ignoring.")
                        # Attempt to fix or use zeros
                        if signal_tensor.numel() == batch_size:
                            signal_tensor = signal_tensor.view(batch_size, 1, 1)
                        else:
                            signal_tensor = torch.zeros(batch_size, 1, 1, device=asset_features_batch.device)
                    
                    all_strategy_signals[:, asset_idx, i] = signal_tensor.squeeze(-1).squeeze(-1) # Store as (batch_size)

                except Exception as e:
                    self.logger.error(f"Error in strategy {strategy_module.config.name} forward for asset {asset_idx}: {e}", exc_info=True)
                    # Store zeros if a strategy fails
                    all_strategy_signals[:, asset_idx, i] = 0.0


        # 3. Combine signals using weights
        # strategy_weights is (batch_size, num_actual_strategies)
        # all_strategy_signals is (batch_size, num_assets, num_actual_strategies)
        
        # We want to compute: sum_s (weight_s * signal_s_a) for each asset a
        # Reshape weights for broadcasting: (batch_size, 1, num_actual_strategies)
        expanded_weights = strategy_weights.unsqueeze(1)
        
        # Element-wise multiplication and sum over strategies dimension
        # (batch_size, num_assets, num_actual_strategies) * (batch_size, 1, num_actual_strategies)
        # -> (batch_size, num_assets, num_actual_strategies) then sum over dim 2
        combined_signals = torch.sum(all_strategy_signals * expanded_weights, dim=2) # (batch_size, num_assets)
        
        # Reshape to final output: (batch_size, num_assets, 1)
        final_actions = combined_signals.unsqueeze(-1)
        
        self.logger.debug(f"ESS Forward: Input asset_features_batch shape: {asset_features_batch.shape}")
        self.logger.debug(f"ESS Forward: Strategy weights shape: {strategy_weights.shape}")
        self.logger.debug(f"ESS Forward: All strategy signals shape: {all_strategy_signals.shape}")
        self.logger.debug(f"ESS Forward: Final actions shape: {final_actions.shape}")

        return final_actions

    # ... (rest of EnhancedStrategySuperposition, e.g., dynamic generator integration)

class DynamicStrategyGenerator:
    def __init__(self, base_strategies: Optional[List[Type[BaseStrategy]]] = None, generation_config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(f"{__name__}.DynamicStrategyGenerator")
        self.base_strategies = base_strategies if base_strategies else []
        self.generation_config = generation_config if generation_config else {}
        if not self.base_strategies:
            self.logger.warning("DynamicStrategyGenerator initialized with no base strategies.")

    def generate_new_strategy(self, market_conditions: Dict[str, Any], existing_strategies: List[BaseStrategy]) -> Optional[Tuple[Type[BaseStrategy], StrategyConfig]]:
        if not self.base_strategies:
            self.logger.warning("No base strategies available for dynamic generation.")
            return None
        
        try:
            selected_base_strategy_class = random.choice(self.base_strategies)
        except IndexError:
            self.logger.error("Base strategies list is empty, cannot select a strategy for generation.")
            return None
        
        self.logger.info(f"Attempting to generate a new strategy based on {selected_base_strategy_class.__name__}")

        base_config = selected_base_strategy_class.default_config()
        if not isinstance(base_config, StrategyConfig):
            self.logger.warning(f"{selected_base_strategy_class.__name__}.default_config() did not return a StrategyConfig object (got {type(base_config)}). Using default StrategyConfig.")
            base_config = StrategyConfig(name=selected_base_strategy_class.__name__)
        
        # Placeholder for actual generation logic (e.g., modifying base_config based on market_conditions)
        # For now, returns the selected class and its (potentially modified) base config
        # Example modification:
        # new_params = base_config.default_params.copy()
        # if 'volatility' in market_conditions and 'window' in new_params:
        #     new_params['window'] = int(new_params['window'] * (1 + market_conditions['volatility'])) # Adjust window by volatility
        # generated_config = StrategyConfig(name=f"Dynamic_{selected_base_strategy_class.__name__}", default_params=new_params, ...)
        
        # For this iteration, we just return the base class and its default/base config
        # The layer would then instantiate it.
        # A more advanced generator might return an already configured *instance* or a new *subclass*.
        # Returning (Class, Config) seems a reasonable contract for now.
        self.logger.info(f"Generated strategy config for {selected_base_strategy_class.__name__} (using base config).")
        return selected_base_strategy_class, base_config

    # ... (rest of DynamicStrategyGenerator)
