# src/oanda_trading_bot/training_system/strategies/strategy_pool_manager.py
"""
策略池管理器 (Strategy Pool Manager)。
負責管理和組合多個交易策略，並根據市場狀態動態調整策略權重。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Type, Callable, Set
import logging
from abc import ABC, abstractmethod
import random
from dataclasses import asdict
import json
import copy

# 導入必要的模組
# 這裡的 .strategies 是指 src/oanda_trading_bot/training_system/agent/strategies/__init__.py
# .base_strategy 是指 src/oanda_trading_bot/training_system/agent/strategies/base_strategy.py
# .optimizers.genetic_optimizer 是指 src/oanda_trading_bot/training_system/agent/optimizers/genetic_optimizer.py
# .optimizers.neural_architecture_search 是指 src/oanda_trading_bot/training_system/agent/optimizers/neural_architecture_search.py

from . import STRATEGY_REGISTRY
from .base_strategy import StrategyConfig, BaseStrategy
from ..agent.optimizers.genetic_optimizer import GeneticOptimizer
from ..agent.optimizers.neural_architecture_search import NeuralArchitectureSearch

class StrategyPoolManager(nn.Module):
    def __init__(self, input_dim: int, num_strategies: int, # num_strategies is the target/max number
                 strategy_configs: Optional[List[Union[Dict[str, Any], StrategyConfig]]] = None,
                 explicit_strategies: Optional[List[Type[BaseStrategy]]] = None,
                 strategy_config_file_path: Optional[str] = None,
                 dropout_rate: float = 0.1,
                 initial_temperature: float = 1.0,
                 use_gumbel_softmax: bool = True,
                 strategy_input_dim: int = 64, # Default input_dim for individual strategies
                 dynamic_loading_enabled: bool = True,
                 adaptive_learning_rate: float = 0.01,
                 performance_ema_alpha: float = 0.1,
                 raw_feature_dim: int = 0, # 新增原始特徵維度
                 timesteps_history: int = 0): # 新增歷史時間步長
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.input_dim = input_dim # For attention network
        self.target_num_strategies = num_strategies

        # Initialize self.strategy_input_dim with the value from __init__ param first
        self.strategy_input_dim = strategy_input_dim

        loaded_strategy_configs_from_file: List[Union[Dict[str, Any], StrategyConfig]] = []
        if strategy_config_file_path:
            try:
                with open(strategy_config_file_path, 'r') as f:
                    config_data = json.load(f)
                    if "strategies" in config_data and isinstance(config_data["strategies"], list):
                        loaded_strategy_configs_from_file.extend(config_data["strategies"])
                        self.logger.info(f"Successfully loaded {len(loaded_strategy_configs_from_file)} strategy configurations from {strategy_config_file_path}")
                    else:
                        self.logger.warning(f"Strategy config file {strategy_config_file_path} does not contain a 'strategies' list.")

                    # Load and apply global_strategy_input_dim from file, potentially overriding self.strategy_input_dim
                    if "global_strategy_input_dim" in config_data and isinstance(config_data["global_strategy_input_dim"], int):
                        file_global_dim = config_data["global_strategy_input_dim"]
                        self.logger.info(f"Found global_strategy_input_dim {file_global_dim} in config file. This will be used as the default strategy input_dim.")
                        self.strategy_input_dim = file_global_dim # Update self.strategy_input_dim

            except FileNotFoundError:
                self.logger.warning(f"Strategy config file not found: {strategy_config_file_path}")
            except json.JSONDecodeError as e:
                self.logger.error(f"Error decoding strategy config file {strategy_config_file_path}: {type(e).__name__} - {e}")
            except Exception as e:
                self.logger.error(f"An unexpected error occurred while loading strategy config file {strategy_config_file_path}: {e}")

        final_strategy_configs = loaded_strategy_configs_from_file
        if strategy_configs:
            final_strategy_configs.extend(strategy_configs)

        self.strategy_configs = final_strategy_configs
        self.explicit_strategies = explicit_strategies
        self.dropout_rate = dropout_rate
        self.initial_temperature = initial_temperature
        self.temperature = nn.Parameter(torch.tensor(initial_temperature), requires_grad=False)
        self.use_gumbel_softmax = use_gumbel_softmax
        self.dynamic_loading_enabled = dynamic_loading_enabled
        self.adaptive_learning_rate = adaptive_learning_rate
        self.performance_ema_alpha = performance_ema_alpha

        self.strategies = nn.ModuleList()
        self.strategy_names: List[str] = []
        self.num_actual_strategies = 0

        self._initialize_strategies(
            self.input_dim,
            self.strategy_configs,
            self.explicit_strategies,
            self.strategy_input_dim,
            self.dynamic_loading_enabled,
            self.raw_feature_dim,
            self.timesteps_history
        )

        if self.num_actual_strategies > 0:
            self.attention_network = nn.Linear(self.input_dim, self.num_actual_strategies)
            if self.dropout_rate > 0:
                self.dropout = nn.Dropout(self.dropout_rate)
            else:
                self.dropout = nn.Identity()

            self.adaptive_bias_weights = nn.Parameter(torch.zeros(self.num_actual_strategies), requires_grad=False)
            self.strategy_performance_ema = nn.Parameter(torch.zeros(self.num_actual_strategies), requires_grad=False)
            self.logger.info(f"Initialized adaptive weighting components for {self.num_actual_strategies} strategies.")
        else:
            self.logger.warning("No strategies loaded, attention network, and adaptive components not created.")
            self.attention_network = None
            self.dropout = nn.Identity()
            self.adaptive_bias_weights = None
            self.strategy_performance_ema = None

        self.gumbel_selector = None
        self.last_strategy_weights: Optional[torch.Tensor] = None
        self.last_all_strategy_signals: Optional[torch.Tensor] = None
        self.last_combined_signals: Optional[torch.Tensor] = None
        self.last_attention_logits: Optional[torch.Tensor] = None
        self.last_market_state_features: Optional[torch.Tensor] = None

    def _get_strategy_registry(self) -> Dict[str, Type[BaseStrategy]]:
        return copy.deepcopy(STRATEGY_REGISTRY)

    def _initialize_strategies(self, layer_input_dim: int,
                                processed_strategy_configs: Optional[List[Union[Dict[str, Any], StrategyConfig]]],
                                explicit_strategies: Optional[List[Type[BaseStrategy]]],
                                default_strategy_input_dim: int,
                                dynamic_loading_enabled: bool,
                                raw_feature_dim: int, # 新增參數
                                timesteps_history: int): # 新增參數
        self.strategies = nn.ModuleList()
        self.strategy_names = []

        processed_strategy_names: Set[str] = set()
        STRATEGY_REGISTRY_LOCAL = self._get_strategy_registry()

        self.logger.info(f"Initializing strategies. Default strategy input_dim: {default_strategy_input_dim}. Dynamic loading: {dynamic_loading_enabled}.")

        if processed_strategy_configs:
            self.logger.info(f"Processing {len(processed_strategy_configs)} strategy configurations.")
            for cfg_item_idx, cfg_item in enumerate(processed_strategy_configs):
                strategy_class: Optional[Type[BaseStrategy]] = None
                final_config_obj: Optional[StrategyConfig] = None
                strategy_name_from_cfg: Optional[str] = None

                if isinstance(cfg_item, type) and issubclass(cfg_item, BaseStrategy):
                    strategy_class = cfg_item
                    strategy_name_from_cfg = strategy_class.__name__
                    base_default_cfg = strategy_class.default_config()
                    if not isinstance(base_default_cfg, StrategyConfig):
                        self.logger.warning(f"Default config for {strategy_name_from_cfg} is not StrategyConfig. Using minimal.")
                        base_default_cfg = StrategyConfig(name=strategy_name_from_cfg)
                    final_config_obj = base_default_cfg

                elif isinstance(cfg_item, StrategyConfig):
                    strategy_name_from_cfg = cfg_item.name
                    if strategy_name_from_cfg in STRATEGY_REGISTRY_LOCAL:
                        strategy_class = STRATEGY_REGISTRY_LOCAL[strategy_name_from_cfg]
                        base_default_cfg = strategy_class.default_config()
                        if not isinstance(base_default_cfg, StrategyConfig):
                             self.logger.warning(f"Default config for {strategy_name_from_cfg} is not StrategyConfig. Using minimal.")
                             base_default_cfg = StrategyConfig(name=strategy_name_from_cfg)
                        final_config_obj = StrategyConfig.merge_configs(base_default_cfg, cfg_item)
                    else:
                        self.logger.warning(f"Strategy name '{strategy_name_from_cfg}' from StrategyConfig not in STRATEGY_REGISTRY. Skipping.")
                        continue

                elif isinstance(cfg_item, dict):
                    strategy_name_from_cfg = cfg_item.get("name")
                    if not strategy_name_from_cfg:
                        self.logger.warning(f"Strategy config dict item at index {cfg_item_idx} missing 'name'. Skipping: {cfg_item}")
                        continue

                    if strategy_name_from_cfg in STRATEGY_REGISTRY_LOCAL:
                        strategy_class = STRATEGY_REGISTRY_LOCAL[strategy_name_from_cfg]
                        base_default_cfg = strategy_class.default_config()
                        if not isinstance(base_default_cfg, StrategyConfig):
                             self.logger.warning(f"Default config for {strategy_name_from_cfg} is not StrategyConfig. Using minimal.")
                             base_default_cfg = StrategyConfig(name=strategy_name_from_cfg)

                        dict_based_cfg = StrategyConfig(**cfg_item)
                        final_config_obj = StrategyConfig.merge_configs(base_default_cfg, dict_based_cfg)
                    else:
                        self.logger.warning(f"Strategy name '{strategy_name_from_cfg}' from dict config not in STRATEGY_REGISTRY. Skipping.")
                        continue
                else:
                    self.logger.warning(f"Unsupported item type in strategy_configs at index {cfg_item_idx}: {type(cfg_item)}. Skipping.")
                    continue

                if strategy_class and final_config_obj and strategy_name_from_cfg:
                    if strategy_name_from_cfg in processed_strategy_names:
                        self.logger.warning(f"Strategy '{strategy_name_from_cfg}' already processed. Skipping duplicate from config list.")
                        continue

                    if final_config_obj.input_dim is None:
                        final_config_obj.input_dim = default_strategy_input_dim
                    elif final_config_obj.input_dim != default_strategy_input_dim:
                        self.logger.info(f"Strategy '{strategy_name_from_cfg}' config specifies input_dim {final_config_obj.input_dim}, "
                                         f"overriding layer's default_strategy_input_dim {default_strategy_input_dim}.")

                    try:
                        # 傳遞 raw_feature_dim 和 timesteps_history 給策略
                        instance = strategy_class(
                            config=final_config_obj,
                            raw_feature_dim=raw_feature_dim,
                            timesteps_history=timesteps_history
                        )
                        self.strategies.append(instance)
                        self.strategy_names.append(strategy_name_from_cfg)
                        processed_strategy_names.add(strategy_name_from_cfg)
                        self.logger.info(f"Loaded strategy '{strategy_name_from_cfg}' from config with input_dim {final_config_obj.input_dim}.")
                    except Exception as e_load_cfg:
                        self.logger.error(f"Error loading strategy '{strategy_name_from_cfg}' from config: {e_load_cfg}", exc_info=True)


        if explicit_strategies:
            self.logger.info(f"Processing {len(explicit_strategies)} explicit strategies provided to constructor.")
            for explicit_class in explicit_strategies:
                name = explicit_class.__name__
                if name not in processed_strategy_names:
                    self.logger.info(f"Loading explicit strategy '{name}'.")
                    try:
                        default_cfg_obj = explicit_class.default_config()
                        if not isinstance(default_cfg_obj, StrategyConfig):
                             self.logger.warning(f"Default config for explicit strategy {name} is not a StrategyConfig instance. Using minimal.")
                             default_cfg_obj = StrategyConfig(name=name)

                        if default_cfg_obj.input_dim is None:
                            default_cfg_obj.input_dim = default_strategy_input_dim
                        elif default_cfg_obj.input_dim != default_strategy_input_dim:
                             self.logger.info(f"Explicit strategy '{name}' default_config.input_dim ({default_cfg_obj.input_dim}) "
                                                f"differs from layer's default_strategy_input_dim ({default_strategy_input_dim}). Using strategy's default.")

                       # 傳遞 raw_feature_dim 和 timesteps_history 給策略
                       instance = explicit_class(
                           config=default_cfg_obj,
                           raw_feature_dim=raw_feature_dim,
                           timesteps_history=timesteps_history
                       )
                       self.strategies.append(instance)
                       self.strategy_names.append(name)
                       processed_strategy_names.add(name)
                       self.logger.info(f"Successfully loaded and added explicit strategy '{name}' with input_dim {default_cfg_obj.input_dim}.")
                   except Exception as e_expl_load:
                       self.logger.error(f"Error loading explicit strategy '{name}': {e_expl_load}", exc_info=True)
               else:
                   self.logger.info(f"Explicit strategy '{name}' was already loaded via configuration. Skipping duplicate.")

        if dynamic_loading_enabled:
            self.logger.info(f"Dynamic loading enabled. Checking STRATEGY_REGISTRY for additional strategies.")
            for name, strategy_class_from_registry in STRATEGY_REGISTRY_LOCAL.items():
                if name not in processed_strategy_names:
                    self.logger.info(f"Dynamically loading strategy '{name}' from registry.")
                    try:
                        default_cfg_obj = strategy_class_from_registry.default_config()
                        if not isinstance(default_cfg_obj, StrategyConfig):
                             self.logger.warning(f"Default config for {name} from registry is not a StrategyConfig instance. Using minimal.")
                             default_cfg_obj = StrategyConfig(name=name)

                        if default_cfg_obj.input_dim is None:
                            default_cfg_obj.input_dim = default_strategy_input_dim
                        elif default_cfg_obj.input_dim != default_strategy_input_dim:
                            self.logger.info(f"Dynamically loaded strategy '{name}' default_config.input_dim ({default_cfg_obj.input_dim}) "
                                                f"differs from layer's default_strategy_input_dim ({default_strategy_input_dim}). Using strategy's default.")

                       # 傳遞 raw_feature_dim 和 timesteps_history 給策略
                       instance = strategy_class_from_registry(
                           config=default_cfg_obj,
                           raw_feature_dim=raw_feature_dim,
                           timesteps_history=timesteps_history
                       )
                       self.strategies.append(instance)
                       self.strategy_names.append(name)
                       processed_strategy_names.add(name)
                       self.logger.info(f"Successfully loaded and added strategy '{name}' from registry with input_dim {default_cfg_obj.input_dim}.")
                   except Exception as e_dyn_load:
                       self.logger.error(f"Error dynamically loading strategy '{name}' from registry: {e_dyn_load}", exc_info=True)

        self.num_actual_strategies = len(self.strategies)
        if self.num_actual_strategies > 0:
            self.logger.info(f"Total strategies initialized: {self.num_actual_strategies}. Names: {self.strategy_names}")
            if self.target_num_strategies < self.num_actual_strategies:
                self.logger.info(f"Actual number of strategies ({self.num_actual_strategies}) exceeds target_num_strategies ({self.target_num_strategies}). Updating target to actual.")
                self.target_num_strategies = self.num_actual_strategies
        else:
            self.logger.warning("No strategies were loaded after all initialization attempts.")

    def forward(self,
                asset_features_batch: torch.Tensor,
                market_state_features: Optional[torch.Tensor] = None,
                current_positions_batch: Optional[torch.Tensor] = None,
                timestamps: Optional[List[pd.Timestamp]] = None,
                external_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Main forward pass for the StrategyPoolManager layer.
        Includes adaptive weighting if enabled and not overridden by external_weights.

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
        else:
            current_logits: Optional[torch.Tensor] = None

            if self.adaptive_bias_weights is not None:
                current_logits = self.adaptive_bias_weights.unsqueeze(0).expand(batch_size, -1)
                self.logger.debug("Initialized logits with adaptive bias weights.")

            if self.attention_network is not None and market_state_features is not None:
                if market_state_features.shape[0] != batch_size or market_state_features.shape[1] != self.input_dim:
                    self.logger.warning(
                        f"market_state_features shape mismatch. Expected ({batch_size}, {self.input_dim}), "
                        f"got {market_state_features.shape}. Not using attention for this batch."
                    )
                else:
                    target_device = market_state_features.device
                    try:
                        current_attn_device = next(self.attention_network.parameters()).device
                        if current_attn_device != target_device:
                            self.logger.info(f"Moving attention_network from {current_attn_device} to {target_device}.")
                            self.attention_network.to(target_device)
                    except StopIteration:
                        self.logger.warning("attention_network has no parameters. Cannot verify/move device.")

                    attention_logits = self.attention_network(market_state_features)
                    self.last_attention_logits = attention_logits.detach().cpu()
                    if current_logits is not None:
                        current_logits = current_logits + attention_logits
                        self.logger.debug("Added attention logits to adaptive bias weights.")
                    else:
                        current_logits = attention_logits
                        self.logger.debug("Using attention logits as base for strategy weights.")

            if current_logits is not None:
                if self.use_gumbel_softmax:
                    if self.training:
                        strategy_weights = gumbel_softmax(current_logits, tau=self.temperature, hard=False, dim=1)
                    else:
                        strategy_weights = F.softmax(current_logits / self.temperature, dim=1)
                else:
                    strategy_weights = F.softmax(current_logits / self.temperature, dim=1)
                self.logger.debug(f"Calculated strategy weights using {'Gumbel Softmax' if self.use_gumbel_softmax else 'Softmax'}.")
            else:
                self.logger.warning("No external, adaptive, or attention-based weights. Using uniform weights.")
                strategy_weights = torch.ones(batch_size, self.num_actual_strategies, device=asset_features_batch.device) / self.num_actual_strategies

        strategy_weights = self.dropout(strategy_weights)

        # 2. Get signals from each strategy for each asset
        all_strategy_signals = torch.zeros(batch_size, num_assets, self.num_actual_strategies, device=asset_features_batch.device)

        for asset_idx in range(num_assets):
            single_asset_features = asset_features_batch[:, asset_idx, :, :]

            current_pos_for_asset = None
            if current_positions_batch is not None:
                if current_positions_batch.ndim == 3 and current_positions_batch.shape[1] == num_assets:
                    current_pos_for_asset = current_positions_batch[:, asset_idx, :]
                elif current_positions_batch.ndim == 2 and current_positions_batch.shape[0] == batch_size and num_assets == 1:
                     current_pos_for_asset = current_positions_batch
                else:
                    self.logger.warning(f"current_positions_batch shape {current_positions_batch.shape} not as expected for asset slicing. Skipping for asset {asset_idx}.")

            for i, strategy_module in enumerate(self.strategies):
                current_timestamp_for_strategy = timestamps[0] if timestamps and len(timestamps) > 0 else None

                try:
                    signal_tensor = strategy_module.forward(
                        single_asset_features,
                        current_positions=current_pos_for_asset,
                        timestamp=current_timestamp_for_strategy
                    )
                    if signal_tensor.shape != (batch_size, 1, 1):
                        self.logger.warning(f"Strategy {strategy_module.config.name} for asset {asset_idx} produced signal of shape {signal_tensor.shape}, expected ({batch_size}, 1, 1). Reshaping/ignoring.")
                        if signal_tensor.numel() == batch_size:
                            signal_tensor = signal_tensor.view(batch_size, 1, 1)
                        else:
                            signal_tensor = torch.zeros(batch_size, 1, 1, device=asset_features_batch.device)

                    all_strategy_signals[:, asset_idx, i] = signal_tensor.squeeze(-1).squeeze(-1)

                except Exception as e:
                    self.logger.error(f"Error in strategy {strategy_module.config.name} forward for asset {asset_idx}: {e}", exc_info=True)
                    all_strategy_signals[:, asset_idx, i] = 0.0


        # 3. Combine signals using weights
        expanded_weights = strategy_weights.unsqueeze(1)
        combined_signals = torch.sum(all_strategy_signals * expanded_weights, dim=2)
        final_actions = combined_signals.unsqueeze(-1)

        self.last_strategy_weights = strategy_weights.detach().cpu() if strategy_weights is not None else None
        self.last_all_strategy_signals = all_strategy_signals.detach().cpu() if all_strategy_signals is not None else None
        self.last_combined_signals = combined_signals.detach().cpu() if combined_signals is not None else None
        self.last_market_state_features = market_state_features.detach().cpu() if market_state_features is not None else None

        self.logger.debug(f"ESS Forward: Input asset_features_batch shape: {asset_features_batch.shape}")
        self.logger.debug(f"ESS Forward: All strategy signals shape: {all_strategy_signals.shape}")
        self.logger.debug(f"ESS Forward: Final actions shape: {final_actions.shape}")

        return final_actions

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing the last computed diagnostic values.
        """
        return {
            "strategy_names": self.strategy_names,
            "last_strategy_weights": self.last_strategy_weights,
            "last_all_strategy_signals": self.last_all_strategy_signals,
            "last_combined_signals": self.last_combined_signals,
            "last_attention_logits": self.last_attention_logits,
            "last_market_state_features": self.last_market_state_features,
            "temperature": self.temperature.item() if self.temperature is not None else None,
            "adaptive_bias_weights": self.adaptive_bias_weights.detach().cpu() if self.adaptive_bias_weights is not None else None,
            "strategy_performance_ema": self.strategy_performance_ema.detach().cpu() if self.strategy_performance_ema is not None else None,
        }

    def update_adaptive_weights(self, per_strategy_rewards: torch.Tensor):
        """
        Updates the adaptive components (performance EMA and bias weights) based on recent strategy performance.

        Args:
            per_strategy_rewards (torch.Tensor): A tensor representing the performance of each strategy.
                                                 Expected shape: (num_actual_strategies,).
                                                 These rewards should be on the same device as the model.
        """
        if self.adaptive_bias_weights is None or self.strategy_performance_ema is None or self.num_actual_strategies == 0:
            self.logger.warning("Adaptive components not initialized or no strategies. Skipping weight adaptation.")
            return

        if not isinstance(per_strategy_rewards, torch.Tensor):
            self.logger.error(f"per_strategy_rewards must be a torch.Tensor. Got {type(per_strategy_rewards)}.")
            return

        expected_shape = (self.num_actual_strategies,)
        if per_strategy_rewards.shape != expected_shape:
            self.logger.error(f"per_strategy_rewards shape mismatch. Expected {expected_shape}, got {per_strategy_rewards.shape}.")
            return

        target_device = self.adaptive_bias_weights.device
        if per_strategy_rewards.device != target_device:
            self.logger.info(f"Moving per_strategy_rewards from {per_strategy_rewards.device} to {target_device}.")
            per_strategy_rewards = per_strategy_rewards.to(target_device)

        self.strategy_performance_ema.data = (1 - self.performance_ema_alpha) * self.strategy_performance_ema.data + \
                                             self.performance_ema_alpha * per_strategy_rewards
        self.logger.debug(f"Updated strategy performance EMA: {self.strategy_performance_ema.data}")

        performance_deviation = self.strategy_performance_ema.data - self.strategy_performance_ema.data.mean()
        self.adaptive_bias_weights.data += self.adaptive_learning_rate * performance_deviation

        self.logger.info(f"Adaptive bias weights updated. Current values: {self.adaptive_bias_weights.data}")
        self.logger.debug(f"Performance EMA used for update: {self.strategy_performance_ema.data}")
        self.logger.debug(f"Performance deviation from mean: {performance_deviation}")

class DynamicStrategyGenerator:
    def __init__(self,
                 optimizer_config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None,
                 ):
        self.logger = logger if logger else logging.getLogger(f"{__name__}.DynamicStrategyGenerator")
        self.optimizer_config = optimizer_config if optimizer_config else {}
        self.genetic_optimizer: Optional[GeneticOptimizer] = None
        self.nas_optimizer: Optional[NeuralArchitectureSearch] = None

    def generate_new_strategy(self,
                              strategy_class: Type[BaseStrategy],
                              fitness_function: Optional[Callable] = None,
                              initial_parameters: Optional[Dict[str, Any]] = None,
                              strategy_config_override: Optional[StrategyConfig] = None,
                              market_data_for_ga: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
                              current_context: Optional[Dict[str, Any]] = None,
                              **kwargs) -> Optional[BaseStrategy]:
        if not strategy_class or not issubclass(strategy_class, BaseStrategy):
            self.logger.error(f"Invalid strategy_class provided: {strategy_class}. It must be a subclass of BaseStrategy.")
            return None

        if current_context is not None and not isinstance(current_context, dict):
            self.logger.error(f"Invalid current_context type: {type(current_context)}. Expected Dict or None.")
            return None

        self.logger.info(f"Attempting to generate strategy instance for: {strategy_class.__name__}")

        call_specific_optimizer_config = self.optimizer_config.copy() if self.optimizer_config else {}
        optimizer_type_override = kwargs.get('optimizer_type_override')
        optimizer_settings_override = kwargs.get('optimizer_settings_override')

        if optimizer_type_override:
            call_specific_optimizer_config["name"] = optimizer_type_override
            if optimizer_settings_override is not None:
                call_specific_optimizer_config["settings"] = optimizer_settings_override
            elif "settings" not in call_specific_optimizer_config:
                 call_specific_optimizer_config["settings"] = {}
        elif optimizer_settings_override is not None and call_specific_optimizer_config:
            call_specific_optimizer_config["settings"] = optimizer_settings_override

        final_config: StrategyConfig
        base_default_config = strategy_class.default_config()
        if not isinstance(base_default_config, StrategyConfig):
            self.logger.warning(
                f"{strategy_class.__name__}.default_config() did not return StrategyConfig. Using minimal."
            )
            base_default_config = StrategyConfig(name=strategy_class.__name__)

        if strategy_config_override:
            final_config = StrategyConfig.merge_configs(base_default_config, strategy_config_override)
            if strategy_config_override.default_params is not None:
                 final_config.default_params = strategy_config_override.default_params.copy()
        else:
            final_config = base_default_config

        if final_config.default_params is None:
            final_config.default_params = {}


        current_best_params = final_config.default_params.copy()

        if initial_parameters is not None:
            current_best_params.update(initial_parameters)
            self.logger.info(f"Updated current_best_params with initial_parameters before optimization: {current_best_params}")

        use_optimizer = bool(call_specific_optimizer_config and call_specific_optimizer_config.get("name") and fitness_function)
        optimized_params_from_optimizer: Optional[Dict[str, Any]] = None

        if use_optimizer:
            optimizer_name = call_specific_optimizer_config.get("name")
            optimizer_settings = call_specific_optimizer_config.get("settings", {})

            def _fitness_wrapper_for_optimizer(params_to_evaluate: Dict[str, Any], context_for_optimizer_fitness: Dict[str, Any]) -> float:
                if not fitness_function:
                    self.logger.error("Fitness function is None inside optimizer wrapper.")
                    return -float('inf')

                temp_eval_config = copy.deepcopy(final_config)

                try:
                    temp_strategy_instance = strategy_class(config=temp_eval_config, params=params_to_evaluate, logger=self.logger)
                except Exception as e_init:
                    self.logger.error(f"Error instantiating {strategy_class.__name__} in fitness wrapper with params {params_to_evaluate}: {e_init}", exc_info=True)
                    return -float('inf')

                _market_data = context_for_optimizer_fitness.get('market_data')
                _portfolio_context = context_for_optimizer_fitness.get('portfolio_context')

                try:
                    return fitness_function(temp_strategy_instance, _portfolio_context, _market_data, params_to_evaluate)
                except Exception as e_fit:
                    self.logger.error(f"Error in provided fitness_function for {strategy_class.__name__} with params {params_to_evaluate}: {e_fit}", exc_info=True)
                    return -float('inf')

            optimizer_context_for_fitness = {
                'market_data': market_data_for_ga,
                'portfolio_context': current_context
            }

            if optimizer_name == "GeneticOptimizer":
                param_space_ga = strategy_class.get_parameter_space(optimizer_type="genetic")
                if not param_space_ga:
                    self.logger.warning(f"GeneticOptimizer selected, but {strategy_class.__name__} has no GA parameter space. Skipping optimization.")
                else:
                    try:
                        self.genetic_optimizer = GeneticOptimizer(
                            fitness_function=_fitness_wrapper_for_optimizer,
                            param_space=param_space_ga,
                            base_config=optimizer_context_for_fitness,
                            logger=self.logger,
                            ga_settings=optimizer_settings
                        )
                        self.logger.info(f"Running GeneticOptimizer for {strategy_class.__name__}.")
                        optimized_params_from_optimizer, best_fitness = self.genetic_optimizer.run_optimizer(current_context=optimizer_context_for_fitness)

                        if optimized_params_from_optimizer:
                            self.logger.info(f"Genetic Optimizer found solution with fitness {best_fitness:.4f}. Params: {optimized_params_from_optimizer}")
                        else:
                            self.logger.warning(f"Genetic Optimizer for {strategy_class.get_strategy_name()} did not return a solution. Best fitness: {best_fitness:.4f}.")
                    except Exception as e_ga:
                        self.logger.error(f"Error during genetic optimization for {strategy_class.get_strategy_name()}: {e_ga}", exc_info=True)
                    finally:
                        self.genetic_optimizer = None

            elif optimizer_name == "NeuralArchitectureSearch":
                if not issubclass(strategy_class, torch.nn.Module):
                    self.logger.error(f"NeuralArchitectureSearch requires strategy_class to be an nn.Module. {strategy_class.__name__} is not. Skipping NAS.")
                else:
                    param_space_nas = strategy_class.get_parameter_space_for_nas()
                    if not param_space_nas:
                         self.logger.warning(f"NeuralArchitectureSearch selected, but {strategy_class.__name__} has no NAS parameter space. Skipping optimization.")
                    else:
                        try:
                            self.nas_optimizer = NeuralArchitectureSearch(
                                strategy_class=strategy_class,
                                search_space=param_space_nas,
                                fitness_function_callback=_fitness_wrapper_for_optimizer,
                                nas_settings=optimizer_settings,
                                base_config_for_fitness=optimizer_context_for_fitness,
                                logger=self.logger
                            )
                            self.logger.info(f"Running NeuralArchitectureSearch for {strategy_class.__name__}.")
                            best_nas_params = self.nas_optimizer.run_optimizer()

                            if best_nas_params:
                                self.logger.info(f"NeuralArchitectureSearch found best params for {strategy_class.__name__}: {best_nas_params}")
                                optimized_params_from_optimizer = best_nas_params
                            else:
                                self.logger.info(f"NeuralArchitectureSearch did not find improved parameters for {strategy_class.__name__}.")
                        except Exception as e_nas:
                            self.logger.error(f"Error during Neural Architecture Search for {strategy_class.__name__}: {e_nas}", exc_info=True)
                        finally:
                            self.nas_optimizer = None
            else:
                self.logger.warning(f"Unknown optimizer: {optimizer_name}. Skipping optimization.")

        if optimized_params_from_optimizer is not None:
            current_best_params.update(optimized_params_from_optimizer)
            self.logger.info(f"Applied optimized parameters. Final params for {strategy_class.__name__}: {current_best_params}")
        elif initial_parameters is not None and not use_optimizer:
             self.logger.info(f"Using initial parameters (no optimizer run) for {strategy_class.__name__}: {current_best_params}")
        else:
            self.logger.info(f"Using default config parameters for {strategy_class.__name__}: {current_best_params}")

        try:
            strategy_instance = strategy_class(config=final_config, params=current_best_params, logger=self.logger)
            self.logger.info(f"Successfully generated strategy instance of {strategy_class.__name__} with final params: {strategy_instance.params}")
            return strategy_instance
        except Exception as e:
            self.logger.error(f"Failed to instantiate strategy {strategy_class.__name__} with final_config and params: {e}", exc_info=True)
            return None

    def _prepare_optimizer_config(self, base_config: Optional[Dict], strategy_class: Type[BaseStrategy],
                                  strategy_params: Optional[Dict] = None, optimizer_type: str = "GA") -> Dict:
        self.logger.debug(f"Preparing optimizer config. Base: {base_config}, Strategy: {strategy_class.__name__}, Params: {strategy_params}, OptType: {optimizer_type}")

        final_config = copy.deepcopy(base_config) if base_config else {}
        if strategy_params is None:
            strategy_params = {}

        strategy_class_param_space_list = strategy_class.get_parameter_space(optimizer_type=optimizer_type)
        if strategy_class_param_space_list is None:
            strategy_class_param_space_list = []

        strategy_class_param_space_map = {
            p['name']: p for p in strategy_class_param_space_list if isinstance(p, dict) and 'name' in p
        }

        original_base_param_space_list = base_config.get('parameter_space', []) if base_config else []
        original_base_param_space_map = {
            p['name']: p for p in original_base_param_space_list if isinstance(p, dict) and 'name' in p
        }

        final_config['parameter_space'] = copy.deepcopy(strategy_class_param_space_list)

        for name, strat_def in strategy_class_param_space_map.items():
            if name in original_base_param_space_map:
                orig_def = original_base_param_space_map[name]
                diffs = []
                for field in ['low', 'high', 'step', 'type']:
                    strat_val = strat_def.get(field)
                    orig_val = orig_def.get(field)
                    if strat_val != orig_val:
                        diffs.append(f"{field}: strategy '{strat_val}' vs base '{orig_val}'")
                if diffs:
                    self.logger.warning(
                        f"Parameter space definition for '{name}' differs between strategy class and base optimizer_config. "
                        f"Using strategy class definition. Differences: {'; '.join(diffs)}"
                    )
            else:
                self.logger.info(f"Parameter '{name}' defined by strategy class, not in base optimizer_config's parameter_space.")

        for name in original_base_param_space_map:
            if name not in strategy_class_param_space_map:
                self.logger.info(f"Parameter '{name}' was in base optimizer_config's parameter_space but not defined by strategy class for optimizer type '{optimizer_type}'. It will not be part of the current 'parameter_space'.")

        final_config.setdefault('initial_guess', {})
        final_config.setdefault('fixed_parameters', {})

        base_initial_guesses = base_config.get('initial_guess', {}) if base_config else {}
        base_fixed_params = base_config.get('fixed_parameters', {}) if base_config else {}

        processed_params_for_logging = {}

        for name, value in base_initial_guesses.items():
            if name in strategy_class_param_space_map:
                final_config['initial_guess'][name] = value
                processed_params_for_logging[name] = {'value': value, 'source': "base_config's initial_guess"}
            else:
                final_config['fixed_parameters'][name] = value
                processed_params_for_logging[name] = {'value': value, 'source': "base_config as fixed_param"}

        for name, value in base_fixed_params.items():
            if name in strategy_class_param_space_map and name in final_config['initial_guess']:
                del final_config['initial_guess'][name]
            final_config['fixed_parameters'][name] = value
            processed_params_for_logging[name] = {'value': value, 'source': "base_config's fixed_parameters"}


        for name, value_from_strat_params in strategy_params.items():
            original_value_info = processed_params_for_logging.get(name)
            log_msg_prefix = f"Parameter '{name}' from strategy_params (value: {value_from_strat_params})"

            if name in strategy_class_param_space_map:
                if original_value_info and original_value_info['value'] != value_from_strat_params:
                    self.logger.info(f"{log_msg_prefix} takes precedence over {original_value_info['source']} (value: {original_value_info['value']})")
                elif not original_value_info:
                     self.logger.info(f"{log_msg_prefix} set as initial guess.")

                final_config['initial_guess'][name] = value_from_strat_params
                if name in final_config['fixed_parameters']:
                    del final_config['fixed_parameters'][name]
                processed_params_for_logging[name] = {'value': value_from_strat_params, 'source': 'strategy_params (as initial_guess)'}
            else:
                if original_value_info and original_value_info['value'] != value_from_strat_params:
                     self.logger.info(f"{log_msg_prefix} (fixed) takes precedence over {original_value_info['source']} (value: {original_value_info['value']})")
                elif not original_value_info:
                    self.logger.info(f"{log_msg_prefix} set as fixed parameter.")

                final_config['fixed_parameters'][name] = value_from_strat_params
                if name in final_config['initial_guess']:
                    del final_config['initial_guess'][name]
                processed_params_for_logging[name] = {'value': value_from_strat_params, 'source': 'strategy_params (as fixed_param)'}

        for name in list(processed_params_for_logging.keys()):
            info = processed_params_for_logging[name]
            if name in final_config['initial_guess'] and final_config['initial_guess'][name] == info['value']:
                 self.logger.info(f"Final initial guess for '{name}' for optimizer: {info['value']}")
            elif name in final_config['fixed_parameters'] and final_config['fixed_parameters'][name] == info['value']:
                 self.logger.info(f"Final fixed value for '{name}': {info['value']}")

        final_config['parameters'] = final_config.get('fixed_parameters', {}).copy()

        if not final_config.get('initial_guess'):
            final_config.pop('initial_guess', None)
        if not final_config.get('fixed_parameters'):
            final_config.pop('fixed_parameters', None)
            if not final_config.get('parameters'):
                 final_config.pop('parameters', None)
        elif not final_config.get('parameters'):
            final_config.pop('parameters', None)


        self.logger.debug(f"Final prepared optimizer config: {final_config}")
        return final_config