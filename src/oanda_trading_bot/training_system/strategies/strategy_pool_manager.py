import logging
from typing import Dict, List, Optional, Any, Union, Type, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
import pandas as pd

from oanda_trading_bot.training_system.agent.strategies import STRATEGY_REGISTRY
from oanda_trading_bot.training_system.agent.strategies.base_strategy import StrategyConfig, BaseStrategy


class StrategyPoolManager(nn.Module):
    """
    Manage a pool of strategies and produce combined signals via learnable + online (bandit/hedge) weights.
    - Learnable prior: attention over market_state_features -> logits
    - Adaptive bias: EMA-driven bias logits (performance based)
    - Bandit/Hedge: exponentiated-gradient style online combiner (log-weights)
    """

    def __init__(
        self,
        input_dim: int,
        num_strategies: int,
        strategy_configs: Optional[List[Union[Dict[str, Any], StrategyConfig]]] = None,
        explicit_strategies: Optional[List[Type[BaseStrategy]]] = None,
        strategy_config_file_path: Optional[str] = None,
        dropout_rate: float = 0.1,
        initial_temperature: float = 1.0,
        use_gumbel_softmax: bool = True,
        strategy_input_dim: int = 64,
        dynamic_loading_enabled: bool = True,
        adaptive_learning_rate: float = 0.01,
        performance_ema_alpha: float = 0.1,
        raw_feature_dim: int = 0,
        timesteps_history: int = 0,
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.input_dim = input_dim
        self.target_num_strategies = int(max(0, num_strategies))
        self.strategy_input_dim = strategy_input_dim
        self.dropout_rate = dropout_rate
        self.initial_temperature = float(initial_temperature)
        self.use_gumbel_softmax = bool(use_gumbel_softmax)
        self.dynamic_loading_enabled = bool(dynamic_loading_enabled)
        self.adaptive_learning_rate = float(adaptive_learning_rate)
        self.performance_ema_alpha = float(performance_ema_alpha)
        self.raw_feature_dim = raw_feature_dim
        self.timesteps_history = timesteps_history

        # Load strategy configs from file if provided
        loaded_cfgs: List[Union[Dict[str, Any], StrategyConfig]] = []
        if strategy_config_file_path:
            try:
                import json
                with open(strategy_config_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and isinstance(data.get('strategies'), list):
                        loaded_cfgs.extend(data['strategies'])
                        if isinstance(data.get('global_strategy_input_dim'), int):
                            self.strategy_input_dim = int(data['global_strategy_input_dim'])
                        self.logger.info(f"Loaded {len(data['strategies'])} strategies from {strategy_config_file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load strategy config file {strategy_config_file_path}: {e}")

        # Merge input strategy configurations
        merged_cfgs: List[Union[Dict[str, Any], StrategyConfig, Type[BaseStrategy]]] = []
        merged_cfgs.extend(loaded_cfgs)
        if strategy_configs:
            merged_cfgs.extend(strategy_configs)

        # Initialize strategies
        self.strategies = nn.ModuleList()
        self.strategy_names: List[str] = []
        self._initialize_strategies(
            layer_input_dim=self.input_dim,
            processed_strategy_configs=merged_cfgs,
            explicit_strategies=explicit_strategies or [],
            default_strategy_input_dim=self.strategy_input_dim,
            dynamic_loading_enabled=self.dynamic_loading_enabled,
            raw_feature_dim=self.raw_feature_dim,
            timesteps_history=self.timesteps_history,
        )
        self.num_actual_strategies = len(self.strategies)

        # Weighting modules
        if self.num_actual_strategies > 0:
            self.attention_network = nn.Linear(self.input_dim, self.num_actual_strategies)
            self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
            # Adaptive bias + EMA for strategy performance
            self.adaptive_bias_weights = nn.Parameter(torch.zeros(self.num_actual_strategies), requires_grad=False)
            self.strategy_performance_ema = nn.Parameter(torch.zeros(self.num_actual_strategies), requires_grad=False)
            # Temperature
            self.temperature = nn.Parameter(torch.tensor(self.initial_temperature, dtype=torch.float32), requires_grad=False)
            # Bandit (Hedge/Exp) log-weights
            self.bandit_eta: float = 0.05
            self.bandit_log_weights = nn.Parameter(torch.zeros(self.num_actual_strategies), requires_grad=False)
        else:
            self.attention_network = None
            self.dropout = nn.Identity()
            self.adaptive_bias_weights = None
            self.strategy_performance_ema = None
            self.temperature = nn.Parameter(torch.tensor(self.initial_temperature, dtype=torch.float32), requires_grad=False)
            self.bandit_log_weights = None

        # Diagnostics
        self.last_strategy_weights: Optional[torch.Tensor] = None
        self.last_all_strategy_signals: Optional[torch.Tensor] = None
        self.last_combined_signals: Optional[torch.Tensor] = None
        self.last_attention_logits: Optional[torch.Tensor] = None
        self.last_market_state_features: Optional[torch.Tensor] = None

        # Regime-aware placeholders (to be filled later if needed)
        self.regime_weights: Dict[str, torch.Tensor] = {}
        self.current_regime: Optional[str] = None

    def _get_strategy_registry(self) -> Dict[str, Type[BaseStrategy]]:
        return dict(STRATEGY_REGISTRY)

    def _initialize_strategies(
        self,
        layer_input_dim: int,
        processed_strategy_configs: Optional[List[Union[Dict[str, Any], StrategyConfig, Type[BaseStrategy]]]],
        explicit_strategies: List[Type[BaseStrategy]],
        default_strategy_input_dim: int,
        dynamic_loading_enabled: bool,
        raw_feature_dim: int,
        timesteps_history: int,
    ) -> None:
        processed_names = set()
        registry = self._get_strategy_registry()

        # From configs
        if processed_strategy_configs:
            for item in processed_strategy_configs:
                strategy_class: Optional[Type[BaseStrategy]] = None
                final_cfg: Optional[StrategyConfig] = None
                if isinstance(item, type) and issubclass(item, BaseStrategy):
                    strategy_class = item
                    base_cfg = strategy_class.default_config()
                    final_cfg = base_cfg if isinstance(base_cfg, StrategyConfig) else StrategyConfig(name=strategy_class.__name__)
                elif isinstance(item, StrategyConfig):
                    name = item.name
                    if name in registry:
                        strategy_class = registry[name]
                        base_cfg = strategy_class.default_config()
                        base_cfg = base_cfg if isinstance(base_cfg, StrategyConfig) else StrategyConfig(name=name)
                        final_cfg = StrategyConfig.merge_configs(base_cfg, item)
                    else:
                        self.logger.warning(f"Config references unknown strategy '{name}', skipping.")
                elif isinstance(item, dict):
                    name = item.get('name')
                    if not name or name not in registry:
                        self.logger.warning(f"Dict config missing/unknown 'name': {item}")
                    else:
                        strategy_class = registry[name]
                        base_cfg = strategy_class.default_config()
                        base_cfg = base_cfg if isinstance(base_cfg, StrategyConfig) else StrategyConfig(name=name)
                        final_cfg = StrategyConfig.merge_configs(base_cfg, StrategyConfig(**item))

                if strategy_class is None or final_cfg is None:
                    continue
                if final_cfg.input_dim is None:
                    final_cfg.input_dim = default_strategy_input_dim
                self._instantiate_strategy(strategy_class, final_cfg, processed_names, raw_feature_dim, timesteps_history)

        # From explicit list
        for cls_ in explicit_strategies or []:
            name = cls_.__name__
            if name in processed_names:
                continue
            base_cfg = cls_.default_config()
            if not isinstance(base_cfg, StrategyConfig):
                base_cfg = StrategyConfig(name=name)
            if base_cfg.input_dim is None:
                base_cfg.input_dim = default_strategy_input_dim
            self._instantiate_strategy(cls_, base_cfg, processed_names, raw_feature_dim, timesteps_history)

        # Dynamic load remaining from registry if enabled
        if dynamic_loading_enabled:
            for name, cls_ in registry.items():
                if name in processed_names:
                    continue
                base_cfg = cls_.default_config()
                if not isinstance(base_cfg, StrategyConfig):
                    base_cfg = StrategyConfig(name=name)
                if base_cfg.input_dim is None:
                    base_cfg.input_dim = default_strategy_input_dim
                self._instantiate_strategy(cls_, base_cfg, processed_names, raw_feature_dim, timesteps_history)

    def _instantiate_strategy(
        self,
        strategy_class: Type[BaseStrategy],
        config: StrategyConfig,
        processed_names: set,
        raw_feature_dim: int,
        timesteps_history: int,
    ) -> None:
        try:
            instance = strategy_class(config=config, raw_feature_dim=raw_feature_dim, timesteps_history=timesteps_history)
            self.strategies.append(instance)
            self.strategy_names.append(config.name or strategy_class.__name__)
            processed_names.add(config.name or strategy_class.__name__)
            self.logger.info(f"Loaded strategy '{config.name}' with input_dim {config.input_dim}.")
        except Exception as e:
            self.logger.error(f"Failed to instantiate strategy {strategy_class.__name__}: {e}")

    def forward(
        self,
        asset_features_batch: torch.Tensor,
        market_state_features: Optional[torch.Tensor] = None,
        current_positions_batch: Optional[torch.Tensor] = None,
        timestamps: Optional[List[pd.Timestamp]] = None,
        external_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.num_actual_strategies == 0:
            b, n = asset_features_batch.shape[0], asset_features_batch.shape[1]
            return torch.zeros(b, n, 1, device=asset_features_batch.device)

        b, n, t, f = asset_features_batch.shape
        # 1) Determine weights
        if external_weights is not None and external_weights.shape == (b, self.num_actual_strategies):
            weights = F.softmax(external_weights / self.temperature, dim=1)
        else:
            logits = None
            if self.adaptive_bias_weights is not None:
                logits = self.adaptive_bias_weights.unsqueeze(0).expand(b, -1)
            if self.attention_network is not None and market_state_features is not None:
                if market_state_features.shape == (b, self.input_dim):
                    attn_logits = self.attention_network(market_state_features)
                    self.last_attention_logits = attn_logits.detach().cpu()
                    logits = attn_logits if logits is None else (logits + attn_logits)
                else:
                    self.logger.warning("market_state_features shape mismatch; skipping attention")
            if logits is not None:
                if self.use_gumbel_softmax and self.training:
                    weights = gumbel_softmax(logits, tau=self.temperature, hard=False, dim=1)
                else:
                    weights = F.softmax(logits / self.temperature, dim=1)
            else:
                weights = torch.ones(b, self.num_actual_strategies, device=asset_features_batch.device) / self.num_actual_strategies

        # Blend with bandit weights
        if self.bandit_log_weights is not None:
            bw = torch.softmax(self.bandit_log_weights, dim=0).to(weights.device)
            bw = bw[: self.num_actual_strategies]
            weights = weights * bw.unsqueeze(0)
            weights = weights / (weights.sum(dim=1, keepdim=True).clamp_min(1e-8))

        weights = self.dropout(weights)
        # Regime-aware blending (if regime set and weights exist)
        if self.current_regime and self.current_regime in self.regime_weights:
            rw = self.regime_weights[self.current_regime].to(weights.device)
            if rw.numel() >= self.num_actual_strategies:
                rw = rw[: self.num_actual_strategies]
                rw = torch.softmax(rw, dim=0)
                weights = weights * rw.unsqueeze(0)
                weights = weights / (weights.sum(dim=1, keepdim=True).clamp_min(1e-8))

        # 2) Collect strategy signals per asset
        all_signals = torch.zeros(b, n, self.num_actual_strategies, device=asset_features_batch.device)
        for asset_idx in range(n):
            single = asset_features_batch[:, asset_idx, :, :]
            cur_pos = None
            if current_positions_batch is not None:
                if current_positions_batch.ndim == 3 and current_positions_batch.shape[1] == n:
                    cur_pos = current_positions_batch[:, asset_idx, :]
                elif current_positions_batch.ndim == 2 and current_positions_batch.shape[0] == b and n == 1:
                    cur_pos = current_positions_batch
            for i, strat in enumerate(self.strategies):
                try:
                    ts = timestamps[0] if timestamps else None
                    sig = strat.forward(single, current_positions=cur_pos, timestamp=ts)
                    if sig.shape != (b, 1, 1):
                        if sig.numel() == b:
                            sig = sig.view(b, 1, 1)
                        else:
                            sig = torch.zeros(b, 1, 1, device=asset_features_batch.device)
                    all_signals[:, asset_idx, i] = sig.squeeze(-1).squeeze(-1)
                except Exception as e:
                    self.logger.error(f"Strategy {self.strategy_names[i]} failed for asset {asset_idx}: {e}")
                    all_signals[:, asset_idx, i] = 0.0

        combined = torch.sum(all_signals * weights.unsqueeze(1), dim=2)  # [B,N]\n        actions = combined.unsqueeze(-1)  # default output [B,N,1]

        # diagnostics
        self.last_strategy_weights = weights.detach().cpu()
        self.last_all_strategy_signals = all_signals.detach().cpu()
        self.last_combined_signals = combined.detach().cpu()
        self.last_market_state_features = market_state_features.detach().cpu() if market_state_features is not None else None

        return actions

    def update_adaptive_weights(self, per_strategy_rewards: torch.Tensor):
        """
        Update EMA and adaptive bias + bandit log-weights.
        per_strategy_rewards: shape (num_actual_strategies,)
        """
        if self.num_actual_strategies == 0:
            return
        if not isinstance(per_strategy_rewards, torch.Tensor):
            self.logger.error("per_strategy_rewards must be a tensor")
            return
        if per_strategy_rewards.shape != (self.num_actual_strategies,):
            self.logger.error(f"per_strategy_rewards shape mismatch: {per_strategy_rewards.shape}")
            return

        dev = self.adaptive_bias_weights.device if self.adaptive_bias_weights is not None else per_strategy_rewards.device
        r = per_strategy_rewards.to(dev)
        # EMA
        self.strategy_performance_ema.data = (1 - self.performance_ema_alpha) * self.strategy_performance_ema.data + self.performance_ema_alpha * r
        # Adaptive bias logits
        perf_dev = self.strategy_performance_ema.data - self.strategy_performance_ema.data.mean()
        self.adaptive_bias_weights.data += self.adaptive_learning_rate * perf_dev
        # Bandit (Hedge)
        if self.bandit_log_weights is not None:
            centered = self.strategy_performance_ema.data - self.strategy_performance_ema.data.mean()
            self.bandit_log_weights.data += self.bandit_eta * centered
            self.bandit_log_weights.data.clamp_(-20.0, 20.0)

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            'strategy_names': self.strategy_names,
            'last_strategy_weights': self.last_strategy_weights,
            'last_all_strategy_signals': self.last_all_strategy_signals,
            'last_combined_signals': self.last_combined_signals,
            'last_attention_logits': self.last_attention_logits,
            'last_market_state_features': self.last_market_state_features,
            'temperature': float(self.temperature.item()) if self.temperature is not None else None,
        }

    # Optional: expose a helper to map alpha -> portfolio weights via differentiable head
    def compute_portfolio_weights(self, alpha: torch.Tensor, vol_est: Optional[torch.Tensor] = None) -> torch.Tensor:
        try:
            from oanda_trading_bot.training_system.models.portfolio_head import DifferentiableMeanVarianceHead
        except Exception:
            raise RuntimeError("portfolio_head module not available")
        if not hasattr(self, '_portfolio_head'):
            self._portfolio_head = DifferentiableMeanVarianceHead(risk_aversion=1.0, scale=10.0).to(alpha.device)
        return self._portfolio_head(alpha, vol_est)

    # Regime control helpers
    def set_current_regime(self, name: Optional[str]):
        self.current_regime = name

    def set_regime_weights(self, name: str, weights: torch.Tensor):
        if not isinstance(weights, torch.Tensor):
            raise ValueError("weights must be a torch.Tensor")
        self.regime_weights[name] = weights.detach().clone()
