# src/agent/sac_policy.py
"""
定義自定義的 SAC 策略，它使用 AdvancedTransformerFeatureExtractor。
"""
from stable_baselines3.sac.policies import SACPolicy, Actor, ContinuousCritic as Critic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp, NatureCNN, CombinedExtractor, FlattenExtractor # Added BaseFeaturesExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution, SquashedDiagGaussianDistribution, Distribution
from src.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition
from src.agent.enhanced_feature_extractor import EnhancedTransformerFeatureExtractor
from src.common.config import (
    ENHANCED_MODEL_CONFIG_PATH,
    QUANTUM_STRATEGY_CONFIG_PATH,
    QUANTUM_STRATEGY_NUM_STRATEGIES,
    QUANTUM_STRATEGY_DROPOUT_RATE,
    QUANTUM_STRATEGY_INITIAL_TEMPERATURE,
    QUANTUM_STRATEGY_USE_GUMBEL_SOFTMAX,
    QUANTUM_ADAPTIVE_LR,
    QUANTUM_PERFORMANCE_EMA_ALPHA,
    DEVICE, # This is a global config, might differ from policy's device
    TRANSFORMER_OUTPUT_DIM_PER_SYMBOL, 
    MAX_SYMBOLS_ALLOWED
)
from src.common.logger_setup import logger
from src.agent.strategies import STRATEGY_REGISTRY, BaseStrategy
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic # ActorCriticPolicy is imported via SACPolicy
from stable_baselines3.common.type_aliases import Schedule, PyTorchObs, TensorDict
from stable_baselines3.common.utils import get_device
from typing import Any, Dict, List, Optional, Type, Union
import torch # Added for torch.device
import torch as th # Common alias in SB3
import torch.nn as nn
from gymnasium import spaces
import logging
import inspect # Add inspect if not already there

logger = logging.getLogger(__name__) # Ensure logger is defined

# Ensure all necessary imports are present
import torch
import torch as th
import torch.nn as nn
from gymnasium import spaces # Make sure spaces is imported

from stable_baselines3.common.policies import BasePolicy, ContinuousCritic # ContinuousCritic for QuantumCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    create_mlp,
    FlattenExtractor,
    NatureCNN,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.utils import get_device
from stable_baselines3.sac.policies import Actor, SACPolicy # Import SACPolicy and Actor

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import logging # Ensure logging is imported
logger = logging.getLogger(__name__)

class QuantumActor(Actor):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int], 
        features_extractor: BaseFeaturesExtractor, # 實例化的特徵提取器
        features_dim: int, # 特徵提取後的維度
        activation_fn: Type[nn.Module] = nn.ReLU, # 激活函數
        # --- SB3 Actor constructor arguments ---
        log_std_init: float = -3, # SAC Actor 的 log_std 初始化值
        use_sde: bool = False, # 是否使用狀態依賴探索 (SDE)
        # --- 其餘自定義參數 ---
        sde_net_arch: Optional[List[int]] = None, # SDE 特徵網絡的架構
        use_expln: bool = False, # 是否在 SDE 中使用指數線性單元 (expln)
        clip_mean: float = 2.0, # SDE 中裁剪平均值的範圍
        normalize_images: bool = True, # 是否標準化圖像觀測 (來自 BasePolicy)
        use_ess_layer: bool = False, # 是否使用增強策略疊加層 (ESS)
        ess_config: Optional[Dict[str, Any]] = None, # ESS 層的配置
        use_adaptive_action_noise: bool = False, # 是否使用自適應動作噪聲
        adaptive_noise_config: Optional[Dict[str, Any]] = None, # 自適應噪聲配置
        device: Union[torch.device, str] = "auto", # 設備 (cpu/cuda)
        dropout_rate: float = 0.0, # Actor 網絡的 dropout 率
        use_layer_norm_actor: bool = False, # 是否在 Actor 網絡中使用 LayerNorm
        use_quantum_layer: bool = False, # 是否使用量子啟發層
        quantum_config: Optional[Dict[str, Any]] = None, # 量子層的配置
    ):
        # 僅傳遞 Actor 支援的參數給 super
        super(QuantumActor, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            log_std_init=log_std_init,
            use_sde=use_sde,
        )
        # 其餘自定義參數本地保存
        self.custom_activation_fn = activation_fn
        self.sde_net_arch = sde_net_arch
        self.use_expln = use_expln
        self.clip_mean = clip_mean
        self.normalize_images = normalize_images
        self.use_ess_layer = use_ess_layer
        self.ess_config = ess_config if ess_config else {}
        self.use_adaptive_action_noise = use_adaptive_action_noise
        self.adaptive_noise_config = adaptive_noise_config if adaptive_noise_config else {}
        self.device_val = get_device(device)
        self.dropout_rate = dropout_rate
        self.use_layer_norm_actor = use_layer_norm_actor
        self.use_quantum_layer = use_quantum_layer
        self.quantum_config = quantum_config if quantum_config else {}

        # --- 實務設計：自動建立 policy_head_net ---
        # features_dim -> net_arch -> latent_dim (通常 net_arch[-1])
        # 若 net_arch 為空，預設一層 [features_dim -> features_dim]
        mlp_arch = net_arch if net_arch else [features_dim]
        # 只傳遞 SB3 支援參數，不傳 dropout_rate
        mlp_layers = create_mlp(
            input_dim=features_dim,
            net_arch=mlp_arch,
            activation_fn=activation_fn,
            output_dim=mlp_arch[-1] if len(mlp_arch) > 0 else features_dim
        )
        # 若需 dropout，可於每層後手動插入
        if dropout_rate > 0.0:
            layers_with_dropout = []
            for layer in mlp_layers:
                layers_with_dropout.append(layer)
                if isinstance(layer, nn.Linear):
                    layers_with_dropout.append(nn.Dropout(dropout_rate))
            self.policy_head_net = nn.Sequential(*layers_with_dropout)
        else:
            self.policy_head_net = nn.Sequential(*mlp_layers)
        # 若未來需插入量子策略層，可在此處包裝 self.policy_head_net
        # if self.use_quantum_layer:
        #     self.policy_head_net = QuantumStrategyLayerWrapper(self.policy_head_net, self.quantum_config)

    # forward 方法: 定義如何從觀測中計算動作分佈的參數
    # 父類 Actor 的 forward 方法返回 (actions, mean_actions, log_std) 或類似 SDE 的輸出
    # 我們需要重寫它以使用 self.policy_head_net
    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        # 從觀測中提取特徵
        # self.features_extractor 是在 BasePolicy 中設置的，由父類 Actor 繼承
        features = self.extract_features(obs, self.features_extractor) # 明確傳遞特徵提取器

        # 通過我們自定義的策略頭部網絡處理特徵
        latent_pi = self.policy_head_net(features)

        # 使用父類 Actor 已創建的 self.mu 和 self.log_std (或 SDE 相關層)
        # 這些層現在以 latent_pi (self.policy_head_net 的輸出) 作為輸入
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            # 在 SDE 模式下，self.log_std (由父類 Actor 創建) 實際上是一個網絡，
            # 它從 SDE 特徵網絡 (sde_features_net) 的輸出計算 SDE 分佈所需的特徵。
            # sde_features_net 的輸入通常是 latent_pi 或原始 features。
            # SB3 Actor 的 sde_features_net (如果 sde_net_arch 不為空) 輸入是 latent_pi。
            # 如果 sde_net_arch 為空，則 SDE 特徵是 latent_pi 本身。
            sde_features = self.log_std(latent_pi) # self.log_std 在 SDE 模式下是 SDE 特徵生成網絡
            # 返回均值動作和 SDE 特徵 (用於 StateDependentNoiseDistribution)
            return mean_actions, sde_features
        else:
            # 非 SDE 模式下，self.log_std 是直接輸出 log 標準差的層
            log_std = self.log_std(latent_pi)
            # 返回均值動作和 log 標準差
            return mean_actions, log_std

    # _get_constructor_parameters 方法: 用於保存和加載模型
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters() # 從父類 Actor 獲取參數

        # 添加 QuantumActor 特有的構造函數參數
        # 父類 Actor 的 _get_constructor_parameters 已經包含了 observation_space, action_space,
        # net_arch (用於 latent_pi), features_extractor_class, features_extractor_kwargs,
        # features_dim, log_std_init, use_sde, sde_net_arch, use_expln, clip_mean, normalize_images.
        
        # 我們需要添加 QuantumActor __init__ 簽名中額外定義的參數
        data.update(
            dict(
                activation_fn=self.custom_activation_fn, # 我們存儲的自定義激活函數
                # use_sde 已經由父類處理
                # sde_net_arch 已經由父類處理 (如果它用於父類的 SDE 網絡)
                # log_std_init, use_expln, clip_mean, normalize_images 也由父類處理
                
                # QuantumActor 特有參數
                use_ess_layer=self.use_ess_layer,
                ess_config=self.ess_config,
                use_adaptive_action_noise=self.use_adaptive_action_noise,
                adaptive_noise_config=self.adaptive_noise_config,
                device=self.device_val, # 存儲實際使用的設備
                dropout_rate=self.dropout_rate,
                use_layer_norm_actor=self.use_layer_norm_actor,
                use_quantum_layer=self.use_quantum_layer,
                quantum_config=self.quantum_config,
            )
        )
        # 確保 net_arch 是 actor head 的 net_arch (已經由父類處理)
        # 確保 features_extractor 和 features_dim 也被正確處理 (父類處理)
        return data

# Define QuantumCritic (Placeholder - needs full implementation similar to SB3 ContinuousCritic)
class QuantumCritic(ContinuousCritic):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box, # In SB3 SAC, critic takes action_space
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True, # From BasePolicy args, passed to ContinuousCritic
        use_sde: bool = False, # Added, though critic might not directly use SDE itself
        use_ess_layer: bool = False,
        ess_config: Optional[Dict[str, Any]] = None,
        dropout_rate: float = 0.0,
        use_layer_norm_critic: bool = False,
        n_critics: int = 2, # From SACPolicy
        device: Union[torch.device, str] = "auto",
        share_features_extractor: bool = True, # From SACPolicy
    ):
        super().__init__(
            observation_space,
            action_space, # Pass action_space
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            normalize_images,
            share_features_extractor=share_features_extractor # Pass to super
        )
        self.use_ess_layer = use_ess_layer
        self.ess_config = ess_config if ess_config else {}
        self.dropout_rate = dropout_rate
        self.use_layer_norm_critic = use_layer_norm_critic
        self.n_critics_val = n_critics # Renamed to avoid conflict if super has n_critics
        self.device_val = get_device(device)

        # Critic network (q_networks) construction is handled by ContinuousCritic superclass.
        # It creates `self.q_networks` as a ModuleList of Sequential networks.
        # We can modify them here if needed, e.g., add dropout or layer norm if not handled by super.
        # Or, if ESS layer needs to be integrated into each Q-network.

        # Example of modifying q_networks if custom layers are needed and not handled by net_arch processing in super:
        # new_q_networks = nn.ModuleList()
        # for q_net in self.q_networks:
        #     # q_net is a Sequential network. We might rebuild it or add layers.
        #     # This requires knowing the structure of q_net from ContinuousCritic.
        #     # For simplicity, assume net_arch already incorporates these needs, or use hooks.
        #     pass 
        # self.q_networks = new_q_networks

        logger.info(f"QuantumCritic initialized. Number of Q networks: {len(self.q_networks) if self.q_networks else 0}")

    # Forward method is inherited from ContinuousCritic, which takes obs and actions
    # def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
    #    return super().forward(obs, actions)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                use_ess_layer=self.use_ess_layer,
                ess_config=self.ess_config,
                dropout_rate=self.dropout_rate,
                use_layer_norm_critic=self.use_layer_norm_critic,
                n_critics=self.n_critics_val, # Use stored value
                device=self.device_val,
            )
        )
        return data

class CustomSACPolicy(SACPolicy):
    net_arch_actor = []
    net_arch_critic = []
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        # Custom params
        use_quantum_layer: bool = False,
        quantum_config: Optional[Dict[str, Any]] = None,
        use_ess_layer: bool = False, 
        ess_config: Optional[Dict[str, Any]] = None, 
        use_adaptive_action_noise: bool = False,
        adaptive_noise_config: Optional[Dict[str, Any]] = None,
        dropout_rate_actor: float = 0.0,
        dropout_rate_critic: float = 0.0,
        use_layer_norm_actor: bool = False,
        use_layer_norm_critic: bool = False,
    ):
        # Log received feature extractor parameters BEFORE super().__init__
        logger.info(f"CustomSACPolicy.__init__ (before super): features_extractor_class name: {features_extractor_class.__name__ if hasattr(features_extractor_class, '__name__') else features_extractor_class}")
        logger.info(f"CustomSACPolicy.__init__ (before super): type(features_extractor_class)={type(features_extractor_class)}")
        logger.info(f"CustomSACPolicy.__init__ (before super): features_extractor_kwargs keys={list(features_extractor_kwargs.keys()) if features_extractor_kwargs else None}")
        if features_extractor_kwargs and 'model_config' in features_extractor_kwargs:
            logger.info(f"CustomSACPolicy.__init__ (before super): model_config in kwargs type={type(features_extractor_kwargs['model_config'])}")
            logger.info(f"CustomSACPolicy.__init__ (before super): model_config content (first level keys)={list(features_extractor_kwargs['model_config'].keys()) if isinstance(features_extractor_kwargs['model_config'], dict) else None}")


        # --- 修正: 明確設置 self.use_sde ---
        self.use_sde = use_sde

        self.use_quantum_layer = use_quantum_layer
        self.quantum_config = quantum_config if quantum_config else {}
        self.use_ess_layer = use_ess_layer
        self.ess_config = ess_config if ess_config else {}
        self.use_adaptive_action_noise = use_adaptive_action_noise
        self.adaptive_noise_config = adaptive_noise_config if adaptive_noise_config else {}
        self.dropout_rate_actor = dropout_rate_actor
        self.dropout_rate_critic = dropout_rate_critic
        self.use_layer_norm_actor = use_layer_norm_actor
        self.use_layer_norm_critic = use_layer_norm_critic
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.clip_mean = clip_mean
        self.sde_net_arch = sde_net_arch
        self.n_critics = n_critics
        self.share_features_extractor = share_features_extractor

        _features_extractor_kwargs = features_extractor_kwargs or {}

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde, # SACPolicy constructor accepts use_sde
            log_std_init=log_std_init, # SACPolicy constructor accepts log_std_init
            # sde_net_arch, use_expln, clip_mean are NOT direct parameters for SACPolicy.__init__
            # They are handled internally or when the actor is created.
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=_features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )
        
        # --- net_arch 解析與屬性設置（必須在 super().__init__ 之前）---
        if isinstance(net_arch, dict):
            self.net_arch = net_arch
            self.net_arch_actor = net_arch.get('pi', [])
            self.net_arch_critic = net_arch.get('qf', [])
        else:
            self.net_arch = net_arch
            self.net_arch_actor = net_arch
            self.net_arch_critic = net_arch

        logger.info(f"CustomSACPolicy.__init__ (after super): self.features_extractor type: {type(self.features_extractor)}")
        if hasattr(self.features_extractor, 'model_config_dict'): # Check for the attribute name used in EnhancedTransformerFeatureExtractor
            logger.info(f"  FE model_config_dict: {getattr(self.features_extractor, 'model_config_dict', 'N/A')}")
        if hasattr(self.features_extractor, 'use_symbol_embedding'):
            logger.info(f"  FE use_symbol_embedding: {getattr(self.features_extractor, 'use_symbol_embedding', 'N/A')}")
        if hasattr(self.features_extractor, 'use_msfe'):
            logger.info(f"  FE use_msfe: {getattr(self.features_extractor, 'use_msfe', 'N/A')}")
        if hasattr(self.features_extractor, 'use_cts_fusion'):
            logger.info(f"  FE use_cts_fusion: {getattr(self.features_extractor, 'use_cts_fusion', 'N/A')}")


    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> QuantumActor:
        actor_features_extractor = features_extractor
        if actor_features_extractor is None:
            actor_features_extractor = self.features_extractor # 使用策略的特徵提取器
            logger.info(f"CustomSACPolicy.make_actor: features_extractor 參數為 None. 使用 self.features_extractor: {type(self.features_extractor)}")
        else:
            logger.info(f"CustomSACPolicy.make_actor: 使用提供的 features_extractor: {type(features_extractor)}")

        if actor_features_extractor is None:
            logger.error("[CRITICAL ERROR] CustomSACPolicy.make_actor: actor_features_extractor 為 None.")
            raise ValueError("actor_features_extractor 在 make_actor 中不能為 None.")

        # actor_net_arch 是 actor 頭部網絡的架構 (pi 部分)
        actor_net_arch = self.net_arch if isinstance(self.net_arch, list) else self.net_arch.get('pi', [])
        
        logger.info(f"CustomSACPolicy.make_actor: actor_features_extractor 類型: {type(actor_features_extractor)}, features_dim: {actor_features_extractor.features_dim}")
        logger.info(f"CustomSACPolicy.make_actor: actor_net_arch: {actor_net_arch}")
        # logger.info(f"CustomSACPolicy.make_actor: Quantum kwargs for Actor: {q_kwargs_for_actor}, use_quantum: {use_quantum}") # 移除舊的 q_kwargs_for_Actor 日誌

        # 創建 QuantumActor 實例，傳遞所有必要的參數
        return QuantumActor(
            observation_space=self.observation_space,
            action_space=self.action_space,
            net_arch=actor_net_arch, # Actor 頭部網絡的架構
            features_extractor=actor_features_extractor, # 實例化的特徵提取器
            features_dim=actor_features_extractor.features_dim, # 特徵維度
            
            # SB3 Actor 所需的參數 (從策略自身獲取)
            log_std_init=self.log_std_init,
            use_sde=self.use_sde,
            sde_net_arch=self.sde_net_arch, # 策略的 SDE 網絡架構 (用於 Actor 的 SDE 特徵網絡)
            use_expln=self.use_expln,
            clip_mean=self.clip_mean,
            normalize_images=self.normalize_images, # BasePolicy 的參數

            # QuantumActor 特有的參數 (從策略自身獲取)
            activation_fn=self.activation_fn, # 策略的激活函數
            use_quantum_layer=self.use_quantum_layer,
            quantum_config=self.quantum_config,
            use_ess_layer=self.use_ess_layer,
            ess_config=self.ess_config,
            # use_adaptive_action_noise 和 adaptive_noise_config 也是策略的屬性
            use_adaptive_action_noise=self.use_adaptive_action_noise,
            adaptive_noise_config=self.adaptive_noise_config,
            dropout_rate=self.dropout_rate_actor, # Actor 特定的 dropout 率
            use_layer_norm_actor=self.use_layer_norm_actor, # Actor 特定的 LayerNorm 使用標誌
            device=self.device # 策略的設備
        )

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> QuantumCritic:
        logger.info(f"CustomSACPolicy.make_critic called. Provided features_extractor type: {type(features_extractor)}")
        critic_features_extractor = features_extractor if features_extractor is not None else self.features_extractor

        logger.info(f"CustomSACPolicy.make_critic: features_extractor arg was {'provided' if features_extractor else 'None'}. Using self.features_extractor: {type(self.features_extractor)}")
        if critic_features_extractor is None:
            logger.error("[CRITICAL ERROR] CustomSACPolicy.make_critic: critic_features_extractor is None.")
            raise ValueError("critic_features_extractor cannot be None in make_critic for QuantumCritic.")

        critic_net_arch = self.net_arch if isinstance(self.net_arch, list) else self.net_arch.get('vf', [])
        q_kwargs_for_critic = self.quantum_critic_kwargs.copy()
        use_quantum = q_kwargs_for_critic.pop("use_quantum_layer", False)

        logger.info(f"CustomSACPolicy.make_critic: critic_features_extractor type: {type(critic_features_extractor)}, features_dim: {critic_features_extractor.features_dim}")
        logger.info(f"CustomSACPolicy.make_critic: critic_net_arch: {critic_net_arch}")
        logger.info(f"CustomSACPolicy.make_critic: Quantum kwargs for Critic: {q_kwargs_for_critic}, use_quantum: {use_quantum}")

        # 正確做法：只回傳一個 QuantumCritic，並將 n_critics 傳給它，讓其內部管理多個 Q-network
        critic = QuantumCritic(
            self.observation_space,
            self.action_space,
            net_arch=critic_net_arch,
            features_extractor=critic_features_extractor,
            features_dim=critic_features_extractor.features_dim,
            activation_fn=self.activation_fn,
            normalize_images=self.normalize_images,
            n_critics=self.n_critics,  # 讓 QuantumCritic 內部管理多 Q-network
            # 其餘自定義參數可安全傳遞
            # use_ess_layer=self.use_ess_layer,
            # ess_config=self.ess_config,
            # dropout_rate=self.dropout_rate,
            # use_layer_norm_critic=self.use_layer_norm_critic,
            # device=self.device,
            # share_features_extractor=self.share_features_extractor,
        )
        return critic

    def _build_features_extractor(self) -> BaseFeaturesExtractor:
        # Ensure self.logger is available
        current_logger = getattr(self, 'logger', logging.getLogger(__name__))
        current_logger.info("[CUSTOM_BUILD_EXTRACTOR] CustomSACPolicy._build_features_extractor: ENTERING.")
        print("[CUSTOM_BUILD_EXTRACTOR_PRINT] CustomSACPolicy._build_features_extractor: ENTERING.")
        
        if not hasattr(self, 'features_extractor_class') or self.features_extractor_class is None:
            current_logger.error("[CUSTOM_BUILD_EXTRACTOR] self.features_extractor_class is not set or is None!")
            raise AttributeError("features_extractor_class not set on policy instance before _build_features_extractor call.")
        
        if not hasattr(self, 'observation_space') or self.observation_space is None:
            current_logger.error("[CUSTOM_BUILD_EXTRACTOR] self.observation_space is not set or is None!")
            raise AttributeError("observation_space not set on policy instance before _build_features_extractor call.")
            
        current_logger.info(f"[CUSTOM_BUILD_EXTRACTOR] features_extractor_class: {self.features_extractor_class}")
        current_logger.info(f"[CUSTOM_BUILD_EXTRACTOR] features_extractor_kwargs: {self.features_extractor_kwargs}")

        try:
            features_extractor = self.features_extractor_class(
                self.observation_space,
                **self.features_extractor_kwargs,
            )
            current_logger.info(f"[CUSTOM_BUILD_EXTRACTOR] Instantiated features_extractor of type: {type(features_extractor)}")
            if not isinstance(features_extractor, BaseFeaturesExtractor):
                current_logger.error(f"[CUSTOM_BUILD_EXTRACTOR] Created extractor is not a BaseFeaturesExtractor! Type: {type(features_extractor)}")
            if not hasattr(features_extractor, 'features_dim'):
                current_logger.error(f"[CUSTOM_BUILD_EXTRACTOR] Created extractor has no features_dim! Type: {type(features_extractor)}")
            else:
                current_logger.info(f"[CUSTOM_BUILD_EXTRACTOR] Created extractor features_dim: {features_extractor.features_dim}")

        except Exception as e:
            current_logger.error(f"[CUSTOM_BUILD_EXTRACTOR] Exception during instantiation: {e}", exc_info=True)
            raise
            
        current_logger.info("[CUSTOM_BUILD_EXTRACTOR] CustomSACPolicy._build_features_extractor: EXITING SUCCESSFULLY.")
        return features_extractor

    def _build(self, lr_schedule: Schedule) -> None:
        # Ensure self.logger is available
        current_logger = getattr(self, 'logger', logging.getLogger(__name__))
        current_logger.info(f"CustomSACPolicy._build: ENTERING. share_features_extractor={self.share_features_extractor}")
        current_logger.info(f"CustomSACPolicy._build (ENTRY): id(self)={id(self)}")
        current_logger.info(f"CustomSACPolicy._build (ENTRY): self.features_extractor IS {self.features_extractor}")
        current_logger.info(f"CustomSACPolicy._build (ENTRY): type(self.features_extractor) IS {type(self.features_extractor)}")
        current_logger.info(f"CustomSACPolicy._build (ENTRY): self.features_extractor_class type: {type(self.features_extractor_class) if hasattr(self, 'features_extractor_class') else 'Not Set'}")

        # CRITICAL FIX: If features_extractor is None, build it explicitly
        if self.features_extractor is None:
            current_logger.warning("CustomSACPolicy._build: self.features_extractor is None. Building it explicitly.")
            print("[EXPLICIT_BUILD] Building features_extractor explicitly in _build")
            self.features_extractor = self._build_features_extractor()
            current_logger.info(f"CustomSACPolicy._build: Explicitly built features_extractor: {self.features_extractor}")
            current_logger.info(f"CustomSACPolicy._build: Type of explicitly built features_extractor: {type(self.features_extractor)}")

        # 確保 net_arch_actor 屬性存在，避免 make_actor 報錯
        if not hasattr(self, 'net_arch_actor') or self.net_arch_actor is None:
            self.net_arch_actor = self.net_arch if hasattr(self, 'net_arch') else [256, 256]

        super()._build(lr_schedule) # This calls SACPolicy._build, which then calls make_actor, make_critic

        current_logger.info(f"CustomSACPolicy._build (after super()._build): self.features_extractor IS {self.features_extractor}")
        current_logger.info(f"CustomSACPolicy._build (after super()._build): type(self.features_extractor) IS {type(self.features_extractor)}")
        
        if self.actor and hasattr(self.actor, 'features_extractor'):
            actor_fe = self.actor.features_extractor
            current_logger.info(f"CustomSACPolicy._build (after super()._build): self.actor.features_extractor IS {actor_fe}")
            current_logger.info(f"CustomSACPolicy._build (after super()._build): type(self.actor.features_extractor) IS {type(actor_fe)}")
            if self.share_features_extractor:
                if actor_fe is not self.features_extractor:
                    current_logger.warning("CustomSACPolicy._build: actor.features_extractor is NOT THE SAME INSTANCE as self.features_extractor, even though shared!")
                    current_logger.warning(f"ID self.features_extractor: {id(self.features_extractor) if self.features_extractor else 'N/A'}, ID actor.features_extractor: {id(actor_fe) if actor_fe else 'N/A'}")
                else:
                    current_logger.info("CustomSACPolicy._build: actor.features_extractor IS THE SAME INSTANCE as self.features_extractor (as expected for shared).")

        current_logger.info(f"CustomSACPolicy._build: EXITING.")
    
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> QuantumActor:
        # Ensure self.logger is available
        current_logger = getattr(self, 'logger', logging.getLogger(__name__))
        current_logger.info(f"CustomSACPolicy.make_actor: ENTERING.")
        current_logger.info(f"CustomSACPolicy.make_actor: Received features_extractor argument IS {features_extractor}")
        current_logger.info(f"CustomSACPolicy.make_actor: Received features_extractor argument type IS {type(features_extractor)}")
        current_logger.info(f"CustomSACPolicy.make_actor: self.features_extractor (at make_actor entry) IS {self.features_extractor}")
        current_logger.info(f"CustomSACPolicy.make_actor: type(self.features_extractor) (at make_actor entry) IS {type(self.features_extractor)}")

        actor_features_extractor = features_extractor or self.features_extractor
        
        current_logger.info(f"CustomSACPolicy.make_actor: actor_features_extractor (chosen) IS {actor_features_extractor}")
        current_logger.info(f"CustomSACPolicy.make_actor: type(actor_features_extractor) (chosen) IS {type(actor_features_extractor)}")

        if actor_features_extractor is None:
            current_logger.critical("CustomSACPolicy.make_actor: actor_features_extractor is None. RAISING ValueError.")
            current_logger.critical(f"Debug info: self.share_features_extractor = {self.share_features_extractor if hasattr(self, 'share_features_extractor') else 'Not Set'}")
            current_logger.critical(f"Debug info: self.observation_space = {self.observation_space if hasattr(self, 'observation_space') else 'Not Set'}")
            current_logger.critical(f"Debug info: self.features_extractor_class = {self.features_extractor_class if hasattr(self, 'features_extractor_class') else 'Not Set'}")
            current_logger.critical(f"Debug info: self.features_extractor_kwargs = {self.features_extractor_kwargs if hasattr(self, 'features_extractor_kwargs') else 'Not Set'}")
            raise ValueError("actor_features_extractor 在 make_actor 中不能為 None.")
        
        # actor_net_arch 是 actor 頭部網絡的架構 (pi 部分)
        actor_net_arch = self.net_arch if isinstance(self.net_arch, list) else self.net_arch.get('pi', [])
        
        logger.info(f"CustomSACPolicy.make_actor: actor_features_extractor 類型: {type(actor_features_extractor)}, features_dim: {actor_features_extractor.features_dim}")
        logger.info(f"CustomSACPolicy.make_actor: actor_net_arch: {actor_net_arch}")
        # logger.info(f"CustomSACPolicy.make_actor: Quantum kwargs for Actor: {q_kwargs_for_actor}, use_quantum: {use_quantum}") # 移除舊的 q_kwargs_for_Actor 日誌

        # 創建 QuantumActor 實例，傳遞所有必要的參數
        return QuantumActor(
            observation_space=self.observation_space,
            action_space=self.action_space,
            net_arch=actor_net_arch, # Actor 頭部網絡的架構
            features_extractor=actor_features_extractor, # 實例化的特徵提取器
            features_dim=actor_features_extractor.features_dim, # 特徵維度
            
            # SB3 Actor 所需的參數 (從策略自身獲取)
            log_std_init=self.log_std_init,
            use_sde=self.use_sde,
            sde_net_arch=self.sde_net_arch, # 策略的 SDE 網絡架構 (用於 Actor 的 SDE 特徵網絡)
            use_expln=self.use_expln,
            clip_mean=self.clip_mean,
            normalize_images=self.normalize_images, # BasePolicy 的參數

            # QuantumActor 特有的參數 (從策略自身獲取)
            activation_fn=self.activation_fn, # 策略的激活函數
            use_quantum_layer=self.use_quantum_layer,
            quantum_config=self.quantum_config,
            use_ess_layer=self.use_ess_layer,
            ess_config=self.ess_config,
            # use_adaptive_action_noise 和 adaptive_noise_config 也是策略的屬性
            use_adaptive_action_noise=self.use_adaptive_action_noise,
            adaptive_noise_config=self.adaptive_noise_config,
            dropout_rate=self.dropout_rate_actor, # Actor 特定的 dropout 率
            use_layer_norm_actor=self.use_layer_norm_actor, # Actor 特定的 LayerNorm 使用標誌
            device=self.device # 策略的設備
        )

    # ... make_critic method, potentially with similar logging if issues persist there ...
    # Ensure QuantumCritic is defined or imported
    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> QuantumCritic:
        current_logger = getattr(self, 'logger', logging.getLogger(__name__))
        current_logger.info(f"CustomSACPolicy.make_critic: ENTERING.")
        current_logger.info(f"CustomSACPolicy.make_critic: Received features_extractor argument IS {features_extractor}")
        current_logger.info(f"CustomSACPolicy.make_critic: self.features_extractor (at make_critic entry) IS {self.features_extractor}")

        critic_features_extractor = features_extractor or self.features_extractor

        current_logger.info(f"CustomSACPolicy.make_critic: critic_features_extractor (chosen) IS {critic_features_extractor}")
        
        if critic_features_extractor is None:
            current_logger.critical("CustomSACPolicy.make_critic: critic_features_extractor is None. RAISING ValueError.")
            # Add similar debug info as in make_actor if needed
            raise ValueError("critic_features_extractor 在 make_critic 中不能為 None.")

        # Assuming QuantumCritic is defined and has a similar signature to SB3's Critic
        # You might need to adjust this part based on your QuantumCritic definition
        critic = QuantumCritic(
            self.observation_space,
            self.action_space,
            net_arch=self.net_arch_critic, # Use resolved critic_net_arch
            features_extractor=critic_features_extractor,
            features_dim=critic_features_extractor.features_dim, # Pass features_dim
            activation_fn=self.activation_fn,
            normalize_images=self.normalize_images,
            # 僅傳遞必要參數，**self.critic_kwargs 需移除 net_arch 相關重複項
            # 其餘自定義參數可安全傳遞
            # 例如:
            # use_ess_layer=self.use_ess_layer,
            # ess_config=self.ess_config,
            # dropout_rate=self.dropout_rate,
            # use_layer_norm_critic=self.use_layer_norm_critic,
            # n_critics=self.n_critics,
            # device=self.device,
            # share_features_extractor=self.share_features_extractor,
        )
        return critic