# src/agent/sac_policy.py
"""
定義自定義的 SAC 策略，它使用 AdvancedTransformerFeatureExtractor。
"""
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import torch
import torch.nn as nn
from typing import List, Type, Optional, Dict, Any, Union, Callable
import sys
from pathlib import Path
import numpy as np
import logging # 確保 logging 導入

# Flag to prevent duplicate import logging
_import_logged = False

# --- Simplified Import Block ---
try:
    from src.common.logger_setup import logger
    if not _import_logged:
        logger.debug("sac_policy.py: Successfully imported logger from src.common.logger_setup.")
except ImportError:
    logger = logging.getLogger("sac_policy_fallback") # type: ignore
    logger.setLevel(logging.DEBUG)
    _ch_policy_fallback = logging.StreamHandler(sys.stdout)
    _ch_policy_fallback.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    if not logger.handlers: logger.addHandler(_ch_policy_fallback)
    logger.warning("sac_policy.py: Failed to import logger from src.common.logger_setup. Using fallback logger.")

try:
    from src.agent.enhanced_feature_extractor import EnhancedTransformerFeatureExtractor
    from src.common.config import (
        MAX_SYMBOLS_ALLOWED, TIMESTEPS, TRANSFORMER_OUTPUT_DIM_PER_SYMBOL
    )
    if not _import_logged:
        logger.info("sac_policy.py: Successfully imported EnhancedTransformerFeatureExtractor and common.config.")
        _import_logged = True
except ImportError as e:
    logger.error(f"sac_policy.py: Failed to import EnhancedTransformerFeatureExtractor or common.config: {e}. Using fallback values.", exc_info=True) # type: ignore
    EnhancedTransformerFeatureExtractor = None # type: ignore
    MAX_SYMBOLS_ALLOWED = 20
    TIMESTEPS = 128
    TRANSFORMER_OUTPUT_DIM_PER_SYMBOL = 64
    logger.warning("sac_policy.py: Using fallback values for config due to import error.") # type: ignore

# 量子策略層導入
try:
    from src.agent.quantum_policy import QuantumPolicyWrapper
    quantum_available = True
except ImportError as e:
    logger.warning(f"量子策略層導入失敗: {e}")
    QuantumPolicyWrapper = None
    quantum_available = False


class CustomSACPolicy(SACPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = EnhancedTransformerFeatureExtractor, # type: ignore
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = False,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        use_sde: bool = False,
        # sde_sample_freq 被移除
        use_expln: bool = False,
        # squash_output 被移除，它由Actor內部處理
        log_std_init: float = -3.0,
        clip_mean: float = 2.0,
    ):
        if net_arch is None:
            net_arch = dict(pi=[256, 256], qf=[256, 256])
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}        # 確保 features_extractor_class 不是 None (在後備導入情況下可能發生)
        actual_feat_ext_class = features_extractor_class
        if actual_feat_ext_class is None and 'EnhancedTransformerFeatureExtractor' in globals() and EnhancedTransformerFeatureExtractor is not None:
            logger.info("Using default feature extractor: EnhancedTransformerFeatureExtractor")
            actual_feat_ext_class = EnhancedTransformerFeatureExtractor # type: ignore
        elif actual_feat_ext_class is None:
            logger.error("CustomSACPolicy: features_extractor_class is None and fallback EnhancedTransformerFeatureExtractor not available!")
            # 這種情況下，父類初始化會失敗，或者需要提供一個默認的 BaseFeaturesExtractor
            # 為了能繼續，我們可能需要一個非常基礎的默認，但這會偏離我們的設計
            # super().__init__ 中 features_extractor_class 是必需的
            raise ValueError("features_extractor_class cannot be None for CustomSACPolicy")


        logger.info(f"Initializing CustomSACPolicy with feature_extractor: {actual_feat_ext_class.__name__}")
        logger.info(f"Actor/Critic net_arch: {net_arch}")

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=actual_feat_ext_class, # 使用修正後的
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
            use_sde=use_sde,
            # sde_sample_freq, # 已移除
            # use_expln, # Actor的參數，SACPolicy不直接收
            # squash_output, # Actor的參數，SACPolicy不直接收
            log_std_init=log_std_init,
            # clip_mean # Actor的參數，SACPolicy不直接收
        )
        # Actor的 squash_output, use_expln, clip_mean 等參數會在 SACPolicy 的 _build 方法中
        # 創建 Actor 實例時傳遞給 Actor 的構造函數。
        # 我們不需要在 CustomSACPolicy 的 __init__ 簽名中包含它們，
        # 除非我們打算重寫 _build_actor_critic 或 make_actor 方法並手動傳遞。
        # 為了保持與SACPolicy的接口一致性並讓它內部處理，我們從 __init__ 移除它們。
        # SACPolicy.__init__ 實際上也沒有直接接收 use_expln 和 clip_mean
        # 它們是 actor_kwargs 的一部分，或者由 make_actor 方法處理。
        # 為了安全，我們也從 super().__init__ 的調用中移除它們，讓 SACPolicy 的默認行為生效。
        # 經過查閱 SB3 源碼，SACPolicy 的 __init__ 確實不包含 use_expln 和 clip_mean
        # 而 log_std_init 是傳遞給 Actor 的。
        # 最安全的做法是只傳遞 SACPolicy.__init__ 實際定義的參數。

        # 初始化量子策略層（如果可用）
        self.quantum_layer = None
        self.use_quantum_layer = quantum_available
        
        if self.use_quantum_layer and QuantumPolicyWrapper is not None:
            try:
                # 獲取特徵提取器的輸出維度
                if hasattr(self.features_extractor, '_features_dim'):
                    features_dim = self.features_extractor._features_dim
                else:
                    features_dim = 256  # 默認值
                
                # 獲取動作空間維度
                if hasattr(action_space, 'shape'):
                    action_dim = action_space.shape[0]
                else:
                    action_dim = MAX_SYMBOLS_ALLOWED
                
                # 創建量子策略包裝器
                self.quantum_layer = QuantumPolicyWrapper(
                    state_dim=features_dim,
                    action_dim=action_dim,
                    latent_dim=256,
                    num_strategies=3,
                    num_energy_levels=8
                )
                
                logger.info(f"✅ 量子策略層已成功集成到SAC策略中")
                logger.info(f"   - 狀態維度: {features_dim}")
                logger.info(f"   - 動作維度: {action_dim}")
                logger.info(f"   - 策略數量: 3")
                logger.info(f"   - 能級數量: 8")
                
            except Exception as e:
                logger.error(f"❌ 量子策略層初始化失敗: {e}")
                self.quantum_layer = None
                self.use_quantum_layer = False
        else:
            if not quantum_available:
                logger.warning("⚠️ 量子策略層不可用，將使用標準SAC策略")
            else:
                logger.warning("⚠️ QuantumPolicyWrapper為None，將使用標準SAC策略")
    
    def forward_with_quantum_layer(self, features: torch.Tensor, deterministic: bool = False):
        """
        如果量子策略層可用，使用它來處理特徵
        
        Args:
            features: 來自特徵提取器的特徵
            deterministic: 是否使用確定性動作
            
        Returns:
            處理後的特徵或動作
        """
        if self.use_quantum_layer and self.quantum_layer is not None:
            try:
                # 使用量子策略層處理特徵
                quantum_output = self.quantum_layer(features, deterministic=deterministic)
                
                # 獲取量子指標用於監控
                if hasattr(self.quantum_layer, 'get_quantum_metrics'):
                    quantum_metrics = self.quantum_layer.get_quantum_metrics()
                    # 可以在這裡記錄量子指標
                
                return quantum_output
            except Exception as e:
                logger.warning(f"量子策略層處理失敗，回退到標準處理: {e}")
                return features
        else:
            return features
    
    def get_quantum_info(self):
        """獲取量子策略層的信息"""
        if self.use_quantum_layer and self.quantum_layer is not None:
            return {
                'quantum_available': True,
                'quantum_metrics': self.quantum_layer.get_quantum_metrics() if hasattr(self.quantum_layer, 'get_quantum_metrics') else {},
                'last_quantum_info': self.quantum_layer.get_last_quantum_info() if hasattr(self.quantum_layer, 'get_last_quantum_info') else None
            }
        else:
            return {
                'quantum_available': False,
                'quantum_metrics': {},
                'last_quantum_info': None
            }
        


if __name__ == "__main__":
    logger.info("正在直接運行 sac_policy.py 進行測試 (僅基礎導入和類定義檢查)...")
    if 'EnhancedTransformerFeatureExtractor' not in globals() or EnhancedTransformerFeatureExtractor is None:
        logger.error("EnhancedTransformerFeatureExtractor is None. Test cannot proceed.")
        sys.exit(1)
    required_configs = ['MAX_SYMBOLS_ALLOWED', 'TIMESTEPS', 'TRANSFORMER_OUTPUT_DIM_PER_SYMBOL']
    for cfg_var in required_configs:
        if cfg_var not in globals():
            logger.error(f"配置變量 {cfg_var} 未定義。請檢查頂部導入或後備邏輯。")
            sys.exit(1)

    num_test_input_features_policy = 9
    _max_symbols = MAX_SYMBOLS_ALLOWED
    _timesteps = TIMESTEPS
    _transformer_out_dim = TRANSFORMER_OUTPUT_DIM_PER_SYMBOL

    dummy_obs_space = spaces.Dict({
        "features_from_dataset": spaces.Box(low=-np.inf, high=np.inf, shape=(_max_symbols, _timesteps, num_test_input_features_policy), dtype=np.float32),
        "current_positions_nominal_ratio_ac": spaces.Box(low=-1.0, high=1.0, shape=(_max_symbols,), dtype=np.float32),
        "unrealized_pnl_ratio_ac": spaces.Box(low=-1.0, high=1.0, shape=(_max_symbols,), dtype=np.float32),
        "margin_level": spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
        "padding_mask": spaces.Box(low=0, high=1, shape=(_max_symbols,), dtype=np.bool_)    })
    dummy_action_space = spaces.Box(low=-1.0, high=1.0, shape=(_max_symbols,), dtype=np.float32)

    def dummy_lr_schedule(progress_remaining: float) -> float:
        return 3e-4 * progress_remaining
    
    try:
        # 創建 CustomSACPolicy 實例時，只傳遞它和其父類 __init__ 方法實際接受的參數
        policy = CustomSACPolicy(
            observation_space=dummy_obs_space,
            action_space=dummy_action_space,
            lr_schedule=dummy_lr_schedule,
            # net_arch, activation_fn 等使用默認值
            features_extractor_kwargs=dict( # 這些會傳給 EnhancedTransformerFeatureExtractor
                enhanced_transformer_output_dim_per_symbol=_transformer_out_dim
            ),
            # use_sde=False, # 如果需要可以設置
            # log_std_init, clip_mean, use_expln, squash_output 這些是Actor的參數，
            # SACPolicy 會在創建Actor時通過 actor_kwargs 處理或有默認值。
            # 我們可以在創建 SAC Agent 時通過 policy_kwargs={"actor_kwargs": {...}} 來傳遞它們。
            # 為了 __main__ 測試的簡潔，這裡不顯式傳遞它們給 CustomSACPolicy 的構造函數。
        )
        logger.info("CustomSACPolicy 實例化成功！")
        logger.info(f"  Policy features_extractor type: {type(policy.features_extractor)}")
        logger.info(f"  Policy actor type: {type(policy.actor)}")
        logger.info(f"  Policy critic type: {type(policy.critic)}")

    except Exception as e:
        logger.error(f"CustomSACPolicy 測試過程中發生錯誤: {e}", exc_info=True)
    logger.info("sac_policy.py 測試執行完畢。")