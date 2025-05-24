# src/agent/sac_agent_wrapper.py
"""
SAC智能體包裝器。
封裝Stable Baselines3的SAC智能體，提供簡化的接口，並集成TensorBoard日誌。
"""
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as sb3_logger_configure # 重命名以避免衝突
from gymnasium import spaces # <-- gymnasium.spaces
import gymnasium as gym # <-- 導入 gymnasium as gym 以便MockEnv使用
import torch
from typing import Optional, Dict, Any, Type, Callable, Union, List, Tuple # <--- 添加 Tuple
from pathlib import Path
import time
import os
import numpy as np # <--- 在文件頂部導入 numpy
from datetime import datetime # <--- 在文件頂部導入 datetime
import pandas as pd # <--- 在文件頂部導入 pandas
import sys # 確保導入

try:
    from agent.sac_policy import CustomSACPolicy
    from common.config import (
        DEVICE, SAC_LEARNING_RATE, SAC_BATCH_SIZE, SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR,
        SAC_LEARNING_STARTS_FACTOR, SAC_GAMMA, SAC_ENT_COEF,
        SAC_TRAIN_FREQ_STEPS, SAC_GRADIENT_STEPS, SAC_TAU,
        TIMESTEPS, LOGS_DIR, MAX_SYMBOLS_ALLOWED # <--- 確保 MAX_SYMBOLS_ALLOWED 也被導入
    )
    from common.logger_setup import logger
except ImportError:
    project_root_wrapper = Path(__file__).resolve().parent.parent.parent
    src_path_wrapper = project_root_wrapper / "src"
    if str(project_root_wrapper) not in sys.path:
        sys.path.insert(0, str(project_root_wrapper))
    try:
        from src.agent.sac_policy import CustomSACPolicy
        from src.common.config import (
            DEVICE, SAC_LEARNING_RATE, SAC_BATCH_SIZE, SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR,
            SAC_LEARNING_STARTS_FACTOR, SAC_GAMMA, SAC_ENT_COEF,
            SAC_TRAIN_FREQ_STEPS, SAC_GRADIENT_STEPS, SAC_TAU, TIMESTEPS, LOGS_DIR, MAX_SYMBOLS_ALLOWED
        )
        from src.common.logger_setup import logger
        logger.info("Direct run SACAgentWrapper: Successfully re-imported modules.")
    except ImportError as e_retry_wrapper:
        import logging
        logger = logging.getLogger("sac_agent_wrapper_fallback") # type: ignore
        logger.setLevel(logging.INFO)
        _ch_wrapper = logging.StreamHandler(sys.stdout)
        _ch_wrapper.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        if not logger.handlers: logger.addHandler(_ch_wrapper)
        logger.error(f"Direct run SACAgentWrapper: Critical import error: {e_retry_wrapper}", exc_info=True)
        CustomSACPolicy = None # type: ignore
        DEVICE="cpu"; SAC_LEARNING_RATE=3e-4; SAC_BATCH_SIZE=256; SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR=1000
        SAC_LEARNING_STARTS_FACTOR=10; SAC_GAMMA=0.99; SAC_ENT_COEF='auto'
        SAC_TRAIN_FREQ_STEPS=1; SAC_GRADIENT_STEPS=1; SAC_TAU=0.005; TIMESTEPS=128
        LOGS_DIR=Path("./logs_fallback"); MAX_SYMBOLS_ALLOWED=20


class SACAgentWrapper:
    def __init__(self,
                 env: DummyVecEnv,
                 policy_class: Type[CustomSACPolicy] = CustomSACPolicy, # type: ignore
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 learning_rate: Union[float, Callable[[float], float]] = SAC_LEARNING_RATE,
                 batch_size: int = SAC_BATCH_SIZE,
                 buffer_size_factor: int = SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR,
                 learning_starts_factor: int = SAC_LEARNING_STARTS_FACTOR,
                 gamma: float = SAC_GAMMA,
                 ent_coef: Union[str, float] = SAC_ENT_COEF,
                 train_freq_steps: int = SAC_TRAIN_FREQ_STEPS,
                 gradient_steps: int = SAC_GRADIENT_STEPS,
                 tau: float = SAC_TAU,
                 verbose: int = 0,
                 tensorboard_log_path: Optional[str] = None,
                 seed: Optional[int] = None,
                 custom_objects: Optional[Dict[str, Any]] = None,
                 device: Union[torch.device, str] = DEVICE
                ):
        self.env = env
        self.policy_class = policy_class
        self.policy_kwargs = policy_kwargs if policy_kwargs is not None else {}
        num_active_symbols = getattr(self.env.envs[0], 'num_active_symbols_in_slots', 1)
        calculated_buffer_size = num_active_symbols * TIMESTEPS * buffer_size_factor
        self.buffer_size = min(max(calculated_buffer_size, batch_size * 200, 50000),200000)
        calculated_learning_starts = num_active_symbols * batch_size * learning_starts_factor
        self.learning_starts = max(calculated_learning_starts, batch_size * 20, 2000)
        logger.info(f"SAC Agent Wrapper: num_active_symbols={num_active_symbols}, BufferSize={self.buffer_size}, LearningStarts={self.learning_starts}")
        if tensorboard_log_path is None:
            current_time_str = datetime.now().strftime("%Y%m%d-%H%M%S") # datetime 已導入
            self.tensorboard_log_path = str(LOGS_DIR / f"sac_tensorboard_logs_{current_time_str}")
            os.makedirs(self.tensorboard_log_path, exist_ok=True)
            logger.info(f"TensorBoard日誌將保存到: {self.tensorboard_log_path}")
        else:
            self.tensorboard_log_path = tensorboard_log_path
        self.agent = SAC(
            policy=self.policy_class, env=self.env, learning_rate=learning_rate,
            buffer_size=self.buffer_size, learning_starts=self.learning_starts, batch_size=batch_size,
            tau=tau, gamma=gamma, train_freq=(train_freq_steps, "step"), gradient_steps=gradient_steps,
            ent_coef=ent_coef, policy_kwargs=self.policy_kwargs, verbose=verbose, seed=seed,
            device=device, tensorboard_log=self.tensorboard_log_path
        )
        self.custom_objects = custom_objects if custom_objects is not None else {}
        self.custom_objects["policy_class"] = self.policy_class
        if "features_extractor_class" in self.policy_kwargs:
            self.custom_objects["features_extractor_class"] = self.policy_kwargs["features_extractor_class"]
        elif hasattr(self.policy_class, "features_extractor_class"):
             self.custom_objects["features_extractor_class"] = self.policy_class.features_extractor_class # type: ignore
        logger.info("SACAgentWrapper 初始化完成。")

    def train(self, total_timesteps: int, callback: Optional[Union[BaseCallback, List[BaseCallback]]] = None,
              log_interval: int = 1, reset_num_timesteps: bool = True):
        logger.info(f"開始訓練 SAC 智能體，總步數: {total_timesteps}...")
        start_time = time.time()
        try:
            self.agent.learn(total_timesteps=total_timesteps, callback=callback, log_interval=log_interval,
                             reset_num_timesteps=reset_num_timesteps, progress_bar=False)
            training_duration = time.time() - start_time
            logger.info(f"智能體訓練完成。耗時: {training_duration:.2f} 秒。")
        except Exception as e: logger.error(f"智能體訓練過程中發生錯誤: {e}", exc_info=True); raise

    def predict(self, observation: np.ndarray, state: Optional[Tuple[np.ndarray, ...]] = None, # np, Tuple 已導入
                episode_start: Optional[np.ndarray] = None, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        return self.agent.predict(observation, state=state, episode_start=episode_start, deterministic=deterministic)

    def save(self, path: Union[str, Path]):
        try: self.agent.save(path); logger.info(f"智能體模型已保存到: {path}")
        except Exception as e: logger.error(f"保存智能體模型失敗: {e}", exc_info=True); raise

    def load(self, path: Union[str, Path], env: Optional[DummyVecEnv] = None):
        try:
            self.agent = SAC.load(path, env=env or self.env, custom_objects=self.custom_objects, device=self.agent.device)
            logger.info(f"智能體模型已從 {path} 加載。")
            current_time_str = datetime.now().strftime("%Y%m%d-%H%M%S") # datetime 已導入
            new_tb_log_path = str(LOGS_DIR / f"sac_tensorboard_logs_loaded_{current_time_str}")
            os.makedirs(new_tb_log_path, exist_ok=True)
            logger.warning(f"加載模型後，TensorBoard日誌可能仍在原路徑記錄。建議重新創建Agent或手動管理日誌路徑。")
        except Exception as e: logger.error(f"加載智能體模型失敗: {e}", exc_info=True); raise

    def get_policy_parameters(self):
        if self.agent and self.agent.policy: return self.agent.policy.parameters()
        return None
    def get_feature_extractor_parameters(self):
        if self.agent and self.agent.policy and hasattr(self.agent.policy, 'features_extractor') \
           and self.agent.policy.features_extractor is not None \
           and hasattr(self.agent.policy.features_extractor, 'transformer'):
            return self.agent.policy.features_extractor.transformer.parameters() # type: ignore
        return None

if __name__ == "__main__":
    logger.info("正在直接運行 sac_agent_wrapper.py 進行測試...")
    # 確保 CustomSACPolicy, UniversalMemoryMappedDataset, TIMESTEPS, MAX_SYMBOLS_ALLOWED, DEVICE 在此作用域可用
    if 'CustomSACPolicy' not in globals() or CustomSACPolicy is None:
        logger.error("CustomSACPolicy is None. Test cannot proceed."); sys.exit(1)
    # UniversalMemoryMappedDataset 的後備導入已經是 type('Dummy',...)，所以不需要再用 MockDataset
    
    required_configs_main = ['TIMESTEPS', 'MAX_SYMBOLS_ALLOWED', 'DEVICE']
    for cfg_var_main in required_configs_main:
        if cfg_var_main not in globals():
            logger.error(f"配置變量 {cfg_var_main} 在 __main__ 中未定義。")
            sys.exit(1)

    test_symbols_main = ["FAKE_SYM1", "FAKE_SYM2"] # 使用不同的變量名以避免衝突
    num_input_feat_main = 9

    # 模擬 UniversalTradingEnvV2 的觀察和動作空間
    # np, spaces, gym 應在頂部導入
    mock_obs_space_main = spaces.Dict({
        "features_from_dataset": spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_SYMBOLS_ALLOWED, TIMESTEPS, num_input_feat_main), dtype=np.float32),
        "current_positions_nominal_ratio_ac": spaces.Box(low=-1.0, high=1.0, shape=(MAX_SYMBOLS_ALLOWED,), dtype=np.float32),
        "unrealized_pnl_ratio_ac": spaces.Box(low=-1.0, high=1.0, shape=(MAX_SYMBOLS_ALLOWED,), dtype=np.float32),
        "margin_level": spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
        "padding_mask": spaces.Box(low=0, high=1, shape=(MAX_SYMBOLS_ALLOWED,), dtype=np.bool_)
    })
    mock_action_space_main = spaces.Box(low=-1.0, high=1.0, shape=(MAX_SYMBOLS_ALLOWED,), dtype=np.float32)

    class MockEnvMain(gym.Env): # 使用不同的類名
        def __init__(self):
            super().__init__()
            self.observation_space = mock_obs_space_main
            self.action_space = mock_action_space_main
            self.num_active_symbols_in_slots = len(test_symbols_main)
        def reset(self, seed=None, options=None): return self.observation_space.sample(), {}
        def step(self, action): return self.observation_space.sample(), 0.0, False, False, {}
        def render(self): pass
        def close(self): pass

    dummy_vec_env_main = DummyVecEnv([lambda: MockEnvMain()])

    test_policy_kwargs_main = {
        "features_extractor_kwargs": dict(transformer_output_dim_per_symbol=64, model_dim=128, num_time_encoder_layers=1, num_cross_asset_layers=1, num_heads=2, ffn_dim=128),
        "net_arch": dict(pi=[64, 64], qf=[64, 64])
    }
    try:
        logger.info("初始化 SACAgentWrapper...")
        agent_wrapper = SACAgentWrapper(env=dummy_vec_env_main, policy_kwargs=test_policy_kwargs_main, batch_size=32, buffer_size_factor=10, learning_starts_factor=2, verbose=0)
        logger.info("SACAgentWrapper 初始化成功。")
        model_save_path_main = LOGS_DIR / "test_sac_agent.zip"
        logger.info(f"測試保存模型到: {model_save_path_main}")
        agent_wrapper.save(model_save_path_main)
        assert model_save_path_main.exists(), "模型文件未成功保存"
        logger.info("測試加載模型...")
        agent_wrapper.load(model_save_path_main)
        logger.info("模型加載成功。")
        if agent_wrapper.get_policy_parameters() is not None: logger.info("成功獲取策略參數。")
        if agent_wrapper.get_feature_extractor_parameters() is not None: logger.info("成功獲取特徵提取器參數。")
    except Exception as e: logger.error(f"SACAgentWrapper 測試過程中發生錯誤: {e}", exc_info=True)
    finally:
        if 'model_save_path_main' in locals() and model_save_path_main.exists():
            try: os.remove(model_save_path_main); logger.info(f"已刪除測試模型文件: {model_save_path_main}")
            except Exception as e_del: logger.warning(f"刪除測試模型文件失敗: {e_del}")
    logger.info("sac_agent_wrapper.py 測試執行完畢。")