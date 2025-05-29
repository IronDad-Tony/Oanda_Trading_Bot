# src/agent/sac_agent_wrapper.py
"""
SAC智能體包裝器。
封裝Stable Baselines3的SAC智能體，提供簡化的接口，並集成TensorBoard日誌。
支援GPU加速訓練和混合精度訓練。
"""
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as sb3_logger_configure # 重命名以避免衝突
from gymnasium import spaces # <-- gymnasium.spaces
import gymnasium as gym # <-- 導入 gymnasium as gym 以便MockEnv使用
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Type, Callable, Union, List, Tuple # <--- 添加 Tuple
from pathlib import Path
import time
import os
import numpy as np # <--- 在文件頂部導入 numpy
from datetime import datetime # <--- 在文件頂部導入 datetime
import pandas as pd # <--- 在文件頂部導入 pandas
import sys # 確保導入
import gc  # 垃圾回收

try:
    from agent.sac_policy import CustomSACPolicy
    from common.config import (
        DEVICE, SAC_LEARNING_RATE, SAC_BATCH_SIZE, SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR,
        SAC_LEARNING_STARTS_FACTOR, SAC_GAMMA, SAC_ENT_COEF,
        SAC_TRAIN_FREQ_STEPS, SAC_GRADIENT_STEPS, SAC_TAU,
        TIMESTEPS, LOGS_DIR, MAX_SYMBOLS_ALLOWED, USE_AMP # <--- 添加混合精度訓練支持
    )
    from common.logger_setup import logger
except ImportError:
    # project_root_wrapper = Path(__file__).resolve().parent.parent.parent # 移除
    # src_path_wrapper = project_root_wrapper / "src" # 移除
    # if str(project_root_wrapper) not in sys.path: # 移除
    #     sys.path.insert(0, str(project_root_wrapper)) # 移除
    try:
        # 假設 PYTHONPATH 已設定，這些導入應該能工作
        from src.agent.sac_policy import CustomSACPolicy
        from src.common.config import (
            DEVICE, SAC_LEARNING_RATE, SAC_BATCH_SIZE, SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR,
            SAC_LEARNING_STARTS_FACTOR, SAC_GAMMA, SAC_ENT_COEF,
            SAC_TRAIN_FREQ_STEPS, SAC_GRADIENT_STEPS, SAC_TAU, TIMESTEPS, LOGS_DIR, MAX_SYMBOLS_ALLOWED, USE_AMP
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
        LOGS_DIR=Path("./logs_fallback"); MAX_SYMBOLS_ALLOWED=20; USE_AMP=False


class SACAgentWrapper:
    def __init__(self,
                 env: DummyVecEnv,
                 policy_class: Type[CustomSACPolicy] = CustomSACPolicy, # type: ignore
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 learning_rate: Union[float, Callable[[float], float]] = SAC_LEARNING_RATE,
                 batch_size: int = SAC_BATCH_SIZE, # SAC_BATCH_SIZE 現在是 64
                 buffer_size: Optional[int] = None, # 改為直接接收 buffer_size
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
                 device: Union[torch.device, str] = DEVICE,
                 use_amp: bool = USE_AMP
                ):
        self.env = env
        self.policy_class = policy_class
        _policy_kwargs = policy_kwargs if policy_kwargs is not None else {}
        self.use_amp = use_amp # This is the USE_AMP from common.config

        # Ensure features_extractor_kwargs exists and add use_amp to it
        if "features_extractor_kwargs" not in _policy_kwargs:
            _policy_kwargs["features_extractor_kwargs"] = {}
        _policy_kwargs["features_extractor_kwargs"]["use_amp"] = self.use_amp
        
        self.policy_kwargs = _policy_kwargs
        
        # 優化設備配置
        self.device = self._setup_device(device)
        logger.info(f"SAC Agent Wrapper: 使用設備 {self.device}, 混合精度訓練: {self.use_amp}")
        
        # num_active_symbols = getattr(self.env.envs[0], 'num_active_symbols_in_slots', 1) # Removed this line
        # Check if env is a list of environments (VecEnv) or a single environment
        if hasattr(self.env, 'envs') and isinstance(self.env.envs, list) and len(self.env.envs) > 0:
            # This is a VecEnv, access the first environment
            first_env = self.env.envs[0]
        else:
            # This is a single environment
            first_env = self.env
        
        num_active_symbols = getattr(first_env, 'num_tradable_symbols_this_episode', 1) # V5.0環境使用此屬性名

        calculated_buffer_size = num_active_symbols * TIMESTEPS * buffer_size_factor
        self.buffer_size = min(max(calculated_buffer_size, batch_size * 200, 50000),200000)
        calculated_learning_starts = num_active_symbols * batch_size * learning_starts_factor
        self.learning_starts = max(calculated_learning_starts, batch_size * 20, 2000)          # 根據設備調整批次大小以優化GPU利用率 - 恢復動態調整
        if self.device.type == 'cuda':
            # 啟用動態批次大小調整，根據GPU內存自動優化
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory_gb >= 16:  # 16GB以上GPU
                self.optimized_batch_size = min(batch_size * 3, 384)
            elif gpu_memory_gb >= 12:  # 12-16GB GPU
                self.optimized_batch_size = min(batch_size * 2, 256)
            elif gpu_memory_gb >= 8:  # 8-12GB GPU
                self.optimized_batch_size = min(int(batch_size * 1.5), 192)
            else:  # 小於8GB GPU
                self.optimized_batch_size = batch_size
            # 確保批次大小不會太小
            self.optimized_batch_size = max(self.optimized_batch_size, 32)
            logger.info(f"GPU模式 ({gpu_memory_gb:.1f}GB)：原始批次大小 {batch_size}，動態調整後批次大小 {self.optimized_batch_size}")
        else:
            self.optimized_batch_size = batch_size
            logger.info(f"CPU模式：批次大小 {self.optimized_batch_size}")
        logger.info(f"強制設定優化後批次大小為: {self.optimized_batch_size}")
            
        logger.info(f"SAC Agent Wrapper: num_active_symbols={num_active_symbols}, BufferSize={self.buffer_size}, LearningStarts={self.learning_starts}, BatchSize={self.optimized_batch_size}")
        if tensorboard_log_path is None:
            current_time_str = datetime.now().strftime("%Y%m%d-%H%M%S") # datetime 已導入
            self.tensorboard_log_path = str(LOGS_DIR / f"sac_tensorboard_logs_{current_time_str}")
            os.makedirs(self.tensorboard_log_path, exist_ok=True)
            logger.info(f"TensorBoard日誌將保存到: {self.tensorboard_log_path}")
        else:
            self.tensorboard_log_path = tensorboard_log_path
        self.agent = SAC(
            policy=self.policy_class, env=self.env, learning_rate=learning_rate,
            buffer_size=self.buffer_size, learning_starts=self.learning_starts, batch_size=self.optimized_batch_size,
            tau=tau, gamma=gamma, train_freq=(train_freq_steps, "step"), gradient_steps=gradient_steps,
            ent_coef=ent_coef, policy_kwargs=self.policy_kwargs, verbose=verbose, seed=seed,
            device=self.device, tensorboard_log=self.tensorboard_log_path
        )
        self.custom_objects = custom_objects if custom_objects is not None else {}
        self.custom_objects["policy_class"] = self.policy_class
        if "features_extractor_class" in self.policy_kwargs:
            self.custom_objects["features_extractor_class"] = self.policy_kwargs["features_extractor_class"]
        elif hasattr(self.policy_class, "features_extractor_class"):
             self.custom_objects["features_extractor_class"] = self.policy_class.features_extractor_class # type: ignore
        logger.info("SACAgentWrapper 初始化完成。")

    def _setup_device(self, device: Union[torch.device, str]) -> torch.device:
        """
        設置和優化計算設備
        
        Args:
            device: 指定的設備
            
        Returns:
            優化後的設備對象
        """
        if isinstance(device, str):
            if device == "auto":
                # 自動選擇最佳設備
                if torch.cuda.is_available():
                    device_obj = torch.device("cuda")
                    # 檢查GPU內存和性能
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"檢測到GPU: {gpu_name}, 內存: {gpu_memory:.1f}GB")
                    
                    # 設置GPU優化
                    self._optimize_gpu_settings()
                        
                else:
                    device_obj = torch.device("cpu")
                    logger.info("未檢測到CUDA，使用CPU")
            else:
                device_obj = torch.device(device)
        else:
            device_obj = device
            
        # 驗證設備可用性
        if device_obj.type == "cuda" and not torch.cuda.is_available():
            logger.warning("指定使用CUDA但CUDA不可用，回退到CPU")
            device_obj = torch.device("cpu")
        
        return device_obj
    
    def _optimize_gpu_settings(self):
        """優化GPU設置以提高訓練效率"""
        try:            # 清理GPU內存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
                # 啟用cuDNN基準模式以優化卷積操作
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                
                # 設置cuDNN確定性（可選，會稍微降低性能但提高可重現性）
                # torch.backends.cudnn.deterministic = True
                
                # 啟用TensorFloat-32 (TF32) 以提高Ampere架構GPU性能
                if hasattr(torch.backends.cuda, 'matmul'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                  # GPU內存配置已在config.py中統一設置
                
                logger.info("GPU優化設置已啟用：cuDNN基準模式、TF32、統一內存配置")
                
        except Exception as e:
            logger.warning(f"GPU優化設置時發生錯誤: {e}")

    def train(self, total_timesteps: int, callback: Optional[Union[BaseCallback, List[BaseCallback]]] = None,
              log_interval: int = 1, reset_num_timesteps: bool = True):
        logger.info(f"開始訓練 SAC 智能體，總步數: {total_timesteps}...")
        start_time = time.time()
        
        # 訓練前的GPU內存清理
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            logger.info(f"訓練開始時GPU內存使用: {initial_memory:.1f}MB")
        
        # 初始化混合精度訓練的梯度縮放器
        scaler = None
        if self.use_amp and self.device.type == 'cuda':
            scaler = torch.amp.GradScaler(device='cuda', enabled=self.use_amp)
            logger.info("使用混合精度訓練模式，已初始化梯度縮放器")
            
            # 設置更保守的損失縮放以避免溢出
            scaler._init_scale = 2**15  # 降低初始縮放因子
            
        try:
            # 添加AMP溢出監控
            if scaler is not None:
                # 在SAC的learn方法中，我們無法直接控制梯度縮放
                # 但可以在訓練前後檢查GPU內存和模型參數狀態
                logger.info("AMP模式：監控訓練前模型狀態...")
                self._check_model_health_amp()
            
            self.agent.learn(total_timesteps=total_timesteps, callback=callback, log_interval=log_interval,
                             reset_num_timesteps=reset_num_timesteps, progress_bar=False)
            
            training_duration = time.time() - start_time
            
            # 訓練後檢查
            if scaler is not None:
                logger.info("AMP模式：檢查訓練後模型狀態...")
                self._check_model_health_amp()
            
            # 訓練後的GPU內存統計
            if self.device.type == 'cuda':
                final_memory = torch.cuda.memory_allocated() / 1024**2  # MB
                max_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                logger.info(f"訓練完成時GPU內存使用: {final_memory:.1f}MB, 峰值: {max_memory:.1f}MB")
                torch.cuda.reset_peak_memory_stats()
            
            logger.info(f"智能體訓練完成。耗時: {training_duration:.2f} 秒。")
            
        except Exception as e:
            logger.error(f"智能體訓練過程中發生錯誤: {e}", exc_info=True)
            # 訓練失敗時清理GPU內存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            raise

    def _check_model_health_amp(self):
        """檢查AMP模式下模型的健康狀態"""
        if not self.use_amp or not hasattr(self.agent, 'policy'):
            return
            
        try:
            nan_count = 0
            inf_count = 0
            
            # 檢查策略網絡參數
            for name, param in self.agent.policy.named_parameters():
                if param.data is not None:
                    if torch.isnan(param.data).any():
                        nan_count += torch.isnan(param.data).sum().item()
                        logger.warning(f"AMP檢查：檢測到NaN在 {name}")
                    
                    if torch.isinf(param.data).any():
                        inf_count += torch.isinf(param.data).sum().item()
                        logger.warning(f"AMP檢查：檢測到Infinity在 {name}")
            
            if nan_count > 0 or inf_count > 0:
                logger.error(f"AMP健康檢查失敗：NaN數量={nan_count}, Infinity數量={inf_count}")
                
                # 如果檢測到數值問題，可以考慮重置某些層或降低學習率
                if nan_count > 100 or inf_count > 100:  # 如果問題很嚴重
                    logger.warning("檢測到嚴重的數值不穩定，建議檢查模型架構和學習率設置")
            else:
                logger.debug("AMP健康檢查通過：未檢測到數值異常")
                
        except Exception as e:
            logger.warning(f"AMP健康檢查過程中發生錯誤: {e}")

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