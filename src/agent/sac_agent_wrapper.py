# src/agent/sac_agent_wrapper.py
"""
SAC智能體包裝器 (整合量子策略層)。
封裝Stable Baselines3的SAC智能體，提供簡化的接口，並集成TensorBoard日誌。
支援GPU加速訓練、混合精度訓練和量子策略層。
"""
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as sb3_logger_configure # 重命名以避免衝突
from gymnasium import spaces
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F  # 導入 functional 模組解決 F 未定義問題
from typing import Optional, Dict, Any, Type, Callable, Union, List, Tuple # <--- 添加 Tuple
from pathlib import Path
import time
import os
import numpy as np # <--- 在文件頂部導入 numpy
from datetime import datetime # <--- 在文件頂部導入 pandas
import pandas as pd # <--- 在文件頂部導入 pandas
import sys # 確保導入
import gc  # 垃圾回收
import warnings # To issue warnings
import logging # Ensure logging is imported

try:
    from src.agent.sac_policy import CustomSACPolicy
    from src.agent.high_level_integration_system import HighLevelIntegrationSystem
    from src.agent.strategy_innovation_module import create_strategy_innovation_module
    from src.agent.market_state_awareness_system import MarketStateAwarenessSystem
    from src.agent.meta_learning_optimizer import MetaLearningOptimizer
    from src.common.config import (
        DEVICE, SAC_LEARNING_RATE, SAC_BATCH_SIZE, SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR,
        SAC_LEARNING_STARTS_FACTOR, SAC_GAMMA, SAC_ENT_COEF,
        SAC_TRAIN_FREQ_STEPS, SAC_GRADIENT_STEPS, SAC_TAU,
        TIMESTEPS, LOGS_DIR, MAX_SYMBOLS_ALLOWED, USE_AMP # <--- 添加混合精度訓練支持
    )
    from src.common.logger_setup import logger
except ImportError:
    try:
        from src.agent.sac_policy import CustomSACPolicy
        from src.agent.high_level_integration_system import HighLevelIntegrationSystem
        from src.agent.strategy_innovation_module import create_strategy_innovation_module
        from src.agent.market_state_awareness_system import MarketStateAwarenessSystem
        from src.agent.meta_learning_optimizer import MetaLearningOptimizer
        from src.common.config import (
            DEVICE, SAC_LEARNING_RATE, SAC_BATCH_SIZE, SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR,
            SAC_LEARNING_STARTS_FACTOR, SAC_GAMMA, SAC_ENT_COEF,
            SAC_TRAIN_FREQ_STEPS, SAC_GRADIENT_STEPS, SAC_TAU,
            TIMESTEPS, LOGS_DIR, MAX_SYMBOLS_ALLOWED, USE_AMP
        )
        from src.common.logger_setup import logger
        logger.info("Direct run SACAgentWrapper: Successfully re-imported modules.")
    except ImportError as e_retry_wrapper:
        logger = logging.getLogger("sac_agent_wrapper_fallback")  # type: ignore
        logger.setLevel(logging.INFO)
        _ch_wrapper = logging.StreamHandler(sys.stdout)
        _ch_wrapper.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        if not logger.handlers:
            logger.addHandler(_ch_wrapper)
        logger.error(f"Direct run SACAgentWrapper: Critical import error: {e_retry_wrapper}", exc_info=True)
        CustomSACPolicy = None  # type: ignore
        HighLevelIntegrationSystem = None  # type: ignore
        create_strategy_innovation_module = None  # type: ignore
        MarketStateAwarenessSystem = None  # type: ignore
        MetaLearningOptimizer = None  # type: ignore
        DEVICE = "cpu"; SAC_LEARNING_RATE = 3e-4; SAC_BATCH_SIZE = 256; SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR = 1000
        SAC_LEARNING_STARTS_FACTOR = 10; SAC_GAMMA = 0.99; SAC_ENT_COEF = 'auto'
        SAC_TRAIN_FREQ_STEPS = 1; SAC_GRADIENT_STEPS = 1; SAC_TAU = 0.005; TIMESTEPS = 128
        LOGS_DIR = Path("./logs_fallback"); MAX_SYMBOLS_ALLOWED = 20; USE_AMP = False


# Attempt to import QuantumEnhancedTransformer, handling potential import errors
try:
    from src.models.enhanced_transformer import EnhancedTransformer as QuantumEnhancedTransformer
except ImportError as e:
    logger.warning(f"Importing QuantumEnhancedTransformer failed: {e}")
    QuantumEnhancedTransformer = None # Define as None if import fails

# Import the new model configuration
from configs.enhanced_model_config import ModelConfig as EnhancedModelConfig

class QuantumEnhancedSAC:
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
                 use_amp: bool = USE_AMP                ):
        self.env = env
        # 使用传入的 policy_class（默认为 CustomSACPolicy）
        self.policy_class = policy_class

        # 設置實例屬性
        self.use_amp = use_amp # This is the USE_AMP from common.config
          # 計算 buffer_size（如果沒有提供）
        if buffer_size is None:
            # 根據環境中的貨幣對數量和因子計算 buffer_size
            # 先嘗試從環境獲取 num_symbols，如果沒有則用默認值
            try:
                if hasattr(self.env, 'num_active_symbols_in_slots'):
                    num_symbols = self.env.num_active_symbols_in_slots
                elif hasattr(self.env, 'envs') and len(self.env.envs) > 0:
                    # DummyVecEnv case
                    first_env = self.env.envs[0]
                    if hasattr(first_env, 'num_tradable_symbols_this_episode'):
                        num_symbols = first_env.num_tradable_symbols_this_episode
                    elif hasattr(first_env, 'active_symbols_for_episode'):
                        num_symbols = len(first_env.active_symbols_for_episode)
                    else:
                        num_symbols = 5  # 默認值
                else:
                    num_symbols = 5  # 默認值
            except:
                num_symbols = 5  # 默認值
            
            self.buffer_size = max(10000, num_symbols * buffer_size_factor)
        else:
            self.buffer_size = buffer_size
            
        # 計算 learning_starts
        self.learning_starts = max(1000, self.buffer_size // learning_starts_factor)
          # 設置優化的批次大小
        self.optimized_batch_size = min(batch_size, self.buffer_size // 4)        # 設置 TensorBoard 日誌路徑，使用統一的目錄結構
        if tensorboard_log_path is None:
            # 使用統一的 TensorBoard 目錄，直接指向根目錄以避免層次過深
            self.tensorboard_log_path = str(LOGS_DIR / "tensorboard")
            # 確保目錄存在
            os.makedirs(self.tensorboard_log_path, exist_ok=True)
            
            # 不再創建子目錄，直接使用主目錄
            self.session_subdir = ""  # 空字符串表示直接使用根目錄
            logger.info(f"TensorBoard 將記錄到: {self.tensorboard_log_path}")
        else:
            self.tensorboard_log_path = tensorboard_log_path
            self.session_subdir = ""        # 準備 policy_kwargs，合併用戶提供的和默認的設置
        from src.agent.enhanced_feature_extractor import EnhancedTransformerFeatureExtractor
        default_policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], qf=[256, 256]),
            features_extractor_class=EnhancedTransformerFeatureExtractor,
            features_extractor_kwargs={"enhanced_transformer_output_dim_per_symbol": 128},
        )
        
        # 如果用戶提供了 policy_kwargs，則合併
        if policy_kwargs is not None:
            default_policy_kwargs.update(policy_kwargs)
        self.policy_kwargs = default_policy_kwargs        # 優化設備配置
        self.device = self._setup_device(device)
        
        logger.info(f"量子增強SAC: 使用設備 {self.device}, 混合精度訓練: {self.use_amp}")
        logger.info(f"Buffer size: {self.buffer_size}, Learning starts: {self.learning_starts}, Batch size: {self.optimized_batch_size}")
        
        # 創建 SAC 智能體，直接使用 TensorBoard 根目錄
        # 使用主 TensorBoard 目錄，不創建額外的子目錄
        full_tensorboard_path = self.tensorboard_log_path
        
        self.agent = SAC(
            policy=self.policy_class, env=self.env, learning_rate=learning_rate,
            buffer_size=self.buffer_size, learning_starts=self.learning_starts,
            batch_size=self.optimized_batch_size, tau=tau, gamma=gamma,
            train_freq=(train_freq_steps, "step"), gradient_steps=gradient_steps,
            ent_coef=ent_coef, policy_kwargs=self.policy_kwargs,
            verbose=verbose, seed=seed, device=self.device,
            tensorboard_log=full_tensorboard_path
        )        # CustomSACPolicy 在內部已使用 TransformerFeatureExtractor，無需手動替換        # Initialize High-Level Integration System for Phase 5
        try:
            if HighLevelIntegrationSystem is not None:
                # Create real components for integration system
                real_strategy_innovation = self._create_strategy_innovation()
                real_market_state_awareness = self._create_market_state_awareness()
                real_meta_learning_optimizer = self._create_meta_learning_optimizer()
                
                # Create required placeholder components
                from src.agent.high_level_integration_system import (
                    DynamicPositionManager, 
                    AnomalyDetector, 
                    EmergencyStopLoss
                )
                
                position_manager = DynamicPositionManager(feature_dim=768)
                anomaly_detector = AnomalyDetector(input_dim=768)
                emergency_stop_loss_system = EmergencyStopLoss()
                  # Create config with feature_dim and other settings
                integration_config = {
                    'feature_dim': 768,
                    'enable_dynamic_adaptation': True,
                    'expected_maml_input_dim': 768,
                    'num_maml_tasks': 5,
                    'maml_shots': 5
                }
                
                self.high_level_integration = HighLevelIntegrationSystem(
                    strategy_innovation_module=real_strategy_innovation,
                    market_state_awareness_system=real_market_state_awareness,
                    meta_learning_optimizer=real_meta_learning_optimizer,
                    position_manager=position_manager,
                    anomaly_detector=anomaly_detector,
                    emergency_stop_loss_system=emergency_stop_loss_system,
                    config=integration_config
                )
                logger.info("✅ HighLevelIntegrationSystem initialized successfully")
            else:
                self.high_level_integration = None
                logger.warning("⚠️ HighLevelIntegrationSystem not available")
        except Exception as e:
            logger.error(f"❌ Failed to initialize HighLevelIntegrationSystem: {e}")
            self.high_level_integration = None

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
        try:
            # 清理GPU內存
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
        """修復版訓練方法"""
        logger.info(f"開始訓練 SAC 智能體，總步數: {total_timesteps}...")
        
        # 修復1: 確保 learning_starts 合理
        if hasattr(self.agent, 'learning_starts'):
            original_learning_starts = self.agent.learning_starts
            # 設置為較小的值以確保訓練能夠開始
            self.agent.learning_starts = min(self.agent.learning_starts, max(100, total_timesteps // 10))
            logger.info(f"調整 learning_starts: {original_learning_starts} -> {self.agent.learning_starts}")
        
        # 修復2: 確保模型在訓練模式
        if hasattr(self.agent, 'policy'):
            self.agent.policy.train()
            if hasattr(self.agent.policy, 'actor'):
                self.agent.policy.actor.train()
            if hasattr(self.agent.policy, 'critic'):
                self.agent.policy.critic.train()
            logger.info("設置模型為訓練模式")
        
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
            
            # 修復3: 訓練後檢查參數更新
            if hasattr(self.agent, 'num_timesteps'):
                logger.info(f"實際訓練步數: {self.agent.num_timesteps}")
            
        except Exception as e:
            logger.error(f"智能體訓練過程中發生錯誤: {e}", exc_info=True)
            # 訓練失敗時清理GPU內存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            raise
            self.agent.policy.train()
            if hasattr(self.agent.policy, 'actor'):
                self.agent.policy.actor.train()
            if hasattr(self.agent.policy, 'critic'):
                self.agent.policy.critic.train()
            logger.info("設置模型為訓練模式")
        
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
            
            # 修復3: 訓練後檢查參數更新
            if hasattr(self.agent, 'num_timesteps'):
                logger.info(f"實際訓練步數: {self.agent.num_timesteps}")
            
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

    def predict(self, observation: np.ndarray, state: Optional[Tuple[np.ndarray, ...]] = None,
                episode_start: Optional[np.ndarray] = None, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        return self.agent.predict(observation, state=state, episode_start=episode_start, deterministic=deterministic)
    
    def select_action(self, state_dict: Dict[str, np.ndarray], market_volatility: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用量子策略層選擇動作 (處理多維度輸入)
        
        Args:
            state_dict: 狀態字典 (包含時間序列特徵)
            market_volatility: 市場波動率數組 (每個交易對)
            
        Returns:
            action: 選擇的動作
            amplitudes: 策略振幅分布
        """
        # 將狀態字典轉換為張量
        features = torch.FloatTensor(state_dict['features_from_dataset']).to(self.device)
        batch_size, num_symbols, timesteps, features_dim = features.shape
        
        # 展平時間序列特徵
        flat_features = features.view(batch_size * num_symbols, timesteps * features_dim)
          # 創建波動率張量
        volatility_tensor = torch.FloatTensor(market_volatility).to(self.device).view(-1, 1)
        
        with torch.no_grad():
            action_tensor, amplitudes = self.quantum_policy.forward_compatible(flat_features, volatility_tensor)
        
        # 重塑動作張量
        action = action_tensor.view(batch_size, num_symbols, -1).cpu().numpy()
        amplitudes = amplitudes.view(batch_size, num_symbols, -1).cpu().numpy()
        
        return action, amplitudes
    
    def update_amplitudes(self, rewards: np.ndarray):
        """
        根據策略表現更新振幅 (批量處理)
        
        Args:
            rewards: 各策略在最近episode的累積獎勵 (形狀: [batch_size, num_strategies])
        """
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        probs = F.softmax(self.quantum_policy.amplitudes, dim=0)        # 計算策略損失
        loss = -torch.mean(torch.sum(torch.log(probs) * rewards_tensor, dim=1))
        
        # 優化振幅參數
        self.quantum_policy.quantum_annealing_step(rewards_tensor)

    def save(self, path: Union[str, Path]):
        """保存智能體模型"""
        try:
            # 為量子策略層添加臨時優化器屬性
            if hasattr(self.agent.policy, 'quantum_policy_layer'):
                self.agent.policy.quantum_policy_layer.optimizer = self.agent.policy.quantum_policy_layer.quantum_optimizer
                
            self.agent.save(path)
            logger.info(f"智能體模型已保存到: {path}")
        except Exception as e:
            logger.error(f"保存智能體模型失敗: {e}", exc_info=True)
            raise
        finally:
            # 移除臨時優化器屬性
            if hasattr(self.agent.policy, 'quantum_policy_layer') and hasattr(self.agent.policy.quantum_policy_layer, 'optimizer'):
                del self.agent.policy.quantum_policy_layer.optimizer

    def load(self, path: Union[str, Path], env: Optional[DummyVecEnv] = None):
        try:
            self.agent = SAC.load(path, env=env or self.env, custom_objects=self.custom_objects, device=self.agent.device)
            logger.info(f"智能體模型已從 {path} 加載。")
            
            # 使用統一的 TensorBoard 目錄，不創建新的時間戳目錄
            self.session_subdir = ""
            logger.info(f"模型載入後，TensorBoard 將記錄到: {self.tensorboard_log_path}")
              # 重新配置 TensorBoard 日誌路徑
            if hasattr(self.agent, 'tensorboard_log') and self.agent.tensorboard_log:
                # 如果 agent 已經有 TensorBoard 配置，更新它
                sb3_logger = sb3_logger_configure(self.tensorboard_log_path, ["stdout", "csv", "tensorboard"])
                self.agent.set_logger(sb3_logger)
        except Exception as e:
            logger.error(f"加載智能體模型失敗: {e}", exc_info=True)
            raise

    def get_policy_parameters(self):
        if self.agent and self.agent.policy:        return self.agent.policy.parameters()
        return None
    
    def get_feature_extractor_parameters(self):
        if self.agent and self.agent.policy and hasattr(self.agent.policy, 'features_extractor') \
           and self.agent.policy.features_extractor is not None:
            # 檢查增強版Transformer特徵提取器
            if hasattr(self.agent.policy.features_extractor, 'enhanced_transformer'):
                return self.agent.policy.features_extractor.enhanced_transformer.parameters() # type: ignore
            # 檢查原版Transformer特徵提取器
            elif hasattr(self.agent.policy.features_extractor, 'transformer'):
                return self.agent.policy.features_extractor.transformer.parameters() # type: ignore
        return None

    def _create_strategy_innovation(self):
        """Create a real strategy innovation module"""
        try:
            # Use the factory function to create the real StrategyInnovationModule
            # Note: Don't pass config_adapter as the factory function creates one automatically
            # hidden_dim must be divisible by num_heads (24), so use 768 instead of 512
            strategy_innovation = create_strategy_innovation_module(
                input_dim=768,  # Market data dimension
                hidden_dim=768,  # Must be divisible by num_heads (24): 768 % 24 = 0
                population_size=10,
                max_generations=50
            )
            logger.info("Created real StrategyInnovationModule")
            return strategy_innovation
        except Exception as e:
            logger.error(f"Failed to create real StrategyInnovationModule: {e}")
            # Fallback to mock if real component fails
            return self._create_mock_strategy_innovation()
    
    def _create_mock_strategy_innovation(self):
        """Create a mock strategy innovation module (fallback)"""
        class MockStrategyInnovationModule(nn.Module):
            def __init__(self, input_dim=768, output_dim=256):
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.innovation_network = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, output_dim)
                )
                
            def forward(self, market_data, existing_strategies=None):
                batch_size = market_data.size(0)
                innovation_output = self.innovation_network(market_data)
                return {
                    'task_batches': innovation_output,
                    'generated_strategies': innovation_output,
                    'strategy_confidence': torch.rand(batch_size, 1)
                }
        
        return MockStrategyInnovationModule()

    def _create_market_state_awareness(self):
        """Create a real market state awareness system"""
        try:
            # Create the real MarketStateAwarenessSystem
            # Note: MarketStateAwarenessSystem expects input_dim and num_strategies, not hidden_dim
            market_state_awareness = MarketStateAwarenessSystem(
                input_dim=768,  # Market data dimension - correct parameter name
                num_strategies=5,  # Number of trading strategies
                enable_real_time_monitoring=True            )
            logger.info("Created real MarketStateAwarenessSystem")
            return market_state_awareness
        except Exception as e:
            logger.error(f"Failed to create real MarketStateAwarenessSystem: {e}")
            # Fallback to mock if real component fails
            return self._create_mock_market_state_awareness()

    def _create_mock_market_state_awareness(self):
        """Create a mock market state awareness system (fallback)"""
        class MockMarketStateAwareness(nn.Module):
            def __init__(self, input_dim=768, output_dim=256):
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.state_network = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, output_dim)
                )
                
            def forward(self, market_data):
                batch_size = market_data.size(0)
                state_output = self.state_network(market_data)
                return {
                    'market_state': {
                        'regime': torch.randint(0, 4, (batch_size,)),
                        'volatility_level': torch.rand(batch_size, 1),
                        'trend_strength': torch.rand(batch_size, 1)
                    },
                    'state_features': state_output,
                    'confidence': torch.rand(batch_size, 1)
                }
        
        return MockMarketStateAwareness()

    def _create_meta_learning_optimizer(self):
        """Create a real meta-learning optimizer"""
        try:
            # Create a dummy model for the MetaLearningOptimizer since it requires a model
            # This will be a simple neural network that can be used for meta-learning
            dummy_model = nn.Sequential(
                nn.Linear(768, 256),  # Input feature dimension
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)  # Output dimension for task prediction
            )
            
            # Create the real MetaLearningOptimizer
            # Note: MetaLearningOptimizer requires a model parameter
            meta_learning_optimizer = MetaLearningOptimizer(
                model=dummy_model,  # Pass the dummy model instead of None
                feature_dim=768,  # Market data dimension
                adaptation_dim=256,  # Adaptation layer dimension
                inner_lr=0.01,  # Inner loop learning rate
                outer_lr=0.001,  # Outer loop learning rate
                inner_steps=5  # Number of inner loop steps
            )
            logger.info("Created real MetaLearningOptimizer with dummy model")
            return meta_learning_optimizer
        except Exception as e:
            logger.error(f"Failed to create real MetaLearningOptimizer: {e}")
            # Fallback to mock if real component fails
            return self._create_mock_meta_learning_optimizer()

    def _create_mock_meta_learning_optimizer(self):
        """Create a mock meta-learning optimizer (fallback)"""
        class MockMetaLearningOptimizer(nn.Module):
            def __init__(self, input_dim=768, output_dim=256):
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.meta_network = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, output_dim)
                )
                
            def forward(self, task_data, strategy_data=None):
                batch_size = task_data.size(0)
                meta_output = self.meta_network(task_data)
                return {
                    'meta_gradients': meta_output,
                    'adaptation_params': meta_output,
                    'learning_rate_adjustments': torch.rand(batch_size, 1)
                }
        
        return MockMetaLearningOptimizer()

    def process_with_high_level_integration(self,
                                           market_data: torch.Tensor,
                                           position_data: Optional[Dict[str, torch.Tensor]] = None,
                                           portfolio_metrics: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Process market data through the High-Level Integration System
        
        Args:
            market_data: Market data tensor (expected to be the primary features)
            position_data: Position data dictionary
            portfolio_metrics: Portfolio metrics tensor
            
        Returns:
            Dictionary containing integration results
        """
        if self.high_level_integration is None:
            logger.warning("HighLevelIntegrationSystem not available, skipping high-level processing")
            return {}
        
        try:
            # Construct the market_data_raw dictionary
            # The HighLevelIntegrationSystem._get_tensor_from_market_data will look for 'features_768', 'features', etc.
            # We should pass the main market_data tensor with a common key.
            # Other data like position_data and portfolio_metrics are also expected to be in this dict.
            market_data_raw_dict = {
                "features": market_data, # Assuming market_data is the primary feature tensor
                "current_positions": position_data if position_data is not None else [],
                "portfolio_metrics": portfolio_metrics if portfolio_metrics is not None else {}
            }
            
            # Process through the high-level integration system
            integration_results = self.high_level_integration.process_market_data(
                market_data_raw=market_data_raw_dict
            )
            
            # Log key metrics
            if 'system_health' in integration_results:
                health_score = integration_results['system_health'].get('health_score', 'unknown')
                logger.debug(f"System health score: {health_score}")
            
            if 'emergency_status' in integration_results:
                emergency_status = integration_results['emergency_status']
                if emergency_status.get('emergency_active', False):
                    logger.warning(f"Emergency conditions detected: {emergency_status}")
            
            return integration_results
            
        except Exception as e:
            logger.error(f"Error in high-level integration processing: {e}")
            return {}

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
        "features_extractor_kwargs": dict(enhanced_transformer_output_dim_per_symbol=128),
        "net_arch": dict(pi=[64, 64], qf=[64, 64])
    }
    try:
        logger.info("初始化 SACAgentWrapper...")
        agent_wrapper = QuantumEnhancedSAC(env=dummy_vec_env_main, policy_kwargs=test_policy_kwargs_main, batch_size=32, buffer_size_factor=10, learning_starts_factor=2, verbose=0)
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