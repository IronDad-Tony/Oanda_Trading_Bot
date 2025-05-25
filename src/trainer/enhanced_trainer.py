# src/trainer/enhanced_trainer.py
"""
增強版訓練器 - 整合智能貨幣管理和自動數據下載
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import numpy as np
import time
import torch
import gc
import os

# 導入邏輯與其他模組相同
logger: logging.Logger = logging.getLogger("enhanced_trainer_module_init")
_logger_initialized_by_common_et = False

try:
    from common.logger_setup import logger as common_configured_logger
    logger = common_configured_logger
    _logger_initialized_by_common_et = True
    logger.debug("enhanced_trainer.py: Successfully imported logger from common.logger_setup.")
    
    from common.config import (
        TIMESTEPS, MAX_SYMBOLS_ALLOWED, ACCOUNT_CURRENCY, INITIAL_CAPITAL,
        OANDA_MARGIN_CLOSEOUT_LEVEL, TRADE_COMMISSION_PERCENTAGE, OANDA_API_KEY,
        ATR_PERIOD, STOP_LOSS_ATR_MULTIPLIER, MAX_ACCOUNT_RISK_PERCENTAGE,
        LOGS_DIR, DEVICE, USE_AMP
    )
    logger.info("enhanced_trainer.py: Successfully imported common.config values.")
    
    # 導入共享數據管理器
    from common.shared_data_manager import get_shared_data_manager
    logger.info("enhanced_trainer.py: Successfully imported shared data manager.")
    
    # 導入所需模組
    from data_manager.currency_manager import CurrencyDependencyManager, ensure_currency_data_for_trading
    from data_manager.mmap_dataset import UniversalMemoryMappedDataset
    from data_manager.instrument_info_manager import InstrumentInfoManager
    from data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols
    from environment.trading_env import UniversalTradingEnvV4
    from agent.sac_agent_wrapper import SACAgentWrapper
    from trainer.callbacks import UniversalCheckpointCallback
    logger.info("enhanced_trainer.py: Successfully imported other dependencies.")
    
except ImportError as e_initial_import_et:
    logger_temp_et = logging.getLogger("enhanced_trainer_fallback_initial")
    logger_temp_et.addHandler(logging.StreamHandler(sys.stdout))
    logger_temp_et.setLevel(logging.DEBUG)
    logger = logger_temp_et
    logger.warning(f"enhanced_trainer.py: Initial import failed: {e_initial_import_et}. Attempting path adjustment...")
    
    project_root_et = Path(__file__).resolve().parent.parent.parent
    if str(project_root_et) not in sys.path:
        sys.path.insert(0, str(project_root_et))
        logger.info(f"enhanced_trainer.py: Added project root to sys.path: {project_root_et}")
    
    try:
        from src.common.logger_setup import logger as common_logger_retry_et
        logger = common_logger_retry_et
        _logger_initialized_by_common_et = True
        logger.info("enhanced_trainer.py: Successfully re-imported common_logger after path adj.")
        
        from src.common.config import (
            TIMESTEPS, MAX_SYMBOLS_ALLOWED, ACCOUNT_CURRENCY, INITIAL_CAPITAL,
            OANDA_MARGIN_CLOSEOUT_LEVEL, TRADE_COMMISSION_PERCENTAGE, OANDA_API_KEY,
            ATR_PERIOD, STOP_LOSS_ATR_MULTIPLIER, MAX_ACCOUNT_RISK_PERCENTAGE,
            LOGS_DIR, DEVICE, USE_AMP
        )
        logger.info("enhanced_trainer.py: Successfully re-imported common.config after path adjustment.")
        
        # 導入共享數據管理器
        from src.common.shared_data_manager import get_shared_data_manager
        logger.info("enhanced_trainer.py: Successfully re-imported shared data manager.")
        
        from src.data_manager.currency_manager import CurrencyDependencyManager, ensure_currency_data_for_trading
        from src.data_manager.mmap_dataset import UniversalMemoryMappedDataset
        from src.data_manager.instrument_info_manager import InstrumentInfoManager
        from src.data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols
        from src.environment.trading_env import UniversalTradingEnvV4
        from src.agent.sac_agent_wrapper import SACAgentWrapper
        from src.trainer.callbacks import UniversalCheckpointCallback
        logger.info("enhanced_trainer.py: Successfully re-imported other dependencies after path adjustment.")
        
    except ImportError as e_retry_critical_et:
        logger.error(f"enhanced_trainer.py: Critical import error after path adjustment: {e_retry_critical_et}", exc_info=True)
        logger.warning("enhanced_trainer.py: Using fallback mode - some features may not work.")
        
        # 創建後備類
        class CurrencyDependencyManager:
            def __init__(self, **kwargs):
                pass
            def download_required_currency_data(self, *args, **kwargs):
                return True
        
        def ensure_currency_data_for_trading(*args, **kwargs):
            return True
        
        def get_shared_data_manager():
            """後備共享數據管理器"""
            class FallbackManager:
                def update_training_status(self, *args, **kwargs): pass
                def is_stop_requested(self): return False
                def request_stop(self): pass
                def add_training_metric(self, *args, **kwargs): pass
                def add_trade_record(self, *args, **kwargs): pass
            return FallbackManager()


class EnhancedUniversalTrainer:
    """
    增強版通用交易模型訓練器
    
    特點：
    1. 智能貨幣依賴管理 - 自動判斷並下載所需匯率對
    2. 自動數據準備 - 確保所有必需數據完整
    3. 完整的訓練流程 - 從數據到模型一站式
    4. 實時監控和回調 - 支持斷點續練
    """
    
    def __init__(self,
                 trading_symbols: List[str],
                 start_time: datetime,
                 end_time: datetime,
                 granularity: str = "S5",
                 timesteps_history: int = TIMESTEPS,
                 account_currency: str = ACCOUNT_CURRENCY,
                 initial_capital: float = float(INITIAL_CAPITAL),
                 max_episode_steps: Optional[int] = None,
                 total_timesteps: int = 10000,
                 save_freq: int = 1000,
                 eval_freq: int = 2000,
                 model_name_prefix: str = "sac_universal_trader",
                 streamlit_session_state=None):
        
        self.trading_symbols = sorted(list(set(trading_symbols)))
        self.start_time = start_time
        self.end_time = end_time
        self.granularity = granularity
        self.timesteps_history = timesteps_history
        self.account_currency = account_currency.upper()
        self.initial_capital = initial_capital
        self.max_episode_steps = max_episode_steps
        self.total_timesteps = total_timesteps
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.model_name_prefix = model_name_prefix
        self.streamlit_session_state = streamlit_session_state  # 用於更新Streamlit UI
        
        # 初始化共享數據管理器
        self.shared_data_manager = get_shared_data_manager()
        logger.info("已連接到共享數據管理器")
        
        # 設置GPU優化
        self._setup_gpu_optimization()
        
        # 生成基於參數的模型名稱
        self.model_identifier = self._generate_model_identifier()
        self.existing_model_path = self._find_existing_model()
        
        # 初始化組件
        self.currency_manager = CurrencyDependencyManager(account_currency)
        self.instrument_manager = InstrumentInfoManager(force_refresh=False)
        
        # 將在 prepare_data 中初始化
        self.dataset = None
        self.env = None
        self.agent = None
        self.callback = None
        self._stop_training = False  # 停止訓練的標誌
        
        # 訓練數據收集
        self.training_start_time = None
        
        logger.info(f"EnhancedUniversalTrainer 初始化完成")
        logger.info(f"交易symbols: {self.trading_symbols}")
        logger.info(f"時間範圍: {self.start_time} 到 {self.end_time}")
        logger.info(f"帳戶貨幣: {self.account_currency}")
        logger.info(f"模型標識符: {self.model_identifier}")
        if self.existing_model_path:
            logger.info(f"發現既有模型: {self.existing_model_path}")
        else:
            logger.info("未發現既有模型，將創建新模型")
    
    def _setup_gpu_optimization(self):
        """設置GPU優化配置"""
        try:
            if torch.cuda.is_available():
                # 檢查GPU信息
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
                
                logger.info(f"檢測到 {gpu_count} 個GPU設備")
                logger.info(f"當前使用GPU {current_device}: {gpu_name} ({gpu_memory:.1f}GB)")
                
                # 清理GPU內存
                torch.cuda.empty_cache()
                gc.collect()
                
                # 設置GPU內存管理
                torch.cuda.set_per_process_memory_fraction(0.85)  # 使用85%的GPU內存
                
                # 啟用cuDNN優化
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                
                # 啟用TensorFloat-32 (TF32) 以提高Ampere架構GPU性能
                if hasattr(torch.backends.cuda, 'matmul'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                
                # 設置環境變量以優化GPU內存分配
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
                
                # 如果啟用混合精度訓練
                if USE_AMP:
                    logger.info("混合精度訓練已啟用")
                    # 檢查GPU是否支持混合精度
                    if torch.cuda.get_device_capability(current_device)[0] >= 7:  # Volta架構或更新
                        logger.info("GPU支援混合精度訓練")
                    else:
                        logger.warning("GPU可能不完全支援混合精度訓練，但仍會嘗試使用")
                
                logger.info("GPU優化設置完成")
                
            else:
                logger.info("未檢測到CUDA，將使用CPU訓練")
                # CPU優化設置
                torch.set_num_threads(min(8, torch.get_num_threads()))  # 限制CPU線程數
                logger.info(f"CPU線程數設置為: {torch.get_num_threads()}")
                
        except Exception as e:
            logger.warning(f"GPU優化設置時發生錯誤: {e}")
    
    def _generate_model_identifier(self) -> str:
        """
        基於關鍵參數生成模型標識符
        
        Returns:
            模型標識符字符串
        """
        # 使用交易品種數量和時間步長作為主要參數
        max_symbols = len(self.trading_symbols)
        timestep = self.timesteps_history
        
        # 生成標識符：sac_model_symbols{數量}_timestep{步長}
        identifier = f"sac_model_symbols{max_symbols}_timestep{timestep}"
        return identifier
    
    def _find_existing_model(self) -> Optional[str]:
        """
        查找是否存在相同參數的模型
        
        Returns:
            既有模型路徑，如果不存在則返回None
        """
        try:
            # 檢查多個可能的位置
            search_paths = [
                Path("weights"),
                Path("logs"),
                Path(LOGS_DIR) if LOGS_DIR else None
            ]
            
            for search_path in search_paths:
                if search_path is None or not search_path.exists():
                    continue
                
                # 查找匹配的模型文件
                pattern = f"{self.model_identifier}*.zip"
                matching_files = list(search_path.glob(pattern))
                
                if matching_files:
                    # 返回最新的文件
                    latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
                    return str(latest_file)
            
            return None
            
        except Exception as e:
            logger.warning(f"查找既有模型時發生錯誤: {e}")
            return None
    
    def get_model_save_path(self, suffix: str = "") -> Path:
        """
        獲取模型保存路徑
        
        Args:
            suffix: 文件名後綴
            
        Returns:
            模型保存路徑
        """
        if suffix:
            filename = f"{self.model_identifier}_{suffix}.zip"
        else:
            filename = f"{self.model_identifier}.zip"
        
        save_dir = Path(LOGS_DIR) if LOGS_DIR else Path("logs")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        return save_dir / filename
    
    def prepare_data(self) -> bool:
        """
        準備訓練所需的所有數據
        
        Returns:
            是否成功準備所有數據
        """
        try:
            logger.info("開始準備訓練數據...")
            
            # 1. 轉換時間格式
            start_iso = format_datetime_for_oanda(self.start_time)
            end_iso = format_datetime_for_oanda(self.end_time)
            
            logger.info(f"時間範圍 (ISO): {start_iso} 到 {end_iso}")
            
            # 2. 下載交易symbols的數據
            logger.info(f"下載交易symbols數據: {self.trading_symbols}")
            manage_data_download_for_symbols(
                symbols=self.trading_symbols,
                overall_start_str=start_iso,
                overall_end_str=end_iso,
                granularity=self.granularity
            )
            
            # 3. 智能下載匯率對數據
            logger.info("分析並下載所需的匯率對數據...")
            currency_success = ensure_currency_data_for_trading(
                trading_symbols=self.trading_symbols,
                start_time_iso=start_iso,
                end_time_iso=end_iso,
                granularity=self.granularity,
                account_currency=self.account_currency
            )
            
            if not currency_success:
                logger.warning("匯率對數據下載可能不完整，但繼續進行...")
            
            # 4. 確定數據集需要包含的所有symbols (交易symbols + 匯率對)
            required_pairs = self.currency_manager.determine_required_currency_pairs(self.trading_symbols)
            
            # 過濾出實際存在的匯率對
            valid_currency_pairs = []
            for pair in required_pairs:
                details = self.instrument_manager.get_details(pair)
                if details:
                    valid_currency_pairs.append(pair)
            
            # 合併所有需要的symbols
            all_dataset_symbols = list(set(self.trading_symbols + valid_currency_pairs))
            logger.info(f"數據集將包含 {len(all_dataset_symbols)} 個symbols: {sorted(all_dataset_symbols)}")
            
            # 5. 創建數據集
            logger.info("創建內存映射數據集...")
            self.dataset = UniversalMemoryMappedDataset(
                symbols=all_dataset_symbols,
                start_time_iso=start_iso,
                end_time_iso=end_iso,
                granularity=self.granularity,
                timesteps_history=self.timesteps_history,
                force_reload=False
            )
            
            if len(self.dataset) == 0:
                logger.error("數據集為空！")
                return False
            
            logger.info(f"數據集創建成功，包含 {len(self.dataset)} 個時間步")
            
            # 6. 驗證匯率覆蓋
            is_complete, missing_pairs = self.currency_manager.validate_currency_coverage(
                self.trading_symbols, self.dataset.symbols
            )
            
            if not is_complete:
                logger.warning(f"匯率覆蓋不完整，缺失: {missing_pairs}")
                logger.warning("這可能會影響某些貨幣轉換的準確性")
            
            # 7. 顯示貨幣轉換信息
            conversion_info = self.currency_manager.get_currency_conversion_info(self.trading_symbols)
            logger.info("貨幣轉換路徑:")
            for symbol, info in conversion_info.items():
                logger.info(f"  {symbol}: {info['conversion_path']}")
            
            logger.info("數據準備完成！")
            return True
            
        except Exception as e:
            logger.error(f"數據準備失敗: {e}", exc_info=True)
            return False
    
    def setup_environment(self) -> bool:
        """
        設置交易環境
        
        Returns:
            是否成功設置環境
        """
        try:
            if self.dataset is None:
                logger.error("數據集未準備，請先調用 prepare_data()")
                return False
            
            logger.info("設置交易環境...")
            
            # 創建交易環境
            self.env = UniversalTradingEnvV4(
                dataset=self.dataset,
                instrument_info_manager=self.instrument_manager,
                active_symbols_for_episode=self.trading_symbols,
                initial_capital=self.initial_capital,
                max_episode_steps=self.max_episode_steps
            )
            
            logger.info(f"交易環境創建成功")
            logger.info(f"觀察空間: {self.env.observation_space}")
            logger.info(f"動作空間: {self.env.action_space}")
            
            return True
            
        except Exception as e:
            logger.error(f"環境設置失敗: {e}", exc_info=True)
            return False
    
    def setup_agent(self, load_model_path: Optional[str] = None) -> bool:
        """
        設置智能體
        
        Args:
            load_model_path: 可選的模型加載路徑，如果為None則自動檢查既有模型
            
        Returns:
            是否成功設置智能體
        """
        try:
            if self.env is None:
                logger.error("環境未設置，請先調用 setup_environment()")
                return False
            
            logger.info("設置SAC智能體...")
            
            # 創建SAC智能體包裝器
            # 創建向量化環境
            from stable_baselines3.common.vec_env import DummyVecEnv
            self.vec_env = DummyVecEnv([lambda: self.env])
            
            # 設置TensorBoard日誌路徑
            current_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.tensorboard_log_path = str(LOGS_DIR / f"sac_tensorboard_logs_{current_time_str}")
            
            # 創建SAC智能體包裝器，確保使用GPU優化設置
            self.agent = SACAgentWrapper(
                env=self.vec_env,
                verbose=1,
                tensorboard_log_path=self.tensorboard_log_path,
                device=DEVICE,
                use_amp=USE_AMP
            )
            
            # 決定要載入的模型路徑
            model_to_load = load_model_path or self.existing_model_path
            
            # 如果有模型路徑，加載模型
            if model_to_load:
                logger.info(f"載入既有模型: {model_to_load}")
                self.agent.load(model_to_load)
                
                # 更新Streamlit狀態
                if self.streamlit_session_state:
                    self.streamlit_session_state.loaded_existing_model = True
                    self.streamlit_session_state.loaded_model_path = model_to_load
                    self.streamlit_session_state.loaded_model_info = self._get_model_info(model_to_load)
            else:
                logger.info("創建新的模型")
                if self.streamlit_session_state:
                    self.streamlit_session_state.loaded_existing_model = False
                    self.streamlit_session_state.loaded_model_path = None
                    self.streamlit_session_state.loaded_model_info = None
            
            logger.info("SAC智能體創建成功")
            # 注意：SACAgentWrapper 可能沒有 get_model_parameter_count 方法
            try:
                if hasattr(self.agent, 'get_model_parameter_count'):
                    logger.info(f"模型參數數量: {self.agent.get_model_parameter_count()}")
                else:
                    logger.info("智能體已創建（參數數量信息不可用）")
            except Exception as e:
                logger.warning(f"無法獲取模型參數數量: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"智能體設置失敗: {e}", exc_info=True)
            return False
    
    def _get_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        獲取模型信息
        
        Args:
            model_path: 模型文件路徑
            
        Returns:
            模型信息字典
        """
        try:
            model_file = Path(model_path)
            if not model_file.exists():
                return {}
            
            stat = model_file.stat()
            return {
                'name': model_file.name,
                'path': str(model_file),
                'size_mb': stat.st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'max_symbols': len(self.trading_symbols),
                'timestep': self.timesteps_history,
                'model_identifier': self.model_identifier
            }
            
        except Exception as e:
            logger.warning(f"獲取模型信息時發生錯誤: {e}")
            return {}
    
    def setup_callbacks(self) -> bool:
        """
        設置訓練回調
        
        Returns:
            是否成功設置回調
        """
        try:
            if self.agent is None:
                logger.error("智能體未設置，請先調用 setup_agent()")
                return False
            
            logger.info("設置訓練回調...")
            
            # 使用模型標識符作為保存路徑，確保相同參數的訓練使用相同的文件
            save_dir = Path(LOGS_DIR) if LOGS_DIR else Path("logs")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 創建檢查點回調
            self.callback = UniversalCheckpointCallback(
                save_freq=self.save_freq,
                save_path=save_dir,
                name_prefix=self.model_identifier,  # 使用模型標識符而不是通用前綴
                eval_freq=self.eval_freq,
                n_eval_episodes=5,
                deterministic_eval=True,
                verbose=1,
                streamlit_session_state=self.streamlit_session_state
            )
            
            logger.info("訓練回調設置成功")
            return True
            
        except Exception as e:
            logger.error(f"回調設置失敗: {e}", exc_info=True)
            return False
    
    def train(self) -> bool:
        """
        執行訓練
        
        Returns:
            是否成功完成訓練
        """
        try:
            if not all([self.dataset, self.env, self.agent, self.callback]):
                logger.error("訓練組件未完全設置，請確保已調用所有setup方法")
                return False
            
            logger.info(f"開始訓練，總步數: {self.total_timesteps}")
            logger.info(f"保存頻率: 每 {self.save_freq} 步")
            logger.info(f"評估頻率: 每 {self.eval_freq} 步")
            
            # 重置停止標誌
            self._stop_training = False
            self.training_start_time = datetime.now()
            
            # 創建一個自定義回調來檢查停止標誌和更新訓練數據
            from stable_baselines3.common.callbacks import BaseCallback
            
            class TrainingMonitorCallback(BaseCallback):
                def __init__(self, trainer, verbose=0):
                    super().__init__(verbose)
                    self.trainer = trainer
                    self.last_update_time = time.time()
                    self.update_interval = 1.0  # 每1秒更新一次，避免過於頻繁
                    self.step_check_interval = 1  # 每1步檢查一次停止信號，提高響應性
                    self.last_step_check = 0
                    self.last_reward = 0.0
                    self.episode_count = 0
                
                def _on_step(self) -> bool:
                    # 每步都檢查停止信號，確保快速響應
                    if self.num_timesteps - self.last_step_check >= self.step_check_interval:
                        self.last_step_check = self.num_timesteps
                        
                        # 檢查共享數據管理器的停止信號
                        if self.trainer.shared_data_manager.is_stop_requested():
                            logger.info("檢測到共享數據管理器停止信號")
                            return False
                        
                        # 如果設置了停止標誌，返回False來停止訓練
                        if self.trainer._stop_training:
                            logger.info("檢測到停止訓練信號")
                            return False
                    
                    # 定期更新訓練進度
                    current_time = time.time()
                    if current_time - self.last_update_time >= self.update_interval:
                        self._update_training_metrics_safe()
                        self.last_update_time = current_time
                    
                    return True
                
                def _update_training_metrics_safe(self):
                    """線程安全的訓練指標更新"""
                    try:
                        # 計算訓練進度
                        progress = (self.num_timesteps / self.trainer.total_timesteps) * 100
                        
                        # 更新共享數據管理器狀態
                        self.trainer.shared_data_manager.update_training_status('running', progress)
                        
                        # 收集當前訓練數據
                        current_reward = 0.0
                        current_portfolio_value = self.trainer.initial_capital
                        actor_loss = 0.0
                        critic_loss = 0.0
                        l2_norm = 0.0
                        grad_norm = 0.0
                        
                        # 從環境獲取當前狀態
                        if hasattr(self.trainer.env, 'get_current_info'):
                            try:
                                env_info = self.trainer.env.get_current_info()
                                current_reward = env_info.get('episode_reward', 0.0)
                                current_portfolio_value = env_info.get('portfolio_value_ac', self.trainer.initial_capital)
                                
                                # 檢測新的episode
                                if current_reward != self.last_reward:
                                    self.last_reward = current_reward
                                    
                            except Exception as e:
                                logger.debug(f"獲取環境信息失敗: {e}")
                        
                        # 從logger獲取損失和範數數據
                        if self.logger is not None and hasattr(self.logger, 'name_to_value'):
                            try:
                                for key, value in self.logger.name_to_value.items():
                                    if isinstance(value, (int, float)):
                                        if 'loss' in key.lower():
                                            if 'actor' in key.lower():
                                                actor_loss = float(value)
                                            elif 'critic' in key.lower():
                                                critic_loss = float(value)
                                        elif 'norm' in key.lower():
                                            if 'l2' in key.lower():
                                                l2_norm = float(value)
                                            elif 'grad' in key.lower():
                                                grad_norm = float(value)
                            except Exception as e:
                                logger.debug(f"獲取logger數據失敗: {e}")
                        
                        # 更新共享數據管理器
                        self.trainer.shared_data_manager.add_training_metric(
                            step=self.num_timesteps,
                            reward=current_reward,
                            portfolio_value=current_portfolio_value,
                            actor_loss=actor_loss,
                            critic_loss=critic_loss,
                            l2_norm=l2_norm,
                            grad_norm=grad_norm
                        )
                        
                        # 模擬交易記錄（從環境獲取）
                        if hasattr(self.trainer.env, 'get_recent_trades'):
                            try:
                                recent_trades = self.trainer.env.get_recent_trades()
                                for trade in recent_trades:
                                    self.trainer.shared_data_manager.add_trade_record(
                                        symbol=trade.get('symbol', 'UNKNOWN'),
                                        action=trade.get('action', 'hold'),
                                        price=trade.get('price', 1.0),
                                        quantity=trade.get('quantity', 0.0),
                                        profit_loss=trade.get('profit_loss', 0.0)
                                    )
                            except Exception as e:
                                logger.debug(f"獲取交易記錄失敗: {e}")
                        
                        # 記錄訓練進度（每50步記錄一次）
                        if self.num_timesteps % 50 == 0:
                            logger.info(f"訓練進度: {progress:.1f}% ({self.num_timesteps}/{self.trainer.total_timesteps}), "
                                      f"獎勵: {current_reward:.3f}, 淨值: {current_portfolio_value:.2f}")
                        
                    except Exception as e:
                        logger.warning(f"更新訓練指標時發生錯誤: {e}")
            
            # 組合回調
            from stable_baselines3.common.callbacks import CallbackList
            monitor_callback = TrainingMonitorCallback(self)
            combined_callback = CallbackList([self.callback, monitor_callback])
            
            # 執行訓練前的GPU內存檢查
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
                logger.info(f"訓練開始前GPU內存使用: {initial_memory:.1f}MB")
            
            # 執行訓練
            self.agent.train(
                total_timesteps=self.total_timesteps,
                callback=combined_callback,
                log_interval=100
            )
            
            # 執行訓練後的GPU內存檢查
            if torch.cuda.is_available():
                final_memory = torch.cuda.memory_allocated() / 1024**2  # MB
                max_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                logger.info(f"訓練完成後GPU內存使用: {final_memory:.1f}MB, 峰值: {max_memory:.1f}MB")
                torch.cuda.reset_peak_memory_stats()
            
            # 檢查是否是因為停止信號而結束
            if self._stop_training or self.shared_data_manager.is_stop_requested():
                logger.info("訓練已被停止信號中斷")
                self.shared_data_manager.update_training_status('idle')
                return False
            
            logger.info("訓練完成！")
            
            # 更新共享數據管理器狀態
            self.shared_data_manager.update_training_status('completed', 100)
            
            # 保存最終模型，使用統一的命名策略
            final_model_path = self.get_model_save_path("final")
            self.agent.save(final_model_path)
            logger.info(f"最終模型已保存: {final_model_path}")
            
            return True
            
        except KeyboardInterrupt:
            logger.info("訓練被用戶中斷")
            # 保存當前模型
            if self.agent:
                interrupted_model_path = self.get_model_save_path("interrupted")
                self.agent.save(interrupted_model_path)
                logger.info(f"中斷時模型已保存: {interrupted_model_path}")
            return False
            
        except Exception as e:
            logger.error(f"訓練失敗: {e}", exc_info=True)
            # 更新共享數據管理器錯誤狀態
            self.shared_data_manager.update_training_status('error', error=str(e))
            return False
    
    def run_full_training_pipeline(self, load_model_path: Optional[str] = None) -> bool:
        """
        執行完整的訓練流程
        
        Args:
            load_model_path: 可選的模型加載路徑（用於斷點續練）
            
        Returns:
            是否成功完成整個流程
        """
        logger.info("=" * 60)
        logger.info("開始完整的訓練流程")
        logger.info("=" * 60)
        
        try:
            # 初始化共享數據管理器狀態
            self.shared_data_manager.update_training_status('running', 0)
            
            # 1. 準備數據
            if not self.prepare_data():
                logger.error("數據準備失敗，終止訓練")
                self.shared_data_manager.update_training_status('error', error="數據準備失敗")
                return False
            
            # 2. 設置環境
            if not self.setup_environment():
                logger.error("環境設置失敗，終止訓練")
                self.shared_data_manager.update_training_status('error', error="環境設置失敗")
                return False
            
            # 3. 設置智能體
            if not self.setup_agent(load_model_path):
                logger.error("智能體設置失敗，終止訓練")
                self.shared_data_manager.update_training_status('error', error="智能體設置失敗")
                return False
            
            # 4. 設置回調
            if not self.setup_callbacks():
                logger.error("回調設置失敗，終止訓練")
                self.shared_data_manager.update_training_status('error', error="回調設置失敗")
                return False
            
            # 5. 執行訓練
            success = self.train()
            
        except Exception as e:
            logger.error(f"完整訓練流程發生錯誤: {e}", exc_info=True)
            self.shared_data_manager.update_training_status('error', error=str(e))
            success = False
        
            # 6. 清理資源
            self.cleanup()
            
            if success:
                logger.info("=" * 60)
                logger.info("完整訓練流程成功完成！")
                logger.info("=" * 60)
            else:
                logger.warning("=" * 60)
                logger.warning("訓練流程未完全成功")
                logger.warning("=" * 60)
            
            return success
    
    def cleanup(self):
        """清理資源"""
        try:
            if self.env:
                self.env.close()
                logger.debug("環境已關閉")
            
            if self.dataset:
                self.dataset.close()
                logger.debug("數據集已關閉")
            
            # 清理GPU內存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.debug("GPU內存已清理")
                
        except Exception as e:
            logger.warning(f"清理資源時發生錯誤: {e}")
    
    def stop(self):
        """
        停止訓練
        
        設置停止標誌，訓練循環會在下一個步驟檢查到這個標誌並停止
        """
        logger.info("收到停止訓練請求")
        self._stop_training = True
        
        # 同時通知共享數據管理器
        self.shared_data_manager.request_stop()
        logger.info("已通知共享數據管理器停止訓練")
        
        # 如果有智能體正在訓練，嘗試中斷
        if hasattr(self, 'agent') and self.agent:
            try:
                # 嘗試保存當前模型
                current_model_path = self.get_model_save_path("interrupted")
                self.agent.save(current_model_path)
                logger.info(f"已保存中斷時的模型: {current_model_path}")
            except Exception as e:
                logger.warning(f"保存中斷模型時發生錯誤: {e}")
    
    def save_current_model(self, suffix: str = "checkpoint") -> Optional[str]:
        """
        保存當前模型
        
        Args:
            suffix: 模型文件名後綴
            
        Returns:
            保存的模型路徑，如果保存失敗則返回None
        """
        try:
            if self.agent is None:
                logger.warning("沒有可保存的模型（agent未初始化）")
                return None
            
            # 使用統一的命名策略
            model_path = self.get_model_save_path(suffix)
            
            self.agent.save(model_path)
            logger.info(f"模型已保存: {model_path}")
            
            return str(model_path)
            
        except Exception as e:
            logger.error(f"保存模型時發生錯誤: {e}", exc_info=True)
            return None


def create_training_time_range(days_back: int = 30) -> tuple[datetime, datetime]:
    """
    創建訓練時間範圍
    
    Args:
        days_back: 從現在往前多少天
        
    Returns:
        (start_time, end_time)
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)
    
    # 調整到工作日和交易時間
    # 這裡可以添加更複雜的邏輯來避開週末和節假日
    
    return start_time, end_time


if __name__ == "__main__":
    # 測試和示例用法
    logger.info("測試 EnhancedUniversalTrainer...")
    
    # 配置訓練參數
    test_symbols = ["EUR_USD", "USD_JPY", "GBP_USD"]
    start_time, end_time = create_training_time_range(days_back=7)  # 最近7天的數據
    
    logger.info(f"測試配置:")
    logger.info(f"  交易symbols: {test_symbols}")
    logger.info(f"  時間範圍: {start_time} 到 {end_time}")
    logger.info(f"  帳戶貨幣: {ACCOUNT_CURRENCY}")
    
    # 創建訓練器
    trainer = EnhancedUniversalTrainer(
        trading_symbols=test_symbols,
        start_time=start_time,
        end_time=end_time,
        granularity="S5",
        total_timesteps=1000,  # 測試用小數值
        save_freq=200,
        eval_freq=400,
        model_name_prefix="test_sac_enhanced"
    )
    
    # 只測試數據準備部分
    logger.info("測試數據準備...")
    success = trainer.prepare_data()
    
    if success:
        logger.info("數據準備測試成功！")
        
        # 測試環境設置
        logger.info("測試環境設置...")
        env_success = trainer.setup_environment()
        
        if env_success:
            logger.info("環境設置測試成功！")
            
            # 測試一個簡單的環境重置
            obs, info = trainer.env.reset()
            logger.info(f"環境重置成功，觀察形狀: {obs['features_from_dataset'].shape}")
            
        trainer.cleanup()
    else:
        logger.error("數據準備測試失敗")
    
    logger.info("EnhancedUniversalTrainer 測試完成")