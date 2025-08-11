# src/trainer/universal_trainer.py
"""
Universal Trading Model Trainer
整合了智能貨幣管理、自動數據下載、改進的模型識別和保存邏輯，
以及用於Streamlit UI的訓練進度監控功能。
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

logger: logging.Logger = logging.getLogger("universal_trainer_module_init")
_logger_initialized_by_common_ut = False

# Flag to prevent duplicate import logging
_import_logged = False

try:
    from oanda_trading_bot.training_system.common.logger_setup import logger as common_configured_logger
    logger = common_configured_logger
    _logger_initialized_by_common_ut = True
    if not _import_logged:
        logger.debug("universal_trainer.py: Successfully imported logger from common.logger_setup.")
        _import_logged = True
    
    from oanda_trading_bot.training_system.common.config import (
        TIMESTEPS, MAX_SYMBOLS_ALLOWED, ACCOUNT_CURRENCY, INITIAL_CAPITAL,
        OANDA_MARGIN_CLOSEOUT_LEVEL, TRADE_COMMISSION_PERCENTAGE, OANDA_API_KEY,
                WEIGHTS_DIR, LOGS_DIR, DEVICE, USE_AMP,
        TRAINER_SAVE_FREQ_STEPS, TRAINER_EVAL_FREQ_STEPS,
        # 不 import 不存在的 *_DEFAULT
    )
    if not _import_logged:
        logger.info("universal_trainer.py: Successfully imported common.config values.")
        _import_logged = True
    
    from oanda_trading_bot.training_system.common.shared_data_manager import get_shared_data_manager
    if not _import_logged:
        logger.info("universal_trainer.py: Successfully imported shared data manager.")
    
    from oanda_trading_bot.training_system.data_manager.currency_manager import CurrencyDependencyManager, ensure_currency_data_for_trading
    from oanda_trading_bot.training_system.data_manager.mmap_dataset import UniversalMemoryMappedDataset
    from oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager
    from oanda_trading_bot.training_system.data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols
    from oanda_trading_bot.training_system.environment.trading_env import UniversalTradingEnvV4
    from oanda_trading_bot.training_system.agent.sac_agent_wrapper import QuantumEnhancedSAC
    from oanda_trading_bot.training_system.agent.enhanced_feature_extractor import EnhancedTransformerFeatureExtractor
    from oanda_trading_bot.training_system.trainer.callbacks import UniversalCheckpointCallback  # callback 定义在 src/trainer/callbacks.py
    if not _import_logged:
        logger.info("universal_trainer.py: Successfully imported other dependencies.")
        _import_logged = True
    
except ImportError as e_initial_import_ut:
    logger_temp_ut = logging.getLogger("universal_trainer_fallback_initial")
    logger_temp_ut.addHandler(logging.StreamHandler(sys.stdout))
    logger_temp_ut.setLevel(logging.DEBUG)
    logger = logger_temp_ut
    if not _import_logged:
        logger.warning(f"universal_trainer.py: Initial import failed: {e_initial_import_ut}. Attempting path adjustment.")
        _import_logged = True
    
    try:
        from oanda_trading_bot.training_system.common.logger_setup import logger as common_logger_retry_ut
        logger = common_logger_retry_ut
        _logger_initialized_by_common_ut = True
        if not _import_logged:
            logger.info("universal_trainer.py: Successfully re-imported common_logger after path adj.")
            _import_logged = True
        
        from oanda_trading_bot.training_system.common.config import (
            TIMESTEPS, MAX_SYMBOLS_ALLOWED, ACCOUNT_CURRENCY, INITIAL_CAPITAL,
            OANDA_MARGIN_CLOSEOUT_LEVEL, TRADE_COMMISSION_PERCENTAGE, OANDA_API_KEY,
            WEIGHTS_DIR, LOGS_DIR, DEVICE, USE_AMP,
            TRAINER_SAVE_FREQ_STEPS, TRAINER_EVAL_FREQ_STEPS,
            # 不 import 不存在的 *_DEFAULT
        )
        if not _import_logged:
            logger.info("universal_trainer.py: Successfully re-imported common.config after path adjustment.")
        
        from oanda_trading_bot.training_system.common.shared_data_manager import get_shared_data_manager
        if not _import_logged:
            logger.info("universal_trainer.py: Successfully re-imported shared data manager.")
        
        from oanda_trading_bot.training_system.data_manager.currency_manager import CurrencyDependencyManager, ensure_currency_data_for_trading
        from oanda_trading_bot.training_system.data_manager.mmap_dataset import UniversalMemoryMappedDataset
        from oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager
        from oanda_trading_bot.training_system.data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols
        from oanda_trading_bot.training_system.environment.trading_env import UniversalTradingEnvV4
        from oanda_trading_bot.training_system.agent.sac_agent_wrapper import QuantumEnhancedSAC
        from oanda_trading_bot.training_system.agent.enhanced_feature_extractor import EnhancedTransformerFeatureExtractor
        from oanda_trading_bot.training_system.trainer.callbacks import UniversalCheckpointCallback
        if not _import_logged:
            logger.info("universal_trainer.py: Successfully re-imported other dependencies after path adjustment.")
            _import_logged = True
        
    except ImportError as e_retry_critical_ut:
        logger.error(f"universal_trainer.py: Critical import error after path adjustment: {e_retry_critical_ut}", exc_info=True)
        logger.warning("universal_trainer.py: Using fallback mode - some features may not work.")        # Fallback classes for critical imports
        class CurrencyDependencyManager:
            def __init__(self, account_currency=None, **kwargs):
                pass
            def download_required_currency_data(self, *args, **kwargs):
                return True
        def ensure_currency_data_for_trading(*args, **kwargs):
            trading_symbols = kwargs.get('trading_symbols', [])
            return True, set(trading_symbols)
        class UniversalMemoryMappedDataset:
            def __init__(self, **kwargs): pass
            def is_valid(self): return True
            def __len__(self): return 1000
        class InstrumentInfoManager:
            def __init__(self, **kwargs): pass
        def manage_data_download_for_symbols(*args, **kwargs): pass
        class UniversalTradingEnvV4:
            def __init__(self, **kwargs):
                self.observation_space = None
                self.action_space = None
            def close(self): pass
        class QuantumEnhancedSAC:
            def __init__(self, **kwargs): pass
            def load(self, path, **kwargs): pass
            def save(self, path): pass
            def learn(self, **kwargs): pass
            def train(self, **kwargs): pass
        class UniversalCheckpointCallback:
            def __init__(self, **kwargs): pass
        class EnhancedTransformerFeatureExtractor: pass
        
        # Fallback for config
        TIMESTEPS, MAX_SYMBOLS_ALLOWED, ACCOUNT_CURRENCY, INITIAL_CAPITAL = 128, 20, "AUD", 100000
        OANDA_MARGIN_CLOSEOUT_LEVEL, TRADE_COMMISSION_PERCENTAGE = 0.5, 0.0001
        OANDA_API_KEY, WEIGHTS_DIR, LOGS_DIR, DEVICE, USE_AMP = None, Path("weights"), Path("logs"), "cpu", False
        TRAINER_SAVE_FREQ_STEPS, TRAINER_EVAL_FREQ_STEPS = 20000, 10000
        # fallback 也只用這些，不用 *_DEFAULT


class UniversalTrainer:
    """
    通用交易模型訓練器
    
    用於管理數據準備、環境設置、代理訓練及結果監控的端到端管道。
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
                 total_timesteps: int = 1_000_000,
                 save_freq: int = TRAINER_SAVE_FREQ_STEPS,
                 eval_freq: int = TRAINER_EVAL_FREQ_STEPS,
                 model_name_prefix: str = "sac_universal_trader",
                 # Streamlit UI組件，用於更新前端進度
                 streamlit_progress_bar=None,
                 streamlit_status_text=None,
                 streamlit_session_state=None,
                 # 新增訓練參數配置
                 risk_percentage: float = 5.0,
                 atr_stop_loss_multiplier: float = 2.0,
                 max_position_percentage: float = 10.0,
                 custom_atr_period: int = 14): # ATR_PERIOD default to 14 if not provided via UI
        
        self.trading_symbols = sorted(list(set(trading_symbols)))
        # 確保選擇的交易對數量不超過配置中允許的最大值
        if len(self.trading_symbols) > MAX_SYMBOLS_ALLOWED:
            logger.error(f"選擇的交易對數量 ({len(self.trading_symbols)}) 超出了 MAX_SYMBOLS_ALLOWED ({MAX_SYMBOLS_ALLOWED})。")
            raise ValueError(f"交易對數量超出限制：({len(self.trading_symbols)} > {MAX_SYMBOLS_ALLOWED})")
            
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
        
        # Streamlit UI組件
        # Streamlit UI組件不再直接在此處使用，通過shared_data_manager更新
        self.streamlit_progress_bar = None
        self.streamlit_status_text = None
        self.streamlit_session_state = None
        
        # 新增訓練參數配置存儲
        self.risk_percentage = risk_percentage / 100.0
        self.atr_stop_loss_multiplier = atr_stop_loss_multiplier
        self.max_position_percentage = max_position_percentage / 100.0
        self.custom_atr_period = custom_atr_period
          # 初始化共享數據管理器
        self.shared_data_manager = get_shared_data_manager()
        logger.info("Connected to shared data manager")
        
        # Set the actual initial capital in the shared data manager for accurate return calculations
        self.shared_data_manager.set_actual_initial_capital(self.initial_capital)
        
        # 設置GPU優化
        self._setup_gpu_optimization()
        
        # 生成基於參數的模型識別符，包含 MAX_SYMBOLS_ALLOWED
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
        self._stop_training = False  # 訓練停止標誌
        
        # 訓練數據收集
        self.training_start_time = None
        self.current_training_step = 0 # 新增，用於訓練進度顯示
        
        logger.info(f"UniversalTrainer 初始化完成")
        logger.info(f"交易品種: {self.trading_symbols}")
        logger.info(f"時間範圍: {self.start_time} 至 {self.end_time}")
        logger.info(f"帳戶貨幣: {self.account_currency}")
        logger.info(f"模型識別符: {self.model_identifier}")
        if self.existing_model_path:
            logger.info(f"找到現有模型: {self.existing_model_path}")
        else:
            logger.info("未找到現有模型，將創建新模型")
    
    def _setup_gpu_optimization(self):
        """設置GPU優化配置"""
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
                
                logger.info(f"Detected {gpu_count} GPU devices")
                logger.info(f"Using GPU {current_device}: {gpu_name} ({gpu_memory:.1f}GB)")
                torch.cuda.empty_cache()
                gc.collect()
                
                # GPU optimization settings
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                
                if hasattr(torch.backends.cuda, 'matmul'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                
                if USE_AMP:
                    logger.info("Mixed precision training enabled")
                    if torch.cuda.get_device_capability(current_device)[0] >= 7:
                        logger.info("GPU supports mixed precision training")
                    else:
                        logger.warning("GPU may not fully support mixed precision, but will try to use it")
                
                logger.info("GPU optimization settings applied")
                
            else:
                logger.info("CUDA not detected, using CPU for training")
                torch.set_num_threads(min(8, torch.get_num_threads()))
                logger.info(f"CPU threads set to: {torch.get_num_threads()}")
                
        except Exception as e:
            logger.warning(f"GPU optimization setup error: {e}")
    
    def _generate_model_identifier(self) -> str:
        """
        只根據MAX_SYMBOLS_ALLOWED產生唯一模型識別符
        """
        # 只用MAX_SYMBOLS_ALLOWED，不考慮實際選取的symbol數量
        identifier = f"sac_model_symbols{MAX_SYMBOLS_ALLOWED}"
        return identifier
    
    def _find_existing_model(self) -> Optional[str]:
        """
        查找是否存在相同MAX_SYMBOLS_ALLOWED的模型
        """
        try:
            search_path = WEIGHTS_DIR
            if not search_path.exists():
                return None
            filename = f"sac_model_symbols{MAX_SYMBOLS_ALLOWED}.zip"
            model_path = search_path / filename
            if model_path.exists():
                return str(model_path)
            return None
        except Exception as e:
            logger.warning(f"查找現有模型時發生錯誤: {e}")
            return None
    
    def get_model_save_path(self) -> Path:
        """
        取得唯一模型儲存路徑（不加任何suffix/prefix）
        """
        save_dir = WEIGHTS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"sac_model_symbols{MAX_SYMBOLS_ALLOWED}.zip"
        return save_dir / filename
    
    def prepare_data(self) -> bool:
        """
        準備訓練數據，包含歷史數據下載進度顯示
        
        返回:
            數據準備是否成功
        """
        try:
            logger.info("開始數據準備...")
            
            overall_start_iso = format_datetime_for_oanda(self.start_time)
            overall_end_iso = format_datetime_for_oanda(self.end_time)

            # Initial download for explicitly selected trading symbols (optional, as ensure_currency_data_for_trading will cover them)
            # However, this provides early feedback on primary symbols if desired.
            # This call might be redundant if ensure_currency_data_for_trading robustly handles all downloads.
            # For now, keeping it for potential separate progress reporting on primary symbols.
            logger.info(f"將為主要交易品種管理數據下載: {self.trading_symbols}, 範圍: {overall_start_iso} 到 {overall_end_iso}")
            manage_data_download_for_symbols(
                symbols=self.trading_symbols,
                overall_start_str=overall_start_iso,
                overall_end_str=overall_end_iso,
                granularity=self.granularity,
                streamlit_progress_bar=self.streamlit_progress_bar, # These might be None if using shared_data_manager
                streamlit_status_text=self.streamlit_status_text   # These might be None if using shared_data_manager
            )
            
            # 確保貨幣依賴數據完整 (this will also download if necessary)
            logger.info("確保交易所需貨幣數據完整性 (包括轉換對)...")
            # MODIFIED: Capture the full list of symbols from ensure_currency_data_for_trading
            success, all_symbols_for_dataset = ensure_currency_data_for_trading(
                trading_symbols=self.trading_symbols,
                account_currency=self.account_currency,
                start_time_iso=overall_start_iso,
                end_time_iso=overall_end_iso,
                granularity=self.granularity
            )
            
            if not success:
                logger.error("未能確保交易所需貨幣數據完整性。終止訓練。")
                self.shared_data_manager.update_training_status(status='error', error="數據依賴檢查失敗。請檢查日誌。")
                return False

            # 創建數據集 using ALL symbols for which data was ensured
            logger.info(f"創建記憶體映射數據集 for symbols: {list(all_symbols_for_dataset)}")
            self.dataset = UniversalMemoryMappedDataset(
                symbols=list(all_symbols_for_dataset), # MODIFIED: Use the full list of symbols
                start_time_iso=overall_start_iso, 
                end_time_iso=overall_end_iso,     # 更改參數名稱
                granularity=self.granularity,
                timesteps_history=self.timesteps_history # 更改參數名稱
            )
            
            # Removed self.dataset.is_valid() check as per previous fix.
            # The UniversalMemoryMappedDataset constructor should handle validation or raise an error.
            
            logger.info(f"數據集創建成功。Dataset length: {len(self.dataset) if self.dataset else 'N/A'}") # Added length log
            logger.info("數據準備完成。")
            return True
            
        except Exception as e:
            logger.error(f"數據準備失敗: {e}", exc_info=True)
            self.shared_data_manager.update_training_status(status='error', error=f"數據準備失敗: {e}")
            return False
    
    def setup_environment(self) -> bool:
        """
        設置交易環境
        
        返回:
            環境設置是否成功
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
                active_symbols_for_episode=self.trading_symbols, # ensure this is the correct parameter name
                initial_capital=self.initial_capital,
                max_episode_steps=self.max_episode_steps,
                commission_percentage_override=TRADE_COMMISSION_PERCENTAGE,
                atr_period=self.custom_atr_period,
                stop_loss_atr_multiplier=self.atr_stop_loss_multiplier,
                max_account_risk_per_trade=self.risk_percentage,
                shared_data_manager=self.shared_data_manager,
                training_step_offset=self.current_training_step
            )
            
            logger.info("交易環境設置成功。")
            logger.info(f"環境觀察空間: {self.env.observation_space}")
            logger.info(f"環境動作空間: {self.env.action_space}")
            
            return True
            
        except Exception as e:
            logger.error(f"環境設置失敗: {e}", exc_info=True)
            self.shared_data_manager.update_training_status(status='error', error=f"環境設置失敗: {e}")
            return False
    
    def setup_agent(self, load_model_path: Optional[str] = None) -> bool:
        """
        設置SAC代理
        
        Args:
            load_model_path: 可選的模型加載路徑 (用於恢復檢查點)
            
        返回:
            代理設置是否成功
        """
        try:
            if self.env is None:
                logger.error("環境未設置，請先調用 setup_environment()")
                return False
            
            logger.info("設置SAC代理...")

            # 1. 加載並動態更新模型配置
            logger.info("加載並配置模型...")
            
            def load_model_config():
                """Helper to load model config from the new location."""
                try:
                    # Construct path relative to this file's location to the project root
                    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
                    config_path = project_root / 'configs' / 'training' / 'enhanced_model_config.json'
                    with open(config_path, 'r') as f:
                        import json
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load model config: {e}", exc_info=True)
                    return {} # Return empty dict as a fallback

            model_config = load_model_config()
            if not model_config:
                logger.error("Model configuration could not be loaded. Aborting agent setup.")
                return False

            # 確保 device 正確設置
            model_config['device'] = DEVICE
            logger.info(f"模型配置更新: device 設置為 {DEVICE}")
            logger.info(f"模型配置確認: num_symbols 保持為 {model_config.get('num_symbols')} (應等於 MAX_SYMBOLS_ALLOWED)")


            # 2. 構建 policy_kwargs，將配置傳遞給特徵提取器
            policy_kwargs = {
                "features_extractor_class": EnhancedTransformerFeatureExtractor,
                "features_extractor_kwargs": {
                    "model_config": model_config # 直接傳遞字典
                }
            }
            logger.info(f"Policy kwargs 構建完成，特徵提取器為: {policy_kwargs['features_extractor_class'].__name__}")
            
            # 判斷模型加載路徑
            model_path_to_load = load_model_path or self.existing_model_path
            
            # 3. 創建SAC代理，傳入 policy_kwargs and model_config
            logger.info("創建 QuantumEnhancedSAC 代理...")
            self.agent = QuantumEnhancedSAC(
                env=self.env,
                model_config=model_config, # Pass the loaded model config
                device=DEVICE,
                use_amp=USE_AMP,
                verbose=1,
                policy_kwargs=policy_kwargs # 傳遞配置
            )
            
            # 如果存在，加載現有模型
            if model_path_to_load and Path(model_path_to_load).exists():
                logger.info(f"正在從 {model_path_to_load} 加載現有模型...")
                try:
                    # 加載時也傳遞 policy_kwargs，以強制使用當前配置，避免舊配置問題
                    self.agent.load(str(model_path_to_load), policy_kwargs=policy_kwargs)
                    logger.info("模型加載成功。")
                except Exception as e:
                    logger.warning(f"加載模型失敗，將創建一個新模型。錯誤: {e}", exc_info=True)
            else:
                logger.info("未找到現有模型或未提供路徑，將創建一個新模型。")
            
            logger.info("SAC代理設置成功。")
            return True
            
        except Exception as e:
            logger.error(f"代理設置失敗: {e}", exc_info=True)
            self.shared_data_manager.update_training_status(status='error', error=f"代理設置失敗: {e}")
            return False
    
    def setup_callbacks(self) -> bool:
        """
        設置訓練回調
        
        返回:
            回調設置是否成功
        """
        try:
            if self.agent is None:
                logger.error("代理未設置，請先調用 setup_agent()")
                return False
            
            logger.info("設置訓練回調...")
            
            save_dir = WEIGHTS_DIR # 使用 config 中定義的 WEIGHTS_DIR
            save_dir.mkdir(parents=True, exist_ok=True)
              # 創建 checkpoint 回調，包含數值穩定性配置
            self.callback = UniversalCheckpointCallback(
                save_freq=self.save_freq,
                save_path=save_dir,
                name_prefix=self.model_identifier,
                eval_freq=self.eval_freq,
                n_eval_episodes=5,
                deterministic_eval=True,
                verbose=1,
                streamlit_session_state=self.streamlit_session_state, # 傳遞給回調，以便更新UI
                shared_data_manager=self.shared_data_manager, # 傳遞共享數據管理器
                enable_gradient_clipping=True, # 啟用梯度裁剪以獲取梯度範數
                gradient_clip_norm=1.0, # 梯度裁剪範數
                nan_check_frequency=100 # 每100步檢查一次NaN值
            )
            
            logger.info("訓練回調設置成功 (已啟用梯度裁剪和NaN檢查)。")
            return True
            
        except Exception as e:
            logger.error(f"回調設置失敗: {e}", exc_info=True)
            self.shared_data_manager.update_training_status(status='error', error=f"回調設置失敗: {e}")
            return False
    
    def train(self) -> bool:
        """
        執行訓練過程，包含增強監控 (訓練速度與預估完成時間)
        
        返回:
            訓練是否成功
        """
        try:
            if self.agent is None or self.callback is None:
                logger.error("Agent or callback not configured.")
                return False
            
            logger.info("=" * 60)
            logger.info("開始SAC模型訓練")
            logger.info(f"總訓練步數: {self.total_timesteps}")
            logger.info("=" * 60)
            
            self.training_start_time = datetime.now(timezone.utc)
            self.current_training_step = 0 # 重置步數

            self.shared_data_manager.update_training_status('running', 0)
            self.shared_data_manager.training_start_time = self.training_start_time # 同步給管理器
            
            try:
                # 為了計算訓練速度和預估完成時間，我們需要一個內部的更新機制。
                # SimpleStableBaselines3 的 learn 方法內部沒有直接的每步回調，
                # UniversalCheckpointCallback 已經提供了一個 `_on_step` 方法，
                # 我們可以藉由它來更新共享數據管理器的性能指標。
                # 我們需要確保 UniversalCheckpointCallback 能夠將訓練步數資訊傳遞給 shared_data_manager。
                # 這需要在 UniversalCheckpointCallback 內部進行修改。
                # 這裡假設回調已經被正確設置以更新共享數據管理器中的訓練步數。

                self.agent.train( # Changed from self.agent.learn
                    total_timesteps=self.total_timesteps,
                    callback=self.callback, # UniversalCheckpointCallback 將會處理進度更新
                    log_interval=100, # 假設每100步記錄一次                    # eval_env=None, # eval_env is not a parameter of QuantumEnhancedSAC.train
                    # eval_freq=self.eval_freq, # eval_freq is not a parameter of QuantumEnhancedSAC.train
                    # n_eval_episodes=5, # n_eval_episodes is not a parameter of QuantumEnhancedSAC.train
                    # tb_log_name="sac_training", # tb_log_name is not a parameter of QuantumEnhancedSAC.train
                    # eval_log_path=None, # eval_log_path is not a parameter of QuantumEnhancedSAC.train
                    reset_num_timesteps=False
                )
                
                # 訓練成功完成
                self.shared_data_manager.update_training_status('completed', 100)
                # 保存最終模型（只覆蓋同一檔案）
                final_model_path = self.get_model_save_path()
                self.agent.save(str(final_model_path))
                logger.info(f"最終模型已保存: {final_model_path}")
                
                training_duration = datetime.now(timezone.utc) - self.training_start_time
                logger.info("=" * 60)
                logger.info("訓練成功完成！")
                logger.info(f"訓練持續時間: {training_duration}")
                logger.info(f"總步數: {self.total_timesteps}")
                logger.info(f"最終模型已保存: {final_model_path}")
                logger.info("=" * 60)
                
                return True
                
            except KeyboardInterrupt:
                logger.info("訓練被用戶中斷。正在嘗試保存當前模型...")
                self.save_current_model() # 中斷時保存模型
                self.shared_data_manager.update_training_status('idle')
                return False
                
            except Exception as e:
                logger.error(f"訓練過程中發生錯誤: {e}", exc_info=True)
                self.save_current_model() # 錯誤時保存模型
                self.shared_data_manager.update_training_status('error', error=str(e))
                return False
            
        except Exception as e:
            logger.error(f"訓練設置錯誤: {e}", exc_info=True)
            self.save_current_model() # 設置錯誤時也嘗試保存
            self.shared_data_manager.update_training_status('error', error=str(e))
            return False
    
    def stop(self):
        """請求停止訓練過程"""
        self._stop_training = True
        self.shared_data_manager.request_stop()
        logger.info("已請求停止訓練。")
    
    def save_current_model(self):
        """保存當前訓練進度（只覆蓋同一檔案）"""
        if self.agent:
            try:
                checkpoint_path = self.get_model_save_path()
                self.agent.save(str(checkpoint_path))
                logger.info(f"當前模型已保存: {checkpoint_path}")
            except Exception as e:
                logger.error(f"未能保存當前模型: {e}")
    
    def cleanup(self):
        """清理資源"""
        try:
            logger.info("正在清理訓練資源...")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("GPU內存已清理。")
            
            if self.env:
                try:
                    self.env.close()
                    logger.info("環境已關閉。")
                except Exception as e:
                    logger.warning(f"關閉環境時發生錯誤: {e}")
            
            # 清除引用
            self.agent = None
            self.env = None
            self.dataset = None
            self.callback = None
            
            logger.info("資源清理完成。")
            
        except Exception as e:
            logger.warning(f"清理過程中發生錯誤: {e}")
    
    def run_full_training_pipeline(self, load_model_path: Optional[str] = None) -> bool:
        """
        執行完整的訓練管道
        
        Args:
            load_model_path: 可選的模型加載路徑 (用於恢復檢查點)
              返回:
            整個管道是否成功完成
        """
        logger.info("=" * 60)
        logger.info("開始完整的訓練管道")
        logger.info("=" * 60)
        
        success = False  # 初始化 success 變量
        try:
            self.shared_data_manager.clear_data() # 開始新訓練前清除舊數據
            self.shared_data_manager.update_training_status('running', 0)
            
            # 1. 準備數據 (包含下載進度)
            self.shared_data_manager.update_training_status(status='running', progress=0)
            if not self.prepare_data():
                logger.error("數據準備失敗，終止訓練管道。")
                self.shared_data_manager.update_training_status('error', error="數據準備失敗")
                return False
            
            self.shared_data_manager.update_training_status(status='running', progress=10)

            # 2. 設置環境
            if not self.setup_environment():
                logger.error("環境設置失敗，終止訓練管道。")
                self.shared_data_manager.update_training_status('error', error="環境設置失敗")
                return False
            
            self.shared_data_manager.update_training_status(status='running', progress=20)

            # 3. 設置代理
            if not self.setup_agent(load_model_path):
                logger.error("代理設置失敗，終止訓練管道。")
                self.shared_data_manager.update_training_status('error', error="代理設置失敗")
                return False
            
            self.shared_data_manager.update_training_status(status='running', progress=30)

            # 4. 設置回調
            if not self.setup_callbacks():
                logger.error("回調設置失敗，終止訓練管道。")
                self.shared_data_manager.update_training_status('error', error="回調設置失敗")
                return False
            
            self.shared_data_manager.update_training_status(status='running', progress=40)

            # 5. 執行訓練
            success = self.train()
            
        except Exception as e:
            logger.error(f"整個訓練管道錯誤: {e}", exc_info=True)
            self.shared_data_manager.update_training_status('error', error=str(e))
            success = False
        finally:
            # 6. 清理資源
            self.cleanup()
            
            if success:
                logger.info("=" * 60)
                logger.info("完整的訓練管道成功完成！")
                logger.info("=" * 60)
                self.shared_data_manager.update_training_status(status='completed', progress=100)
            else:
                logger.warning("=" * 60)
                logger.warning("訓練管道未能成功完成。")
                logger.warning("=" * 60)
                if self.shared_data_manager.training_status != 'error':
                    self.shared_data_manager.update_training_status(status='warning')
            
            return success


def create_training_time_range(days_back: int = 30) -> tuple[datetime, datetime]:
    """
    創建訓練時間範圍
    
    Args:
        days_back: 從當前時間回溯的天數
        
    返回:
        (start_time, end_time) 元組
    """
    # 確保不會請求未來的數據，將結束時間設為昨天
    end_time = datetime.now(timezone.utc).replace(hour=23, minute=59, second=59, microsecond=0) - timedelta(days=1)
    start_time = end_time - timedelta(days=days_back)
    
    return start_time, end_time


if __name__ == "__main__":
    # 示例配置
    symbols_for_test = ["EUR_USD", "USD_JPY", "GBP_USD"]
    start_time_test, end_time_test = create_training_time_range(7) # 測試用7天數據
    
    print(f"Running UniversalTrainer test (only data preparation and environment setup):")
    print(f"  Symbols: {symbols_for_test}")
    print(f"  Time Range: {start_time_test.isoformat()} to {end_time_test.isoformat()}")

    # 模擬Streamlit UI組件
    class MockProgressBar:
        def __init__(self):
            self.progress_val = 0.0
        def progress(self, value):
            self.progress_val = value
            print(f"\rDownload progress: {value*100:.1f}%", end="")
    
    class MockStatusText:
        def info(self, text):
            # print(f"\n[狀態信息] {text}")
            logger.info(f"[Mock Status] {text}")
        def success(self, text):
            # print(f"\n[成功] {text}")
            logger.info(f"[Mock Success] {text}")
        def error(self, text):
            # print(f"\n[錯誤] {text}")
            logger.error(f"[Mock Error] {text}")
        def warning(self, text):
            # print(f"\n[警告] {text}")
            logger.warning(f"[Mock Warning] {text}")

    mock_progress_bar = MockProgressBar()
    mock_status_text = MockStatusText()

    trainer = UniversalTrainer(
        trading_symbols=symbols_for_test,
        start_time=start_time_test,
        end_time=end_time_test,
        total_timesteps=10000, # 縮短測試的總步數
        save_freq=1000,
        eval_freq=500,
        streamlit_progress_bar=None, # 設定為None，因為不再直接使用
        streamlit_status_text=None, # 設定為None，因為不再直接使用
        streamlit_session_state=None
    )
    
    # 執行完整的訓練管道
    success = trainer.run_full_training_pipeline()
    
    if success:
        print("\nUniversalTrainer test completed successfully!")
    else:
        print("\nUniversalTrainer test failed!")