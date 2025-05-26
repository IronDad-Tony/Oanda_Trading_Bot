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

try:
    from common.logger_setup import logger as common_configured_logger
    logger = common_configured_logger
    _logger_initialized_by_common_ut = True
    logger.debug("universal_trainer.py: Successfully imported logger from common.logger_setup.")
    
    from common.config import (
        TIMESTEPS, MAX_SYMBOLS_ALLOWED, ACCOUNT_CURRENCY, INITIAL_CAPITAL,
        OANDA_MARGIN_CLOSEOUT_LEVEL, TRADE_COMMISSION_PERCENTAGE, OANDA_API_KEY,
        WEIGHTS_DIR, LOGS_DIR, DEVICE, USE_AMP,
        TRAINER_SAVE_FREQ_STEPS, TRAINER_EVAL_FREQ_STEPS,
        # 新增訓練參數的預設值
        TRAINING_INITIAL_CAPITAL_DEFAULT, TRAINING_RISK_PERCENTAGE_DEFAULT,
        TRAINING_ATR_MULTIPLIER_DEFAULT, TRAINING_MAX_POSITION_DEFAULT
    )
    logger.info("universal_trainer.py: Successfully imported common.config values.")
    
    from common.shared_data_manager import get_shared_data_manager
    logger.info("universal_trainer.py: Successfully imported shared data manager.")
    
    from data_manager.currency_manager import CurrencyDependencyManager, ensure_currency_data_for_trading
    from data_manager.mmap_dataset import UniversalMemoryMappedDataset
    from data_manager.instrument_info_manager import InstrumentInfoManager
    from data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols
    from environment.trading_env import UniversalTradingEnvV4
    from agent.sac_agent_wrapper import SACAgentWrapper
    from trainer.callbacks import UniversalCheckpointCallback # 假設這個Callback已經存在並被正確實作
    logger.info("universal_trainer.py: Successfully imported other dependencies.")
    
except ImportError as e_initial_import_ut:
    logger_temp_ut = logging.getLogger("universal_trainer_fallback_initial")
    logger_temp_ut.addHandler(logging.StreamHandler(sys.stdout))
    logger_temp_ut.setLevel(logging.DEBUG)
    logger = logger_temp_ut
    logger.warning(f"universal_trainer.py: Initial import failed: {e_initial_import_ut}. Attempting path adjustment.")
    
    try:
        from src.common.logger_setup import logger as common_logger_retry_ut
        logger = common_logger_retry_ut
        _logger_initialized_by_common_ut = True
        logger.info("universal_trainer.py: Successfully re-imported common_logger after path adj.")
        
        from src.common.config import (
            TIMESTEPS, MAX_SYMBOLS_ALLOWED, ACCOUNT_CURRENCY, INITIAL_CAPITAL,
            OANDA_MARGIN_CLOSEOUT_LEVEL, TRADE_COMMISSION_PERCENTAGE, OANDA_API_KEY,
            WEIGHTS_DIR, LOGS_DIR, DEVICE, USE_AMP,
            TRAINER_SAVE_FREQ_STEPS, TRAINER_EVAL_FREQ_STEPS,
            TRAINING_INITIAL_CAPITAL_DEFAULT, TRAINING_RISK_PERCENTAGE_DEFAULT,
            TRAINING_ATR_MULTIPLIER_DEFAULT, TRAINING_MAX_POSITION_DEFAULT
        )
        logger.info("universal_trainer.py: Successfully re-imported common.config after path adjustment.")
        
        from src.common.shared_data_manager import get_shared_data_manager
        logger.info("universal_trainer.py: Successfully re-imported shared data manager.")
        
        from src.data_manager.currency_manager import CurrencyDependencyManager, ensure_currency_data_for_trading
        from src.data_manager.mmap_dataset import UniversalMemoryMappedDataset
        from src.data_manager.instrument_info_manager import InstrumentInfoManager
        from src.data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols
        from src.environment.trading_env import UniversalTradingEnvV4
        from src.agent.sac_agent_wrapper import SACAgentWrapper
        from src.trainer.callbacks import UniversalCheckpointCallback
        logger.info("universal_trainer.py: Successfully re-imported other dependencies after path adjustment.")
        
    except ImportError as e_retry_critical_ut:
        logger.error(f"universal_trainer.py: Critical import error after path adjustment: {e_retry_critical_ut}", exc_info=True)
        logger.warning("universal_trainer.py: Using fallback mode - some features may not work.")
        # Fallback classes for critical imports
        class CurrencyDependencyManager:
            def __init__(self, **kwargs): pass
            def download_required_currency_data(self, *args, **kwargs): return True
        def ensure_currency_data_for_trading(*args, **kwargs): return True
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
        class SACAgentWrapper:
            def __init__(self, **kwargs): pass
            def load(self, path): pass
            def save(self, path): pass
            def learn(self, **kwargs): pass
        class UniversalCheckpointCallback:
            def __init__(self, **kwargs): pass
        
        # Fallback for config
        TIMESTEPS, MAX_SYMBOLS_ALLOWED, ACCOUNT_CURRENCY, INITIAL_CAPITAL = 128, 20, "AUD", 100000
        OANDA_MARGIN_CLOSEOUT_LEVEL, TRADE_COMMISSION_PERCENTAGE = 0.5, 0.0001
        OANDA_API_KEY, WEIGHTS_DIR, LOGS_DIR, DEVICE, USE_AMP = None, Path("weights"), Path("logs"), "cpu", False
        TRAINER_SAVE_FREQ_STEPS, TRAINER_EVAL_FREQ_STEPS = 20000, 10000
        TRAINING_INITIAL_CAPITAL_DEFAULT, TRAINING_RISK_PERCENTAGE_DEFAULT = 100000, 5.0
        TRAINING_ATR_MULTIPLIER_DEFAULT, TRAINING_MAX_POSITION_DEFAULT = 2.0, 10.0


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
                 risk_percentage: float = TRAINING_RISK_PERCENTAGE_DEFAULT / 100.0,
                 atr_stop_loss_multiplier: float = TRAINING_ATR_MULTIPLIER_DEFAULT,
                 max_position_percentage: float = TRAINING_MAX_POSITION_DEFAULT / 100.0,
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
        self.risk_percentage = risk_percentage
        self.atr_stop_loss_multiplier = atr_stop_loss_multiplier
        self.max_position_percentage = max_position_percentage
        self.custom_atr_period = custom_atr_period
        
        # 初始化共享數據管理器
        self.shared_data_manager = get_shared_data_manager()
        logger.info("已連接到共享數據管理器")
        
        # 設置GPU優化
        self._setup_gpu_optimization()
        
        # 生成基於參數的模型識別符，包含 MAX_SYMBOLS_ALLOWED
        self.model_identifier = self._generate_model_identifier(MAX_SYMBOLS_ALLOWED)
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
                
                logger.info(f"檢測到 {gpu_count} 個GPU設備")
                logger.info(f"當前使用GPU {current_device}: {gpu_name} ({gpu_memory:.1f}GB)")
                
                torch.cuda.empty_cache()
                gc.collect()
                
                torch.cuda.set_per_process_memory_fraction(0.85)
                
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                
                if hasattr(torch.backends.cuda, 'matmul'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
                
                if USE_AMP:
                    logger.info("已啟用混合精度訓練")
                    if torch.cuda.get_device_capability(current_device)[0] >= 7:
                        logger.info("GPU支持混合精度訓練")
                    else:
                        logger.warning("GPU可能不完全支持混合精度訓練，但仍將嘗試使用它")
                
                logger.info("GPU優化設置完成")
                
            else:
                logger.info("未檢測到CUDA，將使用CPU訓練")
                torch.set_num_threads(min(8, torch.get_num_threads()))
                logger.info(f"CPU線程數設置為: {torch.get_num_threads()}")
                
        except Exception as e:
            logger.warning(f"GPU優化設置過程中發生錯誤: {e}")
    
    def _generate_model_identifier(self, max_symbols_allowed: int) -> str:
        """
        根據關鍵參數生成模型識別符，包含 MAX_SYMBOLS_ALLOWED
        
        返回:
            模型識別符字符串
        """
        # 使用配置中的 MAX_SYMBOLS_ALLOWED 和時間步長作為主要參數
        # 這確保了如果 MAX_SYMBOLS_ALLOWED 變更，即使實際選擇的交易對數量相同，也會訓練新模型。
        effective_symbols_count = len(self.trading_symbols) # 實際選擇的交易對數量
        timestep = self.timesteps_history
        
        # 識別符格式: sac_model_max_allowed{config_val}_actual{count}_timestep{steps}
        identifier = f"sac_model_max_allowed{max_symbols_allowed}_actual{effective_symbols_count}_timestep{timestep}"
        return identifier
    
    def _find_existing_model(self) -> Optional[str]:
        """
        查找是否存在具有相同參數的模型
        
        返回:
            現有模型路徑，如果未找到則為 None
        """
        try:
            # 優先從 WEIGHTS_DIR 查找模型
            search_paths = [WEIGHTS_DIR]
            
            for search_path in search_paths:
                if not search_path.exists():
                    continue
                
                # 查找匹配的模型文件
                pattern = f"{self.model_identifier}*.zip"
                matching_files = list(search_path.glob(pattern))
                
                if matching_files:
                    # 返回最新創建的文件
                    latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
                    return str(latest_file)
            
            return None
            
        except Exception as e:
            logger.warning(f"查找現有模型時發生錯誤: {e}")
            return None
    
    def get_model_save_path(self, suffix: str = "") -> Path:
        """
        獲取模型保存路徑
        
        Args:
            suffix: 文件名後綴
            
        返回:
            模型保存路徑
        """
        if suffix:
            filename = f"{self.model_identifier}_{suffix}.zip"
        else:
            filename = f"{self.model_identifier}.zip"
        
        save_dir = WEIGHTS_DIR # 使用 config 中定義的 WEIGHTS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        
        return save_dir / filename
    
    def prepare_data(self) -> bool:
        """
        準備訓練數據，包含歷史數據下載進度顯示
        
        返回:
            數據準備是否成功
        """
        try:
            logger.info("開始數據準備...")
            
            # 使用 oanda_downloader 中的 manage_data_download_for_symbols
            # 並傳遞 Streamlit 的進度條和狀態文本組件
            overall_start_iso = format_datetime_for_oanda(self.start_time)
            overall_end_iso = format_datetime_for_oanda(self.end_time)

            logger.info(f"將為以下品種管理數據下載: {self.trading_symbols}, 範圍: {overall_start_iso} 到 {overall_end_iso}")

            manage_data_download_for_symbols(
                symbols=self.trading_symbols,
                overall_start_str=overall_start_iso,
                overall_end_str=overall_end_iso,
                granularity=self.granularity,
                streamlit_progress_bar=self.streamlit_progress_bar,
                streamlit_status_text=self.streamlit_status_text
            )
            
            # 確保貨幣依賴數據完整
            logger.info("確保交易所需貨幣數據完整性...")
            success = ensure_currency_data_for_trading(
                currency_symbols=self.trading_symbols,
                account_currency=self.account_currency,
                start_time=self.start_time,
                end_time=self.end_time,
                granularity=self.granularity
            )
            
            if not success:
                logger.error("未能確保交易所需貨幣數據完整性。終止訓練。")
                self.shared_data_manager.update_training_status(status='error', error="數據依賴檢查失敗。請檢查日誌。", message="數據依賴檢查失敗")
                return False

            # 創建數據集
            logger.info("創建記憶體映射數據集...")
            self.dataset = UniversalMemoryMappedDataset(
                symbols=self.trading_symbols,
                start_time=self.start_time,
                end_time=self.end_time,
                granularity=self.granularity,
                timesteps=self.timesteps_history
            )
            
            if not self.dataset.is_valid():
                logger.error("數據集創建失敗或無效。終止訓練。")
                self.shared_data_manager.update_training_status(status='error', error="數據集創建失敗或無效。請檢查所選日期和品種。", message="數據集創建失敗")
                return False
            
            logger.info(f"數據集創建成功，包含 {len(self.dataset)} 個樣本。")
            logger.info("數據準備完成。")
            return True
            
        except Exception as e:
            logger.error(f"數據準備失敗: {e}", exc_info=True)
            self.shared_data_manager.update_training_status(status='error', error=f"數據準備失敗: {e}", message=f"數據準備失敗: {e}")
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
                active_symbols_for_episode=self.trading_symbols,
                initial_capital=self.initial_capital,
                max_episode_steps=self.max_episode_steps,
                commission_percentage_override=TRADE_COMMISSION_PERCENTAGE,
                atr_period=self.custom_atr_period,
                stop_loss_atr_multiplier=self.atr_stop_loss_multiplier,
                max_account_risk_per_trade=self.risk_percentage
            )
            
            logger.info("交易環境設置成功。")
            logger.info(f"環境觀察空間: {self.env.observation_space}")
            logger.info(f"環境動作空間: {self.env.action_space}")
            
            return True
            
        except Exception as e:
            logger.error(f"環境設置失敗: {e}", exc_info=True)
            self.shared_data_manager.update_training_status(status='error', error=f"環境設置失敗: {e}", message=f"環境設置失敗: {e}")
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
            
            # 判斷模型加載路徑
            model_path_to_load = load_model_path or self.existing_model_path
            
            # 創建SAC代理
            self.agent = SACAgentWrapper(
                env=self.env,
                device=DEVICE,
                use_amp=USE_AMP,
                verbose=1
            )
            
            # 如果存在，加載現有模型
            if model_path_to_load and Path(model_path_to_load).exists():
                logger.info(f"加載現有模型: {model_path_to_load}")
                try:
                    self.agent.load(str(model_path_to_load))
                    logger.info("模型加載成功。")
                except Exception as e:
                    logger.warning(f"未能加載模型，將創建新模型: {e}")
            else:
                logger.info("創建新模型。")
            
            logger.info("SAC代理設置成功。")
            return True
            
        except Exception as e:
            logger.error(f"代理設置失敗: {e}", exc_info=True)
            self.shared_data_manager.update_training_status(status='error', error=f"代理設置失敗: {e}", message=f"代理設置失敗: {e}")
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
            
            # 創建 checkpoint 回調
            self.callback = UniversalCheckpointCallback(
                save_freq=self.save_freq,
                save_path=save_dir,
                name_prefix=self.model_identifier,
                eval_freq=self.eval_freq,
                n_eval_episodes=5,
                deterministic_eval=True,
                verbose=1,
                streamlit_session_state=self.streamlit_session_state # 傳遞給回調，以便更新UI
            )
            
            logger.info("訓練回調設置成功。")
            return True
            
        except Exception as e:
            logger.error(f"回調設置失敗: {e}", exc_info=True)
            self.shared_data_manager.update_training_status(status='error', error=f"回調設置失敗: {e}", message=f"回調設置失敗: {e}")
            return False
    
    def train(self) -> bool:
        """
        執行訓練過程，包含增強監控 (訓練速度與預估完成時間)
        
        返回:
            訓練是否成功
        """
        try:
            if self.agent is None or self.callback is None:
                logger.error("代理或回調未設置。")
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

                self.agent.learn(
                    total_timesteps=self.total_timesteps,
                    callback=self.callback, # UniversalCheckpointCallback 將會處理進度更新
                    log_interval=100, # 假設每100步記錄一次
                    eval_env=None, 
                    eval_freq=self.eval_freq,
                    n_eval_episodes=5,
                    tb_log_name="sac_training",
                    eval_log_path=None,
                    reset_num_timesteps=False
                )
                
                # 訓練成功完成
                self.shared_data_manager.update_training_status('completed', 100)
                
                # 保存最終模型
                final_model_path = self.get_model_save_path("final")
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
                self.shared_data_manager.update_training_status('idle', message="訓練被用戶中斷。當前模型已保存。")
                return False
                
            except Exception as e:
                logger.error(f"訓練過程中發生錯誤: {e}", exc_info=True)
                self.save_current_model() # 錯誤時保存模型
                self.shared_data_manager.update_training_status('error', error=str(e), message=f"訓練錯誤: {e}。當前模型已保存。")
                return False
            
        except Exception as e:
            logger.error(f"訓練設置錯誤: {e}", exc_info=True)
            self.save_current_model() # 設置錯誤時也嘗試保存
            self.shared_data_manager.update_training_status('error', error=str(e), message=f"訓練設置錯誤: {e}。當前模型可能已保存。")
            return False
    
    def stop(self):
        """請求停止訓練過程"""
        self._stop_training = True
        self.shared_data_manager.request_stop()
        logger.info("已請求停止訓練。")
    
    def save_current_model(self):
        """保存當前訓練進度"""
        if self.agent:
            try:
                # 創建一個帶有當前步數時間戳的檢查點文件名稱
                current_step_str = str(self.shared_data_manager.current_metrics.get('step', '0'))
                checkpoint_path = self.get_model_save_path(f"checkpoint_step_{current_step_str}")
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
        
        try:
            self.shared_data_manager.clear_data() # 開始新訓練前清除舊數據
            self.shared_data_manager.update_training_status('running', 0)
            
            # 1. 準備數據 (包含下載進度)
            self.shared_data_manager.update_training_status(status='running', progress=0, message="正在準備數據...")
            if not self.prepare_data():
                logger.error("數據準備失敗，終止訓練管道。")
                self.shared_data_manager.update_training_status('error', error="數據準備失敗", message="數據準備失敗")
                return False
            
            self.shared_data_manager.update_training_status(status='running', progress=10, message="正在設置環境...")

            # 2. 設置環境
            if not self.setup_environment():
                logger.error("環境設置失敗，終止訓練管道。")
                self.shared_data_manager.update_training_status('error', error="環境設置失敗")
                return False
            
            self.shared_data_manager.update_training_status(status='running', progress=20, message="正在設置代理...")

            # 3. 設置代理
            if not self.setup_agent(load_model_path):
                logger.error("代理設置失敗，終止訓練管道。")
                self.shared_data_manager.update_training_status('error', error="代理設置失敗")
                return False
            
            self.shared_data_manager.update_training_status(status='running', progress=30, message="正在設置回調...")

            # 4. 設置回調
            if not self.setup_callbacks():
                logger.error("回調設置失敗，終止訓練管道。")
                self.shared_data_manager.update_training_status('error', error="回調設置失敗")
                return False
            
            self.shared_data_manager.update_training_status(status='running', progress=40, message="訓練即將開始...")

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
                self.shared_data_manager.update_training_status(status='completed', progress=100, message="訓練成功完成！")
            else:
                logger.warning("=" * 60)
                logger.warning("訓練管道未能成功完成。")
                logger.warning("=" * 60)
                if self.shared_data_manager.training_status != 'error':
                    self.shared_data_manager.update_training_status(status='warning', message="訓練管道未完成。請檢查日誌獲取詳細信息。")
            
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
    
    print(f"運行 UniversalTrainer 測試 (只進行數據準備和環境設置):")
    print(f"  Symbols: {symbols_for_test}")
    print(f"  Time Range: {start_time_test.isoformat()} to {end_time_test.isoformat()}")

    # 模擬Streamlit UI組件
    class MockProgressBar:
        def __init__(self):
            self.progress_val = 0.0
        def progress(self, value):
            self.progress_val = value
            print(f"\r下載進度: {value*100:.1f}%", end="")
    
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
        print("\nUniversalTrainer 測試成功完成！")
    else:
        print("\nUniversalTrainer 測試失敗！")