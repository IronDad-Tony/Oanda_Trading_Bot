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
    logger.warning(f"enhanced_trainer.py: Initial import failed: {e_initial_import_et}. Assuming PYTHONPATH is set correctly or this is a critical issue.")
    
    try:
        # 假設 PYTHONPATH 已設定，這些導入應該能工作
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
        
        # 修改：模型儲存在 /weights 資料夾下，而不是 /logs 下
        save_dir = Path("weights")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        return save_dir / filename
    
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
            # 修改：模型儲存在 /weights 資料夾下，而不是 /logs 下
            save_dir = Path("weights")
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
        finally:
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