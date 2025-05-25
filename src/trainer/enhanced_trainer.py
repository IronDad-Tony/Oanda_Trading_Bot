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
        LOGS_DIR
    )
    logger.info("enhanced_trainer.py: Successfully imported common.config values.")
    
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
            LOGS_DIR
        )
        logger.info("enhanced_trainer.py: Successfully re-imported common.config after path adjustment.")
        
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
                 model_name_prefix: str = "sac_universal_trader"):
        
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
        
        # 初始化組件
        self.currency_manager = CurrencyDependencyManager(account_currency)
        self.instrument_manager = InstrumentInfoManager(force_refresh=False)
        
        # 將在 prepare_data 中初始化
        self.dataset = None
        self.env = None
        self.agent = None
        self.callback = None
        
        logger.info(f"EnhancedUniversalTrainer 初始化完成")
        logger.info(f"交易symbols: {self.trading_symbols}")
        logger.info(f"時間範圍: {self.start_time} 到 {self.end_time}")
        logger.info(f"帳戶貨幣: {self.account_currency}")
    
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
            load_model_path: 可選的模型加載路徑
            
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
            
            # 創建SAC智能體包裝器
            self.agent = SACAgentWrapper(
                env=self.vec_env,
                verbose=1,
                tensorboard_log_path=self.tensorboard_log_path
            )
            
            # 如果有模型路徑，加載模型
            if load_model_path:
                logger.info(f"加載模型: {load_model_path}")
                self.agent.load(load_model_path)
            
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
            
            # 創建檢查點回調
            self.callback = UniversalCheckpointCallback(
                save_freq=self.save_freq,
                save_path=LOGS_DIR / f"{self.model_name_prefix}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                name_prefix=self.model_name_prefix,
                eval_freq=self.eval_freq,
                n_eval_episodes=5,
                deterministic_eval=True,
                verbose=1
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
            
            # 執行訓練
            self.agent.train(
                total_timesteps=self.total_timesteps,
                callback=self.callback,
                log_interval=100
            )
            
            logger.info("訓練完成！")
            
            # 保存最終模型
            from pathlib import Path
            final_model_path = Path(LOGS_DIR) / f"{self.model_name_prefix}_final.zip"
            self.agent.save(final_model_path)
            logger.info(f"最終模型已保存: {final_model_path}")
            
            return True
            
        except KeyboardInterrupt:
            logger.info("訓練被用戶中斷")
            # 保存當前模型
            if self.agent:
                from pathlib import Path
                interrupted_model_path = Path(LOGS_DIR) / f"{self.model_name_prefix}_interrupted.zip"
                self.agent.save(interrupted_model_path)
                logger.info(f"中斷時模型已保存: {interrupted_model_path}")
            return False
            
        except Exception as e:
            logger.error(f"訓練失敗: {e}", exc_info=True)
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
        
        # 1. 準備數據
        if not self.prepare_data():
            logger.error("數據準備失敗，終止訓練")
            return False
        
        # 2. 設置環境
        if not self.setup_environment():
            logger.error("環境設置失敗，終止訓練")
            return False
        
        # 3. 設置智能體
        if not self.setup_agent(load_model_path):
            logger.error("智能體設置失敗，終止訓練")
            return False
        
        # 4. 設置回調
        if not self.setup_callbacks():
            logger.error("回調設置失敗，終止訓練")
            return False
        
        # 5. 執行訓練
        success = self.train()
        
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
                
        except Exception as e:
            logger.warning(f"清理資源時發生錯誤: {e}")


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