"""
訓練流程測試腳本
驗證從數據下載到模型訓練的整個流程是否正常運作
"""
import os
import sys
import logging
import sys
import io

# 強制標準輸出使用UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 設置系統環境變數強制UTF-8
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # 啟用oneDNN優化
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from src.data_manager.oanda_downloader import manage_data_download_for_symbols
from src.data_manager.mmap_dataset import UniversalMemoryMappedDataset
from src.data_manager.instrument_info_manager import InstrumentInfoManager
from src.trainer.universal_trainer import UniversalTrainer
from src.environment.trading_env import UniversalTradingEnvV4
from src.agent.sac_agent_wrapper import QuantumEnhancedSAC as SACAgentWrapper

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test_training")

def test_training_workflow():
    """測試整個訓練流程"""
    try:
        logger.info("===== 開始訓練流程測試 =====")
        
        # 1. 數據準備
        logger.info("步驟1: 數據下載...")
        symbols = ["EUR_USD", "USD_JPY"]  # 使用少量品種測試
        # 使用工作日的歷史時間範圍（2024-01-03至2024-01-04），確保有交易數據
        test_start_datetime = datetime(2024, 1, 3, 0, 0, 0, tzinfo=timezone.utc)
        test_end_datetime = datetime(2024, 1, 4, 0, 0, 0, tzinfo=timezone.utc)
        start_time = test_start_datetime
        end_time = test_end_datetime
        logger.info(f"使用固定時間範圍: {start_time.isoformat()} 至 {end_time.isoformat()}")
        granularity = "S5"
        
        manage_data_download_for_symbols(
            symbols=symbols,
            overall_start_str=start_time.isoformat(),
            overall_end_str=end_time.isoformat(),
            granularity=granularity
        )
        logger.info("數據下載完成")
        
        # 2. 創建記憶體映射數據集
        logger.info("步驟2: 創建記憶體映射數據集...")
        dataset = UniversalMemoryMappedDataset(
            symbols=symbols,
            start_time_iso=start_time.isoformat(),
            end_time_iso=end_time.isoformat(),
            granularity=granularity,
            timesteps_history=128,
            force_reload=False
        )
        logger.info(f"數據集創建完成，包含 {len(dataset)} 個樣本")
        
        # 3. 初始化交易品種信息管理器
        logger.info("步驟3: 初始化交易品種信息...")
        instrument_manager = InstrumentInfoManager(force_refresh=False)
        logger.info("交易品種信息初始化完成")
        
        # 4. 設置訓練環境
        logger.info("步驟4: 設置訓練環境...")
        env = UniversalTradingEnvV4(
            dataset=dataset,
            instrument_info_manager=instrument_manager,
            active_symbols_for_episode=symbols,
            initial_capital=Decimal("10000"),
            max_episode_steps=10  # 測試少量步驟
        )
        logger.info("訓練環境設置完成")
        
        # 5. 初始化SAC代理
        logger.info("步驟5: 初始化SAC代理...")
        agent = SACAgentWrapper(
            env=env,
            tensorboard_log_path="./logs/test_tensorboard",
            buffer_size=1000,
            batch_size=32,
            device="auto"  # 自動選擇最佳設備（優先GPU）
        )
        logger.info("SAC代理初始化完成")
        
        # 6. 初始化訓練器
        logger.info("步驟6: 初始化訓練器...")
        trainer = UniversalTrainer(
            trading_symbols=symbols,
            start_time=start_time,
            end_time=end_time,
            account_currency="USD",
            model_name_prefix="test_model",
            total_timesteps=50  # 測試少量時間步
        )
        logger.info("訓練器初始化完成")
        
        # 7. 執行訓練
        logger.info("步驟7: 開始訓練...")
        trainer.train()
        logger.info("訓練完成，無錯誤發生")
        
        # 8. 清理資源
        logger.info("步驟8: 清理資源...")
        dataset.close()
        env.close()
        logger.info("資源清理完成")
        
        logger.info("===== 訓練流程測試成功完成 =====")
        return True
        
    except Exception as e:
        logger.error(f"訓練流程測試失敗: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    # 執行測試
    success = test_training_workflow()
    
    # 輸出測試結果
    if success:
        logger.info("測試結果: 成功")
        print("\n測試成功！整個訓練流程正常運作。")
    else:
        logger.error("測試結果: 失敗")
        print("\n測試失敗！請檢查日誌了解詳細錯誤信息。")
    
    # 確保程序退出
    sys.exit(0 if success else 1)