#!/usr/bin/env python3
"""
OANDA 通用自動交易模型 - 整合版啟動腳本
包含所有修復和優化的統一啟動器
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
import logging

# 確保能找到src模組
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def check_system_requirements():
    """檢查系統需求"""
    logger.info("檢查系統需求...")
    
    # 檢查Python版本
    if sys.version_info < (3, 8):
        logger.error("需要Python 3.8或更高版本")
        return False
    
    # 檢查必要的目錄
    required_dirs = ['src', 'logs', 'data', 'weights']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"創建目錄: {dir_path}")
    
    # 檢查GPU可用性
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"檢測到 {gpu_count} 個GPU: {gpu_name}")
        else:
            logger.info("未檢測到GPU，將使用CPU訓練")
    except ImportError:
        logger.warning("PyTorch未安裝，請檢查依賴")
        return False
    
    logger.info("✅ 系統需求檢查通過")
    return True

def cleanup_mmap_files():
    """清理舊的mmap檔案"""
    logger.info("清理舊的mmap檔案...")
    
    try:
        # 清理data目錄中的mmap檔案
        data_dir = Path('data')
        if data_dir.exists():
            mmap_files = list(data_dir.glob('*.mmap')) + list(data_dir.glob('*.dat'))
            for mmap_file in mmap_files:
                try:
                    mmap_file.unlink()
                    logger.info(f"刪除mmap檔案: {mmap_file}")
                except Exception as e:
                    logger.warning(f"無法刪除 {mmap_file}: {e}")
        
        # 清理臨時檔案
        temp_files = list(Path('.').glob('*.tmp')) + list(Path('.').glob('*.temp'))
        for temp_file in temp_files:
            try:
                temp_file.unlink()
                logger.info(f"刪除臨時檔案: {temp_file}")
            except Exception as e:
                logger.warning(f"無法刪除 {temp_file}: {e}")
        
        logger.info("✅ mmap檔案清理完成")
        
    except Exception as e:
        logger.warning(f"mmap檔案清理過程中發生錯誤: {e}")

def main():
    """主函數"""
    
    print("=" * 60)
    print("🚀 OANDA 通用自動交易模型訓練系統 - 整合版")
    print("=" * 60)
    
    # 檢查系統需求
    if not check_system_requirements():
        print("❌ 系統需求檢查失敗，請檢查環境配置")
        return False
    
    # 清理舊檔案
    cleanup_mmap_files()
    
    try:
        # 導入訓練器
        from src.trainer.enhanced_trainer_complete import EnhancedUniversalTrainer, create_training_time_range
        from src.common.logger_setup import logger as common_logger
        from src.common.shared_data_manager import get_shared_data_manager
        
        logger.info("✅ 所有模組導入成功")
        
        # 配置訓練參數
        trading_symbols = [
            "EUR_USD",    # 歐元/美元
            "USD_JPY",    # 美元/日元
            "GBP_USD",    # 英鎊/美元
            "AUD_USD",    # 澳元/美元
            "USD_CAD",    # 美元/加元
        ]
        
        # 使用最近30天的數據進行訓練
        start_time, end_time = create_training_time_range(days_back=30)
        
        print(f"📊 訓練配置:")
        print(f"   交易品種: {', '.join(trading_symbols)}")
        print(f"   數據時間: {start_time.strftime('%Y-%m-%d %H:%M')} 到 {end_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"   數據粒度: S5 (5秒)")
        print(f"   訓練步數: 50,000")
        print(f"   保存頻率: 每 2,000 步")
        print(f"   評估頻率: 每 5,000 步")
        print()
        
        # 詢問用戶確認
        response = input("🤔 是否開始訓練？(y/N): ").strip().lower()
        if response not in ['y', 'yes', '是']:
            print("❌ 訓練已取消")
            return False
        
        # 初始化共享數據管理器
        shared_manager = get_shared_data_manager()
        shared_manager.clear_data()
        
        # 創建訓練器
        trainer = EnhancedUniversalTrainer(
            trading_symbols=trading_symbols,
            start_time=start_time,
            end_time=end_time,
            granularity="S5",
            total_timesteps=50000,      # 50K步，約需要30-60分鐘
            save_freq=2000,             # 每2K步保存
            eval_freq=5000,             # 每5K步評估
            model_name_prefix="sac_universal_trader"
        )
        
        print("\n🎯 開始完整訓練流程...")
        
        # 執行完整訓練流程
        success = trainer.run_full_training_pipeline()
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 訓練成功完成！")
            print("=" * 60)
            print("📁 模型文件保存在: weights/ 目錄")
            print("📊 TensorBoard日誌: logs/sac_tensorboard_logs_*/")
            print("🔍 查看訓練進度: tensorboard --logdir=logs/")
            print("🌐 啟動Streamlit UI: streamlit run streamlit_app_complete.py")
            print("=" * 60)
            return True
        else:
            print("\n" + "=" * 60)
            print("⚠️  訓練未完全成功")
            print("=" * 60)
            print("💡 請檢查日誌文件了解詳情")
            print("📋 運行整合測試: python integration_test.py")
            return False
            
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("⏹️  訓練被用戶中斷")
        print("=" * 60)
        print("💾 模型已自動保存")
        return False
        
    except ImportError as e:
        print("\n" + "=" * 60)
        print(f"❌ 模組導入錯誤: {e}")
        print("=" * 60)
        print("💡 建議解決方案:")
        print("   1. 檢查所有依賴是否已安裝: pip install -r requirements.txt")
        print("   2. 運行整合測試: python integration_test.py")
        print("   3. 檢查Python路徑配置")
        return False
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 訓練過程中發生錯誤: {e}")
        print("=" * 60)
        logger.error(f"訓練錯誤: {e}", exc_info=True)
        print("💡 建議解決方案:")
        print("   1. 運行整合測試: python integration_test.py")
        print("   2. 檢查日誌文件: training.log")
        print("   3. 確保有足夠的磁碟空間和記憶體")
        return False

def show_help():
    """顯示幫助信息"""
    print("OANDA AI Trading Bot - 整合版啟動腳本")
    print()
    print("使用方法:")
    print("  python start_training_integrated.py        # 開始訓練")
    print("  python start_training_integrated.py --help # 顯示此幫助")
    print()
    print("相關命令:")
    print("  python integration_test.py                 # 運行整合測試")
    print("  streamlit run streamlit_app_complete.py    # 啟動Web UI")
    print("  tensorboard --logdir=logs/                 # 查看訓練進度")
    print()
    print("故障排除:")
    print("  1. 確保已安裝所有依賴: pip install -r requirements.txt")
    print("  2. 檢查Python版本 >= 3.8")
    print("  3. 確保有足夠的磁碟空間 (至少5GB)")
    print("  4. 如果使用GPU，確保CUDA驅動正確安裝")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        show_help()
        sys.exit(0)
    
    success = main()
    sys.exit(0 if success else 1)