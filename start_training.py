#!/usr/bin/env python3
"""
OANDA 通用自動交易模型 - 主訓練腳本
簡化的訓練啟動器，適合日常使用
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# 確保能找到src模組
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.trainer.enhanced_trainer import EnhancedUniversalTrainer, create_training_time_range
from src.common.logger_setup import logger

def main():
    """主訓練函數"""
    
    print("=" * 60)
    print("🚀 OANDA 通用自動交易模型訓練系統")
    print("=" * 60)
    
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
    
    try:
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
            print("📁 模型文件保存在: logs/ 目錄")
            print("📊 TensorBoard日誌: logs/sac_tensorboard_logs_*/")
            print("🔍 查看訓練進度: tensorboard --logdir=logs/")
            print("=" * 60)
            return True
        else:
            print("\n" + "=" * 60)
            print("⚠️  訓練未完全成功")
            print("=" * 60)
            print("💡 請檢查日誌文件了解詳情")
            return False
            
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("⏹️  訓練被用戶中斷")
        print("=" * 60)
        print("💾 模型已自動保存")
        return False
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 訓練過程中發生錯誤: {e}")
        print("=" * 60)
        logger.error(f"訓練錯誤: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)