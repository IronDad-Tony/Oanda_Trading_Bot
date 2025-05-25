#!/usr/bin/env python3
# test_system.py
"""
系統測試腳本 - 快速驗證所有組件是否正常工作
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# 添加項目根目錄到路徑
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.trainer.enhanced_trainer import EnhancedUniversalTrainer, create_training_time_range
from src.common.logger_setup import logger


def test_quick_training():
    """快速訓練測試 - 使用少量數據和步數"""
    logger.info("🧪 開始快速系統測試...")
    
    # 測試配置 - 使用較小的參數
    test_symbols = ['EUR_USD', 'USD_JPY']  # 只用2個symbols
    start_time, end_time = create_training_time_range(days_back=3)  # 只用3天數據
    
    logger.info(f"測試配置:")
    logger.info(f"  交易symbols: {test_symbols}")
    logger.info(f"  時間範圍: {start_time} 到 {end_time}")
    
    try:
        # 創建訓練器
        trainer = EnhancedUniversalTrainer(
            trading_symbols=test_symbols,
            start_time=start_time,
            end_time=end_time,
            granularity="S5",
            total_timesteps=500,  # 很少的步數，只是測試
            save_freq=100,
            eval_freq=200,
            model_name_prefix="test_sac_system"
        )
        
        # 只測試數據準備和環境設置
        logger.info("📊 測試數據準備...")
        if not trainer.prepare_data():
            logger.error("❌ 數據準備失敗")
            return False
        
        logger.info("🏗️ 測試環境設置...")
        if not trainer.setup_environment():
            logger.error("❌ 環境設置失敗")
            return False
        
        logger.info("🤖 測試智能體設置...")
        if not trainer.setup_agent():
            logger.error("❌ 智能體設置失敗")
            return False
        
        logger.info("📋 測試回調設置...")
        if not trainer.setup_callbacks():
            logger.error("❌ 回調設置失敗")
            return False
        
        # 測試環境重置和一個步驟
        logger.info("🔄 測試環境重置...")
        obs, info = trainer.env.reset()
        logger.info(f"✅ 環境重置成功，觀察形狀: {obs['features_from_dataset'].shape}")
        
        # 測試一個隨機動作
        logger.info("🎯 測試隨機動作...")
        action = trainer.env.action_space.sample()
        obs_next, reward, terminated, truncated, info = trainer.env.step(action)
        logger.info(f"✅ 動作執行成功，獎勵: {reward:.4f}")
        
        # 清理
        trainer.cleanup()
        
        logger.info("✅ 所有組件測試通過！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 測試失敗: {e}")
        logger.exception("詳細錯誤:")
        return False


def test_mini_training():
    """迷你訓練測試 - 運行很少的訓練步數"""
    logger.info("🚀 開始迷你訓練測試...")
    
    test_symbols = ['EUR_USD']  # 只用1個symbol
    start_time, end_time = create_training_time_range(days_back=2)  # 只用2天數據
    
    try:
        trainer = EnhancedUniversalTrainer(
            trading_symbols=test_symbols,
            start_time=start_time,
            end_time=end_time,
            granularity="S5",
            total_timesteps=100,  # 非常少的步數
            save_freq=50,
            eval_freq=100,
            model_name_prefix="mini_test_sac"
        )
        
        # 運行完整流程但步數很少
        success = trainer.run_full_training_pipeline()
        
        if success:
            logger.info("✅ 迷你訓練測試成功！")
            return True
        else:
            logger.error("❌ 迷你訓練測試失敗")
            return False
            
    except Exception as e:
        logger.error(f"❌ 迷你訓練測試失敗: {e}")
        logger.exception("詳細錯誤:")
        return False


def main():
    """主測試函數"""
    logger.info("=" * 60)
    logger.info("🔬 OANDA 交易系統完整測試")
    logger.info("=" * 60)
    
    # 測試1: 快速組件測試
    logger.info("\n" + "=" * 40)
    logger.info("測試 1: 快速組件測試")
    logger.info("=" * 40)
    
    if not test_quick_training():
        logger.error("❌ 快速測試失敗，停止後續測試")
        return 1
    
    # 測試2: 迷你訓練測試
    logger.info("\n" + "=" * 40)
    logger.info("測試 2: 迷你訓練測試")
    logger.info("=" * 40)
    
    if not test_mini_training():
        logger.error("❌ 迷你訓練測試失敗")
        return 1
    
    # 所有測試通過
    logger.info("\n" + "=" * 60)
    logger.info("🎉 所有測試通過！系統運行正常")
    logger.info("=" * 60)
    logger.info("✅ 數據管理系統正常")
    logger.info("✅ 貨幣轉換系統正常") 
    logger.info("✅ 交易環境正常")
    logger.info("✅ 智能體系統正常")
    logger.info("✅ 訓練流程正常")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)