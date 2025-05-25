#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速訓練測試腳本
用於測試系統的完整訓練流程
"""

import sys
import os
sys.path.append('src')

from datetime import datetime, timedelta
import pytz
from pathlib import Path

# 導入所有必要的模組
from common.config import *
from common.logger_setup import setup_logger
from data_manager.database_manager import DatabaseManager
from data_manager.instrument_info_manager import InstrumentInfoManager
from data_manager.mmap_dataset import UniversalMemoryMappedDataset
from environment.trading_env import UniversalTradingEnvV4
from agent.sac_agent_wrapper import SACAgentWrapper
from trainer.callbacks import UniversalCheckpointCallback

def main():
    """執行快速訓練測試"""
    
    # 設置日誌
    logger = setup_logger("test_training", LOGS_DIR / "test_training.log")
    logger.info("開始快速訓練測試...")
    
    # 測試參數
    TEST_SYMBOLS = ['EUR_USD', 'USD_JPY', 'GBP_USD']  # 3個測試symbols
    TEST_TIMESTEPS = 500  # 短期訓練步數
    
    # 設定訓練時間範圍（使用最近的數據）
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(days=2)  # 使用最近2天的數據
    
    train_start = start_time.isoformat()
    train_end = (end_time - timedelta(hours=6)).isoformat()  # 訓練到6小時前
    eval_start = (end_time - timedelta(hours=6)).isoformat()
    eval_end = end_time.isoformat()
    
    logger.info(f"測試symbols: {TEST_SYMBOLS}")
    logger.info(f"訓練時間範圍: {train_start} 到 {train_end}")
    logger.info(f"評估時間範圍: {eval_start} 到 {eval_end}")
    
    try:
        # 1. 初始化儀器信息管理器
        logger.info("初始化儀器信息管理器...")
        instrument_manager = InstrumentInfoManager()
        
        # 2. 創建訓練數據集
        logger.info("創建訓練數據集...")
        train_dataset = UniversalMemoryMappedDataset(
            symbols=TEST_SYMBOLS,
            start_time_iso=train_start,
            end_time_iso=train_end,
            timesteps=TIMESTEPS,
            force_reload=True  # 強制重新加載以確保使用最新數據
        )
        
        logger.info(f"訓練數據集大小: {len(train_dataset)}")
        
        # 3. 創建評估數據集
        logger.info("創建評估數據集...")
        eval_dataset = UniversalMemoryMappedDataset(
            symbols=TEST_SYMBOLS,
            start_time_iso=eval_start,
            end_time_iso=eval_end,
            timesteps=TIMESTEPS,
            force_reload=True
        )
        
        logger.info(f"評估數據集大小: {len(eval_dataset)}")
        
        # 4. 創建訓練環境
        logger.info("創建訓練環境...")
        train_env = UniversalTradingEnvV4(
            dataset=train_dataset,
            instrument_info_manager=instrument_manager,
            initial_capital=INITIAL_CAPITAL,
            max_episode_steps=min(1000, len(train_dataset) // 2),  # 限制episode長度
            symbols_for_episode=TEST_SYMBOLS
        )
        
        # 5. 創建評估環境
        logger.info("創建評估環境...")
        eval_env = UniversalTradingEnvV4(
            dataset=eval_dataset,
            instrument_info_manager=instrument_manager,
            initial_capital=INITIAL_CAPITAL,
            max_episode_steps=min(500, len(eval_dataset) // 2),
            symbols_for_episode=TEST_SYMBOLS
        )
        
        # 6. 創建SAC智能體
        logger.info("創建SAC智能體...")
        
        # 創建時間戳標識的模型保存目錄
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_save_dir = WEIGHTS_DIR / f"test_training_{timestamp}"
        model_save_dir.mkdir(exist_ok=True)
        
        agent_wrapper = SACAgentWrapper(
            env=train_env,
            model_save_dir=model_save_dir,
            tensorboard_log_dir=LOGS_DIR / f"test_tensorboard_{timestamp}",
            n_symbols=len(TEST_SYMBOLS),
            timesteps=TIMESTEPS
        )
        
        # 7. 創建回調
        logger.info("創建訓練回調...")
        callback = UniversalCheckpointCallback(
            save_freq=100,  # 每100步保存一次
            eval_freq=200,  # 每200步評估一次
            eval_env=eval_env,
            n_eval_episodes=2,  # 評估2個episode
            save_path=str(model_save_dir),
            name_prefix="test_model",
            verbose=1
        )
        
        # 8. 開始訓練
        logger.info(f"開始訓練 {TEST_TIMESTEPS} 步...")
        print(f"\n🚀 開始快速訓練測試!")
        print(f"📊 Symbols: {TEST_SYMBOLS}")
        print(f"⏱️  訓練步數: {TEST_TIMESTEPS}")
        print(f"💾 模型保存目錄: {model_save_dir}")
        print(f"📈 TensorBoard日誌: {LOGS_DIR / f'test_tensorboard_{timestamp}'}")
        print(f"\n您可以在另一個終端運行以下命令查看TensorBoard:")
        print(f"tensorboard --logdir {LOGS_DIR / f'test_tensorboard_{timestamp}'}")
        print(f"\n訓練開始...")
        
        # 執行訓練
        agent_wrapper.learn(
            total_timesteps=TEST_TIMESTEPS,
            callback=callback,
            progress_bar=True
        )
        
        # 9. 保存最終模型
        final_model_path = model_save_dir / "final_test_model.zip"
        agent_wrapper.save(str(final_model_path))
        
        logger.info("訓練完成!")
        print(f"\n✅ 訓練測試完成!")
        print(f"📁 最終模型保存在: {final_model_path}")
        
        # 10. 簡單測試模型
        logger.info("測試訓練好的模型...")
        obs, info = train_env.reset()
        for i in range(10):
            action, _states = agent_wrapper.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = train_env.step(action)
            print(f"步驟 {i+1}: 獎勵 = {reward:.4f}, 投資組合價值 = {info.get('portfolio_value_ac', 0):.2f}")
            
            if terminated or truncated:
                obs, info = train_env.reset()
        
        # 11. 渲染最終狀態
        logger.info("渲染環境狀態...")
        train_env.render()
        
        logger.info("測試完成!")
        print(f"\n🎉 所有測試完成! 系統運行正常!")
        
        return True
        
    except Exception as e:
        logger.error(f"訓練測試過程中出現錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理資源
        try:
            train_env.close()
            eval_env.close()
        except:
            pass

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 系統已準備好進行完整訓練!")
    else:
        print("\n❌ 測試過程中出現問題，請檢查日誌")