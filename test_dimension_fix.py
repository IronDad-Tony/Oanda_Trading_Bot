#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試維度不匹配修復
驗證 trading_env.py 和 mmap_dataset.py 中的維度一致性修復是否有效
"""

import sys
import os
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 設置環境變量
os.environ['PYTHONPATH'] = str(project_root)

try:
    from src.common.config import TIMESTEPS, DEFAULT_SYMBOLS
    from src.common.logger_setup import logger
    from src.data_manager.mmap_dataset import UniversalMemoryMappedDataset
    from src.data_manager.instrument_info_manager import InstrumentInfoManager
    from src.environment.trading_env import UniversalTradingEnvV4
    from src.data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols
    from datetime import datetime, timezone
    
    logger.info("=== 維度不匹配修復測試開始 ===")
    
    # 測試參數
    test_symbols = ["EUR_USD", "USD_JPY"]
    test_start_datetime = datetime(2024, 5, 22, 10, 0, 0, tzinfo=timezone.utc)
    test_end_datetime = datetime(2024, 5, 22, 11, 0, 0, tzinfo=timezone.utc)
    test_start_iso = format_datetime_for_oanda(test_start_datetime)
    test_end_iso = format_datetime_for_oanda(test_end_datetime)
    test_granularity = "S5"
    
    logger.info(f"測試配置:")
    logger.info(f"  TIMESTEPS (配置): {TIMESTEPS}")
    logger.info(f"  測試交易對: {test_symbols}")
    logger.info(f"  時間範圍: {test_start_iso} 到 {test_end_iso}")
    logger.info(f"  粒度: {test_granularity}")
    
    # 1. 測試數據集維度一致性
    logger.info("\n--- 測試 1: 數據集維度一致性 ---")
    try:
        # 確保有測試數據
        logger.info("確保測試數據存在...")
        manage_data_download_for_symbols(
            symbols=test_symbols,
            overall_start_str=test_start_iso,
            overall_end_str=test_end_iso,
            granularity=test_granularity
        )
        
        # 創建數據集
        logger.info("創建測試數據集...")
        dataset = UniversalMemoryMappedDataset(
            symbols=test_symbols,
            start_time_iso=test_start_iso,
            end_time_iso=test_end_iso,
            granularity=test_granularity,
            timesteps_history=TIMESTEPS,
            force_reload=True
        )
        
        logger.info(f"數據集創建成功:")
        logger.info(f"  timesteps_history: {dataset.timesteps_history}")
        logger.info(f"  配置 TIMESTEPS: {TIMESTEPS}")
        logger.info(f"  數據集長度: {len(dataset)}")
        logger.info(f"  每個symbol特徵數: {dataset.num_features_per_symbol}")
        
        # 檢查維度一致性
        if dataset.timesteps_history == TIMESTEPS:
            logger.info("✅ 數據集 timesteps_history 與配置 TIMESTEPS 一致")
        else:
            logger.error(f"❌ 維度不一致: dataset.timesteps_history={dataset.timesteps_history}, TIMESTEPS={TIMESTEPS}")
        
        # 測試獲取樣本
        if len(dataset) > 0:
            sample = dataset[0]
            features_shape = sample['features'].shape
            raw_prices_shape = sample['raw_prices'].shape
            
            logger.info(f"樣本維度:")
            logger.info(f"  features: {features_shape}")
            logger.info(f"  raw_prices: {raw_prices_shape}")
            
            expected_features_shape = (len(test_symbols), TIMESTEPS, dataset.num_features_per_symbol)
            expected_prices_shape = (len(test_symbols), TIMESTEPS, 2)
            
            if features_shape == expected_features_shape:
                logger.info("✅ 特徵維度正確")
            else:
                logger.error(f"❌ 特徵維度錯誤: 期望 {expected_features_shape}, 實際 {features_shape}")
                
            if raw_prices_shape == expected_prices_shape:
                logger.info("✅ 價格維度正確")
            else:
                logger.error(f"❌ 價格維度錯誤: 期望 {expected_prices_shape}, 實際 {raw_prices_shape}")
        
        dataset.close()
        
    except Exception as e:
        logger.error(f"數據集測試失敗: {e}", exc_info=True)
    
    # 2. 測試交易環境維度處理
    logger.info("\n--- 測試 2: 交易環境維度處理 ---")
    try:
        # 重新創建數據集
        dataset = UniversalMemoryMappedDataset(
            symbols=test_symbols,
            start_time_iso=test_start_iso,
            end_time_iso=test_end_iso,
            granularity=test_granularity,
            timesteps_history=TIMESTEPS,
            force_reload=False
        )
        
        # 創建交易品種信息管理器
        instrument_manager = InstrumentInfoManager(force_refresh=False)
        
        # 創建交易環境
        logger.info("創建交易環境...")
        env = UniversalTradingEnvV4(
            dataset=dataset,
            instrument_info_manager=instrument_manager,
            active_symbols_for_episode=test_symbols
        )
        
        logger.info("交易環境創建成功")
        
        # 重置環境
        logger.info("重置環境...")
        obs, info = env.reset()
        
        logger.info(f"觀察空間維度:")
        for key, value in obs.items():
            logger.info(f"  {key}: {value.shape}")
        
        # 檢查特徵維度
        features_obs_shape = obs['features_from_dataset'].shape
        expected_obs_shape = (env.num_env_slots, dataset.timesteps_history, dataset.num_features_per_symbol)
        
        if features_obs_shape == expected_obs_shape:
            logger.info("✅ 環境觀察特徵維度正確")
        else:
            logger.error(f"❌ 環境觀察特徵維度錯誤: 期望 {expected_obs_shape}, 實際 {features_obs_shape}")
        
        # 執行一步
        logger.info("執行一個隨機動作...")
        action = env.action_space.sample()
        obs_next, reward, terminated, truncated, info = env.step(action)
        
        logger.info(f"步驟執行成功:")
        logger.info(f"  獎勵: {reward:.4f}")
        logger.info(f"  終止: {terminated}")
        logger.info(f"  截斷: {truncated}")
        
        env.close()
        dataset.close()
        
        logger.info("✅ 交易環境測試成功")
        
    except Exception as e:
        logger.error(f"交易環境測試失敗: {e}", exc_info=True)
    
    logger.info("\n=== 維度不匹配修復測試完成 ===")
    logger.info("如果看到所有 ✅ 標記，說明修復成功！")
    
except ImportError as e:
    print(f"導入錯誤: {e}")
    print("請確保已正確設置 PYTHONPATH 和項目結構")
    sys.exit(1)
except Exception as e:
    print(f"測試過程中發生錯誤: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)