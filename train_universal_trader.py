#!/usr/bin/env python3
"""
訓練通用SAC交易模型腳本
1. 構建MMAP數據集
2. 創建環境（支持多符號通用模型）
3. 使用QuantumEnhancedSAC（已集成Transformer特徵提取器）訓練
4. 確保梯度從Actor/Critic到Transformer正常流動
"""
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from src.data_manager.mmap_dataset import UniversalMemoryMappedDataset
from src.data_manager.instrument_info_manager import InstrumentInfoManager
from src.environment.trading_env import UniversalTradingEnvV4
from src.agent.sac_agent_wrapper import QuantumEnhancedSAC
from src.common.config import DEFAULT_TRAIN_START_ISO, DEFAULT_TRAIN_END_ISO, DEFAULT_EVAL_START_ISO, DEFAULT_EVAL_END_ISO, DEFAULT_SYMBOLS

def make_env(symbols):
    # 創建MMAP數據集
    dataset = UniversalMemoryMappedDataset(
        symbols=symbols,
        start_time_iso=DEFAULT_TRAIN_START_ISO,
        end_time_iso=DEFAULT_TRAIN_END_ISO,
    )
    # 創建instrument info manager
    info_mgr = InstrumentInfoManager()
    # 創建環境實例
    env = UniversalTradingEnvV4(
        dataset=dataset,
        instrument_info_manager=info_mgr,
        active_symbols_for_episode=symbols
    )
    return env

if __name__ == '__main__':
    # 選擇要訓練的符號集，不超過MAX_SYMBOLS_ALLOWED
    symbols = DEFAULT_SYMBOLS  # 預設5個符號
    env = DummyVecEnv([lambda: make_env(symbols)])

    # 創建SAC智能體包裝器
    sac_wrapper = QuantumEnhancedSAC(env)

    # 訓練
    total_timesteps = 200_000
    print(f"開始訓練 {total_timesteps} 步驟...")
    # 調用Stable-Baselines3的learn方法
    sac_wrapper.agent.learn(total_timesteps=total_timesteps)
    print("訓練完成，模型權重保存在weights目錄。")
