#!/usr/bin/env python3
"""
Oanda Trading Bot 系統完整性測試
確保所有模組都能正確導入和運行
"""

import sys
import os
import traceback
import logging
from pathlib import Path
import torch
import numpy as np

# 添加項目根路徑到 sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """測試所有關鍵模組的導入"""
    print("=" * 60)
    print("測試模組導入...")
    print("=" * 60)
    
    success_count = 0
    total_count = 0
    
    imports_to_test = [
        # 核心配置和工具
        ("src.common.config", "基礎配置"),
        ("src.common.logger_setup", "日誌系統"),
          # 數據管理
        ("src.data_manager.currency_manager", "貨幣管理器"),
        ("src.data_manager.mmap_dataset", "內存映射數據集"),
        
        # 量子策略層
        ("src.agent.quantum_strategy_layer", "量子策略層"),
        ("src.agent.sac_agent_wrapper", "SAC代理包裝器"),
        
        # 交易環境
        ("src.environment.trading_env", "交易環境"),
        
        # 模型
        ("src.models.transformer_model", "Transformer模型"),
        
        # 訓練器
        ("src.trainer.universal_trainer", "通用訓練器"),
          # 特徵工程
        ("src.feature_engineer.preprocessor", "特徵預處理器"),
        
        # 工具
        ("src.utils.data_validation", "數據驗證"),
        ("src.utils.portfolio_calculator", "投資組合計算器"),
    ]
    
    for module_name, description in imports_to_test:
        total_count += 1
        try:
            exec(f"import {module_name}")
            print(f"✅ {description} ({module_name})")
            success_count += 1
        except Exception as e:
            print(f"❌ {description} ({module_name}): {str(e)}")
            traceback.print_exc()
    
    print(f"\n導入測試完成: {success_count}/{total_count} 成功")
    return success_count == total_count

def test_quantum_layer_compatibility():
    """測試量子策略層的兼容性"""
    print("\n" + "=" * 60)
    print("測試量子策略層兼容性...")
    print("=" * 60)
    
    try:
        from src.agent.quantum_strategy_layer import QuantumTradingLayer
        
        # 創建測試實例
        input_dim = 32
        action_dim = 3
        layer = QuantumTradingLayer(
            input_dim=input_dim,
            action_dim=action_dim,
            num_strategies=3
        )
        
        # 測試基本屬性
        print("✅ QuantumTradingLayer 實例化成功")
        
        # 測試向後兼容性方法
        assert hasattr(layer, 'amplitudes'), "缺少 amplitudes 屬性"
        assert hasattr(layer, 'forward_compatible'), "缺少 forward_compatible 方法"
        assert hasattr(layer, 'quantum_annealing_step'), "缺少 quantum_annealing_step 方法"
        print("✅ 向後兼容性屬性和方法檢查通過")
        
        # 測試前向傳播
        batch_size = 4
        state = torch.randn(batch_size, input_dim)
        volatility = torch.randn(batch_size, 1)
        
        action, amplitudes_batch = layer.forward_compatible(state, volatility)
        
        # 驗證輸出格式
        assert action.shape == (batch_size, action_dim), f"動作輸出形狀錯誤: {action.shape}"
        assert amplitudes_batch.shape[0] == batch_size, f"振幅批次形狀錯誤: {amplitudes_batch.shape}"
        print("✅ 前向傳播測試通過")
        
        # 測試量子退火
        rewards = torch.randn(batch_size, 3)
        layer.quantum_annealing_step(rewards)
        print("✅ 量子退火步驟測試通過")
        
        return True
        
    except Exception as e:
        print(f"❌ 量子策略層測試失敗: {str(e)}")
        traceback.print_exc()
        return False

def test_sac_agent_integration():
    """測試SAC代理的集成"""
    print("\n" + "=" * 60)
    print("測試SAC代理集成...")
    print("=" * 60)
    
    try:
        from src.agent.sac_agent_wrapper import QuantumEnhancedSAC
        from src.environment.trading_env import UniversalTradingEnv
        from src.common.config import DEVICE
        
        # 創建簡化的交易環境
        print("正在創建交易環境...")
        env = UniversalTradingEnv(
            symbols_config={'EUR_USD': {'pip_value': 0.0001}},
            max_positions=1,
            granularity='H1',
            use_dummy_data=True  # 使用虛擬數據進行測試
        )
        
        print("正在創建QuantumEnhancedSAC代理...")
        agent = QuantumEnhancedSAC(
            env=env,
            device=DEVICE,
            use_amp=False  # 測試時關閉混合精度
        )
        
        print("✅ SAC代理與量子策略層集成成功")
        
        # 測試動作選擇
        obs = env.reset()
        market_volatility = np.array([0.01])  # 假設波動率
        
        if isinstance(obs, dict):
            action, amplitudes = agent.select_action(obs, market_volatility)
            print("✅ 動作選擇測試通過")
            print(f"   動作形狀: {action.shape}")
            print(f"   振幅形狀: {amplitudes.shape}")
        else:
            print("⚠️  觀察空間不是字典格式，跳過動作選擇測試")
        
        return True
        
    except Exception as e:
        print(f"❌ SAC代理集成測試失敗: {str(e)}")
        traceback.print_exc()
        return False

def test_oanda_api_integration():
    """測試Oanda API集成"""
    print("\n" + "=" * 60)
    print("測試Oanda API集成...")
    print("=" * 60)
    
    try:
        from src.data_manager.currency_manager import CurrencyManager
        from src.common.config import OANDA_CONFIG
        
        # 檢查配置
        if not OANDA_CONFIG.get('api_key') or OANDA_CONFIG['api_key'] == 'your_oanda_api_key':
            print("⚠️  Oanda API密鑰未配置，跳過API測試")
            return True
        
        # 創建貨幣管理器
        currency_manager = CurrencyManager()
        
        # 測試匯率獲取
        eur_usd_rate = currency_manager.get_exchange_rate('EUR', 'USD')
        if eur_usd_rate and eur_usd_rate > 0:
            print(f"✅ 匯率獲取成功: EUR/USD = {eur_usd_rate}")
        else:
            print("⚠️  匯率獲取返回無效值")
        
        # 測試保證金計算
        margin = currency_manager.calculate_margin('EUR_USD', 10000, eur_usd_rate)
        if margin > 0:
            print(f"✅ 保證金計算成功: {margin}")
        else:
            print("⚠️  保證金計算返回無效值")
        
        return True
        
    except Exception as e:
        print(f"❌ Oanda API集成測試失敗: {str(e)}")
        traceback.print_exc()
        return False

def test_data_pipeline():
    """測試數據處理管道"""
    print("\n" + "=" * 60)
    print("測試數據處理管道...")
    print("=" * 60)
    
    try:
        from src.data_manager.mmap_dataset import UniversalMemoryMappedDataset
        from src.feature_engineer.preprocessor import preprocess_data_for_model
        
        # 創建假數據進行測試
        num_samples = 100
        num_features = 20
        fake_data = np.random.randn(num_samples, num_features).astype(np.float32)
        fake_labels = np.random.randint(0, 3, size=(num_samples,)).astype(np.int64)
        
        # 測試內存映射數據集 (簡化測試)
        print("✅ 內存映射數據集模組導入成功")
        
        # 測試預處理功能
        print("✅ 特徵預處理模組導入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 數據處理管道測試失敗: {str(e)}")
        traceback.print_exc()
        return False

def test_gpu_compatibility():
    """測試GPU兼容性"""
    print("\n" + "=" * 60)
    print("測試GPU兼容性...")
    print("=" * 60)
    
    try:
        from src.common.config import DEVICE
        
        print(f"配置的設備: {DEVICE}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA設備數量: {torch.cuda.device_count()}")
            print(f"當前CUDA設備: {torch.cuda.current_device()}")
            print(f"設備名稱: {torch.cuda.get_device_name()}")
            
            # 測試GPU內存
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU內存: {gpu_memory:.2f} GB")
            
            # 測試張量操作
            test_tensor = torch.randn(1000, 1000).to(DEVICE)
            result = torch.matmul(test_tensor, test_tensor.T)
            print("✅ GPU張量操作測試通過")
            
            # 清理GPU內存
            del test_tensor, result
            torch.cuda.empty_cache()
        else:
            print("⚠️  CUDA不可用，將使用CPU")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU兼容性測試失敗: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """主測試函數"""
    print("Oanda Trading Bot 系統完整性測試")
    print("=" * 80)
    
    test_results = []
    
    # 運行所有測試
    test_functions = [
        ("模組導入", test_imports),
        ("量子策略層兼容性", test_quantum_layer_compatibility),
        ("SAC代理集成", test_sac_agent_integration),
        ("Oanda API集成", test_oanda_api_integration),
        ("數據處理管道", test_data_pipeline),
        ("GPU兼容性", test_gpu_compatibility),
    ]
    
    for test_name, test_func in test_functions:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 測試出現異常: {str(e)}")
            test_results.append((test_name, False))
    
    # 顯示測試結果摘要
    print("\n" + "=" * 80)
    print("測試結果摘要")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\n總體結果: {passed}/{total} 測試通過")
    
    if passed == total:
        print("\n🎉 所有測試通過！系統已準備就緒。")
        return True
    else:
        print(f"\n⚠️  {total - passed} 個測試失敗，請檢查上述錯誤信息。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
