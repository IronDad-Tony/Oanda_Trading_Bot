#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
元學習系統功能測試
測試MAML學習器、策略記憶庫和知識蒸餾器的功能
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from typing import Dict, List, Tuple

# 添加項目根目錄到路徑
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from src.agent.meta_learning_system import (
        MetaLearningSystem, 
        TaskDefinition, 
        MAMLLearner, 
        StrategyMemoryBank,
        KnowledgeDistiller
    )
    from src.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition
    from src.environment.progressive_reward_system import ProgressiveRewardSystem
except ImportError as e:
    print(f"導入錯誤: {e}")
    print("請確保在項目根目錄運行此測試")
    sys.exit(1)


def create_mock_strategy_model():
    """創建模擬策略模型"""
    class MockStrategyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.state_dim = 64
            self.num_strategies = 20
            self.action_dim = 3
            
            # 簡單的線性層
            self.fc = nn.Linear(self.state_dim, self.action_dim)
        
        def forward(self, x):
            return self.fc(x)
    
    return MockStrategyModel()


def test_strategy_memory_bank():
    """測試策略記憶庫"""
    print("\n=== 測試策略記憶庫 ===")
    
    memory_bank = StrategyMemoryBank(max_strategies=10)
    
    # 創建測試任務
    task1 = TaskDefinition(
        task_id="test_1",
        market_regime="trending",
        time_horizon="short",
        risk_profile="moderate",
        asset_class="forex"
    )
    
    task2 = TaskDefinition(
        task_id="test_2", 
        market_regime="ranging",
        time_horizon="medium",
        risk_profile="conservative",
        asset_class="forex"
    )
    
    # 添加策略
    embedding1 = torch.randn(128)
    parameters1 = {"weight": torch.randn(10, 5), "bias": torch.randn(10)}
    performance1 = {"sharpe_ratio": 1.5, "max_drawdown": 0.1}
    
    embedding2 = torch.randn(128)
    parameters2 = {"weight": torch.randn(10, 5), "bias": torch.randn(10)}
    performance2 = {"sharpe_ratio": 1.2, "max_drawdown": 0.15}
    
    memory_bank.add_strategy(embedding1, parameters1, performance1, task1)
    memory_bank.add_strategy(embedding2, parameters2, performance2, task2)
    
    print(f"✓ 成功添加2個策略到記憶庫")
    print(f"  記憶庫大小: {len(memory_bank.strategy_embeddings)}")
    
    # 檢索相似策略
    query_embedding = torch.randn(128)
    similar_strategies = memory_bank.retrieve_similar_strategies(
        query_embedding, task1, top_k=2
    )
    
    print(f"✓ 檢索到 {len(similar_strategies)} 個相似策略")
    for i, (idx, score, params) in enumerate(similar_strategies):
        print(f"  策略 {i+1}: 索引={idx}, 相似度={score:.4f}")
    
    # 獲取最佳策略
    best_strategies = memory_bank.get_best_strategies_for_task(task1, top_k=1)
    print(f"✓ 獲取到 {len(best_strategies)} 個最佳策略")
    
    return True


def test_maml_learner():
    """測試MAML學習器"""
    print("\n=== 測試MAML學習器 ===")
    
    # 創建基礎模型
    base_model = create_mock_strategy_model()
    
    # 創建MAML學習器
    maml = MAMLLearner(
        base_model=base_model,
        inner_lr=0.01,
        outer_lr=0.001,
        num_inner_steps=3
    )
    
    print(f"✓ 創建MAML學習器")
    print(f"  內層學習率: {maml.inner_lr}")
    print(f"  外層學習率: {maml.outer_lr}")
    print(f"  內層步數: {maml.num_inner_steps}")
    
    # 測試內層更新
    support_data = (torch.randn(10, 64), torch.randn(10, 3))
    
    def mock_loss_fn(model, data):
        x, y = data
        pred = model(x)
        return nn.MSELoss()(pred, y)
    
    try:
        fast_model = maml.inner_update(support_data, mock_loss_fn)
        print(f"✓ 內層更新成功")
        print(f"  快速適應模型類型: {type(fast_model).__name__}")
    except Exception as e:
        print(f"✗ 內層更新失敗: {e}")
        return False
    
    return True


def test_knowledge_distiller():
    """測試知識蒸餾器"""
    print("\n=== 測試知識蒸餾器 ===")
    
    # 創建教師和學生模型
    teacher_model = create_mock_strategy_model()
    student_model = create_mock_strategy_model()
    
    # 創建知識蒸餾器
    distiller = KnowledgeDistiller(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=3.0,
        alpha=0.7
    )
    
    print(f"✓ 創建知識蒸餾器")
    print(f"  溫度參數: {distiller.temperature}")
    print(f"  蒸餾權重: {distiller.alpha}")
    
    # 測試蒸餾損失計算
    student_logits = torch.randn(10, 3)
    teacher_logits = torch.randn(10, 3)
    true_labels = torch.randint(0, 3, (10,))
    
    try:
        distill_loss = distiller.compute_distillation_loss(
            student_logits, teacher_logits, true_labels
        )
        print(f"✓ 蒸餾損失計算成功: {distill_loss.item():.4f}")
    except Exception as e:
        print(f"✗ 蒸餾損失計算失敗: {e}")
        return False
    
    return True


def test_meta_learning_system():
    """測試完整的元學習系統"""
    print("\n=== 測試元學習系統 ===")
    
    # 創建策略模型
    strategy_model = EnhancedStrategySuperposition(
        state_dim=64,
        action_dim=3,
        num_strategies=20
    )
    
    # 創建漸進式獎勵系統
    reward_system = ProgressiveRewardSystem(
        state_dim=64,
        action_dim=3
    )
    
    # 創建元學習系統
    try:
        meta_system = MetaLearningSystem(
            strategy_model=strategy_model,
            reward_system=reward_system,
            config={
                'embedding_dim': 128,
                'memory_size': 100,
                'inner_lr': 0.01,
                'outer_lr': 0.001,
                'num_inner_steps': 5,
                'distill_temperature': 3.0,
                'distill_alpha': 0.7
            }
        )
        print(f"✓ 元學習系統創建成功")
    except Exception as e:
        print(f"✗ 元學習系統創建失敗: {e}")
        return False
    
    # 測試策略編碼
    try:
        state = torch.randn(1, 64)
        strategy_weights = torch.randn(1, 20)
        
        embedding = meta_system.encode_strategy(state, strategy_weights)
        print(f"✓ 策略編碼成功，嵌入維度: {embedding.shape}")
    except Exception as e:
        print(f"✗ 策略編碼失敗: {e}")
        return False
    
    # 測試任務編碼
    try:
        task = TaskDefinition(
            task_id="test_task",
            market_regime="trending", 
            time_horizon="short",
            risk_profile="moderate",
            asset_class="forex"
        )
        
        task_embedding = meta_system.encode_task(task)
        print(f"✓ 任務編碼成功，嵌入維度: {task_embedding.shape}")
    except Exception as e:
        print(f"✗ 任務編碼失敗: {e}")
        return False
    
    # 測試快速適應
    try:
        support_states = torch.randn(10, 64)
        support_actions = torch.randn(10, 3)
        query_states = torch.randn(5, 64)
        
        adapted_params = meta_system.fast_adapt(
            support_states, support_actions, query_states, task
        )
        print(f"✓ 快速適應成功，參數數量: {len(adapted_params)}")
    except Exception as e:
        print(f"✗ 快速適應失敗: {e}")
        return False
    
    # 測試參數分析
    try:
        analysis = meta_system.analyze_model_parameters()
        print(f"✓ 參數分析成功")
        print(f"  總參數量: {analysis['total_params']:,}")
        print(f"  策略編碼器參數: {analysis['strategy_encoder_params']:,}")
        print(f"  任務編碼器參數: {analysis['task_encoder_params']:,}")
    except Exception as e:
        print(f"✗ 參數分析失敗: {e}")
        return False
    
    return True


def main():
    """主測試函數"""
    print("🚀 開始元學習系統功能測試...")
    
    tests = [
        ("策略記憶庫", test_strategy_memory_bank),
        ("MAML學習器", test_maml_learner), 
        ("知識蒸餾器", test_knowledge_distiller),
        ("完整元學習系統", test_meta_learning_system)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}測試出現異常: {e}")
            results.append((test_name, False))
    
    # 輸出總結
    print("\n" + "="*50)
    print("📊 測試結果總結:")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\n通過率: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")
    
    if passed == len(results):
        print("🎉 所有測試通過！元學習系統功能正常")
        return True
    else:
        print("⚠️  部分測試失敗，請檢查問題")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
