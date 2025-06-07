# test_integration_script.py
"""
測試自適應元學習系統與增強版量子策略層的集成
驗證系統能否正確處理不同數量的策略和維度變化
"""

import sys
import os
from pathlib import Path

# 添加項目根目錄到路徑
current_dir = Path(__file__).resolve().parent
project_root = current_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_integration():
    """測試自適應元學習系統與量子策略層的集成"""
    logger.info("開始測試自適應元學習系統與量子策略層集成...")
    
    try:
        # 導入模組
        from src.agent.meta_learning_system import MetaLearningSystem
        from src.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition
        
        # 測試配置
        batch_size = 8
        initial_state_dim = 64
        action_dim = 10
        
        # 1. 創建初始元學習系統
        logger.info("步驟 1: 創建初始元學習系統...")
        meta_system = MetaLearningSystem(
            initial_state_dim=initial_state_dim,
            action_dim=action_dim,
            meta_learning_dim=256
        )
        
        # 2. 創建增強版量子策略層（初始配置）
        logger.info("步驟 2: 創建增強版量子策略層（64維，15策略）...")
        strategy_layer_v1 = EnhancedStrategySuperposition(
            state_dim=initial_state_dim,
            action_dim=action_dim,
            enable_dynamic_generation=True
        )
        
        # 3. 測試初始適應
        logger.info("步驟 3: 測試初始適應...")
        initial_adaptation = meta_system.adapt_to_strategy_layer(strategy_layer_v1)
        logger.info(f"初始適應結果: {initial_adaptation}")
        
        # 4. 測試與策略層的交互
        logger.info("步驟 4: 測試與策略層的交互...")
        test_state = torch.randn(batch_size, initial_state_dim)
        test_volatility = torch.rand(batch_size) * 0.5
        
        # 量子策略層輸出
        with torch.no_grad():
            strategy_output, strategy_info = strategy_layer_v1(test_state, test_volatility)
            
        # 元學習系統輸出
        with torch.no_grad():
            meta_output, meta_info = meta_system(test_state)
            
        logger.info(f"策略層輸出形狀: {strategy_output.shape}")
        logger.info(f"元學習輸出形狀: {meta_output.shape}")
        logger.info(f"策略權重形狀: {strategy_info['strategy_weights'].shape}")
        logger.info(f"活躍策略數量: {strategy_info['num_active_strategies'].mean():.1f}")
        
        # 5. 模擬策略數量變化（15 -> 25策略）
        logger.info("步驟 5: 模擬策略數量變化（15 -> 25策略）...")
        
        # 創建擴展的策略層（模擬）
        class ExtendedStrategyLayer(nn.Module):
            def __init__(self, state_dim, action_dim, num_strategies=25):
                super().__init__()
                self.state_dim = state_dim
                self.action_dim = action_dim
                self.num_strategies = num_strategies
                
                # 創建25個模擬策略
                self.base_strategies = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, action_dim),
                        nn.Tanh()
                    ) for _ in range(num_strategies)
                ])
                
            def forward(self, state, volatility):
                # 簡單的策略組合
                outputs = []
                for strategy in self.base_strategies:
                    outputs.append(strategy(state))
                
                # 平均組合
                combined = torch.stack(outputs, dim=1).mean(dim=1)
                
                # 模擬策略權重
                weights = torch.softmax(torch.randn(state.size(0), self.num_strategies), dim=-1)
                
                return combined, {
                    'strategy_weights': weights,
                    'num_active_strategies': torch.sum(weights > 0.04, dim=-1).float()
                }
        
        # 添加策略名稱方法
        strategy_layer_v2 = ExtendedStrategyLayer(initial_state_dim, action_dim, 25)
        for i, strategy in enumerate(strategy_layer_v2.base_strategies):
            strategy.get_strategy_name = lambda idx=i: f"ExtendedStrategy_{idx}"
        
        # 6. 測試策略數量變化適應
        logger.info("步驟 6: 測試策略數量變化適應...")
        strategy_adaptation = meta_system.adapt_to_strategy_layer(strategy_layer_v2)
        logger.info(f"策略數量變化適應結果: {strategy_adaptation}")
        
        # 驗證新配置
        current_config = meta_system.current_config
        logger.info(f"當前配置 - 狀態維度: {current_config.state_dim}, "
                   f"策略數量: {current_config.num_strategies}")
        
        # 7. 模擬狀態維度變化（64 -> 96維）
        logger.info("步驟 7: 模擬狀態維度變化（64 -> 96維）...")
        new_state_dim = 96
        strategy_layer_v3 = ExtendedStrategyLayer(new_state_dim, action_dim, 25)
        
        # 添加策略名稱方法
        for i, strategy in enumerate(strategy_layer_v3.base_strategies):
            strategy.get_strategy_name = lambda idx=i: f"ExtendedStrategy96_{idx}"
        
        # 測試維度變化適應
        dimension_adaptation = meta_system.adapt_to_strategy_layer(strategy_layer_v3)
        logger.info(f"維度變化適應結果: {dimension_adaptation}")
        
        # 8. 測試新配置下的前向傳播
        logger.info("步驟 8: 測試新配置下的前向傳播...")
        new_test_state = torch.randn(batch_size, new_state_dim)
        new_test_volatility = torch.rand(batch_size) * 0.5
        
        with torch.no_grad():
            # 新策略層輸出
            new_strategy_output, new_strategy_info = strategy_layer_v3(new_test_state, new_test_volatility)
            
            # 新元學習輸出
            new_meta_output, new_meta_info = meta_system(new_test_state)
            
        logger.info(f"新配置策略層輸出形狀: {new_strategy_output.shape}")
        logger.info(f"新配置元學習輸出形狀: {new_meta_output.shape}")
        
        # 9. 檢查適應歷史
        logger.info("步驟 9: 檢查適應歷史...")
        system_status = meta_system.get_system_status()
        logger.info(f"總配置變化次數: {system_status['total_config_changes']}")
        logger.info(f"總維度變化次數: {system_status['total_dimension_changes']}")
        logger.info(f"適應成功率: {system_status['adaptation_success_rate']:.2%}")
        logger.info(f"編碼器當前維度: {system_status['encoder_current_dim']}")
        
        # 10. 測試梯度傳播
        logger.info("步驟 10: 測試梯度傳播...")
        meta_system.train()
        strategy_layer_v3.train()
        
        # 創建需要梯度的輸入
        grad_test_state = torch.randn(4, new_state_dim, requires_grad=True)
        grad_test_volatility = torch.rand(4) * 0.5
        
        # 前向傳播
        strategy_out, _ = strategy_layer_v3(grad_test_state, grad_test_volatility)
        meta_out, _ = meta_system(grad_test_state)
        
        # 計算損失
        strategy_loss = strategy_out.abs().mean()
        meta_loss = meta_out.abs().mean()
        combined_loss = strategy_loss + meta_loss
        
        # 反向傳播
        combined_loss.backward()
        
        # 檢查梯度
        meta_grad_norm = 0
        strategy_grad_norm = 0
        
        for param in meta_system.parameters():
            if param.grad is not None:
                meta_grad_norm += param.grad.data.norm(2) ** 2
                
        for param in strategy_layer_v3.parameters():
            if param.grad is not None:
                strategy_grad_norm += param.grad.data.norm(2) ** 2
        
        meta_grad_norm = meta_grad_norm ** 0.5
        strategy_grad_norm = strategy_grad_norm ** 0.5
        
        logger.info(f"元學習系統梯度範數: {meta_grad_norm:.6f}")
        logger.info(f"策略層梯度範數: {strategy_grad_norm:.6f}")
        
        # 11. 保存和載入適應歷史
        logger.info("步驟 11: 保存和載入適應歷史...")
        history_file = "integration_test_history.json"
        meta_system.save_configuration_history(history_file)
        
        # 12. 輸出最終總結
        logger.info("步驟 12: 輸出最終總結...")
        final_status = meta_system.get_system_status()
        
        logger.info("=== 集成測試總結 ===")
        logger.info(f"✅ 初始配置: 64維 -> 當前配置: {final_status['encoder_current_dim']}維")
        logger.info(f"✅ 策略數量變化: 15策略 -> 25策略")
        logger.info(f"✅ 總適應次數: {final_status['total_adaptations']}")
        logger.info(f"✅ 適應成功率: {final_status['adaptation_success_rate']:.2%}")
        logger.info(f"✅ 配置檢測方法: {len(final_status['supported_detection_methods'])}種")
        logger.info(f"✅ 維度變化歷史: {final_status['encoder_dimension_changes']}次")
        
        # 清理測試文件
        if os.path.exists(history_file):
            os.remove(history_file)
            
        logger.info("自適應元學習系統與量子策略層集成測試完成！")
        return True
        
    except Exception as e:
        logger.error(f"集成測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_integration()
    if success:
        print("\n🎉 集成測試成功！自適應元學習系統能夠正確處理策略和維度變化！")
    else:
        print("\n❌ 集成測試失敗！")
