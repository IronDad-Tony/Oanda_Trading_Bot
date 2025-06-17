"""
直接運行測試，驗證所有模組的流程和梯度流
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_complete_model_flow():
    """完整模型流程測試"""
    logger.info("開始完整模型流程測試...")
    
    try:
        # Import required modules
        from src.models.enhanced_transformer import EnhancedTransformer
        from src.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition
        from src.agent.meta_learning_system import MetaLearningSystem
        from src.environment.progressive_reward_system import ProgressiveLearningSystem
        
        logger.info("✓ 所有模組導入成功")
        
        # Setup device and parameters
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 4
        seq_len = 100
        num_features = 24
        num_symbols = 3
        
        logger.info(f"使用設備: {device}")
        
        # 1. 測試 Enhanced Transformer
        logger.info("1. 測試 Enhanced Transformer...")
        transformer_config = {
            'input_size': num_features,
            'hidden_size': 256,
            'num_layers': 4,
            'num_heads': 8,
            'num_symbols': num_symbols,
            'dropout': 0.1,
            'use_msfe': True,
            'use_wavelet': True,
            'use_fourier': True,
            'use_market_state_detection': True
        }
        
        transformer = EnhancedTransformer(**transformer_config).to(device)
        
        # 創建測試數據
        market_data = torch.randn(batch_size, seq_len, num_symbols, num_features, device=device)
        input_tensor = market_data.view(batch_size, seq_len, num_symbols * num_features)
        src_key_padding_mask = torch.zeros(batch_size, num_symbols, dtype=torch.bool, device=device)
        
        # Transformer前向傳播
        transformer_output = transformer(input_tensor, src_key_padding_mask=src_key_padding_mask)
        logger.info(f"✓ Transformer輸出形狀: {transformer_output.shape}")
        assert transformer_output.requires_grad, "Transformer輸出應該需要梯度"
        
        # 2. 測試量子策略層
        logger.info("2. 測試量子策略層...")
        quantum_layer = EnhancedStrategySuperposition(
            input_dim=256,
            num_strategies=5,
            dropout_rate=0.1,
            strategy_input_dim=256
        ).to(device)
        
        # 量子策略層前向傳播
        last_hidden = transformer_output[:, -1, :]  # 使用最後一個時間步
        strategy_output = quantum_layer(last_hidden)
        logger.info(f"✓ 量子策略層輸出形狀: {strategy_output.shape}")
        assert strategy_output.requires_grad, "量子策略層輸出應該需要梯度"
        
        # 3. 測試元學習系統
        logger.info("3. 測試元學習系統...")
        meta_learning = MetaLearningSystem(
            strategy_pool_size=10,
            adaptation_lr=1e-3,
            memory_size=1000,
            device=device
        )
        
        # 模擬策略表現數據
        strategy_performance = {
            'returns': np.random.randn(5),
            'sharpe_ratios': np.random.uniform(0.5, 2.0, 5),
            'max_drawdowns': np.random.uniform(0.05, 0.15, 5),
            'win_rates': np.random.uniform(0.4, 0.7, 5)
        }
        
        market_state = 1
        evaluation_results = meta_learning.evaluate_strategy_performance(strategy_performance, market_state)
        logger.info(f"✓ 元學習評估結果: {evaluation_results}")
        
        # 4. 測試漸進式獎勵系統
        logger.info("4. 測試漸進式獎勵系統...")
        reward_system = ProgressiveLearningSystem(
            initial_stage='simple',
            stage_criteria={'simple': 100, 'intermediate': 500}
        )
        
        # 計算獎勵
        trading_results = {
            'profit_loss': 0.05,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.08,
            'volatility': 0.15,
            'trade_count': 50
        }
        
        current_reward_func = reward_system.get_current_reward_function()
        reward = current_reward_func.calculate_reward(trading_results, market_state)
        logger.info(f"✓ 計算獎勵: {reward}")
        
        # 5. 測試梯度流
        logger.info("5. 測試梯度流和權重更新...")
        
        # 記錄初始權重
        initial_transformer_weights = {}
        initial_quantum_weights = {}
        
        for name, param in transformer.named_parameters():
            if param.requires_grad:
                initial_transformer_weights[name] = param.clone().detach()
        
        for name, param in quantum_layer.named_parameters():
            if param.requires_grad:
                initial_quantum_weights[name] = param.clone().detach()
        
        # 計算損失並反向傳播
        target = torch.randn_like(strategy_output)
        loss = nn.MSELoss()(strategy_output, target)
        
        logger.info(f"計算損失: {loss.item():.6f}")
        
        # 清除舊梯度
        transformer.zero_grad()
        quantum_layer.zero_grad()
        
        # 反向傳播
        loss.backward()
        
        # 檢查梯度
        transformer_gradients_exist = False
        quantum_gradients_exist = False
        
        for name, param in transformer.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 1e-8:
                    transformer_gradients_exist = True
                    logger.info(f"Transformer參數 {name} 梯度範數: {grad_norm:.8f}")
                    break
        
        for name, param in quantum_layer.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 1e-8:
                    quantum_gradients_exist = True
                    logger.info(f"量子策略層參數 {name} 梯度範數: {grad_norm:.8f}")
                    break
        
        logger.info(f"✓ Transformer梯度存在: {transformer_gradients_exist}")
        logger.info(f"✓ 量子策略層梯度存在: {quantum_gradients_exist}")
        
        # 優化器步驟
        optimizer = torch.optim.Adam(
            list(transformer.parameters()) + list(quantum_layer.parameters()),
            lr=1e-4
        )
        optimizer.step()
        
        # 檢查權重更新
        transformer_weights_updated = False
        quantum_weights_updated = False
        
        for name, param in transformer.named_parameters():
            if param.requires_grad and name in initial_transformer_weights:
                initial = initial_transformer_weights[name]
                current = param.detach()
                diff = torch.norm(current - initial).item()
                if diff > 1e-8:
                    transformer_weights_updated = True
                    logger.info(f"Transformer參數 {name} 權重變化: {diff:.8f}")
                    break
        
        for name, param in quantum_layer.named_parameters():
            if param.requires_grad and name in initial_quantum_weights:
                initial = initial_quantum_weights[name]
                current = param.detach()
                diff = torch.norm(current - initial).item()
                if diff > 1e-8:
                    quantum_weights_updated = True
                    logger.info(f"量子策略層參數 {name} 權重變化: {diff:.8f}")
                    break
        
        logger.info(f"✓ Transformer權重更新: {transformer_weights_updated}")
        logger.info(f"✓ 量子策略層權重更新: {quantum_weights_updated}")
        
        # 總結測試結果
        logger.info("\n" + "="*80)
        logger.info("完整模型流程測試結果總結")
        logger.info("="*80)
        logger.info(f"✓ Enhanced Transformer: 正常運行")
        logger.info(f"✓ 量子策略層: 正常運行")
        logger.info(f"✓ 元學習系統: 正常運行")
        logger.info(f"✓ 漸進式獎勵系統: 正常運行")
        logger.info(f"✓ 梯度計算: Transformer={transformer_gradients_exist}, 量子層={quantum_gradients_exist}")
        logger.info(f"✓ 權重更新: Transformer={transformer_weights_updated}, 量子層={quantum_weights_updated}")
        
        # 驗證所有關鍵要求
        all_tests_passed = all([
            transformer_gradients_exist,
            quantum_gradients_exist,
            transformer_weights_updated,
            quantum_weights_updated
        ])
        
        if all_tests_passed:
            logger.info("🎉 所有測試全部通過！整個流程運行正常，梯度流暢通，權重正確更新！")
        else:
            logger.error("❌ 部分測試未通過，請檢查相關模組")
        
        logger.info("="*80)
        
        return all_tests_passed
        
    except Exception as e:
        logger.error(f"測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_model_flow()
    if success:
        print("\n🎉 完整模型流程測試成功完成！")
    else:
        print("\n❌ 完整模型流程測試失敗！")
        sys.exit(1)
