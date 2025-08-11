"""
全面測試模型流程，包含所有模組的梯度流驗證
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any
import traceback

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

# Import required modules
try:
    from src.models.enhanced_transformer import EnhancedTransformer
    from src.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition
    from src.agent.meta_learning_system import MetaLearningSystem
    from src.environment.progressive_reward_system import ProgressiveLearningSystem
except ImportError as e:
    print(f"Import error: {e}")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompleteModelFlowTester:
    """完整模型流程測試器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.seq_len = 100
        self.num_features = 24
        self.num_symbols = 3
        self.action_dim = 3
        
        # 初始化所有模組
        self.setup_models()
        
    def setup_models(self):
        """設置所有模型組件"""
        logger.info("設置模型組件...")
        
        # 1. Enhanced Transformer
        self.transformer_config = {
            'input_size': self.num_features,
            'hidden_size': 256,
            'num_layers': 4,
            'num_heads': 8,
            'num_symbols': self.num_symbols,
            'dropout': 0.1,
            'use_msfe': True,
            'use_wavelet': True,
            'use_fourier': True,
            'use_market_state_detection': True
        }
        
        self.transformer = EnhancedTransformer(**self.transformer_config).to(self.device)
        
        # 2. Enhanced Strategy Superposition (Quantum Strategy Layer)
        self.quantum_layer = EnhancedStrategySuperposition(
            input_dim=256,
            num_strategies=5,
            dropout_rate=0.1,
            strategy_input_dim=256
        ).to(self.device)
        
        # 3. Meta Learning System
        self.meta_learning = MetaLearningSystem(
            strategy_pool_size=10,
            adaptation_lr=1e-3,
            memory_size=1000,
            device=self.device
        )
        
        # 4. Progressive Reward System
        self.reward_system = ProgressiveLearningSystem(
            initial_stage='simple',
            stage_criteria={'simple': 100, 'intermediate': 500}
        )
        
        logger.info("所有模型組件設置完成")
    
    def create_sample_data(self) -> Dict[str, torch.Tensor]:
        """創建樣本數據"""
        # 市場數據
        market_data = torch.randn(
            self.batch_size, 
            self.seq_len, 
            self.num_symbols,
            self.num_features,
            device=self.device
        )
        
        # Padding mask for symbols
        src_key_padding_mask = torch.zeros(
            self.batch_size,
            self.num_symbols,
            dtype=torch.bool,
            device=self.device
        )
        
        # 市場狀態
        market_state = torch.randint(0, 3, (self.batch_size,), device=self.device)
        
        # 獎勵
        rewards = torch.randn(self.batch_size, device=self.device)
        
        return {
            'market_data': market_data,
            'src_key_padding_mask': src_key_padding_mask,
            'market_state': market_state,
            'rewards': rewards
        }
    
    def test_transformer_forward_pass(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """測試Transformer前向傳播"""
        logger.info("測試Transformer前向傳播...")
        
        try:
            # 重塑輸入數據
            batch_size, seq_len, num_symbols, num_features = data['market_data'].shape
            input_tensor = data['market_data'].view(batch_size, seq_len, num_symbols * num_features)
            
            # 前向傳播
            output = self.transformer(
                input_tensor,
                src_key_padding_mask=data['src_key_padding_mask']
            )
            
            logger.info(f"Transformer輸出形狀: {output.shape}")
            assert output.shape == (batch_size, seq_len, 256), f"期望形狀 {(batch_size, seq_len, 256)}, 實際 {output.shape}"
            assert output.requires_grad, "Transformer輸出應該需要梯度"
            
            logger.info("✓ Transformer前向傳播測試通過")
            return output
            
        except Exception as e:
            logger.error(f"✗ Transformer前向傳播測試失敗: {e}")
            raise
    
    def test_quantum_strategy_layer(self, transformer_output: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """測試量子策略層"""
        logger.info("測試量子策略層...")
        
        try:
            # 使用最後一個時間步的輸出
            last_hidden = transformer_output[:, -1, :]  # [batch, hidden_size]
            
            # 前向傳播
            strategy_output = self.quantum_layer(last_hidden)
            
            logger.info(f"量子策略層輸出形狀: {strategy_output.shape}")
            assert strategy_output.requires_grad, "量子策略層輸出應該需要梯度"
            
            logger.info("✓ 量子策略層測試通過")
            return strategy_output
            
        except Exception as e:
            logger.error(f"✗ 量子策略層測試失敗: {e}")
            raise
    
    def test_meta_learning_adaptation(self, data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """測試元學習適應"""
        logger.info("測試元學習適應...")
        
        try:
            # 模擬策略表現數據
            strategy_performance = {
                'returns': np.random.randn(5),
                'sharpe_ratios': np.random.uniform(0.5, 2.0, 5),
                'max_drawdowns': np.random.uniform(0.05, 0.15, 5),
                'win_rates': np.random.uniform(0.4, 0.7, 5)
            }
            
            # 評估策略表現
            evaluation_results = self.meta_learning.evaluate_strategy_performance(
                strategy_performance,
                data['market_state'][0].item()
            )
            
            logger.info(f"策略評估結果: {evaluation_results}")
            assert 'overall_score' in evaluation_results, "缺少整體評分"
            
            logger.info("✓ 元學習適應測試通過")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"✗ 元學習適應測試失敗: {e}")
            raise
    
    def test_gradient_flow(self, data: Dict[str, torch.Tensor]) -> Dict[str, bool]:
        """測試梯度流和權重更新"""
        logger.info("測試梯度流和權重更新...")
        
        try:
            results = {}
            
            # 記錄初始權重
            initial_weights = {}
            for name, param in self.transformer.named_parameters():
                if param.requires_grad:
                    initial_weights[f"transformer_{name}"] = param.clone().detach()
            
            for name, param in self.quantum_layer.named_parameters():
                if param.requires_grad:
                    initial_weights[f"quantum_{name}"] = param.clone().detach()
            
            # 完整前向傳播
            market_data_reshaped = data['market_data'].view(
                self.batch_size, self.seq_len, self.num_symbols * self.num_features
            )
            
            transformer_output = self.transformer(
                market_data_reshaped,
                src_key_padding_mask=data['src_key_padding_mask']
            )
            
            last_hidden = transformer_output[:, -1, :]
            strategy_output = self.quantum_layer(last_hidden)
            
            # 計算損失
            target = torch.randn_like(strategy_output)
            loss = nn.MSELoss()(strategy_output, target)
            
            logger.info(f"計算損失: {loss.item()}")
            
            # 反向傳播
            loss.backward()
            
            # 檢查梯度
            transformer_has_gradients = False
            for name, param in self.transformer.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 1e-8:
                        transformer_has_gradients = True
                        break
            
            quantum_has_gradients = False
            for name, param in self.quantum_layer.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 1e-8:
                        quantum_has_gradients = True
                        break
            
            results['transformer_gradients'] = transformer_has_gradients
            results['quantum_gradients'] = quantum_has_gradients
            
            # 模擬優化器步驟
            optimizer = torch.optim.Adam(
                list(self.transformer.parameters()) + list(self.quantum_layer.parameters()),
                lr=1e-4
            )
            optimizer.step()
            
            # 檢查權重更新
            weights_updated = False
            for name, param in self.transformer.named_parameters():
                if param.requires_grad and f"transformer_{name}" in initial_weights:
                    initial = initial_weights[f"transformer_{name}"]
                    current = param.detach()
                    diff = torch.norm(current - initial).item()
                    if diff > 1e-8:
                        weights_updated = True
                        break
            
            results['weights_updated'] = weights_updated
            
            logger.info(f"梯度檢查結果: transformer_gradients={transformer_has_gradients}, quantum_gradients={quantum_has_gradients}, weights_updated={weights_updated}")
            logger.info("✓ 梯度流和權重更新測試通過")
            
            return results
            
        except Exception as e:
            logger.error(f"✗ 梯度流和權重更新測試失敗: {e}")
            traceback.print_exc()
            raise
    
    def test_progressive_reward_system(self, data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """測試漸進式獎勵系統"""
        logger.info("測試漸進式獎勵系統...")
        
        try:
            # 模擬交易結果
            trading_results = {
                'profit_loss': data['rewards'].cpu().numpy(),
                'sharpe_ratio': np.random.uniform(0.5, 2.0, self.batch_size),
                'max_drawdown': np.random.uniform(0.05, 0.15, self.batch_size),
                'volatility': np.random.uniform(0.1, 0.3, self.batch_size),
                'trade_count': np.random.randint(10, 100, self.batch_size)
            }
            
            # 計算當前階段獎勵
            current_reward_func = self.reward_system.get_current_reward_function()
            rewards = []
            
            for i in range(self.batch_size):
                batch_results = {k: v[i] if isinstance(v, np.ndarray) else v for k, v in trading_results.items()}
                reward = current_reward_func.calculate_reward(batch_results, data['market_state'][i].item())
                rewards.append(reward)
            
            rewards = np.array(rewards)
            
            logger.info(f"當前階段: {self.reward_system.current_stage}")
            logger.info(f"計算獎勵: {rewards}")
            
            results = {
                'current_stage': self.reward_system.current_stage,
                'rewards': rewards.tolist(),
                'mean_reward': np.mean(rewards)
            }
            
            logger.info("✓ 漸進式獎勵系統測試通過")
            return results
            
        except Exception as e:
            logger.error(f"✗ 漸進式獎勵系統測試失敗: {e}")
            raise
    
    def run_complete_test(self) -> Dict[str, Any]:
        """運行完整的測試流程"""
        logger.info("開始完整模型流程測試...")
        
        test_results = {}
        
        try:
            # 1. 創建測試數據
            data = self.create_sample_data()
            logger.info("✓ 測試數據創建完成")
            
            # 2. 測試Transformer
            transformer_output = self.test_transformer_forward_pass(data)
            test_results['transformer'] = True
            
            # 3. 測試量子策略層
            strategy_output = self.test_quantum_strategy_layer(transformer_output, data)
            test_results['quantum_strategy'] = True
            
            # 4. 測試元學習
            meta_results = self.test_meta_learning_adaptation(data)
            test_results['meta_learning'] = True
            test_results['meta_results'] = meta_results
            
            # 5. 測試梯度流
            gradient_results = self.test_gradient_flow(data)
            test_results['gradient_flow'] = True
            test_results['gradient_results'] = gradient_results
            
            # 6. 測試漸進式獎勵系統
            reward_results = self.test_progressive_reward_system(data)
            test_results['progressive_reward'] = True
            test_results['reward_results'] = reward_results
            
            # 整體成功
            test_results['overall_success'] = True
            test_results['all_tests_passed'] = all([
                test_results['transformer'],
                test_results['quantum_strategy'],
                test_results['meta_learning'],
                test_results['gradient_flow'],
                test_results['progressive_reward']
            ])
            
            logger.info("🎉 完整模型流程測試全部通過！")
            
        except Exception as e:
            test_results['overall_success'] = False
            test_results['error'] = str(e)
            test_results['traceback'] = traceback.format_exc()
            logger.error(f"完整模型流程測試失敗: {e}")
            
        return test_results


def test_complete_model_flow():
    """完整模型流程測試函數"""
    tester = CompleteModelFlowTester()
    results = tester.run_complete_test()
    
    # 打印詳細結果
    print("\n" + "="*80)
    print("完整模型流程測試結果")
    print("="*80)
    
    for key, value in results.items():
        if key not in ['gradient_results', 'meta_results', 'reward_results']:
            print(f"{key}: {value}")
    
    print("\n詳細結果:")
    if 'gradient_results' in results:
        print(f"梯度流結果: {results['gradient_results']}")
    if 'meta_results' in results:
        print(f"元學習結果: {results['meta_results']}")
    
    print("="*80)
    
    # 斷言測試結果
    assert results.get('overall_success', False), f"測試失敗: {results.get('error', 'Unknown error')}"
    assert results.get('all_tests_passed', False), "不是所有測試都通過"
    
    # 特別檢查梯度流
    gradient_results = results.get('gradient_results', {})
    assert gradient_results.get('transformer_gradients', False), "Transformer沒有計算梯度"
    assert gradient_results.get('quantum_gradients', False), "量子策略層沒有計算梯度"
    assert gradient_results.get('weights_updated', False), "權重沒有更新"
    
    print("✅ 所有測試檢查通過！")


if __name__ == "__main__":
    # 運行測試
    test_complete_model_flow()
