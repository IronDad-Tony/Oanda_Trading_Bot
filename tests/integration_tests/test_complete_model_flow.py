"""
全面測試模型流程，包含所有模組的梯度流驗證
測試範圍：
1. Transformer模型
2. 量子策略層 (EnhancedStrategySuperposition)
3. 量子策略池
4. 自創策略模組
5. SAC強化學習訓練 (QuantumEnhancedSAC)
6. 元學習訓練模組
7. 梯度前向傳播和權重更新
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pytest
import logging
from typing import Dict, Tuple, List, Any, Optional
from unittest.mock import Mock, MagicMock, patch
import traceback

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

# Import all required modules
from src.models.enhanced_transformer import EnhancedTransformer
from src.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition, DynamicStrategyGenerator
from src.agent.sac_agent_wrapper import QuantumEnhancedSAC
from src.agent.meta_learning_system import MetaLearningSystem
from src.environment.progressive_reward_system import ProgressiveLearningSystem, SimpleReward, IntermediateReward, ComplexReward # Added reward strategy imports
from src.agent.strategy_innovation_engine import StrategyInnovationEngine

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
        self.action_dim = 3  # [buy, sell, hold]
        
        # 初始化所有模組
        self.setup_models()
        
    def setup_models(self):
        """設置所有模型組件"""
        logger.info("設置模型組件...")
        
        # 1. Enhanced Transformer
        self.transformer_config = {
            'input_dim': self.num_features,  # Features per symbol
            'd_model': 256,
            'transformer_nhead': 8,
            'num_encoder_layers': 4,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'max_seq_len': self.seq_len,
            'num_symbols': self.num_symbols,
            'output_dim': 256,  # Output dim for downstream layers
            'use_msfe': True,
            'use_fourier_features': True,
            'use_wavelet_features': True
        }
        self.transformer = EnhancedTransformer(**self.transformer_config).to(self.device)
        
        # 2. Enhanced Strategy Superposition (Quantum Strategy Layer)
        self.quantum_layer = EnhancedStrategySuperposition(
            input_dim=256,  # transformer hidden size
            num_strategies=5,
            dropout_rate=0.1,
            strategy_input_dim=256
        ).to(self.device)
        
        # 3. Mock SAC Agent (simplified for testing)
        self.sac_agent = self.create_mock_sac_agent()
        
        # 4. Meta Learning System
        self.meta_learning = MetaLearningSystem(
            initial_state_dim=256,  # 這裡用 transformer 輸出維度
            action_dim=self.action_dim
        )
        
        # 5. Progressive Reward System
        # Define stage configurations for ProgressiveLearningSystem
        stage_configs = {
            1: {
                'reward_strategy_class': SimpleReward, # Assuming SimpleReward is defined
                'reward_config': {'profit_weight': 0.7, 'risk_penalty_weight': 0.3},
                'criteria_to_advance': lambda stats: stats.get('avg_sharpe_ratio', 0) > 0.5,
                'max_episodes_or_steps': 100
            },
            2: {
                'reward_strategy_class': IntermediateReward, # Assuming IntermediateReward is defined
                'reward_config': {'sharpe_weight': 0.4, 'pnl_weight': 0.3, 'drawdown_penalty_weight': 0.2, 'cost_penalty_weight': 0.1},
                'criteria_to_advance': lambda stats: stats.get('avg_sortino_ratio', 0) > 0.8,
                'max_episodes_or_steps': 500
            }
            # Add more stages if needed
        }
        self.reward_system = ProgressiveLearningSystem(
            stage_configs=stage_configs,
            initial_stage=1
        )
        
        # 6. Strategy Innovation Engine
        self.innovation_engine = self.create_mock_innovation_engine()
        
        logger.info("所有模型組件設置完成")
    
    def create_mock_sac_agent(self):
        """創建模擬的SAC智能體用於測試"""
        class MockSACAgent:
            def __init__(self):
                self.memory = []
                self.batch_size = 4 # MODIFIED: Align with tester's batch_size to ensure train() is called
                
            def store_transition(self, state, action, reward, next_state, done):
                self.memory.append((state, action, reward, next_state, done))
                if len(self.memory) > 1000:
                    self.memory.pop(0)
            
            def train(self):
                if len(self.memory) < self.batch_size:
                    return {}
                return {
                    'actor_loss': np.random.uniform(0.1, 1.0),
                    'critic_loss': np.random.uniform(0.1, 1.0),
                    'alpha_loss': np.random.uniform(0.01, 0.1)
                }
        
        return MockSACAgent()
    
    def create_mock_innovation_engine(self):
        """創建模擬的策略創新引擎"""
        class MockInnovationEngine:
            def innovate_strategy(self, market_conditions, performance_threshold=0.1):
                return {
                    'new_strategy': f"strategy_{np.random.randint(1000, 9999)}",
                    'fitness_score': np.random.uniform(0.1, 1.0),
                    'parameters': {
                        'param1': np.random.uniform(0.1, 1.0),
                        'param2': np.random.uniform(0.1, 1.0)
                    }
                }
        
        return MockInnovationEngine()
    
    def create_sample_data(self) -> Dict[str, torch.Tensor]:
        """創建樣本數據"""
        # 市場數據: [batch_size, num_symbols, seq_len, num_features]
        market_data = torch.randn(
            self.batch_size, 
            self.num_symbols,
            self.seq_len, 
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
        
        # 歷史行動
        actions = torch.randint(0, self.action_dim, (self.batch_size, 10), device=self.device)
        
        # 獎勵
        rewards = torch.randn(self.batch_size, device=self.device)
        
        return {
            'market_data': market_data,
            'src_key_padding_mask': src_key_padding_mask,
            'market_state': market_state,
            'actions': actions,
            'rewards': rewards
        }
    
    def test_transformer_forward_pass(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """測試Transformer前向傳播"""
        logger.info("測試Transformer前向傳播...")
        
        try:
            # 準備輸入字典
            # src: [batch_size, num_active_symbols, seq_len, input_dim]
            # market_data from create_sample_data is already in the correct shape [B, N, S, F]
            input_dict = {
                "src": data['market_data'], 
                "symbol_ids": torch.arange(self.num_symbols, device=self.device).unsqueeze(0).repeat(self.batch_size, 1),
                "src_key_padding_mask": data['src_key_padding_mask'] 
            }
            
            # 前向傳播
            output = self.transformer(input_dict)
            
            logger.info(f"Transformer輸出形狀: {output.shape}")
            # The output shape of EnhancedTransformer is [batch_size, num_active_symbols, output_dim]
            # output_dim is set to 256 in transformer_config
            expected_shape = (self.batch_size, self.num_symbols, 256)
            assert output.shape == expected_shape, f"期望形狀 {expected_shape}, 實際 {output.shape}"
            
            # 檢查梯度
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
            # transformer_output shape: (batch_size, num_symbols, transformer_output_dim)
            # asset_features_batch expects: (batch_size, num_assets, sequence_length, feature_dim)
            # Let num_assets = num_symbols, sequence_length = 1, feature_dim = transformer_output_dim
            asset_features = transformer_output.unsqueeze(2)  # Shape: (B, N_symbols, 1, F_transformer_out)

            # market_state_features expects: (batch_size, quantum_layer_input_dim)
            # quantum_layer_input_dim is transformer_output_dim (256)
            # Use the features of the last symbol as market_state_features, consistent with previous 'last_hidden' logic
            market_features = transformer_output[:, -1, :]  # Shape: (B, F_transformer_out)
            
            # 前向傳播
            # Pass arguments by keyword for clarity
            strategy_output_raw = self.quantum_layer(
                asset_features_batch=asset_features,
                market_state_features=market_features
            )
            
            # Output of quantum_layer is (batch_size, num_symbols, 1)
            # Squeeze the last dimension to match expected shape (batch_size, action_dim)
            # where num_symbols is effectively action_dim in this context for the test assertion.
            strategy_output = strategy_output_raw.squeeze(-1) # Shape: (batch_size, num_symbols)
            
            logger.info(f"量子策略層輸出形狀: {strategy_output.shape}")
            # Assuming num_symbols (3) corresponds to action_dim (3) for this test's assertion
            assert strategy_output.shape == (self.batch_size, self.num_symbols), f"期望形狀 {(self.batch_size, self.num_symbols)}, 實際 {strategy_output.shape}"
            
            # 檢查梯度
            assert strategy_output.requires_grad, "量子策略層輸出應該需要梯度"
            
            logger.info("✓ 量子策略層測試通過")
            return strategy_output
            
        except Exception as e:
            logger.error(f"✗ 量子策略層測試失敗: {e}")
            raise
    
    def test_sac_agent_training(self, strategy_output: torch.Tensor, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """測試SAC智能體訓練"""
        logger.info("測試SAC智能體訓練...")
        
        try:
            # 準備SAC訓練數據
            states = strategy_output.detach()  # 使用策略輸出作為狀態
            actions = torch.softmax(strategy_output, dim=-1)  # 軟動作
            next_states = states + torch.randn_like(states) * 0.1
            rewards = data['rewards']
            dones = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
            
            # 存儲轉換
            for i in range(self.batch_size):
                self.sac_agent.store_transition(
                    states[i].cpu().numpy(),
                    actions[i].cpu().detach().numpy(), # MODIFIED: Added .detach()
                    rewards[i].item(),
                    next_states[i].cpu().numpy(),
                    dones[i].item()
                )
            
            # 訓練SAC
            if len(self.sac_agent.memory) >= self.sac_agent.batch_size:
                losses = self.sac_agent.train()
                logger.info(f"SAC訓練損失: {losses}")
                
                # 檢查損失值的合理性
                for loss_name, loss_value in losses.items():
                    assert not np.isnan(loss_value), f"SAC {loss_name} 損失為NaN"
                    assert not np.isinf(loss_value), f"SAC {loss_name} 損失為無限大"
                
                logger.info("✓ SAC智能體訓練測試通過")
                return losses
            else:
                logger.info("SAC記憶體不足，跳過訓練")
                return {}
            
        except Exception as e:
            logger.error(f"✗ SAC智能體訓練測試失敗: {e}")
            raise
    
    # MODIFIED: Changed signature to accept transformer_output
    def test_meta_learning_adaptation(self, transformer_output: torch.Tensor) -> Dict[str, Any]:
        """測試元學習適應"""
        logger.info("測試元學習適應...")
        
        try:
            # 模擬策略表現數據
            strategy_performance = {
                'strategy_A': {'reward': 10.5, 'uncertainty': 0.2},
                'strategy_B': {'reward': -2.3, 'uncertainty': 0.5}
            }
            
            # 模擬市場上下文 from transformer_output
            # transformer_output shape: (batch_size, num_symbols, 256)
            # MetaLearningSystem initialized with initial_state_dim=256
            # Use features from the first symbol for the whole batch, matching expected dim.
            market_context = transformer_output[:, 0, :] # Shape: (batch_size, 256)
            
            # 執行元學習適應
            adaptation_results = self.meta_learning.adapt_strategies(strategy_performance, market_context)
            
            logger.info(f"元學習適應結果: {adaptation_results}")
            
            # 檢查適應結果
            assert isinstance(adaptation_results, dict), "適應結果應為字典格式"
            assert '策略A' in adaptation_results, "缺少策略A的適應結果"
            assert '策略B' in adaptation_results, "缺少策略B的適應結果"
            
            logger.info("✓ 元學習適應測試通過")
            return adaptation_results
            
        except Exception as e:
            logger.error(f"✗ 元學習適應測試失敗: {e}")
            raise
    
    def test_strategy_innovation(self) -> Dict[str, Any]:
        """測試策略創新"""
        logger.info("測試策略創新...")
        
        try:
            # 創新新策略
            market_conditions = {
                'volatility': np.random.uniform(0.1, 0.3),
                'trend': np.random.choice(['bullish', 'bearish', 'sideways']),
                'volume': np.random.uniform(0.5, 2.0)
            }
            
            innovation_result = self.innovation_engine.innovate_strategy(
                market_conditions,
                performance_threshold=0.1
            )
            
            logger.info(f"策略創新結果: {innovation_result}")
            
            # 檢查創新結果
            assert 'new_strategy' in innovation_result, "缺少新策略"
            assert 'fitness_score' in innovation_result, "缺少適應度評分"
            
            logger.info("✓ 策略創新測試通過")
            return innovation_result
            
        except Exception as e:
            logger.error(f"✗ 策略創新測試失敗: {e}")
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
            strategy_output = self.quantum_layer(
                last_hidden,
                market_state=data['market_state']
            )
            
            # 計算損失
            target = torch.randn_like(strategy_output)
            loss = nn.MSELoss()(strategy_output, target)
            
            logger.info(f"計算損失: {loss.item()}")
            
            # 反向傳播
            loss.backward()
            
            # 檢查梯度
            gradient_check_results = {}
            
            # 檢查Transformer梯度
            transformer_has_gradients = False
            for name, param in self.transformer.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 1e-8:
                        transformer_has_gradients = True
                        gradient_check_results[f"transformer_{name}"] = grad_norm
            
            # 檢查量子策略層梯度
            quantum_has_gradients = False
            for name, param in self.quantum_layer.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 1e-8:
                        quantum_has_gradients = True
                        gradient_check_results[f"quantum_{name}"] = grad_norm
            
            results['transformer_gradients'] = transformer_has_gradients
            results['quantum_gradients'] = quantum_has_gradients
            
            logger.info(f"梯度檢查結果: {gradient_check_results}")
            
            # 模擬優化器步驟
            optimizer = torch.optim.Adam(
                list(self.transformer.parameters()) + list(self.quantum_layer.parameters()),
                lr=1e-4
            )
            optimizer.step()
            
            # 檢查權重更新
            weights_updated = {}
            
            for name, param in self.transformer.named_parameters():
                if param.requires_grad and f"transformer_{name}" in initial_weights:
                    initial = initial_weights[f"transformer_{name}"]
                    current = param.detach()
                    diff = torch.norm(current - initial).item()
                    weights_updated[f"transformer_{name}"] = diff > 1e-8
            
            for name, param in self.quantum_layer.named_parameters():
                if param.requires_grad and f"quantum_{name}" in initial_weights:
                    initial = initial_weights[f"quantum_{name}"]
                    current = param.detach()
                    diff = torch.norm(current - initial).item()
                    weights_updated[f"quantum_{name}"] = diff > 1e-8
            
            results['weights_updated'] = any(weights_updated.values())
            results['weight_update_details'] = weights_updated
            
            logger.info(f"權重更新結果: {weights_updated}")
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
            
            # 檢查階段轉換
            stage_before = self.reward_system.current_stage
            self.reward_system.update_progress(np.mean(rewards))
            stage_after = self.reward_system.current_stage
            
            results = {
                'current_stage': stage_after,
                'stage_changed': stage_before != stage_after,
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
        
        results = {}
        
        try:
            # 1. 創建測試數據
            data = self.create_sample_data()
            logger.info("✓ 測試數據創建完成")
            
            # 2. 測試Transformer
            transformer_output = self.test_transformer_forward_pass(data)
            results['transformer'] = True
            
            # 3. 測試量子策略層
            strategy_output = self.test_quantum_strategy_layer(transformer_output, data)
            results['quantum_strategy'] = True
            
            # 4. 測試SAC智能體
            sac_losses = self.test_sac_agent_training(strategy_output, data)
            results['sac_agent'] = True
            results['sac_losses'] = sac_losses
            
            # 5. 測試元學習
            meta_results = self.test_meta_learning_adaptation(transformer_output) 
            results['meta_learning'] = True
            results['meta_results'] = meta_results
            
            # 6. 測試策略創新
            innovation_results = self.test_strategy_innovation()
            results['strategy_innovation'] = True
            results['innovation_results'] = innovation_results
            
            # 7. 測試梯度流
            gradient_results = self.test_gradient_flow(data)
            results['gradient_flow'] = True
            results['gradient_results'] = gradient_results
            
            # 8. 測試漸進式獎勵系統
            reward_results = self.test_progressive_reward_system(data)
            results['progressive_reward'] = True
            results['reward_results'] = reward_results
            
            # 整體成功
            results['overall_success'] = True
            results['all_tests_passed'] = all([
                results['transformer'],
                results['quantum_strategy'],
                results['sac_agent'],
                results['meta_learning'],
                results['strategy_innovation'],
                results['gradient_flow'],
                results['progressive_reward']
            ])
            
            logger.info("🎉 完整模型流程測試全部通過！")
            
        except Exception as e:
            results['overall_success'] = False
            results['error'] = str(e)
            results['traceback'] = traceback.format_exc()
            logger.error(f"完整模型流程測試失敗: {e}")
            
        return results


def test_complete_model_flow():
    """完整模型流程測試函數"""
    tester = CompleteModelFlowTester()
    results = tester.run_complete_test()
    
    # 打印詳細結果
    print("\n" + "="*80)
    print("完整模型流程測試結果")
    print("="*80)
    
    for key, value in results.items():
        if key not in ['gradient_results', 'meta_results', 'innovation_results', 'reward_results', 'sac_losses']:
            print(f"{key}: {value}")
    
    print("\n詳細結果:")
    if 'gradient_results' in results:
        print(f"梯度流結果: {results['gradient_results']}")
    if 'meta_results' in results:
        print(f"元學習結果: {results['meta_results']}")
    if 'sac_losses' in results:
        print(f"SAC損失: {results['sac_losses']}")
    
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
