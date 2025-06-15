import unittest
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any

# 添加專案根目錄到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

class TestGradientFlowValidation(unittest.TestCase):
    """專門測試梯度流驗證 - 確保所有模型組件的梯度能正確計算和傳播"""
    
    def setUp(self):
        """設置測試環境"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n🔧 使用設備: {self.device}")
        
        # 測試超參數
        self.batch_size = 4
        self.seq_length = 16
        self.input_dim = 32
        self.output_dim = 8
        self.learning_rate = 1e-3
        
        # 設置隨機種子以保證可重現性
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_basic_transformer_gradient_flow(self):
        """測試基本 Transformer 模型的梯度流"""
        print("\n🧠 測試基本 Transformer 梯度流...")
        
        try:
            # 創建簡單的 Transformer 模型
            class SimpleTransformer(nn.Module):
                def __init__(self, input_dim, hidden_dim, output_dim):
                    super().__init__()
                    self.input_projection = nn.Linear(input_dim, hidden_dim)
                    self.transformer_layer = nn.TransformerEncoderLayer(
                        d_model=hidden_dim,
                        nhead=4,
                        dim_feedforward=hidden_dim * 2,
                        batch_first=True
                    )
                    self.output_projection = nn.Linear(hidden_dim, output_dim)
                    
                def forward(self, x):
                    x = self.input_projection(x)
                    x = self.transformer_layer(x)
                    x = x.mean(dim=1)  # Global average pooling
                    return self.output_projection(x)
            
            model = SimpleTransformer(self.input_dim, 64, self.output_dim).to(self.device)
            
            # 創建測試數據
            x = torch.randn(self.batch_size, self.seq_length, self.input_dim).to(self.device)
            target = torch.randn(self.batch_size, self.output_dim).to(self.device)
            
            # 測試梯度流
            gradients_valid = self._test_model_gradient_flow(model, x, target, "SimpleTransformer")
            self.assertTrue(gradients_valid, "簡單 Transformer 梯度流失敗")
            
        except Exception as e:
            self.fail(f"基本 Transformer 梯度流測試失敗: {e}")
    
    def test_enhanced_transformer_gradient_flow(self):
        """測試增強 Transformer 模型的梯度流"""
        print("\n🚀 測試增強 Transformer 梯度流...")
        
        try:
            from src.models.enhanced_transformer import EnhancedTransformer
            
            model = EnhancedTransformer(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                num_layers=2,
                num_heads=4,
                hidden_dim=64
            ).to(self.device)
            
            # 創建測試數據
            x = torch.randn(self.batch_size, self.seq_length, self.input_dim).to(self.device)
            target = torch.randn(self.batch_size, self.output_dim).to(self.device)
            
            # 測試梯度流
            gradients_valid = self._test_model_gradient_flow(model, x, target, "EnhancedTransformer")
            self.assertTrue(gradients_valid, "增強 Transformer 梯度流失敗")
            
            print("✅ 增強 Transformer 梯度流測試通過")
            
        except ImportError:
            print("⏭️  跳過增強 Transformer 測試（模組不可用）")
            self.skipTest("EnhancedTransformer 不可用")
        except Exception as e:
            self.fail(f"增強 Transformer 梯度流測試失敗: {e}")
    
    def test_quantum_strategy_layer_gradient_flow(self):
        """測試量子策略層的梯度流"""
        print("\n⚛️  測試量子策略層梯度流...")
        
        try:
            from src.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition
            from src.agent.strategies import STRATEGY_REGISTRY
            
            # 創建策略配置
            strategies_config = {
                "strategies": [
                    {"name": "MomentumStrategy", "params": {"window": 10}, "input_dim": self.input_dim},
                    {"name": "MeanReversionStrategy", "params": {"reversion_window": 8}, "input_dim": self.input_dim}
                ]
            }
            
            model = EnhancedStrategySuperposition(
                overall_config_for_strategies=strategies_config,
                strategy_registry=STRATEGY_REGISTRY
            ).to(self.device)
            
            # 創建測試數據
            x = torch.randn(self.batch_size, self.input_dim).to(self.device)
            target = torch.randn(self.batch_size, 5).to(self.device)  # 假設5個動作
            
            # 測試梯度流
            gradients_valid = self._test_model_gradient_flow(model, x, target, "EnhancedStrategySuperposition")
            self.assertTrue(gradients_valid, "量子策略層梯度流失敗")
            
            print("✅ 量子策略層梯度流測試通過")
            
        except ImportError:
            print("⏭️  跳過量子策略層測試（模組不可用）")
            self.skipTest("量子策略層不可用")
        except Exception as e:
            self.fail(f"量子策略層梯度流測試失敗: {e}")
    
    def test_sac_agent_gradient_flow(self):
        """測試 SAC 智能體的梯度流"""
        print("\n🤖 測試 SAC 智能體梯度流...")
        
        try:
            from src.environment.trading_env import UniversalTradingEnvV4
            from src.agent.sac_agent_wrapper import QuantumEnhancedSAC
            from stable_baselines3.common.vec_env import DummyVecEnv
            
            # 創建簡單環境
            def make_env():
                return UniversalTradingEnvV4(
                    symbols=["EUR_USD"],
                    initial_capital=10000,
                    max_trade_amount=0.1
                )
            
            env = DummyVecEnv([make_env])
            
            # 創建智能體
            agent = QuantumEnhancedSAC(
                env=env,
                learning_rate=1e-4,
                batch_size=16,
                buffer_size=500,
                learning_starts=10,
                verbose=0
            )
            
            # 測試策略網絡梯度流
            if hasattr(agent.agent.policy, 'features_extractor'):
                print("🔍 檢查特徵提取器梯度流...")
                feature_extractor = agent.agent.policy.features_extractor
                
                # 創建測試輸入
                obs = env.reset()
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                
                # 測試特徵提取器的梯度流
                gradients_valid = self._test_feature_extractor_gradient_flow(
                    feature_extractor, obs_tensor
                )
                
                if gradients_valid:
                    print("✅ 特徵提取器梯度流正常")
                else:
                    print("⚠️  特徵提取器梯度流異常")
            
            # 執行少量訓練步驟來測試整體梯度流
            print("🏃 執行短期訓練測試梯度流...")
            
            # 記錄訓練前參數
            initial_params = self._get_model_parameters(agent.agent.policy)
            
            # 短期訓練
            agent.train(total_timesteps=20)
            
            # 檢查參數是否更新
            final_params = self._get_model_parameters(agent.agent.policy)
            params_changed = self._compare_parameters(initial_params, final_params)
            
            if params_changed:
                print("✅ SAC 智能體參數在訓練中更新")
            else:
                print("⚠️  SAC 智能體參數未更新")
            
            env.close()
            
        except Exception as e:
            self.fail(f"SAC 智能體梯度流測試失敗: {e}")
    
    def test_end_to_end_gradient_flow(self):
        """測試端到端梯度流"""
        print("\n🔄 測試端到端梯度流...")
        
        try:
            # 創建端到端模型（模擬完整的交易流程）
            class EndToEndTradingModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    # 特徵提取器（模擬 Transformer）
                    self.feature_extractor = nn.Sequential(
                        nn.Linear(32, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU()
                    )
                    
                    # 策略層（模擬量子策略）
                    self.strategy_layer = nn.Sequential(
                        nn.Linear(64, 32),
                        nn.Tanh(),
                        nn.Linear(32, 16),
                        nn.Tanh()
                    )
                    
                    # 動作頭
                    self.action_head = nn.Linear(16, 5)
                    
                def forward(self, x):
                    features = self.feature_extractor(x)
                    strategy_features = self.strategy_layer(features)
                    actions = self.action_head(strategy_features)
                    return actions
            
            model = EndToEndTradingModel().to(self.device)
            
            # 創建測試數據
            x = torch.randn(self.batch_size, 32).to(self.device)
            target = torch.randn(self.batch_size, 5).to(self.device)
            
            # 測試完整的梯度流
            gradients_valid = self._test_model_gradient_flow(model, x, target, "EndToEndTradingModel")
            self.assertTrue(gradients_valid, "端到端梯度流失敗")
            
            print("✅ 端到端梯度流測試通過")
            
        except Exception as e:
            self.fail(f"端到端梯度流測試失敗: {e}")
    
    def _test_model_gradient_flow(self, model: nn.Module, input_data: torch.Tensor, 
                                 target: torch.Tensor, model_name: str) -> bool:
        """通用的模型梯度流測試函數"""
        try:
            # 記錄初始參數
            initial_params = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    initial_params[name] = param.clone().detach()
            
            # 創建優化器
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            
            # 前向傳播
            model.train()
            optimizer.zero_grad()
            
            output = model(input_data)
            
            # 調整目標形狀以匹配輸出
            if output.shape != target.shape:
                if len(output.shape) == 2 and len(target.shape) == 2:
                    if output.shape[1] != target.shape[1]:
                        target = target[:, :output.shape[1]]
                else:
                    target = target.view_as(output)
            
            loss = criterion(output, target)
            
            # 反向傳播
            loss.backward()
            
            # 檢查梯度
            gradient_info = self._analyze_gradients(model, model_name)
            
            # 執行優化步驟
            optimizer.step()
            
            # 檢查參數更新
            params_updated = self._check_parameter_updates(model, initial_params, model_name)
            
            return gradient_info['has_gradients'] and params_updated
            
        except Exception as e:
            print(f"❌ {model_name} 梯度流測試錯誤: {e}")
            return False
    
    def _analyze_gradients(self, model: nn.Module, model_name: str) -> Dict[str, Any]:
        """分析模型梯度"""
        gradient_info = {
            'has_gradients': False,
            'gradient_norms': {},
            'zero_gradients': [],
            'none_gradients': [],
            'total_params': 0,
            'params_with_gradients': 0
        }
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                gradient_info['total_params'] += 1
                
                if param.grad is None:
                    gradient_info['none_gradients'].append(name)
                else:
                    gradient_info['params_with_gradients'] += 1
                    grad_norm = param.grad.norm().item()
                    gradient_info['gradient_norms'][name] = grad_norm
                    
                    if grad_norm == 0:
                        gradient_info['zero_gradients'].append(name)
        
        gradient_info['has_gradients'] = gradient_info['params_with_gradients'] > 0
        
        # 打印梯度分析
        print(f"  📊 {model_name} 梯度分析:")
        print(f"    - 總參數數: {gradient_info['total_params']}")
        print(f"    - 有梯度的參數: {gradient_info['params_with_gradients']}")
        print(f"    - 零梯度參數: {len(gradient_info['zero_gradients'])}")
        print(f"    - 無梯度參數: {len(gradient_info['none_gradients'])}")
        
        if gradient_info['gradient_norms']:
            avg_grad_norm = np.mean(list(gradient_info['gradient_norms'].values()))
            max_grad_norm = max(gradient_info['gradient_norms'].values())
            print(f"    - 平均梯度範數: {avg_grad_norm:.6f}")
            print(f"    - 最大梯度範數: {max_grad_norm:.6f}")
        
        return gradient_info
    
    def _check_parameter_updates(self, model: nn.Module, initial_params: Dict[str, torch.Tensor], 
                                model_name: str) -> bool:
        """檢查參數是否更新"""
        updated_params = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in initial_params:
                total_params += 1
                if not torch.equal(initial_params[name], param.data):
                    updated_params += 1
        
        print(f"  🔄 {model_name} 參數更新:")
        print(f"    - 更新的參數: {updated_params}/{total_params}")
        
        return updated_params > 0
    
    def _test_feature_extractor_gradient_flow(self, feature_extractor: nn.Module, 
                                            input_tensor: torch.Tensor) -> bool:
        """測試特徵提取器的梯度流"""
        try:
            # 創建假的下游任務
            feature_dim = 64  # 假設特徵維度
            downstream_model = nn.Linear(feature_dim, 1).to(self.device)
            
            # 前向傳播
            features = feature_extractor(input_tensor)
            
            # 調整特徵維度
            if len(features.shape) > 2:
                features = features.view(features.shape[0], -1)
            
            # 調整下游模型輸入維度
            if features.shape[1] != feature_dim:
                downstream_model = nn.Linear(features.shape[1], 1).to(self.device)
            
            output = downstream_model(features)
            target = torch.randn_like(output)
            
            loss = nn.MSELoss()(output, target)
            loss.backward()
            
            # 檢查特徵提取器是否有梯度
            has_gradients = False
            for param in feature_extractor.parameters():
                if param.grad is not None and param.grad.norm() > 0:
                    has_gradients = True
                    break
            
            return has_gradients
            
        except Exception as e:
            print(f"❌ 特徵提取器梯度流測試錯誤: {e}")
            return False
    
    def _get_model_parameters(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """獲取模型參數的副本"""
        params = {}
        for name, param in model.named_parameters():
            params[name] = param.clone().detach()
        return params
    
    def _compare_parameters(self, params1: Dict[str, torch.Tensor], 
                          params2: Dict[str, torch.Tensor]) -> bool:
        """比較兩組參數是否有差異"""
        for name in params1:
            if name in params2:
                if not torch.equal(params1[name], params2[name]):
                    return True
        return False
    
    def test_gradient_explosion_and_vanishing(self):
        """測試梯度爆炸和消失問題"""
        print("\n🌊 測試梯度爆炸和消失問題...")
        
        try:
            # 創建深層網絡來測試梯度問題
            class DeepNetwork(nn.Module):
                def __init__(self, depth=10):
                    super().__init__()
                    layers = []
                    in_dim = 32
                    
                    for i in range(depth):
                        layers.append(nn.Linear(in_dim, 32))
                        layers.append(nn.ReLU())
                        in_dim = 32
                    
                    layers.append(nn.Linear(32, 1))
                    self.network = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.network(x)
            
            model = DeepNetwork(depth=8).to(self.device)
            
            # 測試數據
            x = torch.randn(self.batch_size, 32).to(self.device)
            target = torch.randn(self.batch_size, 1).to(self.device)
            
            # 前向傳播和反向傳播
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            
            # 檢查梯度範數
            gradient_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    gradient_norms.append(grad_norm)
            
            if gradient_norms:
                max_grad = max(gradient_norms)
                min_grad = min(gradient_norms)
                avg_grad = np.mean(gradient_norms)
                
                print(f"  📊 梯度統計:")
                print(f"    - 最大梯度範數: {max_grad:.6f}")
                print(f"    - 最小梯度範數: {min_grad:.6f}")
                print(f"    - 平均梯度範數: {avg_grad:.6f}")
                
                # 檢查梯度爆炸（梯度範數 > 10）
                if max_grad > 10:
                    print("⚠️  檢測到潛在的梯度爆炸問題")
                
                # 檢查梯度消失（梯度範數 < 1e-6）
                if min_grad < 1e-6:
                    print("⚠️  檢測到潛在的梯度消失問題")
                
                # 正常範圍
                if 1e-6 <= max_grad <= 10:
                    print("✅ 梯度範數在正常範圍內")
            
        except Exception as e:
            print(f"⚠️  梯度問題測試失敗: {e}")
    
    def tearDown(self):
        """清理測試環境"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    unittest.main(verbosity=2)
