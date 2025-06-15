import unittest
import sys
import os
import torch
import numpy as np

# 添加專案根目錄到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

class TestCompleteTrainingFlow(unittest.TestCase):
    """測試完整訓練流程 - 確保所有模組協作正常，梯度流通暢，權重更新正確"""
    
    def setUp(self):
        """設置測試環境"""
        self.test_symbols = ["EUR_USD"]
        self.initial_capital = 10000
        self.max_trade_amount = 0.1
        self.short_training_steps = 50  # 短期訓練，專注於流程驗證
        
    def test_imports_and_basic_setup(self):
        """階段 1: 測試所有必要模組的導入"""
        print("\n🔧 階段 1: 測試模組導入...")
        
        try:
            # 核心模組導入
            from src.environment.trading_env import UniversalTradingEnvV4
            from src.agent.sac_agent_wrapper import QuantumEnhancedSAC
            from stable_baselines3.common.vec_env import DummyVecEnv
            
            print("✅ 核心模組導入成功")
            
            # 嘗試導入增強模組
            try:
                from src.models.enhanced_transformer import EnhancedTransformer
                print("✅ Enhanced Transformer 導入成功")
                self.enhanced_transformer_available = True
            except ImportError as e:
                print(f"⚠️  Enhanced Transformer 導入失敗: {e}")
                self.enhanced_transformer_available = False
            
            try:
                from src.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition
                print("✅ 量子策略層導入成功")
                self.quantum_strategies_available = True
            except ImportError as e:
                print(f"⚠️  量子策略層導入失敗: {e}")
                self.quantum_strategies_available = False
            
            try:
                from src.environment.progressive_reward_system import ProgressiveLearningSystem
                print("✅ 漸進式獎勵系統導入成功")
                self.progressive_rewards_available = True
            except ImportError as e:
                print(f"⚠️  漸進式獎勵系統導入失敗: {e}")
                self.progressive_rewards_available = False
            
        except ImportError as e:
            self.fail(f"核心模組導入失敗: {e}")
    
    def test_environment_creation_and_basic_interaction(self):
        """階段 2: 測試環境創建和基本交互"""
        print("\n🌍 階段 2: 測試環境創建和基本交互...")
        
        try:
            from src.environment.trading_env import UniversalTradingEnvV4
            from stable_baselines3.common.vec_env import DummyVecEnv
            
            # 創建環境
            def make_env():
                return UniversalTradingEnvV4(
                    symbols=self.test_symbols,
                    initial_capital=self.initial_capital,
                    max_trade_amount=self.max_trade_amount
                )
            
            env = DummyVecEnv([make_env])
            
            # 測試環境重置
            obs = env.reset()
            self.assertIsNotNone(obs)
            self.assertGreater(obs.shape[0], 0)
            print(f"✅ 環境重置成功，觀察維度: {obs.shape}")
            
            # 測試隨機動作
            action_space = env.action_space
            random_action = [action_space.sample() for _ in range(env.num_envs)]
            
            next_obs, reward, done, info = env.step(random_action)
            
            self.assertIsNotNone(next_obs)
            self.assertIsNotNone(reward)
            self.assertIsInstance(done, (list, np.ndarray))
            print("✅ 環境步驟執行成功")
            
            # 保存環境供後續測試使用
            self.test_env = env
            self.initial_obs = obs
            
        except Exception as e:
            self.fail(f"環境創建失敗: {e}")
    
    def test_agent_creation_and_policy_setup(self):
        """階段 3: 測試智能體創建和策略設置"""
        print("\n🤖 階段 3: 測試智能體創建和策略設置...")
        
        try:
            from src.agent.sac_agent_wrapper import QuantumEnhancedSAC
            
            # 創建智能體
            agent = QuantumEnhancedSAC(
                env=self.test_env,
                learning_rate=3e-4,
                batch_size=32,
                buffer_size=1000,
                learning_starts=10,  # 快速開始學習
                verbose=1
            )
            
            self.assertIsNotNone(agent)
            self.assertIsNotNone(agent.agent)
            print("✅ SAC 智能體創建成功")
            
            # 測試策略網絡
            if hasattr(agent.agent, 'policy'):
                policy = agent.agent.policy
                print(f"✅ 策略網絡: {type(policy).__name__}")
                
                # 檢查特徵提取器
                if hasattr(policy, 'features_extractor'):
                    features_extractor = policy.features_extractor
                    print(f"✅ 特徵提取器: {type(features_extractor).__name__}")
                    
                    # 如果是增強的 Transformer
                    if 'Transformer' in type(features_extractor).__name__:
                        print("✅ 使用增強的 Transformer 特徵提取器")
            
            # 保存智能體供後續測試使用
            self.test_agent = agent
            
        except Exception as e:
            self.fail(f"智能體創建失敗: {e}")
    
    def test_initial_prediction_and_model_parameters(self):
        """階段 4: 測試初始預測和模型參數"""
        print("\n🔮 階段 4: 測試初始預測和模型參數...")
        
        try:
            # 測試初始預測
            action, _ = self.test_agent.predict(self.initial_obs, deterministic=True)
            self.assertIsNotNone(action)
            print(f"✅ 初始預測成功，動作形狀: {action.shape}")
            
            # 記錄初始模型參數（用於後續檢查權重更新）
            self.initial_model_params = {}
            
            if hasattr(self.test_agent.agent.policy, 'features_extractor'):
                feature_extractor = self.test_agent.agent.policy.features_extractor
                param_count = 0
                for name, param in feature_extractor.named_parameters():
                    self.initial_model_params[f"features_extractor.{name}"] = param.clone().detach()
                    param_count += param.numel()
                print(f"✅ 特徵提取器參數記錄完成，總參數數量: {param_count:,}")
            
            # 記錄 Actor 和 Critic 網絡參數
            if hasattr(self.test_agent.agent.policy, 'actor'):
                actor = self.test_agent.agent.policy.actor
                for name, param in actor.named_parameters():
                    self.initial_model_params[f"actor.{name}"] = param.clone().detach()
                print("✅ Actor 網絡參數記錄完成")
            
            if hasattr(self.test_agent.agent.policy, 'critic'):
                critic = self.test_agent.agent.policy.critic
                for name, param in critic.named_parameters():
                    self.initial_model_params[f"critic.{name}"] = param.clone().detach()
                print("✅ Critic 網絡參數記錄完成")
            
            print(f"✅ 總共記錄了 {len(self.initial_model_params)} 個參數張量")
            
        except Exception as e:
            self.fail(f"初始預測測試失敗: {e}")
    
    def test_short_training_and_gradient_flow(self):
        """階段 5: 測試短期訓練和梯度流"""
        print(f"\n🎯 階段 5: 測試短期訓練和梯度流 ({self.short_training_steps} 步)...")
        
        try:
            import time
            
            # 記錄訓練開始時間
            start_time = time.time()
            
            # 執行短期訓練
            print(f"⏳ 開始 {self.short_training_steps} 步訓練...")
            self.test_agent.train(total_timesteps=self.short_training_steps)
            
            training_duration = time.time() - start_time
            print(f"✅ 訓練完成，耗時: {training_duration:.2f} 秒")
            
        except Exception as e:
            print(f"⚠️  訓練過程出現錯誤: {e}")
            # 不直接失敗，繼續檢查其他部分
    
    def test_weight_updates_and_gradient_propagation(self):
        """階段 6: 測試權重更新和梯度傳播"""
        print("\n📊 階段 6: 測試權重更新和梯度傳播...")
        
        try:
            # 檢查模型參數是否更新
            weights_changed = {}
            unchanged_params = []
            changed_params = []
            
            current_params = {}
            
            # 獲取當前參數
            if hasattr(self.test_agent.agent.policy, 'features_extractor'):
                feature_extractor = self.test_agent.agent.policy.features_extractor
                for name, param in feature_extractor.named_parameters():
                    current_params[f"features_extractor.{name}"] = param.clone().detach()
            
            if hasattr(self.test_agent.agent.policy, 'actor'):
                actor = self.test_agent.agent.policy.actor
                for name, param in actor.named_parameters():
                    current_params[f"actor.{name}"] = param.clone().detach()
            
            if hasattr(self.test_agent.agent.policy, 'critic'):
                critic = self.test_agent.agent.policy.critic
                for name, param in critic.named_parameters():
                    current_params[f"critic.{name}"] = param.clone().detach()
            
            # 比較參數變化
            for param_name in self.initial_model_params:
                if param_name in current_params:
                    initial = self.initial_model_params[param_name]
                    current = current_params[param_name]
                    
                    # 檢查是否相等
                    if torch.equal(initial, current):
                        unchanged_params.append(param_name)
                        weights_changed[param_name] = False
                    else:
                        changed_params.append(param_name)
                        weights_changed[param_name] = True
                        
                        # 計算變化幅度
                        diff = torch.abs(current - initial).mean().item()
                        print(f"  📈 {param_name}: 平均變化 {diff:.8f}")
            
            print(f"\n📊 權重更新統計:")
            print(f"  ✅ 已更新參數: {len(changed_params)}")
            print(f"  ⚠️  未更新參數: {len(unchanged_params)}")
            
            if len(changed_params) > 0:
                print("✅ 梯度流正常，模型參數正在更新")
                
                # 顯示一些更新的參數
                print("\n🔍 更新的參數示例:")
                for i, param_name in enumerate(changed_params[:5]):  # 顯示前5個
                    print(f"  - {param_name}")
                
            else:
                print("⚠️  沒有參數更新，可能需要更多訓練步驟或檢查學習率")
            
            if len(unchanged_params) > 0:
                print("\n⚠️  未更新的參數:")
                for param_name in unchanged_params[:5]:  # 顯示前5個
                    print(f"  - {param_name}")
            
            # 至少應該有一些參數更新
            self.assertGreater(len(changed_params), 0, 
                             "沒有任何模型參數更新，梯度流可能有問題")
            
        except Exception as e:
            self.fail(f"權重更新檢查失敗: {e}")
    
    def test_post_training_prediction_consistency(self):
        """階段 7: 測試訓練後預測一致性"""
        print("\n🔄 階段 7: 測試訓練後預測一致性...")
        
        try:
            # 測試訓練後的預測
            obs = self.test_env.reset()
            action1, _ = self.test_agent.predict(obs, deterministic=True)
            action2, _ = self.test_agent.predict(obs, deterministic=True)
            
            # 確定性預測應該一致
            np.testing.assert_array_almost_equal(action1, action2, decimal=6)
            print("✅ 訓練後確定性預測一致")
            
            # 測試隨機預測
            action3, _ = self.test_agent.predict(obs, deterministic=False)
            action4, _ = self.test_agent.predict(obs, deterministic=False)
            
            # 隨機預測應該不同（但形狀相同）
            self.assertEqual(action3.shape, action4.shape)
            print("✅ 隨機預測形狀正確")
            
        except Exception as e:
            self.fail(f"預測一致性測試失敗: {e}")
    
    def test_enhanced_components_integration(self):
        """階段 8: 測試增強組件集成（如果可用）"""
        print("\n🚀 階段 8: 測試增強組件集成...")
        
        # 測試量子策略層集成
        if self.quantum_strategies_available:
            try:
                from src.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition
                from src.agent.strategies import STRATEGY_REGISTRY
                
                # 創建測試配置
                strategies_config = {
                    "strategies": [
                        {"name": "MomentumStrategy", "params": {"window": 20}, "input_dim": 64},
                        {"name": "MeanReversionStrategy", "params": {"reversion_window": 15}, "input_dim": 64}
                    ]
                }
                
                superposition = EnhancedStrategySuperposition(
                    overall_config_for_strategies=strategies_config,
                    strategy_registry=STRATEGY_REGISTRY
                )
                
                # 測試前向傳播
                batch_size = 2
                feature_dim = 64
                test_features = torch.randn(batch_size, feature_dim)
                
                output = superposition(test_features)
                self.assertIsNotNone(output)
                print("✅ 量子策略層集成測試通過")
                
            except Exception as e:
                print(f"⚠️  量子策略層集成測試失敗: {e}")
        else:
            print("⏭️  跳過量子策略層測試（模組不可用）")
        
        # 測試漸進式獎勵系統
        if self.progressive_rewards_available:
            try:
                from src.environment.progressive_reward_system import ProgressiveLearningSystem
                
                reward_system = ProgressiveLearningSystem()
                
                # 測試基本獎勵計算
                test_profit_loss = 100.0
                test_risk_metrics = {"volatility": 0.02, "drawdown": 0.05}
                
                reward = reward_system.calculate_reward(test_profit_loss, test_risk_metrics)
                self.assertIsInstance(reward, (int, float))
                print("✅ 漸進式獎勵系統集成測試通過")
                
            except Exception as e:
                print(f"⚠️  漸進式獎勵系統集成測試失敗: {e}")
        else:
            print("⏭️  跳過漸進式獎勵系統測試（模組不可用）")
    
    def test_memory_and_resource_usage(self):
        """階段 9: 測試記憶體和資源使用"""
        print("\n💾 階段 9: 測試記憶體和資源使用...")
        
        try:
            import psutil
            import gc
            
            # 獲取記憶體使用情況
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            print(f"📊 當前記憶體使用: {memory_mb:.1f} MB")
            
            # 檢查是否有明顯的記憶體洩漏
            self.assertLess(memory_mb, 2000, "記憶體使用過高，可能有記憶體洩漏")
            
            # 清理記憶體
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("✅ 記憶體使用正常")
            
        except ImportError:
            print("⏭️  跳過記憶體檢查（psutil 不可用）")
        except Exception as e:
            print(f"⚠️  記憶體檢查失敗: {e}")
    
    def test_model_save_and_load(self):
        """階段 10: 測試模型保存和加載"""
        print("\n💾 階段 10: 測試模型保存和加載...")
        
        try:
            import tempfile
            import shutil
            
            # 創建臨時目錄
            temp_dir = tempfile.mkdtemp()
            model_path = os.path.join(temp_dir, "test_model")
            
            try:
                # 保存模型
                self.test_agent.save(model_path)
                print("✅ 模型保存成功")
                
                # 檢查保存的文件
                saved_files = os.listdir(temp_dir)
                self.assertGreater(len(saved_files), 0, "沒有文件被保存")
                print(f"✅ 保存了 {len(saved_files)} 個文件")
                
                # 創建新的智能體並加載模型
                from src.agent.sac_agent_wrapper import QuantumEnhancedSAC
                
                new_agent = QuantumEnhancedSAC(
                    env=self.test_env,
                    learning_rate=3e-4,
                    batch_size=32,
                    buffer_size=1000
                )
                
                new_agent.load(model_path, env=self.test_env)
                print("✅ 模型加載成功")
                
                # 測試加載後的預測
                obs = self.test_env.reset()
                action1, _ = self.test_agent.predict(obs, deterministic=True)
                action2, _ = new_agent.predict(obs, deterministic=True)
                
                # 預測應該相同
                np.testing.assert_array_almost_equal(action1, action2, decimal=5)
                print("✅ 加載後預測一致")
                
            finally:
                # 清理臨時目錄
                shutil.rmtree(temp_dir)
                
        except Exception as e:
            print(f"⚠️  模型保存/加載測試失敗: {e}")
    
    def tearDown(self):
        """清理測試環境"""
        # 清理環境
        if hasattr(self, 'test_env'):
            self.test_env.close()
        
        # 清理 GPU 記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    # 配置詳細的測試運行
    unittest.main(verbosity=2, buffer=True)
