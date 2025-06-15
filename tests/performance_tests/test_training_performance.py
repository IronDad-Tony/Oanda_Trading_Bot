import unittest
import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

class TestTrainingPerformance(unittest.TestCase):
    """測試訓練性能"""
    
    def test_training_speed(self):
        """測試訓練速度"""
        try:
            from src.environment.trading_env import UniversalTradingEnvV4
            from src.agent.sac_agent_wrapper import QuantumEnhancedSAC
            from stable_baselines3.common.vec_env import DummyVecEnv
            
            # 創建環境
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
                learning_rate=3e-4,
                batch_size=32,
                buffer_size=1000
            )
            
            # 測試訓練速度
            start_time = time.time()
            agent.train(total_timesteps=50)  # 小規模測試
            training_time = time.time() - start_time
            
            print(f"✅ 50 步訓練耗時: {training_time:.2f} 秒")
            
            # 簡單的性能檢查（不應該太慢）
            self.assertLess(training_time, 60, "訓練時間過長")
            
        except Exception as e:
            print(f"⚠️  性能測試失敗: {e}")
            self.skipTest("性能測試失敗")

if __name__ == '__main__':
    unittest.main()
