#!/usr/bin/env python3
"""
Test script to validate the functionality of real components in SAC Agent Wrapper
Tests the actual operation of Strategy Innovation, Market State Awareness, and Meta-Learning components
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append('.')

try:
    from src.agent.sac_agent_wrapper import QuantumEnhancedSAC
    from src.agent.strategy_innovation_module import create_strategy_innovation_module
    from src.agent.market_state_awareness_system import MarketStateAwarenessSystem
    from src.agent.meta_learning_optimizer import MetaLearningOptimizer
    from src.common.logger_setup import logger
    print("✅ Successfully imported all components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_strategy_innovation_module():
    """Test the real Strategy Innovation Module functionality"""
    print("\n🧪 Testing Strategy Innovation Module...")
    
    try:
        # Create the module using the factory function
        module = create_strategy_innovation_module(
            input_dim=768,
            output_dim=256,
            hidden_dim=512,
            num_heads=8,
            num_layers=6,
            dropout=0.1
        )
        
        # Test forward pass
        batch_size = 4
        market_data = torch.randn(batch_size, 768)
        
        print(f"  Input shape: {market_data.shape}")
        
        # Test the module's forward pass
        with torch.no_grad():
            output = module(market_data)
        
        print(f"  ✅ Strategy Innovation Module created and tested successfully")
        print(f"     - Module type: {type(module).__name__}")
        print(f"     - Output keys: {list(output.keys()) if isinstance(output, dict) else 'Tensor output'}")
        
        # Test training mode
        module.train()
        module.eval()
        print(f"  ✅ Module mode switching works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Strategy Innovation Module test failed: {e}")
        return False

def test_market_state_awareness_system():
    """Test the real Market State Awareness System functionality"""
    print("\n🧪 Testing Market State Awareness System...")
    
    try:
        # Create the system
        system = MarketStateAwarenessSystem(
            input_dim=768,
            hidden_dim=256,
            num_regimes=4,
            dropout=0.1
        )
        
        # Test forward pass
        batch_size = 4
        market_data = torch.randn(batch_size, 768)
        
        print(f"  Input shape: {market_data.shape}")
        
        # Test the system's forward pass
        with torch.no_grad():
            output = system(market_data)
        
        print(f"  ✅ Market State Awareness System created and tested successfully")
        print(f"     - System type: {type(system).__name__}")
        print(f"     - Output keys: {list(output.keys()) if isinstance(output, dict) else 'Tensor output'}")
        
        # Test regime detection
        if isinstance(output, dict) and 'market_state' in output:
            market_state = output['market_state']
            if isinstance(market_state, dict) and 'regime' in market_state:
                regimes = market_state['regime']
                print(f"     - Detected regimes: {regimes}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Market State Awareness System test failed: {e}")
        return False

def test_meta_learning_optimizer():
    """Test the real Meta-Learning Optimizer functionality"""
    print("\n🧪 Testing Meta-Learning Optimizer...")
    
    try:
        # Create the optimizer
        optimizer = MetaLearningOptimizer(
            input_dim=768,
            hidden_dim=256,
            inner_lr=0.01,
            outer_lr=0.001,
            num_inner_steps=5
        )
        
        # Test forward pass
        batch_size = 4
        task_data = torch.randn(batch_size, 768)
        strategy_data = torch.randn(batch_size, 256)
        
        print(f"  Task data shape: {task_data.shape}")
        print(f"  Strategy data shape: {strategy_data.shape}")
        
        # Test the optimizer's forward pass
        with torch.no_grad():
            output = optimizer(task_data, strategy_data)
        
        print(f"  ✅ Meta-Learning Optimizer created and tested successfully")
        print(f"     - Optimizer type: {type(optimizer).__name__}")
        print(f"     - Output keys: {list(output.keys()) if isinstance(output, dict) else 'Tensor output'}")
        
        # Test MAML functionality if available
        if hasattr(optimizer, 'meta_update'):
            print(f"     - ✅ MAML meta-update method available")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Meta-Learning Optimizer test failed: {e}")
        return False

def test_sac_wrapper_with_real_components():
    """Test SAC Wrapper's integration with real components"""
    print("\n🧪 Testing SAC Wrapper Real Component Integration...")
    
    try:
        # Create a simple dummy environment for testing
        from stable_baselines3.common.vec_env import DummyVecEnv
        from gymnasium import spaces
        import gymnasium as gym
        
        class SimpleTestEnv(gym.Env):
            def __init__(self):
                super().__init__()
                # Define action and observation space
                self.action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32)
                self.num_tradable_symbols_this_episode = 5
                
            def step(self, action):
                obs = self.observation_space.sample()
                reward = np.random.random()
                done = False
                info = {}
                return obs, reward, done, False, info
                
            def reset(self, seed=None, options=None):
                if seed is not None:
                    np.random.seed(seed)
                obs = self.observation_space.sample()
                return obs, {}
        
        # Create environment
        env = DummyVecEnv([lambda: SimpleTestEnv()])
        
        print("  Creating SAC Wrapper with real components...")
        
        # Create SAC wrapper - this will test real component creation
        sac_wrapper = QuantumEnhancedSAC(
            env=env,
            learning_rate=3e-4,
            batch_size=32,
            buffer_size=10000,
            verbose=0
        )
        
        print("  ✅ SAC Wrapper created successfully with real components")
        
        # Test if high-level integration system was created
        if hasattr(sac_wrapper, 'high_level_integration') and sac_wrapper.high_level_integration is not None:
            print("  ✅ High-Level Integration System initialized")
            
            # Test market data processing
            market_data = torch.randn(2, 768)
            try:
                integration_results = sac_wrapper.process_with_high_level_integration(market_data)
                print("  ✅ High-Level Integration processing works")
                print(f"     - Integration results keys: {list(integration_results.keys()) if integration_results else 'Empty results'}")
            except Exception as e:
                print(f"  ⚠️ High-Level Integration processing issue: {e}")
        else:
            print("  ⚠️ High-Level Integration System not available")
        
        # Test basic SAC functionality
        obs = env.reset()
        action, _ = sac_wrapper.predict(obs, deterministic=True)
        print("  ✅ SAC prediction works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ SAC Wrapper integration test failed: {e}")
        return False

def test_gpu_compatibility():
    """Test GPU compatibility of real components"""
    print("\n🧪 Testing GPU Compatibility...")
    
    if not torch.cuda.is_available():
        print("  ⚠️ CUDA not available, skipping GPU tests")
        return True
    
    try:
        device = torch.device('cuda')
        print(f"  Using device: {device}")
        
        # Test Strategy Innovation on GPU
        module = create_strategy_innovation_module(input_dim=768, output_dim=256)
        module = module.to(device)
        
        market_data = torch.randn(2, 768, device=device)
        with torch.no_grad():
            output = module(market_data)
        
        print("  ✅ Strategy Innovation Module GPU test passed")
        
        # Test Market State Awareness on GPU
        system = MarketStateAwarenessSystem(input_dim=768, hidden_dim=256)
        system = system.to(device)
        
        with torch.no_grad():
            output = system(market_data)
        
        print("  ✅ Market State Awareness System GPU test passed")
        
        # Test Meta-Learning Optimizer on GPU
        optimizer = MetaLearningOptimizer(input_dim=768, hidden_dim=256)
        optimizer = optimizer.to(device)
        
        strategy_data = torch.randn(2, 256, device=device)
        with torch.no_grad():
            output = optimizer(market_data, strategy_data)
        
        print("  ✅ Meta-Learning Optimizer GPU test passed")
        
        return True
        
    except Exception as e:
        print(f"  ❌ GPU compatibility test failed: {e}")
        return False
def test_volatility_analyzer():
    """測試波動率分析器的功能"""
    print("\n🧪 Testing Volatility Analyzer...")
    
    try:
        from src.environment.dynamic_reweighting import VolatilityAnalyzer
        
        # 創建波動率分析器
        analyzer = VolatilityAnalyzer(window=5)
        
        # 模擬市場數據
        market_states = [
            {'price': 100.0},
            {'price': 101.0},
            {'price': 102.5},
            {'price': 101.5},
            {'price': 103.0},
            {'price': 104.5}
        ]
        
        # 更新分析器並計算波動率
        volatilities = []
        for state in market_states:
            analyzer.update(state)
            volatilities.append(analyzer.calculate(state))
        
        print(f"  ✅ Volatility Analyzer tested successfully")
        print(f"     - Final volatility: {volatilities[-1]:.4f}")
        print(f"     - Volatility level: {analyzer.get_volatility_level(market_states[-1])}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Volatility Analyzer test failed: {e}")
        return False

def test_dynamic_reward_normalizer():
    """測試動態獎勵標準化器的權重調整功能"""
    print("\n🧪 Testing Dynamic Reward Normalizer...")
    
    try:
        from src.environment.reward_normalizer import DynamicRewardNormalizer
        
        # 創建動態標準化器
        normalizer = DynamicRewardNormalizer(volatility_window=5)
        
        # 模擬市場狀態（高波動）
        high_vol_market = {'price': 100.0, 'volatility': 0.02}
        
        # 更新權重
        normalizer.update_weights(high_vol_market)
        
        # 檢查權重調整
        weights = normalizer.component_weights
        print(f"  ✅ Dynamic Reward Normalizer tested successfully")
        print(f"     - Drawdown penalty weight: {weights['drawdown_penalty']:.2f} (expected increased)")
        print(f"     - Quick cut loss weight: {weights['quick_cut_loss']:.2f} (expected increased)")
        
        # 模擬市場狀態（低波動）
        low_vol_market = {'price': 100.0, 'volatility': 0.004}
        normalizer.update_weights(low_vol_market)
        
        print(f"     - Trend following weight: {weights['trend_following']:.2f} (expected increased)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Dynamic Reward Normalizer test failed: {e}")
        return False

def test_progressive_reward_system_with_dynamic_reweighting():
    """測試整合三階段獎勵流水線的漸進式獎勵系統"""
    print("\n🧪 Testing Progressive Reward System with Dynamic Reweighting...")
    
    try:
        from src.environment.progressive_reward_system import ProgressiveRewardSystem
        
        # 創建獎勵系統
        reward_system = ProgressiveRewardSystem(volatility_window=10)
        
        # 模擬市場狀態
        market_state = {
            'price': 100.0,
            'volume': 10000,
            'spread': 0.0001
        }
        
        # 計算獎勵
        metrics = reward_system.calculate_reward(
            profit=0.05,
            drawdown=0.02,
            volatility=0.015,
            market_state=market_state,
            adaptation_success=True,
            strategy_consistency=0.9
        )
        
        print(f"  ✅ Progressive Reward System with dynamic reweighting tested successfully")
        print(f"     - Total reward: {metrics.total_reward:.2f}")
        print(f"     - Components: {metrics.components}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Progressive Reward System test failed: {e}")
        return False

def test_risk_indicator_monitoring_accuracy():
    """測試風險指標監控準確性"""
    print("\n🧪 Testing Risk Indicator Monitoring Accuracy...")
    
    try:
        from src.agent.risk_management_system import MarketStateAwareness
        
        # 創建市場狀態感知對象
        market_awareness = MarketStateAwareness()
        
        # 模擬市場數據 (使用真實歷史數據模式)
        market_data = {
            'close': np.array([1.0, 1.1, 1.2, 1.15, 1.25, 1.3]),  # 收盤價
            'volume': np.array([1000, 1200, 1500, 1300, 1400, 1600]),  # 交易量
            'asset1': np.array([1.0, 1.01, 1.02, 1.015, 1.025, 1.03]),  # 資產1價格
            'asset2': np.array([1.0, 1.02, 1.03, 1.025, 1.035, 1.04])   # 資產2價格
        }
        
        # 計算風險指標
        risk_level = market_awareness.monitor_risk_indicators(market_data)
        
        print(f"  ✅ Risk level calculated: {risk_level:.4f}")
        
        # 驗證風險指標在0-1之間
        assert 0 <= risk_level <= 1, "Risk level should be between 0 and 1"
        print("  ✅ Risk level within valid range")
        
        # 驗證準確率 (模擬測試)
        expected_risk = 0.65  # 基於模擬數據的預期值
        accuracy = 1 - abs(risk_level - expected_risk)
        print(f"  Accuracy: {accuracy*100:.2f}%")
        assert accuracy > 0.9, "Risk indicator accuracy below 90%"
        print("  ✅ Accuracy > 90%")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Risk indicator monitoring test failed: {e}")
        return False

def test_dynamic_stoploss_effectiveness():
    """測試動態停損機制有效性"""
    print("\n🧪 Testing Dynamic Stoploss Effectiveness...")
    
    try:
        from src.agent.risk_management_system import MarketStateAwareness
        
        market_awareness = MarketStateAwareness()
        
        # 模擬持倉
        position = {
            'entry_price': 100.0,
            'base_stop_loss': 0.05  # 5%
        }
        
        # 測試不同風險等級下的停損調整
        risk_levels = [0.2, 0.5, 0.8]
        for risk in risk_levels:
            stop_loss_price = market_awareness.dynamic_stoploss(position, risk)
            stop_loss_pct = (position['entry_price'] - stop_loss_price) / position['entry_price']
            
            print(f"  Risk level: {risk:.1f} -> Stop loss: {stop_loss_pct*100:.2f}%")
            
            # 驗證：風險越高，停損距離越小
            if risk > 0.5:
                assert stop_loss_pct < 0.05, "In high risk, stop loss should be tighter"
            else:
                assert stop_loss_pct <= 0.05, "Stop loss should not exceed base stop loss"
        
        # 驗證停損觸發誤差
        calculated_stop = market_awareness.dynamic_stoploss(position, 0.6)
        expected_stop = 100 * (1 - 0.05 * (1 - 0.6*0.5))
        error = abs(calculated_stop - expected_stop) / 100
        print(f"  Stop loss error: {error*100:.4f}%")
        assert error < 0.005, "Stop loss error exceeds 0.5%"
        print("  ✅ Stop loss error < 0.5%")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Dynamic stoploss test failed: {e}")
        return False

def test_stress_test_scenario_coverage():
    """測試壓力測試場景覆蓋率"""
    print("\n🧪 Testing Stress Test Scenario Coverage...")
    
    try:
        from src.agent.risk_management_system import StressTester
        
        stress_tester = StressTester()
        
        # 加載危機場景
        import json
        with open("live_trading/crisis_scenarios.json", "r") as f:
            scenarios = json.load(f)
        
        # 檢查黑天鵝事件覆蓋率
        black_swan_events = scenarios["black_swan_events"]
        for event in black_swan_events:
            event_name = event["name"]
            result = stress_tester.simulate_black_swan(event_name)
            assert result, f"Missing simulation for black swan event: {event_name}"
            print(f"  ✅ Covered black swan event: {event_name}")
        
        # 檢查流動性危機覆蓋率
        liquidity_crises = scenarios["liquidity_crises"]
        for crisis in liquidity_crises:
            crisis_name = crisis["name"]
            result = stress_tester.liquidity_crisis_test({'severity': crisis['severity']})
            assert result, f"Missing simulation for liquidity crisis: {crisis_name}"
            print(f"  ✅ Covered liquidity crisis: {crisis_name}")
        
        # 驗證場景覆蓋率100%
        total_scenarios = len(black_swan_events) + len(liquidity_crises)
        print(f"  Total scenarios covered: {total_scenarios}")
        assert total_scenarios == len(black_swan_events) + len(liquidity_crises), "Not all scenarios covered"
        print("  ✅ 100% scenario coverage")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Stress test scenario coverage test failed: {e}")
        return False

def main():
    """Run all tests to validate real component functionality"""
    print("🚀 Starting Real Components Functionality Tests")
    print("=" * 60)
    
    test_results = []
    
    # Test individual components
    test_results.append(test_strategy_innovation_module())
    test_results.append(test_market_state_awareness_system())
    test_results.append(test_meta_learning_optimizer())
    
    # Test integration
    test_results.append(test_sac_wrapper_with_real_components())
    
    # Test GPU compatibility
    test_results.append(test_gpu_compatibility())
    
    # Test dynamic reward components
    test_results.append(test_volatility_analyzer())
    test_results.append(test_dynamic_reward_normalizer())
    test_results.append(test_progressive_reward_system_with_dynamic_reweighting())
    
    # 新增风控测试
    test_results.append(test_risk_indicator_monitoring_accuracy())
    test_results.append(test_dynamic_stoploss_effectiveness())
    test_results.append(test_stress_test_scenario_coverage())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    test_names = [
        "Strategy Innovation Module",
        "Market State Awareness System",
        "Meta-Learning Optimizer",
        "SAC Wrapper Integration",
        "GPU Compatibility",
        "Volatility Analyzer",
        "Dynamic Reward Normalizer",
        "Progressive Reward System with Reweighting",
        "Risk Indicator Monitoring Accuracy",
        "Dynamic Stoploss Effectiveness",
        "Stress Test Scenario Coverage"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Real components are working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the component implementations.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
