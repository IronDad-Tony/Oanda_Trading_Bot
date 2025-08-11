import torch
import numpy as np
import pytest
from oanda_trading_bot.training_system.agent.strategy_innovation_module import QuantumInspiredGenerator, StateAwareAdapter
from oanda_trading_bot.training_system.agent.meta_learning_optimizer import MetaLearningOptimizer, GeneticSelector

def test_quantum_generator_effectiveness():
    """測試量子策略生成器的有效性"""
    # 初始化生成器
    generator = QuantumInspiredGenerator(num_strategies=5, strategy_dim=256)
    
    # 創建市場狀態
    market_state = torch.randn(1, 128)  # [batch_size, state_dim]
    
    # 生成策略組合
    strategies = generator.generate_strategy(market_state)
    
    # 驗證輸出形狀
    assert strategies.shape == (1, 1, 256), "策略形狀應為 [batch_size, 1, strategy_dim]"
    
    # 驗證策略多樣性
    diversity = torch.std(strategies).item()
    assert diversity > 0.1, "策略應有足夠多樣性 (std > 0.1)"
    
    # 驗證量子態更新
    initial_states = generator.quantum_states.clone()
    _ = generator.generate_strategy(market_state)
    assert not torch.allclose(initial_states, generator.quantum_states), "量子態應在每次生成後更新"

def test_state_adapter_stability():
    """測試狀態適配器的穩定性"""
    # 初始化適配器
    adapter = StateAwareAdapter(volatility_factor=0.7, risk_aversion=0.8)
    
    # 創建策略和市場狀態
    strategy = torch.randn(1, 256)  # [batch_size, strategy_dim]
    market_state = torch.tensor([[0.5, 0.8, 0.3, 0.9]])  # 最後一個值為波動率
    
    # 適配策略
    adapted_strategy = adapter.adapt_strategy(strategy, market_state)
    
    # 驗證輸出形狀
    assert adapted_strategy.shape == strategy.shape, "適配前後策略形狀應一致"
    
    # 驗證調整因子應用
    adjustment_factor = 1.0 - (market_state[..., -1].unsqueeze(-1) * adapter.volatility_factor) * adapter.risk_aversion
    expected_strategy = strategy * adjustment_factor
    assert torch.allclose(adapted_strategy, expected_strategy, atol=1e-5), "應正確應用調整因子"
    
    # 測試極端值穩定性
    extreme_market = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # 最高波動率
    extreme_adapted = adapter.adapt_strategy(strategy, extreme_market)
    assert torch.isfinite(extreme_adapted).all(), "極端市場條件下應保持穩定"

def test_genetic_algorithm_convergence():
    """測試遺傳算法的收斂性"""
    # 初始化遺傳選擇器
    selector = GeneticSelector(population_size=20, mutation_rate=0.2)
    
    # 模擬適應度進化
    convergence_history = []
    for generation in range(10):
        # 評估當前種群
        fitness_scores = []
        for params in selector.population:
            # 模擬適應度函數：學習率越高、折扣因子越高則適應度越好
            fitness = params['learning_rate'] * 10000 + params['discount_factor'] * 10
            fitness_scores.append(fitness)
        
        # 記錄最佳適應度
        best_fitness = max(fitness_scores)
        convergence_history.append(best_fitness)
        
        # 進化到下一代
        selector.evolve(fitness_scores)
    
    # 驗證收斂性：後三代應優於前三代
    early_avg = np.mean(convergence_history[:3])
    late_avg = np.mean(convergence_history[-3:])
    assert late_avg > early_avg, "遺傳算法應隨代數增加而改進"
    
    # 驗證超參數範圍
    best_params = selector.population[np.argmax(fitness_scores)]
    assert 1e-5 <= best_params['learning_rate'] <= 1e-2, "學習率應在合理範圍"
    assert 0.8 <= best_params['discount_factor'] <= 0.99, "折扣因子應在合理範圍"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
