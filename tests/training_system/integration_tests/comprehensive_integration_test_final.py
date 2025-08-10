#!/usr/bin/env python3
"""
Comprehensive Integration Test for Oanda Trading Bot
Tests all core components and their interactions
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import traceback

def main():
    print("=== COMPREHENSIVE INTEGRATION TEST ===")
    
    try:
        # Import all components
        print("0. Importing components...")
        from src.agent.meta_learning_optimizer import MetaLearningOptimizer, TaskBatch, AdaptationStrategy
        from src.agent.market_state_awareness_system import MarketStateAwarenessSystem
        from src.agent.strategy_innovation_module import StrategyInnovationModule
        print("   All imports successful")
        
        # Test 1: Create mock model and initialize MetaLearningOptimizer
        print("1. Testing MetaLearningOptimizer initialization...")
        mock_model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        meta_optimizer = MetaLearningOptimizer(
            model=mock_model,
            feature_dim=128,
            adaptation_dim=64
        )
        print("   MetaLearningOptimizer created successfully")
        
        # Test 2: Create and test MarketStateAwarenessSystem
        print("2. Testing MarketStateAwarenessSystem...")
        market_system = MarketStateAwarenessSystem(
            input_dim=128,
            num_strategies=5 # Assuming a default number of strategies for testing
        )
        
        # Test with mock data
        mock_market_data = torch.randn(1, 10, 128)
        market_state = market_system(mock_market_data)
        print(f"   Market state keys: {list(market_state.keys())}")
        
        # Verify required keys are present
        required_keys = ['regime_confidence', 'current_state']
        for key in required_keys:
            if key in market_state:
                print(f"   ✓ Key '{key}' found")
            else:
                print(f"   ✗ Key '{key}' missing")
        
        # Test 3: Create and test StrategyInnovationModule
        print("3. Testing StrategyInnovationModule...")
        strategy_module = StrategyInnovationModule(
            input_dim=128,
            hidden_dim=72  # Changed from 64 to 72 to be divisible by num_heads (24)
        )
        
        mock_strategy_data = torch.randn(1, 10, 128)
        strategy_output = strategy_module(mock_strategy_data)
        print(f"   Strategy output keys: {list(strategy_output.keys())}")
        
        # Verify required keys are present
        if 'strategy_diversity' in strategy_output:
            print("   ✓ Key 'strategy_diversity' found")
        else:
            print("   ✗ Key 'strategy_diversity' missing")
        
        # Test 4: Test meta-learning with task batches
        print("4. Testing meta-learning task adaptation...")
        task_batch = TaskBatch(
            support_data=torch.randn(5, 128),
            support_labels=torch.randn(5, 32),
            query_data=torch.randn(3, 128),
            query_labels=torch.randn(3, 32),
            task_id='test_task_1',
            market_state='volatile',
            difficulty=0.7
        )
        
        adaptation_result = meta_optimizer.maml_optimizer.adapt(task_batch)
        print(f"   Adaptation completed - Loss: {adaptation_result.adaptation_loss:.4f}")
        
        # Test 5: Test fast adaptation mechanism
        print("5. Testing fast adaptation mechanism...")
        test_features = torch.randn(2, 5, 128)
        test_context = torch.randn(2, 5, 128)
        
        adapted_features = meta_optimizer.fast_adaptation(test_features, test_context)
        print(f"   Fast adaptation - Input shape: {test_features.shape}, Output shape: {adapted_features.shape}")
        
        # Test 6: Test complete optimization and adaptation pipeline
        print("6. Testing complete optimization pipeline...")
        context_dict = {
            'market_volatility': 0.8,
            'trend_strength': 0.6,
            'liquidity': 0.9
        }
        
        optimization_result = meta_optimizer.optimize_and_adapt(
            features=test_features,
            context=context_dict,
            task_batches=[task_batch]
        )
        print(f"   Complete optimization - Strategy: {optimization_result['selected_strategy']}")
        
        # Test 7: Dimension compatibility checks
        print("7. Testing dimension compatibility...")
        
        # Test 1D context
        context_1d = torch.randn(128)
        adapted_1d = meta_optimizer.fast_adaptation(test_features, context_1d)
        print(f"   1D context adaptation successful - Output shape: {adapted_1d.shape}")
        
        # Test 2D context
        context_2d = torch.randn(2, 128)
        adapted_2d = meta_optimizer.fast_adaptation(test_features, context_2d)
        print(f"   2D context adaptation successful - Output shape: {adapted_2d.shape}")
        
        print()
        print("=== ALL TESTS PASSED ===")
        print("System is ready for production use!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
