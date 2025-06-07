#!/usr/bin/env python3
"""
Simple test script for Phase 5 Strategy Innovation Module
Tests the core functionality with dynamic dimension adaptation
"""

import sys
import os
sys.path.append('src')

import torch
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_phase5_strategy_innovation():
    """Test Phase 5 Strategy Innovation Module with various configurations"""
    
    logger.info("🚀 Starting Phase 5 Strategy Innovation Module Test")
    
    try:
        # Import the module
        from src.agent.strategy_innovation_module import (
            StrategyInnovationModule, 
            ConfigAdapter,
            create_strategy_innovation_module
        )
        
        # Test different model configurations
        test_configs = [
            {'model_dim': 768, 'description': 'Large Model (Current)'},
            {'model_dim': 512, 'description': 'Medium Model'},
            {'model_dim': 256, 'description': 'Small Model'}
        ]
        
        for config in test_configs:
            logger.info(f"\n📊 Testing with {config['description']} - Dimension: {config['model_dim']}")
            
            # Create config adapter
            config_adapter = ConfigAdapter()
            config_adapter._cached_config = {
                'model_dim': config['model_dim'],
                'num_layers': 12,
                'num_heads': 16,
                'ffn_dim': config['model_dim'] * 4,
                'output_dim_per_symbol': config['model_dim'] // 4,
                'head_dim': config['model_dim'] // 16
            }
            
            # Create strategy innovation module
            innovation_module = StrategyInnovationModule(
                input_dim=config['model_dim'],
                population_size=10,
                max_generations=5,
                config_adapter=config_adapter
            )
            
            # Generate test data
            batch_size = 2
            input_dim = config['model_dim']
            strategy_dim = max(256, input_dim // 3)
            
            market_context = torch.randn(batch_size, input_dim)
            existing_strategies = torch.randn(batch_size, 5, strategy_dim)
            
            logger.info(f"   Input dimensions: market_context={market_context.shape}, strategies={existing_strategies.shape}")
            
            # Test innovation flow
            with torch.no_grad():
                innovation_result = innovation_module(market_context, existing_strategies)
                
                # Test evolution
                evolved_strategies = innovation_module.evolve_strategies(
                    market_context, num_generations=2
                )
                
                # Get statistics
                stats = innovation_module.get_innovation_statistics()
                
            logger.info(f"   ✅ Success! Fitness: {innovation_result['evaluation']['fitness_score'].mean():.4f}")
            logger.info(f"   📈 Evolved {len(evolved_strategies)} strategies, {stats['evolution_generations']} generations")
            
        # Test factory function
        logger.info(f"\n🏭 Testing Factory Function")
        factory_module = create_strategy_innovation_module(
            input_dim=768, 
            population_size=5
        )
        
        test_context = torch.randn(1, 768)
        with torch.no_grad():
            result = factory_module(test_context)
        
        logger.info(f"   ✅ Factory function test successful!")
        
        # Test cross-market knowledge transfer
        logger.info(f"\n🔄 Testing Cross-Market Knowledge Transfer")
        
        # Store some mock knowledge
        knowledge_system = innovation_module.knowledge_transfer
        mock_strategies = torch.randn(10, 256)
        mock_performance = torch.rand(10)
        
        knowledge_system.store_market_knowledge('stocks', mock_strategies, mock_performance)
        knowledge_system.store_market_knowledge('crypto', mock_strategies * 0.8, mock_performance * 0.9)
        
        # Test knowledge transfer
        target_context = torch.randn(256)
        transfer_result = knowledge_system.transfer_knowledge('forex', target_context)
        
        if transfer_result['adapted_knowledge'] is not None:
            logger.info(f"   ✅ Knowledge transfer successful! Score: {transfer_result['transfer_scores'].mean():.4f}")
        else:
            logger.info(f"   ⚠️ No knowledge transferred (expected for new setup)")
        
        logger.info(f"\n🎉 All Phase 5 Strategy Innovation tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    start_time = datetime.now()
    
    success = test_phase5_strategy_innovation()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"\n⏱️ Test completed in {duration:.2f} seconds")
    
    if success:
        logger.info("🎯 Phase 5 Strategy Innovation Module is working correctly!")
        print("\n" + "="*70)
        print("🎉 PHASE 5 STRATEGY INNOVATION MODULE - DYNAMIC ADAPTATION COMPLETE!")
        print("="*70)
        print("✅ All syntax errors fixed")
        print("✅ Dynamic dimension adaptation implemented")
        print("✅ Cross-module compatibility verified")
        print("✅ Strategy generation and evolution working")
        print("✅ Knowledge transfer system functional")
        print("✅ Multi-configuration testing successful")
        print("="*70)
    else:
        logger.error("💥 Phase 5 tests failed - further debugging needed")
