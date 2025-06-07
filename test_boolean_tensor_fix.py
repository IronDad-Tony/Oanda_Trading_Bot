#!/usr/bin/env python3
"""
Test script to reproduce and fix the boolean tensor ambiguity error
in the High Level Integration System's Emergency Stop Loss
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

def test_emergency_stop_loss_boolean_tensor_error():
    """Test to reproduce the boolean tensor ambiguity error"""
    
    logger.info("🚀 Testing EmergencyStopLoss for Boolean Tensor Error")
    
    try:
        from src.agent.high_level_integration_system import EmergencyStopLoss
        
        # Create EmergencyStopLoss instance
        emergency_stop = EmergencyStopLoss()
        
        # Create test data that will trigger the boolean tensor error
        portfolio_metrics = torch.randn(256)  # Random portfolio metrics tensor
        current_drawdown = 0.06  # Above threshold (0.05)
        portfolio_risk = 0.12    # Above threshold (0.10)
        
        logger.info("📊 Running emergency condition check...")
        logger.info(f"Portfolio metrics shape: {portfolio_metrics.shape}")
        logger.info(f"Current drawdown: {current_drawdown}")
        logger.info(f"Portfolio risk: {portfolio_risk}")
        
        # This should trigger the boolean tensor ambiguity error
        emergency_response = emergency_stop.check_emergency_conditions(
            portfolio_metrics=portfolio_metrics,
            current_drawdown=current_drawdown,
            portfolio_risk=portfolio_risk
        )
        
        logger.info("✅ Emergency condition check completed successfully!")
        logger.info(f"Emergency triggered: {emergency_response['emergency_triggered']}")
        
    except Exception as e:
        logger.error(f"❌ Error occurred: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        return False
    
    return True

def test_high_level_integration_system():
    """Test the complete HighLevelIntegrationSystem"""
    
    logger.info("🚀 Testing HighLevelIntegrationSystem for Boolean Tensor Error")
    
    try:
        from src.agent.high_level_integration_system import HighLevelIntegrationSystem
        from src.agent.strategy_innovation_module import create_strategy_innovation_module
        from src.agent.market_state_awareness_system import MarketStateAwarenessSystem
        from src.agent.meta_learning_optimizer import MetaLearningOptimizer
          # Create required components
        strategy_innovation = create_strategy_innovation_module()
        market_state_awareness = MarketStateAwarenessSystem(input_dim=768)
        
        # Create a dummy model for MetaLearningOptimizer
        dummy_model = torch.nn.Linear(768, 256)
        meta_learning_optimizer = MetaLearningOptimizer(model=dummy_model, feature_dim=768)
        
        # Create HighLevelIntegrationSystem
        integration_system = HighLevelIntegrationSystem(
            strategy_innovation_module=strategy_innovation,
            market_state_awareness_system=market_state_awareness,
            meta_learning_optimizer=meta_learning_optimizer,
            feature_dim=768
        )
        
        # Create test market data
        market_data = torch.randn(2, 768)  # Batch size 2, feature dim 768
        portfolio_metrics = torch.randn(2, 256)  # Batch size 2, metrics dim 256
        
        logger.info("📊 Running process_market_data...")
        logger.info(f"Market data shape: {market_data.shape}")
        logger.info(f"Portfolio metrics shape: {portfolio_metrics.shape}")
        
        # This should trigger the boolean tensor ambiguity error in emergency stop loss
        results = integration_system.process_market_data(
            market_data=market_data,
            portfolio_metrics=portfolio_metrics
        )
        
        logger.info("✅ Market data processing completed successfully!")
        logger.info(f"Emergency status: {results['emergency_status']['emergency_triggered']}")
        
    except Exception as e:
        logger.error(f"❌ Error occurred: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        if "Boolean value of Tensor with more than one value is ambiguous" in str(e):
            logger.error("🎯 CONFIRMED: This is the boolean tensor ambiguity error!")
        return False
    
    return True

if __name__ == "__main__":
    logger.info("🧪 Starting Boolean Tensor Error Reproduction Test\n")
    
    # Test 1: Direct EmergencyStopLoss test
    logger.info("=" * 60)
    test1_success = test_emergency_stop_loss_boolean_tensor_error()
    
    # Test 2: Complete HighLevelIntegrationSystem test
    logger.info("\n" + "=" * 60)
    test2_success = test_high_level_integration_system()
    
    logger.info("\n" + "=" * 60)
    if test1_success and test2_success:
        logger.info("✅ All tests passed - no boolean tensor error detected")
    else:
        logger.error("❌ Boolean tensor error reproduced - applying fix...")
