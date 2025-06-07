#!/usr/bin/env python3
"""
Debug script to find the exact location of the boolean tensor ambiguity error
"""

import sys
import os
sys.path.append('src')

import torch
import numpy as np
from datetime import datetime
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_exact_integration_system_error():
    """Test to find the exact location of the boolean tensor error"""
    
    logger.info("🎯 Debug: Finding exact location of boolean tensor ambiguity error")
    
    try:
        from src.agent.high_level_integration_system import HighLevelIntegrationSystem
        from src.agent.strategy_innovation_module import create_strategy_innovation_module
        from src.agent.market_state_awareness_system import MarketStateAwarenessSystem
        from src.agent.meta_learning_optimizer import MetaLearningOptimizer
        
        # Create required components
        logger.info("✅ Creating components...")
        strategy_innovation = create_strategy_innovation_module()
        market_state_awareness = MarketStateAwarenessSystem(input_dim=768)
        meta_learning_optimizer = MetaLearningOptimizer(feature_dim=768)
        
        # Create HighLevelIntegrationSystem
        logger.info("✅ Creating HighLevelIntegrationSystem...")
        integration_system = HighLevelIntegrationSystem(
            strategy_innovation_module=strategy_innovation,
            market_state_awareness_system=market_state_awareness,
            meta_learning_optimizer=meta_learning_optimizer,
            feature_dim=768
        )
        
        # Create test market data - exact same as in the test that fails
        logger.info("📊 Creating test data...")
        market_data = torch.randn(2, 768)  # Batch size 2, feature dim 768
        portfolio_metrics = torch.randn(2, 256)  # Batch size 2, metrics dim 256
        
        logger.info(f"Market data shape: {market_data.shape}")
        logger.info(f"Portfolio metrics shape: {portfolio_metrics.shape}")
        
        # Step by step processing to find the exact error location
        
        # 1. Test market state awareness
        logger.info("🔍 Step 1: Testing market state awareness...")
        try:
            market_state_results = market_state_awareness(market_data)
            logger.info("✅ Market state awareness completed")
        except Exception as e:
            logger.error(f"❌ Error in market state awareness: {e}")
            logger.error(traceback.format_exc())
            return False
        
        # 2. Test strategy innovation
        logger.info("🔍 Step 2: Testing strategy innovation...")
        try:
            innovation_results = strategy_innovation(
                market_data,
                existing_strategies=None
            )
            logger.info("✅ Strategy innovation completed")
        except Exception as e:
            logger.error(f"❌ Error in strategy innovation: {e}")
            logger.error(traceback.format_exc())
            return False
        
        # 3. Test meta-learning
        logger.info("🔍 Step 3: Testing meta-learning...")
        try:
            meta_results = {'adapted_features': market_data}  # Simplified
            logger.info("✅ Meta-learning completed")
        except Exception as e:
            logger.error(f"❌ Error in meta-learning: {e}")
            logger.error(traceback.format_exc())
            return False
        
        # 4. Test anomaly detection
        logger.info("🔍 Step 4: Testing anomaly detection...")
        try:
            anomaly_results = integration_system.anomaly_detector(market_data)
            logger.info("✅ Anomaly detection completed")
        except Exception as e:
            logger.error(f"❌ Error in anomaly detection: {e}")
            logger.error(traceback.format_exc())
            return False
        
        # 5. Test emergency stop loss
        logger.info("🔍 Step 5: Testing emergency stop loss...")
        try:
            emergency_status = integration_system.emergency_stop_loss.check_emergency_conditions(
                portfolio_metrics.mean(dim=0) if portfolio_metrics.dim() > 1 else portfolio_metrics,
                current_drawdown=0.02,
                portfolio_risk=0.03
            )
            logger.info("✅ Emergency stop loss completed")
        except Exception as e:
            logger.error(f"❌ Error in emergency stop loss: {e}")
            logger.error(traceback.format_exc())
            return False
        
        # 6. Test system health assessment
        logger.info("🔍 Step 6: Testing system health assessment...")
        try:
            system_health = integration_system._assess_system_health(
                market_state_results,
                innovation_results,
                anomaly_results,
                emergency_status
            )
            logger.info("✅ System health assessment completed")
        except Exception as e:
            logger.error(f"❌ Error in system health assessment: {e}")
            logger.error(traceback.format_exc())
            return False
        
        # 7. Test the complete pipeline
        logger.info("🔍 Step 7: Testing complete pipeline...")
        try:
            results = integration_system.process_market_data(
                market_data=market_data,
                portfolio_metrics=portfolio_metrics
            )
            logger.info("✅ Complete pipeline test completed successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ Error in complete pipeline: {e}")
            logger.error(traceback.format_exc())
            
            # If error contains boolean tensor message, print detailed info
            if "Boolean value of Tensor with more than one value is ambiguous" in str(e):
                logger.error("🎯 FOUND IT! This is the boolean tensor ambiguity error location!")
                logger.error("Looking at the traceback above to find the exact line...")
            
            return False
        
    except Exception as e:
        logger.error(f"❌ Failed to set up test: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting Boolean Tensor Debug Session\n")
    
    success = test_exact_integration_system_error()
    
    if success:
        logger.info("\n✅ All tests passed - no boolean tensor error found")
    else:
        logger.error("\n❌ Boolean tensor error found - check logs above for location")
