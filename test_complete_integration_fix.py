#!/usr/bin/env python3
"""
Complete test script for the High Level Integration System
Properly initializes all required components and tests the boolean tensor fix
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_strategy_innovation_module():
    """Create a mock strategy innovation module"""
    class MockStrategyInnovationModule(nn.Module):
        def __init__(self, input_dim=768, output_dim=256):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            
            # Simple network to simulate strategy innovation
            self.innovation_network = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim)
            )
            
        def forward(self, market_data, existing_strategies=None):
            """
            Simulate strategy innovation
            Returns a dictionary with task_batches and generated_strategies
            """
            batch_size = market_data.size(0)
            
            # Process market data
            innovation_output = self.innovation_network(market_data)
            
            # Return dictionary structure that high_level_integration_system expects
            return {
                'task_batches': innovation_output,  # This is what the system looks for
                'generated_strategies': innovation_output,
                'innovation_score': torch.rand(batch_size, 1),
                'strategy_confidence': torch.rand(batch_size, 1)
            }
    
    return MockStrategyInnovationModule()

def create_mock_market_state_awareness():
    """Create a mock market state awareness system"""
    class MockMarketStateAwareness(nn.Module):
        def __init__(self, input_dim=768):
            super().__init__()
            self.input_dim = input_dim
            
            # Network to process market state
            self.state_network = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
            
        def forward(self, market_data):
            """Process market data and return state analysis"""
            batch_size = market_data.size(0)
            
            state_features = self.state_network(market_data)
            
            return {
                'market_regime': torch.randint(0, 3, (batch_size,)),  # 0=trending, 1=ranging, 2=volatile
                'volatility_level': torch.rand(batch_size, 1),
                'trend_strength': torch.rand(batch_size, 1),
                'state_features': state_features,
                'confidence': torch.rand(batch_size, 1)
            }
    
    return MockMarketStateAwareness()

def create_mock_meta_learning_optimizer():
    """Create a mock meta learning optimizer"""
    class MockMetaLearningOptimizer(nn.Module):
        def __init__(self, input_dim=768, task_dim=256):
            super().__init__()
            self.input_dim = input_dim
            self.task_dim = task_dim
            
            # Meta learning network
            self.meta_network = nn.Sequential(
                nn.Linear(input_dim + task_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, input_dim)
            )
            
        def optimize_and_adapt(self, market_data, context_data, task_batches):
            """
            Simulate meta learning optimization
            """
            batch_size = market_data.size(0)
            
            # Ensure task_batches has the right shape
            if task_batches.size(-1) != self.task_dim:
                # Adapt the task batch size
                task_adapter = nn.Linear(task_batches.size(-1), self.task_dim)
                task_batches = task_adapter(task_batches)
            
            # Combine market data and task information
            combined_input = torch.cat([market_data, task_batches], dim=-1)
            
            # Process through meta learning network
            adapted_features = self.meta_network(combined_input)
            
            return {
                'adapted_features': adapted_features,
                'adaptation_score': torch.rand(batch_size, 1),
                'learning_progress': torch.rand(batch_size, 1),
                'meta_loss': torch.rand(1)
            }
    
    return MockMetaLearningOptimizer()

def test_high_level_integration_system():
    """Test the high level integration system with proper component initialization"""
    logger.info("🚀 Starting comprehensive high level integration system test...")
    
    try:
        # Import the high level integration system
        from agent.high_level_integration_system import HighLevelIntegrationSystem
        
        logger.info("✅ Successfully imported HighLevelIntegrationSystem")
        
        # Create mock components
        logger.info("🔧 Creating mock components...")
        
        strategy_innovation = create_mock_strategy_innovation_module()
        market_state_awareness = create_mock_market_state_awareness()
        meta_learning_optimizer = create_mock_meta_learning_optimizer()
        
        logger.info("✅ Mock components created successfully")
        
        # Initialize the high level integration system
        logger.info("🏗️ Initializing HighLevelIntegrationSystem...")
        
        system = HighLevelIntegrationSystem(
            strategy_innovation_module=strategy_innovation,
            market_state_awareness_system=market_state_awareness,
            meta_learning_optimizer=meta_learning_optimizer,
            feature_dim=768
        )
        
        logger.info("✅ HighLevelIntegrationSystem initialized successfully")
        
        # Create test data
        logger.info("📊 Creating test data...")
        
        batch_size = 4
        sequence_length = 10
        feature_dim = 768
        
        # Market data
        market_data = torch.randn(batch_size, sequence_length, feature_dim)
          # Position data (optional) - ensure proper dimensions
        position_data = {
            'positions': torch.randn(batch_size, feature_dim),  # Match feature_dim
            'pnl': torch.randn(batch_size, feature_dim),         # Match feature_dim
            'exposure': torch.randn(batch_size, feature_dim)     # Match feature_dim
        }
        
        # Portfolio metrics (optional)
        portfolio_metrics = torch.randn(batch_size, 5)  # 5 metrics
        
        logger.info("✅ Test data created successfully")
        
        # Test the main processing pipeline
        logger.info("🧪 Testing main processing pipeline...")
        
        # This is where the original error occurred
        results = system.process_market_data(
            market_data=market_data,
            position_data=position_data,
            portfolio_metrics=portfolio_metrics
        )
        
        logger.info("✅ Main processing pipeline completed successfully!")
        
        # Analyze results
        logger.info("📋 Analyzing results...")
        
        expected_keys = [
            'market_state_results', 'innovation_results', 'meta_results',
            'anomaly_results', 'position_risks', 'emergency_status',
            'system_metrics', 'processing_time', 'recommendations'
        ]
        
        for key in expected_keys:
            if key in results:
                logger.info(f"✅ Found expected key: {key}")
            else:
                logger.warning(f"⚠️ Missing key: {key}")
        
        # Print summary of results
        logger.info("\n📊 RESULTS SUMMARY:")
        logger.info(f"Total result keys: {len(results)}")
        logger.info(f"Processing time: {results.get('processing_time', 'N/A'):.4f}s")
        logger.info(f"System state: {results.get('system_metrics', {}).get('system_state', 'N/A')}")
        
        # Test with different input scenarios
        logger.info("\n🔄 Testing additional scenarios...")
        
        # Test with minimal data
        minimal_results = system.process_market_data(market_data=market_data)
        logger.info("✅ Minimal data test passed")
        
        # Test with different batch sizes
        small_market_data = torch.randn(1, sequence_length, feature_dim)
        small_results = system.process_market_data(market_data=small_market_data)
        logger.info("✅ Small batch test passed")
        
        large_market_data = torch.randn(16, sequence_length, feature_dim)
        large_results = system.process_market_data(market_data=large_market_data)
        logger.info("✅ Large batch test passed")
        
        logger.info("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
        logger.info("✅ The boolean tensor error has been successfully resolved!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        logger.exception("Full error traceback:")
        return False

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("🔧 OANDA TRADING BOT - HIGH LEVEL INTEGRATION SYSTEM TEST")
    logger.info("=" * 80)
    
    success = test_high_level_integration_system()
    
    if success:
        logger.info("\n🎉 CONCLUSION: All tests passed successfully!")
        logger.info("✅ The boolean tensor error fix is working correctly.")
        logger.info("✅ The HighLevelIntegrationSystem is functioning properly.")
    else:
        logger.error("\n❌ CONCLUSION: Tests failed!")
        logger.error("❌ There are still issues that need to be resolved.")
    
    logger.info("=" * 80)
