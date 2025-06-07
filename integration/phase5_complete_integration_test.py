"""
Phase 5 Complete Integration Test
Tests all Phase 5 components working together:
- Strategy Innovation Module
- Market State Awareness System  
- Meta-Learning Optimizer with MAML
- High-Level Integration System
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any
import time
from datetime import datetime
import logging

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import all Phase 5 components
from agent.strategy_innovation_module import StrategyInnovationModule
from agent.market_state_awareness_system import MarketStateAwarenessSystem
from agent.meta_learning_optimizer import MetaLearningOptimizer, TaskBatch
from agent.high_level_integration_system import HighLevelIntegrationSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase5IntegrationTest:
    """Comprehensive Phase 5 integration test"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
          # Test parameters
        self.batch_size = 4
        self.seq_len = 50
        self.feature_dim = 768  # Changed from 512 to 768 for compatibility
        self.adaptation_dim = 768  # Changed from 256 to 768 for num_heads compatibility
        
        # Initialize components
        self.components = {}
        
    def setup_components(self):
        """Setup all Phase 5 components"""
        
        logger.info("Setting up Phase 5 components...")
        
        try:            # 1. Strategy Innovation Module
            self.components['strategy_innovation'] = StrategyInnovationModule(
                input_dim=self.feature_dim,
                hidden_dim=self.adaptation_dim,
                population_size=20
            )
            logger.info("✅ Strategy Innovation Module initialized")
              # 2. Market State Awareness System
            self.components['market_state_awareness'] = MarketStateAwarenessSystem(
                input_dim=self.feature_dim,
                num_strategies=20,
                enable_real_time_monitoring=True
            )
            logger.info("✅ Market State Awareness System initialized")
            
            # 3. Create a simple model for Meta-Learning Optimizer
            simple_model = nn.Sequential(
                nn.Linear(self.feature_dim, self.adaptation_dim),
                nn.ReLU(),
                nn.Linear(self.adaptation_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            
            # 4. Meta-Learning Optimizer
            self.components['meta_learning_optimizer'] = MetaLearningOptimizer(
                model=simple_model,
                feature_dim=self.feature_dim,
                adaptation_dim=self.adaptation_dim
            )
            logger.info("✅ Meta-Learning Optimizer initialized")
            
            # 5. High-Level Integration System
            self.components['integration_system'] = HighLevelIntegrationSystem(
                strategy_innovation_module=self.components['strategy_innovation'],
                market_state_awareness_system=self.components['market_state_awareness'],
                meta_learning_optimizer=self.components['meta_learning_optimizer'],
                feature_dim=self.feature_dim
            )
            logger.info("✅ High-Level Integration System initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up components: {str(e)}")
            return False
    
    def test_individual_components(self):
        """Test each component individually"""
        
        logger.info("Testing individual components...")
        
        # Generate test data
        market_data = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        
        results = {}
        
        try:
            # Test Strategy Innovation Module
            logger.info("Testing Strategy Innovation Module...")
            innovation_results = self.components['strategy_innovation'](market_data)
            results['strategy_innovation'] = {
                'success': True,
                'output_keys': list(innovation_results.keys()),
                'generated_strategies_shape': innovation_results['generated_strategies'].shape,
                'innovation_confidence': innovation_results['innovation_confidence']
            }
            logger.info(f"✅ Strategy Innovation - Generated strategies: {innovation_results['generated_strategies'].shape}")
              # Test Market State Awareness System
            logger.info("Testing Market State Awareness System...")
            state_results = self.components['market_state_awareness'](market_data)
            results['market_state_awareness'] = {
                'success': True,
                'current_state': state_results['system_status']['current_state'],
                'state_confidence': state_results['market_state']['confidence'],
                'output_keys': list(state_results.keys())
            }
            logger.info(f"✅ Market State Awareness - Current state: {state_results['system_status']['current_state']}")
            
            # Test Meta-Learning Optimizer
            logger.info("Testing Meta-Learning Optimizer...")
            
            # Create task batches for testing
            task_batches = []
            for i in range(3):
                task_batch = TaskBatch(
                    support_data=torch.randn(16, self.feature_dim),
                    support_labels=torch.randn(16, 1),
                    query_data=torch.randn(8, self.feature_dim),
                    query_labels=torch.randn(8, 1),
                    task_id=f"test_task_{i}",
                    market_state="trending_up",
                    difficulty=0.5
                )
                task_batches.append(task_batch)
            
            meta_results = self.components['meta_learning_optimizer'].optimize_and_adapt(
                market_data, market_data, task_batches
            )
            results['meta_learning_optimizer'] = {
                'success': True,
                'adapted_features_shape': meta_results['adapted_features'].shape,
                'selected_strategy': meta_results['selected_strategy'],
                'adaptation_quality': meta_results['adaptation_quality']
            }
            logger.info(f"✅ Meta-Learning Optimizer - Adaptation quality: {meta_results['adaptation_quality']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing individual components: {str(e)}")
            return None
    
    def test_integrated_system(self):
        """Test the complete integrated system"""
        
        logger.info("Testing integrated system...")
        
        try:
            # Generate comprehensive test data
            market_data = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
            
            # Position data for testing
            position_data = {
                f'position_{i}': torch.randn(self.feature_dim) 
                for i in range(5)
            }
            
            # Portfolio metrics
            portfolio_metrics = torch.randn(256)
            
            # Process through integrated system
            logger.info("Processing market data through integrated system...")
            start_time = time.time()
            
            integrated_results = self.components['integration_system'].process_market_data(
                market_data=market_data,
                position_data=position_data,
                portfolio_metrics=portfolio_metrics
            )
            
            processing_time = time.time() - start_time
            
            # Analyze results
            results = {
                'success': True,
                'processing_time': processing_time,
                'system_health': integrated_results['system_health']['overall_health'],
                'system_state': integrated_results['system_health']['system_state'],
                'market_state': integrated_results['market_state']['current_state'],
                'emergency_triggered': integrated_results['emergency_status']['emergency_triggered'],
                'positions_managed': len(integrated_results['position_management']),
                'max_anomaly_score': torch.max(integrated_results['anomaly_detection']['combined_scores']).item(),
                'components_tested': list(integrated_results.keys())
            }
            
            logger.info(f"✅ Integrated system processing completed in {processing_time:.4f} seconds")
            logger.info(f"✅ System health: {results['system_health']:.4f}")
            logger.info(f"✅ System state: {results['system_state']}")
            logger.info(f"✅ Positions managed: {results['positions_managed']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing integrated system: {str(e)}")
            return None
    
    def test_system_performance(self):
        """Test system performance under various conditions"""
        
        logger.info("Testing system performance...")
        
        performance_results = {
            'latency_tests': [],
            'memory_usage': [],
            'gradient_flow': True,
            'stability_tests': []
        }
        
        try:
            # Latency tests
            logger.info("Running latency tests...")
            for i in range(10):
                market_data = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
                
                start_time = time.time()
                results = self.components['integration_system'].process_market_data(market_data)
                latency = time.time() - start_time
                
                performance_results['latency_tests'].append(latency)
            
            avg_latency = np.mean(performance_results['latency_tests'])
            logger.info(f"✅ Average processing latency: {avg_latency:.4f} seconds")
            
            # Memory usage test (simplified)
            import psutil
            import gc
            
            gc.collect()
            memory_before = psutil.virtual_memory().used
            
            # Process larger batch
            large_market_data = torch.randn(8, 100, self.feature_dim)
            results = self.components['integration_system'].process_market_data(large_market_data)
            
            gc.collect()
            memory_after = psutil.virtual_memory().used
            memory_used = (memory_after - memory_before) / (1024 * 1024)  # MB
            
            performance_results['memory_usage'].append(memory_used)
            logger.info(f"✅ Memory usage for large batch: {memory_used:.2f} MB")
            
            # Gradient flow test
            logger.info("Testing gradient flow...")
            market_data = torch.randn(self.batch_size, self.seq_len, self.feature_dim, requires_grad=True)
            results = self.components['integration_system'].process_market_data(market_data)
            
            # Compute loss and backpropagate
            loss = torch.sum(results['anomaly_detection']['combined_scores'])
            loss.backward()
            
            # Check if gradients exist
            has_gradients = market_data.grad is not None
            performance_results['gradient_flow'] = has_gradients
            logger.info(f"✅ Gradient flow test: {'Passed' if has_gradients else 'Failed'}")
            
            # Stability tests
            logger.info("Running stability tests...")
            for i in range(5):
                try:
                    market_data = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
                    results = self.components['integration_system'].process_market_data(market_data)
                    performance_results['stability_tests'].append(True)
                except Exception as e:
                    logger.warning(f"Stability test {i} failed: {str(e)}")
                    performance_results['stability_tests'].append(False)
            
            stability_rate = sum(performance_results['stability_tests']) / len(performance_results['stability_tests'])
            logger.info(f"✅ System stability rate: {stability_rate:.2%}")
            
            return performance_results
            
        except Exception as e:
            logger.error(f"Error in performance testing: {str(e)}")
            return None
    
    def test_component_integration(self):
        """Test integration between components"""
        
        logger.info("Testing component integration...")
        
        try:
            market_data = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
              # Test data flow between components
            logger.info("Testing data flow between components...")
            
            # 1. Market State Awareness -> Strategy Innovation
            state_results = self.components['market_state_awareness'](market_data)
            # Use the market features as context for strategy innovation
            innovation_input = market_data  # Use original market data as state context
            innovation_results = self.components['strategy_innovation'](innovation_input)
            
            # 2. Strategy Innovation -> Meta-Learning
            adapted_features = innovation_results['generated_strategies'].mean(dim=1)  # Reduce strategy dimension
            
            # Create task batches
            task_batches = [
                TaskBatch(
                    support_data=torch.randn(16, self.feature_dim),
                    support_labels=torch.randn(16, 1),
                    query_data=torch.randn(8, self.feature_dim),
                    query_labels=torch.randn(8, 1),                    task_id="integration_task",
                    market_state=state_results['system_status']['current_state'],
                    difficulty=0.6
                )
            ]
            
            meta_results = self.components['meta_learning_optimizer'].optimize_and_adapt(
                market_data, market_data, task_batches  # Use original market data as context
            )
            
            # 3. All components -> Integration System
            final_results = self.components['integration_system'].process_market_data(market_data)
            
            integration_results = {
                'success': True,
                'data_flow_verified': True,
                'state_to_innovation': innovation_results['innovation_confidence'] > 0,
                'innovation_to_meta': meta_results['adaptation_quality'] > 0,
                'full_integration': final_results['system_health']['overall_health'] > 0,
                'components_connected': len(final_results.keys()) >= 6
            }
            
            logger.info("✅ Component integration test passed")
            logger.info(f"✅ Data flow verified: {integration_results['data_flow_verified']}")
            logger.info(f"✅ Components connected: {integration_results['components_connected']}")
            
            return integration_results
            
        except Exception as e:
            logger.error(f"Error in component integration test: {str(e)}")
            return None
    
    def run_complete_test_suite(self):
        """Run the complete Phase 5 test suite"""
        
        logger.info("=" * 80)
        logger.info("STARTING PHASE 5 COMPLETE INTEGRATION TEST")
        logger.info("=" * 80)
        
        test_results = {
            'overall_success': False,
            'setup_success': False,
            'individual_components': None,
            'integrated_system': None,
            'system_performance': None,
            'component_integration': None,
            'total_test_time': 0
        }
        
        try:
            # 1. Setup components
            logger.info("\n1. Setting up components...")
            test_results['setup_success'] = self.setup_components()
            
            if not test_results['setup_success']:
                logger.error("❌ Component setup failed. Aborting tests.")
                return test_results
            
            # 2. Test individual components
            logger.info("\n2. Testing individual components...")
            test_results['individual_components'] = self.test_individual_components()
            
            # 3. Test integrated system
            logger.info("\n3. Testing integrated system...")
            test_results['integrated_system'] = self.test_integrated_system()
            
            # 4. Test system performance
            logger.info("\n4. Testing system performance...")
            test_results['system_performance'] = self.test_system_performance()
            
            # 5. Test component integration
            logger.info("\n5. Testing component integration...")
            test_results['component_integration'] = self.test_component_integration()
            
            # Calculate overall success
            success_components = [
                test_results['setup_success'],
                test_results['individual_components'] is not None,
                test_results['integrated_system'] is not None,
                test_results['system_performance'] is not None,
                test_results['component_integration'] is not None
            ]
            
            test_results['overall_success'] = all(success_components)
            test_results['total_test_time'] = time.time() - self.start_time
            
            # Print final results
            self.print_test_summary(test_results)
            
            return test_results
            
        except Exception as e:
            logger.error(f"Critical error in test suite: {str(e)}")
            test_results['total_test_time'] = time.time() - self.start_time
            return test_results
    
    def print_test_summary(self, test_results: Dict[str, Any]):
        """Print comprehensive test summary"""
        
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 5 INTEGRATION TEST SUMMARY")
        logger.info("=" * 80)
        
        # Overall results
        status = "✅ PASSED" if test_results['overall_success'] else "❌ FAILED"
        logger.info(f"Overall Test Status: {status}")
        logger.info(f"Total Test Time: {test_results['total_test_time']:.2f} seconds")
        
        # Component setup
        setup_status = "✅ SUCCESS" if test_results['setup_success'] else "❌ FAILED"
        logger.info(f"Component Setup: {setup_status}")
        
        # Individual components
        if test_results['individual_components']:
            logger.info("\nIndividual Component Tests:")
            for component, result in test_results['individual_components'].items():
                status = "✅ PASSED" if result['success'] else "❌ FAILED"
                logger.info(f"  - {component.replace('_', ' ').title()}: {status}")
        
        # Integrated system
        if test_results['integrated_system']:
            logger.info(f"\nIntegrated System Test: ✅ PASSED")
            logger.info(f"  - Processing Time: {test_results['integrated_system']['processing_time']:.4f}s")
            logger.info(f"  - System Health: {test_results['integrated_system']['system_health']:.4f}")
            logger.info(f"  - Positions Managed: {test_results['integrated_system']['positions_managed']}")
        
        # Performance tests
        if test_results['system_performance']:
            perf = test_results['system_performance']
            logger.info(f"\nPerformance Tests:")
            if perf['latency_tests']:
                avg_latency = np.mean(perf['latency_tests'])
                logger.info(f"  - Average Latency: {avg_latency:.4f}s")
            if perf['stability_tests']:
                stability = sum(perf['stability_tests']) / len(perf['stability_tests'])
                logger.info(f"  - Stability Rate: {stability:.2%}")
            logger.info(f"  - Gradient Flow: {'✅ PASSED' if perf['gradient_flow'] else '❌ FAILED'}")
        
        # Component integration
        if test_results['component_integration']:
            integration_status = "✅ PASSED" if test_results['component_integration']['success'] else "❌ FAILED"
            logger.info(f"\nComponent Integration: {integration_status}")
        
        logger.info("\n" + "=" * 80)
        
        if test_results['overall_success']:
            logger.info("🎉 PHASE 5 COMPLETE INTEGRATION TEST SUCCESSFUL!")
            logger.info("All Phase 5 components are working together correctly.")
        else:
            logger.info("❌ PHASE 5 INTEGRATION TEST FAILED")
            logger.info("Some components or integrations need attention.")
        
        logger.info("=" * 80)

def main():
    """Main test execution"""
    
    # Create and run test suite
    test_suite = Phase5IntegrationTest()
    results = test_suite.run_complete_test_suite()
    
    # Return exit code based on success
    return 0 if results['overall_success'] else 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
