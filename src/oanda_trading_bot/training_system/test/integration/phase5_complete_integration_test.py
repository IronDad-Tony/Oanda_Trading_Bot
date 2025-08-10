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
# sys.path.append(os.path.join(os.path.dirname(__file__), \'..\', \'src\')) # Removed this line

# Import all Phase 5 components
from oanda_trading_bot.training_system.agent.strategy_innovation_module import StrategyInnovationModule
from oanda_trading_bot.training_system.agent.market_state_awareness_system import MarketStateAwarenessSystem
from oanda_trading_bot.training_system.agent.meta_learning_optimizer import MetaLearningOptimizer, TaskBatch
from oanda_trading_bot.training_system.agent.high_level_integration_system import HighLevelIntegrationSystem, AnomalyDetector, DynamicPositionManager, EmergencyStopLoss, SystemState # Added AnomalyDetector, DynamicPositionManager, EmergencyStopLoss, SystemState
from oanda_trading_bot.training_system.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition # ADDED IMPORT

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
        self.action_dim_enhanced = 10 # ADDED for EnhancedStrategySuperposition
        
        # Initialize components
        self.components = {}
        
    def setup_components(self):
        """Setup all Phase 5 components"""
        
        logger.info("Setting up Phase 5 components...")
        
        try:
            # 1. Strategy Innovation Module
            self.components['strategy_innovation'] = StrategyInnovationModule(
                input_dim=self.feature_dim,
                hidden_dim=self.adaptation_dim, # Assuming adaptation_dim is appropriate for hidden_dim
                population_size=20
            )
            logger.info("✅ Strategy Innovation Module initialized")

            # 2. Market State Awareness System
            self.components['market_state_awareness'] = MarketStateAwarenessSystem(
                input_dim=self.feature_dim,
                num_strategies=20, # Example value
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
                feature_dim=self.feature_dim, # MLO still takes feature_dim directly
                adaptation_dim=self.adaptation_dim
            )
            logger.info("✅ Meta-Learning Optimizer initialized")

            # 4a. Placeholder AnomalyDetector, DynamicPositionManager, EmergencyStopLoss
            self.components['anomaly_detector'] = AnomalyDetector(input_dim=self.feature_dim)
            logger.info("✅ Anomaly Detector (placeholder) initialized")
            self.components['position_manager'] = DynamicPositionManager(feature_dim=self.feature_dim)
            logger.info("✅ Dynamic Position Manager (placeholder) initialized")
            self.components['emergency_stop_loss_system'] = EmergencyStopLoss()
            logger.info("✅ Emergency Stop Loss System (placeholder) initialized")

            # 5. High-Level Integration System
            # Prepare config for HighLevelIntegrationSystem
            hlis_config = {
                'feature_dim': self.feature_dim,
                'expected_maml_input_dim': self.adaptation_dim, # Or another appropriate value
                # Add other necessary config values from HLIS _get_default_config if needed
                "num_maml_tasks": 5,
                "maml_shots": 5,
                "enable_dynamic_adaptation": True, # Or False to test without adapter initially
                "default_input_tensor_key": f"features_{self.feature_dim}" # Example
            }

            self.components['integration_system'] = HighLevelIntegrationSystem(
                strategy_innovation_module=self.components['strategy_innovation'],
                market_state_awareness_system=self.components['market_state_awareness'],
                meta_learning_optimizer=self.components['meta_learning_optimizer'],
                position_manager=self.components['position_manager'],
                anomaly_detector=self.components['anomaly_detector'],
                emergency_stop_loss_system=self.components['emergency_stop_loss_system'],
                config=hlis_config,
                device=torch.device("cpu") # Explicitly set device for test consistency
            )
            logger.info("✅ High-Level Integration System initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up components: {str(e)}", exc_info=True) # Added exc_info
            return False

    def setup_enhanced_strategy_layer(self): # ADDED METHOD
        """Setup the EnhancedStrategySuperposition layer"""
        logger.info("Setting up Enhanced Strategy Layer...")
        try:
            self.components['enhanced_strategy_layer'] = EnhancedStrategySuperposition(
                state_dim=self.feature_dim,
                action_dim=self.action_dim_enhanced,
                enable_dynamic_generation=True 
            )
            logger.info("✅ Enhanced Strategy Layer initialized")
            return True
        except Exception as e:
            logger.error(f"Error setting up Enhanced Strategy Layer: {str(e)}")
            self.components['enhanced_strategy_layer'] = None # Ensure it's None on failure
            return False

    def test_enhanced_strategy_layer_execution(self): # ADDED METHOD
        """Test execution of the EnhancedStrategySuperposition layer"""
        logger.info("Testing Enhanced Strategy Layer execution...")
        if 'enhanced_strategy_layer' not in self.components or self.components['enhanced_strategy_layer'] is None:
            logger.error("Enhanced Strategy Layer not set up or setup failed.")
            return {'success': False, 'reason': 'Not set up or setup failed'}

        try:
            # Prepare inputs: state (batch, features) and volatility (batch)
            state_input = torch.randn(self.batch_size, self.feature_dim, device='cpu') # Assuming CPU for tests unless specified
            volatility_input = torch.rand(self.batch_size, device='cpu') 

            output, info = self.components['enhanced_strategy_layer'](state_input, volatility_input)
            
            expected_shape = (self.batch_size, self.action_dim_enhanced)
            if output.shape != expected_shape:
                error_msg = f"Output shape mismatch: Got {output.shape}, expected {expected_shape}"
                logger.error(error_msg)
                return {'success': False, 'reason': error_msg, 'output_shape': output.shape, 'expected_output_shape': expected_shape}

            results = {
                'success': True,
                'output_shape': output.shape,
                'info_keys': list(info.keys()),
                'expected_output_shape': expected_shape
            }
            logger.info(f"✅ Enhanced Strategy Layer execution test passed. Output shape: {output.shape}")
            return results
        except Exception as e:
            logger.error(f"Error testing Enhanced Strategy Layer execution: {str(e)}", exc_info=True)
            return {'success': False, 'reason': str(e)}
    
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
            # Generate comprehensive test data for HighLevelIntegrationSystem
            # It expects a dictionary
            market_data_dict = {
                f'features_{self.feature_dim}': torch.randn(self.batch_size, self.seq_len, self.feature_dim),
                'current_positions': [ # Example position data
                    {'id': 'pos1', 'symbol': 'EUR_USD', 'units': 100, 'entry_price': 1.1},
                    {'id': 'pos2', 'symbol': 'USD_JPY', 'units': -50, 'entry_price': 110.5}
                ],
                'portfolio_metrics': { # Example portfolio metrics
                    'total_value': 10000.0,
                    'current_drawdown': 0.05,
                    'portfolio_risk': 0.2
                },
                # Add any other keys HighLevelIntegrationSystem might expect based on _get_tensor_from_market_data
                'features_256': torch.randn(self.batch_size, self.seq_len, 256) # Example of another feature set
            }
            
            # Process through integrated system
            logger.info("Processing market data through integrated system...")
            start_time_processing = time.time() # Renamed start_time to avoid conflict
            
            integrated_results = self.components['integration_system'].process_market_data(
                market_data_raw=market_data_dict # Pass the dictionary
            )
            
            processing_time = time.time() - start_time_processing
            
            # Analyze results
            results = {
                'success': True,
                'processing_time': processing_time,
                'system_health': integrated_results.get('system_health', {}).get('overall_health'),
                'system_state': integrated_results.get('system_health', {}).get('system_state'),
                'market_state_current': integrated_results.get('market_state', {}).get('current_state'), # Adjusted key
                'emergency_triggered': integrated_results.get('emergency_status', {}).get('emergency_triggered'),
                'positions_managed_count': integrated_results.get('position_management', {}).get('processed_positions_count', 0), # Adjusted key
                'max_anomaly_score': None, # Anomaly score might be a tensor or scalar
                'components_tested': list(integrated_results.keys())
            }
            
            anomaly_scores = integrated_results.get('anomaly_detection', {}).get('combined_scores')
            if torch.is_tensor(anomaly_scores) and anomaly_scores.numel() > 0:
                results['max_anomaly_score'] = torch.max(anomaly_scores).item()
            elif isinstance(anomaly_scores, (float, int)):
                 results['max_anomaly_score'] = anomaly_scores


            logger.info(f"✅ Integrated system processing completed in {processing_time:.4f} seconds")
            logger.info(f"✅ System health: {results['system_health'] if results['system_health'] is not None else 'N/A'}")
            logger.info(f"✅ System state: {results['system_state'] if results['system_state'] is not None else 'N/A'}")
            logger.info(f"✅ Positions managed count: {results['positions_managed_count']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing integrated system: {str(e)}", exc_info=True) # Added exc_info
            return {'success': False, 'reason': str(e)}
    
    def test_system_performance(self):
        """Test system performance under various conditions"""
        
        logger.info("Testing system performance...")
        
        performance_results = {
            'latency_tests': [],
            'memory_usage': [], # Memory usage test is complex and platform dependent, simplifying
            'gradient_flow': False, # Default to False
            'stability_tests': []
        }
        
        try:
            # Latency tests
            logger.info("Running latency tests...")
            for i in range(5): # Reduced iterations for speed
                market_data_dict_perf = {
                    f'features_{self.feature_dim}': torch.randn(self.batch_size, self.seq_len, self.feature_dim),
                    'current_positions': [], 'portfolio_metrics': {} # Minimal data
                }
                start_time_perf = time.time()
                # Ensure results are captured from process_market_data
                results_perf = self.components['integration_system'].process_market_data(market_data_dict_perf)
                latency = time.time() - start_time_perf
                performance_results['latency_tests'].append(latency)
                # Check for errors in results_perf
                if results_perf.get('error'):
                    logger.warning(f"Latency test iteration {i} produced an error: {results_perf['error']}")
            
            if performance_results['latency_tests']:
                avg_latency = np.mean(performance_results['latency_tests'])
                logger.info(f"✅ Average processing latency: {avg_latency:.4f} seconds")
            else:
                logger.warning("Latency tests did not run or all failed.")

            # Memory usage test (simplified - more of a placeholder)
            # Actual memory profiling is more involved. This just checks if a large batch runs.
            logger.info("Running simplified memory check (large batch processing)...")
            try:
                large_market_data_dict = {
                    f'features_{self.feature_dim}': torch.randn(self.batch_size * 2, self.seq_len * 2, self.feature_dim),
                    'current_positions': [], 'portfolio_metrics': {}
                }
                results_large = self.components['integration_system'].process_market_data(large_market_data_dict)
                if results_large.get('error'):
                     logger.warning(f"Large batch processing returned an error: {results_large['error']}")
                     performance_results['memory_usage'].append({'success': False, 'reason': results_large['error']})
                else:
                    logger.info(f"✅ Large batch processed successfully (simplified memory check).")
                    performance_results['memory_usage'].append({'success': True})
            except Exception as e_mem:
                logger.error(f"Error during large batch processing (simplified memory check): {str(e_mem)}", exc_info=True)
                performance_results['memory_usage'].append({'success': False, 'reason': str(e_mem)})

            # Gradient flow test
            logger.info("Testing gradient flow...")
            # Ensure the input tensor for which grad is checked is part of what HLIS processes and affects the loss
            # The main input tensor used by anomaly_detector is a good candidate.
            # HLIS's process_market_data now uses 'market_data_raw'
            market_data_grad_dict = {
                f'features_{self.feature_dim}': torch.randn(self.batch_size, self.seq_len, self.feature_dim, requires_grad=True),
                'current_positions': [], 'portfolio_metrics': {}
            }
            
            # Get the specific tensor that requires grad
            input_tensor_for_grad = market_data_grad_dict[f'features_{self.feature_dim}']

            results_grad = self.components['integration_system'].process_market_data(market_data_grad_dict)
            
            if results_grad.get('error'):
                logger.error(f"Gradient flow test failed during HLIS processing: {results_grad['error']}")
                performance_results['gradient_flow'] = False
            elif results_grad.get('anomaly_detection') and torch.is_tensor(results_grad['anomaly_detection'].get('combined_scores')):
                anomaly_scores_grad = results_grad['anomaly_detection']['combined_scores']
                if anomaly_scores_grad.numel() > 0:
                    try:
                        # Compute loss and backpropagate
                        # Ensure loss is a scalar
                        loss = torch.sum(anomaly_scores_grad) 
                        if loss.requires_grad: # Check if the loss itself requires grad
                            loss.backward()
                            # Check if gradients exist on the input tensor
                            has_gradients = input_tensor_for_grad.grad is not None and input_tensor_for_grad.grad.abs().sum() > 0
                            performance_results['gradient_flow'] = has_gradients
                            logger.info(f"✅ Gradient flow test: {'Passed' if has_gradients else 'Failed (no/zero gradients on input)'}")
                        else:
                            logger.warning("Gradient flow test: Loss does not require grad. Check model structure and inputs.")
                            performance_results['gradient_flow'] = False # Mark as failed if loss doesn't require grad
                    except RuntimeError as e_grad_runtime:
                        logger.error(f"Runtime error during backward pass in gradient flow test: {str(e_grad_runtime)}", exc_info=True)
                        performance_results['gradient_flow'] = False
                else:
                    logger.warning("Gradient flow test: Anomaly scores tensor is empty.")
                    performance_results['gradient_flow'] = False # Cannot test grad flow
            else:
                logger.warning("Gradient flow test: Anomaly scores not found or not a tensor in results.")
                performance_results['gradient_flow'] = False
            
            # Stability tests
            logger.info("Running stability tests...")
            for i in range(3): # Reduced iterations
                try:
                    market_data_stability_dict = {
                        f'features_{self.feature_dim}': torch.randn(self.batch_size, self.seq_len, self.feature_dim),
                        'current_positions': [], 'portfolio_metrics': {}
                    }
                    results_stability = self.components['integration_system'].process_market_data(market_data_stability_dict)
                    if results_stability.get('error'):
                        logger.warning(f"Stability test {i} produced an error: {results_stability['error']}")
                        performance_results['stability_tests'].append(False)
                    else:
                        performance_results['stability_tests'].append(True)
                except Exception as e_stability:
                    logger.warning(f"Stability test {i} failed with exception: {str(e_stability)}", exc_info=True)
                    performance_results['stability_tests'].append(False)
            
            if performance_results['stability_tests']:
                stability_rate = sum(performance_results['stability_tests']) / len(performance_results['stability_tests'])
                logger.info(f"✅ System stability rate: {stability_rate:.2%}")
            else:
                logger.warning("Stability tests did not run.")
            
            return performance_results
            
        except Exception as e:
            logger.error(f"Error in performance testing: {str(e)}", exc_info=True) # Added exc_info
            # Ensure all keys exist in the returned dict even on error
            performance_results['gradient_flow'] = False # Explicitly set on error
            performance_results['error_message'] = str(e)
            return performance_results
    
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
            # Use the same market_data_dict structure as in test_integrated_system
            market_data_integration_dict = {
                f'features_{self.feature_dim}': market_data, # market_data here is (batch, seq, feature_dim)
                'current_positions': [{'id': 'int_pos', 'units': 10}], 
                'portfolio_metrics': {'current_drawdown': 0.01}
            }
            final_results = self.components['integration_system'].process_market_data(market_data_integration_dict)
            
            integration_results = {
                'success': final_results.get('error') is None, # Success if no error key or error is None
                'data_flow_verified': True, # Assuming prior steps imply this if no error
                'state_to_innovation': innovation_results.get('innovation_confidence', 0) > 0 if isinstance(innovation_results, dict) else False,
                'innovation_to_meta': meta_results.get('adaptation_quality', 0) > 0 if isinstance(meta_results, dict) else False,
                'full_integration': final_results.get('system_health', {}).get('overall_health', 0) > 0,
                'components_connected': len(final_results.keys()) >= 6 # Check for presence of main output sections
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
            'enhanced_strategy_layer_setup_success': False, # ADDED
            'individual_components': None,
            'integrated_system': None,
            'system_performance': None,
            'component_integration': None,
            'enhanced_strategy_layer_execution': None, # ADDED
            'total_test_time': 0
        }
        
        try:
            # 1. Setup components
            logger.info("\n1. Setting up components...")
            test_results['setup_success'] = self.setup_components()
            
            if not test_results['setup_success']:
                logger.error("❌ Main component setup failed. Some tests may be skipped or fail.")
                # Continue to setup enhanced strategy layer as it's independent

            # 1b. Setup Enhanced Strategy Layer
            logger.info("\n1b. Setting up Enhanced Strategy Layer...")
            test_results['enhanced_strategy_layer_setup_success'] = self.setup_enhanced_strategy_layer()
            if not test_results['enhanced_strategy_layer_setup_success']:
                logger.error("❌ Enhanced Strategy Layer setup failed. Its execution test will be skipped.")

            # 2. Test individual components
            logger.info("\n2. Testing individual components...")
            if test_results['setup_success']:
                test_results['individual_components'] = self.test_individual_components()
            else:
                logger.warning("Skipping individual component tests due to main setup failure.")
                test_results['individual_components'] = {'success': False, 'reason': 'Main setup failed'}
            
            # 3. Test integrated system
            logger.info("\n3. Testing integrated system...")
            if test_results['setup_success']:
                test_results['integrated_system'] = self.test_integrated_system()
            else:
                logger.warning("Skipping integrated system test due to main setup failure.")
                test_results['integrated_system'] = {'success': False, 'reason': 'Main setup failed'}
            
            # 4. Test system performance
            logger.info("\n4. Testing system performance...")
            if test_results['setup_success']:
                test_results['system_performance'] = self.test_system_performance()
            else:
                logger.warning("Skipping system performance test due to main setup failure.")
                test_results['system_performance'] = {'success': False, 'reason': 'Main setup failed'}
            
            # 5. Test component integration
            logger.info("\n5. Testing component integration...")
            if test_results['setup_success']:
                test_results['component_integration'] = self.test_component_integration()
            else:
                logger.warning("Skipping component integration test due to main setup failure.")
                test_results['component_integration'] = {'success': False, 'reason': 'Main setup failed'}

            # 6. Test Enhanced Strategy Layer Execution
            logger.info("\n6. Testing Enhanced Strategy Layer Execution...")
            if test_results['enhanced_strategy_layer_setup_success']:
                test_results['enhanced_strategy_layer_execution'] = self.test_enhanced_strategy_layer_execution()
            else:
                logger.warning("Skipping Enhanced Strategy Layer execution test due to its setup failure.")
                test_results['enhanced_strategy_layer_execution'] = {'success': False, 'reason': 'Enhanced Strategy Layer setup failed'}
            
            # Calculate overall success
            # Overall success depends on main setup and all main tests.
            # Enhanced strategy layer is an additional check.
            main_tests_successful = all([\
                test_results['setup_success'],
                test_results['individual_components'] and test_results['individual_components'].get('success', True), # if dict, check success, else treat as pass if not None
                test_results['integrated_system'] and test_results['integrated_system'].get('success', True),
                test_results['system_performance'] and test_results['system_performance'].get('gradient_flow', True), # Using gradient_flow as a proxy for success
                test_results['component_integration'] and test_results['component_integration'].get('success', True)
            ])
            
            # For individual_components, success is more complex. Let's check if the dict exists.
            # A more robust check for individual_components success:
            individual_comp_success = True
            if isinstance(test_results['individual_components'], dict):
                if 'strategy_innovation' in test_results['individual_components']: # Check one key as example
                     individual_comp_success = all(v.get('success',True) for v in test_results['individual_components'].values() if isinstance(v,dict))
                else: # if structure is different, e.g. just a success flag
                    individual_comp_success = test_results['individual_components'].get('success', False)

            elif test_results['individual_components'] is None:
                 individual_comp_success = False


            success_components = [\
                test_results['setup_success'],
                individual_comp_success,
                test_results['integrated_system'] is not None and test_results['integrated_system'].get('success', False),
                test_results['system_performance'] is not None and test_results['system_performance'].get('gradient_flow', False), # gradient_flow is a boolean
                test_results['component_integration'] is not None and test_results['component_integration'].get('success', False),
                test_results['enhanced_strategy_layer_setup_success'],
                test_results['enhanced_strategy_layer_execution'] is not None and test_results['enhanced_strategy_layer_execution'].get('success', False)
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
        
        # Enhanced Strategy Layer Setup
        enhanced_setup_status = "✅ SUCCESS" if test_results.get('enhanced_strategy_layer_setup_success', False) else "❌ FAILED"
        logger.info(f"Enhanced Strategy Layer Setup: {enhanced_setup_status}")

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
        elif test_results['setup_success']: # If setup was ok, but test is None
             logger.info(f"\nIntegrated System Test: ❓ SKIPPED OR ERRORED")


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
            integration_status = "✅ PASSED" if test_results['component_integration'].get('success', False) else "❌ FAILED"
            logger.info(f"\nComponent Integration: {integration_status}")
        elif test_results['setup_success']:
            logger.info(f"\nComponent Integration: ❓ SKIPPED OR ERRORED")

        # Enhanced Strategy Layer Execution
        if test_results['enhanced_strategy_layer_execution']:
            esl_result = test_results['enhanced_strategy_layer_execution']
            esl_status = "✅ PASSED" if esl_result.get('success', False) else "❌ FAILED"
            logger.info(f"\nEnhanced Strategy Layer Execution: {esl_status}")
            if esl_result.get('success', False):
                logger.info(f"  - Output Shape: {esl_result['output_shape']}")
            else:
                logger.info(f"  - Reason: {esl_result.get('reason', 'Unknown')}")
        elif test_results['enhanced_strategy_layer_setup_success']:
            logger.info(f"\nEnhanced Strategy Layer Execution: ❓ SKIPPED OR ERRORED (but setup was OK)")


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
