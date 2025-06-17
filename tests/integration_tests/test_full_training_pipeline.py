\
#!/usr/bin/env python3
"""
Integration test for the full training pipeline.

This test verifies that the training process can be initiated,
the SAC agent (with EnhancedTransformer and QuantumStrategyLayer) runs,
and that model weights are updated, indicating learning.
"""
import sys
import os
import torch
import logging
import unittest
from datetime import datetime
import copy # For deepcopying model state_dict

# Add the project root to the Python path
# For a test file in tests/integration_tests/, project_root is two levels up.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Attempt to import necessary components from the training pipeline and other modules
try:
    from scripts.training_pipeline import create_training_env, CONFIG as DEFAULT_PIPELINE_CONFIG
    from src.agent.sac_agent_wrapper import QuantumEnhancedSAC
    from src.agent.sac_policy import CustomSACPolicy # MODIFIED: Import CustomSACPolicy
    from src.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition
    from stable_baselines3.common.callbacks import CheckpointCallback
    TRAINING_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Error importing training components: {e}")
    TRAINING_COMPONENTS_AVAILABLE = False
    raise e # MODIFIED: Re-raise the import error to see the underlying cause

# --- Logger Setup ---
def setup_test_logging(log_level: int = logging.INFO):
    """Configures basic logging for the test."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)] # Log to stdout for test visibility
    )
    logging.getLogger("stable_baselines3").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)
    # Suppress matplotlib font manager spam
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


logger = logging.getLogger(__name__)

@unittest.skipIf(not TRAINING_COMPONENTS_AVAILABLE, "Training components not available, skipping integration test.") # MODIFIED: Restored skipIf
class TestFullTrainingPipeline(unittest.TestCase):

    def setUp(self):
        """Set up test parameters and environment."""
        setup_test_logging(logging.INFO)
        logger.info("Setting up test for full training pipeline.")

        self.test_config = copy.deepcopy(DEFAULT_PIPELINE_CONFIG)
        
        # Override with minimal settings for a quick test run
        self.test_config["symbols"] = ["EUR_USD"] 
        self.test_config["train_start_iso"] = "2023-01-01T00:00:00Z" 
        self.test_config["train_end_iso"] = "2023-01-03T00:00:00Z"   
        self.test_config["total_timesteps"] = 100  
        self.test_config["model_save_freq"] = 50
        self.test_config["model_save_path"] = os.path.join(project_root, "trained_models_test/sac_universal_trader_test")
        self.test_config["log_level"] = logging.DEBUG 
        self.test_config["sac_params"]["batch_size"] = 32 
        self.test_config["sac_params"]["learning_starts"] = 10 
        self.test_config["sac_params"]["buffer_size"] = 1000 

        # Ensure test model save path exists
        os.makedirs(os.path.dirname(self.test_config["model_save_path"]), exist_ok=True)
        
        self.env = None

    def tearDown(self):
        """Clean up after the test."""
        logger.info("Tearing down test for full training pipeline.")
        if self.env:
            self.env.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Optionally, clean up created model files if necessary,
        # but often it\'s good to inspect them after a test run.

    def get_model_parameters(self, model_component):
        """Helper to get a deepcopy of model parameters."""
        return {name: param.clone().detach() for name, param in model_component.named_parameters()}

    def test_training_run_and_weight_update(self):
        """
        Tests a short training run, ensuring it completes and model weights are updated.
        """
        logger.info("Starting training run and weight update test.")
        logger.info(f"Test configuration: {self.test_config}")

        try:
            self.env = create_training_env(
                symbols=self.test_config["symbols"],
                start_time_iso=self.test_config["train_start_iso"],
                end_time_iso=self.test_config["train_end_iso"],
                env_params=self.test_config["env_params"]
            )
            self.assertIsNotNone(self.env, "Environment creation failed.")

            logger.info("Initializing QuantumEnhancedSAC agent with CustomSACPolicy for testing.")
            
            agent_wrapper = QuantumEnhancedSAC(
                env=self.env, 
                policy=CustomSACPolicy, # MODIFIED: Pass CustomSACPolicy class
                verbose=1, 
                **self.test_config["sac_params"]
            )
            sb3_agent = agent_wrapper.agent 
            self.assertIsNotNone(sb3_agent, "SAC Agent initialization failed.")
            self.assertIsNotNone(sb3_agent.policy, "SAC Agent policy is None.")
            self.assertIsInstance(sb3_agent.policy, CustomSACPolicy, "Agent policy is not an instance of CustomSACPolicy.")

            # --- EnhancedTransformer Weight Check ---
            transformer_component = None
            # In CustomSACPolicy, EnhancedTransformerFeatureExtractor is self.features_extractor
            if hasattr(sb3_agent.policy, 'features_extractor') and \
               hasattr(sb3_agent.policy.features_extractor, 'transformer_model') and \
               sb3_agent.policy.features_extractor.transformer_model is not None:
                transformer_component = sb3_agent.policy.features_extractor.transformer_model
                logger.info("Checking weights of EnhancedTransformer (policy.features_extractor.transformer_model).")
            else:
                logger.warning("Could not locate EnhancedTransformer in policy.features_extractor. Will check actor parameters as a whole.")
                # Fallback: check the actor if the specific transformer isn't found as expected.
                # This might indicate a structural change or misconfiguration.
                if hasattr(sb3_agent.policy, 'actor'):
                    transformer_component = sb3_agent.policy.actor
                    logger.info("Falling back to check weights of the entire actor network.")
                else:
                    self.fail("Neither policy.features_extractor.transformer_model nor policy.actor found for weight check.")

            self.assertIsNotNone(transformer_component, "EnhancedTransformer component not found in the agent's policy.")
            self.assertTrue(len(list(transformer_component.parameters())) > 0,
                            "EnhancedTransformer component for weight check has no parameters.")
            initial_transformer_weights = self.get_model_parameters(transformer_component)

            # --- Quantum Strategy Layer Check ---
            quantum_layer_component = None
            # In CustomSACPolicy, quantum_layer is a direct attribute of the policy instance
            if hasattr(sb3_agent.policy, 'quantum_layer') and sb3_agent.policy.quantum_layer is not None:
                quantum_layer_component = sb3_agent.policy.quantum_layer
                logger.info("Located Quantum Strategy Layer (policy.quantum_layer).")
                self.assertTrue(sb3_agent.policy.use_quantum_layer, "policy.use_quantum_layer is False, but layer was found.")
            elif hasattr(sb3_agent.policy, 'use_quantum_layer') and not sb3_agent.policy.use_quantum_layer:
                logger.warning("Quantum Strategy Layer is explicitly disabled (policy.use_quantum_layer is False). Skipping quantum checks.")
                # If it's disabled, we might not want the test to fail, but to acknowledge it.
                # For this test, we expect it to be active.
                self.fail("Quantum Strategy Layer is disabled (policy.use_quantum_layer is False), but expected to be active.")
            else:
                logger.warning("Could not locate Quantum Strategy Layer directly in policy.quantum_layer.")
                self.fail("Quantum Strategy Layer (policy.quantum_layer) not found.")

            self.assertIsNotNone(quantum_layer_component, "Quantum Strategy Layer not found in the agent's policy.")
            
            self.assertIsInstance(quantum_layer_component, EnhancedStrategySuperposition, "Quantum layer is not an EnhancedStrategySuperposition instance.")
            logger.info(f"Number of strategies loaded in Quantum Layer: {quantum_layer_component.num_actual_strategies}")
            self.assertTrue(quantum_layer_component.num_actual_strategies > 0, "No strategies were loaded into the Quantum Strategy Layer.")

            initial_quantum_layer_weights = {}
            if quantum_layer_component.attention_network is not None and \
               len(list(quantum_layer_component.attention_network.parameters())) > 0:
                logger.info("Checking weights of Quantum Layer's attention_network.")
                initial_quantum_layer_weights.update(self.get_model_parameters(quantum_layer_component.attention_network))
            
            for i, strategy_module in enumerate(quantum_layer_component.strategies):
                if hasattr(strategy_module, 'parameters') and len(list(strategy_module.parameters())) > 0:
                    logger.info(f"Checking weights of strategy: {quantum_layer_component.strategy_names[i]}")
                    initial_quantum_layer_weights.update(
                        {f"strategy_{i}_{name}": param.clone().detach() for name, param in strategy_module.named_parameters()}
                    )
            
            self.assertTrue(len(initial_quantum_layer_weights) > 0, 
                            "Quantum layer (attention or strategies) has no trainable parameters to check for updates.")
            initial_adaptive_bias_weights = None
            if quantum_layer_component.adaptive_bias_weights is not None:
                 initial_adaptive_bias_weights = quantum_layer_component.adaptive_bias_weights.clone().detach()
                 logger.info(f"Initial adaptive_bias_weights: {initial_adaptive_bias_weights}")


            # Select a representative part of the EnhancedTransformer to check for weight updates.
            # This path depends on how EnhancedTransformer is integrated into the SAC policy.
            # Common SB3 paths: sb3_agent.policy.features_extractor or sb3_agent.policy.actor / sb3_agent.policy.critic
            # Let\'s assume EnhancedTransformer is part of the actor\'s features_extractor or main net.
            # If EnhancedTransformer is complex, pick a specific, non-trivial layer.
            # For this example, we\'ll try to get actor parameters.
            # This might need adjustment based on your actual model structure.
            
            target_component_to_check = sb3_agent.policy.actor
            if hasattr(sb3_agent.policy, 'features_extractor') and sb3_agent.policy.features_extractor is not None:
                 # If a dedicated features_extractor (potentially the EnhancedTransformer) exists
                if len(list(sb3_agent.policy.features_extractor.parameters())) > 0:
                    target_component_to_check = sb3_agent.policy.features_extractor
                    logger.info("Checking weights of policy.features_extractor.")
                else:
                    logger.info("policy.features_extractor has no parameters, checking policy.actor instead.")
            else:
                logger.info("No policy.features_extractor, checking policy.actor.")

            self.assertTrue(len(list(target_component_to_check.parameters())) > 0, 
                            "Target component for weight check has no parameters.")

            initial_weights = self.get_model_parameters(target_component_to_check)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename_prefix = f"{os.path.basename(self.test_config['model_save_path'])}_{timestamp}"
            
            checkpoint_callback = CheckpointCallback(
                save_freq=self.test_config["model_save_freq"],
                save_path=os.path.join(os.path.dirname(self.test_config["model_save_path"]), "checkpoints_test"),
                name_prefix=model_filename_prefix,
                save_replay_buffer=False, # Don\'t save replay buffer for quick test
                save_vecnormalize=False,
            )

            logger.info(f"Starting agent learning for {self.test_config['total_timesteps']} timesteps.")
            sb3_agent.learn(
                total_timesteps=self.test_config["total_timesteps"],
                callback=checkpoint_callback,
                log_interval=5 # Log more frequently for tests
            )

            logger.info("Agent learning finished.")

            # --- Verify EnhancedTransformer Weight Update ---
            final_transformer_weights = self.get_model_parameters(transformer_component)
            transformer_weights_changed = False
            for name in initial_transformer_weights:
                self.assertIn(name, final_transformer_weights, f"Transformer parameter {name} not found in final weights.")
                if not torch.equal(initial_transformer_weights[name], final_transformer_weights[name]):
                    transformer_weights_changed = True
                    logger.info(f"Transformer weight changed for parameter: {name}")
                    break
            self.assertTrue(transformer_weights_changed, "EnhancedTransformer weights did not change after training.")

            # --- Verify Quantum Strategy Layer Weight Update ---
            final_quantum_layer_weights = {}
            if quantum_layer_component.attention_network is not None and \
               len(list(quantum_layer_component.attention_network.parameters())) > 0:
                final_quantum_layer_weights.update(self.get_model_parameters(quantum_layer_component.attention_network))

            for i, strategy_module in enumerate(quantum_layer_component.strategies):
                if hasattr(strategy_module, 'parameters') and len(list(strategy_module.parameters())) > 0:
                    final_quantum_layer_weights.update(
                        {f"strategy_{i}_{name}": param.clone().detach() for name, param in strategy_module.named_parameters()}
                    )
            
            quantum_weights_changed = False
            for name in initial_quantum_layer_weights:
                if name in final_quantum_layer_weights: # Ensure key exists if a strategy was dynamically removed (not expected here)
                    if not torch.equal(initial_quantum_layer_weights[name], final_quantum_layer_weights[name]):
                        quantum_weights_changed = True
                        logger.info(f"Quantum Layer weight changed for parameter: {name}")
                        break
                else:
                    logger.warning(f"Initial quantum weight \'{name}\' not found in final weights. This might indicate dynamic changes.")
            self.assertTrue(quantum_weights_changed, "Quantum Strategy Layer weights (attention or strategies) did not change after training.")
            
            # Check adaptive_bias_weights update (if they are meant to be updated by SAC directly or via a callback)
            # For this test, we'll assume they might be updated. If not, this check might fail or need adjustment.
            if initial_adaptive_bias_weights is not None and quantum_layer_component.adaptive_bias_weights is not None:
                final_adaptive_bias_weights = quantum_layer_component.adaptive_bias_weights.clone().detach()
                logger.info(f"Final adaptive_bias_weights: {final_adaptive_bias_weights}")
                self.assertFalse(torch.equal(initial_adaptive_bias_weights, final_adaptive_bias_weights),
                                 "Quantum Layer adaptive_bias_weights did not change after training.")
            elif initial_adaptive_bias_weights is None and quantum_layer_component.adaptive_bias_weights is not None:
                logger.info("Adaptive bias weights were initialized during training. This is a valid change.")
            elif initial_adaptive_bias_weights is not None and quantum_layer_component.adaptive_bias_weights is None:
                self.fail("Adaptive bias weights were present initially but are None after training.")
            # If both are None, no assertion needed.

            logger.info("Weight update verification successful.")

        except Exception as e:
            logger.error(f"Exception during test_training_run_and_weight_update: {e}", exc_info=True)
            self.fail(f"Test failed due to exception: {e}")

if __name__ == "__main__":
    # This allows running the test directly, e.g., python test_full_training_pipeline.py
    # For integration with a test runner (like pytest or unittest discovery),
    # the runner would typically discover and execute the TestFullTrainingPipeline class.
    if not TRAINING_COMPONENTS_AVAILABLE:
        print("Skipping test execution: Training components not available.")
        sys.exit(0)
        
    unittest.main()
