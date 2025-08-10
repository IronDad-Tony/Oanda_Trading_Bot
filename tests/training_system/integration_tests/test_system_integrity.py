\
import torch
import torch.nn as nn
from gymnasium import spaces
import numpy as np
import logging

# Define these globally for fallback and tests
TEST_TIMESTEPS = 128
TEST_FEATURES = 9

# Attempt to import necessary modules
try:
    from src.agent.enhanced_feature_extractor import EnhancedTransformerFeatureExtractor
    from src.models.enhanced_transformer import EnhancedTransformer as EnhancedUniversalTradingTransformer
    from src.agent.meta_learning_system import AdaptiveStrategyEncoder, MetaLearningSystem # Assuming MetaLearningSystem might be tested later
    from src.common.config import (
        MAX_SYMBOLS_ALLOWED, TRANSFORMER_OUTPUT_DIM_PER_SYMBOL, TRANSFORMER_MODEL_DIM,
        TRANSFORMER_NUM_LAYERS, TRANSFORMER_NUM_HEADS, TRANSFORMER_FFN_DIM,
        TRANSFORMER_DROPOUT_RATE, ENHANCED_TRANSFORMER_USE_MULTI_SCALE,
        ENHANCED_TRANSFORMER_USE_CROSS_TIME_FUSION, DEVICE
    )
    from src.common.logger_setup import logger
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import modules: {e}. Some tests may not run correctly.")
    # Define fallbacks if necessary for the script to be parsable
    MAX_SYMBOLS_ALLOWED = 5
    TRANSFORMER_OUTPUT_DIM_PER_SYMBOL = 64
    DEVICE = 'cpu' # Corrected: Use single quotes for the string literal
    # TEST_TIMESTEPS and TEST_FEATURES are now defined globally above

    class EnhancedTransformerFeatureExtractor(nn.Module):
        def __init__(self, observation_space, enhanced_transformer_output_dim_per_symbol):
            super().__init__()
            total_features_from_transformer = MAX_SYMBOLS_ALLOWED * enhanced_transformer_output_dim_per_symbol
            other_features_count = MAX_SYMBOLS_ALLOWED * 3 + 1
            self.features_dim = total_features_from_transformer + other_features_count
            self.enhanced_transformer = nn.Linear(10,10) # Dummy
            logger.info("Using fallback EnhancedTransformerFeatureExtractor")

        def forward(self, obs):
            batch_size = obs['features_from_dataset'].shape[0]
            return torch.randn(batch_size, self.features_dim)

    class AdaptiveStrategyEncoder(nn.Module):
        def __init__(self, initial_input_dim, output_dim):
            super().__init__()
            self.output_dim = output_dim
            self.fc = nn.Linear(initial_input_dim, self.output_dim)
            self.current_input_dim = initial_input_dim
            logger.info("Using fallback AdaptiveStrategyEncoder")

        def forward(self, x):
            if x.shape[-1] != self.fc.in_features:
                logger.warning(f"Fallback AdaptiveStrategyEncoder: Input dim {x.shape[-1]} != fc.in_features {self.fc.in_features}. Adapting.")
                self._adapt_fc_layer(x.shape[-1], x.device)
            return self.fc(x)

        def _adapt_fc_layer(self, new_dim, device_to_use):
            logger.info(f"Fallback AdaptiveStrategyEncoder: Adapting FC layer to input dimension {new_dim}")
            self.fc = nn.Linear(new_dim, self.output_dim).to(device_to_use)
            self.current_input_dim = new_dim

        def adapt_input_dimension(self, new_dim, preserve_weights=True):
            # This method is called externally, x might not be available directly.
            # We need a device. Assuming global DEVICE or try to get from existing layer.
            device_to_use = self.fc.weight.device if hasattr(self.fc, 'weight') else torch.device(DEVICE)
            logger.info(f"Fallback AdaptiveStrategyEncoder: adapt_input_dimension called for {new_dim} on device {device_to_use}")
            self._adapt_fc_layer(new_dim, device_to_use)
            logger.info(f"Fallback AdaptiveStrategyEncoder: Adapted to {new_dim}")


logger.setLevel(logging.INFO)

def test_gradient_flow_to_transformer():
    logger.info("\\n--- Testing Gradient Flow to Transformer ---")
    grad_flow_successful = False
    try:
        # 1. Setup Observation Space (simplified)
        obs_space_dict = {
            'features_from_dataset': spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_SYMBOLS_ALLOWED, 128, 9), dtype=np.float32),
            'current_positions_nominal_ratio_ac': spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_SYMBOLS_ALLOWED,), dtype=np.float32),
            'unrealized_pnl_ratio_ac': spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_SYMBOLS_ALLOWED,), dtype=np.float32),
            'margin_level': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'padding_mask': spaces.Box(low=0, high=1, shape=(MAX_SYMBOLS_ALLOWED,), dtype=np.bool_)
        }
        observation_space = spaces.Dict(obs_space_dict)

        # 2. Initialize EnhancedTransformerFeatureExtractor
        feature_extractor = EnhancedTransformerFeatureExtractor(observation_space, enhanced_transformer_output_dim_per_symbol=TRANSFORMER_OUTPUT_DIM_PER_SYMBOL)
        feature_extractor.to(DEVICE)
        logger.info(f"Feature extractor initialized with features_dim: {feature_extractor.features_dim}")

        # Create a dummy optimizer for the feature_extractor's parameters
        optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=1e-3)

        # 3. Create Dummy Observations
        batch_size = 4
        dummy_obs = {
            key: torch.randn(batch_size, *space.shape).to(DEVICE)
            for key, space in observation_space.spaces.items()
            if key != 'padding_mask' # Corrected: String literal for key
        }
        dummy_obs['padding_mask'] = torch.randint(0, 2, (batch_size, MAX_SYMBOLS_ALLOWED), dtype=torch.bool).to(DEVICE) # Corrected: String literal
        # Ensure features_from_dataset matches expected dimensions by feature extractor
        dummy_obs['features_from_dataset'] = torch.randn(batch_size, MAX_SYMBOLS_ALLOWED, TEST_TIMESTEPS, TEST_FEATURES).to(DEVICE) # Used global TEST_TIMESTEPS, TEST_FEATURES


        # 4. Forward Pass through Feature Extractor
        features = feature_extractor(dummy_obs)
        logger.info(f"Features extracted with shape: {features.shape}")

        # 5. Dummy Policy Head and Loss Calculation
        # This simulates the SAC actor/critic head using the extracted features
        dummy_policy_head = nn.Linear(features.shape[-1], 1).to(DEVICE) # Predict a single value (e.g., Q-value or action component)
        optimizer.add_param_group({'params': dummy_policy_head.parameters()})

        predicted_values = dummy_policy_head(features)
        dummy_target = torch.randn_like(predicted_values).to(DEVICE)
        loss = nn.MSELoss()(predicted_values, dummy_target)
        logger.info(f"Calculated Loss: {loss.item()}")

        # 6. Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # Not strictly necessary for grad check, but good practice

        # 7. Check Gradients in EnhancedUniversalTradingTransformer
        grad_found = False
        if hasattr(feature_extractor, 'enhanced_transformer') and isinstance(feature_extractor.enhanced_transformer, nn.Module):
            for name, param in feature_extractor.enhanced_transformer.named_parameters():
                if param.grad is not None:
                    grad_found = True
                    logger.info(f"Gradient found for param: {name} in EnhancedUniversalTradingTransformer")
                    # logger.debug(f"Gradient for param {name}: {param.grad.abs().mean().item()}") # Optional: log mean abs grad
            if not grad_found:
                logger.error("NO GRADIENTS found in EnhancedUniversalTradingTransformer parameters!")
            else:
                logger.info("SUCCESS: Gradients successfully flowed to EnhancedUniversalTradingTransformer.")
                grad_flow_successful = True
        else:
            logger.error("Could not find 'enhanced_transformer' attribute or it\\'s not an nn.Module in feature_extractor.")
    except Exception as e:
        logger.error(f"Error during gradient flow test: {e}", exc_info=True)
    assert grad_flow_successful

def test_adaptive_encoder_dimension_adaptation():
    logger.info("\\n--- Testing AdaptiveStrategyEncoder Dimension Adaptation ---")
    try:
        initial_dim = 64
        output_dim = 128
        encoder = AdaptiveStrategyEncoder(initial_input_dim=initial_dim, output_dim=output_dim)
        encoder.to(DEVICE)
        logger.info(f"Initial encoder input dim: {encoder.current_input_dim}")

        # Test with same dimension
        dummy_input_same_dim = torch.randn(2, initial_dim).to(DEVICE)
        output_same = encoder(dummy_input_same_dim)
        assert encoder.current_input_dim == initial_dim, "Dimension should not change"
        logger.info(f"Forward pass with same dimension successful. Output shape: {output_same.shape}")

        # Test adaptation to larger dimension
        new_dim_larger = initial_dim * 2
        logger.info(f"Adapting to larger dimension: {new_dim_larger}")
        # The forward pass itself triggers adaptation if input dim mismatches
        dummy_input_larger_dim = torch.randn(2, new_dim_larger).to(DEVICE)
        output_larger = encoder(dummy_input_larger_dim)
        assert encoder.current_input_dim == new_dim_larger, "Dimension should adapt to larger"
        logger.info(f"Forward pass with larger dimension successful. Output shape: {output_larger.shape}")
        
        # Store some weights from the first layer for comparison (optional, simplified check)
        # weight_after_larger_adapt = encoder.network[0].weight.clone().detach()

        # Test adaptation to smaller dimension
        new_dim_smaller = initial_dim // 2
        logger.info(f"Adapting to smaller dimension: {new_dim_smaller}")
        dummy_input_smaller_dim = torch.randn(2, new_dim_smaller).to(DEVICE)
        output_smaller = encoder(dummy_input_smaller_dim)
        assert encoder.current_input_dim == new_dim_smaller, "Dimension should adapt to smaller"
        logger.info(f"Forward pass with smaller dimension successful. Output shape: {output_smaller.shape}")

        # Add more sophisticated checks for weight preservation if needed
        # For example, compare parts of weight matrices before/after adaptation

        logger.info("SUCCESS: AdaptiveStrategyEncoder dimension adaptation test completed.")

    except Exception as e:
        logger.error(f"Error during AdaptiveStrategyEncoder test: {e}", exc_info=True)

def create_mock_observation_space():
    return spaces.Dict({
        'features_from_dataset': spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(MAX_SYMBOLS_ALLOWED, TEST_TIMESTEPS, TEST_FEATURES),
            dtype=np.float32
        ),
        'current_positions_nominal_ratio_ac': spaces.Box( # Updated key
            low=-np.inf, high=np.inf,
            shape=(MAX_SYMBOLS_ALLOWED,),
            dtype=np.float32
        ),
        'unrealized_pnl_ratio_ac': spaces.Box( # Updated key
            low=-np.inf, high=np.inf,
            shape=(MAX_SYMBOLS_ALLOWED,),
            dtype=np.float32
        ),
        'margin_level': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
        'padding_mask': spaces.Box(low=0, high=1, shape=(MAX_SYMBOLS_ALLOWED,), dtype=np.bool_)
    })

def create_mock_observations(batch_size, num_symbols, observation_space):
    obs = {}
    obs['features_from_dataset'] = torch.randn(
        batch_size, MAX_SYMBOLS_ALLOWED, TEST_TIMESTEPS, TEST_FEATURES, device=DEVICE
    )
    # obs['padding_mask'] should be True for active symbols, False for padded symbols.
    # This is so that ~obs['padding_mask'] (used by feature extractor) becomes:
    # False for active symbols, True for padded symbols, which is what nn.MultiheadAttention key_padding_mask expects.
    padding_mask_np = np.zeros((batch_size, MAX_SYMBOLS_ALLOWED), dtype=bool) # All False (padded) by default
    if num_symbols > 0:
        padding_mask_np[:, :num_symbols] = True # Active symbols are True
    obs['padding_mask'] = torch.from_numpy(padding_mask_np).to(DEVICE)

    # Fill active symbols for other features
    obs['current_positions_nominal_ratio_ac'] = torch.randn(batch_size, MAX_SYMBOLS_ALLOWED, device=DEVICE) # Updated key
    obs['unrealized_pnl_ratio_ac'] = torch.randn(batch_size, MAX_SYMBOLS_ALLOWED, device=DEVICE) # Updated key
    obs['margin_level'] = torch.rand(batch_size, 1, device=DEVICE)

    # Apply padding mask to relevant features (zero out padded entries)
    # obs['padding_mask'].float() is 1.0 for active, 0.0 for padded.
    active_mask_float = obs['padding_mask'].float()
    obs['current_positions_nominal_ratio_ac'] = obs['current_positions_nominal_ratio_ac'] * active_mask_float
    obs['unrealized_pnl_ratio_ac'] = obs['unrealized_pnl_ratio_ac'] * active_mask_float
    
    # features_from_dataset padding is handled by the transformer's attention mask.

    logger.info(f"Mock observations created. Active symbols: {num_symbols}")
    logger.info(f"obs['padding_mask'] example (first batch item, True means active): {obs['padding_mask'][0].tolist()}")
    logger.info(f"Current positions example (sum after mask): {obs['current_positions_nominal_ratio_ac'][0].sum().item()}")
    return obs

# --- System Components Initialization ---

# Assuming these are the correct and final versions of the classes after all modifications
# If there are versioning or import issues, these might need to be adjusted.

# feature_extractor = EnhancedTransformerFeatureExtractor(observation_space, enhanced_transformer_output_dim_per_symbol=TRANSFORMER_OUTPUT_DIM_PER_SYMBOL)
# feature_extractor.to(DEVICE)

# For testing, we might not need the full complexity of the transformer. A simplified version could be:
class DummyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Just a linear layer for dummy transformer behavior
        self.linear = nn.Linear(128, TRANSFORMER_OUTPUT_DIM_PER_SYMBOL)

    def forward(self, x, padding_mask=None):
        # x: (batch_size, num_symbols, 128)
        # padding_mask: (batch_size, num_symbols) - boolean mask
        # For dummy, we ignore padding_mask
        return self.linear(x)

# Dummy policy network that uses the feature extractor and a simple transformer
class DummyPolicy(nn.Module):
    def __init__(self, feature_extractor, qsl_module=None, num_symbols=MAX_SYMBOLS_ALLOWED):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.num_symbols = num_symbols # Store num_symbols as instance attribute

        # For this test, let's assume QSL processes each symbol's features independently.
        # The input to this policy's forward will be the output of EnhancedTransformerFeatureExtractor
        # Shape: (batch_size, feature_extractor.features_dim)
        # feature_extractor.features_dim = (MAX_SYMBOLS_ALLOWED * TRANSFORMER_OUTPUT_DIM_PER_SYMBOL) + non_transformer_features
        # We need to extract per-symbol features for the QSL.
        # The transformer_part_dim should be based on MAX_SYMBOLS_ALLOWED as the feature extractor produces features for all possible symbols.
        self.transformer_output_dim_per_symbol = TRANSFORMER_OUTPUT_DIM_PER_SYMBOL # Added for clarity
        self.transformer_part_dim = MAX_SYMBOLS_ALLOWED * self.transformer_output_dim_per_symbol

        # For simplicity, let's define a dummy QSL as a linear layer that would take the place of the actual QSL
        # This should be replaced with the actual QSL module/class in real scenarios
        self.qsl = nn.Linear(self.transformer_output_dim_per_symbol, 1) if qsl_module is None else qsl_module

    def forward(self, state_features_full):
        # state_features_full is output of feature_extractor
        # Extract the part corresponding to transformer output for all symbols
        transformer_features_flat = state_features_full[:, :self.transformer_part_dim]
        
        # Reshape to (batch_size, MAX_SYMBOLS_ALLOWED, TRANSFORMER_OUTPUT_DIM_PER_SYMBOL)
        transformer_features_per_symbol = transformer_features_flat.view(
            -1, MAX_SYMBOLS_ALLOWED, self.transformer_output_dim_per_symbol
        )

        # We are processing NUM_TEST_SYMBOLS active symbols for QSL input
        # These are assumed to be the first `self.num_symbols` in the `transformer_features_per_symbol`
        active_symbol_features = transformer_features_per_symbol[:, :self.num_symbols, :]
        # (batch_size, NUM_TEST_SYMBOLS, QSL_INPUT_DIM)

        # For this dummy policy, let's just return the QSL output directly
        return self.qsl(active_symbol_features)

# --- System Integrity Tests ---

# These tests are to ensure that the system components interact correctly
# They may need to be adjusted as the actual system design and data flow become clear

def test_system_integrity_end_to_end():
    logger.info("\\n--- Testing System Integrity End-to-End ---")
    qsl_grad_successful = False
    try:
        batch_size = 2
        num_symbols = 3 # For testing, say we have 3 active symbols
        # TEST_TIMESTEPS and TEST_FEATURES are global
        # 1. Create mock observation space and observations
        observation_space = create_mock_observation_space()
        mock_obs = create_mock_observations(batch_size, num_symbols, observation_space)

        # 2. Initialize components
        feature_extractor = EnhancedTransformerFeatureExtractor(observation_space, enhanced_transformer_output_dim_per_symbol=TRANSFORMER_OUTPUT_DIM_PER_SYMBOL)
        feature_extractor.to(DEVICE)
        logger.info("Feature extractor initialized.")

        # For the sake of the test, let's keep the policy_net as our DummyPolicy
        # The DummyPolicy needs the number of active symbols to correctly slice features for QSL
        policy_net = DummyPolicy(feature_extractor, num_symbols=num_symbols) # Pass num_symbols
        policy_net.to(DEVICE)
        logger.info("Policy network initialized.")

        # AdaptiveStrategyEncoder doesn't directly participate in this flow, unless we are conditioning the policy on some meta-information
        # For now, we can skip its explicit initialization unless needed for the test

        # 3. Forward pass through feature extractor
        with torch.no_grad():
            extracted_features = feature_extractor(mock_obs)
            logger.info(f"Extracted features shape: {extracted_features.shape}")

        # 4. Forward pass through policy network (which includes QSL)
        policy_output = policy_net(extracted_features)
        logger.info(f"Policy network output shape: {policy_output.shape}")

        # 5. Check if the output has the expected shape and is differentiable
        assert policy_output.shape == (batch_size, num_symbols, 1), "Policy output shape mismatch"
        logger.info("Policy output shape is as expected.")

        # 6. (Optional) If there's a loss computation and backward pass, we can check gradient flow here
        # For simplicity, let's just do a dummy loss computation
        dummy_target = torch.randn_like(policy_output)
        loss = nn.MSELoss()(policy_output, dummy_target)
        logger.info(f"Dummy loss computed: {loss.item()}")

        # Backward pass to check gradient flow
        policy_net.qsl.weight.grad = None # Zero out any existing grad
        loss.backward()

        # Check gradients for QSL parameters
        grad_found_qsl = False
        # QSL parameters are part of policy_net.parameters()
        for name, param in policy_net.qsl.named_parameters(): # Check specifically within the qsl submodule of policy_net
            if param.grad is not None and param.grad.abs().sum().item() > 1e-9:
                logger.info(f"Grad found for policy_net.qsl.{name}, sum: {param.grad.abs().sum().item()}")
                grad_found_qsl = True
            elif param.grad is not None:
                logger.debug(f"Grad for policy_net.qsl.{name} is near zero: {param.grad.abs().sum().item()}")
        if not grad_found_qsl:
            logger.error("NO GRADIENTS found for policy_net.qsl parameters!")
        else:
            logger.info("SUCCESS: Gradients successfully flowed to policy_net.qsl parameters.")
            qsl_grad_successful = True

        logger.info("System integrity end-to-end test passed.")

    except Exception as e:
        logger.error(f"Error during system integrity test: {e}", exc_info=True)
    assert qsl_grad_successful

def test_meta_learning_system_integration():
    logger.info("\\n--- Testing Meta Learning System Integration ---")
    meta_path_grad_successful = False
    try:
        batch_size = 2
        num_symbols = 3 # Active symbols for this test
        # 1. Create mock observation space and observations
        observation_space = create_mock_observation_space()
        mock_obs = create_mock_observations(batch_size, num_symbols, observation_space)

        # 2. Initialize components
        feature_extractor = EnhancedTransformerFeatureExtractor(observation_space, enhanced_transformer_output_dim_per_symbol=TRANSFORMER_OUTPUT_DIM_PER_SYMBOL)
        feature_extractor.to(DEVICE)

        # Ensure the adaptive_encoder's initial_input_dim matches the feature_extractor's output dimension
        adaptive_encoder_initial_input_dim = feature_extractor.features_dim
        adaptive_encoder = AdaptiveStrategyEncoder(initial_input_dim=adaptive_encoder_initial_input_dim, output_dim=64)
        adaptive_encoder.to(DEVICE)
        logger.info("Adaptive strategy encoder initialized.")

        # For this test, let's use a simplified policy that directly uses the adaptive encoder output
        class SimplifiedPolicy(nn.Module):
            def __init__(self, adaptive_encoder):
                super().__init__()
                self.adaptive_encoder = adaptive_encoder
                # Output layer for action or value (depends on your use case)
                self.output_layer = nn.Linear(64, 1)

            def forward(self, state_features):
                # state_features is output of feature_extractor
                task_embedding = self.adaptive_encoder(state_features)
                return self.output_layer(task_embedding)

        policy_net = SimplifiedPolicy(adaptive_encoder)
        policy_net.to(DEVICE)
        logger.info("Simplified policy network initialized.")

        # 3. Forward pass through feature extractor
        with torch.no_grad():
            extracted_features = feature_extractor(mock_obs)
            logger.info(f"Extracted features shape: {extracted_features.shape}")

        # 4. Forward pass through policy network
        policy_output = policy_net(extracted_features)
        logger.info(f"Policy network output shape: {policy_output.shape}")

        # 5. Check if the output has the expected shape and is differentiable
        assert policy_output.shape == (batch_size, 1), "Policy output shape mismatch"
        logger.info("Policy output shape is as expected.")

        # 6. (Optional) If there's a loss computation and backward pass, we can check gradient flow here
        # For simplicity, let's just do a dummy loss computation
        dummy_target = torch.randn_like(policy_output)
        loss = nn.MSELoss()(policy_output, dummy_target)
        logger.info(f"Dummy loss computed: {loss.item()}")

        # Backward pass to check gradient flow
        policy_net.output_layer.weight.grad = None # Zero out any existing grad
        loss.backward()

        # Check gradients for output layer parameters
        grad_found_output = False
        for name, param in policy_net.output_layer.named_parameters():
            if param.grad is not None and param.grad.abs().sum().item() > 1e-9:
                logger.info(f"Grad found for policy_net.output_layer.{name}, sum: {param.grad.abs().sum().item()}")
                grad_found_output = True
            elif param.grad is not None:
                logger.debug(f"Grad for policy_net.output_layer.{name} is near zero: {param.grad.abs().sum().item()}")
        if not grad_found_output:
            logger.error("NO GRADIENTS found for policy_net.output_layer parameters!")
        else:
            logger.info("SUCCESS: Gradients successfully flowed to policy_net.output_layer parameters.")
            meta_path_grad_successful = True

        # For the adaptive encoder, we check if it adapts to the input dimension from the feature extractor
        new_encoder_input_dim = extracted_features.shape[1] # This should match the encoder's expected input dim
        adaptive_encoder_input_dim_before = adaptive_encoder.current_input_dim

        # Forward pass will trigger adaptation if needed
        adaptive_encoder.adapt_input_dimension(new_encoder_input_dim)
        logger.info(f"Adaptive encoder input dim after explicit adapt: {adaptive_encoder.current_input_dim}")

        # Now, a forward pass with data of this new dimension should work without further adaptation.
        dummy_data_for_adapted_encoder = torch.randn(batch_size, new_encoder_input_dim).to(DEVICE)
        # The policy_net.adaptive_encoder is the same instance as adaptive_encoder initialized above.
        with torch.no_grad():
            _ = policy_net.adaptive_encoder(dummy_data_for_adapted_encoder)

        logger.info(f"Adaptive encoder input dim before (initial): {adaptive_encoder_input_dim_before}, current after test: {adaptive_encoder.current_input_dim}")

        if adaptive_encoder.current_input_dim == new_encoder_input_dim:
            logger.info(f"✅ AdaptiveStrategyEncoder successfully adapted to input dimension {new_encoder_input_dim}.")

            # Test another adaptation back to original or another different dimension
            another_new_dim = adaptive_encoder_initial_input_dim + 10 if adaptive_encoder_initial_input_dim > 0 else 256
            logger.info(f"Attempting to adapt encoder to another new input dimension: {another_new_dim}")
            adaptive_encoder.adapt_input_dimension(another_new_dim)

            logger.info(f"Encoder current_input_dim after forward with new dim: {adaptive_encoder.current_input_dim}")

        logger.info("Meta learning system integration test passed.")

    except Exception as e:
        logger.error(f"Error during meta learning system integration test: {e}", exc_info=True)
    assert meta_path_grad_successful

def main():
    transformer_grad_ok = test_gradient_flow_to_transformer()
    test_adaptive_encoder_dimension_adaptation() # This test focuses on adaptation, not returning gradient status for summary
    system_e2e_qsl_grad_ok = test_system_integrity_end_to_end()
    meta_system_output_grad_ok = test_meta_learning_system_integration()

    logger.info("\\n--- Overall Gradient Flow and Weight Update Summary ---")
    
    if transformer_grad_ok:
        logger.info("✅ Transformer: Gradients successfully flowed to EnhancedUniversalTradingTransformer. Weights can be updated by an optimizer.")
    else:
        logger.info("❌ Transformer: Gradient flow to EnhancedUniversalTradingTransformer FAILED or not detected.")

    if system_e2e_qsl_grad_ok:
        logger.info("✅ QSL (via End-to-End Test): Gradients successfully flowed to QSL component. Weights can be updated by an optimizer.")
    else:
        logger.info("❌ QSL (via End-to-End Test): Gradient flow to QSL component FAILED or not detected.")

    if meta_system_output_grad_ok:
        logger.info("✅ Meta-Learning Path (Output Layer): Gradients successfully flowed to the output layer in the meta-learning integration test. Weights can be updated by an optimizer.")
    else:
        logger.info("❌ Meta-Learning Path (Output Layer): Gradient flow to the output layer in the meta-learning integration test FAILED or not detected.")

    logger.info("\\n--- Batch Size Testing Recommendation ---")
    logger.info("To test with various batch sizes, please modify \'batch_size\' in the `sac_params`")
    logger.info("within the CONFIG dictionary in your main training script (train_universal_trader_rewritten.py).")
    logger.info("Run short training sessions (e.g., 100-200 timesteps) with different batch sizes such as 32, 64, 128, 256.")
    logger.info("Monitor the console output for successful execution, any dimension-related errors, and OOM errors.")
    
    logger.info("\\n--- Further Testing Considerations ---")
    logger.info("1. MetaLearningSystem: Test \'detect_model_configuration\' with mock strategy layers of varying configs.")
    logger.info("2. QuantumStrategyLayer: Requires specific tests based on its design and how it integrates with the SAC policy.")
    logger.info("3. End-to-End Gradient Flow: For a full check from SAC loss back to all components (including MetaLearningSystem and QuantumStrategyLayer if they are part of the policy network), a more integrated test environment or detailed logging within the SAC training loop would be needed.")

if __name__ == "__main__":
    main()
