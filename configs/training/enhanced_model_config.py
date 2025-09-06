# Configuration for the Enhanced Transformer Model
# Adjusted import path to be an absolute path from the 'src' root
from oanda_trading_bot.training_system.common.config import MAX_SYMBOLS_ALLOWED

ModelConfig = {
    # Phase 1: safe-but-larger baseline to enable richer modules
    'use_enhanced_transformer': True,  # Use the EnhancedTransformer inside the features extractor

    # Core model size (ensure hidden_dim % num_heads == 0)
    'hidden_dim': 768,
    'num_layers': 12,
    'num_heads': 12,
    'intermediate_dim': 3072,
    'dropout_rate': 0.1,

    # Sequence/context
    'max_sequence_length': 512,
    'num_symbols': MAX_SYMBOLS_ALLOWED,
    'output_dim': 256,

    # Attention and structure
    'use_adaptive_attention': True,
    'use_cross_asset_attention': True,
    'num_cross_asset_layers': 2,

    # Market state and auxiliary
    'num_market_states': 4,
    'use_gmm_market_state_detector': False,
    'gmm_market_state_detector_path': None,
    'gmm_ohlcv_feature_config': False,

    # Spectral/multi-scale modules
    'use_fourier_features': True,
    'fourier_num_modes': 32,
    'use_wavelet_features': True,
    'wavelet_name': 'db4',
    'wavelet_levels': 3,
    'use_graph_attn': True,

    # Positional/symbol
    'positional_encoding_type': 'sinusoidal',
    'output_activation': None,
    'use_symbol_embedding': True,
    'symbol_embedding_dim': 32,

    # Multi-scale feature extractor & cross-time fusion
    'use_msfe': True,
    'msfe_hidden_dim': 256,
    'msfe_scales': [3, 5, 7, 11],
    'use_cts_fusion': True,
    'cts_time_scales': [1, 3, 5],
    'cts_fusion_type': 'hierarchical_attention',
    'advanced_fusion_params': {
        'context_vector_dim': 128,
        'attention_heads': 4,
        'dropout_rate': 0.1
    },

    # Optional multi-resolution inputs (kept on, not used by all paths)
    'use_multi_resolution_input': True,
    'multi_resolution_params': {
        'resolutions': ['S5', 'M1', 'M15'],
        'fusion_method': 'attention_weighted_sum',
        's5_weight': 0.5,
        'm1_weight': 0.3,
        'm15_weight': 0.2
    },

    # Dynamic feature weighting
    'use_dynamic_feature_weighting': True,
    'dynamic_weighting_params': {
        'weighting_type': 'attention',
        'attention_dim': 64
    },

    # Contextualized embeddings (kept for compatibility)
    'use_contextualized_embeddings': True,
    'contextualized_embedding_params': {
        'embedding_source': 'bert_like',
        'bert_model_name': 'bert-base-uncased',
        'embedding_dim_override': 128
    },

    # Hierarchical structure
    'use_hierarchical_structure': True,
    'hierarchical_params': {
        'num_levels': 2,
        'level_fusion_method': 'recursive_attention'
    },

    # Regularization
    'use_regularization': True,
    'regularization_params': {
        'type': 'l2_norm_clipping',
        'l2_lambda': 0.0001,
        'norm_clip_value': 1.0,
        'spectral_norm_iterations': 1
    },

    # Auxiliary losses
    'use_auxiliary_losses': True,
    'auxiliary_loss_params': {
        'market_state_prediction_weight': 0.1,
        'volatility_prediction_weight': 0.05,
        'reconstruction_weight': 0.15
    },

    # Explainability (kept on, cheap during forward)
    'use_explainability_features': True,
    'explainability_params': {
        'integrated_gradients_steps': 20,
        'shap_approximation_samples': 50
    }
}

# You can add other model-related configurations here if needed.
