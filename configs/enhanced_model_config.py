# Configuration for the Enhanced Transformer Model

ModelConfig = {
    'hidden_dim': 256,
    'num_layers': 6,
    'num_heads': 8,
    'intermediate_dim': 1024,
    'dropout_rate': 0.1,
    'max_sequence_length': 128, # This can also be reviewed if sequences are shorter
    'num_symbols': None,
    'output_dim': 128,
    'use_adaptive_attention': True,
    'num_market_states': 4,
    'use_gmm_market_state_detector': False,
    'gmm_market_state_detector_path': None,
    'gmm_ohlcv_feature_config': False,
    'use_fourier_features': False,
    'use_wavelet_features': False,
    'positional_encoding_type': 'sinusoidal',
    'output_activation': None,
    'use_symbol_embedding': True,
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
    'use_multi_resolution_input': True,
    'multi_resolution_params': {
        'resolutions': ['S5', 'M1', 'M15'],
        'fusion_method': 'attention_weighted_sum',
        's5_weight': 0.5,
        'm1_weight': 0.3,
        'm15_weight': 0.2
    },
    'use_dynamic_feature_weighting': True,
    'dynamic_weighting_params': {
        'weighting_type': 'attention',
        'attention_dim': 64
    },
    'use_contextualized_embeddings': True,
    'contextualized_embedding_params': {
        'embedding_source': 'bert_like',
        'bert_model_name': 'bert-base-uncased',
        'embedding_dim_override': 128
    },
    'use_hierarchical_structure': True,
    'hierarchical_params': {
        'num_levels': 2,
        'level_fusion_method': 'recursive_attention'
    },
    'use_regularization': True,
    'regularization_params': {
        'type': 'l2_norm_clipping',
        'l2_lambda': 0.0001,
        'norm_clip_value': 1.0,
        'spectral_norm_iterations': 1
    },
    'use_auxiliary_losses': True,
    'auxiliary_loss_params': {
        'market_state_prediction_weight': 0.1,
        'volatility_prediction_weight': 0.05,
        'reconstruction_weight': 0.15
    },
    'use_explainability_features': True,
    'explainability_params': {
        'integrated_gradients_steps': 20,
        'shap_approximation_samples': 50
    }
}

# You can add other model-related configurations here if needed.
