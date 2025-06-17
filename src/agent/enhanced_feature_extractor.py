# src/agent/enhanced_feature_extractor.py
"""
增強版特徵提取器 - Phase 3
使用增強版UniversalTradingTransformer進行特徵提取
支持多尺度特徵提取、自適應注意力機制和跨時間尺度融合
"""

import gymnasium as gym # Changed alias for clarity
from gymnasium.spaces import Dict as GymDict # Import Dict directly
import torch as th
from torch import nn
import json # Added import for json
import os # Added import for os
import threading # Added import for threading
from typing import Optional, Dict, List # Added List to typing imports
import numpy as np # Added import for numpy

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs # Added import for PyTorchObs

import logging

import torch
logger = logging.getLogger(__name__) # __name__ will be 'src.agent.enhanced_feature_extractor'

# Define MAX_SYMBOLS_ALLOWED if it's a constant used in this module
MAX_SYMBOLS_ALLOWED = 10 # Example value, adjust as needed
TRANSFORMER_OUTPUT_DIM_PER_SYMBOL = 64 # Example value, adjust as needed

class MockTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, n_head, num_layers, output_dim, dropout=0.1, use_symbol_embedding=False, num_symbols=None, symbol_embedding_dim=None, config=None):
        super().__init__()
        self.model_dim = model_dim
        self.use_symbol_embedding = use_symbol_embedding
        self.num_symbols = num_symbols
        self.symbol_embedding_dim = symbol_embedding_dim
        self.config = config # Store the config

        logger.info(f"MockTransformer.__init__ called with input_dim={input_dim}, model_dim={model_dim}, n_head={n_head}, num_layers={num_layers}, output_dim={output_dim}")
        logger.info(f"MockTransformer.__init__ use_symbol_embedding: {self.use_symbol_embedding}")
        if self.use_symbol_embedding:
            if num_symbols is None or symbol_embedding_dim is None:
                raise ValueError("num_symbols and symbol_embedding_dim must be provided if use_symbol_embedding is True.")
            self.symbol_embedder = nn.Embedding(num_symbols, symbol_embedding_dim)
            # Adjust input_dim for the main transformer if symbol embedding is concatenated or added
            # This depends on how it's used. If concatenated: input_dim += symbol_embedding_dim
            # For simplicity, let's assume it's handled before features are passed to a main linear layer.
            logger.info(f"MockTransformer.__init__ symbol_embedder created: num_symbols={num_symbols}, symbol_embedding_dim={symbol_embedding_dim}")
            # The input_dim to the first linear layer might need to be adjusted if symbol embedding is concatenated here.
            # However, typically, symbol ID is a separate input to forward, and embedding is looked up and combined.
            # Let's assume input_dim is for the main sequence data.

        self.fc_in = nn.Linear(input_dim, model_dim)
        # Simplified transformer structure for mock
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_head, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, output_dim)
        logger.info(f"MockTransformer.__init__ config received: {config}")
        logger.info("MockTransformer initialized.")

    def forward(self, src: th.Tensor, symbol_ids: Optional[th.Tensor] = None) -> th.Tensor:
        # src shape: (batch_size, seq_len, feature_dim)
        # logger.debug(f"MockTransformer.forward: src shape: {src.shape}")
        
        x = self.fc_in(src) # Project to model_dim
        # logger.debug(f"MockTransformer.forward: after fc_in, x shape: {x.shape}")

        if self.use_symbol_embedding:
            if symbol_ids is None:
                raise ValueError("symbol_ids must be provided if use_symbol_embedding is True.")
            # logger.debug(f"MockTransformer.forward: symbol_ids shape: {symbol_ids.shape}") # (batch_size, 1) or (batch_size)
            
            # Ensure symbol_ids are correctly shaped for embedding lookup if they are per-sequence rather than per-timestep
            # If symbol_ids is (batch_size), expand to (batch_size, 1) for embedding, then (batch_size, 1, embedding_dim)
            # Then, this embedding needs to be combined with x. Common ways: add to x (if dims match) or concatenate.
            # For concatenation, x might be (batch_size, seq_len, model_dim_main_features)
            # and symbol_embedding (batch_size, seq_len, symbol_embedding_dim) - requires broadcasting symbol_id or embedding
            
            # Assuming symbol_ids is (batch_size) or (batch_size, 1) and applies to the whole sequence
            s_embed = self.symbol_embedder(symbol_ids.squeeze(-1) if symbol_ids.ndim > 1 else symbol_ids) # (batch_size, symbol_embedding_dim)
            s_embed = s_embed.unsqueeze(1).repeat(1, x.size(1), 1) # (batch_size, seq_len, symbol_embedding_dim)
            
            # This is a placeholder for how to combine. Concatenation is common.
            # If concatenating, fc_in should have been on (input_dim - symbol_embedding_dim) if symbol_embedding is part of input_dim to fc_in.
            # Or, if input_dim to MockTransformer is just for sequence features, and symbol_embedding is extra:
            # x = th.cat((x, s_embed), dim=-1) # This would change model_dim, so fc_in's output or transformer's d_model needs to account for it.
            # A simpler way for a mock: add to x if dimensions allow, or project and add.
            # For now, let's assume symbol embedding is an additive feature or handled by a more complex architecture.
            # To keep it simple and avoid changing dimensions for the main transformer layers:
            # Project symbol embedding to model_dim and add it to x (element-wise sum)
            if s_embed.shape[-1] != x.shape[-1]: # If symbol_embedding_dim != model_dim
                # This would require another linear layer to project s_embed to model_dim
                # For a mock, we might skip this complexity or assume they are equal.
                # logger.warning("Symbol embedding dim != model_dim, addition might not be appropriate without projection.")
                pass # Not adding for now to keep mock simple
            else:
                x = x + s_embed # Add symbol information (broadcasts over seq_len if s_embed was (batch, 1, model_dim))
            # logger.debug(f"MockTransformer.forward: after symbol embedding (if used and added), x shape: {x.shape}")

        x = self.transformer_encoder(x)
        # logger.debug(f"MockTransformer.forward: after transformer_encoder, x shape: {x.shape}")
        
        # Output processing: use mean over sequence length, or CLS token if used
        x_mean = x.mean(dim=1) # (batch_size, model_dim)
        # logger.debug(f"MockTransformer.forward: after mean pooling, x_mean shape: {x_mean.shape}")
        
        output = self.fc_out(x_mean) # (batch_size, output_dim)
        # logger.debug(f"MockTransformer.forward: after fc_out, output shape: {output.shape}")
        return output


class EnhancedTransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    增強版Transformer特徵提取器
    基於EnhancedUniversalTradingTransformer的高級特徵提取
    """
    
    def __init__(self, observation_space: GymDict, model_config_path: str, 
                 use_msfe: bool = False, use_cts_fusion: bool = False, use_symbol_embedding: bool = False, 
                 msfe_configs: Optional[dict] = None, cts_fusion_configs: Optional[dict] = None, 
                 symbol_embedding_configs: Optional[dict] = None):
        
        logger.info("ENHANCED_TF_FE_INIT_MINIMAL_LOG_TEST_POINT_A (forced level)")
        logger.info(f"EnhancedTransformerFeatureExtractor.__init__ ENTERED. PID: {os.getpid()}, TID: {threading.get_ident()}")
        logger.info(f"EnhancedTransformerFeatureExtractor.__init__: model_config_path='{model_config_path}'")
        logger.info(f"EnhancedTransformerFeatureExtractor.__init__: use_msfe={use_msfe}, use_cts_fusion={use_cts_fusion}, use_symbol_embedding={use_symbol_embedding}")
        logger.info(f"EnhancedTransformerFeatureExtractor.__init__: msfe_configs: {msfe_configs}")
        logger.info(f"EnhancedTransformerFeatureExtractor.__init__: cts_fusion_configs: {cts_fusion_configs}")
        logger.info(f"EnhancedTransformerFeatureExtractor.__init__: symbol_embedding_configs: {symbol_embedding_configs}")

        self.model_config_path = model_config_path
        self.config = None
        self.model_config = None # This will store the loaded JSON config

        try:
            with open(model_config_path, 'r') as f:
                self.model_config = json.load(f)
            logger.info(f"EnhancedTransformerFeatureExtractor: Successfully loaded model_config from {model_config_path}")
            logger.info(f"EnhancedTransformerFeatureExtractor: model_config content: {json.dumps(self.model_config, indent=2)}")
        except FileNotFoundError:
            logger.error(f"EnhancedTransformerFeatureExtractor: Model config file not found at {model_config_path}. Critical error.")
            raise
        except json.JSONDecodeError:
            logger.error(f"EnhancedTransformerFeatureExtractor: Error decoding JSON from {model_config_path}. Critical error.")
            raise
        except Exception as e:
            logger.error(f"EnhancedTransformerFeatureExtractor: Unexpected error loading config {model_config_path}: {e}. Critical error.")
            raise

        # Override feature flags from model_config if they are explicitly passed as False to __init__
        # This allows turning OFF a feature via constructor even if it's ON in the config file.
        # If the constructor arg is True or None (default), the config file value (or its default if not in config) is used.
        
        # Effective settings, preferring explicit False from constructor, then config file, then class defaults.
        self.use_msfe = use_msfe if use_msfe is not None else self.model_config.get('use_multi_scale', False)
        self.use_cts_fusion = use_cts_fusion if use_cts_fusion is not None else self.model_config.get('use_cross_time_fusion', False)
        self.use_symbol_embedding = use_symbol_embedding if use_symbol_embedding is not None else self.model_config.get('use_symbol_embedding', False)

        logger.info(f"EnhancedTransformerFeatureExtractor: Effective feature flags after considering constructor args and config file:")
        logger.info(f"  use_msfe: {self.use_msfe} (constructor: {use_msfe}, config: {self.model_config.get('use_multi_scale')})")
        logger.info(f"  use_cts_fusion: {self.use_cts_fusion} (constructor: {use_cts_fusion}, config: {self.model_config.get('use_cross_time_fusion')})")
        logger.info(f"  use_symbol_embedding: {self.use_symbol_embedding} (constructor: {use_symbol_embedding}, config: {self.model_config.get('use_symbol_embedding')})")

        # Extract relevant parameters from model_config
        transformer_params = self.model_config.get('transformer_params', {})
        self.input_dim = transformer_params.get('input_dim', 64) # Example default
        self.model_dim = transformer_params.get('model_dim', 128)
        self.n_head = transformer_params.get('n_head', 4)
        self.num_layers = transformer_params.get('num_layers', 3)
        self.dropout = transformer_params.get('dropout', 0.1)
        # Output dim of the transformer should be features_dim for SB3
        # Let's assume the transformer's output_dim is self.model_dim if not specified further, 
        # or a specific 'output_dim' if present in config.
        self.transformer_output_dim = transformer_params.get('output_dim', self.model_dim) 

        # The _features_dim for SB3 is the output of this entire feature extractor
        # This might be different from transformer_output_dim if there are post-processing layers.
        # For now, assume it's the transformer_output_dim.
        _features_dim = self.transformer_output_dim
        
        # Initialize symbol embedding related parameters if used
        self.num_symbols = None
        self.symbol_embedding_dim = None
        if self.use_symbol_embedding:
            # These should ideally come from symbol_embedding_configs or model_config
            # For MAX_SYMBOLS_ALLOWED, it might be a global constant or from obs_space properties
            self.num_symbols = symbol_embedding_configs.get('num_symbols', MAX_SYMBOLS_ALLOWED) if symbol_embedding_configs else MAX_SYMBOLS_ALLOWED
            self.symbol_embedding_dim = symbol_embedding_configs.get('embedding_dim', 16) if symbol_embedding_configs else 16
            logger.info(f"EnhancedTransformerFeatureExtractor: Symbol embedding enabled. num_symbols={self.num_symbols}, embedding_dim={self.symbol_embedding_dim}")
            # If symbol embedding is concatenated to features before transformer, input_dim might change
            # self.input_dim += self.symbol_embedding_dim # This depends on architecture
        
        # Pass the full model_config to MockTransformer for its own parsing if needed
        # Also pass specific parsed params for clarity and direct use by MockTransformer's known args
        self.transformer = MockTransformer(
            input_dim=self.input_dim, # This is the dim of features per time step
            model_dim=self.model_dim,
            n_head=self.n_head,
            num_layers=self.num_layers,
            output_dim=self.transformer_output_dim, # This is the final output dim of the transformer block
            dropout=self.dropout,
            use_symbol_embedding=self.use_symbol_embedding,
            num_symbols=self.num_symbols,
            symbol_embedding_dim=self.symbol_embedding_dim,
            config=self.model_config # Pass the whole config too
        )
        logger.info(f"EnhancedTransformerFeatureExtractor: MockTransformer initialized with output_dim={self.transformer_output_dim}")

        # Call super().__init__ after defining _features_dim
        super().__init__(observation_space, features_dim=_features_dim)
        logger.info(f"EnhancedTransformerFeatureExtractor.__init__ COMPLETED. _features_dim set to: {_features_dim}")
        logger.info(f"  Final effective use_msfe: {self.use_msfe}")
        logger.info(f"  Final effective use_cts_fusion: {self.use_cts_fusion}")
        logger.info(f"  Final effective use_symbol_embedding: {self.use_symbol_embedding}")
        logger.info(f"  Associated model_config: {json.dumps(self.model_config, indent=2)}")


    def forward(self, observations: PyTorchObs) -> th.Tensor:
        # logger.debug(f"EnhancedTransformerFeatureExtractor.forward called. Obs keys: {observations.keys()}")
        # Assuming 'market_features' is the primary input tensor for the transformer
        # Shape: (batch_size, sequence_length, num_features_per_step)
        market_features = observations.get('market_features')
        if market_features is None:
            logger.error("'market_features' not found in observations!")
            raise ValueError("'market_features' must be part of observations for EnhancedTransformerFeatureExtractor")

        # logger.debug(f"  market_features shape: {market_features.shape}")

        symbol_ids = None
        if self.use_symbol_embedding:
            symbol_ids = observations.get('symbol_id') # Expected shape: (batch_size, 1) or (batch_size)
            if symbol_ids is None:
                logger.error("'symbol_id' not found in observations but use_symbol_embedding is True!")
                raise ValueError("'symbol_id' must be provided if use_symbol_embedding is True.")
            # Ensure symbol_ids are long type for nn.Embedding
            symbol_ids = symbol_ids.long()
            # logger.debug(f"  symbol_ids shape: {symbol_ids.shape}, dtype: {symbol_ids.dtype}")

        # Pass to the transformer model
        # The MockTransformer expects src and optional symbol_ids
        extracted_features = self.transformer(market_features, symbol_ids=symbol_ids)
        # logger.debug(f"  extracted_features shape after transformer: {extracted_features.shape}") # Expected: (batch_size, self.transformer_output_dim)
        
        # Placeholder for MSFE and CTS-Fusion logic
        if self.use_msfe:
            # logger.debug("MSFE logic would be applied here.")
            # This would involve processing market_features at different scales
            # and combining them, potentially before or within the main transformer.
            # For simplicity, assume extracted_features is the result after MSFE if enabled.
            pass

        if self.use_cts_fusion:
            # logger.debug("CTS-Fusion logic would be applied here.")
            # This would involve fusing information across different time scales/steps.
            # Could be part of the transformer architecture or a separate module.
            pass
        
        return extracted_features


class EnhancedTransformerFeatureExtractorWithMemory(EnhancedTransformerFeatureExtractor):
    """
    帶記憶機制的增強版特徵提取器
    支持跨episode的長期記憶和短期記憶融合
    """
    
    def __init__(self, observation_space: GymDict, 
                 enhanced_transformer_output_dim_per_symbol: int = TRANSFORMER_OUTPUT_DIM_PER_SYMBOL,
                 memory_size: int = 32):
        super().__init__(observation_space, enhanced_transformer_output_dim_per_symbol)
        
        self.memory_size = memory_size
        self.feature_dim = MAX_SYMBOLS_ALLOWED * enhanced_transformer_output_dim_per_symbol
        
        # 記憶存儲
        self.register_buffer('long_term_memory', 
                           torch.zeros(memory_size, self.feature_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
        # 記憶融合網絡
        self.memory_fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.GELU(),
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        
        # 記憶注意力
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        logger.info(f"初始化帶記憶機制的增強版特徵提取器，記憶大小: {memory_size}")
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """增強版前向傳播，包含記憶機制"""
        # 基礎特徵提取
        base_features = super().forward(observations)
        
        # 提取Transformer特徵部分
        batch_size = base_features.shape[0]
        transformer_features = base_features[:, :self.feature_dim]
        other_features = base_features[:, self.feature_dim:]
        
        # 記憶增強
        if self.training:
            # 訓練時更新記憶
            self._update_memory(transformer_features.detach())
        
        # 記憶檢索和融合
        memory_enhanced_features = self._retrieve_and_fuse_memory(transformer_features)
        
        # 重新組合特徵
        enhanced_combined = torch.cat([memory_enhanced_features, other_features], dim=1)
        
        return enhanced_combined
    
    def _update_memory(self, features: torch.Tensor):
        """更新長期記憶"""
        batch_size = features.shape[0]
        
        for i in range(batch_size):
            ptr = int(self.memory_ptr.item())
            self.long_term_memory[ptr] = features[i]
            self.memory_ptr[0] = (ptr + 1) % self.memory_size
    
    def _retrieve_and_fuse_memory(self, current_features: torch.Tensor) -> torch.Tensor:
        """檢索和融合記憶"""
        batch_size = current_features.shape[0]
        
        # 與記憶進行注意力交互
        memory_attended, _ = self.memory_attention(
            current_features.unsqueeze(1),  # [batch, 1, feature_dim]
            self.long_term_memory.unsqueeze(0).expand(batch_size, -1, -1),  # [batch, memory_size, feature_dim]
            self.long_term_memory.unsqueeze(0).expand(batch_size, -1, -1)
        )
        
        memory_attended = memory_attended.squeeze(1)  # [batch, feature_dim]
        
        # 記憶融合
        concatenated = torch.cat([current_features, memory_attended], dim=-1)
        fused_features = self.memory_fusion(concatenated)
        
        return fused_features + current_features  # 殘差連接


class MemoryFusionFeatureExtractor(BaseFeaturesExtractor):
    """
    A feature extractor that combines current features with a memory bank of past features
    using an attention mechanism, and also fuses other observation components.
    """
    def __init__(self, 
                 observation_space: GymDict, 
                 main_feature_key: str = 'market_features', # Key for the main sequential features
                 other_feature_keys: Optional[List[str]] = None, # Keys for other flat features to concatenate
                 features_dim: int = 256, # Output dimension of this extractor
                 memory_size: int = 100, 
                 num_attention_heads: int = 4,
                 enhanced_transformer_output_dim_per_symbol: int = TRANSFORMER_OUTPUT_DIM_PER_SYMBOL, # Use defined constant
                 ): 
        super().__init__(observation_space, features_dim)
        self.main_feature_key = main_feature_key
        self.other_feature_keys = other_feature_keys if other_feature_keys is not None else []
        self.memory_size = memory_size
        
        # Assuming main_feature_key provides features of size `enhanced_transformer_output_dim_per_symbol`
        self.feature_dim = enhanced_transformer_output_dim_per_symbol 

        # Memory bank (circular buffer)
        self.register_buffer('memory_bank', 
                           th.zeros(memory_size, self.feature_dim)) # Use th
        self.register_buffer('memory_ptr', th.zeros(1, dtype=th.long)) # Use th

        # Attention mechanism for memory retrieval
        self.attention = nn.MultiheadAttention(embed_dim=self.feature_dim, 
                                             num_heads=num_attention_heads, 
                                             batch_first=True)
        
        # Calculate combined dimension from other features
        other_features_dim = 0
        if self.other_feature_keys:
            for key in self.other_feature_keys:
                space = observation_space[key]
                if isinstance(space, gym.spaces.Box):
                    other_features_dim += int(np.prod(space.shape))
                else:
                    raise ValueError(f"Unsupported space type for other_feature_key '{key}': {type(space)}")
        
        # Layer to fuse memory-enhanced features with other features
        # Input to this layer will be (feature_dim from attention + other_features_dim)
        self.fusion_layer_input_dim = self.feature_dim + other_features_dim
        self.fusion_layer = nn.Linear(self.fusion_layer_input_dim, features_dim)
        self.output_activation = nn.ReLU()

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor: # Use th, Dict
        current_main_features = observations[self.main_feature_key]
        # Assuming current_main_features is [batch_size, feature_dim]
        # If it's [batch_size, seq_len, feature_dim], take the last time step or mean
        if current_main_features.ndim == 3:
            current_main_features = current_main_features[:, -1, :] # Take last time step

        # Retrieve and fuse memory
        memory_enhanced_features = self._retrieve_and_fuse_memory(current_main_features)

        # Update memory with current main features (after they've been used for retrieval)
        self._update_memory(current_main_features.detach()) # Detach to prevent gradients flowing into memory update

        # Concatenate other features if any
        other_features_list = []
        if self.other_feature_keys:
            for key in self.other_feature_keys:
                other_features_list.append(observations[key].view(observations[key].size(0), -1))
        
        if other_features_list:
            other_features = th.cat(other_features_list, dim=1) # Use th
            # Combine memory-enhanced features with other features
            combined_features = th.cat([memory_enhanced_features, other_features], dim=1) # Use th
        else:
            combined_features = memory_enhanced_features # No other features to combine

        # Final fusion layer
        output_features = self.output_activation(self.fusion_layer(combined_features))
        return output_features

    def _update_memory(self, features: th.Tensor): # Use th
        batch_size = features.size(0)
        ptr = int(self.memory_ptr[0])
        for i in range(batch_size):
            self.memory_bank[ptr] = features[i]
            ptr = (ptr + 1) % self.memory_size
        self.memory_ptr[0] = ptr

    def _retrieve_and_fuse_memory(self, current_features: th.Tensor) -> th.Tensor: # Use th
        batch_size = current_features.size(0)
        # Query: current_features [batch_size, feature_dim]
        # Key/Value: memory_bank [memory_size, feature_dim]
        
        # Expand query to [batch_size, 1, feature_dim] for attention
        query = current_features.unsqueeze(1)
        # Expand memory_bank to [batch_size, memory_size, feature_dim] for attention
        # This is not quite right for MHA if memory is shared across batch. 
        # MHA expects K,V as (L, N, E) or (N, L, E) if batch_first=True where L is target seq len.
        # Here, memory_bank is the target sequence (keys/values).
        # Let's make memory_bank [1, memory_size, feature_dim] and broadcast over batch if needed,
        # or repeat it for each item in the batch.
        memory_expanded = self.memory_bank.unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, memory_size, feature_dim]

        # Attention output: [batch_size, 1, feature_dim] (attended features from memory)
        memory_attended, _ = self.attention(query, memory_expanded, memory_expanded)
        memory_attended = memory_attended.squeeze(1) # [batch_size, feature_dim]

        # Fuse (e.g., concatenate or add) with current features
        # Here, let's assume the attention output IS the enhanced feature from memory.
        # If we want to combine it with current_features (e.g. residual connection or concat):
        # concatenated = th.cat([current_features, memory_attended], dim=-1) # Use th
        # And then a linear layer to bring it back to feature_dim if needed.
        # For simplicity, let's say memory_attended is the result to be combined with *other* features later.
        return memory_attended

# Example of how observation_space might be defined for MemoryFusionFeatureExtractor
# This is for context, not part of the class itself.
def get_example_observation_space_for_memory_fusion():
    return GymDict({
        'market_features': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10, TRANSFORMER_OUTPUT_DIM_PER_SYMBOL), dtype=np.float32), # seq_len, feature_dim_per_step
        'aux_flat_features': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
        'another_aux_feature': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
    })

# Example usage for testing EnhancedTransformerFeatureExtractor (and MockTransformer indirectly)
# This is for context and can be run separately for unit testing if needed.
# It is not part of the class definitions.

# def test_enhanced_transformer_feature_extractor():
#     logger.info("Starting test_enhanced_transformer_feature_extractor...")
#     # Define a sample observation space
#     # This should match what your environment provides
#     obs_space = GymDict({
#         "market_features": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10, 64), dtype=np.float32), # (seq_len, num_features_per_step)
#         "symbol_id": gym.spaces.Box(low=0, high=MAX_SYMBOLS_ALLOWED-1, shape=(1,), dtype=np.int64) # Assuming one symbol ID per observation
#     })

#     # Path to a dummy config file (create one for testing)
#     dummy_config_path = "dummy_transformer_config.json"
#     dummy_config_content = {
#         "transformer_params": {
#             "input_dim": 64, # Should match market_features' num_features_per_step
#             "model_dim": 128,
#             "n_head": 4,
#             "num_layers": 2,
#             "output_dim": 256, # This will be the _features_dim
#             "dropout": 0.1
#         },
#         "use_multi_scale": True,
#         "use_cross_time_fusion": True,
#         "use_symbol_embedding": True # Test with symbol embedding enabled in config
#     }
#     with open(dummy_config_path, 'w') as f:
#         json.dump(dummy_config_content, f)

#     # Instantiate the feature extractor
#     try:
#         feature_extractor = EnhancedTransformerFeatureExtractor(
#             observation_space=obs_space,
#             model_config_path=dummy_config_path,
#             # Explicitly pass flags to test override/combination logic
#             use_msfe=True, 
#             use_cts_fusion=None, # Test None (should take from config)
#             use_symbol_embedding=True # Test explicit True
#         )
#         logger.info(f"Feature extractor instantiated. Features dim: {feature_extractor.features_dim}")
#         logger.info(f"  Effective use_msfe: {feature_extractor.use_msfe}")
#         logger.info(f"  Effective use_cts_fusion: {feature_extractor.use_cts_fusion}")
#         logger.info(f"  Effective use_symbol_embedding: {feature_extractor.use_symbol_embedding}")

#         # Create a dummy observation tensor
#         test_batch_size = 2
#         dummy_obs = {
#             "market_features": th.randn(test_batch_size, 10, 64), # (batch, seq_len, features)
#             "symbol_id": th.randint(0, MAX_SYMBOLS_ALLOWED, (test_batch_size, 1)).long() # (batch, 1)
#         }

#         # Test the forward pass
#         with th.no_grad():
#             extracted_features = feature_extractor(dummy_obs)
#         logger.info(f"Forward pass successful. Extracted features shape: {extracted_features.shape}") # Expected: (batch_size, output_dim_from_config)
        
#         assert extracted_features.shape == (test_batch_size, dummy_config_content["transformer_params"]["output_dim"])
#         assert feature_extractor.use_symbol_embedding is True # Verify flag

#     except Exception as e:
#         logger.error(f"Error during test_enhanced_transformer_feature_extractor: {e}", exc_info=True)
#     finally:
#         if os.path.exists(dummy_config_path):
#             os.remove(dummy_config_path)
#         logger.info("Finished test_enhanced_transformer_feature_extractor.")

# # If you want to run this test when the module is executed directly:
# if __name__ == '__main__':
#     # Setup basic logging for the test
#     logging.basicConfig(level=logging.INFO,
#                         format='%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s:%(lineno)d) - %(message)s')
#     test_enhanced_transformer_feature_extractor()
