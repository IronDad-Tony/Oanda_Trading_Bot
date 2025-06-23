# src/agent/enhanced_feature_extractor.py
"""
增強版特徵提取器 - Phase 3
使用增強版UniversalTradingTransformer進行特徵提取
支持多尺度特徵提取、自適應注意力機制和跨時間尺度融合
"""

import json
import logging
from typing import Optional, Dict, List

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.spaces import Dict as GymDict
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs

logger = logging.getLogger(__name__)

MAX_SYMBOLS_ALLOWED = 10
TRANSFORMER_OUTPUT_DIM_PER_SYMBOL = 64

class UniversalTransformer(nn.Module):
    """
    A universal transformer module that processes input features and captures
    intermediate layer activations for visualization and analysis.
    """
    def __init__(self, input_dim, model_dim, output_dim, nhead, num_layers, dropout_rate=0.1, **kwargs):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, batch_first=True, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(model_dim, output_dim)
        logger.info(f"UniversalTransformer initialized: input_dim={input_dim}, model_dim={model_dim}, output_dim={output_dim}")
        # This dictionary will hold the activations from each layer
        self.activations = {}

    def forward(self, src: th.Tensor) -> th.Tensor:
        """
        Forward pass for the transformer. It processes the source tensor and
        stores the output of each encoder layer in the self.activations dictionary.
        """
        # Reset activations at the beginning of each forward pass
        self.activations = {}
        
        x = self.input_proj(src)
        
        # Iterate through encoder layers to capture activations
        for i, layer in enumerate(self.transformer_encoder.layers):
            x = layer(x)
            # Detach the tensor to prevent gradients from flowing back from this
            # stored value during training, as it's for diagnostics only.
            self.activations[f'encoder_layer_{i}'] = x.detach()

        # The final output is based on the last token's representation,
        # which aggregates information from the sequence.
        last_token_output = x[:, -1, :] 
        output = self.output_proj(last_token_output)
        return output

class EnhancedTransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: GymDict,
                 model_config: Optional[Dict] = None,
                 model_config_path: Optional[str] = None,
                 **kwargs):

        if model_config:
            self.model_config = model_config
        elif model_config_path:
            try:
                with open(model_config_path, 'r') as f:
                    self.model_config = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load model config from {model_config_path}: {e}", exc_info=True)
                raise
        else:
            raise ValueError("EnhancedTransformerFeatureExtractor requires 'model_config' or 'model_config_path'.")

        # --- Parameter Calculation Stage ---
        # Calculate all dimensions and parameters before calling super().__init__

        self.use_symbol_embedding = self.model_config.get('use_symbol_embedding', False)
        self.symbol_embedding_dim = 0
        self.symbol_vocab_size = 0
        if self.use_symbol_embedding:
            self.symbol_vocab_size = self.model_config.get('symbol_universe_size')
            if self.symbol_vocab_size is None:
                logger.warning("'symbol_universe_size' not in config, inferring from observation_space.")
                try:
                    # The space is Box(low, high, shape, dtype), high is inclusive.
                    max_id = int(observation_space.spaces['symbol_id'].high[0])
                    self.symbol_vocab_size = max_id + 1
                    logger.info(f"Inferred symbol_universe_size: {self.symbol_vocab_size}")
                except (KeyError, AttributeError, IndexError) as e:
                    raise ValueError(f"Cannot determine symbol_universe_size. Must be in config or observation_space['symbol_id']. Error: {e}")
            
            context_params = self.model_config.get('contextualized_embedding_params', {})
            self.symbol_embedding_dim = context_params.get('embedding_dim_override', self.model_config.get('symbol_embedding_dim', 16))

        try:
            market_features_dim = observation_space.spaces['market_features'].shape[-1]
        except KeyError:
            raise ValueError("'market_features' not found in observation space. It is required.")
        
        transformer_input_dim = market_features_dim + self.symbol_embedding_dim
        transformer_output_dim = self.model_config.get('output_dim', self.model_config.get('transformer_output_dim', 64))
        
        self.aux_features_dim = 0
        for key, space in observation_space.spaces.items():
            if key not in ['market_features', 'symbol_id']:
                self.aux_features_dim += int(np.prod(space.shape))

        features_dim = transformer_output_dim + self.aux_features_dim

        # --- Superclass Initialization ---
        # This MUST be called before assigning any nn.Module to self.
        super().__init__(observation_space, features_dim=features_dim)
        self.observation_space = observation_space
        
        # This dictionary will be populated with layer activations after each forward pass
        self.activations: Dict[str, th.Tensor] = {}


        # --- Module Initialization Stage ---
        # Now it's safe to create and assign nn.Module instances.

        self.symbol_embedding = None
        if self.use_symbol_embedding:
            self.symbol_embedding = nn.Embedding(self.symbol_vocab_size, self.symbol_embedding_dim)
            logger.info(f"Symbol embedding enabled: vocab_size={self.symbol_vocab_size}, embedding_dim={self.symbol_embedding_dim}")

        self.transformer = UniversalTransformer(
            input_dim=transformer_input_dim,
            model_dim=self.model_config.get('hidden_dim', self.model_config.get('transformer_model_dim', 128)),
            output_dim=transformer_output_dim,
            nhead=self.model_config.get('num_heads', self.model_config.get('transformer_nhead', 4)),
            num_layers=self.model_config.get('num_layers', self.model_config.get('transformer_num_layers', 3)),
            dropout_rate=self.model_config.get('dropout_rate', 0.1)
        )
        
        logger.info(f"EnhancedTransformerFeatureExtractor initialized. Total features_dim: {self._features_dim}")
        logger.info(f"  - Transformer output dim: {transformer_output_dim}")
        logger.info(f"  - Auxiliary features dim: {self.aux_features_dim}")
        logger.info(f"  Associated model_config: {json.dumps(self.model_config, indent=2)}")

    def forward(self, observations: PyTorchObs) -> th.Tensor:
        market_features = observations['market_features']
        transformer_input = market_features

        if self.use_symbol_embedding:
            if 'symbol_id' not in observations:
                 raise ValueError("'symbol_id' not in observations, but use_symbol_embedding is True.")
            symbol_ids = observations['symbol_id'].long()

            max_id = symbol_ids.max().item()
            if max_id >= self.symbol_vocab_size:
                logger.error(f"Symbol ID {max_id} is out of bounds for embedding layer (size {self.symbol_vocab_size}). Clamping.")
                symbol_ids = th.clamp(symbol_ids, 0, self.symbol_vocab_size - 1)

            symbol_embeds = self.symbol_embedding(symbol_ids)
            
            if len(symbol_embeds.shape) == 2:
                symbol_embeds = symbol_embeds.unsqueeze(1)
            
            expanded_embeds = symbol_embeds.expand(-1, market_features.shape[1], -1)
            transformer_input = th.cat([market_features, expanded_embeds], dim=-1)

        transformer_output = self.transformer(transformer_input)
        
        # After the forward pass, the transformer's activations are updated.
        # Copy them to this class's attribute so a callback can access them.
        self.activations = self.transformer.activations


        aux_features_list = []
        for key in self.observation_space.spaces:
            if key not in ['market_features', 'symbol_id']:
                aux_features_list.append(th.flatten(observations[key], start_dim=1))

        if aux_features_list:
            all_aux_features = th.cat(aux_features_list, dim=1)
            combined_features = th.cat([transformer_output, all_aux_features], dim=1)
        else:
            combined_features = transformer_output

        return combined_features

class EnhancedTransformerFeatureExtractorWithMemory(EnhancedTransformerFeatureExtractor):
    def __init__(self, observation_space: GymDict, 
                 enhanced_transformer_output_dim_per_symbol: int = TRANSFORMER_OUTPUT_DIM_PER_SYMBOL,
                 memory_size: int = 32,
                 model_config: Optional[Dict] = None,
                 model_config_path: Optional[str] = None):
        super().__init__(observation_space, model_config=model_config, model_config_path=model_config_path)
        
        self.memory_size = memory_size
        self.feature_dim = self._features_dim
        
        self.register_buffer('long_term_memory', th.zeros(memory_size, self.feature_dim))
        self.register_buffer('memory_ptr', th.zeros(1, dtype=th.long))
        
        self.memory_fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.GELU(),
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        logger.info(f"Initialized EnhancedTransformerFeatureExtractorWithMemory, memory size: {memory_size}")
    
    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        base_features = super().forward(observations)
        
        if self.training:
            self._update_memory(base_features.detach())
        
        memory_enhanced_features = self._retrieve_and_fuse_memory(base_features)
        
        return memory_enhanced_features
    
    def _update_memory(self, features: th.Tensor):
        batch_size = features.shape[0]
        for i in range(batch_size):
            ptr = int(self.memory_ptr.item())
            self.long_term_memory[ptr] = features[i]
            self.memory_ptr[0] = (ptr + 1) % self.memory_size
    
    def _retrieve_and_fuse_memory(self, current_features: th.Tensor) -> th.Tensor:
        batch_size = current_features.shape[0]
        
        memory_attended, _ = self.memory_attention(
            current_features.unsqueeze(1),
            self.long_term_memory.unsqueeze(0).expand(batch_size, -1, -1),
            self.long_term_memory.unsqueeze(0).expand(batch_size, -1, -1)
        )
        
        memory_attended = memory_attended.squeeze(1)
        
        concatenated = th.cat([current_features, memory_attended], dim=-1)
        fused_features = self.memory_fusion(concatenated)
        
        return fused_features + current_features

class MemoryFusionFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space: GymDict, 
                 main_feature_key: str = 'market_features',
                 other_feature_keys: Optional[List[str]] = None,
                 features_dim: int = 256,
                 memory_size: int = 100, 
                 num_attention_heads: int = 4,
                 enhanced_transformer_output_dim_per_symbol: int = TRANSFORMER_OUTPUT_DIM_PER_SYMBOL,
                 ): 
        super().__init__(observation_space, features_dim)
        self.main_feature_key = main_feature_key
        self.other_feature_keys = other_feature_keys if other_feature_keys is not None else []
        self.memory_size = memory_size
        
        self.feature_dim = enhanced_transformer_output_dim_per_symbol 

        self.register_buffer('memory_bank', th.zeros(memory_size, self.feature_dim))
        self.register_buffer('memory_ptr', th.zeros(1, dtype=th.long))

        self.attention = nn.MultiheadAttention(embed_dim=self.feature_dim, 
                                             num_heads=num_attention_heads, 
                                             batch_first=True)
        
        other_features_dim = 0
        if self.other_feature_keys:
            for key in self.other_feature_keys:
                space = observation_space[key]
                if isinstance(space, gym.spaces.Box):
                    other_features_dim += int(np.prod(space.shape))
                else:
                    raise ValueError(f"Unsupported space type for other_feature_key '{key}': {type(space)}")
        
        self.fusion_layer_input_dim = self.feature_dim + other_features_dim
        self.fusion_layer = nn.Linear(self.fusion_layer_input_dim, features_dim)
        self.output_activation = nn.ReLU()

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        current_main_features = observations[self.main_feature_key]
        if current_main_features.ndim == 3:
            current_main_features = current_main_features[:, -1, :]

        memory_enhanced_features = self._retrieve_and_fuse_memory(current_main_features)
        self._update_memory(current_main_features.detach())

        other_features_list = []
        if self.other_feature_keys:
            for key in self.other_feature_keys:
                other_features_list.append(observations[key].view(observations[key].size(0), -1))
        
        if other_features_list:
            other_features = th.cat(other_features_list, dim=1)
            combined_features = th.cat([memory_enhanced_features, other_features], dim=1)
        else:
            combined_features = memory_enhanced_features

        output_features = self.output_activation(self.fusion_layer(combined_features))
        return output_features

    def _update_memory(self, features: th.Tensor):
        batch_size = features.size(0)
        ptr = int(self.memory_ptr[0])
        for i in range(batch_size):
            self.memory_bank[ptr] = features[i]
            ptr = (ptr + 1) % self.memory_size
        self.memory_ptr[0] = ptr

    def _retrieve_and_fuse_memory(self, current_features: th.Tensor) -> th.Tensor:
        batch_size = current_features.size(0)
        query = current_features.unsqueeze(1)
        memory_expanded = self.memory_bank.unsqueeze(0).repeat(batch_size, 1, 1)
        memory_attended, _ = self.attention(query, memory_expanded, memory_expanded)
        memory_attended = memory_attended.squeeze(1)
        return memory_attended

class AttentionMemoryFusion(nn.Module):
    def __init__(self, feature_dim: int, memory_size: int, head_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.head_dim = head_dim

        self.query_proj = nn.Linear(feature_dim, head_dim)
        self.key_proj = nn.Linear(feature_dim, head_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim * 2, feature_dim)

        self.register_buffer('memory', th.zeros(memory_size, self.feature_dim))
        self.register_buffer('memory_ptr', th.zeros(1, dtype=th.long))

    def forward(self, current_features: th.Tensor) -> th.Tensor:
        memory_features = self._retrieve_and_fuse_memory(current_features)
        self._update_memory(current_features)
        return memory_features

    def _update_memory(self, features: th.Tensor):
        batch_size = features.shape[0]
        if self.memory_ptr + batch_size > self.memory_size:
            self.memory_ptr[0] = 0
        
        self.memory[self.memory_ptr[0]:self.memory_ptr[0] + batch_size] = features.detach()
        self.memory_ptr[0] = (self.memory_ptr[0] + batch_size) % self.memory_size

    def _retrieve_and_fuse_memory(self, current_features: th.Tensor) -> th.Tensor:
        query = self.query_proj(current_features)
        keys = self.key_proj(self.memory)
        values = self.value_proj(self.memory)

        attention_scores = th.matmul(query, keys.T) / np.sqrt(self.head_dim)
        attention_weights = th.softmax(attention_scores, dim=-1)

        memory_attended = th.matmul(attention_weights, values)

        concatenated = th.cat([current_features, memory_attended], dim=-1)
        fused_features = self.out_proj(concatenated)
        return fused_features
