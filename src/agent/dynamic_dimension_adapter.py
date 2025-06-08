"""
Dynamic Dimension Adapter System
Handles automatic dimension adaptation between different modules in the Oanda Trading Bot
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import logging
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class DimensionSpec:
    """Specification for tensor dimensions"""
    name: str
    expected_shape: Tuple[int, ...]
    min_dims: int
    max_dims: int
    adaptive: bool = True
    
class DimensionMismatchError(Exception):
    """Raised when dimensions cannot be adapted"""
    pass

class AdapterLayer(nn.Module):
    """Single adapter layer for dimension transformation"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        adapter_type: str = "linear",
        preserve_batch: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adapter_type = adapter_type
        self.preserve_batch = preserve_batch
        
        if adapter_type == "linear":
            self.adapter = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU() if input_dim != output_dim else nn.Identity(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )
        elif adapter_type == "projection":
            # Use projection for dimension reduction/expansion
            if input_dim > output_dim:
                self.adapter = nn.Sequential(
                    nn.Linear(input_dim, (input_dim + output_dim) // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear((input_dim + output_dim) // 2, output_dim)
                )
            else:
                self.adapter = nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.ReLU() if input_dim != output_dim else nn.Identity()
                )
        elif adapter_type == "attention":
            # Use attention mechanism for adaptive dimension mapping
            self.query = nn.Linear(input_dim, min(input_dim, output_dim))
            self.key = nn.Linear(input_dim, min(input_dim, output_dim))
            self.value = nn.Linear(input_dim, output_dim)
            self.scale = 1.0 / np.sqrt(min(input_dim, output_dim))
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dimension adaptation"""
        if self.adapter_type == "attention":
            return self._attention_forward(x)
        else:
            return self._standard_forward(x)
    
    def _standard_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for linear/projection adapters"""
        original_shape = x.shape
        
        # Handle different input shapes
        if x.dim() == 1:
            # 1D tensor
            adapted = self.adapter(x)
        elif x.dim() == 2:
            # 2D tensor (batch_size, features)
            adapted = self.adapter(x)
        elif x.dim() == 3:
            # 3D tensor (batch_size, seq_len, features)
            batch_size, seq_len, _ = x.shape
            x_reshaped = x.view(-1, self.input_dim)
            adapted = self.adapter(x_reshaped)
            adapted = adapted.view(batch_size, seq_len, self.output_dim)
        else:
            # Higher dimensional tensors - flatten last dimension
            *leading_dims, last_dim = x.shape
            x_reshaped = x.view(-1, last_dim)
            adapted = self.adapter(x_reshaped)
            adapted = adapted.view(*leading_dims, self.output_dim)
        
        return adapted
    
    def _attention_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attention-based forward pass"""
        # Apply attention mechanism
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Compute attention weights
        attention_weights = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        
        return attended

class DynamicDimensionAdapter(nn.Module):
    """
    Dynamic dimension adapter that automatically handles dimension mismatches
    between different modules in the trading system
    """
    
    def __init__(
        self,
        default_adapter_type: str = "linear",
        enable_caching: bool = True,
        max_cache_size: int = 100
    ):
        super().__init__()
        self.default_adapter_type = default_adapter_type
        self.enable_caching = enable_caching
        self.max_cache_size = max_cache_size
        
        # Adapter cache for efficient reuse
        self.adapter_cache = nn.ModuleDict()
        self.cache_usage = defaultdict(int)
        
        # Dimension specifications for known modules
        self.dimension_specs = {
            'strategy_innovation': DimensionSpec('strategy_innovation', (-1, 768), 2, 3),
            'market_state_awareness': DimensionSpec('market_state_awareness', (-1, 512), 2, 3),
            'meta_learning': DimensionSpec('meta_learning', (-1, 256), 2, 3),
            'position_manager': DimensionSpec('position_manager', (-1, 768), 1, 2),
            'anomaly_detector': DimensionSpec('anomaly_detector', (-1, -1, 768), 3, 3),
            'emergency_stop_loss': DimensionSpec('emergency_stop_loss', (-1,), 1, 2)
        }
          # Statistics for monitoring
        self.adaptation_stats = {
            'total_adaptations': 0,
            'successful_adaptations': 0,
            'failed_adaptations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def adapt_tensor(
        self,
        tensor: torch.Tensor,
        target_dim: int,
        source_module: str = "unknown",
        target_module: str = "unknown",
        adapter_type: Optional[str] = None
    ) -> torch.Tensor:
        """
        Adapt tensor to target dimension
        
        Args:
            tensor: Input tensor to adapt
            target_dim: Target dimension size
            source_module: Name of source module
            target_module: Name of target module
            adapter_type: Type of adapter to use
            
        Returns:
            Adapted tensor with target dimension
        """
        
        # Critical: Validate input is actually a tensor
        if not isinstance(tensor, torch.Tensor):
            logger.error(f"❌ Non-tensor object passed to adapt_tensor: {type(tensor)}")
            logger.error(f"   Object contents: {tensor}")
            logger.error(f"   Source: {source_module} -> Target: {target_module}")
            
            # Try to extract tensor from common container types
            if isinstance(tensor, dict):
                tensor_values = [v for v in tensor.values() if isinstance(v, torch.Tensor)]
                if tensor_values:
                    tensor = tensor_values[0]  # Use first tensor found
                    logger.warning(f"🔧 Extracted tensor from dictionary for adaptation")
                else:
                    logger.error(f"❌ No tensors found in dictionary, creating default tensor")
                    return torch.zeros(1, target_dim)
            elif isinstance(tensor, (list, tuple)):
                tensor_items = [item for item in tensor if isinstance(item, torch.Tensor)]
                if tensor_items:
                    tensor = tensor_items[0]  # Use first tensor found
                    logger.warning(f"🔧 Extracted tensor from {type(tensor).__name__} for adaptation")
                else:
                    logger.error(f"❌ No tensors found in {type(tensor).__name__}, creating default tensor")
                    return torch.zeros(1, target_dim)
            else:
                try:
                    tensor = torch.tensor(tensor, dtype=torch.float32)
                    logger.warning(f"🔧 Converted {type(tensor).__name__} to tensor for adaptation")
                except Exception as e:
                    logger.error(f"❌ Cannot convert {type(tensor).__name__} to tensor: {e}")
                    return torch.zeros(1, target_dim)
        
        # Ensure tensor has valid shape
        if tensor.numel() == 0:
            logger.warning(f"⚠️ Empty tensor for adaptation, creating default tensor")
            return torch.zeros(1, target_dim)
        
        # Get current dimension
        current_dim = tensor.size(-1)
        
        # No adaptation needed
        if current_dim == target_dim:
            return tensor
        
        # Generate adapter key
        adapter_key = f"{source_module}_{target_module}_{current_dim}_{target_dim}"
        
        # Check cache first
        if self.enable_caching and adapter_key in self.adapter_cache:
            adapter = self.adapter_cache[adapter_key]
            self.cache_usage[adapter_key] += 1
            self.adaptation_stats['cache_hits'] += 1
        else:
            # Create new adapter
            adapter_type = adapter_type or self.default_adapter_type
            adapter = AdapterLayer(
                input_dim=current_dim,
                output_dim=target_dim,
                adapter_type=adapter_type
            )
            
            # Cache the adapter
            if self.enable_caching:
                self._cache_adapter(adapter_key, adapter)
                self.adaptation_stats['cache_misses'] += 1
        
        try:
            # Apply adaptation
            adapted_tensor = adapter(tensor)
            self.adaptation_stats['successful_adaptations'] += 1
            self.adaptation_stats['total_adaptations'] += 1
            
            logger.debug(f"Successfully adapted tensor from {current_dim} to {target_dim} "
                        f"for {source_module} -> {target_module}")
            
            return adapted_tensor
            
        except Exception as e:
            self.adaptation_stats['failed_adaptations'] += 1
            self.adaptation_stats['total_adaptations'] += 1
            
            logger.error(f"Failed to adapt tensor from {current_dim} to {target_dim}: {e}")
            raise DimensionMismatchError(
                f"Cannot adapt tensor from dimension {current_dim} to {target_dim}: {str(e)}"
            )
    
    def adapt_tensor_to_spec(
        self,
        tensor: torch.Tensor,
        target_spec: DimensionSpec,
        source_module: str = "unknown"
    ) -> torch.Tensor:
        """Adapt tensor to match dimension specification"""
          # Get target dimension from spec
        if target_spec.expected_shape[-1] == -1:
            # Flexible dimension - no adaptation needed
            return tensor
        
        target_dim = target_spec.expected_shape[-1]
        return self.adapt_tensor(tensor, target_dim, source_module, target_spec.name)
    
    def batch_adapt_tensors(
        self,
        tensors: Dict[str, torch.Tensor],
        target_dims: Dict[str, int],
        source_module: str = "unknown"
    ) -> Dict[str, torch.Tensor]:
        """Adapt multiple tensors in batch"""
        
        adapted_tensors = {}
        
        for key, tensor in tensors.items():
            if key in target_dims:
                adapted_tensors[key] = self.adapt_tensor(
                    tensor,
                    target_dims[key],
                    source_module,
                    key
                )
            else:
                adapted_tensors[key] = tensor
        
        return adapted_tensors
    
    def smart_adapt(
        self,
        tensor: torch.Tensor,
        target_module: str,
        source_module: str = "unknown",
        fallback_dim: Optional[int] = None
    ) -> torch.Tensor:
        """
        Smart adaptation using known module specifications
        
        Args:
            tensor: Input tensor
            target_module: Target module name
            source_module: Source module name
            fallback_dim: Fallback dimension if target module unknown
            
        Returns:
            Adapted tensor
        """
        
        # Critical: Validate input is actually a tensor
        if not isinstance(tensor, torch.Tensor):
            logger.error(f"❌ Non-tensor object passed to smart_adapt: {type(tensor)}")
            logger.error(f"   Object contents: {tensor}")
            logger.error(f"   Source: {source_module} -> Target: {target_module}")
            
            # Try to extract tensor from common container types
            if isinstance(tensor, dict):
                tensor_values = [v for v in tensor.values() if isinstance(v, torch.Tensor)]
                if tensor_values:
                    tensor = tensor_values[0]  # Use first tensor found
                    logger.warning(f"🔧 Extracted tensor from dictionary for smart adaptation")
                else:
                    logger.error(f"❌ No tensors found in dictionary, creating default tensor")
                    fallback_dim = fallback_dim or 256  # Use fallback or default
                    return torch.zeros(1, fallback_dim)
            elif isinstance(tensor, (list, tuple)):
                tensor_items = [item for item in tensor if isinstance(item, torch.Tensor)]
                if tensor_items:
                    tensor = tensor_items[0]  # Use first tensor found
                    logger.warning(f"🔧 Extracted tensor from {type(tensor).__name__} for smart adaptation")
                else:
                    logger.error(f"❌ No tensors found in {type(tensor).__name__}, creating default tensor")
                    fallback_dim = fallback_dim or 256  # Use fallback or default
                    return torch.zeros(1, fallback_dim)
            else:
                try:
                    tensor = torch.tensor(tensor, dtype=torch.float32)
                    logger.warning(f"🔧 Converted {type(tensor).__name__} to tensor for smart adaptation")
                except Exception as e:
                    logger.error(f"❌ Cannot convert {type(tensor).__name__} to tensor: {e}")
                    fallback_dim = fallback_dim or 256  # Use fallback or default
                    return torch.zeros(1, fallback_dim)
        
        # Ensure tensor has valid shape
        if tensor.numel() == 0:
            logger.warning(f"⚠️ Empty tensor for smart adaptation, creating default tensor")
            fallback_dim = fallback_dim or 256  # Use fallback or default
            return torch.zeros(1, fallback_dim)
        
        # Check if we have specs for the target module
        if target_module in self.dimension_specs:
            spec = self.dimension_specs[target_module]
            
            # Handle flexible dimensions
            if spec.expected_shape[-1] == -1:
                return tensor
            
            target_dim = spec.expected_shape[-1]
            
            # Validate dimension constraints
            if tensor.dim() < spec.min_dims or tensor.dim() > spec.max_dims:
                logger.warning(f"Tensor dimensions {tensor.dim()} outside valid range "
                              f"[{spec.min_dims}, {spec.max_dims}] for {target_module}")
            
            return self.adapt_tensor(tensor, target_dim, source_module, target_module)
        
        elif fallback_dim is not None:
            return self.adapt_tensor(tensor, fallback_dim, source_module, target_module)
        
        else:
            logger.warning(f"No dimension specification found for {target_module}, "
                          f"returning original tensor")
            return tensor
    
    def auto_detect_and_adapt(
        self,
        tensor: torch.Tensor,
        context: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Automatically detect required adaptation based on context
        
        Args:
            tensor: Input tensor
            context: Context information containing target requirements
            
        Returns:
            Adapted tensor
        """
        
        # Extract information from context
        target_module = context.get('target_module', 'unknown')
        source_module = context.get('source_module', 'unknown')
        required_dim = context.get('required_dim')
        
        # Try smart adaptation first
        if target_module != 'unknown':
            return self.smart_adapt(tensor, target_module, source_module, required_dim)
        
        # Fallback to direct dimension adaptation
        elif required_dim is not None:
            return self.adapt_tensor(tensor, required_dim, source_module, target_module)
        
        # No adaptation possible
        else:
            logger.warning("Insufficient context for automatic adaptation")
            return tensor
    
    def _cache_adapter(self, key: str, adapter: AdapterLayer):
        """Cache an adapter with LRU eviction"""
        
        # Check cache size limit
        if len(self.adapter_cache) >= self.max_cache_size:
            # Remove least recently used adapter
            lru_key = min(self.cache_usage.keys(), key=lambda k: self.cache_usage[k])
            del self.adapter_cache[lru_key]
            del self.cache_usage[lru_key]
        
        # Add to cache
        self.adapter_cache[key] = adapter
        self.cache_usage[key] = 0
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics"""
        
        stats = self.adaptation_stats.copy()
        stats['cache_size'] = len(self.adapter_cache)
        stats['success_rate'] = (
            stats['successful_adaptations'] / max(stats['total_adaptations'], 1)
        )
        stats['cache_hit_rate'] = (
            stats['cache_hits'] / max(stats['cache_hits'] + stats['cache_misses'], 1)
        )
        
        return stats
    
    def clear_cache(self):
        """Clear adapter cache"""
        self.adapter_cache.clear()
        self.cache_usage.clear()
        logger.info("Adapter cache cleared")
    
    def register_dimension_spec(self, name: str, spec: DimensionSpec):
        """Register a new dimension specification"""
        self.dimension_specs[name] = spec
        logger.info(f"Registered dimension spec for {name}: {spec}")
    
    def update_dimension_spec(self, name: str, **kwargs):
        """Update existing dimension specification"""
        if name in self.dimension_specs:
            spec = self.dimension_specs[name]
            for key, value in kwargs.items():
                if hasattr(spec, key):
                    setattr(spec, key, value)
            logger.info(f"Updated dimension spec for {name}")
        else:
            logger.warning(f"No dimension spec found for {name}")

# Utility functions for common adaptation patterns

def ensure_batch_dimension(tensor: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
    """Ensure tensor has batch dimension"""
    if tensor.dim() == 1:
        return tensor.unsqueeze(0).expand(batch_size, -1)
    return tensor

def ensure_sequence_dimension(
    tensor: torch.Tensor,
    seq_len: int = 1,
    batch_size: int = 1
) -> torch.Tensor:
    """Ensure tensor has sequence dimension"""
    if tensor.dim() == 1:
        return tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
    elif tensor.dim() == 2:
        return tensor.unsqueeze(1).expand(-1, seq_len, -1)
    return tensor

def flatten_to_feature_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten tensor to feature dimension"""
    if tensor.dim() > 2:
        return tensor.view(-1, tensor.size(-1))
    return tensor

def reshape_for_module(
    tensor: torch.Tensor,
    target_shape: Tuple[int, ...],
    preserve_batch: bool = True
) -> torch.Tensor:
    """Reshape tensor for specific module requirements"""
    if preserve_batch and tensor.dim() > 1:
        batch_size = tensor.size(0)
        return tensor.view(batch_size, *target_shape[1:])
    else:
        return tensor.view(target_shape)
