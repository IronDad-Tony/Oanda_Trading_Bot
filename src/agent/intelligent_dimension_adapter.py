"""
Intelligent Dimension Adapter System
Advanced dynamic dimension adaptation with smart tensor reshaping and feature preservation
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
class ComponentSpec:
    """Component specification for intelligent adaptation"""
    name: str
    expected_input_range: Tuple[int, int]  # (min_dim, max_dim)
    preferred_dim: Optional[int] = None
    adaptive: bool = True
    preserve_sequence: bool = True
    
class IntelligentDimensionAdapter(nn.Module):
    """
    Intelligent dimension adapter that learns optimal transformations
    between different components while preserving important features
    """
    
    def __init__(
        self,
        default_strategy: str = "smart_projection",
        enable_learning: bool = True,
        cache_size: int = 100
    ):
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO) # Or a configurable level
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(ch)
            
        self.default_strategy = default_strategy
        self.enable_learning = enable_learning
        self.cache_size = cache_size
        
        # Component specifications
        self.component_specs = {}
        
        # Adaptive transformation networks
        self.transformation_networks = nn.ModuleDict()
        
        # Usage statistics for optimization
        self.usage_stats = defaultdict(int)
        self.performance_stats = defaultdict(list)
        
        # Cache for computed transformations
        self.transformation_cache = {}
        
    def register_component(
        self,
        name: str,
        expected_input_range: Tuple[int, int],
        preferred_dim: Optional[int] = None,
        adaptive: bool = True,
        preserve_sequence: bool = True
    ):
        """Register a component with its dimension requirements"""
        spec = ComponentSpec(
            name=name,
            expected_input_range=expected_input_range,
            preferred_dim=preferred_dim,
            adaptive=adaptive,
            preserve_sequence=preserve_sequence
        )
        
        self.component_specs[name] = spec
        logger.info(f"Registered component: {name} with input range {expected_input_range}")
        
    def get_optimal_dimension(self, component_name: str, input_tensor: torch.Tensor) -> int:
        """Determine optimal dimension for a component based on input tensor"""
        
        if component_name not in self.component_specs:
            # Auto-register with flexible specs
            input_dim = input_tensor.size(-1)
            self.register_component(
                component_name,
                expected_input_range=(input_dim // 2, input_dim * 2),
                preferred_dim=input_dim,
                adaptive=True
            )
        
        spec = self.component_specs[component_name]
        input_dim = input_tensor.size(-1)
        
        # If input is within expected range, use as-is
        if spec.expected_input_range[0] <= input_dim <= spec.expected_input_range[1]:
            return input_dim
        
        # Use preferred dimension if specified
        if spec.preferred_dim is not None:
            return spec.preferred_dim
        
        # Calculate optimal dimension based on input
        min_dim, max_dim = spec.expected_input_range
        
        if input_dim < min_dim:
            return min_dim
        elif input_dim > max_dim:
            return max_dim
        else:
            return input_dim
    
    def create_transformation_network(
        self,
        input_dim: int,
        output_dim: int,
        strategy: str = None
    ) -> nn.Module:
        """Create a transformation network for dimension adaptation"""
        
        strategy = strategy or self.default_strategy
        
        if strategy == "smart_projection":
            return self._create_smart_projection(input_dim, output_dim)
        elif strategy == "attention_based":
            return self._create_attention_based(input_dim, output_dim)
        elif strategy == "residual_adaptation":
            return self._create_residual_adaptation(input_dim, output_dim)
        else:
            return self._create_linear_adaptation(input_dim, output_dim)
    
    def _create_smart_projection(self, input_dim: int, output_dim: int) -> nn.Module:
        """Create smart projection network that preserves important features"""
        
        if input_dim == output_dim:
            return nn.Identity()
        
        # Use different strategies based on dimension relationship
        if input_dim > output_dim:
            # Dimension reduction with feature selection
            return nn.Sequential(
                nn.Linear(input_dim, max(output_dim * 2, input_dim // 2)),
                nn.LayerNorm(max(output_dim * 2, input_dim // 2)),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(max(output_dim * 2, input_dim // 2), output_dim),
                nn.LayerNorm(output_dim)
            )
        else:
            # Dimension expansion with feature enhancement
            return nn.Sequential(
                nn.Linear(input_dim, min(output_dim // 2, input_dim * 2)),
                nn.LayerNorm(min(output_dim // 2, input_dim * 2)),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(min(output_dim // 2, input_dim * 2), output_dim),
                nn.LayerNorm(output_dim)
            )
    
    def _create_attention_based(self, input_dim: int, output_dim: int) -> nn.Module:
        """Create attention-based transformation"""
        
        if input_dim == output_dim:
            return nn.Identity()
        
        hidden_dim = min(input_dim, output_dim)
        
        class AttentionAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.query = nn.Linear(input_dim, hidden_dim)
                self.key = nn.Linear(input_dim, hidden_dim)
                self.value = nn.Linear(input_dim, output_dim)
                self.scale = 1.0 / np.sqrt(hidden_dim)
                
            def forward(self, x):
                # Ensure proper shape for attention
                original_shape = x.shape
                if x.dim() > 2:
                    x = x.view(-1, x.size(-1))
                
                Q = self.query(x)
                K = self.key(x)
                V = self.value(x)
                
                # Self-attention
                attention_weights = torch.softmax(
                    torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1
                )
                output = torch.matmul(attention_weights, V)
                
                # Restore original shape if needed
                if len(original_shape) > 2:
                    output = output.view(*original_shape[:-1], output_dim)
                
                return output
        
        return AttentionAdapter()
    
    def _create_residual_adaptation(self, input_dim: int, output_dim: int) -> nn.Module:
        """Create residual adaptation network"""
        
        if input_dim == output_dim:
            return nn.Identity()
        
        class ResidualAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(input_dim, output_dim)
                self.residual_path = nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.GELU(),
                    nn.Linear(input_dim, output_dim)
                )
                self.layer_norm = nn.LayerNorm(output_dim)
                
            def forward(self, x):
                main_path = self.projection(x)
                residual = self.residual_path(x)
                return self.layer_norm(main_path + residual)
        
        return ResidualAdapter()
    
    def _create_linear_adaptation(self, input_dim: int, output_dim: int) -> nn.Module:
        """Create simple linear adaptation"""
        
        if input_dim == output_dim:
            return nn.Identity()
        
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
    
    def adapt_tensor(
        self,
        tensor: torch.Tensor,
        component_name: str,
        target_dim: Optional[int] = None,
        source_component: str = "unknown",
        strategy: str = None
    ) -> torch.Tensor:
        """Intelligently adapt tensor for target component"""
        
        # Determine optimal dimension for target component
        # If target_dim is provided, use it. Otherwise, use component_spec.
        if target_dim is None:
            target_dim_resolved = self.get_optimal_dimension(component_name, tensor)
        else:
            target_dim_resolved = target_dim
            
        input_dim = tensor.size(-1)
        
        # No adaptation needed
        if input_dim == target_dim_resolved:
            return tensor
        
        # Generate transformation key
        # Use component_name in the key as it's the primary identifier from HLIS
        transform_key = f"{source_component}_{component_name}_{input_dim}_{target_dim_resolved}"
        
        # Check cache
        if transform_key in self.transformation_cache:
            transformation = self.transformation_cache[transform_key]
            self.usage_stats[transform_key] += 1
        else:
            # Create new transformation
            transformation = self.create_transformation_network(
                input_dim, target_dim_resolved, strategy
            )
            
            # Cache the transformation
            self._cache_transformation(transform_key, transformation)
        
        # Apply transformation
        try:
            adapted_tensor = transformation(tensor)
            
            # Track performance
            self.performance_stats[transform_key].append({
                'input_shape': tensor.shape,
                'output_shape': adapted_tensor.shape,
                'success': True
            })
            
            self.logger.debug(f"Successfully adapted tensor from {tensor.shape} to {adapted_tensor.shape} "
                        f"for {source_component} -> {component_name}")
            
            return adapted_tensor
            
        except Exception as e:
            self.logger.error(f"Failed to adapt tensor for {component_name}: {e}")
            
            # Track failure
            self.performance_stats[transform_key].append({
                'input_shape': tensor.shape,
                'output_shape': None,
                'success': False,
                'error': str(e)
            })
            
            # Return original tensor as fallback
            return tensor
    
    def _cache_transformation(self, key: str, transformation: nn.Module):
        """Cache transformation with size limits"""
        
        if len(self.transformation_cache) >= self.cache_size:
            # Remove least used transformation
            lru_key = min(self.usage_stats.keys(), key=lambda k: self.usage_stats[k])
            del self.transformation_cache[lru_key]
            del self.usage_stats[lru_key]
        
        self.transformation_cache[key] = transformation
        self.usage_stats[key] = 0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        stats = {
            'total_transformations': len(self.performance_stats),
            'cache_size': len(self.transformation_cache),
            'usage_stats': dict(self.usage_stats),
            'component_specs': {name: {
                'input_range': spec.expected_input_range,
                'preferred_dim': spec.preferred_dim,
                'adaptive': spec.adaptive
            } for name, spec in self.component_specs.items()}
        }
        
        # Calculate success rates
        success_rates = {}
        for key, performances in self.performance_stats.items():
            total = len(performances)
            successful = sum(1 for p in performances if p['success'])
            success_rates[key] = successful / total if total > 0 else 0.0
        
        stats['success_rates'] = success_rates
        
        return stats
    
    def optimize_transformations(self):
        """Optimize transformations based on usage patterns"""
        
        if not self.enable_learning:
            return
        
        # Remove unused transformations
        to_remove = [key for key, count in self.usage_stats.items() if count == 0]
        for key in to_remove:
            if key in self.transformation_cache:
                del self.transformation_cache[key]
            if key in self.usage_stats:
                del self.usage_stats[key]
            if key in self.performance_stats:
                del self.performance_stats[key]
        
        logger.info(f"Optimized transformations: removed {len(to_remove)} unused transformations")
    
    def clear_cache(self):
        """Clear all cached transformations"""
        
        self.transformation_cache.clear()
        self.usage_stats.clear()
        self.performance_stats.clear()
        logger.info("Cleared all transformation caches")
    
    def get_component_spec(self, component_name: str) -> Optional[ComponentSpec]:
        """Retrieve the specification for a given component."""
        return self.component_specs.get(component_name)

# Utility functions for tensor shape handling

def ensure_compatible_shape(
    tensor: torch.Tensor,
    target_component: str,
    preserve_sequence: bool = True
) -> torch.Tensor:
    """Ensure tensor has compatible shape for target component"""
    
    if preserve_sequence and tensor.dim() == 2:
        # Add sequence dimension if needed
        return tensor.unsqueeze(1)
    elif not preserve_sequence and tensor.dim() == 3:
        # Remove sequence dimension if needed
        return tensor.squeeze(1) if tensor.size(1) == 1 else tensor.mean(dim=1)
    
    return tensor

def smart_reshape(
    tensor: torch.Tensor,
    target_shape: Tuple[int, ...],
    strategy: str = "preserve_features"
) -> torch.Tensor:
    """Smart tensor reshaping with feature preservation"""
    
    if strategy == "preserve_features":
        # Preserve the last dimension (features)
        if tensor.numel() == np.prod(target_shape):
            return tensor.view(target_shape)
        else:
            # Adapt feature dimension if needed
            current_features = tensor.size(-1)
            target_features = target_shape[-1]
            
            if current_features != target_features:
                # Simple linear adaptation
                adapter = nn.Linear(current_features, target_features)
                tensor = adapter(tensor)
            
            return tensor.view(target_shape)
    
    elif strategy == "interpolate":
        # Use interpolation for smooth adaptation
        target_size = target_shape[-1]
        if tensor.size(-1) != target_size:
            tensor = F.interpolate(
                tensor.unsqueeze(0).unsqueeze(0),
                size=target_size,
                mode='linear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        
        return tensor.view(target_shape)
    
    else:
        # Default reshape
        return tensor.view(target_shape)
