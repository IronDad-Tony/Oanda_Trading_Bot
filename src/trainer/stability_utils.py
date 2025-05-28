#!/usr/bin/env python3
"""
Numerical Stability Utilities for AMP Training
Provides tools for monitoring and maintaining numerical stability during training.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from typing import Dict, Any, Optional, List
import warnings

# Add the project root to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from src.common.logger_setup import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

class NumericalStabilityMonitor:
    """
    A comprehensive numerical stability monitor for deep learning training.
    
    Features:
    - Gradient clipping with configurable norms
    - NaN/Infinity detection for parameters and gradients
    - Gradient statistics computation
    - AMP overflow detection
    """
    
    def __init__(self, 
                 gradient_clip_norm: float = 1.0,
                 nan_check_frequency: int = 500,
                 enable_gradient_clipping: bool = True):
        """
        Initialize the numerical stability monitor.
        
        Args:
            gradient_clip_norm: Maximum gradient norm for clipping
            nan_check_frequency: How often to check for NaN values (in steps)
            enable_gradient_clipping: Whether to enable gradient clipping
        """
        self.gradient_clip_norm = gradient_clip_norm
        self.nan_check_frequency = nan_check_frequency
        self.enable_gradient_clipping = enable_gradient_clipping
        
        # Statistics tracking
        self.nan_count = 0
        self.inf_count = 0
        self.gradient_explosion_count = 0
        self.total_checks = 0
        
        logger.info(f"NumericalStabilityMonitor initialized:")
        logger.info(f"  Gradient clip norm: {self.gradient_clip_norm}")
        logger.info(f"  NaN check frequency: {self.nan_check_frequency}")
        logger.info(f"  Gradient clipping enabled: {self.enable_gradient_clipping}")
    
    def clip_gradients(self, model: nn.Module, gradient_clip_norm_value: float, step: Optional[int] = None) -> float:
        """
        Apply gradient clipping to the model parameters and return the total norm before clipping.
        
        Args:
            model: The neural network model
            gradient_clip_norm_value: Maximum gradient norm for clipping
            step: Current training step (optional, for logging/tracking if needed later)
            
        Returns:
            Total norm of the gradients before clipping as a float.
        """
        if not self.enable_gradient_clipping:
            # Calculate norm without clipping if clipping is disabled
            total_norm_val = 0.0
            params_with_grads_no_clip = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            if not params_with_grads_no_clip:
                return 0.0
            for param in params_with_grads_no_clip:
                if param.grad is not None: # Should always be true due to filter
                    param_norm = param.grad.data.norm(2)
                    total_norm_val += param_norm.item() ** 2
            total_norm_val = total_norm_val ** 0.5
            return float(total_norm_val)

        params_with_grads = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
        if not params_with_grads:
            logger.warning("clip_gradients called on a model with no parameters requiring gradients or no gradients.")
            return 0.0

        total_norm_before_clipping = torch.nn.utils.clip_grad_norm_(
            params_with_grads,
            gradient_clip_norm_value
        )
        
        if total_norm_before_clipping.item() > gradient_clip_norm_value:
            self.gradient_explosion_count += 1
            # Logging of actual clipping event can be done by the caller if needed
        
        return float(total_norm_before_clipping.item())
    
    def should_check_nans(self, step: int) -> bool:
        """
        Determine if NaN/Infinity check should be performed at the current step.

        Args:
            step: Current training step.

        Returns:
            True if NaN check should be performed, False otherwise.
        """
        return step % self.nan_check_frequency == 0

    def check_for_nans(self, model: nn.Module, step: int) -> bool:
        """
        Check for NaN or Infinity values in model parameters and gradients.
        
        Args:
            model: The neural network model
            step: Current training step
            
        Returns:
            True if NaN/Inf detected, False otherwise
        """
        if step % self.nan_check_frequency != 0:
            return False
        
        self.total_checks += 1
        nan_detected = False
        inf_detected = False
        
        # Check parameters
        for name, param in model.named_parameters():
            if param is not None:
                if torch.isnan(param).any():
                    logger.warning(f"NaN detected in parameter {name} at step {step}")
                    nan_detected = True
                    self.nan_count += 1
                
                if torch.isinf(param).any():
                    logger.warning(f"Infinity detected in parameter {name} at step {step}")
                    inf_detected = True
                    self.inf_count += 1
                
                # Check gradients if they exist
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        logger.warning(f"NaN detected in gradient of {name} at step {step}")
                        nan_detected = True
                        self.nan_count += 1
                    
                    if torch.isinf(param.grad).any():
                        logger.warning(f"Infinity detected in gradient of {name} at step {step}")
                        inf_detected = True
                        self.inf_count += 1
        
        if nan_detected or inf_detected:
            logger.error(f"Numerical instability detected at step {step}")
            return True
        
        return False
    
    def compute_gradient_stats(self, model: nn.Module) -> Dict[str, float]:
        """
        Compute gradient statistics for monitoring.
        
        Args:
            model: The neural network model
            
        Returns:
            Dictionary with gradient statistics
        """
        total_norm = 0.0
        max_grad = 0.0
        min_grad = float('inf')
        param_count = 0
        zero_grad_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2)
                total_norm += grad_norm.item() ** 2
                
                grad_max = param.grad.data.abs().max().item()
                grad_min = param.grad.data.abs().min().item()
                
                max_grad = max(max_grad, grad_max)
                min_grad = min(min_grad, grad_min)
                
                param_count += 1
                
                if grad_norm.item() == 0:
                    zero_grad_count += 1
        
        total_norm = total_norm ** 0.5 if param_count > 0 else 0.0
        min_grad = min_grad if min_grad != float('inf') else 0.0
        
        return {
            "total_norm": total_norm,
            "max_gradient": max_grad,
            "min_gradient": min_grad,
            "param_count": param_count,
            "zero_grad_count": zero_grad_count,
            "avg_grad_norm": total_norm / param_count if param_count > 0 else 0.0
        }
    
    def check_amp_overflow(self, scaler: torch.cuda.amp.GradScaler) -> bool:
        """
        Check if AMP scaler has detected overflow.
        
        Args:
            scaler: The GradScaler used for AMP training
            
        Returns:
            True if overflow detected, False otherwise
        """
        if scaler is None:
            return False
        
        # Get the scale factor - if it's decreasing, overflow likely occurred
        current_scale = scaler.get_scale()
        
        # Check if scale is abnormally low (indicating recent overflows)
        if current_scale < 1.0:
            logger.warning(f"AMP scale factor is low: {current_scale}, indicating recent overflows")
            return True
        
        return False
    
    def get_stability_summary(self) -> Dict[str, Any]:
        """
        Get a summary of stability statistics.
        
        Returns:
            Dictionary with stability statistics
        """
        return {
            "total_checks": self.total_checks,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "gradient_explosion_count": self.gradient_explosion_count,
            "nan_rate": self.nan_count / max(self.total_checks, 1),
            "inf_rate": self.inf_count / max(self.total_checks, 1),
            "explosion_rate": self.gradient_explosion_count / max(self.total_checks, 1)
        }
    
    def reset_stats(self):
        """Reset all statistics counters."""
        self.nan_count = 0
        self.inf_count = 0
        self.gradient_explosion_count = 0
        self.total_checks = 0
        logger.info("Numerical stability statistics reset")


def validate_model_health(model: nn.Module, 
                         check_gradients: bool = True) -> Dict[str, Any]:
    """
    Standalone function to validate model health.
    
    Args:
        model: The neural network model to check
        check_gradients: Whether to check gradients as well as parameters
        
    Returns:
        Dictionary with health check results
    """
    health_info = {
        "healthy": True,
        "issues": [],
        "param_stats": {},
        "grad_stats": {}
    }
    
    param_issues = []
    grad_issues = []
    
    # Check parameters
    for name, param in model.named_parameters():
        if param is not None:
            if torch.isnan(param).any():
                param_issues.append(f"NaN in parameter {name}")
                health_info["healthy"] = False
            
            if torch.isinf(param).any():
                param_issues.append(f"Infinity in parameter {name}")
                health_info["healthy"] = False
            
            # Check gradients
            if check_gradients and param.grad is not None:
                if torch.isnan(param.grad).any():
                    grad_issues.append(f"NaN in gradient of {name}")
                    health_info["healthy"] = False
                
                if torch.isinf(param.grad).any():
                    grad_issues.append(f"Infinity in gradient of {name}")
                    health_info["healthy"] = False
    
    health_info["issues"] = param_issues + grad_issues
    health_info["param_issues"] = param_issues
    health_info["grad_issues"] = grad_issues
    
    return health_info


def emergency_amp_disable(scaler: torch.cuda.amp.GradScaler, 
                         reason: str = "numerical instability") -> bool:
    """
    Emergency function to disable AMP training.
    
    Args:
        scaler: The GradScaler to disable
        reason: Reason for disabling AMP
        
    Returns:
        True if successfully disabled, False otherwise
    """
    try:
        if scaler is not None:
            # Reset scaler to initial state
            scaler._scale = torch.tensor(2.**16, dtype=torch.float32)
            scaler._growth_factor = 2.0
            scaler._backoff_factor = 0.5
            scaler._growth_interval = 2000
            scaler._init_growth_tracker = 0
            
            logger.warning(f"AMP emergency disabled due to: {reason}")
            return True
    except Exception as e:
        logger.error(f"Failed to disable AMP: {e}")
    
    return False