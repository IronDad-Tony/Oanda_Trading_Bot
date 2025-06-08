"""
High-Level Integration System for Phase 5
Combines all Phase 5 components with advanced monitoring and control
Includes anomaly detection, dynamic position management, and emergency controls
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
from datetime import datetime, timedelta
import warnings
import json
import traceback # Added import

# Global logger for module-level logging if needed
global_logger = logging.getLogger(__name__) # Use __name__ for module-level logger
if not global_logger.handlers:
    global_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    global_logger.addHandler(ch)
    # Example: Add a file handler to the global_logger if desired
    # fh_global = logging.FileHandler('global_high_level_system.log')
    # fh_global.setFormatter(formatter)
    # global_logger.addHandler(fh_global)

# Fallback TaskBatch definition if not available from .meta_learning_optimizer
try:
    from .meta_learning_optimizer import TaskBatch, MetaLearningOptimizer
except ImportError:
    global_logger.warning("Could not import TaskBatch or MetaLearningOptimizer from .meta_learning_optimizer. Using fallback definition for TaskBatch.") # Use global_logger
    if 'TaskBatch' not in globals(): # Check if TaskBatch is already defined
        @dataclass
        class TaskBatch:
            support_data: torch.Tensor
            support_labels: torch.Tensor
            query_data: torch.Tensor
            query_labels: torch.Tensor
            task_id: str = "unknown_task"
            market_state: Optional[str] = "unknown"
            difficulty: Optional[float] = 0.5
    
    # Fallback for MetaLearningOptimizer if needed for type hinting, though actual instance is passed
    if 'MetaLearningOptimizer' not in globals():
        class MetaLearningOptimizer(nn.Module): # type: ignore
            def optimize_and_adapt(self, features, context, task_batches):
                raise NotImplementedError("Fallback MetaLearningOptimizer cannot be used.")

try:
    from .dynamic_dimension_adapter import DynamicDimensionAdapter, ComponentSpec
except ImportError:
    global_logger.warning("DynamicDimensionAdapter not found. Dynamic adaptation features will be limited.") # Use global_logger
    DynamicDimensionAdapter = None
    ComponentSpec = None 

try:
    from .intelligent_dimension_adapter import IntelligentDimensionAdapter
except ImportError:
    global_logger.warning("IntelligentDimensionAdapter not found. Advanced adaptation features will be limited.") # Use global_logger
    IntelligentDimensionAdapter = None


class AnomalyDetector(nn.Module): # Placeholder
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, 1)
    def forward(self, x):
        # Ensure x is 2D or 3D
        if x.dim() == 1:
            x = x.unsqueeze(0) # Add batch dimension if missing
        if x.dim() == 3 and x.size(1) > 1: # If [batch, seq, features]
            x = x.mean(dim=1) # Average over sequence
        elif x.dim() == 3 and x.size(1) == 1: # If [batch, 1, features]
            x = x.squeeze(1) # Remove seq dim

        if x.size(-1) != self.input_dim:
            # This should ideally be handled by an adapter before reaching here
            # For now, log a warning and try a simple pass-through or error
            global_logger.warning(f"AnomalyDetector: Input dim {x.size(-1)} does not match expected {self.input_dim}. Trying to proceed.")
            # If a projection is absolutely necessary here as a last resort:
            # self.fc = nn.Linear(x.size(-1), 1).to(x.device) # Not recommended to redefine layers in forward
            # For now, we'll let it error if dimensions mismatch severely, or pass if fc handles it.
            # A more robust solution is to ensure adapter handles this.
            pass

        return {'combined_scores': torch.sigmoid(self.fc(x)), 'anomalies_detected': (torch.rand(x.size(0)) > 0.8).int()}


class DynamicPositionManager(nn.Module): # Placeholder
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
    def manage_positions(self, market_features, current_positions_data, market_state_info):
        # market_features: Tensor from market_data
        # current_positions_data: List of dicts representing current positions
        # market_state_info: Dict with market state context
        
        # Placeholder logic
        num_positions = len(current_positions_data)
        actions = []
        for i in range(num_positions):
            actions.append({'action': 'hold', 'confidence': torch.rand(1).item()})
        
        # Example: Use market_features if available and correctly dimensioned
        if market_features is not None and market_features.numel() > 0:
            if market_features.dim() == 3: # BxSxV
                market_features_processed = market_features.mean(dim=1) # Average over sequence
            elif market_features.dim() == 1: # V
                 market_features_processed = market_features.unsqueeze(0) # BxV
            else: # BxV
                market_features_processed = market_features

            # Dummy processing based on features
            if market_features_processed.size(-1) == self.feature_dim:
                # Example: if some feature mean is high, suggest buy for first action
                if actions and market_features_processed[0].mean() > 0.5:
                    actions[0]['action'] = 'buy'
            else:
                global_logger.warning(f"DynamicPositionManager: market_features dim {market_features_processed.size(-1)} != {self.feature_dim}")


        return {
            'suggested_actions': actions, 
            'risk_assessment': torch.rand(1).item(),
            'processed_positions_count': num_positions
        }

class EmergencyStopLoss(nn.Module): # Placeholder
    def __init__(self):
        super().__init__()
    def check_emergency_conditions(self, portfolio_metrics, current_drawdown, portfolio_risk):
        triggered = False
        reasons = []
        if current_drawdown is not None and current_drawdown > 0.2: # Example: 20% drawdown
            triggered = True
            reasons.append("High drawdown")
        if portfolio_risk is not None and portfolio_risk > 0.8: # Example: High portfolio risk
            triggered = True
            reasons.append("High portfolio risk")
        
        return {
            'emergency_triggered': triggered,
            'reasons': reasons,
            'recommended_actions': ['reduce_exposure'] if triggered else []
        }

class SystemState(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    EMERGENCY = "EMERGENCY"
    DEGRADED = "DEGRADED"

def ensure_sequence_dimension(data: Dict[str, Any], feature_key_prefix: str = 'features') -> Dict[str, Any]:
    """Ensures that relevant tensors in market_data have a sequence dimension."""
    for key, value in data.items():
        if key.startswith(feature_key_prefix) and torch.is_tensor(value):
            if value.dim() == 2:  # [batch, features]
                data[key] = value.unsqueeze(1)  # Add sequence dimension -> [batch, 1, features]
            elif value.dim() == 1: # [features]
                data[key] = value.unsqueeze(0).unsqueeze(0) # Add batch and sequence -> [1, 1, features]
    return data

def ensure_batch_dimension(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Ensures that a tensor has a batch dimension."""
    if tensor is None:
        return None
    if torch.is_tensor(tensor):
        if tensor.dim() == 0: # scalar
            return tensor.unsqueeze(0) # [1]
        if tensor.dim() == 1 and tensor.size(0) > 1: # [features] or [batch_or_seq]
             # Heuristic: if it's a common feature size, assume it's features, add batch
             # This is tricky. For now, let's assume if it's 1D, it's a batch of 1 element or needs a batch dim.
             # If it's like [N], it could be N features for 1 batch, or N batches of 1 feature.
             # Let's assume it's [features] and add a batch dim.
            return tensor.unsqueeze(0) # [1, features]
        # If 2D [batch, features] or [seq, features] or 3D [batch, seq, features], assume batch is present or is the first dim.
    return tensor


class HighLevelIntegrationSystem(nn.Module):
    """High-Level Integration System for Phase 5"""
    
    def __init__(
        self,
        strategy_innovation_module: nn.Module,
        market_state_awareness_system: nn.Module,
        meta_learning_optimizer: MetaLearningOptimizer, # Use the (potentially fallback) class
        position_manager: DynamicPositionManager, # Use the placeholder class
        anomaly_detector: AnomalyDetector, # Use the placeholder class
        emergency_stop_loss_system: EmergencyStopLoss, # Use the placeholder class
        enhanced_strategy_layer: Optional[nn.Module] = None, 
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        enable_logging: bool = True,
        log_level: int = logging.INFO,
        log_file: Optional[str] = None
    ):
        super().__init__()
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if enable_logging:
            if not self.logger.handlers: 
                self.logger.setLevel(log_level)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                ch = logging.StreamHandler()
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)
                if log_file:
                    fh = logging.FileHandler(log_file)
                    fh.setFormatter(formatter)
                    self.logger.addHandler(fh)
        else:
            if not self.logger.handlers: 
                self.logger.addHandler(logging.NullHandler())

        self.config = config if config is not None else self._get_default_config()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dim = self.config.get('feature_dim', 768) # Define feature_dim early

        self.strategy_innovation_module = strategy_innovation_module
        self.market_state_awareness = market_state_awareness_system
        self.meta_learning_optimizer = meta_learning_optimizer
        # Use passed-in instances directly
        self.position_manager = position_manager
        self.anomaly_detector = anomaly_detector
        self.emergency_stop_loss_system = emergency_stop_loss_system # Corrected name
        
        self.enhanced_strategy_layer = enhanced_strategy_layer

        self.enable_dynamic_adaptation = self.config.get('enable_dynamic_adaptation', True)
        self.adapter = None
        self.adapter_type = "none"

        if self.enable_dynamic_adaptation:
            if IntelligentDimensionAdapter is not None:
                self.adapter = IntelligentDimensionAdapter(
                    default_strategy=self.config.get('adapter_default_strategy', "smart_projection"),
                    enable_learning=self.config.get('adapter_enable_learning', True),
                    cache_size=self.config.get('adapter_cache_size', 100)
                )
                self.adapter_type = "intelligent"
                self.logger.info("Using IntelligentDimensionAdapter for dynamic adaptation.")
                # Register components with their expected input dimension ranges
                # These ranges are examples and should be defined based on actual component needs
                # Ensure ComponentSpec is available
                if ComponentSpec:
                    self.adapter.register_component("strategy_innovation", ComponentSpec(input_range=(self.config.get('strategy_input_min_dim', 256), self.config.get('strategy_input_max_dim',1024))))
                    self.adapter.register_component("market_state_awareness", ComponentSpec(input_range=(self.config.get('market_state_input_min_dim', 256), self.feature_dim)))
                    self.adapter.register_component("meta_learning_features", ComponentSpec(input_range=(self.config.get('meta_features_input_min_dim',128), self.config.get('meta_features_input_max_dim',768)))) # For meta_input features
                    self.adapter.register_component("meta_learning_task", ComponentSpec(input_range=(self.config.get('meta_task_input_min_dim',128), self.config.get('meta_task_input_max_dim',768)))) # For task_tensor
                    self.adapter.register_component("position_manager", ComponentSpec(input_range=(self.config.get('position_manager_input_min_dim',256), self.config.get('position_manager_input_max_dim',1024))))
                    self.adapter.register_component("anomaly_detector", ComponentSpec(input_range=(self.config.get('anomaly_input_min_dim',256), self.config.get('anomaly_input_max_dim',1024))))
                    self.adapter.register_component("emergency_stop_loss", ComponentSpec(input_range=(self.config.get('emergency_input_min_dim',1), self.config.get('emergency_input_max_dim',128)))) # e.g., for risk scores
                else:
                    self.logger.warning("ComponentSpec not available, cannot register components with IntelligentDimensionAdapter.")
            elif DynamicDimensionAdapter is not None: # Fallback to DynamicDimensionAdapter
                self.adapter = DynamicDimensionAdapter(
                    default_adapter_type=self.config.get('dynamic_adapter_type', "linear"),
                    enable_caching=self.config.get('dynamic_adapter_caching', True),
                    max_cache_size=self.config.get('dynamic_adapter_cache_size', 50)
                )
                self.adapter_type = "dynamic"
                self.logger.info("Using DynamicDimensionAdapter for dynamic adaptation.")
            else:
                self.logger.warning("No dimension adapter available (Intelligent or Dynamic). Direct tensor operations will be used.")
        
        # System monitoring
        self.system_state = SystemState.NORMAL
        self.active_alerts = []
        self.system_metrics_history = deque(maxlen=self.config.get('metrics_history_maxlen', 1000))
        self._maml_task_projector = nn.ModuleDict() # For specific MAML task tensor projections if needed outside adapter
        self.expected_maml_input_dim = self.config.get('expected_maml_input_dim', 768)
        
        # Initialize performance tracker
        self.performance_tracker: Dict[str, Any] = {
            'total_decisions': 0,
            'successful_adaptations': 0,
            'failed_adaptations': 0,
            'emergency_triggers': 0,
            'anomalies_detected': 0,
            'errors': 0,
            'processing_times': deque(maxlen=100) # Store recent processing times
        }

    def _get_default_config(self) -> Dict:
        return {
            "num_maml_tasks": 5,
            "maml_shots": 5,
            "expected_maml_input_dim": 768,
            "feature_dim": 768,
            "enable_dynamic_adaptation": True,
            "adapter_default_strategy": "smart_projection",
            "adapter_enable_learning": True,
            "adapter_cache_size": 100,
            "dynamic_adapter_type": "linear",
            "dynamic_adapter_caching": True,
            "dynamic_adapter_cache_size": 50,
            "metrics_history_maxlen": 1000,
            "default_input_tensor_key": "features_768", # Default key to look for in market_data
            # Component-specific dimension expectations (min, max)
            "strategy_input_min_dim": 256, "strategy_input_max_dim": 1024,
            "market_state_input_min_dim": 256, # market_state_input_max_dim will be self.feature_dim
            "meta_features_input_min_dim": 128, "meta_features_input_max_dim": 768,
            "meta_task_input_min_dim": 128, "meta_task_input_max_dim": 768,
            "position_manager_input_min_dim": 256, "position_manager_input_max_dim": 1024,
            "anomaly_input_min_dim": 256, "anomaly_input_max_dim": 1024,
            "emergency_input_min_dim": 1, "emergency_input_max_dim": 128,
        }

    def _get_tensor_from_market_data(self, market_data: Dict[str, Any], preferred_key: Optional[str] = None, default_dim: Optional[int] = None) -> Optional[torch.Tensor]:
        """Safely extracts a tensor from market_data, trying common keys."""
        keys_to_try = []
        if preferred_key:
            keys_to_try.append(preferred_key)
        
        # Add common variations based on self.feature_dim and other typical dimensions
        keys_to_try.extend([
            self.config.get('default_input_tensor_key', 'features_768'),
            f'features_{self.feature_dim}',
            'features',
            'input_tensor',
            'data',
            'features_256', # Common alternative
            'features_512', # Common alternative
            'raw_features'
        ])
        
        for key in keys_to_try:
            tensor = market_data.get(key)
            if torch.is_tensor(tensor):
                self.logger.debug(f"Found tensor with key '{key}', shape: {tensor.shape}")
                return tensor.to(self.device)
        
        self.logger.warning(f"No suitable tensor found in market_data with keys: {keys_to_try}. Checked data keys: {list(market_data.keys())}")
        if default_dim:
            self.logger.warning(f"Returning a random tensor of dim {default_dim} as fallback.")
            return torch.randn(1, default_dim, device=self.device) # Batch size 1
        return None

    def _adapt_tensor_if_needed(self, tensor: Optional[torch.Tensor], component_name: str, target_dim: Optional[int] = None) -> Optional[torch.Tensor]:
        """Adapts a tensor using self.adapter if available and tensor is valid."""
        if tensor is None:
            self.logger.warning(f"Cannot adapt None tensor for component {component_name}.")
            return None
        if not torch.is_tensor(tensor):
            self.logger.warning(f"Item for component {component_name} is not a tensor (type: {type(tensor)}). Cannot adapt.")
            return None
        if tensor.numel() == 0:
            self.logger.warning(f"Tensor for component {component_name} is empty. Cannot adapt.")
            return tensor # Return as is, or handle as error

        adapted_tensor = tensor
        if self.adapter:
            try:
                # The adapter's adapt_tensor might take target_shape or component_name
                # Assuming it can use component_name to look up spec, or target_dim for direct adaptation
                spec = self.adapter.get_component_spec(component_name)
                expected_dim_to_pass = target_dim
                if spec and not target_dim: # If spec exists and no specific target_dim is given, try to use spec's info
                    # This part depends on how IntelligentDimensionAdapter uses its spec.
                    # Let's assume for now it can infer from component_name or we pass a general target.
                    # If spec has a fixed target dim, use it. Otherwise, adapter might be flexible.
                    pass # Adapter will use its internal logic for the component

                adapted_tensor = self.adapter.adapt_tensor(tensor, component_name=component_name, target_dim=expected_dim_to_pass)
                if adapted_tensor.shape != tensor.shape:
                    self.logger.info(f"Tensor for {component_name} adapted from {tensor.shape} to {adapted_tensor.shape}.")
                self.performance_tracker['successful_adaptations'] = self.performance_tracker.get('successful_adaptations', 0) + 1
            except Exception as e:
                self.logger.error(f"Failed to adapt tensor for {component_name} (shape {tensor.shape}): {e}")
                self.performance_tracker['failed_adaptations'] = self.performance_tracker.get('failed_adaptations', 0) + 1
                # Fallback: return original tensor or handle error
                # adapted_tensor = tensor # Or raise
        return adapted_tensor


    def process_market_data(self, market_data_raw: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing pipeline for market data with dynamic dimension adaptation"""
        start_time = time.time()
        
        # Ensure market_data_raw is on the correct device if it contains tensors
        market_data = {}
        for k, v in market_data_raw.items():
            if torch.is_tensor(v):
                market_data[k] = v.to(self.device)
            else:
                market_data[k] = v
        
        # Extract common data structures (with fallbacks)
        # These might be tensors or other dicts/values
        portfolio_metrics = market_data.get('portfolio_metrics', {}) # Usually a dict
        position_data = market_data.get('current_positions', []) # Usually a list of dicts
        
        # Try to get a primary input tensor from market_data
        # This tensor will be adapted for various components
        main_input_tensor = self._get_tensor_from_market_data(market_data, default_dim=self.feature_dim)
        if main_input_tensor is None:
            self.logger.error("Critical: No main input tensor could be derived from market_data. Aborting processing.")
            self.performance_tracker['errors'] +=1
            return self._get_fallback_results("No main input tensor in market_data")

        # Ensure sequence dimension for relevant tensors if adapter expects it
        # This depends on the adapter's capabilities. Some adapters might handle this.
        # For now, let's assume main_input_tensor might need it for some components.
        if main_input_tensor.dim() == 2: # [B, F]
            main_input_tensor_seq = main_input_tensor.unsqueeze(1) # [B, 1, F]
        elif main_input_tensor.dim() == 1: # [F]
             main_input_tensor_seq = main_input_tensor.unsqueeze(0).unsqueeze(0) # [1, 1, F]
        else: # Already has 3+ dims or 0 dim (scalar, unlikely for features)
            main_input_tensor_seq = main_input_tensor
        
        try:
            # 1. Market State Analysis
            self.logger.debug(f"1. Market State Analysis - input tensor shape: {main_input_tensor_seq.shape}")
            market_state_input = self._adapt_tensor_if_needed(main_input_tensor_seq, 'market_state_awareness')
            if market_state_input is None: # Critical if adaptation fails and returns None
                self.logger.error("Market state input is None after adaptation attempt.")
                return self._get_fallback_results("Market state input adaptation failed")

            market_state_results = self.market_state_awareness(market_state_input)
            self.logger.debug(f"Market state results type: {type(market_state_results)}")
            
            # 2. Strategy Innovation
            self.logger.debug(f"2. Strategy Innovation - input tensor shape: {main_input_tensor_seq.shape}")
            strategy_input_tensor = self._adapt_tensor_if_needed(main_input_tensor_seq, 'strategy_innovation')
            if strategy_input_tensor is None:
                return self._get_fallback_results("Strategy innovation input adaptation failed")

            market_state_context = market_state_results.get('market_state', {}) # Extract context for strategy
            innovation_results = self.strategy_innovation_module(strategy_input_tensor, market_state_context)
            self.logger.debug(f"Innovation results type: {type(innovation_results)}")
            
            # 3. Meta-Learning Optimization
            self.logger.debug(f"3. Meta-Learning - calling _process_meta_learning")
            # _process_meta_learning handles its own input tensor adaptation from market_data and innovation_results
            meta_results = self._process_meta_learning(market_data, innovation_results, market_state_results)
            
            # 4. Anomaly Detection
            self.logger.debug(f"4. Anomaly Detection - input tensor shape: {main_input_tensor_seq.shape}")
            anomaly_input = self._adapt_tensor_if_needed(main_input_tensor_seq, 'anomaly_detector', target_dim=self.anomaly_detector.input_dim)
            if anomaly_input is None:
                 return self._get_fallback_results("Anomaly detection input adaptation failed")
            anomaly_results = self.anomaly_detector(anomaly_input)
            
            # 5. Position Management
            self.logger.debug(f"5. Position Management - using position_data and market_state_results")
            # Position manager might use its own features or a dedicated input
            pm_input_tensor = self._adapt_tensor_if_needed(main_input_tensor, 'position_manager', target_dim=self.position_manager.feature_dim)
            # If pm_input_tensor is None, position_manager's placeholder might handle it or use other data
            position_results = self.position_manager.manage_positions(
                market_features=pm_input_tensor, # Pass adapted tensor
                current_positions_data=position_data, 
                market_state_info=market_state_results 
            )
            
            # 6. Emergency Stop Loss
            self.logger.debug(f"6. Emergency Stop Loss - using portfolio_metrics")
            # This component typically uses high-level metrics, not raw tensors directly.
            # No tensor adaptation needed here unless it takes a specific risk tensor.
            # Assuming portfolio_metrics is a dict.
            current_drawdown = portfolio_metrics.get('current_drawdown', 0.0)
            portfolio_risk_value = portfolio_metrics.get('portfolio_risk', 0.0)
            emergency_results = self.emergency_stop_loss_system.check_emergency_conditions(
                portfolio_metrics, current_drawdown, portfolio_risk_value
            )
            
            # 7. System Health Assessment
            system_health = self._assess_system_health(
                market_state_results, innovation_results, meta_results,
                anomaly_results, position_results, emergency_results
            )
            
            # Update performance tracking
            self.performance_tracker['total_decisions'] += 1
            if isinstance(meta_results, dict) and meta_results.get('adaptation_quality', 0) > 0.7 :
                self.performance_tracker['successful_adaptations'] += 1 # This might be double counted with _adapt_tensor_if_needed            if isinstance(emergency_results, dict) and emergency_results.get('emergency_triggered', False):
                self.performance_tracker['emergency_triggers'] += 1
            if isinstance(anomaly_results, dict):
                anomalies_count = anomaly_results.get('anomalies_detected', 0)
                # Handle tensor vs scalar comparison properly
                if torch.is_tensor(anomalies_count):
                    # For tensor values, check if any anomalies detected
                    if anomalies_count.numel() > 1:
                        # Multiple values - check if any are > 0
                        has_anomalies = (anomalies_count > 0).any().item()
                    else:
                        # Single value tensor
                        has_anomalies = anomalies_count.item() > 0
                    
                    if has_anomalies:
                        self.performance_tracker['anomalies_detected'] += anomalies_count.sum().item()
                else:
                    # Scalar value
                    if anomalies_count > 0:
                        self.performance_tracker['anomalies_detected'] += anomalies_count


            # Compile results
            current_processing_time = time.time() - start_time
            self.performance_tracker['processing_times'].append(current_processing_time)

            results = {
                'market_state': market_state_results,
                'strategy_innovation': innovation_results,
                'meta_learning': meta_results,
                'anomaly_detection': anomaly_results,
                'position_management': position_results,
                'emergency_status': emergency_results,
                'system_health': system_health,
                'processing_time': current_processing_time,
                'adaptation_stats': self.adapter.get_adaptation_stats() if self.adapter else {}
            }
            self._check_conditions_and_generate_alerts(results)
            self._update_system_metrics(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in process_market_data: {e}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            self.performance_tracker['errors'] +=1
            return self._get_fallback_results(str(e))

    def _get_fallback_results(self, error_message: str) -> Dict[str, Any]:
        self.logger.error(f"Returning fallback results due to error: {error_message}")
        return {
            'error': error_message,
            'system_health': {'overall_health': 0.0, 'system_state': SystemState.EMERGENCY.value, 'component_health': {}, 'issues': [error_message]},
            'market_state': {'current_state': 'unknown', 'confidence': 0.0, 'error': error_message},
            'strategy_innovation': {'generated_strategies': torch.empty(0, device=self.device), 'innovation_confidence': 0.0, 'error': error_message},
            'meta_learning': {'adapted_features': torch.empty(0, device=self.device), 'adaptation_quality': 0.0, 'error': error_message},
            'anomaly_detection': {'combined_scores': torch.empty(0, device=self.device), 'anomalies_detected': 0, 'error': error_message},
            'position_management': {'suggested_actions': [], 'error': error_message},
            'emergency_status': {'emergency_triggered': True, 'reasons': [error_message], 'error': error_message},
            'processing_time': 0.0,
            'adaptation_stats': {}
        }

    def _process_strategy_innovation(self, market_data: Dict, market_state_results: Dict) -> Dict:
        """
        DEPRECATED/REPLACED by inline logic in process_market_data.
        This method is kept for reference or if a separate processing step is reinstated.
        """
        self.logger.warning("_process_strategy_innovation is deprecated and was called. Logic should be in process_market_data.")
        # Simplified version, assuming main_input_tensor is passed or extracted
        main_input_tensor = self._get_tensor_from_market_data(market_data, default_dim=self.feature_dim)
        if main_input_tensor is None: return {'error': "No input tensor for strategy innovation"}

        strategy_input_tensor = self._adapt_tensor_if_needed(main_input_tensor, 'strategy_innovation')
        if strategy_input_tensor is None: return {'error': "Strategy input adaptation failed"}
        
        market_state_context = market_state_results.get('market_state', {})
        return self.strategy_innovation_module(strategy_input_tensor, market_state_context)


    def _process_meta_learning(self, market_data: Dict, innovation_results: Dict, market_state_results: Dict) -> Dict:
        """Process meta-learning with robust tensor handling and adaptation."""
        try:
            # 1. Get Task Tensor from Innovation Results
            # 'processed_for_maml_task_tensor' or 'generated_strategies' or similar
            task_tensor_source = innovation_results.get('processed_for_maml_task_tensor', 
                                                      innovation_results.get('generated_strategies'))
            
            if not torch.is_tensor(task_tensor_source) or task_tensor_source.numel() == 0:
                self.logger.warning("No valid task tensor in innovation_results for meta-learning. Using fallback random tensor.")
                num_tasks_fallback = self.config.get('num_maml_tasks', 5)
                shots_fallback = self.config.get('maml_shots', 5)
                task_tensor_source = torch.randn(num_tasks_fallback, shots_fallback * 2, self.expected_maml_input_dim, device=self.device)

            # Adapt task_tensor_source for meta-learning component (e.g. MAML)
            # This ensures it has the right dimension for _create_task_batches / MAML optimizer
            task_tensor_for_batches = self._adapt_tensor_if_needed(task_tensor_source, 'meta_learning_task', target_dim=self.expected_maml_input_dim)
            if task_tensor_for_batches is None:
                 self.logger.error("Task tensor for batches is None after adaptation. Cannot proceed with meta-learning.")
                 return {'error': "Task tensor adaptation failed", 'adaptation_quality': 0.0, 'meta_loss': -1.0}


            # Create TaskBatch objects
            task_batches = self._create_task_batches(task_tensor_for_batches)
            if not task_batches:
                self.logger.warning("No task batches created for meta-learning. This might be due to input tensor issues.")
                # Depending on strictness, either return error or allow optimizer to handle empty task_batches if it can
                # For now, let optimizer decide. Some optimizers might still run with empty task_batches (e.g., just update based on features).

            # 2. Get Features Tensor from Market Data
            # This is the primary input features for the meta-learning optimizer's main adaptation pass
            meta_input_features_raw = self._get_tensor_from_market_data(market_data, preferred_key='features_for_meta', default_dim=self.expected_maml_input_dim)
            if meta_input_features_raw is None:
                self.logger.error("No suitable features tensor found in market_data for meta-learning main input.")
                return {'error': "Meta input features not found", 'adaptation_quality': 0.0, 'meta_loss': -1.0}

            meta_input_features = self._adapt_tensor_if_needed(meta_input_features_raw, 'meta_learning_features', target_dim=self.expected_maml_input_dim)
            if meta_input_features is None:
                self.logger.error("Meta input features are None after adaptation. Cannot proceed.")
                return {'error': "Meta input features adaptation failed", 'adaptation_quality': 0.0, 'meta_loss': -1.0}

            # Ensure meta_input_features is 2D [batch_size, feature_dim] or 3D [batch_size, seq_len, feature_dim]
            # The MLOptimizer might expect a certain shape.
            if meta_input_features.dim() == 3 and meta_input_features.size(1) > 1: # If [B, S, F] with S > 1
                 # Example: average over sequence length if optimizer expects [B, F]
                 # This depends on the specific MLOptimizer's design
                 # meta_input_features = meta_input_features.mean(dim=1) 
                 self.logger.debug(f"Meta input features for MLO has shape {meta_input_features.shape}. Passing as is.")
            elif meta_input_features.dim() == 1: # [F]
                meta_input_features = meta_input_features.unsqueeze(0) # [1, F]


            # 3. Call Meta-Learning Optimizer
            # The 'context' argument can be market_state_results or other relevant context
            meta_context = market_state_results # Or a more specific context tensor derived from it

            meta_results = self.meta_learning_optimizer.optimize_and_adapt(
                features=meta_input_features, 
                context=meta_context,
                task_batches=task_batches
            )
            return meta_results
            
        except Exception as e:
            self.logger.error(f"Meta-learning processing failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback: use the raw market data tensor if adaptation failed, or a zero tensor
            fallback_features = self._get_tensor_from_market_data(market_data, default_dim=self.expected_maml_input_dim)
            if fallback_features is None: fallback_features = torch.zeros(1, self.expected_maml_input_dim, device=self.device)

            return {
                'adapted_features': fallback_features, 
                'adaptation_quality': 0.0, # Indicate failure
                'meta_loss': -1.0, # Indicate error
                'error': str(e)
            }

    def _create_task_batches(self, task_tensor: torch.Tensor) -> List[TaskBatch]:
        task_batches = []
        if task_tensor is None or task_tensor.numel() == 0:
            self.logger.warning(f"⚠️ Empty or None tensor for task batch creation (shape: {task_tensor.shape if task_tensor is not None else 'None'}). Cannot create batches.")
            return []

        # Ensure task_tensor is 3D: [num_tasks, samples_per_task, feature_dim]
        # Adapter should have ideally handled this to match 'meta_learning_task' spec.
        if task_tensor.dim() == 2: # If [samples, features], assume 1 task
            self.logger.debug(f"Task tensor is 2D ({task_tensor.shape}), unsqueezing to 3D for 1 task.")
            task_tensor = task_tensor.unsqueeze(0) 
        elif task_tensor.dim() != 3:
            self.logger.error(f"Task tensor has unexpected dimension {task_tensor.dim()} (shape: {task_tensor.shape}), expected 3. Skipping batch creation.")
            return []

        num_tasks = task_tensor.size(0)
        samples_per_task = task_tensor.size(1)
        # feature_dim_from_tensor = task_tensor.size(2) # Can be used for validation

        shots = self.config.get('maml_shots', 5) # k-shot

        if samples_per_task < shots * 2: # Need k for support, k for query
            self.logger.warning(f"Insufficient samples per task ({samples_per_task}) for {shots} shots (requires {shots*2}). Will attempt to use fewer shots or skip.")
            if samples_per_task < 2 : # Cannot form even one support and one query
                self.logger.error(f"Cannot form task batches with {samples_per_task} samples per task. Skipping all tasks for this tensor.")
                return []
            current_shots = samples_per_task // 2 # Use as many shots as possible
            if current_shots == 0: # Still not enough after integer division
                 self.logger.error(f"Calculated current_shots is 0 with {samples_per_task} samples. Skipping.")
                 return []
            self.logger.info(f"Adjusted shots to {current_shots} due to insufficient samples.")
        else:
            current_shots = shots

        for i in range(num_tasks):
            try:
                current_task_data = task_tensor[i] # Shape: [samples_per_task, feature_dim]
                
                # Ensure we have enough data for the adjusted current_shots
                if current_task_data.size(0) < current_shots * 2:
                    self.logger.warning(f"Task {i}: Still not enough samples ({current_task_data.size(0)}) for {current_shots} support/query pairs. Skipping task.")
                    continue
                
                support_data = current_task_data[:current_shots]
                query_data = current_task_data[current_shots : current_shots * 2]
                
                if support_data.size(0) == 0 or query_data.size(0) == 0:
                    self.logger.warning(f"Task {i}: Support or query data is empty after slicing with {current_shots} shots (support: {support_data.shape}, query: {query_data.shape}). Skipping task.")
                    continue

                # Create dummy labels for now. In a real scenario, these would be meaningful.
                # Assuming binary classification for simplicity.
                support_labels = torch.randint(0, 2, (support_data.size(0),), device=self.device).float()
                query_labels = torch.randint(0, 2, (query_data.size(0),), device=self.device).float()

                task_batch = TaskBatch(
                    support_data=support_data,
                    support_labels=support_labels,
                    query_data=query_data,
                    query_labels=query_labels,
                    task_id=f"task_{i}",
                    market_state="unknown_during_batch_creation", # This could be enriched if context is passed
                    difficulty=torch.rand(1).item() # Random difficulty
                )
                task_batches.append(task_batch)
            except Exception as e:
                self.logger.error(f"Failed to create task batch {i} from tensor slice of shape {current_task_data.shape if 'current_task_data' in locals() else 'unknown'}: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        if not task_batches:
            self.logger.warning("Finished _create_task_batches, but no batches were successfully created.")
        else:
            self.logger.info(f"Successfully created {len(task_batches)} task batches.")
        return task_batches

    def _manage_positions(self, market_features: Optional[torch.Tensor], current_positions: List[Dict], market_state_info: Dict) -> Dict:
        """
        Manage positions using the DynamicPositionManager.
        This method is a wrapper and might be deprecated if direct call to self.position_manager.manage_positions is preferred.
        """
        self.logger.debug(f"Calling position_manager.manage_positions with {len(current_positions)} current positions.")
        try:
            # The DynamicPositionManager's manage_positions method is expected to be called directly.
            # market_features should already be adapted by _adapt_tensor_if_needed in process_market_data
            # and passed to self.position_manager.manage_positions
            # This wrapper might be redundant if process_market_data calls position_manager directly.
            # For now, assuming it's called from somewhere or for clarity.
            
            # If this _manage_positions is still intended to be used as a separate step:
            # 1. Ensure market_features are adapted if not already
            # (already done if called from process_market_data which prepares pm_input_tensor)

            # 2. Call the actual position manager
            position_management_results = self.position_manager.manage_positions(
                market_features=market_features, # This should be the adapted tensor for position manager
                current_positions_data=current_positions,
                market_state_info=market_state_info
            )
            
            self.logger.info(f"Position management results: {position_management_results.get('suggested_actions')}")
            return position_management_results

        except Exception as e:
            self.logger.error(f"Position management failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'error': str(e), 
                'suggested_actions': [], 
                'risk_assessment': 1.0 # Max risk on error
            }


    def _check_emergency_conditions(self, portfolio_metrics: Dict, current_drawdown: float, portfolio_risk: float) -> Dict:
        """
        Check emergency conditions using the EmergencyStopLoss system.
        This is largely a wrapper.
        """
        self.logger.debug(f"Checking emergency conditions. Drawdown: {current_drawdown}, Risk: {portfolio_risk}")
        try:
            # Call the actual emergency stop loss system
            # This is already done in process_market_data. This method might be redundant.
            emergency_response = self.emergency_stop_loss_system.check_emergency_conditions(
                portfolio_metrics=portfolio_metrics,
                current_drawdown=current_drawdown,
                portfolio_risk=portfolio_risk
            )
            
            if emergency_response.get('emergency_triggered', False):
                self.logger.warning(f"Emergency condition triggered: {emergency_response.get('reasons')}")
            else:
                self.logger.debug("No emergency conditions triggered.")
            
            return emergency_response

        except Exception as e:
            self.logger.error(f"Emergency condition checking failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'emergency_triggered': True, # Assume emergency on error
                'reasons': [f"Error in checking: {str(e)}"],
                'recommended_actions': ['MANUAL_INTERVENTION_REQUIRED'],
                'error': str(e)
            }
            
    def _handle_emergency_conditions(self, emergency_status_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles actions based on emergency status.
        Placeholder - actual emergency handling logic (e.g., trade execution, alerts) would go here.
        """
        actions_taken_log = []
        emergency_triggered = emergency_status_results.get('emergency_triggered', False)
        recommended_actions = emergency_status_results.get('recommended_actions', [])

        if emergency_triggered:
            self.logger.critical(f"EMERGENCY TRIGGERED. Reasons: {emergency_status_results.get('reasons')}. Recommended: {recommended_actions}")
            self.system_state = SystemState.EMERGENCY
            # Example: Log actions that would be taken
            for action in recommended_actions:
                log_message = f"Executing emergency action: {action}"
                self.logger.info(log_message)
                actions_taken_log.append(log_message)
            # In a real system, this would interface with an execution module.
        else:
            self.logger.info("No emergency conditions to handle.")
        
        return {
            "emergency_response_initiated": emergency_triggered,
            "actions_taken_log": actions_taken_log, # Log of what would be done
            "final_system_state_after_handling": self.system_state.value
        }

    def _assess_system_health(
        self,
        market_state_results: Optional[Dict],
        innovation_results: Optional[Dict],
        meta_results: Optional[Dict],
        anomaly_results: Optional[Dict],
        position_results: Optional[Dict], # Renamed from position_results_input
        emergency_status_input: Optional[Dict] # Renamed from emergency_status
    ) -> Dict[str, Any]:
        """Assess overall system health based on component outputs."""
        component_health_scores = {}
        issues = []

        # Helper to safely get scores and handle missing data or errors
        def get_score(data: Optional[Dict], key: str, default_healthy: float = 1.0, default_unhealthy: float = 0.0, is_error_key: str = 'error') -> float:
            if data is None:
                issues.append(f"Data for component related to '{key}' is None.")
                return default_unhealthy
            if is_error_key in data and data[is_error_key] is not None:
                issues.append(f"Error in component providing '{key}': {data[is_error_key]}")
                return default_unhealthy
            
            val = data.get(key)
            if val is None:
                # issues.append(f"Metric '{key}' not found in data.") # Too verbose for every missing metric
                return (default_healthy + default_unhealthy) / 2 # Neutral if metric simply missing
            
            if torch.is_tensor(val):
                if val.numel() == 0: return default_unhealthy # Empty tensor
                try:
                    # Attempt to convert to a scalar float. Handle different tensor shapes.
                    if val.numel() == 1:
                        scalar_val = val.item()
                    elif val.dim() > 0 : # For multi-element tensors, take mean or other aggregate
                        scalar_val = val.float().mean().item()
                    else: # Should not happen if numel > 0 and dim == 0
                        scalar_val = val.item() # Try item() for 0-dim tensor
                    return float(scalar_val)
                except Exception as e:
                    issues.append(f"Could not convert tensor metric '{key}' (shape {val.shape}) to scalar: {e}")
                    return default_unhealthy
            elif isinstance(val, (int, float)):
                return float(val)
            else:
                # issues.append(f"Metric '{key}' is of unexpected type: {type(val)}.") # Verbose
                return (default_healthy + default_unhealthy) / 2 

        # Market State Health (e.g., based on confidence, inverse of uncertainty)
        component_health_scores['market_state'] = get_score(market_state_results, 'confidence', default_unhealthy=0.1)
        if market_state_results and market_state_results.get('uncertainty') is not None:
             uncertainty = get_score(market_state_results, 'uncertainty')
             component_health_scores['market_state'] *= (1 - min(max(uncertainty, 0), 1)) # Invert uncertainty


        # Strategy Innovation Health (e.g., confidence, diversity)
        component_health_scores['strategy_innovation'] = get_score(innovation_results, 'innovation_confidence', default_unhealthy=0.1)
        # Could add diversity score if available: get_score(innovation_results, 'strategy_diversity')

        # Meta-Learning Health (e.g., adaptation quality, inverse of loss)
        component_health_scores['meta_learning'] = get_score(meta_results, 'adaptation_quality', default_unhealthy=0.0)
        if meta_results and meta_results.get('meta_loss') is not None:
            meta_loss = get_score(meta_results, 'meta_loss')
            if meta_loss > 0: # Avoid division by zero or negative loss issues
                component_health_scores['meta_learning'] *= (1 / (1 + meta_loss)) # Lower loss is better        # Anomaly Detection Health (e.g., inverse of anomaly score if high means bad, or based on detection rates)
        # Assuming lower combined_scores is better (less anomalous)
        anomaly_score = get_score(anomaly_results, 'combined_scores', default_healthy=0.0, default_unhealthy=1.0) # Inverted logic for score
        component_health_scores['anomaly_detection'] = 1.0 - anomaly_score 
        if anomaly_results and anomaly_results.get('anomalies_detected', 0) is not None:
            anomalies_detected = anomaly_results.get('anomalies_detected', 0)
            # Handle tensor values properly
            if isinstance(anomalies_detected, torch.Tensor):
                if anomalies_detected.numel() == 1:
                    anomalies_count = anomalies_detected.item()
                else:
                    anomalies_count = anomalies_detected.sum().item()  # Sum all detected anomalies
            else:
                anomalies_count = anomalies_detected
                
            if anomalies_count > 0:
                issues.append(f"Anomalies detected: {anomalies_count}")
                # Reduce health if anomalies are high, this is a simple heuristic
                component_health_scores['anomaly_detection'] *= 0.8


        # Position Management Health (e.g., based on risk assessment, successful actions)
        # Assuming 'risk_assessment' in position_results is 0=low risk, 1=high risk
        pm_risk = get_score(position_results, 'risk_assessment', default_healthy=0.0, default_unhealthy=1.0)
        component_health_scores['position_management'] = 1.0 - pm_risk
        if position_results and position_results.get('error'):
            component_health_scores['position_management'] = 0.0


        # Emergency Status (not a score, but an indicator)
        is_emergency = False
        if emergency_status_input:
            component_health_scores['emergency_check'] = 0.0 if emergency_status_input.get('emergency_triggered') else 1.0
            if emergency_status_input.get('emergency_triggered'):
                is_emergency = True
                issues.append(f"Emergency triggered: {emergency_status_input.get('reasons', 'No specific reason')}")
        else:
            component_health_scores['emergency_check'] = 0.5 # Unknown
            issues.append("Emergency status data not available.")


        # Overall Health Calculation (e.g., weighted average, or minimum)
        # Simple average for now, excluding emergency_check from direct average unless it's 0
        valid_scores = [s for s in component_health_scores.values() if s is not None and s != component_health_scores.get('emergency_check')]
        overall_health = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        # Adjust overall health based on critical failures or emergency
        if is_emergency:
            overall_health *= 0.3 # Drastically reduce health if emergency
            self.system_state = SystemState.EMERGENCY
        elif any(s < 0.3 for s in valid_scores): # If any component is very unhealthy
            overall_health *= 0.7
            self.system_state = SystemState.DEGRADED
        elif any(s < 0.6 for s in valid_scores):
            self.system_state = SystemState.WARNING
        else:
            self.system_state = SystemState.NORMAL
            
        # Ensure overall_health is between 0 and 1
        overall_health = min(max(overall_health, 0.0), 1.0)

        # Log final health assessment
        self.logger.info(f"System Health Assessment: Overall={overall_health:.2f}, State={self.system_state.value}, Components={component_health_scores}, Issues: {issues}")

        return {
            'overall_health': overall_health,
            'system_state': self.system_state.value,
            'component_health': component_health_scores,
            'issues': issues # List of identified issues/warnings
        }

    def _update_system_metrics(self, results: Dict[str, Any]):
        """Placeholder for updating detailed system metrics based on processing results."""
        self.logger.debug("Updating system metrics (placeholder).")
        # Example: Store current results or extracted metrics in self.system_metrics_history
        metrics_to_store = {
            'timestamp': time.time(),
            'overall_health': results.get('system_health', {}).get('overall_health'),
            'processing_time': results.get('processing_time'),
            'meta_loss': results.get('meta_learning', {}).get('meta_loss'),
            'anomalies': results.get('anomaly_detection', {}).get('anomalies_detected')
        }
        self.system_metrics_history.append(metrics_to_store)
        # Further logic for trend analysis, etc. could be added here.

    def _check_conditions_and_generate_alerts(self, results: Dict[str, Any]):
        """Placeholder for checking conditions and generating alerts."""
        self.logger.debug("Checking conditions and generating alerts (placeholder).")
        self.active_alerts = [] # Clear previous alerts or manage them

        system_health_info = results.get('system_health', {})
        if system_health_info.get('system_state') == SystemState.EMERGENCY.value:
            self.active_alerts.append({
                'type': 'EMERGENCY', 
                'message': f"System in emergency state. Issues: {system_health_info.get('issues')}"
            })
        elif system_health_info.get('system_state') == SystemState.WARNING.value:
             self.active_alerts.append({
                'type': 'WARNING', 
                'message': f"System in warning state. Issues: {system_health_info.get('issues')}"
            })
        
        if results.get('emergency_status', {}).get('emergency_triggered'):
             self.active_alerts.append({
                'type': 'CRITICAL', 
                'message': f"Emergency stop-loss triggered. Reasons: {results.get('emergency_status', {}).get('reasons')}"
            })

        if self.active_alerts:
            for alert in self.active_alerts:
                self.logger.warning(f"ALERT: Type: {alert['type']}, Message: {alert['message']}")
        # Further logic for sending notifications could be added here.

    def get_system_status(self) -> Dict[str, Any]:
        """Returns the current operational status of the system."""
        avg_processing_time = 0
        if self.performance_tracker['processing_times']:
            avg_processing_time = sum(self.performance_tracker['processing_times']) / len(self.performance_tracker['processing_times'])
        
        return {
            "system_state": self.system_state.value,
            "active_alerts": self.active_alerts,
            "performance_summary": self.performance_tracker,
            "avg_processing_time_ms": avg_processing_time * 1000,
            "adapter_type": self.adapter_type,
            "device": str(self.device),
            "recent_metrics": list(self.system_metrics_history)[-5:] # Last 5 metrics history
        }

    def to(self, device, *args, **kwargs):
        """Override .to() to also move registered adapters and projectors if they are nn.Module."""
        super().to(device, *args, **kwargs)
        self.device = device # Update self.device tracking

        if hasattr(self, 'adapter') and isinstance(self.adapter, nn.Module):
            self.adapter.to(device)
            self.logger.info(f"Moved adapter ({self.adapter_type}) to {device}")

        if hasattr(self, '_maml_task_projector') and isinstance(self._maml_task_projector, nn.ModuleDict):
            for key in self._maml_task_projector:
                self._maml_task_projector[key].to(device)
            self.logger.info(f"Moved _maml_task_projector components to {device}")
        
        # Ensure all sub-modules passed in __init__ are also moved if not handled by super().to()
        # This is usually handled if they are direct attributes and nn.Modules
        # self.strategy_innovation_module.to(device)
        # self.market_state_awareness.to(device)
        # self.meta_learning_optimizer.to(device)
        # self.position_manager.to(device) # If it's an nn.Module
        # self.anomaly_detector.to(device) # If it's an nn.Module
        # self.emergency_stop_loss_system.to(device) # If it's an nn.Module
        # if self.enhanced_strategy_layer: self.enhanced_strategy_layer.to(device)
        return self
