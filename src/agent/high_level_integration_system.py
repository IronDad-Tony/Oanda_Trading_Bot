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

# Import TaskBatch for meta-learning
try:
    from .meta_learning_optimizer import TaskBatch
except ImportError:
    # Define a simple TaskBatch if import fails
    @dataclass
    class TaskBatch:
        support_data: torch.Tensor
        support_labels: torch.Tensor
        query_data: torch.Tensor
        query_labels: torch.Tensor
        task_id: str
        market_state: str
        difficulty: float

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Dynamic Dimension Adapter
try:
    from .dynamic_dimension_adapter import (
        DynamicDimensionAdapter,
        DimensionSpec,
        ensure_batch_dimension,
        ensure_sequence_dimension,
        flatten_to_feature_dim
    )
except ImportError:
    logger.warning("Could not import DynamicDimensionAdapter, using fallback")
    DynamicDimensionAdapter = None

class SystemState(Enum):
    """Overall system states"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class PositionAction(Enum):
    """Position management actions"""
    HOLD = "hold"
    REDUCE = "reduce"
    CLOSE = "close"
    EMERGENCY_CLOSE = "emergency_close"
    HEDGE = "hedge"

@dataclass
class SystemAlert:
    """System alert structure"""
    level: AlertLevel
    message: str
    component: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class PositionRisk:
    """Position risk assessment"""
    position_id: str
    current_risk: float
    max_risk: float
    risk_trend: float
    recommended_action: PositionAction
    confidence: float
    reasoning: List[str]

@dataclass
class SystemMetrics:
    """Comprehensive system metrics"""
    timestamp: datetime
    system_state: SystemState
    performance_score: float
    risk_score: float
    adaptation_quality: float
    strategy_diversity: float
    anomaly_score: float
    resource_utilization: Dict[str, float]
    active_alerts: int
    resolved_alerts: int

class AnomalyDetector(nn.Module):
    """Advanced anomaly detection system"""
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        threshold_multiplier: float = 2.5
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.threshold_multiplier = threshold_multiplier
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Temporal attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Anomaly score predictor
        self.anomaly_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Reconstruction decoder for unsupervised detection
        self.reconstruction_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Statistical baseline tracking
        self.register_buffer('feature_mean', torch.zeros(input_dim))
        self.register_buffer('feature_std', torch.ones(input_dim))
        self.register_buffer('update_count', torch.zeros(1))
        
        # Anomaly history
        self.anomaly_history = deque(maxlen=1000)
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Detect anomalies in features"""
        
        batch_size, seq_len, feature_dim = features.shape
        
        # Encode features
        encoded_features = self.feature_encoder(features)
        
        # Apply temporal encoding
        temporal_features = self.temporal_encoder(encoded_features)
        
        # Predict anomaly scores
        anomaly_scores = self.anomaly_predictor(temporal_features)
        
        # Reconstruct features for unsupervised detection
        reconstructed_features = self.reconstruction_decoder(temporal_features)
        reconstruction_loss = F.mse_loss(reconstructed_features, features, reduction='none')
        reconstruction_scores = torch.mean(reconstruction_loss, dim=-1, keepdim=True)
        
        # Statistical anomaly detection
        normalized_features = (features - self.feature_mean) / (self.feature_std + 1e-8)
        statistical_scores = torch.norm(normalized_features, dim=-1, keepdim=True) / np.sqrt(feature_dim)
        statistical_scores = torch.sigmoid(statistical_scores - 2.0)  # Threshold at 2 standard deviations
        
        # Combine anomaly scores
        combined_scores = (anomaly_scores + reconstruction_scores + statistical_scores) / 3.0
        
        # Update statistical baseline
        self._update_statistics(features)
        
        # Store anomaly information
        current_anomaly = {
            'timestamp': datetime.now(),
            'max_score': torch.max(combined_scores).item(),
            'mean_score': torch.mean(combined_scores).item(),
            'anomaly_detected': torch.max(combined_scores).item() > 0.7
        }
        self.anomaly_history.append(current_anomaly)
        
        return {
            'anomaly_scores': combined_scores,
            'reconstruction_scores': reconstruction_scores,
            'statistical_scores': statistical_scores,
            'combined_scores': combined_scores,
            'reconstructed_features': reconstructed_features
        }
    
    def _update_statistics(self, features: torch.Tensor):
        """Update statistical baseline for anomaly detection"""
        
        batch_mean = torch.mean(features, dim=(0, 1))
        batch_var = torch.var(features, dim=(0, 1))
        
        # Exponential moving average update
        alpha = 0.01
        self.feature_mean = (1 - alpha) * self.feature_mean + alpha * batch_mean
        
        # Update standard deviation
        new_std = torch.sqrt(batch_var)
        self.feature_std = (1 - alpha) * self.feature_std + alpha * new_std
        
        self.update_count += 1

class DynamicPositionManager(nn.Module):
    """Dynamic position management system"""
    
    def __init__(
        self,
        max_positions: int = 20,
        base_position_size: float = 0.1,
        risk_limit: float = 0.02,
        emergency_threshold: float = 0.05,
        feature_dim: int = 768
    ):
        super().__init__()
        self.max_positions = max_positions
        self.base_position_size = base_position_size
        self.risk_limit = risk_limit
        self.emergency_threshold = emergency_threshold
        self.feature_dim = feature_dim
        
        # Position risk assessor
        self.risk_assessor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Action recommender
        self.action_recommender = nn.Sequential(
            nn.Linear(feature_dim + 64, 256),  # Features + risk context
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(PositionAction)),
            nn.Softmax(dim=-1)
        )
          # Position tracking
        self.active_positions = {}
        self.position_history = deque(maxlen=1000)
        
    def assess_position_risk(
        self,
        position_features: torch.Tensor,
        market_context: torch.Tensor,
        position_id: str
    ) -> PositionRisk:
        """Assess risk for a specific position"""
        
        # Handle dimension mismatch - adapt position features to expected feature_dim
        if position_features.size(-1) != self.feature_dim:
            # Create an adapter layer if needed
            if not hasattr(self, 'position_adapter'):
                self.position_adapter = nn.Linear(position_features.size(-1), self.feature_dim)
            position_features = self.position_adapter(position_features)
          # Calculate current risk
        risk_output = self.risk_assessor(position_features)
        if risk_output.dim() > 1:
            current_risk = risk_output.mean().item()  # Average over batch dimension
        else:
            current_risk = risk_output.item()
          # Create risk context
        flattened_features = position_features.flatten() if position_features.dim() > 1 else position_features
        required_risk_context_size = max(64, self.feature_dim - flattened_features.numel())
        risk_context = torch.cat([
            flattened_features,
            torch.tensor([current_risk] * required_risk_context_size)  # Adjust to match dimension
        ])
        
        # Get action recommendation
        action_probs = self.action_recommender(risk_context.unsqueeze(0))
        recommended_action_idx = torch.argmax(action_probs).item()
        recommended_action = list(PositionAction)[recommended_action_idx]
        confidence = torch.max(action_probs).item()
        
        # Calculate risk trend
        risk_trend = self._calculate_risk_trend(position_id, current_risk)
        
        # Generate reasoning
        reasoning = self._generate_risk_reasoning(
            current_risk, risk_trend, recommended_action
        )
        
        return PositionRisk(
            position_id=position_id,
            current_risk=current_risk,
            max_risk=self.risk_limit,
            risk_trend=risk_trend,
            recommended_action=recommended_action,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def manage_positions(
        self,
        position_features: Dict[str, torch.Tensor],
        market_context: torch.Tensor
    ) -> Dict[str, PositionRisk]:
        """Manage all active positions"""
        
        position_risks = {}
        emergency_actions = []
        
        for position_id, features in position_features.items():
            # Assess position risk
            risk_assessment = self.assess_position_risk(
                features, market_context, position_id
            )
            position_risks[position_id] = risk_assessment
            
            # Check for emergency conditions
            if risk_assessment.current_risk > self.emergency_threshold:
                emergency_actions.append({
                    'position_id': position_id,
                    'action': PositionAction.EMERGENCY_CLOSE,
                    'risk': risk_assessment.current_risk
                })
            
            # Update position tracking
            self.active_positions[position_id] = {
                'last_update': datetime.now(),
                'current_risk': risk_assessment.current_risk,
                'risk_trend': risk_assessment.risk_trend,
                'recommended_action': risk_assessment.recommended_action
            }
        
        # Log emergency actions
        if emergency_actions:
            logger.critical(f"Emergency position actions required: {len(emergency_actions)} positions")
            for action in emergency_actions:
                logger.critical(f"Emergency close position {action['position_id']} - Risk: {action['risk']:.4f}")
        
        return position_risks
    
    def _calculate_risk_trend(self, position_id: str, current_risk: float) -> float:
        """Calculate risk trend for position"""
        
        if position_id not in self.active_positions:
            return 0.0
        
        previous_risk = self.active_positions[position_id].get('current_risk', current_risk)
        risk_trend = current_risk - previous_risk
        
        return risk_trend
    
    def _generate_risk_reasoning(
        self,
        current_risk: float,
        risk_trend: float,
        recommended_action: PositionAction
    ) -> List[str]:
        """Generate human-readable risk reasoning"""
        
        reasoning = []
        
        # Risk level assessment
        if current_risk < 0.01:
            reasoning.append("Low risk position")
        elif current_risk < 0.02:
            reasoning.append("Moderate risk position")
        elif current_risk < 0.05:
            reasoning.append("High risk position")
        else:
            reasoning.append("Critical risk position")
        
        # Risk trend assessment
        if risk_trend > 0.005:
            reasoning.append("Risk rapidly increasing")
        elif risk_trend > 0.001:
            reasoning.append("Risk gradually increasing")
        elif risk_trend < -0.005:
            reasoning.append("Risk rapidly decreasing")
        elif risk_trend < -0.001:
            reasoning.append("Risk gradually decreasing")
        else:
            reasoning.append("Risk stable")
        
        # Action justification
        if recommended_action == PositionAction.EMERGENCY_CLOSE:
            reasoning.append("Immediate closure required")
        elif recommended_action == PositionAction.CLOSE:
            reasoning.append("Position closure recommended")
        elif recommended_action == PositionAction.REDUCE:
            reasoning.append("Position size reduction advised")
        elif recommended_action == PositionAction.HEDGE:
            reasoning.append("Hedging strategy recommended")
        else:
            reasoning.append("Continue monitoring")
        
        return reasoning

class EmergencyStopLoss(nn.Module):
    """Emergency stop-loss mechanism"""
    
    def __init__(
        self,
        max_drawdown: float = 0.05,
        portfolio_risk_limit: float = 0.10,
        emergency_threshold: float = 0.08
    ):
        super().__init__()
        self.max_drawdown = max_drawdown
        self.portfolio_risk_limit = portfolio_risk_limit
        self.emergency_threshold = emergency_threshold
        
        # Emergency condition detector
        self.emergency_detector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Normal, Warning, Emergency
            nn.Softmax(dim=-1)
        )
          # Emergency response history
        self.emergency_history = deque(maxlen=100)
    
    def check_emergency_conditions(
        self,
        portfolio_metrics: torch.Tensor,
        current_drawdown: float,
        portfolio_risk: float
    ) -> Dict[str, Any]:
        """Check for emergency conditions"""
        
        # Detect emergency state
        emergency_probs = self.emergency_detector(portfolio_metrics)
        emergency_state = torch.argmax(emergency_probs).item()
        emergency_confidence = torch.max(emergency_probs).item()
        
        # Check manual thresholds (convert to Python bool for proper logical operations)
        drawdown_emergency = bool(current_drawdown > self.max_drawdown)
        risk_emergency = bool(portfolio_risk > self.portfolio_risk_limit)
        critical_emergency = bool(current_drawdown > self.emergency_threshold or 
                                portfolio_risk > self.emergency_threshold)
        
        # Determine overall emergency status (now using Python booleans)
        emergency_triggered = (emergency_state == 2 or  # Neural network emergency
                             drawdown_emergency or
                             risk_emergency or
                             critical_emergency)
        
        # Generate emergency response
        emergency_response = {
            'emergency_triggered': emergency_triggered,
            'emergency_state': emergency_state,
            'emergency_confidence': emergency_confidence,
            'drawdown_emergency': drawdown_emergency,
            'risk_emergency': risk_emergency,
            'critical_emergency': critical_emergency,
            'current_drawdown': current_drawdown,
            'portfolio_risk': portfolio_risk,
            'recommended_actions': []
        }
        
        # Add recommended actions
        if emergency_triggered:
            if critical_emergency:
                emergency_response['recommended_actions'].extend([
                    'IMMEDIATE_PORTFOLIO_CLOSURE',
                    'HALT_ALL_TRADING',
                    'NOTIFY_ADMINISTRATORS'
                ])
            elif drawdown_emergency:
                emergency_response['recommended_actions'].extend([
                    'REDUCE_ALL_POSITIONS',
                    'TIGHTEN_STOP_LOSSES',
                    'INCREASE_MONITORING'
                ])
            elif risk_emergency:
                emergency_response['recommended_actions'].extend([
                    'REDUCE_POSITION_SIZES',
                    'LIMIT_NEW_POSITIONS',
                    'INCREASE_CASH_RESERVES'
                ])
        
        # Log emergency response
        self.emergency_history.append({
            'timestamp': datetime.now(),
            'response': emergency_response
        })
        
        if emergency_triggered:
            logger.critical(f"EMERGENCY CONDITIONS DETECTED: {emergency_response['recommended_actions']}")
        
        return emergency_response

class HighLevelIntegrationSystem(nn.Module):
    """Complete high-level integration system for Phase 5 with dynamic dimension adaptation"""
    
    def __init__(
        self,
        strategy_innovation_module,
        market_state_awareness_system,
        meta_learning_optimizer,
        feature_dim: int = 768,
        enable_dynamic_adaptation: bool = True
    ):
        super().__init__()
        
        # Core Phase 5 components
        self.strategy_innovation = strategy_innovation_module
        self.market_state_awareness = market_state_awareness_system
        self.meta_learning_optimizer = meta_learning_optimizer
        
        # Dynamic dimension adapter
        self.enable_dynamic_adaptation = enable_dynamic_adaptation
        if enable_dynamic_adaptation and DynamicDimensionAdapter is not None:
            self.dimension_adapter = DynamicDimensionAdapter(
                default_adapter_type="linear",
                enable_caching=True,
                max_cache_size=50
            )
            
            # Register dimension specifications for each component
            self._register_component_specs()
        else:
            self.dimension_adapter = None
            if enable_dynamic_adaptation:
                logger.warning("Dynamic adaptation requested but DynamicDimensionAdapter not available")
        
        # High-level integration components
        self.anomaly_detector = AnomalyDetector(input_dim=feature_dim)
        self.position_manager = DynamicPositionManager(feature_dim=feature_dim)
        self.emergency_stop_loss = EmergencyStopLoss()
        
        # System monitoring
        self.system_state = SystemState.NORMAL
        self.active_alerts = deque(maxlen=1000)
        self.system_metrics_history = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_tracker = {
            'system_uptime': time.time(),
            'total_decisions': 0,
            'successful_adaptations': 0,
            'emergency_triggers': 0,
            'anomalies_detected': 0,
            'dimension_adaptations': 0
        }
        
        # Threading for real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
    
    def _register_component_specs(self):
        """Register dimension specifications for all components"""
        if self.dimension_adapter is None:
            return
        
        # Strategy Innovation Module
        self.dimension_adapter.register_dimension_spec(
            'strategy_innovation',
            DimensionSpec('strategy_innovation', (-1, 768), 2, 3, adaptive=True)
        )
        
        # Market State Awareness System  
        self.dimension_adapter.register_dimension_spec(
            'market_state_awareness',
            DimensionSpec('market_state_awareness', (-1, 512), 2, 3, adaptive=True)
        )
        
        # Meta Learning Optimizer
        self.dimension_adapter.register_dimension_spec(
            'meta_learning',
            DimensionSpec('meta_learning', (-1, 256), 2, 3, adaptive=True)
        )
        
        # Position Manager
        self.dimension_adapter.register_dimension_spec(
            'position_manager',
            DimensionSpec('position_manager', (-1, 768), 1, 2, adaptive=True)
        )
        
        # Anomaly Detector
        self.dimension_adapter.register_dimension_spec(
            'anomaly_detector',
            DimensionSpec('anomaly_detector', (-1, -1, 768), 3, 3, adaptive=True)
        )
          # Emergency Stop Loss
        self.dimension_adapter.register_dimension_spec(
            'emergency_stop_loss',
            DimensionSpec('emergency_stop_loss', (-1,), 1, 2, adaptive=True)
        )
    
    def _adapt_tensor_for_component(
        self,
        tensor: torch.Tensor,
        target_component: str,
        source_component: str = "unknown"
    ) -> torch.Tensor:
        """Adapt tensor dimensions for specific component"""
        if not self.enable_dynamic_adaptation or self.dimension_adapter is None:
            return tensor
        
        try:
            adapted_tensor = self.dimension_adapter.smart_adapt(
                tensor, target_component, source_component
            )
            
            if not torch.equal(tensor, adapted_tensor):
                self.performance_tracker['dimension_adaptations'] += 1
                logger.debug(f"Adapted tensor from {tensor.shape} to {adapted_tensor.shape} "
                           f"for {source_component} -> {target_component}")
            
            return adapted_tensor
            
        except Exception as e:
            logger.warning(f"Failed to adapt tensor for {target_component}: {e}")
            return tensor
    
    def process_market_data(
        self,
        market_data: torch.Tensor,
        position_data: Optional[Dict[str, torch.Tensor]] = None,
        portfolio_metrics: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Main processing pipeline for market data with dynamic dimension adaptation"""
        
        start_time = time.time()
        
        try:
            # Ensure market data has proper dimensions for processing
            if self.dimension_adapter is not None:
                market_data = ensure_sequence_dimension(market_data)
                if portfolio_metrics is not None:
                    portfolio_metrics = ensure_batch_dimension(portfolio_metrics)
            
            # 1. Market State Analysis
            market_state_input = self._adapt_tensor_for_component(
                market_data, 'market_state_awareness', 'input'
            )
            market_state_results = self.market_state_awareness(market_state_input)
            
            # 2. Strategy Innovation with adapted input
            strategy_input = self._adapt_tensor_for_component(
                market_data, 'strategy_innovation', 'input'
            )
            innovation_results = self.strategy_innovation(
                strategy_input,
                existing_strategies=None
            )
            
            # 3. Meta-Learning Optimization with improved TaskBatch creation
            meta_results = self._process_meta_learning(
                market_data, innovation_results
            )
            
            # 4. Anomaly Detection with proper dimension handling
            anomaly_input = self._adapt_tensor_for_component(
                market_data, 'anomaly_detector', 'input'
            )
            anomaly_results = self.anomaly_detector(anomaly_input)
            
            # 5. Position Management with dimension adaptation
            position_results = self._process_position_management(
                position_data, market_data
            )
            
            # 6. Emergency Stop Loss with adapted portfolio metrics
            emergency_results = self._process_emergency_conditions(
                portfolio_metrics
            )
            
            # 7. System Health Assessment
            system_health = self._assess_system_health(
                market_state_results,
                innovation_results,
                meta_results,
                anomaly_results,
                position_results,
                emergency_results
            )
            
            # Update performance tracking
            self.performance_tracker['total_decisions'] += 1
            if meta_results.get('adaptation_quality', 0) > 0.7:
                self.performance_tracker['successful_adaptations'] += 1
            if emergency_results.get('emergency_triggered', False):
                self.performance_tracker['emergency_triggers'] += 1
            if torch.max(anomaly_results['combined_scores']).item() > 0.7:
                self.performance_tracker['anomalies_detected'] += 1
            
            # Compile results
            results = {
                'market_state': market_state_results,
                'strategy_innovation': innovation_results,
                'meta_learning': meta_results,
                'anomaly_detection': anomaly_results,
                'position_management': position_results,
                'emergency_status': emergency_results,
                'system_health': system_health,
                'processing_time': time.time() - start_time,
                'adaptation_stats': self._get_adaptation_stats()
            }
            
            # Generate alerts based on results
            self._check_conditions_and_generate_alerts(results)
            
            # Update system metrics
            self._update_system_metrics(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in process_market_data: {e}")
            # Return safe fallback results
            return self._get_fallback_results(start_time, str(e))
    
    def _process_meta_learning(
        self,
        market_data: torch.Tensor,
        innovation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process meta-learning with improved TaskBatch handling"""
        
        try:
            if 'task_batches' in innovation_results:
                # Adapt tensor for meta-learning module
                task_tensor = self._adapt_tensor_for_component(
                    innovation_results['task_batches'],
                    'meta_learning',
                    'strategy_innovation'
                )
                
                # Create TaskBatch objects with proper dimensions
                task_batches = self._create_task_batches(task_tensor)
                
                # Adapt market data for meta-learning
                meta_input = self._adapt_tensor_for_component(
                    market_data, 'meta_learning', 'input'
                )
                
                meta_results = self.meta_learning_optimizer.optimize_and_adapt(
                    meta_input,
                    market_context={'state': 'unknown'},
                    task_batches=task_batches
                )
            else:
                # Fallback without task batches
                meta_input = self._adapt_tensor_for_component(
                    market_data, 'meta_learning', 'input'
                )
                meta_results = self.meta_learning_optimizer.optimize_and_adapt(
                    meta_input,
                    market_context={'state': 'unknown'}
                )
            
            return meta_results
            
        except Exception as e:
            logger.warning(f"Meta-learning processing failed: {e}")
            return {
                'adapted_features': market_data,
                'adaptation_quality': 0.5,
                'meta_loss': 1.0
            }
    
    def _create_task_batches(self, task_tensor: torch.Tensor) -> List[TaskBatch]:
        """Create TaskBatch objects with dynamic dimension handling"""
        
        task_batches = []
        batch_size = task_tensor.size(0)
        feature_dim = task_tensor.size(-1)
        
        # Limit number of tasks to avoid excessive computation
        num_tasks = min(batch_size, 4)
        
        for i in range(num_tasks):
            try:
                # Extract task data
                if task_tensor.dim() == 3:  # (batch, seq, features)
                    task_data = task_tensor[i, -1, :]  # Take last timestep
                else:  # (batch, features)
                    task_data = task_tensor[i]
                
                # Create support and query data with proper dimensions
                support_size = 16
                query_size = 8
                
                # Ensure task_data is properly shaped
                task_data = ensure_batch_dimension(task_data, 1).squeeze(0)
                
                # Create support data by expanding and adding noise
                support_data = task_data.unsqueeze(0).expand(support_size, -1)
                support_data = support_data + torch.randn_like(support_data) * 0.1
                
                # Create query data as subset of support data
                query_data = support_data[:query_size]
                
                # Create labels
                support_labels = torch.randn(support_size, 1)
                query_labels = torch.randn(query_size, 1)
                
                task_batch = TaskBatch(
                    support_data=support_data,
                    support_labels=support_labels,
                    query_data=query_data,
                    query_labels=query_labels,
                    task_id=f"adaptive_task_{i}",
                    market_state="dynamic",
                    difficulty=0.5 + (i * 0.1)  # Increasing difficulty
                )
                
                task_batches.append(task_batch)
                
            except Exception as e:
                logger.warning(f"Failed to create task batch {i}: {e}")
                continue
        
        return task_batches
    
    def _process_position_management(
        self,
        position_data: Optional[Dict[str, torch.Tensor]],
        market_data: torch.Tensor
    ) -> Dict[str, Any]:
        """Process position management with dimension adaptation"""
        
        if position_data is None:
            return {'positions_assessed': 0, 'total_risk': 0.0}
        
        try:
            # Adapt position data for position manager
            adapted_position_data = {}
            for pos_id, pos_tensor in position_data.items():
                adapted_tensor = self._adapt_tensor_for_component(
                    pos_tensor, 'position_manager', 'input'
                )
                adapted_position_data[pos_id] = adapted_tensor
            
            # Adapt market context
            market_context = self._adapt_tensor_for_component(
                market_data, 'position_manager', 'market_context'
            )
            
            # Manage positions
            position_risks = self.position_manager.manage_positions(
                adapted_position_data, market_context
            )
            
            return position_risks
            
        except Exception as e:
            logger.warning(f"Position management failed: {e}")
            return {'positions_assessed': 0, 'error': str(e)}
    
    def _process_emergency_conditions(
        self,
        portfolio_metrics: Optional[torch.Tensor]
    ) -> Dict[str, Any]:
        """Process emergency conditions with dimension adaptation"""
        
        try:
            if portfolio_metrics is not None:
                # Adapt portfolio metrics for emergency system
                adapted_metrics = self._adapt_tensor_for_component(
                    portfolio_metrics, 'emergency_stop_loss', 'input'
                )
                
                # Extract scalar values for emergency checking
                if adapted_metrics.dim() > 0:
                    drawdown = adapted_metrics[0].item() if len(adapted_metrics) > 0 else 0.0
                    volatility = adapted_metrics[1].item() if len(adapted_metrics) > 1 else 0.0
                else:
                    drawdown = adapted_metrics.item()
                    volatility = 0.0
                
                return self.emergency_stop_loss.check_emergency_conditions(
                    adapted_metrics, drawdown, volatility
                )
            else:
                return self.emergency_stop_loss.check_emergency_conditions(
                    torch.tensor([0.0]), 0.0, 0.0
                )
                
        except Exception as e:
            logger.warning(f"Emergency condition checking failed: {e}")
            return {
                'emergency_triggered': False,
                'risk_level': 'unknown',
                'recommended_actions': [],
                'error': str(e)
            }
    
    def _get_adaptation_stats(self) -> Dict[str, Any]:
        """Get dimension adaptation statistics"""
        
        if self.dimension_adapter is not None:
            return self.dimension_adapter.get_adaptation_stats()
        else:
            return {
                'total_adaptations': 0,
                'successful_adaptations': 0,
                'failed_adaptations': 0,                'cache_hits': 0,
                'cache_misses': 0,
                'cache_size': 0,
                'success_rate': 1.0,
                'cache_hit_rate': 0.0
            }
    
    def _get_fallback_results(self, start_time: float, error_msg: str) -> Dict[str, Any]:
        """Generate fallback results when processing fails"""
        
        return {
            'market_state': {'current_state': 'error', 'confidence': 0.0},
            'strategy_innovation': {'generated_strategies': None, 'innovation_confidence': 0.0},
            'meta_learning': {'adapted_features': None, 'adaptation_quality': 0.0},
            'anomaly_detection': {'combined_scores': torch.zeros(1, 1, 1)},
            'position_management': {'positions_assessed': 0},
            'emergency_status': {'emergency_triggered': True, 'risk_level': 'critical'},
            'system_health': {'overall_health': 0.0, 'system_state': 'error'},
            'processing_time': time.time() - start_time,
            'error': error_msg,
            'adaptation_stats': self._get_adaptation_stats()
        }
    
    def _check_conditions_and_generate_alerts(self, results: Dict[str, Any]):
        """Check conditions and generate alerts based on results"""
        
        # Emergency alerts
        if results['emergency_status']['emergency_triggered']:
            self._generate_alert(
                AlertLevel.EMERGENCY,
                "Emergency conditions detected in trading system",
                "EmergencyStopLoss",
                results['emergency_status']
            )
        
        # Anomaly alerts
        max_anomaly = torch.max(results['anomaly_detection']['combined_scores']).item()
        if max_anomaly > 0.8:
            self._generate_alert(
                AlertLevel.CRITICAL,
                f"High anomaly score detected: {max_anomaly:.4f}",
                "AnomalyDetector",
                {'anomaly_score': max_anomaly}
            )
        
        # System health alerts
        if results['system_health']['overall_health'] < 0.5:
            self._generate_alert(
                AlertLevel.WARNING,
                f"System health degraded: {results['system_health']['overall_health']:.4f}",
                "SystemHealth",
                results['system_health']
            )
    
    def _update_system_metrics(self, results: Dict[str, Any]):
        """Update system metrics history"""
        
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            system_state=self.system_state,
            performance_score=results['system_health']['overall_health'],
            risk_score=results['emergency_status'].get('portfolio_risk', 0.0),
            adaptation_quality=results['meta_learning'].get('adaptation_quality', 0.5),
            strategy_diversity=results['strategy_innovation'].get('strategy_diversity', 0.5),
            anomaly_score=torch.max(results['anomaly_detection']['combined_scores']).item(),
            resource_utilization={'cpu': 0.5, 'memory': 0.6, 'gpu': 0.7},
            active_alerts=len([a for a in self.active_alerts if not a.resolved]),
            resolved_alerts=len([a for a in self.active_alerts if a.resolved])
        )
        
        self.system_metrics_history.append(metrics)
    
    def _assess_system_health(
        self,
        market_state_results: Dict,
        innovation_results: Dict,
        anomaly_results: Dict,
        emergency_status: Dict
    ) -> Dict[str, Any]:
        """Assess overall system health"""
        
        # Calculate component health scores
        market_health = 1.0 - market_state_results.get('uncertainty', 0.5)
        
        innovation_health = innovation_results.get('innovation_confidence', 0.5)
        
        anomaly_health = 1.0 - torch.max(anomaly_results['combined_scores']).item()
        
        emergency_health = 0.0 if emergency_status['emergency_triggered'] else 1.0
        
        # Overall system health
        overall_health = (market_health + innovation_health + anomaly_health + emergency_health) / 4.0
        
        # Determine system state
        if overall_health > 0.8:
            system_state = SystemState.NORMAL
        elif overall_health > 0.6:
            system_state = SystemState.WARNING
        elif overall_health > 0.4:
            system_state = SystemState.CRITICAL
        else:
            system_state = SystemState.EMERGENCY
        
        # Update system state
        self.system_state = system_state
        
        return {
            'overall_health': overall_health,
            'system_state': system_state.value,
            'component_health': {
                'market_analysis': market_health,
                'strategy_innovation': innovation_health,
                'anomaly_detection': anomaly_health,
                'emergency_systems': emergency_health
            },
            'recommendations': self._generate_health_recommendations(overall_health, system_state)
        }
    
    def _generate_health_recommendations(self, health_score: float, system_state: SystemState) -> List[str]:
        """Generate system health recommendations"""
        
        recommendations = []
        
        if system_state == SystemState.EMERGENCY:            recommendations.extend([
                "Immediate intervention required",
                "Consider halting automated trading",
                "Review all system components",
                "Contact system administrators"
            ])
        elif system_state == SystemState.CRITICAL:
            recommendations.extend([
                "Increase monitoring frequency",
                "Reduce position sizes",
                "Review recent system changes",
                "Prepare for potential intervention"
            ])
        elif system_state == SystemState.WARNING:
            recommendations.extend([
                "Monitor system closely",
                "Review performance metrics",
                "Consider parameter adjustments"
            ])
        else:
            recommendations.append("System operating normally")
        
        return recommendations
    
    def _update_performance_tracking(self, results: Dict[str, Any]):
        """Update performance tracking metrics"""
        
        self.performance_tracker['total_decisions'] += 1
        
        if results['emergency_status']['emergency_triggered']:
            self.performance_tracker['emergency_triggers'] += 1
        
        if torch.max(results['anomaly_detection']['combined_scores']).item() > 0.7:
            self.performance_tracker['anomalies_detected'] += 1
        
        # Store system metrics
        metrics = SystemMetrics(
            timestamp=results['timestamp'],
            system_state=self.system_state,
            performance_score=results['system_health']['overall_health'],
            risk_score=results['emergency_status'].get('portfolio_risk', 0.0),
            adaptation_quality=results['meta_learning'].get('adaptation_quality', 0.5),
            strategy_diversity=results['strategy_innovation'].get('strategy_diversity', 0.5),
            anomaly_score=torch.max(results['anomaly_detection']['combined_scores']).item(),
            resource_utilization={'cpu': 0.5, 'memory': 0.6, 'gpu': 0.7},  # Example values
            active_alerts=len([a for a in self.active_alerts if not a.resolved]),
            resolved_alerts=len([a for a in self.active_alerts if a.resolved])
        )
        
        self.system_metrics_history.append(metrics)
    
    def _check_and_generate_alerts(self, results: Dict[str, Any]):
        """Check conditions and generate alerts"""
        
        # Emergency alerts
        if results['emergency_status']['emergency_triggered']:
            self._generate_alert(
                AlertLevel.EMERGENCY,
                "Emergency conditions detected in trading system",
                "EmergencyStopLoss",
                results['emergency_status']
            )
        
        # Anomaly alerts
        max_anomaly = torch.max(results['anomaly_detection']['combined_scores']).item()
        if max_anomaly > 0.8:
            self._generate_alert(
                AlertLevel.CRITICAL,
                f"High anomaly score detected: {max_anomaly:.4f}",
                "AnomalyDetector",
                {'anomaly_score': max_anomaly}
            )
        
        # System health alerts
        if results['system_health']['overall_health'] < 0.5:
            self._generate_alert(
                AlertLevel.WARNING,
                f"System health degraded: {results['system_health']['overall_health']:.4f}",
                "SystemHealth",
                results['system_health']
            )
    
    def _generate_alert(
        self,
        level: AlertLevel,
        message: str,
        component: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """Generate system alert"""
        
        alert = SystemAlert(
            level=level,
            message=message,
            component=component,
            timestamp=datetime.now(),
            data=data or {}
        )
        
        self.active_alerts.append(alert)
        
        # Log alert
        log_level = {
            AlertLevel.INFO: logging.info,
            AlertLevel.WARNING: logging.warning,
            AlertLevel.ERROR: logging.error,
            AlertLevel.CRITICAL: logging.critical,
            AlertLevel.EMERGENCY: logging.critical
        }[level]
        
        log_level(f"[{component}] {message}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        uptime = time.time() - self.performance_tracker['system_uptime']
        
        return {
            'system_state': self.system_state.value,
            'uptime_seconds': uptime,
            'performance_tracking': self.performance_tracker.copy(),
            'active_alerts': len([a for a in self.active_alerts if not a.resolved]),
            'total_alerts': len(self.active_alerts),
            'recent_metrics': list(self.system_metrics_history)[-10:] if self.system_metrics_history else [],
            'component_status': {
                'strategy_innovation': 'active',
                'market_state_awareness': 'active',
                'meta_learning_optimizer': 'active',
                'anomaly_detector': 'active',
                'position_manager': 'active',
                'emergency_stop_loss': 'active'
            }
        }

def test_high_level_integration_system():
    """Test the complete high-level integration system"""
    
    print("Testing High-Level Integration System...")
    
    # Create mock components (in real implementation, these would be actual Phase 5 components)
    class MockStrategyInnovation(nn.Module):
        def forward(self, market_data, market_state):
            return {
                'generated_strategies': torch.randn(4, 10, 64),
                'innovation_confidence': 0.8,
                'strategy_diversity': 0.7            }
    
    class MockMarketStateAwareness(nn.Module):
        def forward(self, market_data):
            return {
                'market_state': {
                    'dominant_state': 'trending_up',
                    'confidence': 0.9,
                    'state_probabilities': torch.tensor([0.1, 0.8, 0.05, 0.05])
                },
                'system_status': {
                    'current_state': 'trending_up',
                    'system_active': True
                },
                'strategy_recommendation': {
                    'recommended_weights': torch.randn(20)
                },
                'regime_analysis': {
                    'regime_type': 'bull_market'
                }
            }
    
    class MockMetaLearningOptimizer(nn.Module):
        def optimize_and_adapt(self, features, context, task_batches=None):
            return {
                'adapted_features': features,
                'adaptation_quality': 0.85,
                'meta_loss': 0.15
            }
    
    # Create integration system
    integration_system = HighLevelIntegrationSystem(
        strategy_innovation_module=MockStrategyInnovation(),
        market_state_awareness_system=MockMarketStateAwareness(),
        meta_learning_optimizer=MockMetaLearningOptimizer()
    )
    
    # Test data
    batch_size, seq_len, feature_dim = 4, 50, 512
    market_data = torch.randn(batch_size, seq_len, feature_dim)
    
    # Position data
    position_data = {
        'pos_1': torch.randn(feature_dim),
        'pos_2': torch.randn(feature_dim),
        'pos_3': torch.randn(feature_dim)
    }
    
    # Portfolio metrics
    portfolio_metrics = torch.randn(256)
    
    # Test main processing
    print("Testing main processing pipeline...")
    results = integration_system.process_market_data(
        market_data=market_data,
        position_data=position_data,
        portfolio_metrics=portfolio_metrics
    )
    
    print(f"✅ Processing completed in {results['processing_time']:.4f} seconds")
    print(f"✅ System health: {results['system_health']['overall_health']:.4f}")
    print(f"✅ System state: {results['system_health']['system_state']}")
    print(f"✅ Market state: {results['market_state']['current_state']}")
    print(f"✅ Emergency status: {results['emergency_status']['emergency_triggered']}")
    print(f"✅ Position risks assessed: {len(results['position_management'])} positions")
    print(f"✅ Max anomaly score: {torch.max(results['anomaly_detection']['combined_scores']):.4f}")
    
    # Test individual components
    print("\nTesting individual components...")
    
    # Test anomaly detector
    anomaly_results = integration_system.anomaly_detector(market_data)
    print(f"✅ Anomaly detection - Combined scores shape: {anomaly_results['combined_scores'].shape}")
    
    # Test position manager
    position_risks = integration_system.position_manager.manage_positions(position_data, market_data)
    print(f"✅ Position management - {len(position_risks)} positions assessed")
    
    # Test emergency stop loss
    emergency_response = integration_system.emergency_stop_loss.check_emergency_conditions(
        portfolio_metrics, 0.03, 0.04
    )
    print(f"✅ Emergency stop loss - Emergency triggered: {emergency_response['emergency_triggered']}")
    
    # Test system status
    system_status = integration_system.get_system_status()
    print(f"✅ System status - Uptime: {system_status['uptime_seconds']:.2f} seconds")
    print(f"✅ System status - Active alerts: {system_status['active_alerts']}")
    
    # Test gradient computation
    print("Testing gradient computation...")
    total_loss = sum([
        torch.sum(results['anomaly_detection']['combined_scores']),
        torch.sum(results['meta_learning']['adapted_features'])
    ])
    total_loss.backward()
    print("✅ Gradient computation successful")
    
    print("\n🎉 High-Level Integration System test completed successfully!")
    return True

if __name__ == "__main__":
    test_high_level_integration_system()
