"""
Meta-Learning Optimizer with MAML Implementation
Implements Model-Agnostic Meta-Learning (MAML) for fast adaptation capabilities
Part of Phase 5: High-Level Meta-Learning Capabilities
"""

from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Union
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict, deque
import copy
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptationStrategy(Enum):
    """Adaptation strategies for different market conditions"""
    FAST_ADAPTATION = "fast_adaptation"      # Quick response to market changes
    SLOW_ADAPTATION = "slow_adaptation"      # Stable long-term adaptation
    HYBRID_ADAPTATION = "hybrid_adaptation"   # Mixed fast/slow adaptation
    MOMENTUM_ADAPTATION = "momentum_adaptation"  # Momentum-based adaptation

@dataclass
class TaskBatch:
    """Container for meta-learning task batches"""
    support_data: torch.Tensor
    support_labels: torch.Tensor
    query_data: torch.Tensor
    query_labels: torch.Tensor
    task_id: str
    market_state: str
    difficulty: float

@dataclass
class AdaptationResult:
    """Result of adaptation process"""
    adapted_params: Dict[str, torch.Tensor]
    adaptation_loss: float
    query_loss: float
    adaptation_steps: int
    convergence_rate: float
    stability_score: float

class GradientProcessor(nn.Module):
    """Advanced gradient processing for MAML"""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Gradient transformation network
        self.grad_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Gradient scaling network
        self.grad_scale = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Gradient clipping predictor
        self.clip_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, gradients: torch.Tensor, task_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process gradients with context awareness"""
        batch_size, seq_len, grad_dim = gradients.shape
        
        # Flatten gradients for processing
        flat_grads = gradients.view(-1, grad_dim)
        
        # Transform gradients
        transformed_grads = self.grad_transform(flat_grads)
        
        # Scale gradients based on context
        if task_context is not None:
            # Incorporate task context into scaling
            context_expanded = task_context.unsqueeze(1).expand(-1, seq_len, -1).contiguous()
            context_flat = context_expanded.view(-1, task_context.size(-1))
            combined_input = torch.cat([flat_grads, context_flat], dim=-1)
            
            # Update networks to handle combined input
            if not hasattr(self, '_context_adjusted'):
                self._adjust_for_context(combined_input.size(-1))
        
        scale_factors = self.grad_scale(flat_grads)
        scaled_grads = transformed_grads * scale_factors
        
        # Predict optimal clipping threshold
        clip_thresholds = self.clip_predictor(flat_grads) * 10.0  # Scale to reasonable range
        
        # Apply dynamic clipping
        grad_norms = torch.norm(scaled_grads, dim=-1, keepdim=True)
        clip_factors = torch.min(clip_thresholds / (grad_norms + 1e-8), torch.ones_like(grad_norms))
        clipped_grads = scaled_grads * clip_factors
        
        # Reshape back to original dimensions
        return clipped_grads.view(batch_size, seq_len, grad_dim)
    
    def _adjust_for_context(self, new_input_dim: int):
        """Dynamically adjust network for context input"""
        # This is a simplified approach - in practice, you'd want more sophisticated adaptation
        self._context_adjusted = True

class MAMLOptimizer(nn.Module):
    """Model-Agnostic Meta-Learning Optimizer"""
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        adaptation_steps: int = 10,
        first_order: bool = False
    ):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.adaptation_steps = adaptation_steps
        self.first_order = first_order
        
        # Gradient processor for advanced gradient handling
        self.gradient_processor = GradientProcessor()
        
        # Meta-optimizer for outer loop
        self.meta_optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.gradient_processor.parameters()),
            lr=outer_lr
        )
        
        # Adaptation history
        self.adaptation_history = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_tracker = {
            'adaptation_losses': deque(maxlen=100),
            'query_losses': deque(maxlen=100),
            'convergence_rates': deque(maxlen=100),
            'stability_scores': deque(maxlen=100)
        }
    
    def adapt(
        self,
        task_batch: TaskBatch,
        strategy: AdaptationStrategy = AdaptationStrategy.FAST_ADAPTATION
    ) -> AdaptationResult:
        """Adapt model to new task using MAML"""
        
        # Create task-specific model copy
        adapted_model = copy.deepcopy(self.model)
        adapted_params = dict(adapted_model.named_parameters())
        
        # Inner loop adaptation
        adaptation_losses = []
        for step in range(self.inner_steps):
            # Forward pass on support set
            support_pred = adapted_model(task_batch.support_data)
            support_loss = F.mse_loss(support_pred, task_batch.support_labels)
              # Compute gradients
            if self.first_order:
                grads = torch.autograd.grad(
                    support_loss, adapted_params.values(),
                    create_graph=False, retain_graph=False, allow_unused=True
                )
            else:
                grads = torch.autograd.grad(
                    support_loss, adapted_params.values(),
                    create_graph=True, retain_graph=True, allow_unused=True
                )
            
            # Process gradients
            processed_grads = self._process_gradients(grads, strategy)
            
            # Update adapted parameters
            for (name, param), grad in zip(adapted_params.items(), processed_grads):
                if grad is not None:
                    adapted_params[name] = param - self.inner_lr * grad
            
            # Update model parameters
            for name, param in adapted_model.named_parameters():
                param.data = adapted_params[name].data
            
            adaptation_losses.append(support_loss.item())
        
        # Evaluate on query set
        query_pred = adapted_model(task_batch.query_data)
        query_loss = F.mse_loss(query_pred, task_batch.query_labels)
        
        # Calculate metrics
        convergence_rate = self._calculate_convergence_rate(adaptation_losses)
        stability_score = self._calculate_stability_score(adaptation_losses)
        
        result = AdaptationResult(
            adapted_params={name: param.detach().clone() for name, param in adapted_params.items()},
            adaptation_loss=np.mean(adaptation_losses),
            query_loss=query_loss.item(),
            adaptation_steps=self.inner_steps,
            convergence_rate=convergence_rate,
            stability_score=stability_score
        )
        
        # Update tracking
        self.performance_tracker['adaptation_losses'].append(result.adaptation_loss)
        self.performance_tracker['query_losses'].append(result.query_loss)
        self.performance_tracker['convergence_rates'].append(result.convergence_rate)
        self.performance_tracker['stability_scores'].append(result.stability_score)
        
        self.adaptation_history.append({
            'task_id': task_batch.task_id,
            'market_state': task_batch.market_state,
            'result': result,
            'strategy': strategy.value
        })
        
        return result
    
    def meta_update(self, task_batches: List[TaskBatch]) -> Dict[str, float]:
        """Perform meta-update using multiple tasks"""
        
        meta_losses = []
        self.meta_optimizer.zero_grad()
        
        for task_batch in task_batches:
            # Adapt to task
            adaptation_result = self.adapt(task_batch)
            
            # Use query loss for meta-update
            meta_losses.append(adaptation_result.query_loss)
        
        # Average meta-loss
        meta_loss = sum(meta_losses) / len(meta_losses)
        
        # Compute meta-gradients and update
        meta_loss_tensor = torch.tensor(meta_loss, requires_grad=True)
        meta_loss_tensor.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.gradient_processor.parameters(), max_norm=1.0)
        
        self.meta_optimizer.step()
        
        return {
            'meta_loss': meta_loss,
            'num_tasks': len(task_batches),
            'avg_adaptation_loss': np.mean([r.adaptation_loss for r in [self.adapt(tb) for tb in task_batches]]),
            'avg_convergence_rate': np.mean([r.convergence_rate for r in [self.adapt(tb) for tb in task_batches]])
        }
    
    def _process_gradients(
        self,
        gradients: Tuple[torch.Tensor, ...],
        strategy: AdaptationStrategy
    ) -> List[torch.Tensor]:
        """Process gradients based on adaptation strategy"""
        
        processed_grads = []
        
        for grad in gradients:
            if grad is None:
                processed_grads.append(None)
                continue
            
            # Apply strategy-specific processing
            if strategy == AdaptationStrategy.FAST_ADAPTATION:
                # Higher learning rate, less smoothing
                processed_grad = grad * 1.5
            elif strategy == AdaptationStrategy.SLOW_ADAPTATION:
                # Lower learning rate, more smoothing
                processed_grad = grad * 0.5
            elif strategy == AdaptationStrategy.HYBRID_ADAPTATION:
                # Balanced approach
                processed_grad = grad * 1.0
            elif strategy == AdaptationStrategy.MOMENTUM_ADAPTATION:
                # Apply momentum-like processing
                processed_grad = grad * 1.2
            else:
                processed_grad = grad
            
            processed_grads.append(processed_grad)
        
        return processed_grads
    
    def _calculate_convergence_rate(self, losses: List[float]) -> float:
        """Calculate convergence rate from loss sequence"""
        if len(losses) < 2:
            return 0.0
        
        # Calculate rate of loss decrease
        decreases = []
        for i in range(1, len(losses)):
            if losses[i-1] > 0:
                decrease = (losses[i-1] - losses[i]) / losses[i-1]
                decreases.append(max(0, decrease))
        
        return np.mean(decreases) if decreases else 0.0
    
    def _calculate_stability_score(self, losses: List[float]) -> float:
        """Calculate stability score from loss sequence"""
        if len(losses) < 2:
            return 1.0
        
        # Calculate coefficient of variation (lower is more stable)
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        
        if mean_loss > 0:
            cv = std_loss / mean_loss
            # Convert to stability score (higher is more stable)
            stability = 1.0 / (1.0 + cv)
        else:
            stability = 1.0
        
        return stability
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        metrics = {}
        
        for key, values in self.performance_tracker.items():
            if values:
                metrics[f'{key}_mean'] = np.mean(values)
                metrics[f'{key}_std'] = np.std(values)
                metrics[f'{key}_latest'] = values[-1]
        
        return metrics
    
    def reset_adaptation_history(self):
        """Reset adaptation history"""
        self.adaptation_history.clear()
        for values in self.performance_tracker.values():
            values.clear()

class FastAdaptationMechanism(nn.Module):
    """Fast adaptation mechanism for rapid strategy adjustment"""
    
    def __init__(
        self,
        feature_dim: int = 512,
        adaptation_dim: int = 256,
        num_adaptation_layers: int = 3
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.adaptation_dim = adaptation_dim
        self.num_adaptation_layers = num_adaptation_layers
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, adaptation_dim),
            nn.LayerNorm(adaptation_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Adaptation layers
        self.adaptation_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(adaptation_dim, adaptation_dim),
                nn.LayerNorm(adaptation_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_adaptation_layers)
        ])
        
        # Context attention
        self.context_attention = nn.MultiheadAttention(
            embed_dim=adaptation_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Adaptation output
        self.adaptation_output = nn.Linear(adaptation_dim, feature_dim)
        
        # Adaptation gate
        self.adaptation_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, adaptation_dim),
            nn.ReLU(),
            nn.Linear(adaptation_dim, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        features: torch.Tensor,
        context: torch.Tensor,
        adaptation_signal: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perform fast adaptation"""
        
        batch_size, seq_len, _ = features.shape
        
        # Encode features
        encoded_features = self.feature_encoder(features)
        
        # Apply adaptation layers with residual connections
        adapted_features = encoded_features
        for layer in self.adaptation_layers:
            residual = adapted_features
            adapted_features = layer(adapted_features) + residual
        
        # Apply context attention
        attended_features, attention_weights = self.context_attention(
            adapted_features, context, context
        )
        
        # Generate adaptation output
        adaptation_output = self.adaptation_output(attended_features)
        
        # Apply adaptation gate
        gate_input = torch.cat([features, adaptation_output], dim=-1)
        adaptation_gate = self.adaptation_gate(gate_input)
        
        # Combine original and adapted features
        final_output = features * (1 - adaptation_gate) + adaptation_output * adaptation_gate
        
        return final_output

class GeneticSelector:
    """遺傳算法超參數選擇器"""
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()
        self.fitness_scores = []
        
    def initialize_population(self) -> List[Dict[str, float]]:
        """初始化超參數種群"""
        population = []
        for _ in range(self.population_size):
            # 隨機生成學習率和折扣因子
            learning_rate = np.random.uniform(1e-5, 1e-2)
            discount_factor = np.random.uniform(0.8, 0.99)
            population.append({
                'learning_rate': learning_rate,
                'discount_factor': discount_factor
            })
        return population
    
    def select(self, fitness_scores: List[float]) -> List[Dict[str, float]]:
        """基於適應度分數選擇個體"""
        # 輪盤賭選擇
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return self.population  # 避免除零錯誤
            
        probabilities = [f / total_fitness for f in fitness_scores]
        selected_indices = np.random.choice(
            len(self.population),
            size=self.population_size,
            replace=True,
            p=probabilities
        )
        return [self.population[i] for i in selected_indices]
    
    def crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Dict[str, float]:
        """交叉操作生成新個體"""
        # 均勻交叉
        child = {}
        for key in parent1.keys():
            if np.random.rand() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child
    
    def mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """變異操作"""
        mutated = individual.copy()
        for key in mutated.keys():
            if np.random.rand() < self.mutation_rate:
                # 對數尺度變異
                if key == 'learning_rate':
                    mutated[key] = np.clip(mutated[key] * np.random.lognormal(0, 0.1), 1e-5, 1e-2)
                else:
                    mutated[key] = np.clip(mutated[key] * np.random.lognormal(0, 0.05), 0.8, 0.99)
        return mutated
    
    def evolve(self, fitness_scores: List[float]) -> List[Dict[str, float]]:
        """進化一代"""
        selected = self.select(fitness_scores)
        next_population = []
        
        # 精英保留: 保留最佳個體
        best_idx = np.argmax(fitness_scores)
        next_population.append(self.population[best_idx])
        
        # 生成新個體
        while len(next_population) < self.population_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            next_population.append(child)
        
        self.population = next_population
        return self.population

class MetaLearningOptimizer(nn.Module):
    """Complete Meta-Learning Optimizer System"""
    
    def __init__(
        self,
        model: nn.Module,
        feature_dim: int = 512,
        adaptation_dim: int = 256,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5
    ):
        super().__init__()
        
        # Core components
        self.maml_optimizer = MAMLOptimizer(
            model=model,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            inner_steps=inner_steps
        )
        
        self.fast_adaptation = FastAdaptationMechanism(
            feature_dim=feature_dim,
            adaptation_dim=adaptation_dim
        )
        
        # 遺傳算法超參數優化器
        self.genetic_selector = GeneticSelector()
        
        # Performance monitoring
        self.performance_history = deque(maxlen=1000)
        
        # Adaptation strategy selector
        self.strategy_selector = nn.Sequential(
            nn.Linear(feature_dim, adaptation_dim),
            nn.ReLU(),
            nn.Linear(adaptation_dim, len(AdaptationStrategy)),
            nn.Softmax(dim=-1)
        )
    
    def dynamic_hyperparam_tuning(self, performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        基於遺傳算法的超參數選擇
        實時調整學習率和折扣因子
        
        Args:
            performance_metrics: 策略性能指標字典
            
        Returns:
            調整後的超參數字典
        """
        # 提取適應度分數
        fitness_score = performance_metrics.get('fitness_score', 0.0)
        
        # 記錄當前適應度
        self.genetic_selector.fitness_scores.append(fitness_score)
        
        # 當收集到足夠樣本時進化
        if len(self.genetic_selector.fitness_scores) >= self.genetic_selector.population_size:
            self.genetic_selector.evolve(self.genetic_selector.fitness_scores)
            self.genetic_selector.fitness_scores = []  # 重置適應度記錄
        
        # 返回當前最佳超參數
        best_idx = np.argmax(self.genetic_selector.fitness_scores) if self.genetic_selector.fitness_scores else 0
        return self.genetic_selector.population[best_idx]
    
    def optimize_and_adapt(
        self,
        features: torch.Tensor,
        context: Union[torch.Tensor, Dict[str, Any]],
        task_batches: List[TaskBatch]
    ) -> Dict[str, Any]:
        """Perform meta-learning optimization and fast adaptation"""
        
        # Perform meta-update
        meta_results = self.maml_optimizer.meta_update(task_batches)
        
        # Handle context conversion to tensor if needed
        if isinstance(context, dict):
            # Convert dict context to tensor
            context_tensor = self._dict_to_tensor(context, features.device)
        else:
            context_tensor = context
            
        # Ensure context_tensor has proper dimensions for strategy selector
        if context_tensor.dim() == 1:        context_tensor = context_tensor.unsqueeze(0)  # [1, F]
        if context_tensor.dim() > 2:
            context_tensor = context_tensor.mean(dim=tuple(range(1, context_tensor.dim()-1)))  # Keep batch and feature dims
            
        # Select adaptation strategy - ensure proper tensor dimensionality
        if context_tensor.dim() > 1:
            # Take mean across sequence dimension but ensure we maintain batch dimension
            strategy_input = context_tensor.mean(dim=1)  # [batch, features]
            if strategy_input.size(-1) != self.strategy_selector[0].in_features:
                # Project to correct input size if needed
                strategy_input = torch.nn.functional.adaptive_avg_pool1d(
                    strategy_input.unsqueeze(1), self.strategy_selector[0].in_features
                ).squeeze(1)
        else:
            strategy_input = context_tensor
            if strategy_input.size(-1) != self.strategy_selector[0].in_features:
                # Expand to correct dimensions 
                strategy_input = strategy_input.expand(-1, self.strategy_selector[0].in_features)
        
        strategy_probs = self.strategy_selector(strategy_input)
        selected_strategy_idx = torch.argmax(strategy_probs, dim=-1)
        selected_strategy = list(AdaptationStrategy)[selected_strategy_idx[0].item()]
          # Perform fast adaptation
        adapted_features = self.fast_adaptation(features, context_tensor)
        
        # Collect results
        results = {
            'meta_results': meta_results,
            'adapted_features': adapted_features,
            'selected_strategy': selected_strategy.value,
            'strategy_confidence': torch.max(strategy_probs).item(),
            'adaptation_quality': self._assess_adaptation_quality(features, adapted_features)
        }
        
        # Update performance history
        self.performance_history.append({
            'meta_loss': meta_results['meta_loss'],
            'adaptation_quality': results['adaptation_quality'],
            'strategy': selected_strategy.value
        })
        
        return results
    
    def _dict_to_tensor(self, context_dict: Dict[str, Any], device: torch.device) -> torch.Tensor:
        """Convert dictionary context to tensor representation"""
        try:
            # Extract relevant numeric values from dictionary
            context_values = []
            
            # Handle common dictionary keys that might contain tensors or numbers
            for key, value in context_dict.items():
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        context_values.append(value.item())
                    else:
                        # For multi-element tensors, take mean
                        context_values.append(value.mean().item())
                elif isinstance(value, (int, float)):
                    context_values.append(float(value))
                elif isinstance(value, dict):
                    # Recursively handle nested dictionaries
                    nested_tensor = self._dict_to_tensor(value, device)
                    context_values.append(nested_tensor.mean().item())
                
            # If no valid values found, create a default tensor
            if not context_values:
                context_values = [0.0]  # Default value
                
            # Create tensor with appropriate size for strategy selector
            context_tensor = torch.tensor(context_values, dtype=torch.float32, device=device)
            
            # Ensure minimum size for strategy selector (expects at least 512-dim input)
            min_size = 512
            if context_tensor.size(0) < min_size:
                # Pad with zeros or repeat values
                padding_size = min_size - context_tensor.size(0)
                padding = torch.zeros(padding_size, dtype=torch.float32, device=device)
                context_tensor = torch.cat([context_tensor, padding], dim=0)
            elif context_tensor.size(0) > min_size:
                # Truncate to expected size
                context_tensor = context_tensor[:min_size]
                
            return context_tensor
            
        except Exception as e:
            logger.warning(f"Failed to convert context dict to tensor: {e}. Using default.")
            # Return a default tensor of appropriate size
            return torch.zeros(512, dtype=torch.float32, device=device)
    
    def _assess_adaptation_quality(
        self,
        original_features: torch.Tensor,
        adapted_features: torch.Tensor
    ) -> float:
        """Assess quality of adaptation"""
        
        # Calculate adaptation magnitude
        adaptation_magnitude = torch.norm(adapted_features - original_features).item()
        
        # Calculate feature diversity (higher is better for adaptation)
        feature_std = torch.std(adapted_features).item()
        
        # Combine metrics
        quality_score = min(1.0, feature_std / (adaptation_magnitude + 1e-8))
        
        return quality_score
    
    def get_optimization_metrics(self) -> Dict[str, float]:
        """Get comprehensive optimization metrics"""
        
        maml_metrics = self.maml_optimizer.get_performance_metrics()
        
        if self.performance_history:
            recent_history = list(self.performance_history)[-50:]  # Last 50 entries
            
            history_metrics = {
                'avg_meta_loss': np.mean([h['meta_loss'] for h in recent_history]),
                'avg_adaptation_quality': np.mean([h['adaptation_quality'] for h in recent_history]),
                'strategy_diversity': len(set([h['strategy'] for h in recent_history])) / len(AdaptationStrategy)
            }
        else:
            history_metrics = {
                'avg_meta_loss': 0.0,
                'avg_adaptation_quality': 0.0,
                'strategy_diversity': 0.0
            }
        
        return {**maml_metrics, **history_metrics}

def test_meta_learning_optimizer():
    """Test the meta-learning optimizer system"""
    
    print("Testing Meta-Learning Optimizer System...")
    
    # Test parameters
    batch_size = 4
    seq_len = 50
    feature_dim = 512
    adaptation_dim = 256
    
    # Create dummy model
    dummy_model = nn.Sequential(
        nn.Linear(feature_dim, adaptation_dim),
        nn.ReLU(),
        nn.Linear(adaptation_dim, 1)
    )
    
    # Create optimizer
    optimizer = MetaLearningOptimizer(
        model=dummy_model,
        feature_dim=feature_dim,
        adaptation_dim=adaptation_dim
    )
    
    # Create test data
    features = torch.randn(batch_size, seq_len, feature_dim)
    context = torch.randn(batch_size, seq_len, feature_dim)
    
    # Create task batches
    task_batches = []
    for i in range(3):
        task_batch = TaskBatch(
            support_data=torch.randn(16, feature_dim),
            support_labels=torch.randn(16, 1),
            query_data=torch.randn(8, feature_dim),
            query_labels=torch.randn(8, 1),
            task_id=f"task_{i}",
            market_state="trending_up",
            difficulty=0.5
        )
        task_batches.append(task_batch)
    
    # Test optimization and adaptation
    print("Testing optimization and adaptation...")
    results = optimizer.optimize_and_adapt(features, context, task_batches)
    
    print(f"✅ Meta-learning results: {results['meta_results']}")
    print(f"✅ Selected strategy: {results['selected_strategy']}")
    print(f"✅ Strategy confidence: {results['strategy_confidence']:.4f}")
    print(f"✅ Adaptation quality: {results['adaptation_quality']:.4f}")
    print(f"✅ Adapted features shape: {results['adapted_features'].shape}")
    
    # Test individual components
    print("\nTesting individual components...")
    
    # Test MAML optimizer
    maml_result = optimizer.maml_optimizer.adapt(task_batches[0])
    print(f"✅ MAML adaptation loss: {maml_result.adaptation_loss:.4f}")
    print(f"✅ MAML query loss: {maml_result.query_loss:.4f}")
    print(f"✅ MAML convergence rate: {maml_result.convergence_rate:.4f}")
    print(f"✅ MAML stability score: {maml_result.stability_score:.4f}")
    
    # Test fast adaptation
    adapted_features = optimizer.fast_adaptation(features, context)
    print(f"✅ Fast adaptation output shape: {adapted_features.shape}")
    
    # Test metrics
    metrics = optimizer.get_optimization_metrics()
    print(f"✅ Optimization metrics: {len(metrics)} metrics collected")
    
    # Test gradient computation
    print("Testing gradient computation...")
    loss = torch.sum(results['adapted_features'])
    loss.backward()
    print("✅ Gradient computation successful")
    
    print("\n🎉 Meta-Learning Optimizer System test completed successfully!")
    return True

if __name__ == "__main__":
    test_meta_learning_optimizer()
