# src/agent/strategy_innovation_module.py
"""
策略創新模組 - Phase 5: High-Level Meta-Learning Capabilities
實現策略生成網絡、評估系統和進化算法

主要功能：
1. 策略生成網絡：基於Transformer架構的策略創新網絡
2. 策略評估系統：多維度策略性能評估
3. 策略進化算法：遺傳算法結合神經架構搜索
4. 自動特徵工程：動態特徵選擇和組合
5. 知識遷移系統：跨市場策略知識遷移

技術特點：
- Neural Architecture Search (NAS) 自動搜索最優策略架構
- Multi-Objective Optimization 多目標優化平衡收益與風險
- Cross-Market Knowledge Transfer 跨市場知識遷移能力
- Automatic Feature Engineering 自動特徵工程
- Strategy Evolution Algorithm 策略進化算法
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import copy
import random
from collections import deque, defaultdict

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StrategyGenome:
    """策略基因組"""
    architecture: Dict[str, Any]  # 網絡架構參數
    hyperparameters: Dict[str, float]  # 超參數
    features: List[str]  # 使用的特徵
    objectives: Dict[str, float]  # 目標權重
    performance_metrics: Dict[str, float]  # 性能指標
    generation: int  # 生成代數
    parent_ids: List[str]  # 父代ID
    genome_id: str  # 基因組ID
    creation_time: str  # 創建時間


@dataclass
class StrategyEvaluation:
    """策略評估結果"""
    strategy_id: str
    fitness_score: float
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    efficiency_metrics: Dict[str, float]
    adaptability_score: float
    robustness_score: float
    innovation_score: float
    evaluation_time: str


class StrategyGeneratorTransformer(nn.Module):
    """策略生成Transformer網絡 - 完全動態維度適應"""
    
    def __init__(self, 
                 input_dim: int = None,
                 hidden_dim: int = None,
                 num_layers: int = None,
                 num_heads: int = None,
                 max_sequence_length: int = 256,
                 config_adapter=None):
        super().__init__()
        
        # 動態獲取配置
        if config_adapter is not None:
            config = config_adapter.get_dynamic_config()
            self.input_dim = input_dim or config.get('model_dim', 768)
            self.hidden_dim = hidden_dim or config.get('model_dim', 768)
            self.num_layers = num_layers or max(4, config.get('num_layers', 16) // 2)
            self.num_heads = num_heads or config.get('num_heads', 24)
        else:
            # 從config.py動態導入
            try:
                from oanda_trading_bot.training_system.common.config import (
                    TRANSFORMER_MODEL_DIM, TRANSFORMER_NUM_LAYERS, TRANSFORMER_NUM_HEADS
                )
                self.input_dim = input_dim or TRANSFORMER_MODEL_DIM
                self.hidden_dim = hidden_dim or TRANSFORMER_MODEL_DIM
                self.num_layers = num_layers or max(4, TRANSFORMER_NUM_LAYERS // 2)
                self.num_heads = num_heads or TRANSFORMER_NUM_HEADS
            except ImportError:
                # 後備配置
                self.input_dim = input_dim or 768
                self.hidden_dim = hidden_dim or 768
                self.num_layers = num_layers or 8
                self.num_heads = num_heads or 24
        
        self.max_sequence_length = max_sequence_length
        
        # 驗證維度兼容性
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) 必須能被 num_heads ({self.num_heads}) 整除"
        
        # 動態計算中間層維度
        intermediate_dim_1 = max(256, self.hidden_dim // 2)  # 第一中間層
        intermediate_dim_2 = max(128, self.hidden_dim // 4)  # 第二中間層
        architecture_output_dim = max(64, self.hidden_dim // 6)  # 架構參數輸出
        hyperparameter_output_dim = max(32, self.hidden_dim // 12)  # 超參數輸出
        feature_output_dim = max(50, self.hidden_dim // 8)  # 特徵選擇輸出
        objective_output_dim = max(8, min(16, self.hidden_dim // 48))  # 目標權重輸出
        
        # 輸入嵌入層
        self.input_embedding = nn.Linear(self.input_dim, self.hidden_dim)
        self.position_encoding = nn.Parameter(
            torch.randn(max_sequence_length, self.hidden_dim)
        )
        
        # Transformer編碼器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )
        
        # 動態策略架構生成器
        self.architecture_generator = nn.Sequential(
            nn.Linear(self.hidden_dim, intermediate_dim_1),
            nn.GELU(),
            nn.LayerNorm(intermediate_dim_1),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim_1, intermediate_dim_2),
            nn.GELU(),
            nn.Linear(intermediate_dim_2, architecture_output_dim),
            nn.Sigmoid()
        )
        
        # 動態超參數生成器
        self.hyperparameter_generator = nn.Sequential(
            nn.Linear(self.hidden_dim, intermediate_dim_2),
            nn.GELU(),
            nn.LayerNorm(intermediate_dim_2),
            nn.Linear(intermediate_dim_2, hyperparameter_output_dim),
            nn.Softplus()
        )
        
        # 動態特徵選擇器
        self.feature_selector = nn.Sequential(
            nn.Linear(self.hidden_dim, intermediate_dim_1),
            nn.GELU(),
            nn.Linear(intermediate_dim_1, intermediate_dim_2),
            nn.GELU(),
            nn.Linear(intermediate_dim_2, feature_output_dim),
            nn.Sigmoid()
        )
          # 動態目標權重生成器
        self.objective_weight_generator = nn.Sequential(
            nn.Linear(self.hidden_dim, max(64, self.hidden_dim // 6)),
            nn.GELU(),
            nn.Linear(max(64, self.hidden_dim // 6), objective_output_dim),
            nn.Softmax(dim=-1)
        )
        
        # Explicitly declare projector attributes to be None initially
        self.context_projector: Optional[nn.Linear] = None
        self.input_projector: Optional[nn.Linear] = None

        logger.info(f"動態策略生成Transformer初始化 - 隱藏維度: {self.hidden_dim}, "
                   f"層數: {self.num_layers}, 頭數: {self.num_heads}, "                   f"架構輸出: {architecture_output_dim}, 超參數輸出: {hyperparameter_output_dim}")
    
    def forward(self, market_context: torch.Tensor,
                existing_strategies: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        生成新策略
        
        Args:
            market_context: 市場環境上下文 [batch_size, context_dim] or [batch_size, seq_len, context_dim]
            existing_strategies: 現有策略特徵 [batch_size, num_strategies, strategy_dim]
            
        Returns:
            策略參數字典
        """
        batch_size = market_context.size(0)
        device = market_context.device

        # 1. 確保 market_context 是 3D 張量: [batch_size, seq_len, features]
        if market_context.dim() == 2:
            market_context = market_context.unsqueeze(1)
        elif market_context.dim() > 3:
            # 重塑形狀，例如 [B, S1, S2, F] -> [B, S1*S2, F]
            market_context = market_context.view(batch_size, -1, market_context.size(-1))
        # 如果 market_context.dim() == 3，則已經是正確的形狀，無需更改

        input_sequence: torch.Tensor

        if existing_strategies is not None:
            # 2a. 處理 existing_strategies 為字典的情況
            if isinstance(existing_strategies, dict):
                keys_to_try = ['strategies_tensor', 'features', 'tensor', 'data', 'strategy_features']
                extracted_tensor = None
                for key in keys_to_try:
                    if key in existing_strategies and isinstance(existing_strategies[key], torch.Tensor):
                        extracted_tensor = existing_strategies[key]
                        break
                if extracted_tensor is None:
                    tensor_values = [v for v in existing_strategies.values() if isinstance(v, torch.Tensor)]
                    if len(tensor_values) == 1:
                        extracted_tensor = tensor_values[0]
                    else:
                        logger.error(f"StrategyGenerator: existing_strategies is a dict, but could not unambiguously extract a tensor. Keys: {list(existing_strategies.keys())}")
                        raise ValueError(
                            f"StrategyGenerator: existing_strategies is a dict, but could not unambiguously extract a tensor. "
                            f"Dict keys: {list(existing_strategies.keys())}. Ensure one of {keys_to_try} contains the tensor, "
                            f"or the dict contains exactly one tensor value."
                        )
                existing_strategies = extracted_tensor

            if not isinstance(existing_strategies, torch.Tensor):
                logger.error(f"StrategyGenerator: existing_strategies expected as torch.Tensor, got {type(existing_strategies)}.")
                raise TypeError(
                    f"StrategyGenerator: existing_strategies is expected to be a torch.Tensor, "
                    f"but received type {type(existing_strategies)} after attempting to process."
                )

            # 2b. 確保 existing_strategies 是 3D 張量: [batch_size, num_strategies, strategy_dim]
            if existing_strategies.dim() == 2: # [B, F] -> [B, 1, F]
                existing_strategies = existing_strategies.unsqueeze(1)
            elif existing_strategies.dim() != 3:
                logger.error(f"StrategyGenerator: existing_strategies must be 2D or 3D, got {existing_strategies.dim()}D shape {existing_strategies.shape}")
                raise ValueError(f"StrategyGenerator: existing_strategies must be 2D or 3D, got {existing_strategies.dim()}D")
            
            # 2c. 如果 market_context 的特徵維度與 existing_strategies 不同，則對 market_context 進行投影
            context_feature_dim = market_context.size(-1)
            strategy_feature_dim = existing_strategies.size(-1)
            market_context_projected: torch.Tensor

            if context_feature_dim != strategy_feature_dim:
                recreate_context_projector = False
                if self.context_projector is None: # Check if None first
                    recreate_context_projector = True
                elif not isinstance(self.context_projector, nn.Linear): # Should not happen if None or Linear
                    logger.warning("StrategyGenerator: self.context_projector was not None and not nn.Linear. Recreating.")
                    recreate_context_projector = True
                elif self.context_projector.in_features != context_feature_dim:
                    recreate_context_projector = True
                elif self.context_projector.out_features != strategy_feature_dim:
                    recreate_context_projector = True
                
                if recreate_context_projector:
                    logger.info(f"StrategyGenerator: Recreating context_projector from {context_feature_dim} to {strategy_feature_dim}")
                    self.context_projector = nn.Linear(context_feature_dim, strategy_feature_dim).to(device)
                
                market_context_projected = self.context_projector(market_context)
            else:
                market_context_projected = market_context
            
            # 2d. 將 market_context_projected 和 existing_strategies 沿序列維度連接
            # market_context_projected 是 [B, S_ctx, F_strat]
            # existing_strategies 是 [B, S_strat, F_strat]
            input_sequence = torch.cat([market_context_projected, existing_strategies], dim=1)
        else:
            # 3. 如果沒有 existing_strategies，則直接使用 market_context
            input_sequence = market_context

        # 4. 獲取 input_sequence 當前的特徵維度
        # input_sequence 是 [B, S_combined_or_ctx, F_current]
        current_input_feature_dim = input_sequence.size(-1)
        
        # 5. 確定嵌入層所需的目標特徵維度
        # self.input_dim 是 self.input_embedding 預期的特徵大小
        desired_embedding_input_dim = self.input_dim 

        final_sequence_for_embedding: torch.Tensor

        # 6. 如果 input_sequence 的特徵維度與嵌入層預期的不匹配，則進行投影
        if current_input_feature_dim != desired_embedding_input_dim:
            recreate_input_projector = False
            if self.input_projector is None: # Check if None first
                recreate_input_projector = True
            elif not isinstance(self.input_projector, nn.Linear): # Should not happen
                logger.warning("StrategyGenerator: self.input_projector was not None and not nn.Linear. Recreating.")
                recreate_input_projector = True
            elif self.input_projector.in_features != current_input_feature_dim:
                recreate_input_projector = True
            elif self.input_projector.out_features != desired_embedding_input_dim: # Target dim for projector
                recreate_input_projector = True
            
            if recreate_input_projector:
                logger.info(f"StrategyGenerator: Recreating input_projector from {current_input_feature_dim} to {desired_embedding_input_dim}")
                self.input_projector = nn.Linear(current_input_feature_dim, desired_embedding_input_dim).to(device)
            
            if self.input_projector is None: # Should have been created if needed
                 err_msg = "StrategyGenerator: self.input_projector is None after recreation logic. This should not happen."
                 logger.error(err_msg)
                 raise RuntimeError(err_msg)

            final_sequence_for_embedding = self.input_projector(input_sequence)
        else:
            final_sequence_for_embedding = input_sequence
        
        # 8. 輸入嵌入
        # final_sequence_for_embedding 的特徵維度 = desired_embedding_input_dim (self.input_dim)
        embedded = self.input_embedding(final_sequence_for_embedding)
        
        # 9. 添加位置編碼
        seq_len = final_sequence_for_embedding.size(1)
        if seq_len <= self.max_sequence_length:
            embedded = embedded + self.position_encoding[:seq_len].unsqueeze(0) # 將位置編碼添加到批次維度
        else:
            logger.warning(
                f"Input sequence length {seq_len} exceeds max_sequence_length {self.max_sequence_length}. "
                f"Positional encoding will be applied only to the first {self.max_sequence_length} tokens."
            )
            # 只對與 max_sequence_length 匹配的序列部分應用位置編碼
            embedded[:, :self.max_sequence_length, :] = embedded[:, :self.max_sequence_length, :] + self.position_encoding.unsqueeze(0)

        # Transformer 編碼
        transformer_output = self.transformer(embedded)
        
        # 聚合序列信息（例如，平均池化）
        aggregated = transformer_output.mean(dim=1)  # [batch_size, hidden_dim]
        
        # 生成策略組件
        architecture_params = self.architecture_generator(aggregated)
        hyperparameters = self.hyperparameter_generator(aggregated)
        feature_weights = self.feature_selector(aggregated)
        objective_weights = self.objective_weight_generator(aggregated)
        
        return {
            'architecture_params': architecture_params,
            'hyperparameters': hyperparameters,
            'feature_weights': feature_weights,
            'objective_weights': objective_weights,
            'encoded_context': aggregated  # 對於下游任務或分析很有用
        }


class StrategyEvaluator(nn.Module):
    """策略評估系統 - 動態維度適應"""
    
    def __init__(self, 
                 strategy_dim: int = None,
                 evaluation_metrics: int = None,
                 config_adapter=None):
        super().__init__()
        
        # 動態獲取配置
        if config_adapter is not None:
            config = config_adapter.get_dynamic_config()
            base_dim = config.get('model_dim', 768)
        else:
            try:
                from oanda_trading_bot.training_system.common.config import TRANSFORMER_MODEL_DIM
                base_dim = TRANSFORMER_MODEL_DIM
            except ImportError:
                base_dim = 768
        
        self.strategy_dim = strategy_dim or max(256, base_dim // 3)
        self.evaluation_metrics = evaluation_metrics or max(20, base_dim // 32)
        
        # 動態計算中間維度
        intermediate_dim_1 = max(256, base_dim // 2)
        intermediate_dim_2 = max(128, base_dim // 4)
        risk_metrics_dim = max(8, min(12, base_dim // 64))
        
        # 性能評估網絡
        self.performance_evaluator = nn.Sequential(
            nn.Linear(self.strategy_dim, intermediate_dim_1),
            nn.GELU(),
            nn.LayerNorm(intermediate_dim_1),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim_1, intermediate_dim_2),
            nn.GELU(),
            nn.Linear(intermediate_dim_2, self.evaluation_metrics),
            nn.Sigmoid()
        )
        
        # 風險評估網絡
        self.risk_evaluator = nn.Sequential(
            nn.Linear(self.strategy_dim, intermediate_dim_2),
            nn.GELU(),
            nn.LayerNorm(intermediate_dim_2),
            nn.Linear(intermediate_dim_2, max(64, intermediate_dim_2 // 2)),
            nn.GELU(),
            nn.Linear(max(64, intermediate_dim_2 // 2), risk_metrics_dim),
            nn.Sigmoid()
        )
        
        # 適應性評估網絡
        self.adaptability_evaluator = nn.Sequential(
            nn.Linear(self.strategy_dim, intermediate_dim_2),
            nn.GELU(),
            nn.Linear(intermediate_dim_2, max(64, intermediate_dim_2 // 2)),
            nn.GELU(),
            nn.Linear(max(64, intermediate_dim_2 // 2), 1),
            nn.Sigmoid()
        )
        
        # 創新性評估網絡
        self.innovation_evaluator = nn.Sequential(
            nn.Linear(self.strategy_dim, max(64, self.strategy_dim // 2)),
            nn.GELU(),
            nn.Linear(max(64, self.strategy_dim // 2), max(32, self.strategy_dim // 4)),
            nn.GELU(),
            nn.Linear(max(32, self.strategy_dim // 4), 1),
            nn.Sigmoid()
        )
        
        # 穩健性評估網絡
        self.robustness_evaluator = nn.Sequential(
            nn.Linear(self.strategy_dim, intermediate_dim_2),
            nn.GELU(),
            nn.Linear(intermediate_dim_2, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"初始化動態策略評估器 - 策略維度: {self.strategy_dim}, "
                   f"評估指標: {self.evaluation_metrics}, 風險指標: {risk_metrics_dim}")
        
    def forward(self, strategy_representation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        評估策略
        
        Args:
            strategy_representation: 策略表示向量
            
        Returns:
            評估結果字典
        """
        performance_scores = self.performance_evaluator(strategy_representation)
        risk_scores = self.risk_evaluator(strategy_representation)
        adaptability_score = self.adaptability_evaluator(strategy_representation)
        innovation_score = self.innovation_evaluator(strategy_representation)
        robustness_score = self.robustness_evaluator(strategy_representation)
        
        # 計算綜合適應度分數
        fitness_score = (
            performance_scores.mean(dim=-1, keepdim=True) * 0.4 +
            (1.0 - risk_scores.mean(dim=-1, keepdim=True)) * 0.3 +
            adaptability_score * 0.15 +
            innovation_score * 0.1 +
            robustness_score * 0.05
        )
        
        return {
            'fitness_score': fitness_score,
            'performance_scores': performance_scores,
            'risk_scores': risk_scores,
            'adaptability_score': adaptability_score,
            'innovation_score': innovation_score,
            'robustness_score': robustness_score
        }


class StrategyEvolutionEngine(nn.Module):
    """策略進化引擎"""
    
    def __init__(self,
                 population_size: int = 50,
                 elite_ratio: float = 0.2,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8):
        super().__init__()
        
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # 進化統計
        self.generation_counter = 0
        self.best_fitness_history = []
        self.average_fitness_history = []
        self.diversity_history = []
        
        # 當前種群
        self.current_population = []
        self.fitness_scores = []
        
        # 突變控制器
        self.mutation_controller = nn.Parameter(torch.tensor(mutation_rate))
        
        logger.info(f"初始化策略進化引擎 - 種群大小: {population_size}")
    
    def initialize_population(self, generator: StrategyGeneratorTransformer,
                            market_context: torch.Tensor) -> List[StrategyGenome]:
        """初始化種群"""
        population = []
        
        with torch.no_grad():
            for i in range(self.population_size):
                # 添加隨機噪聲以增加多樣性
                noisy_context = market_context + torch.randn_like(market_context) * 0.1
                
                strategy_params = generator(noisy_context)
                
                # 創建策略基因組
                genome = StrategyGenome(
                    architecture={
                        'params': strategy_params['architecture_params'].cpu().numpy(),
                        'dimensions': strategy_params['architecture_params'].shape
                    },
                    hyperparameters={
                        f'param_{j}': float(strategy_params['hyperparameters'][0, j])
                        for j in range(strategy_params['hyperparameters'].shape[1])
                    },
                    features=[f'feature_{j}' for j in range(strategy_params['feature_weights'].shape[1])
                             if strategy_params['feature_weights'][0, j] > 0.5],
                    objectives={
                        f'objective_{j}': float(strategy_params['objective_weights'][0, j])
                        for j in range(strategy_params['objective_weights'].shape[1])
                    },
                    performance_metrics={},
                    generation=0,
                    parent_ids=[],
                    genome_id=f"gen0_individual_{i}",
                    creation_time=datetime.now().isoformat()
                )
                
                population.append(genome)
        
        self.current_population = population
        logger.info(f"初始化種群完成 - {len(population)} 個個體")
        return population
    
    def evaluate_population(self, evaluator: StrategyEvaluator) -> List[float]:
        """評估種群適應度"""
        fitness_scores = []
        
        for genome in self.current_population:
            # 將基因組轉換為策略表示
            strategy_repr = self._genome_to_representation(genome)
            
            with torch.no_grad():
                evaluation = evaluator(strategy_repr.unsqueeze(0))
                fitness = float(evaluation['fitness_score'][0, 0])
                  # 更新基因組性能指標
                genome.performance_metrics = {
                    'fitness': fitness,
                    'performance': float(evaluation['performance_scores'][0].mean()),
                    'risk': float(evaluation['risk_scores'][0].mean()),
                    'adaptability': float(evaluation['adaptability_score'][0, 0]),
                    'innovation': float(evaluation['innovation_score'][0, 0]),
                    'robustness': float(evaluation['robustness_score'][0, 0])
                }
                
                fitness_scores.append(fitness)
        
        self.fitness_scores = fitness_scores
        return fitness_scores
    
    def evolve_generation(self) -> List[StrategyGenome]:
        """進化一代"""
        if not self.current_population:
            raise ValueError("種群未初始化")
        
        # 選擇精英
        elite_size = int(self.population_size * self.elite_ratio)
        sorted_population = sorted(zip(self.current_population, self.fitness_scores),
                                 key=lambda x: x[1], reverse=True)
        elites = [genome for genome, _ in sorted_population[:elite_size]]
        
        # 生成新一代
        new_population = elites.copy()
        
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                # 交叉
                parent1, parent2 = self._tournament_selection(2)
                child = self._crossover(parent1, parent2)
            else:
                # 選擇
                child = copy.deepcopy(self._tournament_selection(1)[0])
            
            # 突變
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
            
            # 更新基因組信息
            child.generation = self.generation_counter + 1
            child.genome_id = f"gen{child.generation}_individual_{len(new_population)}"
            child.creation_time = datetime.now().isoformat()
            
            new_population.append(child)
        
        # 更新種群和統計
        self.current_population = new_population
        self.generation_counter += 1
        
        # 記錄統計信息
        best_fitness = max(self.fitness_scores)
        avg_fitness = sum(self.fitness_scores) / len(self.fitness_scores)
        diversity = self._calculate_diversity()
        
        self.best_fitness_history.append(best_fitness)
        self.average_fitness_history.append(avg_fitness)
        self.diversity_history.append(diversity)
        
        logger.info(f"進化第 {self.generation_counter} 代完成 - "
                   f"最佳適應度: {best_fitness:.4f}, 平均適應度: {avg_fitness:.4f}")
        
        return new_population
    
    def _genome_to_representation(self, genome: StrategyGenome) -> torch.Tensor:
        """將基因組轉換為策略表示向量 - 動態維度適應"""
        # 簡化實現：連接所有參數
        arch_params = torch.tensor(genome.architecture['params'], dtype=torch.float32)
        hyper_params = torch.tensor(list(genome.hyperparameters.values()), dtype=torch.float32)
        obj_weights = torch.tensor(list(genome.objectives.values()), dtype=torch.float32)
        
        # 確保維度一致
        if arch_params.dim() > 1:
            arch_params = arch_params.flatten()
        
        representation = torch.cat([arch_params, hyper_params, obj_weights])
        
        # 動態計算目標維度
        try:
            from oanda_trading_bot.training_system.common.config import TRANSFORMER_MODEL_DIM
            target_dim = max(256, TRANSFORMER_MODEL_DIM // 3)
        except ImportError:
            target_dim = 256
        
        # 填充或截斷到動態計算的長度
        if representation.size(0) < target_dim:
            padding = torch.zeros(target_dim - representation.size(0))
            representation = torch.cat([representation, padding])
        elif representation.size(0) > target_dim:
            representation = representation[:target_dim]
        
        return representation
    
    def _tournament_selection(self, num_parents: int) -> List[StrategyGenome]:
        """錦標賽選擇"""
        tournament_size = 3
        selected = []
        
        for _ in range(num_parents):
            tournament_indices = random.sample(range(len(self.current_population)), 
                                             min(tournament_size, len(self.current_population)))
            tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            selected.append(self.current_population[winner_idx])
        
        return selected
    
    def _crossover(self, parent1: StrategyGenome, parent2: StrategyGenome) -> StrategyGenome:
        """交叉操作"""
        child = copy.deepcopy(parent1)
        
        # 架構參數交叉
        if random.random() < 0.5:
            child.architecture = copy.deepcopy(parent2.architecture)
        
        # 超參數交叉
        for key in child.hyperparameters:
            if key in parent2.hyperparameters and random.random() < 0.5:
                child.hyperparameters[key] = parent2.hyperparameters[key]
        
        # 特徵交叉
        p1_features = set(parent1.features)
        p2_features = set(parent2.features)
        child.features = list(p1_features.union(p2_features))
        
        # 目標權重交叉
        for key in child.objectives:
            if key in parent2.objectives and random.random() < 0.5:
                child.objectives[key] = parent2.objectives[key]
        
        child.parent_ids = [parent1.genome_id, parent2.genome_id]
        return child
    
    def _mutate(self, genome: StrategyGenome) -> StrategyGenome:
        """突變操作"""
        mutated = copy.deepcopy(genome)
        
        # 超參數突變
        for key in mutated.hyperparameters:
            if random.random() < 0.3:
                mutation_factor = 1.0 + random.gauss(0, 0.1)
                mutated.hyperparameters[key] *= mutation_factor
                mutated.hyperparameters[key] = max(0.001, mutated.hyperparameters[key])
        
        # 特徵突變
        if random.random() < 0.2:
            available_features = [f'feature_{i}' for i in range(100)]
            if random.random() < 0.5 and mutated.features:
                # 移除特徵
                mutated.features.remove(random.choice(mutated.features))
            else:
                # 添加特徵
                new_feature = random.choice(available_features)
                if new_feature not in mutated.features:
                    mutated.features.append(new_feature)
        
        # 目標權重突變
        for key in mutated.objectives:
            if random.random() < 0.2:
                mutation_factor = 1.0 + random.gauss(0, 0.05)
                mutated.objectives[key] *= mutation_factor
        
        # 重新歸一化目標權重
        total_weight = sum(mutated.objectives.values())
        if total_weight > 0:
            for key in mutated.objectives:
                mutated.objectives[key] /= total_weight
        
        return mutated
    
    def _calculate_diversity(self) -> float:
        """計算種群多樣性"""
        if len(self.current_population) < 2:
            return 0.0
        
        # 簡化多樣性計算：基於超參數差異
        diversity_sum = 0.0
        count = 0
        
        for i in range(len(self.current_population)):
            for j in range(i + 1, len(self.current_population)):
                genome1 = self.current_population[i]
                genome2 = self.current_population[j]
                
                # 計算超參數差異
                param_diff = 0.0
                common_keys = set(genome1.hyperparameters.keys()) & set(genome2.hyperparameters.keys())
                
                for key in common_keys:
                    param_diff += abs(genome1.hyperparameters[key] - genome2.hyperparameters[key])
                
                diversity_sum += param_diff
                count += 1
        
        return diversity_sum / count if count > 0 else 0.0


class AutomaticFeatureEngineer(nn.Module):
    """自動特徵工程系統 - 動態維度適應"""
    
    def __init__(self, 
                 base_feature_dim: int = None,
                 engineered_feature_dim: int = None,
                 transformation_layers: int = 3,
                 config_adapter=None):
        super().__init__()
        
        # 動態獲取配置
        if config_adapter is not None:
            config = config_adapter.get_dynamic_config()
            base_dim = config.get('model_dim', 768)
        else:
            try:
                from oanda_trading_bot.training_system.common.config import TRANSFORMER_MODEL_DIM
                base_dim = TRANSFORMER_MODEL_DIM
            except ImportError:
                base_dim = 768
        
        self.base_feature_dim = base_feature_dim or max(100, base_dim // 8)
        self.engineered_feature_dim = engineered_feature_dim or max(50, base_dim // 16)
        self.transformation_layers = transformation_layers
        
        # 特徵變換網絡
        self.feature_transformers = nn.ModuleList()
        current_dim = self.base_feature_dim
        
        for i in range(transformation_layers):
            if i == transformation_layers - 1:
                output_dim = self.engineered_feature_dim
            else:
                output_dim = min(current_dim * 2, base_dim // 2)  # 防止維度過大
            
            transformer = nn.Sequential(
                nn.Linear(current_dim, output_dim),
                nn.GELU(),
                nn.LayerNorm(output_dim),
                nn.Dropout(0.1)
            )
            
            self.feature_transformers.append(transformer)
            current_dim = output_dim
        
        # 特徵選擇器
        selector_intermediate_dim = max(64, self.engineered_feature_dim // 2)
        self.feature_selector = nn.Sequential(
            nn.Linear(self.engineered_feature_dim, selector_intermediate_dim),
            nn.GELU(),
            nn.Linear(selector_intermediate_dim, self.engineered_feature_dim),
            nn.Sigmoid()
        )
        
        # 特徵重要性評估器
        importance_intermediate_dim = max(32, self.engineered_feature_dim // 2)
        self.importance_evaluator = nn.Sequential(
            nn.Linear(self.engineered_feature_dim, importance_intermediate_dim),
            nn.GELU(),
            nn.Linear(importance_intermediate_dim, self.engineered_feature_dim),
            nn.Softmax(dim=-1)
        )
        
        logger.info(f"初始化動態自動特徵工程系統 - 基礎特徵: {self.base_feature_dim}, "
                   f"工程特徵: {self.engineered_feature_dim}, 變換層數: {transformation_layers}")
    
    def forward(self, base_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        自動特徵工程
        
        Args:
            base_features: 基礎特徵 [batch_size, base_feature_dim]
            
        Returns:
            工程特徵字典
        """
        current_features = base_features
        
        # 特徵變換
        for transformer in self.feature_transformers:
            current_features = transformer(current_features)
        
        # 特徵選擇
        selection_weights = self.feature_selector(current_features)
        selected_features = current_features * selection_weights
        
        # 特徵重要性評估
        importance_weights = self.importance_evaluator(selected_features)
        
        return {
            'engineered_features': selected_features,
            'selection_weights': selection_weights,
            'importance_weights': importance_weights,
            'raw_transformed': current_features
        }


class KnowledgeTransferSystem(nn.Module):
    """跨市場知識遷移系統 - 動態維度適應"""
    
    def __init__(self,
                 source_markets: List[str] = None,
                 target_markets: List[str] = None,
                 knowledge_dim: int = None,
                 config_adapter=None):
        super().__init__()
        
        self.source_markets = source_markets or ['forex', 'stocks', 'crypto', 'commodities']
        self.target_markets = target_markets or ['forex']
        
        # 動態獲取配置
        if config_adapter is not None:
            config = config_adapter.get_dynamic_config()
            base_dim = config.get('model_dim', 768)
        else:
            try:
                from oanda_trading_bot.training_system.common.config import TRANSFORMER_MODEL_DIM
                base_dim = TRANSFORMER_MODEL_DIM
            except ImportError:
                base_dim = 768
        
        self.knowledge_dim = knowledge_dim or max(256, base_dim // 3)
        
        # 動態計算中間維度
        encoder_intermediate_dim = max(256, base_dim // 2)
        extractor_intermediate_1 = max(256, base_dim // 2)
        extractor_intermediate_2 = max(128, base_dim // 4)
        adapter_intermediate_1 = max(256, base_dim // 2)
        adapter_intermediate_2 = max(128, base_dim // 4)
        evaluator_intermediate_1 = max(64, base_dim // 8)
        evaluator_intermediate_2 = max(32, base_dim // 16)
        
        # 市場編碼器
        self.market_encoders = nn.ModuleDict({
            market: nn.Sequential(
                nn.Linear(self.knowledge_dim, encoder_intermediate_dim),
                nn.GELU(),
                nn.LayerNorm(encoder_intermediate_dim),
                nn.Linear(encoder_intermediate_dim, self.knowledge_dim),
                nn.Tanh()
            ) for market in self.source_markets
        })
        
        # 知識提取器
        self.knowledge_extractor = nn.Sequential(
            nn.Linear(self.knowledge_dim, extractor_intermediate_1),
            nn.GELU(),
            nn.LayerNorm(extractor_intermediate_1),
            nn.Dropout(0.1),
            nn.Linear(extractor_intermediate_1, extractor_intermediate_2),
            nn.GELU(),
            nn.Linear(extractor_intermediate_2, self.knowledge_dim)
        )
        
        # 知識適配器
        adapter_input_dim = self.knowledge_dim * len(self.source_markets)
        self.knowledge_adapter = nn.Sequential(
            nn.Linear(adapter_input_dim, adapter_intermediate_1),
            nn.GELU(),
            nn.LayerNorm(adapter_intermediate_1),
            nn.Linear(adapter_intermediate_1, adapter_intermediate_2),
            nn.GELU(),
            nn.Linear(adapter_intermediate_2, self.knowledge_dim),
            nn.Tanh()
        )
          # 遷移效果評估器 - 動態適應輸入維度
        evaluator_input_dim = self.knowledge_dim * 2
        self.transfer_evaluator_base = nn.Sequential(
            nn.Linear(evaluator_input_dim, evaluator_intermediate_1),
            nn.GELU(),
            nn.Linear(evaluator_intermediate_1, evaluator_intermediate_2),
            nn.GELU(),
            nn.Linear(evaluator_intermediate_2, 1),
            nn.Sigmoid()
        )
        
        # 動態輸入適配器 - 用於處理不同維度的輸入
        self.transfer_evaluator_adapter = None
          # 知識庫
        self.knowledge_bank = {}
        
        logger.info(f"初始化動态知識遷移系統 - 源市場: {len(self.source_markets)}, "
                   f"目標市場: {len(self.target_markets)}, 知識維度: {self.knowledge_dim}")
    
    def _get_transfer_evaluator(self, input_tensor: torch.Tensor) -> callable:
        """獲取動態適配的遷移評估器"""
        actual_input_dim = input_tensor.size(-1)
        expected_input_dim = self.knowledge_dim * 2
        
        if actual_input_dim == expected_input_dim:
            # 維度匹配，直接使用基礎評估器
            return self.transfer_evaluator_base
        else:
            # 維度不匹配，需要適配器
            adapter_name = f'transfer_evaluator_adapter_{actual_input_dim}_to_{expected_input_dim}'
            
            if not hasattr(self, adapter_name):
                # 創建新的適配器
                adapter = nn.Sequential(
                    nn.Linear(actual_input_dim, expected_input_dim),
                    nn.GELU(),
                    nn.LayerNorm(expected_input_dim),
                    nn.Dropout(0.1)
                ).to(input_tensor.device)
                
                setattr(self, adapter_name, adapter)
                logger.info(f"創建動態遷移評估適配器: {actual_input_dim} -> {expected_input_dim}")
            
            # 組合適配器和基礎評估器
            adapter = getattr(self, adapter_name)
            
            def adapted_evaluator(x):
                adapted_x = adapter(x)
                return self.transfer_evaluator_base(adapted_x)
            
            return adapted_evaluator
    
    def store_market_knowledge(self, market: str, strategies: torch.Tensor, 
                              performance: torch.Tensor):
        """存儲市場知識"""
        if market not in self.knowledge_bank:
            self.knowledge_bank[market] = {
                'strategies': [],
                'performance': [],
                'timestamps': []
            }
        
        self.knowledge_bank[market]['strategies'].append(strategies.detach().cpu())
        self.knowledge_bank[market]['performance'].append(performance.detach().cpu())
        self.knowledge_bank[market]['timestamps'].append(datetime.now().isoformat())
        
        # 保持最近的1000個記錄
        if len(self.knowledge_bank[market]['strategies']) > 1000:
            self.knowledge_bank[market]['strategies'] = self.knowledge_bank[market]['strategies'][-1000:]
            self.knowledge_bank[market]['performance'] = self.knowledge_bank[market]['performance'][-1000:]
            self.knowledge_bank[market]['timestamps'] = self.knowledge_bank[market]['timestamps'][-1000:]
    
    def transfer_knowledge(self, target_market: str, 
                          target_context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        跨市場知識遷移
        
        Args:
            target_market: 目標市場
            target_context: 目標市場上下文
            
        Returns:
            遷移的知識
        """
        transferred_knowledge = []
        transfer_scores = []
        
        for source_market in self.source_markets:
            if source_market == target_market or source_market not in self.knowledge_bank:
                continue
              # 獲取源市場知識
            source_knowledge = self.knowledge_bank[source_market]
            if not source_knowledge['strategies']:
                continue
              # 選擇最佳策略
            best_strategies = []
            best_performance = []
            
            for i, perf in enumerate(source_knowledge['performance']):
                if len(best_strategies) < 10:  # 保持前10個最佳策略
                    best_strategies.append(source_knowledge['strategies'][i])
                    best_performance.append(perf)
                else:
                    min_idx = best_performance.index(min(best_performance))
                    if perf > best_performance[min_idx]:
                        best_strategies[min_idx] = source_knowledge['strategies'][i]
                        best_performance[min_idx] = perf
            
            if best_strategies:                # 編碼源市場策略 - 先取平均得到單一策略表示
                source_strategies = torch.stack(best_strategies).mean(dim=0, keepdim=True)  # 保持batch維度
                
                # 確保源策略維度與知識維度匹配
                if source_strategies.size(-1) != self.knowledge_dim:
                    if not hasattr(self, f'strategy_projector_{source_market}'):
                        projector = nn.Linear(source_strategies.size(-1), self.knowledge_dim).to(source_strategies.device)
                        setattr(self, f'strategy_projector_{source_market}', projector)
                    source_strategies = getattr(self, f'strategy_projector_{source_market}')(source_strategies)
                
                encoded_source = self.market_encoders[source_market](source_strategies)
                
                # 提取可遷移知識
                transferable_knowledge = self.knowledge_extractor(encoded_source)
                  # 評估遷移效果 - 確保維度匹配
                target_context_dim = target_context.size(-1)
                transferable_knowledge_dim = transferable_knowledge.size(-1)
                
                if target_context_dim != transferable_knowledge_dim:
                    # 將target_context投影到transferable_knowledge的維度
                    if not hasattr(self, 'context_adapter'):
                        self.context_adapter = nn.Linear(target_context_dim, transferable_knowledge_dim).to(target_context.device)
                    adapted_target_context = self.context_adapter(target_context)
                else:
                    adapted_target_context = target_context
                
                # 確保兩個張量的維度匹配
                if adapted_target_context.dim() != transferable_knowledge.dim():
                    if adapted_target_context.dim() == 1:
                        adapted_target_context = adapted_target_context.unsqueeze(0)
                    if transferable_knowledge.dim() == 1:
                        transferable_knowledge = transferable_knowledge.unsqueeze(0)
                    
                    # 處理維度不匹配的情況（例如2D vs 3D）
                    if adapted_target_context.dim() < transferable_knowledge.dim():
                        # 如果 adapted_target_context 維度較少，增加維度
                        while adapted_target_context.dim() < transferable_knowledge.dim():
                            adapted_target_context = adapted_target_context.unsqueeze(1)
                    elif transferable_knowledge.dim() < adapted_target_context.dim():
                        # 如果 transferable_knowledge 維度較少，增加維度
                        while transferable_knowledge.dim() < adapted_target_context.dim():
                            transferable_knowledge = transferable_knowledge.unsqueeze(1)
                
                # 如果仍然維度不匹配，展平到2D
                if adapted_target_context.dim() != transferable_knowledge.dim() or adapted_target_context.dim() > 2:
                    adapted_target_context = adapted_target_context.view(adapted_target_context.size(0), -1)
                    transferable_knowledge = transferable_knowledge.view(transferable_knowledge.size(0), -1)                # 確保batch維度匹配 - 適配到較小的batch 大小或廣播
                if adapted_target_context.size(0) != transferable_knowledge.size(0):
                    # 如果其中一個是單batch，擴展到匹配
                    if adapted_target_context.size(0) == 1:
                        # 使用repeat而不是expand來避免維度不匹配問題
                        repeat_dims = [transferable_knowledge.size(0)] + [1] * (adapted_target_context.dim() - 1)
                        adapted_target_context = adapted_target_context.repeat(*repeat_dims)
                    elif transferable_knowledge.size(0) == 1:
                        # 使用repeat而不是expand來避免維度不匹配問題
                        repeat_dims = [adapted_target_context.size(0)] + [1] * (transferable_knowledge.dim() - 1)
                        transferable_knowledge = transferable_knowledge.repeat(*repeat_dims)
                    else:# 如果都不是單batch且大小不同，取第一個樣本
                        min_batch = min(adapted_target_context.size(0), transferable_knowledge.size(0))
                        adapted_target_context = adapted_target_context[:min_batch]
                        transferable_knowledge = transferable_knowledge[:min_batch]
                
                combined_context = torch.cat([adapted_target_context, transferable_knowledge], dim=-1)
                
                # 使用動態適配的遷移評估器
                transfer_evaluator = self._get_transfer_evaluator(combined_context)
                transfer_score = transfer_evaluator(combined_context)
                
                transferred_knowledge.append(transferable_knowledge)
                transfer_scores.append(transfer_score)
        
        if transferred_knowledge:
            # 融合多源知識
            all_knowledge = torch.stack(transferred_knowledge)
            all_scores = torch.stack(transfer_scores)
              # 加權融合
            weights = F.softmax(all_scores.squeeze(), dim=0)
            fused_knowledge = (all_knowledge * weights.unsqueeze(-1)).sum(dim=0)
            
            # 確保融合知識的維度正確 - 應該是 [batch_size, knowledge_dim * num_sources]
            if fused_knowledge.dim() == 2:
                # 如果是2D，需要重塑為正確的維度
                expected_dim = self.knowledge_dim * len(self.source_markets)
                if fused_knowledge.size(-1) != expected_dim:
                    # 創建動態適配器以匹配預期維度
                    if not hasattr(self, 'fusion_adapter'):
                        self.fusion_adapter = nn.Linear(fused_knowledge.size(-1), expected_dim).to(fused_knowledge.device)
                    fused_knowledge = self.fusion_adapter(fused_knowledge)
            
            # 適配到目標市場
            adapted_knowledge = self.knowledge_adapter(fused_knowledge)
            
            return {
                'adapted_knowledge': adapted_knowledge,
                'source_knowledge': all_knowledge,
                'transfer_scores': all_scores,
                'fusion_weights': weights
            }
        
        return {
            'adapted_knowledge': torch.zeros_like(target_context),
            'source_knowledge': None,
            'transfer_scores': None,
            'fusion_weights': None
        }


class StrategyInnovationModule(nn.Module):
    """策略創新模組主類 - 完全動態維度適應"""
    
    def __init__(self,
                 input_dim: int = None,
                 hidden_dim: int = None,
                 population_size: int = 50,
                 max_generations: int = 100,
                 config_adapter=None):
        super().__init__()
        
        # 動態獲取配置
        if config_adapter is not None:
            config = config_adapter.get_dynamic_config()
            base_dim = config.get('model_dim', 768)
        else:
            try:
                from oanda_trading_bot.training_system.common.config import TRANSFORMER_MODEL_DIM
                base_dim = TRANSFORMER_MODEL_DIM
            except ImportError:
                base_dim = 768
        
        self.input_dim = input_dim or base_dim
        self.hidden_dim = hidden_dim or base_dim
        self.population_size = population_size
        self.max_generations = max_generations
        
        # 動態計算策略維度
        strategy_dim = max(256, base_dim // 3)
        knowledge_dim = max(256, base_dim // 3)
        
        # 添加維度適配層
        self.strategy_projector = nn.Linear(self.hidden_dim, strategy_dim)
        
        # 初始化子模組 - 傳遞配置適配器
        self.strategy_generator = StrategyGeneratorTransformer(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            config_adapter=config_adapter
        )
        
        self.strategy_evaluator = StrategyEvaluator(
            strategy_dim=strategy_dim,
            config_adapter=config_adapter
        )
        
        self.evolution_engine = StrategyEvolutionEngine(
            population_size=population_size
        )
        
        self.feature_engineer = AutomaticFeatureEngineer(
            base_feature_dim=self.input_dim,
            engineered_feature_dim=strategy_dim, # 確保維度一致
            config_adapter=config_adapter
        )
        
        self.knowledge_transfer = KnowledgeTransferSystem(
            knowledge_dim=knowledge_dim, # 確保維度一致
            config_adapter=config_adapter
        )
        
        # 創新統計
        self.innovation_history = []
        self.best_strategies = []
        
        logger.info(f"初始化動态策略創新模組 - 輸入維度: {self.input_dim}, "
                   f"隱藏維度: {self.hidden_dim}, 策略維度: {strategy_dim}, "
                   f"種群大小: {population_size}")

    def get_dynamic_config(self) -> Dict[str, Any]:
        """獲取動態配置信息"""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'strategy_dim': getattr(self.strategy_evaluator, 'strategy_dim', 256),
            'feature_dim': getattr(self.feature_engineer, 'engineered_feature_dim', 256),
            'knowledge_dim': getattr(self.knowledge_transfer, 'knowledge_dim', 256)
        }

    def _calculate_strategy_diversity(self, strategies_tensor: torch.Tensor) -> torch.Tensor:
        """
        計算策略張量的多樣性。
        一個簡單的實現是計算策略表示的標準差。
        """
        if strategies_tensor is None or strategies_tensor.numel() == 0:
            return torch.tensor(0.0, device=strategies_tensor.device if strategies_tensor is not None else 'cpu')
        
        if strategies_tensor.dim() < 2: # 至少需要 batch_size x strategy_dim
            return torch.tensor(0.0, device=strategies_tensor.device)
            
        # 沿策略維度計算標準差，然後取平均值
        # 假設 strategies_tensor 的形狀是 [batch_size, num_strategies, strategy_dim] 或 [batch_size, strategy_dim]
        if strategies_tensor.dim() == 3: # batch_size x num_strategies x strategy_dim
            if strategies_tensor.shape[1] <= 1: # 單一策略或沒有策略
                 return torch.tensor(0.0, device=strategies_tensor.device)
            diversity = torch.std(strategies_tensor, dim=1).mean() # 計算不同策略之間的標準差的平均值
        elif strategies_tensor.dim() == 2: # batch_size x features (每個 batch item 是一個策略)
            if strategies_tensor.shape[0] <= 1: # 單個策略或空
                 return torch.tensor(0.0, device=strategies_tensor.device)
            diversity = torch.std(strategies_tensor, dim=0).mean() # 特徵間的標準差的平均
        else: # 其他情況，無法明確計算
            diversity = torch.tensor(0.0, device=strategies_tensor.device)
            
        return diversity.squeeze()


    def forward(self, market_context: torch.Tensor,
                existing_strategies: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        執行策略創新流程
        
        Args:
            market_context: 市場上下文
            existing_strategies: 現有策略
            
        Returns:
            創新結果
        """
        # 自動特徵工程
        feature_result = self.feature_engineer(market_context)
        engineered_features = feature_result['engineered_features']
        
        # 跨市場知識遷移
        transfer_result = self.knowledge_transfer.transfer_knowledge(
            target_market='forex',
            target_context=engineered_features
        )
        
        # 融合工程特徵和遷移知識
        if transfer_result['adapted_knowledge'] is not None:
            enhanced_context = engineered_features + transfer_result['adapted_knowledge']
        else:
            enhanced_context = engineered_features        # 生成新策略
        generation_result = self.strategy_generator(
            enhanced_context, 
            existing_strategies
        )
        
        # 將編碼上下文投影到策略維度
        projected_context = self.strategy_projector(generation_result['encoded_context'])
        
        # 評估策略
        evaluation_result = self.strategy_evaluator(
            projected_context
        )
        
        # 提取實際的策略張量 - 組合所有策略參數為一個張量
        # 將各種策略參數組合成一個統一的策略表示
        strategy_components = [
            generation_result['architecture_params'],
            generation_result['hyperparameters'], 
            generation_result['feature_weights'],
            generation_result['objective_weights']
        ]
        # 沿最後一個維度連接所有組件
        generated_strategies_tensor = torch.cat(strategy_components, dim=-1)
        
        # 記錄創新統計
        innovation_record = {
            'timestamp': datetime.now().isoformat(),
            'fitness_score': float(evaluation_result['fitness_score'].mean()),
            'innovation_score': float(evaluation_result['innovation_score'].mean()),
            'feature_importance': feature_result['importance_weights'].mean(dim=0).tolist(),
            'transfer_score': float(transfer_result['transfer_scores'].mean()) 
                            if transfer_result['transfer_scores'] is not None else 0.0
        }
        self.innovation_history.append(innovation_record)
        
        return {
            'generated_strategies': generated_strategies_tensor,
            'innovation_confidence': evaluation_result['innovation_score'],
            'strategy_diversity': self._calculate_strategy_diversity(generated_strategies_tensor),
            'generation_details': generation_result,
            'evaluation': evaluation_result,
            'feature_engineering': feature_result,
            'knowledge_transfer': transfer_result,
            'innovation_metrics': innovation_record
        }
    
    def evolve_strategies(self, market_context: torch.Tensor, 
                         num_generations: int = 10) -> List[StrategyGenome]:
        """
        執行策略進化
        
        Args:
            market_context: 市場上下文
            num_generations: 進化代數
            
        Returns:
            進化後的策略種群
        """
        # 初始化種群
        if not self.evolution_engine.current_population:
            self.evolution_engine.initialize_population(
                self.strategy_generator, 
                market_context
            )
        
        best_strategies = []
        
        for generation in range(num_generations):
            # 評估當前種群
            fitness_scores = self.evolution_engine.evaluate_population(
                self.strategy_evaluator
            )
            
            # 記錄最佳策略
            best_idx = fitness_scores.index(max(fitness_scores))
            best_strategy = self.evolution_engine.current_population[best_idx]
            best_strategies.append(copy.deepcopy(best_strategy))
            
            # 進化下一代
            if generation < num_generations - 1:
                self.evolution_engine.evolve_generation()
            
            logger.info(f"進化代數 {generation + 1}/{num_generations} - "
                       f"最佳適應度: {max(fitness_scores):.4f}")
        
        self.best_strategies.extend(best_strategies)
        return self.evolution_engine.current_population
    
    def get_innovation_statistics(self) -> Dict[str, Any]:
        """獲取創新統計信息"""
        if not self.innovation_history:
            return {'message': '暫無創新記錄'}
        
        recent_records = self.innovation_history[-100:]  # 最近100條記錄
        
        avg_fitness = sum(r['fitness_score'] for r in recent_records) / len(recent_records)
        avg_innovation = sum(r['innovation_score'] for r in recent_records) / len(recent_records)
        avg_transfer = sum(r['transfer_score'] for r in recent_records) / len(recent_records)
        
        return {
            'total_innovations': len(self.innovation_history),
            'recent_average_fitness': avg_fitness,
            'recent_average_innovation': avg_innovation,
            'recent_average_transfer': avg_transfer,
            'evolution_generations': self.evolution_engine.generation_counter,
            'population_size': len(self.evolution_engine.current_population),
            'best_strategies_count': len(self.best_strategies)
        }
class QuantumInspiredGenerator:
    """量子啟發式策略生成器
    使用量子隨機遊走算法生成創新交易策略
    """
    
    def __init__(self, num_strategies: int = 10, strategy_dim: int = 256):
        """
        初始化量子策略生成器
        
        Args:
            num_strategies: 生成的策略數量
            strategy_dim: 策略參數的維度
        """
        self.num_strategies = num_strategies
        self.strategy_dim = strategy_dim
        self.quantum_states = None
        self.current_position = 0
        
    def generate_strategy(self, market_state: torch.Tensor) -> torch.Tensor:
        """
        基於市場狀態生成策略組合
        
        Args:
            market_state: 市場狀態張量 [batch_size, state_dim]
            
        Returns:
            生成的策略參數 [batch_size, num_strategies, strategy_dim]
        """
        batch_size = market_state.size(0)
        device = market_state.device
        
        # 初始化量子態 (如果未初始化或批次大小改變)
        if self.quantum_states is None or self.quantum_states.size(0) != batch_size:
            # 使用市場狀態作為初始量子態的基礎
            self.quantum_states = torch.randn(batch_size, self.num_strategies, self.strategy_dim, device=device)
            self.current_position = 0
        
        # 量子隨機遊走算法
        # 1. 量子疊加: 混合當前量子態
        superposition = torch.fft.fft(self.quantum_states, dim=-1)
        
        # 2. 相位擾動: 加入市場狀態影響
        market_influence = market_state.unsqueeze(1)  # [batch_size, 1, state_dim]
        # 確保維度匹配
        if market_influence.size(-1) < self.strategy_dim:
            padding = torch.zeros(batch_size, 1, self.strategy_dim - market_influence.size(-1),
                                device=device)
            market_influence = torch.cat([market_influence, padding], dim=-1)
        elif market_influence.size(-1) > self.strategy_dim:
            market_influence = market_influence[..., :self.strategy_dim]
            
        # 應用相位擾動
        phase_shift = torch.exp(1j * market_influence * 0.1)
        superposition = superposition * phase_shift
        
        # 3. 逆傅立葉變換返回時域
        new_states = torch.fft.ifft(superposition, dim=-1).real
        
        # 4. 量子測量: 選取當前策略
        strategies = new_states[:, self.current_position, :].unsqueeze(1)
        
        # 更新量子態和位置
        self.quantum_states = new_states
        self.current_position = (self.current_position + 1) % self.num_strategies
        
        return strategies

class StateAwareAdapter:
    """狀態感知策略適配器
    基於市場狀態動態調整策略參數，整合波動率感知和風險偏好
    """
    
    def __init__(self, volatility_factor: float = 0.5, risk_aversion: float = 0.7):
        """
        初始化適配器
        
        Args:
            volatility_factor: 波動率影響因子 (0-1)
            risk_aversion: 風險規避係數 (0-1)
        """
        self.volatility_factor = volatility_factor
        self.risk_aversion = risk_aversion
        
    def adapt_strategy(self, strategy: torch.Tensor, market_state: torch.Tensor) -> torch.Tensor:
        """
        動態適配策略參數
        
        Args:
            strategy: 原始策略參數 [batch_size, strategy_dim]
            market_state: 市場狀態張量 [batch_size, state_dim]
            
        Returns:
            適配後的策略參數 [batch_size, strategy_dim]
        """
        # 提取波動率信號 (假設market_state最後一個維度是波動率)
        volatility = market_state[..., -1].unsqueeze(-1)  # [batch_size, 1]
        
        # 波動率感知調整
        # 高波動率時降低策略攻擊性
        volatility_adjustment = 1.0 - (volatility * self.volatility_factor)
        
        # 風險偏好調整
        risk_adjustment = 1.0 - self.risk_aversion
        
        # 組合調整因子
        adjustment_factor = volatility_adjustment * risk_adjustment
        
        # 確保維度匹配
        if adjustment_factor.size(-1) < strategy.size(-1):
            padding = torch.ones(strategy.size(0), strategy.size(-1) - adjustment_factor.size(-1),
                               device=strategy.device)
            adjustment_factor = torch.cat([adjustment_factor, padding], dim=-1)
        elif adjustment_factor.size(-1) > strategy.size(-1):
            adjustment_factor = adjustment_factor[..., :strategy.size(-1)]
            
        # 應用調整
        adapted_strategy = strategy * adjustment_factor
        
        return adapted_strategy


class ConfigAdapter:
    """配置適配器 - 統一管理動態維度配置"""
    
    def __init__(self, enhanced_transformer=None):
        self.enhanced_transformer = enhanced_transformer
        self._cached_config = None
    
    def get_dynamic_config(self) -> Dict[str, Any]:
        """獲取動態配置"""
        if self._cached_config is not None:
            return self._cached_config
        
        # 優先從Enhanced Transformer獲取配置
        if self.enhanced_transformer is not None and hasattr(self.enhanced_transformer, 'get_dynamic_config'):
            config = self.enhanced_transformer.get_dynamic_config()
            self._cached_config = config
            return config
        
        # 從配置文件獲取
        try:
            from oanda_trading_bot.training_system.common.config import (
                TRANSFORMER_MODEL_DIM, TRANSFORMER_NUM_LAYERS,
                TRANSFORMER_NUM_HEADS, TRANSFORMER_FFN_DIM,
                TRANSFORMER_OUTPUT_DIM_PER_SYMBOL
            )
            config = {
                'model_dim': TRANSFORMER_MODEL_DIM,
                'num_layers': TRANSFORMER_NUM_LAYERS,
                'num_heads': TRANSFORMER_NUM_HEADS,
                'ffn_dim': TRANSFORMER_FFN_DIM,
                'output_dim_per_symbol': TRANSFORMER_OUTPUT_DIM_PER_SYMBOL,
                'head_dim': TRANSFORMER_MODEL_DIM // TRANSFORMER_NUM_HEADS
            }
        except ImportError:
            # 後備配置 (Large Model)
            config = {
                'model_dim': 768,
                'num_layers': 16,
                'num_heads': 24,
                'ffn_dim': 3072,
                'output_dim_per_symbol': 192,
                'head_dim': 32
            }
        
        self._cached_config = config
        return config
    
    def invalidate_cache(self):
        """使緩存失效"""
        self._cached_config = None



def create_strategy_innovation_module(enhanced_transformer=None, **kwargs) -> StrategyInnovationModule:
    """創建策略創新模組的工廠函數"""
    config_adapter = ConfigAdapter(enhanced_transformer)
    
    return StrategyInnovationModule(
        config_adapter=config_adapter,
        **kwargs
    )


# 測試函數
def test_strategy_innovation_module():
    """測試策略創新模組 - 動態維度適應版本"""
    try:
        logger.info("開始測試動態策略創新模組...")
        
        # 創建配置適配器
        config_adapter = ConfigAdapter()
        config = config_adapter.get_dynamic_config()
        
        logger.info(f"動態配置: {config}")
        
        # 設置測試參數
        batch_size = 4
        input_dim = config['model_dim']
        
        # 初始化模組
        innovation_module = StrategyInnovationModule(
            input_dim=input_dim,
            population_size=20,
            max_generations=5,
            config_adapter=config_adapter
        )
        
        # 創建測試輸入 - 動態維度
        market_context = torch.randn(batch_size, input_dim)
        strategy_dim = max(256, input_dim // 3)
        existing_strategies = torch.randn(batch_size, 10, strategy_dim)  # 10個現有策略
        
        logger.info(f"測試輸入維度 - 市場上下文: {market_context.shape}, "
                   f"現有策略: {existing_strategies.shape}")
        
        # 測試創新流程
        with torch.no_grad():
            innovation_result = innovation_module(market_context, existing_strategies)
            
            logger.info("✅ 策略創新流程測試成功")
            logger.info(f"生成策略輸出維度: {innovation_result['generated_strategy']['architecture_params'].shape}")
            logger.info(f"評估分數: {float(innovation_result['evaluation']['fitness_score'].mean()):.4f}")
            
            # 測試進化過程
            logger.info("測試策略進化...")
            evolved_strategies = innovation_module.evolve_strategies(
                market_context, num_generations=3
            )
            
            logger.info(f"✅ 策略進化測試成功 - 進化了 {len(evolved_strategies)} 個策略")
            
            # 測試統計信息
            stats = innovation_module.get_innovation_statistics()
            logger.info(f"創新統計: {stats}")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ 策略創新模組測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 運行測試
    success = test_strategy_innovation_module()
    if success:
        logger.info("🎉 所有測試通過！動態策略創新模組運行正常")
    else:
        logger.error("💥 測試失敗，需要進一步調試")
