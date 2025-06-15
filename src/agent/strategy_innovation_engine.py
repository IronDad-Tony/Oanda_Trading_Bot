# src/agent/strategy_innovation_engine.py
"""
策略創新引擎 (Strategy Innovation Engine)

此模組實現了自動策略創新和進化的核心功能，包括：
1. 基因算法策略進化
2. 神經架構搜索 (Neural Architecture Search)
3. 自動特徵工程
4. 策略性能優化

主要類：
- StrategyInnovationEngine: 主要引擎類，協調各種創新方法
- GeneticStrategyEvolution: 基因算法實現
- NeuralArchitectureSearch: 神經架構搜索實現
- AutoFeatureEngineering: 自動特徵工程實現
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import logging
from dataclasses import dataclass
import random
import copy
from datetime import datetime
import json

# 添加項目根目錄到路徑
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.common.logger_setup import logger
    from src.common.config import DEVICE
    from src.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition
    from src.agent.meta_learning_system import MetaLearningSystem, MarketKnowledgeBase
except ImportError as e:
    # 基礎日誌設定
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    DEVICE = "cpu"
    
    # 創建空的替代類以避免錯誤
    class EnhancedStrategySuperposition:
        pass
    
    class MetaLearningSystem:
        pass
    
    class MarketKnowledgeBase:
        pass
    
    logger.warning(f"導入錯誤，使用基礎配置: {e}")


@dataclass
class StrategyGenome:
    """策略基因結構"""
    strategy_type: str
    parameters: Dict[str, Any]
    architecture: Dict[str, Any]
    fitness_score: float = 0.0
    generation: int = 0
    parent_ids: List[str] = None
    mutation_history: List[str] = None
    
    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []
        if self.mutation_history is None:
            self.mutation_history = []


@dataclass 
class ArchitectureCandidate:
    """神經架構候選結構"""
    layers: List[Dict[str, Any]]
    connections: List[Tuple[int, int]]
    hyperparameters: Dict[str, Any]
    estimated_performance: float = 0.0
    complexity_score: float = 0.0
    
    
@dataclass
class FeatureCandidate:
    """特徵候選結構"""
    feature_name: str
    transformation: Callable
    input_features: List[str]
    complexity: int
    importance_score: float = 0.0


class GeneticStrategyEvolution:
    """
    基因算法策略進化器
    
    實現策略參數和結構的進化優化
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 elite_ratio: float = 0.2,
                 max_generations: int = 100):
        """
        初始化基因算法參數
        
        Args:
            population_size: 種群大小
            mutation_rate: 突變率
            crossover_rate: 交叉率  
            elite_ratio: 精英保留比例
            max_generations: 最大進化代數
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.max_generations = max_generations
        
        self.population: List[StrategyGenome] = []
        self.generation_history: List[Dict[str, Any]] = []
        self.best_genome: Optional[StrategyGenome] = None
        
        logger.info(f"基因算法策略進化器已初始化 - 種群大小: {population_size}, "
                   f"突變率: {mutation_rate}, 交叉率: {crossover_rate}")
    
    def initialize_population(self, strategy_templates: List[Dict[str, Any]]) -> None:
        """
        初始化種群
        
        Args:
            strategy_templates: 策略模板列表
        """
        self.population = []
        
        for i in range(self.population_size):
            # 隨機選擇策略模板
            template = random.choice(strategy_templates)
            
            # 創建基因組
            genome = StrategyGenome(
                strategy_type=template['type'],
                parameters=self._randomize_parameters(template['parameters']),
                architecture=self._randomize_architecture(template['architecture']),
                generation=0
            )
            
            self.population.append(genome)
        
        logger.info(f"種群已初始化，包含 {len(self.population)} 個策略基因組")
    
    def _randomize_parameters(self, param_template: Dict[str, Any]) -> Dict[str, Any]:
        """隨機化策略參數"""
        randomized = {}
        
        for param_name, param_config in param_template.items():
            param_type = param_config.get('type', 'float')
            param_range = param_config.get('range', [0, 1])
            
            if param_type == 'float':
                randomized[param_name] = random.uniform(param_range[0], param_range[1])
            elif param_type == 'int':
                randomized[param_name] = random.randint(param_range[0], param_range[1])
            elif param_type == 'choice':
                randomized[param_name] = random.choice(param_config.get('choices', []))
            elif param_type == 'bool':
                randomized[param_name] = random.choice([True, False])
        
        return randomized
    
    def _randomize_architecture(self, arch_template: Dict[str, Any]) -> Dict[str, Any]:
        """隨機化策略架構"""
        randomized = copy.deepcopy(arch_template)
        
        # 隨機化層數
        if 'num_layers' in arch_template:
            layer_range = arch_template['num_layers'].get('range', [2, 8])
            randomized['num_layers'] = random.randint(layer_range[0], layer_range[1])
        
        # 隨機化隱藏維度
        if 'hidden_dims' in arch_template:
            dim_choices = arch_template['hidden_dims'].get('choices', [64, 128, 256, 512])
            num_layers = randomized.get('num_layers', 4)
            randomized['hidden_dims'] = [random.choice(dim_choices) for _ in range(num_layers)]
        
        # 隨機化激活函數
        if 'activation' in arch_template:
            activations = arch_template['activation'].get('choices', ['relu', 'tanh', 'gelu'])
            randomized['activation'] = random.choice(activations)
        
        return randomized
    
    def evaluate_fitness(self, genome: StrategyGenome, 
                        fitness_function: Callable[[StrategyGenome], float]) -> float:
        """
        評估基因組適應度
        
        Args:
            genome: 策略基因組
            fitness_function: 適應度評估函數
            
        Returns:
            適應度分數
        """
        try:
            fitness = fitness_function(genome)
            genome.fitness_score = fitness
            return fitness
        except Exception as e:
            logger.error(f"適應度評估失敗: {e}")
            genome.fitness_score = 0.0
            return 0.0
    
    def selection(self, k: int = 3) -> StrategyGenome:
        """
        錦標賽選擇
        
        Args:
            k: 錦標賽大小
            
        Returns:
            選中的基因組
        """
        tournament = random.sample(self.population, min(k, len(self.population)))
        return max(tournament, key=lambda g: g.fitness_score)
    
    def crossover(self, parent1: StrategyGenome, parent2: StrategyGenome) -> Tuple[StrategyGenome, StrategyGenome]:
        """
        交叉操作
        
        Args:
            parent1: 父基因組1
            parent2: 父基因組2
            
        Returns:
            兩個子基因組
        """
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        # 參數交叉
        child1_params = {}
        child2_params = {}
        
        for param_name in parent1.parameters.keys():
            if random.random() < 0.5:
                child1_params[param_name] = parent1.parameters[param_name]
                child2_params[param_name] = parent2.parameters[param_name]
            else:
                child1_params[param_name] = parent2.parameters[param_name]
                child2_params[param_name] = parent1.parameters[param_name]
        
        # 架構交叉（簡化版）
        child1_arch = copy.deepcopy(parent1.architecture)
        child2_arch = copy.deepcopy(parent2.architecture)
        
        if random.random() < 0.5:
            child1_arch, child2_arch = child2_arch, child1_arch
        
        # 創建子基因組
        child1 = StrategyGenome(
            strategy_type=parent1.strategy_type,
            parameters=child1_params,
            architecture=child1_arch,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[id(parent1), id(parent2)]
        )
        
        child2 = StrategyGenome(
            strategy_type=parent2.strategy_type,
            parameters=child2_params,
            architecture=child2_arch,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[id(parent1), id(parent2)]
        )
        
        return child1, child2
    
    def mutate(self, genome: StrategyGenome) -> StrategyGenome:
        """
        突變操作
        
        Args:
            genome: 要突變的基因組
            
        Returns:
            突變後的基因組
        """
        mutated = copy.deepcopy(genome)
        mutations = []
        
        # 參數突變
        for param_name, param_value in mutated.parameters.items():
            if random.random() < self.mutation_rate:
                if isinstance(param_value, float):
                    # 高斯突變
                    mutated.parameters[param_name] = param_value + random.gauss(0, 0.1)
                    mutations.append(f"param_{param_name}_gaussian")
                elif isinstance(param_value, int):
                    # 整數突變
                    mutated.parameters[param_name] = max(1, param_value + random.randint(-2, 2))
                    mutations.append(f"param_{param_name}_integer")
                elif isinstance(param_value, bool):
                    # 布爾翻轉
                    mutated.parameters[param_name] = not param_value
                    mutations.append(f"param_{param_name}_flip")
        
        # 架構突變
        if random.random() < self.mutation_rate:
            if 'num_layers' in mutated.architecture:
                old_layers = mutated.architecture['num_layers']
                mutated.architecture['num_layers'] = max(1, old_layers + random.randint(-1, 1))
                mutations.append(f"arch_num_layers_{old_layers}_to_{mutated.architecture['num_layers']}")
        
        if random.random() < self.mutation_rate:
            if 'hidden_dims' in mutated.architecture and mutated.architecture['hidden_dims']:
                idx = random.randint(0, len(mutated.architecture['hidden_dims']) - 1)
                old_dim = mutated.architecture['hidden_dims'][idx]
                choices = [32, 64, 128, 256, 512, 1024]
                mutated.architecture['hidden_dims'][idx] = random.choice(choices)
                mutations.append(f"arch_hidden_dim_{idx}_{old_dim}_to_{mutated.architecture['hidden_dims'][idx]}")
        
        mutated.mutation_history.extend(mutations)
        return mutated
    
    def evolve_generation(self, fitness_function: Callable[[StrategyGenome], float]) -> Dict[str, Any]:
        """
        進化一代
        
        Args:
            fitness_function: 適應度評估函數
            
        Returns:
            這一代的統計信息
        """
        # 評估當前種群適應度
        for genome in self.population:
            self.evaluate_fitness(genome, fitness_function)
        
        # 排序種群
        self.population.sort(key=lambda g: g.fitness_score, reverse=True)
        
        # 記錄統計信息
        stats = {
            'generation': self.population[0].generation + 1,
            'best_fitness': self.population[0].fitness_score,
            'avg_fitness': np.mean([g.fitness_score for g in self.population]),
            'worst_fitness': self.population[-1].fitness_score,
            'diversity': self._calculate_diversity()
        }
        
        # 更新最佳基因組
        if self.best_genome is None or self.population[0].fitness_score > self.best_genome.fitness_score:
            self.best_genome = copy.deepcopy(self.population[0])
        
        # 選擇精英
        elite_size = int(self.population_size * self.elite_ratio)
        elite = self.population[:elite_size]
        
        # 生成新種群
        new_population = copy.deepcopy(elite)
        
        while len(new_population) < self.population_size:
            parent1 = self.selection()
            parent2 = self.selection()
            
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        # 截斷到指定大小
        self.population = new_population[:self.population_size]
        
        # 更新代數
        for genome in self.population:
            genome.generation = stats['generation']
        
        self.generation_history.append(stats)
        
        logger.info(f"第 {stats['generation']} 代進化完成 - "
                   f"最佳適應度: {stats['best_fitness']:.4f}, "
                   f"平均適應度: {stats['avg_fitness']:.4f}")
        
        return stats
    
    def _calculate_diversity(self) -> float:
        """計算種群多樣性"""
        if len(self.population) < 2:
            return 0.0
        
        # 簡化的多樣性計算：參數值的標準差平均值
        param_stds = []
        
        for param_name in self.population[0].parameters.keys():
            values = [g.parameters[param_name] for g in self.population 
                     if isinstance(g.parameters[param_name], (int, float))]
            if values:
                param_stds.append(np.std(values))
        
        return np.mean(param_stds) if param_stds else 0.0


class NeuralArchitectureSearch:
    """
    神經架構搜索 (Neural Architecture Search, NAS)
    
    自動搜索最優的神經網絡架構
    """
    
    def __init__(self, 
                 search_space: Dict[str, Any],
                 max_architectures: int = 100,
                 performance_threshold: float = 0.8):
        """
        初始化NAS參數
        
        Args:
            search_space: 搜索空間定義
            max_architectures: 最大搜索架構數量
            performance_threshold: 性能閾值
        """
        self.search_space = search_space
        self.max_architectures = max_architectures
        self.performance_threshold = performance_threshold
        
        self.candidates: List[ArchitectureCandidate] = []
        self.evaluated_architectures: List[Dict[str, Any]] = []
        
        logger.info(f"神經架構搜索已初始化 - 最大架構數: {max_architectures}")
    
    def generate_architecture(self) -> ArchitectureCandidate:
        """
        生成新的架構候選
        
        Returns:
            新的架構候選
        """
        layers = []
        connections = []
        
        # 生成層序列
        num_layers = random.randint(
            self.search_space.get('num_layers', {}).get('min', 2),
            self.search_space.get('num_layers', {}).get('max', 10)
        )
        
        for i in range(num_layers):
            layer_type = random.choice(self.search_space.get('layer_types', ['linear', 'conv1d', 'attention']))
            
            if layer_type == 'linear':
                layer = {
                    'type': 'linear',
                    'input_dim': random.choice(self.search_space.get('hidden_dims', [64, 128, 256])),
                    'output_dim': random.choice(self.search_space.get('hidden_dims', [64, 128, 256])),
                    'activation': random.choice(self.search_space.get('activations', ['relu', 'gelu', 'tanh']))
                }
            elif layer_type == 'conv1d':
                layer = {
                    'type': 'conv1d',
                    'in_channels': random.choice([32, 64, 128]),
                    'out_channels': random.choice([32, 64, 128]),
                    'kernel_size': random.choice([3, 5, 7]),
                    'stride': random.choice([1, 2]),
                    'activation': random.choice(self.search_space.get('activations', ['relu', 'gelu']))
                }
            elif layer_type == 'attention':
                layer = {
                    'type': 'attention',
                    'embed_dim': random.choice([128, 256, 512]),
                    'num_heads': random.choice([4, 8, 16]),
                    'dropout': random.uniform(0.0, 0.3)
                }
            
            layers.append(layer)
            
            # 添加連接（簡化：順序連接）
            if i > 0:
                connections.append((i-1, i))
        
        # 生成超參數
        hyperparameters = {
            'learning_rate': random.uniform(
                self.search_space.get('learning_rate', {}).get('min', 1e-5),
                self.search_space.get('learning_rate', {}).get('max', 1e-2)
            ),
            'batch_size': random.choice(self.search_space.get('batch_sizes', [16, 32, 64, 128])),
            'dropout_rate': random.uniform(0.0, 0.5),
            'weight_decay': random.uniform(1e-6, 1e-3)
        }
        
        candidate = ArchitectureCandidate(
            layers=layers,
            connections=connections,
            hyperparameters=hyperparameters,
            complexity_score=self._calculate_complexity(layers)
        )
        
        return candidate
    
    def _calculate_complexity(self, layers: List[Dict[str, Any]]) -> float:
        """計算架構複雜度"""
        complexity = 0.0
        
        for layer in layers:
            if layer['type'] == 'linear':
                complexity += layer['input_dim'] * layer['output_dim']
            elif layer['type'] == 'conv1d':
                complexity += layer['in_channels'] * layer['out_channels'] * layer['kernel_size']
            elif layer['type'] == 'attention':
                complexity += layer['embed_dim'] * layer['num_heads'] * 2  # Simplified
        
        return complexity
    
    def evaluate_architecture(self, candidate: ArchitectureCandidate, 
                            evaluation_function: Callable[[ArchitectureCandidate], float]) -> float:
        """
        評估架構性能
        
        Args:
            candidate: 架構候選
            evaluation_function: 評估函數
            
        Returns:
            性能分數
        """
        try:
            performance = evaluation_function(candidate)
            candidate.estimated_performance = performance
            
            # 記錄評估結果
            self.evaluated_architectures.append({
                'candidate': candidate,
                'performance': performance,
                'complexity': candidate.complexity_score,
                'timestamp': datetime.now().isoformat()
            })
            
            return performance
        except Exception as e:
            logger.error(f"架構評估失敗: {e}")
            return 0.0
    
    def search(self, evaluation_function: Callable[[ArchitectureCandidate], float]) -> List[ArchitectureCandidate]:
        """
        執行架構搜索
        
        Args:
            evaluation_function: 評估函數
            
        Returns:
            搜索到的最優架構列表
        """
        logger.info("開始神經架構搜索...")
        
        best_architectures = []
        
        for i in range(self.max_architectures):
            # 生成新架構
            candidate = self.generate_architecture()
            
            # 評估性能
            performance = self.evaluate_architecture(candidate, evaluation_function)
            
            # 檢查是否滿足閾值
            if performance >= self.performance_threshold:
                best_architectures.append(candidate)
                logger.info(f"發現優秀架構 {i+1}: 性能={performance:.4f}, 複雜度={candidate.complexity_score:.0f}")
            
            # 早停條件
            if len(best_architectures) >= 10:  # 找到足夠多的好架構
                logger.info(f"已找到 {len(best_architectures)} 個優秀架構，提前終止搜索")
                break
        
        # 按性能排序
        best_architectures.sort(key=lambda x: x.estimated_performance, reverse=True)
        
        logger.info(f"神經架構搜索完成 - 評估了 {i+1} 個架構，找到 {len(best_architectures)} 個優秀架構")
        
        return best_architectures


class AutoFeatureEngineering:
    """
    自動特徵工程
    
    自動生成和選擇有用的特徵
    """
    
    def __init__(self, max_features: int = 100, complexity_limit: int = 5):
        """
        初始化自動特徵工程
        
        Args:
            max_features: 最大特徵數量
            complexity_limit: 特徵複雜度限制
        """
        self.max_features = max_features
        self.complexity_limit = complexity_limit
        
        self.feature_candidates: List[FeatureCandidate] = []
        self.selected_features: List[FeatureCandidate] = []
        
        # 基礎變換函數
        self.transformations = {
            'log': lambda x: np.log(np.abs(x) + 1e-8),
            'sqrt': lambda x: np.sqrt(np.abs(x)),
            'square': lambda x: x ** 2,
            'reciprocal': lambda x: 1.0 / (x + 1e-8),
            'exp': lambda x: np.exp(np.clip(x, -10, 10)),
            'tanh': lambda x: np.tanh(x),
            'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))),
            'diff': lambda x: np.diff(x, prepend=x[0]) if len(x.shape) > 0 else 0,
            'rolling_mean': lambda x: self._rolling_mean(x, 5),
            'rolling_std': lambda x: self._rolling_std(x, 5),
            'ema': lambda x: self._exponential_moving_average(x, 0.1)
        }
        
        logger.info(f"自動特徵工程已初始化 - 最大特徵數: {max_features}")
    
    def _rolling_mean(self, x: np.ndarray, window: int) -> np.ndarray:
        """滾動平均"""
        if len(x) < window:
            return np.full_like(x, np.mean(x))
        
        result = np.zeros_like(x)
        for i in range(len(x)):
            start_idx = max(0, i - window + 1)
            result[i] = np.mean(x[start_idx:i+1])
        return result
    
    def _rolling_std(self, x: np.ndarray, window: int) -> np.ndarray:
        """滾動標準差"""
        if len(x) < window:
            return np.full_like(x, np.std(x))
        
        result = np.zeros_like(x)
        for i in range(len(x)):
            start_idx = max(0, i - window + 1)
            result[i] = np.std(x[start_idx:i+1])
        return result
    
    def _exponential_moving_average(self, x: np.ndarray, alpha: float) -> np.ndarray:
        """指數移動平均"""
        result = np.zeros_like(x)
        result[0] = x[0]
        for i in range(1, len(x)):
            result[i] = alpha * x[i] + (1 - alpha) * result[i-1]
        return result
    
    def generate_features(self, base_features: List[str]) -> List[FeatureCandidate]:
        """
        生成特徵候選
        
        Args:
            base_features: 基礎特徵列表
            
        Returns:
            特徵候選列表
        """
        candidates = []
        
        # 單變量變換
        for feature in base_features:
            for transform_name, transform_func in self.transformations.items():
                candidate = FeatureCandidate(
                    feature_name=f"{feature}_{transform_name}",
                    transformation=transform_func,
                    input_features=[feature],
                    complexity=1
                )
                candidates.append(candidate)
        
        # 雙變量組合
        if self.complexity_limit >= 2:
            for i, feature1 in enumerate(base_features):
                for j, feature2 in enumerate(base_features[i+1:], i+1):
                    # 加法
                    candidates.append(FeatureCandidate(
                        feature_name=f"{feature1}_plus_{feature2}",
                        transformation=lambda x1, x2: x1 + x2,
                        input_features=[feature1, feature2],
                        complexity=2
                    ))
                    
                    # 乘法
                    candidates.append(FeatureCandidate(
                        feature_name=f"{feature1}_mult_{feature2}",
                        transformation=lambda x1, x2: x1 * x2,
                        input_features=[feature1, feature2],
                        complexity=2
                    ))
                    
                    # 比率
                    candidates.append(FeatureCandidate(
                        feature_name=f"{feature1}_div_{feature2}",
                        transformation=lambda x1, x2: x1 / (x2 + 1e-8),
                        input_features=[feature1, feature2],
                        complexity=2
                    ))
        
        # 限制候選數量
        if len(candidates) > self.max_features:
            candidates = random.sample(candidates, self.max_features)
        
        self.feature_candidates = candidates
        logger.info(f"生成了 {len(candidates)} 個特徵候選")
        
        return candidates
    
    def evaluate_features(self, candidates: List[FeatureCandidate],
                         data: Dict[str, np.ndarray],
                         target: np.ndarray,
                         evaluation_method: str = 'correlation') -> None:
        """
        評估特徵重要性
        
        Args:
            candidates: 特徵候選列表
            data: 數據字典
            target: 目標變量
            evaluation_method: 評估方法
        """
        for candidate in candidates:
            try:
                # 計算特徵值
                if len(candidate.input_features) == 1:
                    feature_name = candidate.input_features[0]
                    if feature_name in data:
                        feature_values = candidate.transformation(data[feature_name])
                    else:
                        continue
                elif len(candidate.input_features) == 2:
                    feature1, feature2 = candidate.input_features
                    if feature1 in data and feature2 in data:
                        feature_values = candidate.transformation(data[feature1], data[feature2])
                    else:
                        continue
                else:
                    continue
                
                # 評估重要性
                if evaluation_method == 'correlation':
                    # 確保feature_values和target維度匹配
                    min_len = min(len(feature_values), len(target))
                    correlation = np.corrcoef(feature_values[:min_len], target[:min_len])[0, 1]
                    candidate.importance_score = abs(correlation) if not np.isnan(correlation) else 0.0
                
                elif evaluation_method == 'mutual_info':
                    # 簡化的互信息（使用相關性近似）
                    min_len = min(len(feature_values), len(target))
                    correlation = np.corrcoef(feature_values[:min_len], target[:min_len])[0, 1]
                    candidate.importance_score = abs(correlation) if not np.isnan(correlation) else 0.0
                
            except Exception as e:
                logger.warning(f"特徵 {candidate.feature_name} 評估失敗: {e}")
                candidate.importance_score = 0.0
    
    def select_features(self, candidates: List[FeatureCandidate], 
                       top_k: int = 20) -> List[FeatureCandidate]:
        """
        選擇最優特徵
        
        Args:
            candidates: 特徵候選列表
            top_k: 選擇的特徵數量
            
        Returns:
            選中的特徵列表
        """
        # 按重要性排序
        candidates.sort(key=lambda x: x.importance_score, reverse=True)
        
        # 選擇top_k特徵
        selected = candidates[:top_k]
        self.selected_features = selected
        
        logger.info(f"選擇了 {len(selected)} 個特徵:")
        for i, feature in enumerate(selected[:5]):  # 只顯示前5個
            logger.info(f"  {i+1}. {feature.feature_name}: {feature.importance_score:.4f}")
        
        return selected


class StrategyInnovationEngine:
    """
    策略創新引擎
    
    整合基因算法、神經架構搜索和自動特徵工程的主要引擎
    """
    
    def __init__(self,
                 genetic_config: Dict[str, Any] = None,
                 nas_config: Dict[str, Any] = None,
                 feature_config: Dict[str, Any] = None,
                 knowledge_base: Optional[MarketKnowledgeBase] = None):
        """
        初始化策略創新引擎
        
        Args:
            genetic_config: 基因算法配置
            nas_config: 神經架構搜索配置  
            feature_config: 特徵工程配置
            knowledge_base: 知識庫（可選）
        """
        # 默認配置
        default_genetic_config = {
            'population_size': 50,
            'mutation_rate': 0.1,
            'crossover_rate': 0.7,
            'elite_ratio': 0.2,
            'max_generations': 100
        }
        
        default_nas_config = {
            'search_space': {
                'num_layers': {'min': 2, 'max': 10},
                'hidden_dims': [64, 128, 256, 512],
                'layer_types': ['linear', 'conv1d', 'attention'],
                'activations': ['relu', 'gelu', 'tanh'],
                'learning_rate': {'min': 1e-5, 'max': 1e-2},
                'batch_sizes': [16, 32, 64, 128]
            },
            'max_architectures': 100,
            'performance_threshold': 0.8
        }
        
        default_feature_config = {
            'max_features': 100,
            'complexity_limit': 5
        }
        
        # 合併配置
        genetic_config = {**default_genetic_config, **(genetic_config or {})}
        nas_config = {**default_nas_config, **(nas_config or {})}
        feature_config = {**default_feature_config, **(feature_config or {})}
        
        # 初始化組件
        self.genetic_evolution = GeneticStrategyEvolution(**genetic_config)
        self.neural_search = NeuralArchitectureSearch(**nas_config)
        self.feature_engineering = AutoFeatureEngineering(**feature_config)
        
        # 知識庫
        self.knowledge_base = knowledge_base or MarketKnowledgeBase()
        
        # 創新歷史
        self.innovation_history: List[Dict[str, Any]] = []
        
        logger.info("策略創新引擎已初始化")
    
    def innovate_strategies(self,
                          strategy_templates: List[Dict[str, Any]],
                          training_data: Dict[str, np.ndarray],
                          target_data: np.ndarray,
                          innovation_budget: int = 50) -> Dict[str, Any]:
        """
        執行策略創新
        
        Args:
            strategy_templates: 策略模板
            training_data: 訓練數據
            target_data: 目標數據
            innovation_budget: 創新預算（評估次數）
            
        Returns:
            創新結果
        """
        logger.info("開始策略創新流程...")
        
        innovation_session = {
            'timestamp': datetime.now().isoformat(),
            'templates_used': len(strategy_templates),
            'data_features': list(training_data.keys()),
            'innovation_budget': innovation_budget,
            'results': {}
        }
        
        # 1. 自動特徵工程
        logger.info("步驟1: 執行自動特徵工程...")
        base_features = list(training_data.keys())
        feature_candidates = self.feature_engineering.generate_features(base_features)
        self.feature_engineering.evaluate_features(feature_candidates, training_data, target_data)
        selected_features = self.feature_engineering.select_features(feature_candidates, top_k=20)
        
        innovation_session['results']['feature_engineering'] = {
            'candidates_generated': len(feature_candidates),
            'features_selected': len(selected_features),
            'top_features': [f.feature_name for f in selected_features[:5]]
        }
        
        # 2. 神經架構搜索
        logger.info("步驟2: 執行神經架構搜索...")
        
        def architecture_evaluator(candidate: ArchitectureCandidate) -> float:
            # 簡化的架構評估（實際中需要訓練和驗證）
            complexity_penalty = candidate.complexity_score / 10000  # 複雜度懲罰
            estimated_performance = random.uniform(0.5, 0.95) - complexity_penalty
            return max(0.0, estimated_performance)
        
        budget_nas = min(innovation_budget // 3, 30)
        self.neural_search.max_architectures = budget_nas
        best_architectures = self.neural_search.search(architecture_evaluator)
        
        innovation_session['results']['neural_architecture_search'] = {
            'architectures_evaluated': budget_nas,
            'good_architectures_found': len(best_architectures),
            'best_performance': best_architectures[0].estimated_performance if best_architectures else 0.0
        }
        
        # 3. 基因算法策略進化
        logger.info("步驟3: 執行基因算法策略進化...")
        
        def strategy_fitness(genome: StrategyGenome) -> float:
            # 簡化的適應度評估
            param_score = sum(abs(v) if isinstance(v, (int, float)) else 1 
                            for v in genome.parameters.values()) / len(genome.parameters)
            arch_score = genome.architecture.get('num_layers', 4) / 10.0
            return random.uniform(0.3, 0.9) + param_score * 0.1 - arch_score * 0.05
        
        self.genetic_evolution.initialize_population(strategy_templates)
        
        budget_genetic = min(innovation_budget // 3, 20)
        for generation in range(min(budget_genetic, self.genetic_evolution.max_generations)):
            stats = self.genetic_evolution.evolve_generation(strategy_fitness)
            if generation % 5 == 0:
                logger.info(f"基因算法第 {generation} 代: 最佳適應度 {stats['best_fitness']:.4f}")
        
        innovation_session['results']['genetic_evolution'] = {
            'generations_evolved': generation + 1,
            'final_best_fitness': self.genetic_evolution.best_genome.fitness_score if self.genetic_evolution.best_genome else 0.0,
            'population_diversity': stats.get('diversity', 0.0)
        }
        
        # 4. 整合結果
        logger.info("步驟4: 整合創新結果...")
        
        # 將結果存儲到知識庫
        knowledge_key = f"innovation_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.knowledge_base.store_knowledge(knowledge_key, innovation_session)
        
        # 創建最終推薦
        recommendations = {
            'best_strategy_genome': self.genetic_evolution.best_genome,
            'best_architecture': best_architectures[0] if best_architectures else None,
            'top_features': selected_features[:10],
            'innovation_summary': innovation_session
        }
        
        self.innovation_history.append(innovation_session)
        
        logger.info(f"策略創新完成 - 使用預算: {innovation_budget}, "
                   f"生成特徵: {len(selected_features)}, "
                   f"架構: {len(best_architectures)}, "
                   f"最佳策略適應度: {self.genetic_evolution.best_genome.fitness_score if self.genetic_evolution.best_genome else 0:.4f}")
        
        return recommendations


# 測試函數
def test_strategy_innovation_engine():
    """測試策略創新引擎"""
    logger.info("開始測試策略創新引擎...")
    
    # 創建模擬數據
    np.random.seed(42)
    training_data = {
        'price': np.random.randn(1000),
        'volume': np.random.randn(1000),
        'rsi': np.random.randn(1000),
        'macd': np.random.randn(1000)
    }
    target_data = np.random.randn(1000)
    
    # 策略模板
    strategy_templates = [
        {
            'type': 'TrendFollowing',
            'parameters': {
                'lookback_period': {'type': 'int', 'range': [5, 50]},
                'threshold': {'type': 'float', 'range': [0.01, 0.1]},
                'use_volume': {'type': 'bool'}
            },
            'architecture': {
                'num_layers': {'range': [2, 6]},
                'hidden_dims': {'choices': [32, 64, 128]},
                'activation': {'choices': ['relu', 'tanh']}
            }
        },
        {
            'type': 'MeanReversion',
            'parameters': {
                'mean_window': {'type': 'int', 'range': [10, 100]},
                'std_multiplier': {'type': 'float', 'range': [1.0, 3.0]},
                'exit_threshold': {'type': 'float', 'range': [0.5, 2.0]}
            },
            'architecture': {
                'num_layers': {'range': [3, 8]},
                'hidden_dims': {'choices': [64, 128, 256]},
                'activation': {'choices': ['gelu', 'tanh']}
            }
        }
    ]
    
    # 初始化引擎
    engine = StrategyInnovationEngine()
    
    # 執行創新
    results = engine.innovate_strategies(
        strategy_templates=strategy_templates,
        training_data=training_data,
        target_data=target_data,
        innovation_budget=30
    )
    
    # 輸出結果
    logger.info("創新結果:")
    logger.info(f"最佳策略類型: {results['best_strategy_genome'].strategy_type if results['best_strategy_genome'] else 'None'}")
    logger.info(f"最佳策略適應度: {results['best_strategy_genome'].fitness_score if results['best_strategy_genome'] else 0:.4f}")
    logger.info(f"發現優秀架構數: {1 if results['best_architecture'] else 0}")
    logger.info(f"生成頂級特徵數: {len(results['top_features'])}")
    
    if results['top_features']:
        logger.info("頂級特徵:")
        for i, feature in enumerate(results['top_features'][:3]):
            logger.info(f"  {i+1}. {feature.feature_name}: {feature.importance_score:.4f}")
    
    logger.info("策略創新引擎測試完成")


if __name__ == '__main__':
    test_strategy_innovation_engine()
