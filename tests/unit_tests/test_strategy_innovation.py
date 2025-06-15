# tests/unit_tests/test_strategy_innovation.py
"""
策略創新引擎單元測試

測試 StrategyInnovationEngine 及其組件的功能
"""

import sys
import os
import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# 添加項目路徑
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.agent.strategy_innovation_engine import (
    StrategyInnovationEngine,
    GeneticStrategyEvolution,
    NeuralArchitectureSearch, 
    AutoFeatureEngineering,
    StrategyGenome,
    ArchitectureCandidate,
    FeatureCandidate
)


class TestStrategyGenome:
    """測試 StrategyGenome 數據類"""
    
    def test_genome_creation(self):
        """測試基因組創建"""
        genome = StrategyGenome(
            strategy_type="momentum",
            parameters={"window": 20, "threshold": 0.02},
            architecture={"layers": [64, 32, 16]},
            fitness_score=0.85
        )
        
        assert genome.strategy_type == "momentum"
        assert genome.parameters["window"] == 20
        assert genome.fitness_score == 0.85
        assert genome.parent_ids == []
        assert genome.mutation_history == []
    
    def test_genome_with_history(self):
        """測試帶歷史記錄的基因組"""
        genome = StrategyGenome(
            strategy_type="mean_reversion",
            parameters={"lookback": 10},
            architecture={"depth": 3},
            parent_ids=["parent1", "parent2"],
            mutation_history=["param_mutation", "arch_mutation"]
        )
        
        assert len(genome.parent_ids) == 2
        assert len(genome.mutation_history) == 2


class TestGeneticStrategyEvolution:
    """測試基因算法策略進化器"""
    
    @pytest.fixture
    def genetic_evolution(self):
        """創建基因進化器實例"""
        return GeneticStrategyEvolution(
            population_size=10,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elite_ratio=0.2
        )
    
    @pytest.fixture
    def strategy_templates(self):
        """策略模板"""
        return [
            {
                "type": "momentum",
                "parameters": {
                    "window": {"type": "int", "range": [5, 50]},
                    "threshold": {"type": "float", "range": [0.01, 0.1]}
                },
                "architecture": {
                    "num_layers": {"range": [2, 4]},
                    "hidden_dims": {"choices": [32, 64, 128]},
                    "activation": {"choices": ["relu", "tanh"]}
                }
            },
            {
                "type": "mean_reversion", 
                "parameters": {
                    "lookback": {"type": "int", "range": [5, 30]},
                    "std_dev": {"type": "float", "range": [1.0, 3.0]}
                },
                "architecture": {
                    "num_layers": {"range": [2, 5]},
                    "hidden_dims": {"choices": [64, 128, 256]},
                    "activation": {"choices": ["relu", "tanh", "gelu"]}
                }
            }
        ]
    
    def test_initialization(self, genetic_evolution):
        """測試初始化"""
        assert genetic_evolution.population_size == 10
        assert genetic_evolution.mutation_rate == 0.1
        assert genetic_evolution.crossover_rate == 0.7
        assert genetic_evolution.elite_ratio == 0.2
        assert len(genetic_evolution.population) == 0
    
    def test_initialize_population(self, genetic_evolution, strategy_templates):
        """測試種群初始化"""
        genetic_evolution.initialize_population(strategy_templates)
        
        assert len(genetic_evolution.population) == 10
        for genome in genetic_evolution.population:
            assert isinstance(genome, StrategyGenome)
            assert genome.strategy_type in ["momentum", "mean_reversion"]
            assert genome.generation == 0
    
    def test_fitness_evaluation(self, genetic_evolution, strategy_templates):
        """測試適應度評估"""
        genetic_evolution.initialize_population(strategy_templates)
        
        # 模擬適應度評估
        for genome in genetic_evolution.population:
            fitness_score = np.random.random()
            genetic_evolution.evaluate_fitness(genome, lambda g, d, t: fitness_score, {}, np.array([0]))
        
        for genome in genetic_evolution.population:
            assert genome.fitness_score >= 0
    
    def test_selection(self, genetic_evolution, strategy_templates):
        """測試選擇操作"""
        genetic_evolution.initialize_population(strategy_templates)
        
        # 設置適應度分數
        for i, genome in enumerate(genetic_evolution.population):
            genome.fitness_score = i / len(genetic_evolution.population)
        
        selected = genetic_evolution.selection()
        
        # 檢查選擇的個體
        assert isinstance(selected, StrategyGenome)
        assert selected.fitness_score >= 0
    
    def test_crossover(self, genetic_evolution):
        """測試交叉操作"""
        parent1 = StrategyGenome(
            strategy_type="momentum",
            parameters={"window": 20, "threshold": 0.02},
            architecture={"layers": [64, 32]}
        )
        parent2 = StrategyGenome(
            strategy_type="momentum", 
            parameters={"window": 30, "threshold": 0.05},
            architecture={"layers": [32, 16]}
        )
        
        child1, child2 = genetic_evolution.crossover(parent1, parent2)
        
        assert isinstance(child1, StrategyGenome)
        assert isinstance(child2, StrategyGenome)
        assert child1.strategy_type == "momentum"
        assert child2.strategy_type == "momentum"
    
    def test_mutation(self, genetic_evolution):
        """測試突變操作"""
        genome = StrategyGenome(
            strategy_type="momentum",
            parameters={"window": 20, "threshold": 0.02},
            architecture={"layers": [64, 32]}
        )
        
        original_params = genome.parameters.copy()
        mutated = genetic_evolution.mutate(genome)
        
        assert isinstance(mutated, StrategyGenome)
        # 參數可能發生變化
        assert mutated.strategy_type == genome.strategy_type


class TestNeuralArchitectureSearch:
    """測試神經架構搜索"""
    
    @pytest.fixture
    def nas(self):
        """創建NAS實例"""
        return NeuralArchitectureSearch(
            search_space={
                "num_layers": {"min": 2, "max": 4},
                "hidden_dims": [32, 64, 128, 256],
                "layer_types": ["linear", "conv1d", "attention"],
                "activations": ["relu", "tanh", "sigmoid"],
                "learning_rate": {"min": 1e-4, "max": 1e-2},
                "batch_sizes": [16, 32, 64]
            },
            max_architectures=5,
            performance_threshold=0.7
        )
    
    def test_initialization(self, nas):
        """測試初始化"""
        assert nas.max_architectures == 5
        assert nas.performance_threshold == 0.7
        assert "layers" in nas.search_space
    
    def test_generate_architecture(self, nas):
        """測試架構生成"""
        architecture = nas.generate_architecture()
        
        assert isinstance(architecture, ArchitectureCandidate)
        assert len(architecture.layers) >= 2
        assert len(architecture.layers) <= 4
        
        for layer in architecture.layers:
            assert "type" in layer
    
    def test_evaluate_architecture(self, nas):
        """測試架構評估"""
        architecture = nas.generate_architecture()
        
        # 模擬評估函數
        def mock_evaluate_fn(arch):
            return np.random.random()
        
        performance = nas.evaluate_architecture(architecture, mock_evaluate_fn)
        
        assert isinstance(performance, float)
        assert 0 <= performance <= 1
        assert architecture.estimated_performance == performance
    
    def test_search_best_architecture(self, nas):
        """測試最佳架構搜索"""
        def mock_evaluate_fn(arch):
            return np.random.random()
        
        best_architectures = nas.search(mock_evaluate_fn)
        
        assert isinstance(best_architectures, list)
        assert len(best_architectures) > 0
        for arch in best_architectures:
            assert isinstance(arch, ArchitectureCandidate)


class TestAutoFeatureEngineering:
    """測試自動特徵工程"""
    
    @pytest.fixture
    def feature_engine(self):
        """創建特徵工程實例"""
        return AutoFeatureEngineering(
            max_features=10,
            complexity_limit=3
        )
    
    @pytest.fixture
    def sample_data(self):
        """示例數據"""
        return {
            "price": np.random.randn(100),
            "volume": np.random.randn(100),
            "volatility": np.random.randn(100)
        }    
    def test_initialization(self, feature_engine):
        """測試初始化"""
        assert feature_engine.max_features == 10
        assert feature_engine.complexity_limit == 3
    
    def test_generate_feature_candidates(self, feature_engine):
        """測試特徵候選生成"""
        base_features = ["price", "volume", "volatility"]
        candidates = feature_engine.generate_features(base_features)
        
        assert len(candidates) > 0
        for candidate in candidates:
            assert isinstance(candidate, FeatureCandidate)
            assert candidate.feature_name is not None
            assert callable(candidate.transformation)
    
    def test_evaluate_feature(self, feature_engine, sample_data):
        """測試特徵評估"""
        # 創建簡單的特徵候選
        base_features = ["price", "volume", "volatility"]
        candidates = feature_engine.generate_features(base_features)
        
        # 模擬目標變量
        target = np.random.randn(100)
        
        # 評估特徵
        feature_engine.evaluate_features(candidates, sample_data, target)
        
        # 檢查特徵分數
        for candidate in candidates:
            assert hasattr(candidate, 'importance_score')
    
    def test_select_best_features(self, feature_engine, sample_data):
        """測試最佳特徵選擇"""
        # 模擬目標變量
        target = np.random.randn(100)
        
        # 生成和評估特徵
        base_features = ["price", "volume", "volatility"]
        candidates = feature_engine.generate_features(base_features)
        feature_engine.evaluate_features(candidates, sample_data, target)
        
        # 選擇最佳特徵
        best_features = feature_engine.select_features(candidates, top_k=5)
        
        assert len(best_features) <= 5
        for feature in best_features:
            assert isinstance(feature, FeatureCandidate)


class TestStrategyInnovationEngine:
    """測試策略創新引擎主類"""
    
    @pytest.fixture
    def innovation_engine(self):
        """創建創新引擎實例"""
        genetic_config = {
            "population_size": 10,
            "mutation_rate": 0.1,
            "max_generations": 5
        }
        nas_config = {
            "search_space": {
                "num_layers": {"min": 2, "max": 4},
                "hidden_dims": [64, 128, 256],
                "layer_types": ["linear"],
                "activations": ["relu", "tanh"]
            },
            "max_architectures": 5,
            "performance_threshold": 0.7
        }
        feature_config = {
            "max_features": 5,
            "complexity_limit": 2
        }
        
        return StrategyInnovationEngine(
            genetic_config=genetic_config,
            nas_config=nas_config,
            feature_config=feature_config
        )    
    def test_initialization(self, innovation_engine):
        """測試初始化"""
        assert innovation_engine.genetic_evolution is not None
        assert innovation_engine.neural_search is not None
        assert innovation_engine.feature_engineering is not None
        assert innovation_engine.knowledge_base is not None
    
    def test_innovate_strategies(self, innovation_engine, strategy_templates):
        """測試策略創新"""
        # 模擬市場數據
        training_data = {
            "price": np.random.randn(100),
            "volume": np.random.randn(100),
            "returns": np.random.randn(100)
        }
        target_data = np.random.randn(100)
        
        result = innovation_engine.innovate_strategies(
            strategy_templates=strategy_templates,
            training_data=training_data,
            target_data=target_data,
            innovation_budget=5
        )
        
        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "results" in result


class TestIntegration:
    """集成測試"""
    
    def test_component_integration(self):
        """測試組件集成"""
        # 創建完整配置
        genetic_config = {
            "population_size": 5,
            "mutation_rate": 0.2,
            "max_generations": 2
        }
        nas_config = {
            "search_space": {
                "num_layers": {"min": 2, "max": 3},
                "hidden_dims": [64, 128],
                "layer_types": ["linear"],
                "activations": ["relu"]
            },
            "max_architectures": 3,
            "performance_threshold": 0.5
        }
        feature_config = {
            "max_features": 3,
            "complexity_limit": 2
        }
        
        engine = StrategyInnovationEngine(
            genetic_config=genetic_config,
            nas_config=nas_config,
            feature_config=feature_config
        )
        
        # 驗證所有組件都已正確初始化
        assert engine.genetic_evolution.population_size == 5
        assert engine.neural_search.max_architectures == 3
        assert engine.feature_engineering.max_features == 3
    
    def test_error_handling(self):
        """測試錯誤處理"""
        # 創建基本的 StrategyInnovationEngine（應該成功）
        engine = StrategyInnovationEngine()
        assert engine is not None
        
        # 測試空的策略模板
        genetic_evolution = GeneticStrategyEvolution(population_size=5)
        with pytest.raises((ValueError, IndexError)):
            genetic_evolution.initialize_population([])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
