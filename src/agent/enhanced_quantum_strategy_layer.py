# src/agent/enhanced_quantum_strategy_layer.py
"""
增強版量子策略層實現
擴展原有3種策略至15+種策略，實現階段一核心架構增強

主要增強：
1. 15種專業交易策略實現
2. 動態策略生成器
3. 量子策略組合優化
4. 自適應權重極習機制
5. 策略創新引擎（基礎版）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from abc import ABC, abstractmethod
import math
from dataclasses import dataclass

try:
    from src.common.logger_setup import logger
    from src.common.config import DEVICE, MAX_SYMBOLS_ALLOWED
    from src.agent.quantum_strategy_layer import BaseStrategy, QuantumEncoder
except ImportError:
    logger = logging.getLogger(__name__)
    DEVICE = "cpu"
    MAX_SYMBOLS_ALLOWED = 5


# 自定義激活函數實現
class Swish(nn.Module):
    """Swish激活函數實現"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    """Mish激活函數實現"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# 基礎策略類（如果導入失敗則定義）
if 'BaseStrategy' not in globals():
    class BaseStrategy(nn.Module, ABC):
        @abstractmethod
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            pass
        
        @abstractmethod
        def get_strategy_name(self) -> str:
            pass


if 'QuantumEncoder' not in globals():
    class QuantumEncoder(nn.Module):
        def __init__(self, input_dim: int, latent_dim: int, num_qubits: int = 8):
            super().__init__()
            self.encoding_layers =nn.Sequential(
                nn.Linear(input_dim, latent_dim),
                nn.GELU()
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoding_layers(x)


@dataclass
class StrategyConfig:
    """策略配置類"""
    name: str
    description: str
    risk_level: float  # 0.0-1.0
    market_regime: str  # "trending", "ranging", "volatile", "all"
    complexity: int    # 1-5
    base_performance: float = 0.5


# ===============================
# 15種專業交易策略實現
# ===============================

class MomentumStrategy(BaseStrategy):
    """動量策略：基於價格動量進行交易"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.strategy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.momentum_amplifier = nn.Parameter(torch.tensor(1.5))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        base_output = self.strategy_net(state)
        return base_output * self.momentum_amplifier
    
    def get_strategy_name(self) -> str:
        return "Momentum"


class BreakoutStrategy(BaseStrategy):
    """突破策略：識別並跟隨價格突破"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.strategy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.breakout_threshold = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raw_signal = self.strategy_net(state)
        # 應用突破閾值
        breakout_mask = torch.abs(raw_signal) > self.breakout_threshold
        return raw_signal * breakout_mask.float() * 1.3
    
    def get_strategy_name(self) -> str:
        return "Breakout"


class StatisticalArbitrageStrategy(BaseStrategy):
    """統計套利策略：基於統計模型的套利機會"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.strategy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            Swish(),
            nn.LayerNorm(128),
            nn.Linear(128, 96),
            Swish(),
            nn.Linear(96, 64),
            Swish(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.correlation_matrix = nn.Parameter(torch.eye(action_dim) * 0.1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        base_signal = self.strategy_net(state)
        # 應用統計相關性調整
        adjusted_signal = torch.matmul(base_signal.unsqueeze(1), self.correlation_matrix).squeeze(1)
        return adjusted_signal * 0.8  # 保守的套利信號
    
    def get_strategy_name(self) -> str:
        return "StatisticalArbitrage"


class OptionFlowStrategy(BaseStrategy):
    """期權流策略：基於期權市場信號"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.strategy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.option_sensitivity = nn.Parameter(torch.tensor(0.7))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.strategy_net(state) * self.option_sensitivity
    
    def get_strategy_name(self) -> str:
        return "OptionFlow"


class MicrostructureStrategy(BaseStrategy):
    """市場微觀結構策略：基於高頻交易信號"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.strategy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.GroupNorm(8, 128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.microstructure_weight = nn.Parameter(torch.tensor(0.6))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.strategy_net(state) * self.microstructure_weight
    
    def get_strategy_name(self) -> str:
        return "Microstructure"


class VolatilityStrategy(BaseStrategy):
    """波動率策略：基於波動率模式的交易"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.strategy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            Mish(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            Mish(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.vol_scaling = nn.Parameter(torch.tensor(1.1))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.strategy_net(state) * self.vol_scaling
    
    def get_strategy_name(self) -> str:
        return "Volatility"


class CarryTradeStrategy(BaseStrategy):
    """利差交易策略：基於利率差異"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.strategy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.carry_multiplier = nn.Parameter(torch.tensor(0.9))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.strategy_net(state) * self.carry_multiplier
    
    def get_strategy_name(self) -> str:
        return "CarryTrade"


class MacroEconomicStrategy(BaseStrategy):
    """宏觀經濟策略：基於宏觀經濟指標"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.strategy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.SELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.SELU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.macro_influence = nn.Parameter(torch.tensor(0.8))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.strategy_net(state) * self.macro_influence
    
    def get_strategy_name(self) -> str:
        return "MacroEconomic"


class EventDrivenStrategy(BaseStrategy):
    """事件驅動策略：基於市場事件和新聞"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.strategy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.event_intensity = nn.Parameter(torch.tensor(1.2))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.strategy_net(state) * self.event_intensity
    
    def get_strategy_name(self) -> str:
        return "EventDriven"


class SentimentStrategy(BaseStrategy):
    """市場情緒策略：基於市場情緒指標"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.strategy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            Swish(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            Swish(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.sentiment_factor = nn.Parameter(torch.tensor(0.7))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.strategy_net(state) * self.sentiment_factor
    
    def get_strategy_name(self) -> str:
        return "Sentiment"


class QuantitativeStrategy(BaseStrategy):
    """量化因子策略：基於量化因子模型"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.strategy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 96),
            nn.ReLU(),            
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.factor_loadings = nn.Parameter(torch.randn(action_dim, 5) * 0.1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        base_signal = self.strategy_net(state)
        # 應用因子載荷 - 修復維度問題
        factor_effect = torch.matmul(base_signal, self.factor_loadings).mean(dim=-1, keepdim=True)
        enhanced_signal = base_signal + factor_effect * 0.1
        return enhanced_signal * 0.9
    
    def get_strategy_name(self) -> str:
        return "Quantitative"


class PairsTradeStrategy(BaseStrategy):
    """配對交易策略：基於配對股票的相對價值"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.strategy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.pairs_correlation = nn.Parameter(torch.eye(action_dim) * 0.2)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        base_signal = self.strategy_net(state)
        pairs_adjusted = torch.matmul(base_signal.unsqueeze(1), self.pairs_correlation).squeeze(1)
        return pairs_adjusted * 0.85
    
    def get_strategy_name(self) -> str:
        return "PairsTrade"


class MarketMakingStrategy(BaseStrategy):
    """做市策略：提供流動性獲取差價"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.strategy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.spread_sensitivity = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.strategy_net(state) * self.spread_sensitivity
    
    def get_strategy_name(self) -> str:
        return "MarketMaking"


class HighFrequencyStrategy(BaseStrategy):
    """高頻交易策略：基於微秒級市場動態"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.strategy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            Mish(),
            nn.GroupNorm(8, 128),
            nn.Linear(128, 64),
            Mish(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.hft_multiplier = nn.Parameter(torch.tensor(0.4))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.strategy_net(state) * self.hft_multiplier
    
    def get_strategy_name(self) -> str:
        return "HighFrequency"


class AlgorithmicStrategy(BaseStrategy):
    """算法交易策略：基於預定義交易算法"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.strategy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.SELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.SELU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.algo_parameters = nn.Parameter(torch.randn(3) * 0.1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        base_signal = self.strategy_net(state)
        # 應用算法參數調整
        algo_adjustment = torch.sum(self.algo_parameters) * 0.1
        return base_signal * (1.0 + algo_adjustment)
    
    def get_strategy_name(self) -> str:
        return "Algorithmic"


# ===============================
# 動態策略生成器
# ===============================

class DynamicStrategyGenerator(nn.Module):
    """
    動態策略生成器：實時創建和調整交易策略
    基於遺傳算法和神經進化的混合方法
    支持完全動態自適應維度配置
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 base_strategies: List[BaseStrategy],
                 max_generated_strategies: int = 5):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.base_strategies = base_strategies
        self.max_generated_strategies = max_generated_strategies
        
        # 動態計算中間層維度 - 自適應縮放
        self.gene_hidden_dim = max(16, min(128, state_dim // 4))  # 動態隱藏層維度
        self.gene_latent_dim = max(8, min(64, state_dim // 8))   # 動態潛在維度
        self.decoder_hidden_1 = max(32, min(256, action_dim * 2))  # 解碼器第一層
        self.decoder_hidden_2 = max(64, min(512, action_dim * 4))  # 解碼器第二層
        self.fitness_hidden_1 = max(32, min(128, (state_dim + action_dim) // 2))
        self.fitness_hidden_2 = max(16, min(64, (state_dim + action_dim) // 4))
        
        # 策略基因編碼器 - 動態維度
        self.gene_encoder = nn.Sequential(
            nn.Linear(state_dim, self.gene_hidden_dim),
            nn.GELU(),
            nn.Linear(self.gene_hidden_dim, self.gene_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.gene_hidden_dim // 2, self.gene_latent_dim)
        )
        
        # 策略解碼器網絡 - 動態維度
        self.strategy_decoder = nn.ModuleDict({
            f"decoder_{i}": nn.Sequential(
                nn.Linear(self.gene_latent_dim, self.decoder_hidden_1),
                nn.GELU(),
                nn.Linear(self.decoder_hidden_1, self.decoder_hidden_2),
                nn.GELU(),
                nn.Linear(self.decoder_hidden_2, action_dim),
                nn.Tanh()
            ) for i in range(max_generated_strategies)
        })
        
        # 策略適應度評估器 - 動態維度
        self.fitness_evaluator = nn.Sequential(
            nn.Linear(state_dim + action_dim, self.fitness_hidden_1),
            nn.ReLU(),
            nn.Linear(self.fitness_hidden_1, self.fitness_hidden_2),
            nn.ReLU(),
            nn.Linear(self.fitness_hidden_2, 1),
            nn.Sigmoid()
        )
        
        # 基因突變控制器
        self.mutation_controller = nn.Parameter(torch.tensor(0.1))
        
        # 生成策略的性能歷史
        self.register_buffer('strategy_performance', torch.zeros(max_generated_strategies))
        self.register_buffer('generation_counter', torch.tensor(0))
        
    def encode_market_genes(self, state: torch.Tensor) -> torch.Tensor:
        """將市場狀態編碼為策略基因"""
        return self.gene_encoder(state)
    
    def decode_strategy(self, genes: torch.Tensor, strategy_id: int) -> torch.Tensor:
        """將基因解碼為策略輸出"""
        decoder = self.strategy_decoder[f"decoder_{strategy_id}"]
        return decoder(genes)
    
    def roulette_wheel_selection(self, fitness_scores: torch.Tensor) -> torch.Tensor:
        """
        基於適應度的輪盤賭選擇算子（支持批量處理）
        Args:
            fitness_scores: 個體適應度分數，形狀 [batch_size, population_size]
        Returns:
            選擇的個體索引，形狀 [batch_size]
        """
        # 計算選擇概率
        probs = fitness_scores / fitness_scores.sum(dim=1, keepdim=True)
        
        # 逐样本处理以避免广播问题
        selected_indices = []
        for i in range(probs.shape[0]):
            # 生成随机数
            r = torch.rand(1, device=probs.device)
            # 累積概率
            cumulative_probs = torch.cumsum(probs[i], dim=0)
            # 找到第一个大于等于随机数的索引
            idx = torch.nonzero(cumulative_probs >= r, as_tuple=True)[0]
            if idx.numel() > 0:
                selected_indices.append(idx[0])
            else:
                selected_indices.append(torch.tensor(len(cumulative_probs)-1, device=probs.device))
        
        return torch.stack(selected_indices)
    
    def validate_architecture(self, state: torch.Tensor) -> float:
        """驗證架構搜索性能並返回搜索時間"""
        import time
        start_time = time.time()
        
        # 編碼市場基因
        genes = self.encode_market_genes(state)
        
        # 測試所有NAS模塊
        for i in range(self.max_generated_strategies):
            _ = self.decode_strategy(genes, i)
        
        end_time = time.time()
        return end_time - start_time
    
    def single_point_crossover(self, parent1: torch.Tensor, parent2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        單點交叉算子（支持批量處理）
        Args:
            parent1: 父代基因1，形狀 [batch_size, gene_length]
            parent2: 父代基因2，形狀 [batch_size, gene_length]
        Returns:
            (child1, child2): 兩個子代基因
        """
        batch_size, gene_length = parent1.shape
        # 隨機選擇交叉點（每個樣本不同）
        crossover_points = torch.randint(1, gene_length-1, (batch_size,), device=parent1.device)
        
        # 創建掩碼矩陣
        mask = torch.arange(gene_length, device=parent1.device).expand(batch_size, -1)
        mask = mask < crossover_points.unsqueeze(1)
        
        # 執行交叉
        child1 = torch.where(mask, parent1, parent2)
        child2 = torch.where(mask, parent2, parent1)
        return child1, child2
    
    def gaussian_mutation(self, genes: torch.Tensor, 
                         mutation_rate: Optional[float] = None,
                         mutation_scale: float = 0.1) -> torch.Tensor:
        """
        高斯變異算子（支持自動微分和批量處理）
        Args:
            genes: 輸入基因，形狀 [batch_size, gene_length]
            mutation_rate: 變異率（每個基因變異的概率）
            mutation_scale: 變異強度（高斯噪聲的標準差）
        Returns:
            變異後的基因
        """
        if mutation_rate is None:
            mutation_rate = self.mutation_controller.item()
        
        # 創建突變掩碼
        mutation_mask = torch.rand_like(genes) < mutation_rate
        # 生成高斯噪聲
        noise = torch.randn_like(genes) * mutation_scale
        # 應用突變
        mutated_genes = genes + mutation_mask * noise
        return torch.clamp(mutated_genes, -2.0, 2.0)
    
    def evaluate_strategy_fitness(self, state: torch.Tensor, 
                                action: torch.Tensor) -> torch.Tensor:
        """評估策略的適應度"""
        combined_input = torch.cat([state, action], dim=-1)
        return self.fitness_evaluator(combined_input)
    
    def generate_strategies(self, state: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        使用遺傳算法生成動態策略（包含選擇、交叉、變異）
        
        Returns:
            Tuple[策略輸出列表, 適應度分數]
        """
        batch_size = state.size(0)
        
        # 編碼市場基因
        market_genes = self.encode_market_genes(state)  # [batch_size, gene_length]
        
        # 初始化種群（基於市場基因的變異）
        population = []
        for i in range(self.max_generated_strategies):
            # 每個策略使用不同的解碼器
            mutated_genes = self.gaussian_mutation(market_genes)
            population.append(mutated_genes)
        
        # 評估初始適應度
        fitness_scores = []
        strategy_outputs = []
        for i, genes in enumerate(population):
            output = self.decode_strategy(genes, i)  # 使用對應解碼器
            fitness = self.evaluate_strategy_fitness(state, output)
            fitness_scores.append(fitness)
            strategy_outputs.append(output)
        
        fitness_tensor = torch.stack(fitness_scores, dim=1)  # [batch, population]
        
        # 遺傳算法迭代
        for generation in range(3):  # 進行3代進化
            new_population = []
            
            # 選擇父代（輪盤賭選擇）
            parent_indices = []
            for _ in range(self.max_generated_strategies):
                selected_idx = self.roulette_wheel_selection(fitness_tensor)
                parent_indices.append(selected_idx)
            
            # 交叉和變異
            for i in range(0, self.max_generated_strategies, 2):
                if i+1 >= self.max_generated_strategies:
                    break
                    
                # 獲取父代基因
                parent1_idx = parent_indices[i]
                parent1 = torch.stack([population[j][b] for b, j in enumerate(parent1_idx)], dim=0)
                
                parent2_idx = parent_indices[i+1]
                parent2 = torch.stack([population[j][b] for b, j in enumerate(parent2_idx)], dim=0)
                
                # 執行交叉
                child1, child2 = self.single_point_crossover(parent1, parent2)
                
                # 變異
                child1 = self.gaussian_mutation(child1)
                child2 = self.gaussian_mutation(child2)
                
                new_population.append(child1)
                new_population.append(child2)
            
            # 更新種群（保留前N個）
            population = new_population[:self.max_generated_strategies]
            
            # 重新評估適應度
            fitness_scores = []
            strategy_outputs = []
            for i, genes in enumerate(population):
                output = self.decode_strategy(genes, i)
                fitness = self.evaluate_strategy_fitness(state, output)
                fitness_scores.append(fitness)
                strategy_outputs.append(output)
            
            fitness_tensor = torch.stack(fitness_scores, dim=1)
        
        return strategy_outputs, fitness_tensor
    
    def evolve_strateg极(self, performance_feedback: torch.Tensor):
        """基於性能反饋進化策略"""
        # 更新性能歷史
        self.strategy_performance = 0.9 * self.strategy_performance + 0.1 * performance_feedback
        
        # 調整突變率
        avg_performance = torch.mean(self.strategy_performance)
        if avg_performance < 0.5:
            self.mutation_controller.data *= 1.1  # 增加探索
        else:
            self.mutation_controller.data *= 0.95  # 減少探索
        
        # 限制突變率範圍
        self.mutation_controller.data = torch.clamp(self.mutation_controller.data, 0.01, 0.3)
        
        self.generation_counter += 1


# ===============================
# 增強版策略疊加系統
# ===============================

class EnhancedStrategySuperposition(nn.Module):
    """
    增強版策略疊加系統
    管理15+種策略的量子疊加，包含動態策略生成
    支持完全動態自適應維度配置
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 enable_dynamic_generation: bool = True):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.enable_dynamic_generation = enable_dynamic_generation
        
        # 初始化所有15種基礎策略
        self.base_strategies = nn.ModuleList([
            # 原有策略
            MomentumStrategy(state_dim, action_dim),
            BreakoutStrategy(state_dim, action_dim),
            StatisticalArbitrageStrategy(state_dim, action_dim),
            # 新增策略
            OptionFlowStrategy(state_dim, action_dim),
            MicrostructureStrategy(state_dim, action_dim),
            VolatilityStrategy(state_dim, action_dim),
            CarryTradeStrategy(state_dim, action_dim),
            MacroEconomicStrategy(state_dim, action_dim),
            EventDrivenStrategy(state_dim, action_dim),
            SentimentStrategy(state_dim, action_dim),
            QuantitativeStrategy(state_dim, action_dim),
            PairsTradeStrategy(state_dim, action_dim),
            MarketMakingStrategy(state_dim, action_dim),
            HighFrequencyStrategy(state_dim, action_dim),
            AlgorithmicStrategy(state_dim, action_dim),
        ])
        
        self.num_base_strategies = len(self.base_strategies)
        
        # 動態策略生成器
        if enable_dynamic_generation:
            self.dynamic_generator = DynamicStrategyGenerator(
                state_dim, action_dim, self.base_strategies, max_generated_strategies=5
            )
            self.total_strategies = self.num_base_strategies + 5
        else:
            self.total_strategies = self.num_base_strategies
        
        # 動態計算權重網絡的隱藏層維度 - 自適應縮放
        self.vol_hidden_dim = max(16, min(64, self.total_strategies))
        self.regime_hidden_dim = max(32, min(128, state_dim // 4))
        self.corr_hidden_dim = max(32, min(128, (state_dim + 1) // 4))
        
        # 量子振幅參數（可學習）
        self.quantum_amplitudes = nn.Parameter(
            torch.ones(self.total_strategies) / math.sqrt(self.total_strategies)
        )
        
        # 多層次權重調整網絡 - 動態維度配置
        self.weight_networks = nn.ModuleDict({
            'volatility_net': nn.Sequential(
                nn.Linear(1, self.vol_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.vol_hidden_dim, self.total_strategies),
                nn.Sigmoid()
            ),
            'regime_net': nn.Sequential(
                nn.Linear(state_dim, self.regime_hidden_dim),
                nn.GELU(),
                nn.Linear(self.regime_hidden_dim, self.total_strategies),
                nn.Softmax(dim=-1)
            ),
            'correlation_net': nn.Sequential(
                nn.Linear(state_dim + 1, self.corr_hidden_dim),
                nn.GELU(),
                nn.Linear(self.corr_hidden_dim, self.total_strategies),
                nn.Softmax(dim=-1)
            )
        })
        
        # 策略相互作用矩陣
        self.interaction_matrix = nn.Parameter(
            torch.eye(self.total_strategies) + torch.randn(self.total_strategies, self.total_strategies) * 0.05
        )
        
        # 策略性能追蹤
        self.register_buffer('strategy_performance_history', 
                           torch.zeros(self.total_strategies, 100))  # 記錄最近100次性能
        self.register_buffer('performance_index', torch.tensor(0))
        
        # 量子糾纏效應模擬
        self.entanglement_strength = nn.Parameter(torch.tensor(0.1))
        
        logger.info(f"🌟 初始化增強版策略疊加系統: {self.total_strategies}種策略")
        logger.info(f"📐 動態維度配置 - State: {state_dim}, Action: {action_dim}")
        logger.info(f"🔧 權重網絡隱藏層 - Vol: {self.vol_hidden_dim}, Regime: {self.regime_hidden_dim}, Corr: {self.corr_hidden_dim}")
    
    def get_dynamic_dimensions(self) -> Dict[str, int]:
        """獲取當前動態維度配置信息"""
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'total_strategies': self.total_strategies,
            'vol_hidden_dim': self.vol_hidden_dim,
            'regime_hidden_dim': self.regime_hidden_dim,
            'corr_hidden_dim': self.corr_hidden_dim,
            'num_base_strategies': self.num_base_strategies,
            'dynamic_generation_enabled': self.enable_dynamic_generation
        }
    
    def _adaptive_dimension_handler(self, tensor: torch.Tensor, 
                                   expected_dim: int, 
                                   operation_name: str = "unknown") -> torch.Tensor:
        """
        動態維度適配處理器
        自動調整輸入張量以匹配期望維度
        
        Args:
            tensor: 輸入張量
            expected_dim: 期望的最後一個維度
            operation_name: 操作名稱，用於日誌
            
        Returns:
            適配後的張量
        """
        current_shape = tensor.shape
        current_last_dim = current_shape[-1]
        
        if current_last_dim == expected_dim:
            return tensor
        
        batch_dims = current_shape[:-1]
        
        if current_last_dim > expected_dim:
            # 維度過大：使用線性投影降維
            if not hasattr(self, f'_adaptive_projector_{operation_name}_{current_last_dim}_{expected_dim}'):
                projector = nn.Linear(current_last_dim, expected_dim).to(tensor.dev极)
                setattr(self, f'_adaptive_projector_{operation_name}_{current_last_dim}_{expected_dim}', projector)
                logger.info(f"🔧 創建動態投影器: {operation_name} {current_last_dim}→{expected_dim}")
            
            projector = getattr(self, f'_adaptive_projector_{operation_name}_{current_last_dim}_{expected_dim}')
            adapted_tensor = projector(tensor)
            
        elif current_last_dim < expected_dim:
            # 維度過小：使用零填充擴展
            pad_size = expected_dim - current_last_dim
            padding = torch.zeros(*batch_dims, pad_size, device=tensor.device, dtype=tensor.dtype)
            adapted_tensor = torch.cat([tensor, padding], dim=-1)
            logger.info(f"🔧 動態零填充: {operation_name} {current_last_dim}→{expected_dim}")
        
        return adapted_tensor
    
    def _validate_and_adapt_inputs(self, state: torch.Tensor, 
                                  volatility: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        驗證並適配輸入維度
        
        Args:
            state: 市場狀態張量
            volatility: 波動率張量
            
        Returns:
            適配後的狀態和波動率張量
        """
        # 適配狀態張量
        adapted_state = self._adaptive_dimension_handler(
            state, self.state_dim, "state_input"
        )
        
        # 確保波動率是一維的
        if volatility.dim() > 1 and volatility.shape[-1] != 1:
            volatility = volatility.mean(dim=-1, keepdim=True)
        
        if volatility.dim() == 1:
            volatility = volatility.unsqueeze(-1)
        
        return adapted_state, volatility.squeeze(-1)
    
    def forward(self, state: torch.Tensor, volatility: torch.Tensor, 
                market_regime: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向傳播：計算增強策略疊加輸出
        支持動態維度適配
        
        Args:
            state: 量子編碼的市場狀態
            volatility: 市場波動率
            market_regime: 市場環境指標
            
        Returns:
            Tuple[疊加策略輸出, 詳細信息字典]
        """
        # 動態維度適配
        state, volatility = self._validate_and_adapt_inputs(state, volatility)
        batch_size = state.size(0)
        
        # 執行所有基礎策略
        base_strategy_outputs = []
        for strategy in self.base_strategies:
            try:
                output = strategy(state)
                base_strategy_outputs.append(output)
            except RuntimeError as e:
                if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                    logger.warning(f"⚠️ 策略 {strategy.get_strategy_name()} 維度不匹配，使用自適應處理")
                    # 嘗試自適應處理
                    adapted_state = self._adaptive_dimension_handler(state, strategy.strategy_net[0].in_features, f"strategy_{strategy.get_strategy_name()}")
                    output = strategy(adapted_state)
                    base_strategy_outputs.append(output)
                else:
                    raise e
        
        strategy_outputs = base_strategy_outputs.copy()
        num_base_strategies = len(strategy_outputs)
        
        # 動態策略生成
        dynamic_fitness = None
        if self.enable_dynamic_generation:
            try:
                dynamic_strategies, dynamic_fitness = self.dynamic_generator.generate_strategies(state)
                strategy_outputs.extend(dynamic_strategies)
            except RuntimeError as e:
                if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                    logger.warning("⚠️ 動態策略生成維度不匹配，使用自適應處理")
                    adapted_state = self._adaptive_dimension_handler(state, self.dynamic_generator.state_dim, "dynamic_generator")
                    dynamic_strategies, dynamic_fitness = self.dynamic_generator.generate极ategies(adapted_state)
                    strategy_outputs.extend(dynamic_strategies)
                else:
                    raise e
        
        # 確保策略數量一致
        if len(strategy_outputs) != self.total_strategies:
            logger.warning(f"⚠️ 策略數量不一致: 實際 {len(strategy_outputs)} vs 預期 {self.total_strategies}")
            # 調整策略輸出數量以匹配預期
            if len(strategy_outputs) > self.total_strategies:
                strategy_outputs = strategy_outputs[:self.total_strategies]
            else:
                # 添加空策略以補足數量
                for i in range(self.total_strategies - len(strategy_outputs)):
                    strategy_outputs.append(torch.zeros_like(strategy_outputs[0]))
        
        # 堆疊所有策略輸出
        strategy_tensor = torch.stack(strategy_outputs, dim=1)  # [batch, num_strategies, action_dim]
        
        # 計算多層次權重 - 使用動態適配
        vol_weights = self.weight_networks['volatility_net'](volatility.unsqueeze(-1))
        regime_weights = self.weight_networks['regime_net'](state)
        corr_input = torch.cat([state, volatility.unsqueeze(-1)], dim=-1)
        corr_weights = self.weight_networks['correlation_net'](corr_input)
        
        # 量子振幅正規化
        quantum_amps = F.softmax(self.quantum_amplitudes, dim=0)
        
        # 組合權重（考慮量子效應）
        combined_weights = quantum_amps.unsqueeze(0) * vol_weights * regime_weights * corr_weights
        combined_weights = F.softmax(combined_weights, dim=-1)
        
        # 策略相互作用效應
        interaction_effect = torch.matmul(combined_weights.unsqueeze(1), self.interaction_matrix).squeeze(1)
        final_weights = combined_weights + self.entanglement_strength * interaction_effect
        final_weights = F.softmax(final_weights, dim=-1)
        
        # 加權疊加策略輸出
        superposition_output = torch.sum(
            strategy_tensor * final_weights.unsqueeze(-1), dim=1
        )
        
        # 構建詳細信息
        info_dict = {
            'strategy_weights': final_weights,
            'quantum_amplitudes': quantum_amps,
            'vol_weights': vol_weights,
            'regime_weights': regime_weights,
            'corr_weights': corr_weights,
            'num_active_strategies': torch.sum(final_weights > 0.01, dim=-1).float(),
            'dimensions_info': self.get_dynamic_dimensions()
        }
        
        if dynamic_fitness is not None:
            info_dict['dynamic_fitness'] = dynamic_fitness
        
        return superposition_output, info_dict
    
    def update_strategy_performance(self, performance_scores: torch.Tensor):
        """更新策略性能歷史"""
        current_idx = self.performance_index.item() % 100
        self.strategy_performance_history[:, current_idx] = performance_scores.mean(dim=0)
        self.performance_index += 1
        
        # 更新動態策略生成器
        if self.enable_dynamic_generation:
            dynamic_performance = performance_scores[:, self.num_base_strategies:]
            if dynamic_performance.numel() > 0:
                self.dynamic_generator.evolve_strategies(dynamic_performance.mean(dim=0))
    
    def get_strategy_analysis(self) -> Dict[str, Any]:
        """獲取策略分析信息"""
        recent_performance = self.strategy_performance_history[:, :min(100, self.performance_index.item())]
        
        return {
            'num_strategies': len(self.base_strategies) + (5 if self.enable_dynamic_generation else 0),
            'avg_performance': recent_performance.mean(dim=-1),
            'performance_std': recent_performance.std(dim=-1),
            'best_strategy_idx': recent_performance.mean(dim=-1).argmax().item(),
            'worst_strategy_idx': recent_performance.mean(dim=-1).argmin().item(),
            'strategy_names': [s.get_strategy_name() for s in self.base_strategies] + 
                            ([f"Dynamic_{i}" for i in range(5)] if self.enable_dynamic_generation else []),
            'quantum_amplitude_distribution': F.softmax(self.quantum_amplitudes, dim=0),
            'entanglement_strength': self.entanglement_strength.item(),
        }


if __name__ == "__main__":
    # 測試增強版量子策略層
    logger.info("開始測試增強版量子策略層...")
    
    # 測試參數
    batch_size = 8
    state_dim = 64
    action_dim = 10
    
    # 創建測試數據
    test_state = torch.randn(batch_size, state_dim)
    test_volatility = torch.rand(batch_size) * 0.5
    
    # 初始化增強版策略疊加系統
    enhanced_strategy_layer = EnhancedStrategySuperposition(
        state_dim=state_dim,
        action_dim=action_dim,
        enable_dynamic_generation=True
    )
    
    try:
        # 前向傳播測試
        with torch.no_grad():
            output, info = enhanced_strategy_layer(test_state, test_volatility)
            
        logger.info(f"測試成功！")
        logger.info(f"輸入狀態形狀: {test_state.shape}")
        logger.info(f"輸出動作形狀: {output.shape}")
        logger.info(f"策略權重形狀: {info['strategy_weights'].shape}")
        logger.info(f"活躍策略數量: {info['num_active_strategies'].mean():.2f}")
        
        # 獲取策略分析
        analysis = enhanced_strategy_layer.get_strategy_analysis()
        logger.info(f"策略總數: {analysis['num_strategies']}")
        logger.info(f"策略名稱: {analysis['strategy_names']}")
        
        # 測試梯度計算
        enhanced_strategy_layer.train()
        output, info = enhanced_strategy_layer(test_state, test_volatility)
        loss = output.abs().mean()
        loss.backward()
        
        logger.info("梯度計算測試通過")
        logger.info(f"總參數量: {sum(p.numel() for p in enhanced_strategy_layer.parameters()):,}")
        
    except Exception as e:
        logger.error(f"測試失敗: {e}")
        raise e

    # ==============================================
    # 測試動態策略生成器的遺傳算法功能
    # ==============================================
    logger.info("\n開始測試動態策略生成器的遺傳算法功能...")
    
    # 加載真實數據
    try:
        import pandas as pd
        # 加載EUR/USD 5秒數據
        data_path = "data/EUR_USD_5S_20250601.csv"
        df = pd.read_csv(data_path)
        logger.info(f"成功加載數據: {data_path}, 形狀: {df.shape}")
        
        # 數據預處理
        # 使用收盤價作為狀態特徵
        prices = df['close'].values
        # 計算波動率
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        # 轉換為PyTorch張量
        state_tensor = torch.tensor(prices[-state_dim:], dtype=torch.float32).unsqueeze(0)
        volatility_tensor = torch.tensor([volatility], dtype=torch.float32)
        
        # 初始化動態策略生成器
        generator = DynamicStrategyGenerator(
            state_dim=state_dim,
            action_dim=action_dim,
            base_strategies=enhanced_strategy_layer.base_strategies,
            max_generated_strategies=5
        )
        
        # 測試策略生成
        strategies, fitness_scores = generator.generate_strategies(state_tensor)
        logger.info(f"生成策略數量: {len(strategies)}")
        logger.info(f"策略輸出形狀: {strategies[0].shape}")
        logger.info(f"適應度分數形狀: {fitness_scores.shape}")
        
        # 測試遺傳算法操作
        logger.info("\n測試遺傳算法操作:")
        logger.info("1. 測試輪盤賭選擇...")
        selected_idx = generator.roulette_wheel_selection(fitness_scores)
        logger.info(f"選擇的索引: {selected_idx.item()}")
        
        logger.info("2. 測試單點交叉...")
        parent1 = torch.randn(1, generator.gene_latent_dim)
        parent2 = torch.randn(1, generator.gene_latent_dim)
        child1, child2 = generator.single_point_crossover(parent1, parent2)
        logger.info(f"父代1形狀: {parent1.shape}, 父代2形狀: {parent2.shape}")
        logger.info(f"子代1形狀: {child1.shape}, 子代2形狀: {child2.shape}")
        
        logger.info("3. 測試高斯變異...")
        genes = torch.randn(1, generator.gene_latent_dim)
        mutated_genes = generator.gaussian_mutation(genes)
        logger.info(f"變異前: {genes.mean().item():.4f}, 變異後: {mutated_genes.mean().item():.4f}")
        
        logger.info("✅ 動態策略生成器測試通過")
        
    except Exception as e:
        logger.error(f"動態策略生成器測試失敗: {e}")
        raise e
