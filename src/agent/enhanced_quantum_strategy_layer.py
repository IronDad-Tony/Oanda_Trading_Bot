# src/agent/enhanced_quantum_strategy_layer.py
"""
增強版量子策略層實現
擴展原有3種策略至15+種策略，實現階段一核心架構增強

主要增強：
1. 15種專業交易策略實現
2. 動態策略生成器
3. 量子策略組合優化
4. 自適應權重學習機制
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
            self.encoding_layers = nn.Sequential(
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
            nn.ReLU(),            nn.Linear(96, 64),
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
    
    def mutate_genes(self, genes: torch.Tensor, 
                    mutation_rate: Optional[float] = None) -> torch.Tensor:
        """對基因進行突變"""
        if mutation_rate is None:
            mutation_rate = self.mutation_controller.item()
        
        noise = torch.randn_like(genes) * mutation_rate
        mutated_genes = genes + noise
        return torch.clamp(mutated_genes, -2.0, 2.0)
    
    def evaluate_strategy_fitness(self, state: torch.Tensor, 
                                action: torch.Tensor) -> torch.Tensor:
        """評估策略的適應度"""
        combined_input = torch.cat([state, action], dim=-1)
        return self.fitness_evaluator(combined_input)
    
    def generate_strategies(self, state: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        生成動態策略
        
        Returns:
            Tuple[策略輸出列表, 基因編碼]
        """
        batch_size = state.size(0)
        
        # 編碼市場基因
        market_genes = self.encode_market_genes(state)
        
        # 生成多個策略變體
        generated_strategies = []
        fitness_scores = []
        
        for i in range(self.max_generated_strategies):
            # 基因突變
            mutated_genes = self.mutate_genes(market_genes)
            
            # 解碼策略
            strategy_output = self.decode_strategy(mutated_genes, i)
            generated_strategies.append(strategy_output)
            
            # 評估適應度
            fitness = self.evaluate_strategy_fitness(state, strategy_output)
            fitness_scores.append(fitness)
        
        # 選擇最佳策略組合
        fitness_tensor = torch.stack(fitness_scores, dim=1)  # [batch, num_strategies]
        
        return generated_strategies, fitness_tensor
    
    def evolve_strategies(self, performance_feedback: torch.Tensor):
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
                projector = nn.Linear(current_last_dim, expected_dim).to(tensor.device)
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
                    dynamic_strategies, dynamic_fitness = self.dynamic_generator.generate_strategies(adapted_state)
                    strategy_outputs.extend(dynamic_strategies)
                else:
                    raise e
        
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
