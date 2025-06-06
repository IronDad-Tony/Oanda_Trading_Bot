# src/environment/progressive_reward_system.py
"""
漸進式獎勵系統實現
階段二：學習系統重構的核心組件

實現三階段漸進式獎勵機制：
1. 探索階段：鼓勵多樣化策略嘗試
2. 利用階段：專注於最佳策略優化
3. 精煉階段：微調和風險控制

主要特性：
- 動態獎勵權重調整
- 多層次性能評估
- 自適應學習率控制
- 策略風險感知機制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import math

try:
    from src.common.logger_setup import logger
    from src.common.config import DEVICE, MAX_SYMBOLS_ALLOWED
except ImportError:
    logger = logging.getLogger(__name__)
    DEVICE = "cpu"
    MAX_SYMBOLS_ALLOWED = 5


class LearningPhase(Enum):
    """學習階段枚舉"""
    EXPLORATION = "exploration"    # 探索階段
    EXPLOITATION = "exploitation"  # 利用階段
    REFINEMENT = "refinement"      # 精煉階段


@dataclass
class RewardConfig:
    """獎勵配置類"""
    # 基礎獎勵權重
    profit_weight: float = 1.0
    risk_penalty_weight: float = 0.3
    diversity_bonus_weight: float = 0.2
    
    # 階段特定權重
    exploration_diversity_boost: float = 2.0
    exploitation_profit_boost: float = 1.5
    refinement_risk_focus: float = 2.0
    
    # 動態調整參數
    phase_transition_threshold: float = 0.1
    performance_momentum_decay: float = 0.95
    risk_tolerance_range: Tuple[float, float] = (0.1, 0.5)


class PerformanceMetrics:
    """性能指標計算器"""
    
    def __init__(self, history_length: int = 1000):
        self.history_length = history_length
        self.reset()
    
    def reset(self):
        """重置所有指標"""
        self.returns_history = []
        self.volatility_history = []
        self.drawdown_history = []
        self.sharpe_history = []
        self.strategy_diversity_history = []
    
    def update(self, returns: torch.Tensor, volatility: torch.Tensor, 
               strategy_weights: torch.Tensor, portfolio_value: torch.Tensor):
        """更新性能指標"""
        # 計算當前指標
        current_return = returns.mean().item()
        current_vol = volatility.mean().item()
        current_diversity = self._calculate_strategy_diversity(strategy_weights)
        current_drawdown = self._calculate_drawdown(portfolio_value)
        current_sharpe = self._calculate_sharpe_ratio(returns, volatility)
        
        # 更新歷史記錄
        self.returns_history.append(current_return)
        self.volatility_history.append(current_vol)
        self.drawdown_history.append(current_drawdown)
        self.sharpe_history.append(current_sharpe)
        self.strategy_diversity_history.append(current_diversity)
        
        # 保持歷史長度限制
        if len(self.returns_history) > self.history_length:
            self.returns_history.pop(0)
            self.volatility_history.pop(0)
            self.drawdown_history.pop(0)
            self.sharpe_history.pop(0)
            self.strategy_diversity_history.pop(0)
    
    def _calculate_strategy_diversity(self, strategy_weights: torch.Tensor) -> float:
        """計算策略多樣性（基於權重分佈的熵）"""
        weights = F.softmax(strategy_weights.mean(dim=0), dim=0)
        weights = torch.clamp(weights, min=1e-8)  # 避免log(0)
        entropy = -torch.sum(weights * torch.log(weights))
        max_entropy = math.log(weights.size(0))
        return (entropy / max_entropy).item()
    
    def _calculate_drawdown(self, portfolio_value: torch.Tensor) -> float:
        """計算最大回撤"""
        if len(self.returns_history) < 2:
            return 0.0
        
        cumulative_returns = np.cumprod(1 + np.array(self.returns_history))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return abs(np.min(drawdown))
    
    def _calculate_sharpe_ratio(self, returns: torch.Tensor, volatility: torch.Tensor) -> float:
        """計算夏普比率"""
        mean_return = returns.mean().item()
        vol = volatility.mean().item()
        return mean_return / (vol + 1e-8)
    
    def get_performance_summary(self) -> Dict[str, float]:
        """獲取性能摘要"""
        if not self.returns_history:
            return {}
        
        return {
            'avg_return': np.mean(self.returns_history),
            'return_volatility': np.std(self.returns_history),
            'avg_sharpe': np.mean(self.sharpe_history),
            'max_drawdown': np.max(self.drawdown_history),
            'avg_diversity': np.mean(self.strategy_diversity_history),
            'return_trend': self._calculate_trend(self.returns_history),
            'volatility_trend': self._calculate_trend(self.volatility_history),
        }
    
    def _calculate_trend(self, data: List[float], window: int = 50) -> float:
        """計算數據趨勢（斜率）"""
        if len(data) < window:
            return 0.0
        
        recent_data = data[-window:]
        x = np.arange(len(recent_data))
        slope, _ = np.polyfit(x, recent_data, 1)
        return slope


class PhaseController:
    """學習階段控制器"""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.current_phase = LearningPhase.EXPLORATION
        self.phase_duration = 0
        self.phase_performance = []
        self.phase_transition_cooldown = 0
        
        # 階段切換閾值
        self.exploration_to_exploitation_threshold = 0.6  # 策略穩定性閾值
        self.exploitation_to_refinement_threshold = 0.8   # 性能穩定性閾值
        
        logger.info(f"初始化階段控制器，當前階段: {self.current_phase.value}")
    
    def update_phase(self, performance_metrics: Dict[str, float]) -> bool:
        """更新學習階段"""
        self.phase_duration += 1
        self.phase_performance.append(performance_metrics.get('avg_return', 0.0))
        
        # 冷卻期間不切換階段
        if self.phase_transition_cooldown > 0:
            self.phase_transition_cooldown -= 1
            return False
        
        phase_changed = False
        
        # 階段切換邏輯
        if self.current_phase == LearningPhase.EXPLORATION:
            if self._should_transition_to_exploitation(performance_metrics):
                self.current_phase = LearningPhase.EXPLOITATION
                phase_changed = True
                logger.info("階段切換: 探索 → 利用")
                
        elif self.current_phase == LearningPhase.EXPLOITATION:
            if self._should_transition_to_refinement(performance_metrics):
                self.current_phase = LearningPhase.REFINEMENT
                phase_changed = True
                logger.info("階段切換: 利用 → 精煉")
            elif self._should_return_to_exploration(performance_metrics):
                self.current_phase = LearningPhase.EXPLORATION
                phase_changed = True
                logger.info("階段切換: 利用 → 探索 (性能下降)")
                
        elif self.current_phase == LearningPhase.REFINEMENT:
            if self._should_return_to_exploitation(performance_metrics):
                self.current_phase = LearningPhase.EXPLOITATION
                phase_changed = True
                logger.info("階段切換: 精煉 → 利用 (需要重新優化)")
        
        if phase_changed:
            self._reset_phase_state()
            
        return phase_changed
    
    def _should_transition_to_exploitation(self, metrics: Dict[str, float]) -> bool:
        """判斷是否應從探索階段切換到利用階段"""
        # 條件：策略多樣性穩定 + 最小階段持續時間
        min_duration = 100
        diversity_stable = metrics.get('avg_diversity', 0.0) > 0.5
        performance_improving = metrics.get('return_trend', 0.0) > 0
        
        return (self.phase_duration > min_duration and 
                diversity_stable and performance_improving)
    
    def _should_transition_to_refinement(self, metrics: Dict[str, float]) -> bool:
        """判斷是否應從利用階段切換到精煉階段"""
        # 條件：性能穩定 + 低波動性
        min_duration = 150
        performance_stable = abs(metrics.get('return_trend', 0.0)) < 0.001
        low_volatility = metrics.get('return_volatility', 1.0) < 0.1
        good_sharpe = metrics.get('avg_sharpe', 0.0) > 1.0
        
        return (self.phase_duration > min_duration and 
                performance_stable and low_volatility and good_sharpe)
    
    def _should_return_to_exploration(self, metrics: Dict[str, float]) -> bool:
        """判斷是否應從利用階段返回探索階段"""
        # 條件：性能明顯下降
        performance_declining = metrics.get('return_trend', 0.0) < -0.005
        high_drawdown = metrics.get('max_drawdown', 0.0) > 0.15
        
        return performance_declining or high_drawdown
    
    def _should_return_to_exploitation(self, metrics: Dict[str, float]) -> bool:
        """判斷是否應從精煉階段返回利用階段"""
        # 條件：市場環境變化，需要重新優化
        volatility_spike = metrics.get('volatility_trend', 0.0) > 0.01
        performance_unstable = abs(metrics.get('return_trend', 0.0)) > 0.002
        
        return volatility_spike or performance_unstable
    
    def _reset_phase_state(self):
        """重置階段狀態"""
        self.phase_duration = 0
        self.phase_performance = []
        self.phase_transition_cooldown = 50  # 50步冷卻期
    
    def get_phase_weights(self) -> Dict[str, float]:
        """獲取當前階段的獎勵權重"""
        base_weights = {
            'profit': self.config.profit_weight,
            'risk_penalty': self.config.risk_penalty_weight,
            'diversity_bonus': self.config.diversity_bonus_weight,
        }
        
        if self.current_phase == LearningPhase.EXPLORATION:
            base_weights['diversity_bonus'] *= self.config.exploration_diversity_boost
            base_weights['risk_penalty'] *= 0.7  # 降低風險懲罰
            
        elif self.current_phase == LearningPhase.EXPLOITATION:
            base_weights['profit'] *= self.config.exploitation_profit_boost
            base_weights['diversity_bonus'] *= 0.8  # 降低多樣性獎勵
            
        elif self.current_phase == LearningPhase.REFINEMENT:
            base_weights['risk_penalty'] *= self.config.refinement_risk_focus
            base_weights['profit'] *= 0.9  # 稍微降低利潤權重
            
        return base_weights


class ProgressiveRewardSystem(nn.Module):
    """
    漸進式獎勵系統
    實現三階段動態獎勵機制
    """
    
    def __init__(self, 
                 num_strategies: int,
                 config: Optional[RewardConfig] = None,
                 enable_adaptive_learning: bool = True):
        super().__init__()
        
        self.num_strategies = num_strategies
        self.config = config or RewardConfig()
        self.enable_adaptive_learning = enable_adaptive_learning
        
        # 核心組件
        self.performance_metrics = PerformanceMetrics()
        self.phase_controller = PhaseController(self.config)
        
        # 獎勵計算網絡
        self.reward_networks = nn.ModuleDict({
            'profit_evaluator': nn.Sequential(
                nn.Linear(3, 32),  # [return, sharpe, trend]
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            ),
            'risk_evaluator': nn.Sequential(
                nn.Linear(4, 32),  # [volatility, drawdown, var, skew]
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            ),
            'diversity_evaluator': nn.Sequential(
                nn.Linear(num_strategies + 2, 32),  # [strategy_weights, entropy, gini]
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        })
        
        # 動態調整參數
        self.learning_rate_multiplier = nn.Parameter(torch.tensor(1.0))
        self.risk_tolerance = nn.Parameter(torch.tensor(0.3))
        
        # 獎勵歷史追蹤
        self.register_buffer('reward_history', torch.zeros(1000))
        self.register_buffer('reward_index', torch.tensor(0))
        
        logger.info(f"初始化漸進式獎勵系統: {num_strategies}種策略")
    
    def forward(self, 
                returns: torch.Tensor,
                volatility: torch.Tensor,
                strategy_weights: torch.Tensor,
                portfolio_value: torch.Tensor,
                market_state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        計算漸進式獎勵
        
        Args:
            returns: 投資回報 [batch_size]
            volatility: 波動率 [batch_size]
            strategy_weights: 策略權重 [batch_size, num_strategies]
            portfolio_value: 投資組合價值 [batch_size]
            market_state: 市場狀態 [batch_size, state_dim]
            
        Returns:
            Tuple[總獎勵, 詳細信息字典]
        """
        batch_size = returns.size(0)
        
        # 更新性能指標
        self.performance_metrics.update(returns, volatility, strategy_weights, portfolio_value)
        
        # 獲取性能摘要並更新階段
        performance_summary = self.performance_metrics.get_performance_summary()
        phase_changed = self.phase_controller.update_phase(performance_summary)
        
        # 獲取當前階段權重
        phase_weights = self.phase_controller.get_phase_weights()
        
        # 計算各組件獎勵
        profit_reward = self._calculate_profit_reward(returns, volatility)
        risk_penalty = self._calculate_risk_penalty(returns, volatility, portfolio_value)
        diversity_bonus = self._calculate_diversity_bonus(strategy_weights)
        
        # 組合最終獎勵
        total_reward = (
            phase_weights['profit'] * profit_reward -
            phase_weights['risk_penalty'] * risk_penalty +
            phase_weights['diversity_bonus'] * diversity_bonus
        )
        
        # 動態學習率調整
        if self.enable_adaptive_learning:
            lr_adjustment = self._calculate_learning_rate_adjustment(performance_summary)
            total_reward = total_reward * lr_adjustment
        
        # 更新獎勵歷史
        self._update_reward_history(total_reward.mean())
        
        # 構建詳細信息
        info = {
            'current_phase': self.phase_controller.current_phase.value,
            'phase_duration': self.phase_controller.phase_duration,
            'phase_changed': phase_changed,
            'phase_weights': phase_weights,
            'profit_reward': profit_reward.mean().item(),
            'risk_penalty': risk_penalty.mean().item(),
            'diversity_bonus': diversity_bonus.mean().item(),
            'total_reward': total_reward.mean().item(),
            'learning_rate_multiplier': self.learning_rate_multiplier.item(),
            'risk_tolerance': self.risk_tolerance.item(),
            'performance_summary': performance_summary,
        }
        
        return total_reward, info
    
    def _calculate_profit_reward(self, returns: torch.Tensor, volatility: torch.Tensor) -> torch.Tensor:
        """計算利潤獎勵"""
        # 計算風險調整回報
        sharpe_ratio = returns / (volatility + 1e-8)
        
        # 計算趨勢強度
        returns_batch = returns.unsqueeze(-1).expand(-1, 3)
        trend_strength = torch.tanh(returns_batch.std(dim=-1))
        
        # 組合輸入特徵
        profit_features = torch.stack([
            torch.tanh(returns * 10),      # 標準化回報
            torch.tanh(sharpe_ratio),      # 夏普比率
            trend_strength                 # 趨勢強度
        ], dim=-1)
        
        profit_score = self.reward_networks['profit_evaluator'](profit_features).squeeze(-1)
        return profit_score
    
    def _calculate_risk_penalty(self, returns: torch.Tensor, volatility: torch.Tensor, 
                              portfolio_value: torch.Tensor) -> torch.Tensor:
        """計算風險懲罰"""
        batch_size = returns.size(0)
        
        # 計算VaR (Value at Risk)
        var_95 = torch.quantile(returns, 0.05) if returns.numel() > 1 else returns.mean()
        var_95 = var_95.expand(batch_size)
        
        # 計算偏度 (風險指標)
        returns_centered = returns - returns.mean()
        skewness = (returns_centered ** 3).mean() / (volatility ** 3 + 1e-8)
        
        # 組合風險特徵
        risk_features = torch.stack([
            torch.tanh(volatility * 5),     # 標準化波動率
            torch.tanh(-var_95 * 10),       # VaR (負值因為是損失)
            torch.tanh(volatility.std().expand(batch_size)),  # 波動率穩定性
            torch.tanh(skewness)            # 偏度
        ], dim=-1)
        
        risk_score = self.reward_networks['risk_evaluator'](risk_features).squeeze(-1)
        
        # 動態風險容忍度調整
        risk_penalty = risk_score * self.risk_tolerance
        return risk_penalty
    
    def _calculate_diversity_bonus(self, strategy_weights: torch.Tensor) -> torch.Tensor:
        """計算策略多樣性獎勵"""
        batch_size = strategy_weights.size(0)
        
        # 計算策略權重熵
        weights_normalized = F.softmax(strategy_weights, dim=-1)
        weights_clamped = torch.clamp(weights_normalized, min=1e-8)
        entropy = -torch.sum(weights_clamped * torch.log(weights_clamped), dim=-1)
        max_entropy = math.log(self.num_strategies)
        normalized_entropy = entropy / max_entropy
        
        # 計算基尼係數（衡量分佈不均勻程度）
        sorted_weights, _ = torch.sort(weights_normalized, dim=-1)
        n = strategy_weights.size(-1)
        index = torch.arange(1, n + 1, device=strategy_weights.device).float()
        gini = (2 * torch.sum(sorted_weights * index.unsqueeze(0), dim=-1) / 
                (n * torch.sum(sorted_weights, dim=-1)) - (n + 1) / n)
        diversity_gini = 1 - gini  # 轉換為多樣性指標
        
        # 組合多樣性特徵
        diversity_features = torch.cat([
            weights_normalized,                              # 策略權重
            normalized_entropy.unsqueeze(-1),               # 熵
            diversity_gini.unsqueeze(-1)                     # 基尼多樣性
        ], dim=-1)
        
        diversity_score = self.reward_networks['diversity_evaluator'](diversity_features).squeeze(-1)
        return diversity_score
    
    def _calculate_learning_rate_adjustment(self, performance_summary: Dict[str, float]) -> torch.Tensor:
        """計算學習率調整倍數"""
        if not performance_summary:
            return self.learning_rate_multiplier
        
        # 基於性能趨勢調整學習率
        return_trend = performance_summary.get('return_trend', 0.0)
        avg_return = performance_summary.get('avg_return', 0.0)
        
        # 性能改善時降低學習率（穩定），性能下降時提高學習率（探索）
        if return_trend > 0 and avg_return > 0:
            adjustment = 0.95  # 穩定期降低學習率
        elif return_trend < 0 or avg_return < 0:
            adjustment = 1.1   # 性能下降時提高學習率
        else:
            adjustment = 1.0   # 保持不變
        
        # 更新學習率倍數（使用動量）
        self.learning_rate_multiplier.data = (
            0.9 * self.learning_rate_multiplier.data + 0.1 * adjustment
        )
        
        # 限制學習率倍數範圍
        self.learning_rate_multiplier.data = torch.clamp(
            self.learning_rate_multiplier.data, 0.5, 2.0
        )
        
        return self.learning_rate_multiplier
    
    def _update_reward_history(self, reward: torch.Tensor):
        """更新獎勵歷史"""
        idx = self.reward_index.item() % 1000
        self.reward_history[idx] = reward.item()
        self.reward_index += 1
    
    def get_system_analysis(self) -> Dict[str, Any]:
        """獲取系統分析信息"""
        recent_rewards = self.reward_history[:min(1000, self.reward_index.item())]
        
        return {
            'current_phase': self.phase_controller.current_phase.value,
            'phase_duration': self.phase_controller.phase_duration,
            'learning_rate_multiplier': self.learning_rate_multiplier.item(),
            'risk_tolerance': self.risk_tolerance.item(),
            'avg_recent_reward': recent_rewards.mean().item() if recent_rewards.numel() > 0 else 0.0,
            'reward_volatility': recent_rewards.std().item() if recent_rewards.numel() > 0 else 0.0,
            'performance_metrics': self.performance_metrics.get_performance_summary(),
            'total_steps': self.reward_index.item(),
        }
    
    def reset_system(self):
        """重置系統狀態"""
        self.performance_metrics.reset()
        self.phase_controller = PhaseController(self.config)
        self.reward_history.zero_()
        self.reward_index.zero_()
        self.learning_rate_multiplier.data.fill_(1.0)
        self.risk_tolerance.data.fill_(0.3)
        
        logger.info("漸進式獎勵系統已重置")


if __name__ == "__main__":
    # 測試漸進式獎勵系統
    logger.info("開始測試漸進式獎勵系統...")
    
    # 測試參數
    batch_size = 8
    num_strategies = 20
    state_dim = 64
    
    # 創建測試數據
    test_returns = torch.randn(batch_size) * 0.02  # 2%平均回報，帶噪聲
    test_volatility = torch.abs(torch.randn(batch_size)) * 0.1 + 0.05  # 5-15%波動率
    test_strategy_weights = torch.softmax(torch.randn(batch_size, num_strategies), dim=-1)
    test_portfolio_value = torch.ones(batch_size) * 100000  # 初始資金100k
    test_market_state = torch.randn(batch_size, state_dim)
    
    # 初始化獎勵系統
    reward_system = ProgressiveRewardSystem(
        num_strategies=num_strategies,
        enable_adaptive_learning=True
    )
    
    try:
        logger.info("開始獎勵系統測試...")
        
        # 模擬多步驟測試
        for step in range(200):
            # 模擬市場變化
            step_returns = test_returns + torch.randn(batch_size) * 0.01
            step_volatility = test_volatility + torch.randn(batch_size) * 0.01
            step_portfolio_value = test_portfolio_value * (1 + step_returns)
            
            # 計算獎勵
            rewards, info = reward_system(
                step_returns, step_volatility, test_strategy_weights,
                step_portfolio_value, test_market_state
            )
            
            # 每50步報告一次
            if step % 50 == 0:
                logger.info(f"Step {step}: 階段={info['current_phase']}, "
                          f"獎勵={info['total_reward']:.4f}, "
                          f"學習率倍數={info['learning_rate_multiplier']:.3f}")
        
        # 獲取最終分析
        analysis = reward_system.get_system_analysis()
        logger.info("=== 漸進式獎勵系統測試完成 ===")
        logger.info(f"最終階段: {analysis['current_phase']}")
        logger.info(f"平均獎勵: {analysis['avg_recent_reward']:.4f}")
        logger.info(f"學習率倍數: {analysis['learning_rate_multiplier']:.3f}")
        logger.info(f"總步數: {analysis['total_steps']}")
        
        # 測試梯度計算
        reward_system.train()
        rewards, _ = reward_system(test_returns, test_volatility, test_strategy_weights,
                                 test_portfolio_value, test_market_state)
        loss = -rewards.mean()  # 最大化獎勵
        loss.backward()
        
        logger.info("梯度計算測試通過")
        total_params = sum(p.numel() for p in reward_system.parameters())
        logger.info(f"獎勵系統總參數量: {total_params:,}")
        
        logger.info("階段二核心組件：漸進式獎勵系統實施完成 ✅")
        
    except Exception as e:
        logger.error(f"測試失敗: {e}")
        import traceback
        traceback.print_exc()
        raise e
