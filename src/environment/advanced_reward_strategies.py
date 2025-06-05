"""
高級獎勵策略集合
提供多種獎勵策略以適應不同市場狀況和訓練階段
"""

from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class RewardStrategy(Enum):
    """獎勵策略枚舉"""
    CONSERVATIVE = "conservative"  # 保守策略：重視風險控制
    AGGRESSIVE = "aggressive"      # 激進策略：追求高收益
    BALANCED = "balanced"          # 平衡策略：風險收益並重
    ADAPTIVE = "adaptive"          # 自適應策略：根據市場調整
    MOMENTUM = "momentum"          # 動量策略：趨勢跟隨
    MEAN_REVERSION = "mean_reversion"  # 均值回歸策略

class AdvancedRewardStrategies:
    """高級獎勵策略實現"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.current_strategy = RewardStrategy.ADAPTIVE
        
        # 市場狀態檢測
        self.market_volatility_threshold = 0.02
        self.trend_strength_threshold = 0.7
        
        # 策略權重配置
        self.strategy_weights = self._initialize_strategy_weights()
        
    def _initialize_strategy_weights(self) -> Dict[RewardStrategy, Dict[str, float]]:
        """初始化各策略的權重配置"""
        return {
            RewardStrategy.CONSERVATIVE: {
                'profit_weight': 0.3,
                'risk_weight': 0.5,
                'drawdown_penalty': 3.0,
                'volatility_penalty': 2.0,
                'sharpe_bonus': 1.5
            },
            RewardStrategy.AGGRESSIVE: {
                'profit_weight': 0.7,
                'risk_weight': 0.2,
                'return_bonus': 2.0,
                'momentum_bonus': 1.5,
                'drawdown_penalty': 1.0
            },
            RewardStrategy.BALANCED: {
                'profit_weight': 0.5,
                'risk_weight': 0.3,
                'consistency_bonus': 1.2,
                'diversification_bonus': 1.0,
                'drawdown_penalty': 2.0
            },
            RewardStrategy.ADAPTIVE: {
                'adaptation_bonus': 1.5,
                'regime_change_bonus': 1.0,
                'learning_rate_bonus': 0.8,
                'flexibility_bonus': 1.2
            },
            RewardStrategy.MOMENTUM: {
                'trend_following_bonus': 2.0,
                'momentum_consistency': 1.5,
                'breakout_bonus': 1.8,
                'trend_strength_bonus': 1.3
            },
            RewardStrategy.MEAN_REVERSION: {
                'contrarian_bonus': 1.5,
                'oversold_overbought_bonus': 1.8,
                'volatility_capture_bonus': 1.2,
                'range_trading_bonus': 1.0
            }
        }
    
    def detect_market_regime(self, market_data: Dict[str, Any]) -> str:
        """檢測市場狀態"""
        volatility = market_data.get('volatility', 0)
        trend_strength = market_data.get('trend_strength', 0)
        volume = market_data.get('volume', 0)
        
        # 高波動性市場
        if volatility > self.market_volatility_threshold * 2:
            return 'high_volatility'
        # 趨勢市場
        elif trend_strength > self.trend_strength_threshold:
            return 'trending'
        # 震盪市場
        elif volatility < self.market_volatility_threshold * 0.5:
            return 'ranging'
        # 正常市場
        else:
            return 'normal'
    
    def select_optimal_strategy(self, market_regime: str, 
                              performance_metrics: Dict[str, float]) -> RewardStrategy:
        """根據市場狀態和性能選擇最佳策略"""
        
        current_sharpe = performance_metrics.get('sharpe_ratio', 0)
        current_drawdown = abs(performance_metrics.get('max_drawdown', 0))
        current_profit_factor = performance_metrics.get('profit_factor', 1.0)
        
        # 根據市場狀態初步選擇
        if market_regime == 'high_volatility':
            base_strategy = RewardStrategy.CONSERVATIVE
        elif market_regime == 'trending':
            base_strategy = RewardStrategy.MOMENTUM
        elif market_regime == 'ranging':
            base_strategy = RewardStrategy.MEAN_REVERSION
        else:
            base_strategy = RewardStrategy.BALANCED
            
        # 根據當前性能調整
        if current_drawdown > 0.15:  # 回撤過大
            return RewardStrategy.CONSERVATIVE
        elif current_sharpe < 0.5:  # Sharpe比率過低
            return RewardStrategy.ADAPTIVE
        elif current_profit_factor > 1.5:  # 表現良好
            return base_strategy
        else:
            return RewardStrategy.BALANCED
    
    def calculate_strategy_reward(self, strategy: RewardStrategy,
                                trading_metrics: Dict[str, float],
                                market_data: Dict[str, Any]) -> Dict[str, float]:
        """根據策略計算獎勵"""
        
        weights = self.strategy_weights[strategy]
        reward_components = {}
        
        if strategy == RewardStrategy.CONSERVATIVE:
            reward_components = self._calculate_conservative_reward(weights, trading_metrics)
        elif strategy == RewardStrategy.AGGRESSIVE:
            reward_components = self._calculate_aggressive_reward(weights, trading_metrics)
        elif strategy == RewardStrategy.BALANCED:
            reward_components = self._calculate_balanced_reward(weights, trading_metrics)
        elif strategy == RewardStrategy.ADAPTIVE:
            reward_components = self._calculate_adaptive_reward(weights, trading_metrics, market_data)
        elif strategy == RewardStrategy.MOMENTUM:
            reward_components = self._calculate_momentum_reward(weights, trading_metrics, market_data)
        elif strategy == RewardStrategy.MEAN_REVERSION:
            reward_components = self._calculate_mean_reversion_reward(weights, trading_metrics, market_data)
            
        return reward_components
    
    def _calculate_conservative_reward(self, weights: Dict[str, float], 
                                     metrics: Dict[str, float]) -> Dict[str, float]:
        """保守策略獎勵計算"""
        return {
            'risk_adjusted_return': metrics.get('sharpe_ratio', 0) * weights.get('sharpe_bonus', 1.0),
            'drawdown_penalty': -abs(metrics.get('max_drawdown', 0)) * weights.get('drawdown_penalty', 2.0),
            'volatility_control': -metrics.get('volatility', 0) * weights.get('volatility_penalty', 1.0),
            'capital_preservation': metrics.get('capital_preservation_ratio', 0) * weights.get('profit_weight', 0.3)
        }
    
    def _calculate_aggressive_reward(self, weights: Dict[str, float], 
                                   metrics: Dict[str, float]) -> Dict[str, float]:
        """激進策略獎勵計算"""
        return {
            'high_returns': metrics.get('total_return', 0) * weights.get('return_bonus', 2.0),
            'momentum_capture': metrics.get('momentum_score', 0) * weights.get('momentum_bonus', 1.5),
            'profit_maximization': metrics.get('profit_factor', 1.0) * weights.get('profit_weight', 0.7),
            'growth_potential': metrics.get('compound_growth_rate', 0) * 1.5
        }
    
    def _calculate_balanced_reward(self, weights: Dict[str, float], 
                                 metrics: Dict[str, float]) -> Dict[str, float]:
        """平衡策略獎勵計算"""
        return {
            'risk_return_balance': (metrics.get('sharpe_ratio', 0) + metrics.get('profit_factor', 1.0)) * 0.5,
            'consistency_bonus': metrics.get('consistency_score', 0) * weights.get('consistency_bonus', 1.2),
            'drawdown_control': -abs(metrics.get('max_drawdown', 0)) * weights.get('drawdown_penalty', 2.0),
            'diversification': metrics.get('diversification_score', 0) * weights.get('diversification_bonus', 1.0)
        }
    
    def _calculate_adaptive_reward(self, weights: Dict[str, float], 
                                 metrics: Dict[str, float], 
                                 market_data: Dict[str, Any]) -> Dict[str, float]:
        """自適應策略獎勵計算"""
        market_regime = self.detect_market_regime(market_data)
        adaptation_score = self._calculate_adaptation_score(metrics, market_regime)
        
        return {
            'adaptation_performance': adaptation_score * weights.get('adaptation_bonus', 1.5),
            'regime_awareness': metrics.get('regime_detection_accuracy', 0) * weights.get('regime_change_bonus', 1.0),
            'learning_efficiency': metrics.get('learning_rate', 0) * weights.get('learning_rate_bonus', 0.8),
            'flexibility_score': metrics.get('strategy_flexibility', 0) * weights.get('flexibility_bonus', 1.2)
        }
    
    def _calculate_momentum_reward(self, weights: Dict[str, float], 
                                 metrics: Dict[str, float], 
                                 market_data: Dict[str, Any]) -> Dict[str, float]:
        """動量策略獎勵計算"""
        trend_strength = market_data.get('trend_strength', 0)
        
        return {
            'trend_following': metrics.get('trend_following_accuracy', 0) * weights.get('trend_following_bonus', 2.0),
            'momentum_consistency': metrics.get('momentum_consistency', 0) * weights.get('momentum_consistency', 1.5),
            'breakout_capture': metrics.get('breakout_success_rate', 0) * weights.get('breakout_bonus', 1.8),
            'trend_strength_utilization': trend_strength * weights.get('trend_strength_bonus', 1.3)
        }
    
    def _calculate_mean_reversion_reward(self, weights: Dict[str, float], 
                                       metrics: Dict[str, float], 
                                       market_data: Dict[str, Any]) -> Dict[str, float]:
        """均值回歸策略獎勵計算"""
        return {
            'contrarian_success': metrics.get('contrarian_accuracy', 0) * weights.get('contrarian_bonus', 1.5),
            'overbought_oversold': metrics.get('mean_reversion_success', 0) * weights.get('oversold_overbought_bonus', 1.8),
            'volatility_capture': metrics.get('volatility_profit', 0) * weights.get('volatility_capture_bonus', 1.2),
            'range_trading': metrics.get('range_trading_efficiency', 0) * weights.get('range_trading_bonus', 1.0)
        }
    
    def _calculate_adaptation_score(self, metrics: Dict[str, float], market_regime: str) -> float:
        """計算適應性分數"""
        base_performance = metrics.get('sharpe_ratio', 0)
        
        # 根據市場狀態調整
        if market_regime == 'high_volatility':
            # 高波動時期，風險控制更重要
            adaptation_score = base_performance * (1 - abs(metrics.get('max_drawdown', 0)))
        elif market_regime == 'trending':
            # 趨勢時期，趨勢跟隨能力重要
            adaptation_score = base_performance * metrics.get('trend_following_accuracy', 0.5)
        elif market_regime == 'ranging':
            # 震盪時期，均值回歸能力重要
            adaptation_score = base_performance * metrics.get('mean_reversion_success', 0.5)
        else:
            adaptation_score = base_performance
            
        return max(0, adaptation_score)
    
    def get_current_strategy(self) -> RewardStrategy:
        """獲取當前策略"""
        return self.current_strategy
    
    def set_strategy(self, strategy: RewardStrategy):
        """設置策略"""
        self.current_strategy = strategy
        logger.info(f"切換到獎勵策略: {strategy.value}")
    
    def get_strategy_description(self, strategy: RewardStrategy) -> str:
        """獲取策略描述"""
        descriptions = {
            RewardStrategy.CONSERVATIVE: "保守策略：重視風險控制和資本保全",
            RewardStrategy.AGGRESSIVE: "激進策略：追求高收益和成長潛力",
            RewardStrategy.BALANCED: "平衡策略：風險收益並重，追求穩定表現",
            RewardStrategy.ADAPTIVE: "自適應策略：根據市場狀況動態調整",
            RewardStrategy.MOMENTUM: "動量策略：專注趨勢跟隨和突破捕捉",
            RewardStrategy.MEAN_REVERSION: "均值回歸策略：利用價格波動和反轉機會"
        }
        return descriptions.get(strategy, "未知策略")
