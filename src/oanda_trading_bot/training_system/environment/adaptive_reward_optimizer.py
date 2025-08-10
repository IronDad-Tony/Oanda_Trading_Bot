"""
自適應獎勵優化器
動態調整獎勵權重以提升訓練效果
"""

from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)

class AdaptiveRewardOptimizer:
    """
    自適應獎勵優化器
    根據訓練表現動態調整獎勵權重
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 性能追蹤窗口
        self.performance_window = 200
        self.recent_rewards = deque(maxlen=self.performance_window)
        self.recent_profits = deque(maxlen=self.performance_window)
        self.recent_drawdowns = deque(maxlen=self.performance_window)
        
        # 權重調整參數
        self.learning_rate = 0.01
        self.weight_momentum = 0.9
        self.weight_clip = (0.1, 5.0)  # 權重範圍限制
        
        # 初始權重
        self.reward_weights = {
            'profit_factor': 1.0,
            'sharpe_ratio': 1.0,
            'max_drawdown': 1.0,
            'win_rate': 1.0,
            'profit_loss_ratio': 1.0,
            'trade_frequency': 1.0,
            'risk_adjusted_return': 1.0
        }
        
        # 權重動量
        self.weight_momentum_dict = {k: 0.0 for k in self.reward_weights.keys()}
        
        # 目標性能指標
        self.target_metrics = {
            'min_sharpe_ratio': 1.5,
            'max_drawdown_threshold': 0.15,
            'min_profit_factor': 1.3,
            'target_win_rate': 0.55
        }
        
    def update_performance_history(self, reward_components: Dict[str, float], 
                                 portfolio_performance: Dict[str, float]):
        """更新性能歷史"""
        
        # 記錄總獎勵
        total_reward = sum(reward_components.values())
        self.recent_rewards.append(total_reward)
        
        # 記錄關鍵指標
        self.recent_profits.append(portfolio_performance.get('total_return', 0))
        self.recent_drawdowns.append(portfolio_performance.get('max_drawdown', 0))
        
    def calculate_performance_gradients(self) -> Dict[str, float]:
        """計算性能梯度"""
        if len(self.recent_rewards) < 50:
            return {k: 0.0 for k in self.reward_weights.keys()}
            
        # 計算近期性能趨勢
        recent_half = list(self.recent_rewards)[-50:]
        earlier_half = list(self.recent_rewards)[-100:-50] if len(self.recent_rewards) >= 100 else []
        
        if not earlier_half:
            return {k: 0.0 for k in self.reward_weights.keys()}
            
        # 性能改善指標
        recent_avg = np.mean(recent_half)
        earlier_avg = np.mean(earlier_half)
        performance_improvement = recent_avg - earlier_avg
        
        # 風險調整後的性能
        recent_profit_avg = np.mean(list(self.recent_profits)[-50:]) if self.recent_profits else 0
        recent_drawdown_avg = np.mean(list(self.recent_drawdowns)[-50:]) if self.recent_drawdowns else 0
        
        # 計算各個獎勵組件的梯度
        gradients = {}
        
        # 如果性能下降，增加風險控制權重
        if performance_improvement < 0:
            gradients['max_drawdown'] = 0.1
            gradients['risk_adjusted_return'] = 0.05
            gradients['profit_factor'] = -0.02
        else:
            gradients['profit_factor'] = 0.05
            gradients['sharpe_ratio'] = 0.03
            gradients['max_drawdown'] = -0.02
            
        # 根據當前表現調整
        if recent_drawdown_avg > self.target_metrics['max_drawdown_threshold']:
            gradients['max_drawdown'] = gradients.get('max_drawdown', 0) + 0.15
            gradients['risk_adjusted_return'] = gradients.get('risk_adjusted_return', 0) + 0.1
            
        # 勝率相關調整
        if recent_profit_avg > 0:
            gradients['win_rate'] = 0.02
            gradients['profit_loss_ratio'] = 0.03
        else:
            gradients['win_rate'] = 0.05
            gradients['profit_loss_ratio'] = 0.05
            
        return gradients
        
    def optimize_weights(self) -> Dict[str, float]:
        """優化權重"""
        gradients = self.calculate_performance_gradients()
        
        # 使用動量更新權重
        for key in self.reward_weights.keys():
            gradient = gradients.get(key, 0.0)
            
            # 動量更新
            self.weight_momentum_dict[key] = (
                self.weight_momentum * self.weight_momentum_dict[key] + 
                self.learning_rate * gradient
            )
            
            # 更新權重
            self.reward_weights[key] += self.weight_momentum_dict[key]
            
            # 權重限制
            self.reward_weights[key] = np.clip(
                self.reward_weights[key], 
                self.weight_clip[0], 
                self.weight_clip[1]
            )
            
        return self.reward_weights.copy()
        
    def get_adaptive_reward_multipliers(self, current_metrics: Dict[str, float]) -> Dict[str, float]:
        """根據當前指標獲取自適應獎勵乘數"""
        multipliers = {}
        
        # Sharpe比率自適應
        current_sharpe = current_metrics.get('sharpe_ratio', 0)
        if current_sharpe < self.target_metrics['min_sharpe_ratio']:
            multipliers['sharpe_boost'] = 1.5
        else:
            multipliers['sharpe_boost'] = 1.0
            
        # 最大回撤自適應
        current_drawdown = abs(current_metrics.get('max_drawdown', 0))
        if current_drawdown > self.target_metrics['max_drawdown_threshold']:
            multipliers['drawdown_penalty'] = 2.0
        else:
            multipliers['drawdown_penalty'] = 1.0
            
        # 獲利因子自適應
        current_profit_factor = current_metrics.get('profit_factor', 1.0)
        if current_profit_factor < self.target_metrics['min_profit_factor']:
            multipliers['profit_boost'] = 1.3
        else:
            multipliers['profit_boost'] = 1.0
            
        return multipliers
        
    def get_current_weights(self) -> Dict[str, float]:
        """獲取當前權重"""
        return self.reward_weights.copy()
        
    def reset_weights(self):
        """重置權重到初始值"""
        self.reward_weights = {
            'profit_factor': 1.0,
            'sharpe_ratio': 1.0,
            'max_drawdown': 1.0,
            'win_rate': 1.0,
            'profit_loss_ratio': 1.0,
            'trade_frequency': 1.0,
            'risk_adjusted_return': 1.0
        }
        self.weight_momentum_dict = {k: 0.0 for k in self.reward_weights.keys()}
