# src/environment/progressive_reward_system.py
"""
漸進式獎勵系統實現
為元學習系統提供動態獎勵計算和績效評估

主要功能：
1. 策略績效評估
2. 自適應獎勵計算
3. 學習進度追蹤
4. 策略優化指導
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class RewardMetrics:
    """獎勵評估指標"""
    profit_score: float = 0.0
    risk_score: float = 0.0
    adaptation_score: float = 0.0
    consistency_score: float = 0.0
    total_reward: float = 0.0
    timestamp: datetime = None

class ProgressiveRewardSystem:
    """
    漸進式獎勵系統
    
    提供動態獎勵計算，支持：
    - 基於績效的獎勵
    - 自適應學習獎勵  
    - 風險調整獎勵
    - 一致性獎勵
    """
    
    def __init__(self, 
                 profit_weight: float = 0.4,
                 risk_weight: float = 0.3,
                 adaptation_weight: float = 0.2,
                 consistency_weight: float = 0.1,
                 device: str = "cpu"):
        """
        初始化漸進式獎勵系統
        
        Args:
            profit_weight: 利潤權重
            risk_weight: 風險權重  
            adaptation_weight: 適應性權重
            consistency_weight: 一致性權重
            device: 計算設備
        """
        self.profit_weight = profit_weight
        self.risk_weight = risk_weight
        self.adaptation_weight = adaptation_weight
        self.consistency_weight = consistency_weight
        self.device = device
        
        # 初始化獎勵歷史
        self.reward_history: List[RewardMetrics] = []
        self.performance_baseline = 0.0
        self.adaptation_count = 0
        
        logger.info(f"初始化漸進式獎勵系統 - 權重配置: 利潤={profit_weight}, 風險={risk_weight}, 適應={adaptation_weight}, 一致性={consistency_weight}")
    
    def calculate_reward(self,
                        profit: float,
                        drawdown: float,
                        volatility: float,
                        adaptation_success: bool = True,
                        strategy_consistency: float = 1.0) -> RewardMetrics:
        """
        計算綜合獎勵分數
        
        Args:
            profit: 利潤率
            drawdown: 最大回撤
            volatility: 波動率
            adaptation_success: 適應是否成功
            strategy_consistency: 策略一致性分數
            
        Returns:
            RewardMetrics: 獎勵評估結果
        """
        # 1. 利潤分數 (範圍: -1 to 1)
        profit_score = np.tanh(profit * 10)  # 將利潤映射到[-1,1]
        
        # 2. 風險分數 (範圍: 0 to 1，越低越好)
        risk_penalty = max(0, drawdown) + max(0, volatility - 0.1)
        risk_score = max(0, 1 - risk_penalty * 2)
        
        # 3. 適應性分數
        adaptation_score = 1.0 if adaptation_success else 0.5
        if hasattr(self, 'adaptation_count'):
            self.adaptation_count += 1 if adaptation_success else 0
            # 獎勵頻繁成功適應
            adaptation_bonus = min(0.2, self.adaptation_count * 0.01)
            adaptation_score += adaptation_bonus
        
        # 4. 一致性分數
        consistency_score = max(0, min(1, strategy_consistency))
        
        # 5. 計算總獎勵
        total_reward = (
            self.profit_weight * profit_score +
            self.risk_weight * risk_score +
            self.adaptation_weight * adaptation_score +
            self.consistency_weight * consistency_score
        )
        
        # 6. 創建獎勵指標
        metrics = RewardMetrics(
            profit_score=profit_score,
            risk_score=risk_score,
            adaptation_score=adaptation_score,
            consistency_score=consistency_score,
            total_reward=total_reward,
            timestamp=datetime.now()
        )
        
        # 7. 更新歷史
        self.reward_history.append(metrics)
        if len(self.reward_history) > 1000:  # 保持歷史記錄在合理範圍
            self.reward_history = self.reward_history[-1000:]
        
        return metrics
    
    def get_learning_signal(self) -> Dict[str, float]:
        """
        獲取學習指導信號
        
        Returns:
            Dict: 包含學習建議的字典
        """
        if len(self.reward_history) < 5:
            return {"signal": "insufficient_data", "confidence": 0.0}
        
        recent_rewards = [r.total_reward for r in self.reward_history[-10:]]
        avg_reward = np.mean(recent_rewards)
        reward_trend = np.mean(np.diff(recent_rewards)) if len(recent_rewards) > 1 else 0
        
        signal_dict = {
            "current_performance": avg_reward,
            "trend": reward_trend,
            "profit_focus": np.mean([r.profit_score for r in self.reward_history[-5:]]),
            "risk_management": np.mean([r.risk_score for r in self.reward_history[-5:]]),
            "adaptation_success": np.mean([r.adaptation_score for r in self.reward_history[-5:]]),
            "consistency": np.mean([r.consistency_score for r in self.reward_history[-5:]]),
        }
        
        # 判斷主要改進方向
        if signal_dict["profit_focus"] < 0.3:
            signal_dict["recommendation"] = "focus_on_profit"
        elif signal_dict["risk_management"] < 0.5:
            signal_dict["recommendation"] = "improve_risk_control"
        elif signal_dict["consistency"] < 0.6:
            signal_dict["recommendation"] = "enhance_consistency"
        else:
            signal_dict["recommendation"] = "maintain_balance"
        
        return signal_dict
    
    def update_baseline(self, new_baseline: float):
        """更新績效基準線"""
        self.performance_baseline = new_baseline
        logger.info(f"更新績效基準線: {new_baseline:.4f}")
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """
        獲取獎勵統計信息
        
        Returns:
            Dict: 統計信息
        """
        if not self.reward_history:
            return {"status": "no_data"}
        
        rewards = [r.total_reward for r in self.reward_history]
        
        return {
            "total_episodes": len(self.reward_history),
            "average_reward": np.mean(rewards),
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards),
            "reward_std": np.std(rewards),
            "recent_performance": np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards),
            "adaptation_success_rate": self.adaptation_count / max(1, len(self.reward_history)),
            "performance_vs_baseline": np.mean(rewards) - self.performance_baseline
        }
    
    def reset_history(self):
        """重置獎勵歷史"""
        self.reward_history.clear()
        self.adaptation_count = 0
        logger.info("獎勵歷史已重置")
    
    def __repr__(self):
        stats = self.get_reward_statistics()
        if stats.get("status") == "no_data":
            return "ProgressiveRewardSystem(episodes=0)"
        
        return (f"ProgressiveRewardSystem("
                f"episodes={stats['total_episodes']}, "
                f"avg_reward={stats['average_reward']:.3f}, "
                f"adaptation_rate={stats['adaptation_success_rate']:.2f})")