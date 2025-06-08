# src/environment/progressive_reward_system.py
"""
漸進式獎勵系統實現
為元學習系統提供動態獎勵計算和績效評估

主要功能：
1. 策略績效評估
2. 自適應獎勵計算
3. 學習進度追蹤
4. 策略優化指導
5. 三階段獎勵計算流水線（新增）
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from datetime import datetime
from .reward_normalizer import DynamicRewardNormalizer  # 新增導入

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
    漸進式獎勵系統（三階段流水線）
    
    提供動態獎勵計算，支持：
    - 階段1: 基礎獎勵計算
    - 階段2: 動態權重調整
    - 階段3: 獎勵標準化
    - 績效評估
    - 學習進度追蹤
    """
    
    def __init__(self,
                 profit_weight: float = 0.4,
                 risk_weight: float = 0.3,
                 adaptation_weight: float = 0.2,
                 consistency_weight: float = 0.1,
                 device: str = "cpu",
                 volatility_window: int = 100):  # 新增參數
        """
        初始化漸進式獎勵系統（三階段）
        
        Args:
            profit_weight: 利潤權重
            risk_weight: 風險權重
            adaptation_weight: 適應性權重
            consistency_weight: 一致性權重
            device: 計算設備
            volatility_window: 波動率分析窗口大小
        """
        self.profit_weight = profit_weight
        self.risk_weight = risk_weight
        self.adaptation_weight = adaptation_weight
        self.consistency_weight = consistency_weight
        self.device = device
        
        # 初始化三階段組件
        self.dynamic_normalizer = DynamicRewardNormalizer(volatility_window=volatility_window)  # 新增
        
        # 初始化獎勵歷史
        self.reward_history: List[RewardMetrics] = []
        self.performance_baseline = 0.0
        self.adaptation_count = 0
        
        logger.info(f"初始化三階段獎勵系統 - 權重配置: 利潤={profit_weight}, 風險={risk_weight}, 適應={adaptation_weight}, 一致性={consistency_weight}")
        logger.info(f"波動率分析窗口: {volatility_window}")
    
    def calculate_reward(self,
                        profit: float,
                        drawdown: float,
                        volatility: float,
                        market_state: dict,  # 新增市場狀態參數
                        adaptation_success: bool = True,
                        strategy_consistency: float = 1.0) -> RewardMetrics:
        """
        三階段獎勵計算流水線
        
        Args:
            profit: 利潤率
            drawdown: 最大回撤
            volatility: 波動率
            market_state: 市場狀態數據（新增）
            adaptation_success: 適應是否成功
            strategy_consistency: 策略一致性分數
            
        Returns:
            RewardMetrics: 獎勵評估結果
        """
        # ===== 階段1: 基礎獎勵計算 =====
        # 1.1 利潤分數 (範圍: -1 to 1)
        profit_score = np.tanh(profit * 10)
        
        # 1.2 風險分數 (範圍: 0 to 1)
        risk_penalty = max(0, drawdown) + max(0, volatility - 0.1)
        risk_score = max(0, 1 - risk_penalty * 2)
        
        # 1.3 適應性分數
        adaptation_score = 1.0 if adaptation_success else 0.5
        self.adaptation_count += 1 if adaptation_success else 0
        adaptation_bonus = min(0.2, self.adaptation_count * 0.01)
        adaptation_score += adaptation_bonus
        
        # 1.4 一致性分數
        consistency_score = max(0, min(1, strategy_consistency))
        
        # 1.5 計算基礎總獎勵
        base_total = (
            self.profit_weight * profit_score +
            self.risk_weight * risk_score +
            self.adaptation_weight * adaptation_score +
            self.consistency_weight * consistency_score
        )
        
        # ===== 階段2: 動態權重調整 =====
        # 2.1 更新權重基於市場波動率
        self.dynamic_normalizer.update_weights(market_state)
        
        # 2.2 應用動態權重調整
        adjusted_components = {
            'profit_reward': profit_score * self.dynamic_normalizer.component_weights.get('profit_reward', 1.0),
            'risk_penalty': risk_score * self.dynamic_normalizer.component_weights.get('risk_penalty', 1.0),
            'adaptation': adaptation_score * self.dynamic_normalizer.component_weights.get('adaptation', 1.0),
            'consistency': consistency_score * self.dynamic_normalizer.component_weights.get('consistency', 1.0)
        }
        
        # 2.3 計算調整後總獎勵
        adjusted_total = sum(adjusted_components.values())
        
        # ===== 階段3: 獎勵標準化 =====
        # 3.1 準備標準化輸入
        reward_info = {
            'total_reward': adjusted_total,
            'components': adjusted_components
        }
        
        # 3.2 應用標準化
        normalized_info = self.dynamic_normalizer.normalize_reward(reward_info, method='hybrid')
        normalized_total = normalized_info['total_reward']
        
        # ===== 創建最終獎勵指標 =====
        metrics = RewardMetrics(
            profit_score=profit_score,
            risk_score=risk_score,
            adaptation_score=adaptation_score,
            consistency_score=consistency_score,
            total_reward=normalized_total,  # 使用標準化後總獎勵
            timestamp=datetime.now()
        )
        
        # 更新歷史
        self.reward_history.append(metrics)
        if len(self.reward_history) > 1000:
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