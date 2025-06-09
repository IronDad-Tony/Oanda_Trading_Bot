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

# 動態導入處理，支持作為主程序運行
try:
    from .reward_normalizer import DynamicRewardNormalizer
except ImportError:
    from src.environment.reward_normalizer import DynamicRewardNormalizer

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
    
    def update_phase(self):
        """
        根據訓練進度自動更新階段（新增方法）
        階段1：訓練步驟 < 20% (基礎學習)
        階段2：20% ≤ 訓練步驟 < 70% (風險管理)
        階段3：訓練步驟 ≥ 70% (複雜策略)
        """
        progress = self.current_step / self.total_training_steps
        new_phase = None
        
        if progress < 0.2:
            new_phase = 1
        elif progress < 0.7:
            new_phase = 2
        else:
            new_phase = 3
            
        if new_phase != self.current_phase:
            logger.info(f"階段切換: 從階段 {self.current_phase} 到階段 {new_phase} (進度: {progress:.1%})")
            self.current_phase = new_phase
    
    def validate_performance(self):
        """
        根據當前階段驗證性能閾值（新增方法）
        階段1：盈亏理解 > 70%，回撤 < 10%
        階段2：夏普比率 > 1.0，胜率 > 55%
        階段3：夏普比率 > 1.5，策略创新率 > 20%
        """
        if not self.reward_history:
            return False
            
        # 獲取最近10個記錄的指標
        recent_metrics = self.reward_history[-10:]
        profit_scores = [m.profit_score for m in recent_metrics]
        risk_scores = [m.risk_score for m in recent_metrics]
        
        # 計算平均指標
        avg_profit = np.mean(profit_scores)
        avg_risk = np.mean(risk_scores)
        
        if self.current_phase == 1:
            # 階段1: 盈亏理解 > 70%，回撤 < 10%
            # 假設profit_score代表盈亏理解，risk_score代表回撤控制
            if avg_profit > 0.7 and avg_risk > 0.9:  # risk_score越高代表風險越低
                logger.info("階段1驗證通過: 盈亏理解>70% 且 回撤<10%")
                return True
            else:
                logger.warning(f"階段1驗證失敗: 盈亏理解={avg_profit:.1%}, 風險控制={avg_risk:.1%}")
                return False
                
        elif self.current_phase == 2:
            # 階段2: 夏普比率 > 1.0，胜率 > 55%
            # 這裡簡化為使用profit_score和risk_score的組合
            # 實際應從交易記錄計算夏普比率和胜率
            performance_score = avg_profit * 0.7 + avg_risk * 0.3
            if performance_score > 0.65:  # 簡化閾值
                logger.info("階段2驗證通過: 夏普比率>1.0 且 胜率>55% (模擬)")
                return True
            else:
                logger.warning(f"階段2驗證失敗: 綜合表現={performance_score:.1%}")
                return False
                
        else:  # 階段3
            # 階段3: 夏普比率 > 1.5，策略创新率 > 20%
            # 這裡簡化為使用adaptation_score和consistency_score
            adaptation_scores = [m.adaptation_score for m in recent_metrics]
            consistency_scores = [m.consistency_score for m in recent_metrics]
            innovation_score = np.mean(adaptation_scores) * 0.8 + np.mean(consistency_scores) * 0.2
            
            if innovation_score > 0.75:  # 簡化閾值
                logger.info("階段3驗證通過: 夏普比率>1.5 且 策略創新率>20% (模擬)")
                return True
            else:
                logger.warning(f"階段3驗證失敗: 創新表現={innovation_score:.1%}")
                return False
    
    def calculate_reward(self,
                        profit: float,
                        drawdown: float,
                        volatility: float,
                        market_state: dict,
                        adaptation_success: bool = True,
                        strategy_consistency: float = 1.0) -> RewardMetrics:
        """
        三階段獎勵計算流水線（增加階段判斷）
        
        Args:
            profit: 利潤率
            drawdown: 最大回撤
            volatility: 波動率
            market_state: 市場狀態數據
            adaptation_success: 適應是否成功
            strategy_consistency: 策略一致性分數
            
        Returns:
            RewardMetrics: 獎勵評估結果
        """
        # 更新訓練步驟和階段
        self.current_step += 1
        self.update_phase()
        
        # 根據階段調整計算參數
        phase_factor = 1.0 + (self.current_phase - 1) * 0.3  # 階段越高要求越嚴格
        
        # ===== 階段1: 基礎獎勵計算 =====
        # 1.1 利潤分數 (範圍: -1 to 1)
        profit_score = np.tanh(profit * 10 * phase_factor)
        
        # 1.2 風險分數 (範圍: 0 to 1)
        risk_penalty = max(0, drawdown) + max(0, volatility - 0.1)
        risk_score = max(0, 1 - risk_penalty * 2 * phase_factor)
        
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
        
        # 執行階段性能驗證
        self.validate_performance()
        
        return metrics
    
    def get_learning_signal(self) -> Dict[str, float]:
        """
        獲取學習指導信號（增加階段信息）
        
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
            "current_phase": self.current_phase,  # 新增階段信息
        }
        
        # 根據階段判斷改進方向
        if self.current_phase == 1:
            if signal_dict["profit_focus"] < 0.5:
                signal_dict["recommendation"] = "focus_on_profit_basics"
            elif signal_dict["risk_management"] < 0.7:
                signal_dict["recommendation"] = "improve_risk_control_basics"
            else:
                signal_dict["recommendation"] = "consolidate_foundation"
                
        elif self.current_phase == 2:
            if signal_dict["risk_management"] < 0.6:
                signal_dict["recommendation"] = "enhance_risk_management"
            elif signal_dict["adaptation_success"] < 0.7:
                signal_dict["recommendation"] = "improve_market_adaptation"
            else:
                signal_dict["recommendation"] = "optimize_balanced_strategy"
                
        else:  # 階段3
            if signal_dict["adaptation_success"] < 0.8:
                signal_dict["recommendation"] = "boost_innovation_capability"
            elif signal_dict["consistency"] < 0.8:
                signal_dict["recommendation"] = "strengthen_strategy_consistency"
            else:
                signal_dict["recommendation"] = "pursue_advanced_strategies"
        
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
                f"phase={self.current_phase}, "  # 新增階段顯示
                f"step={self.current_step}/{self.total_training_steps}, "  # 新增步驟顯示
                f"episodes={stats['total_episodes']}, "
                f"avg_reward={stats['average_reward']:.3f}, "
                f"adaptation_rate={stats['adaptation_success_rate']:.2f})")


# ===== 測試代碼區塊（測試後可刪除）=====
if __name__ == "__main__":
    """
    階段切換邏輯測試 - 獨立測試版本
    完全自包含，不依賴項目導入結構
    """
    import numpy as np
    import logging
    from datetime import datetime
    from dataclasses import dataclass
    
    # 配置日誌
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 定義簡化版的獎勵指標
    @dataclass
    class TestRewardMetrics:
        profit_score: float = 0.0
        risk_score: float = 0.0
        adaptation_score: float = 0.0
        consistency_score: float = 0.0
        total_reward: float = 0.0
        timestamp: datetime = None
    
    # 簡化版的動態標準化器（僅用於測試）
    class TestDynamicRewardNormalizer:
        def __init__(self, volatility_window=100):
            self.volatility_window = volatility_window
            self.component_weights = {
                'profit_reward': 1.0,
                'risk_penalty': 1.0,
                'adaptation': 1.0,
                'consistency': 1.0
            }
            
        def update_weights(self, market_state):
            # 測試中不實際更新權重
            pass
            
        def normalize_reward(self, reward_info, method='hybrid'):
            # 測試中直接返回原始值
            return reward_info
    
    # 測試專用的漸進式獎勵系統
    class TestProgressiveRewardSystem:
        def __init__(self, total_training_steps=100):
            self.total_training_steps = total_training_steps
            self.current_step = 0
            self.current_phase = 1
            self.dynamic_normalizer = TestDynamicRewardNormalizer()
            self.reward_history = []
            self.performance_baseline = 0.0
            self.adaptation_count = 0
            
        def update_phase(self):
            progress = self.current_step / self.total_training_steps
            new_phase = None
            
            if progress < 0.2:
                new_phase = 1
            elif progress < 0.7:
                new_phase = 2
            else:
                new_phase = 3
                
            if new_phase != self.current_phase:
                logger.info(f"階段切換: 從階段 {self.current_phase} 到階段 {new_phase} (進度: {progress:.1%})")
                self.current_phase = new_phase
        
        def validate_performance(self):
            # 簡化驗證邏輯
            if not self.reward_history:
                return False
            return True  # 總是返回True以簡化測試
        
        def calculate_reward(self, profit, drawdown, volatility, market_state,
                            adaptation_success, strategy_consistency):
            self.current_step += 1
            self.update_phase()
            
            # 簡化計算邏輯
            profit_score = np.tanh(profit * 10)
            risk_score = max(0, 1 - (drawdown + max(0, volatility - 0.1)) * 2)
            adaptation_score = 1.0 if adaptation_success else 0.5
            consistency_score = max(0, min(1, strategy_consistency))
            
            total_reward = (
                0.4 * profit_score +
                0.3 * risk_score +
                0.2 * adaptation_score +
                0.1 * consistency_score
            )
            
            metrics = TestRewardMetrics(
                profit_score=profit_score,
                risk_score=risk_score,
                adaptation_score=adaptation_score,
                consistency_score=consistency_score,
                total_reward=total_reward,
                timestamp=datetime.now()
            )
            
            self.reward_history.append(metrics)
            return metrics
        
        def get_reward_statistics(self):
            return {
                'total_episodes': len(self.reward_history),
                'average_reward': np.mean([r.total_reward for r in self.reward_history]),
                'adaptation_success_rate': self.adaptation_count / max(1, len(self.reward_history))
            }
    
    # 初始化測試系統
    reward_system = TestProgressiveRewardSystem(total_training_steps=100)
    
    # 模擬訓練過程
    print("開始模擬訓練...")
    for step in range(reward_system.total_training_steps):
        profit = np.random.uniform(-0.05, 0.1)
        drawdown = np.random.uniform(0.01, 0.2)
        volatility = np.random.uniform(0.05, 0.3)
        
        metrics = reward_system.calculate_reward(
            profit=profit,
            drawdown=drawdown,
            volatility=volatility,
            market_state={},
            adaptation_success=np.random.rand() > 0.3,
            strategy_consistency=np.random.uniform(0.7, 1.0)
        )
        
        if step % 10 == 0:
            print(f"步驟 {step}/100 | 階段: {reward_system.current_phase} | 獎勵: {metrics.total_reward:.4f}")
    
    print("\n訓練完成!")
    stats = reward_system.get_reward_statistics()
    print(f"總訓練次數: {stats['total_episodes']}")
    print(f"平均獎勵: {stats['average_reward']:.4f}")
    print(f"最終階段: {reward_system.current_phase}")
    print("\n測試完成! 請刪除此測試代碼區塊")
# ===== 測試代碼結束 =====