# src/environment/progressive_learning_system.py
"""
漸進式學習系統 - 模型全面增強計畫階段二實施
實現三階段學習框架，基於實際交易表現進行動態切換

階段一：基礎交易原理學習 (基本盈虧概念和風險控制)
階段二：風險管理強化 (複雜風險指標和績效評估)  
階段三：複雜策略掌握 (完整複雜獎勵函數，發展高級策略)

主要特點：
- 完全基於交易表現的階段切換
- 三種漸進式獎勵函數
- 智能進階條件判斷
- 詳細學習進度追蹤
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math

try:
    from src.common.logger_setup import logger
    from src.common.config import DEVICE
except ImportError:
    logger = logging.getLogger(__name__)
    DEVICE = "cpu"


class LearningStage(Enum):
    """學習階段枚舉"""
    BASIC = 1       # 基礎學習階段
    INTERMEDIATE = 2 # 中級學習階段  
    ADVANCED = 3    # 高級學習階段


@dataclass
class LearningMetrics:
    """學習進度指標"""
    stage: LearningStage
    episode: int
    stage_episodes: int
    stage_progress: float  # 當前階段進度 (0-1)
    stage_performance: float  # 當前階段表現
    advancement_progress: float  # 進階進度 (0-1)
    should_advance: bool = False  # 是否應該進入下一階段
    advancement_reason: str = ""  # 進階原因
    timestamp: datetime = None


@dataclass
class RewardComponents:
    """獎勵組件詳情"""
    basic_pnl: float = 0.0
    risk_penalty: float = 0.0
    trade_frequency: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    sortino_ratio: float = 0.0
    var_risk: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    transaction_costs: float = 0.0
    consistency_bonus: float = 0.0
    learning_bonus: float = 0.0  # 新增學習進步獎勵
    total_reward: float = 0.0


class Stage1BasicReward(nn.Module):
    """
    階段一：基礎交易原理學習獎勵函數
    目標：學習基本的盈虧概念和風險控制
    
    學習重點：
    - 基本的買賣時機
    - 風險意識培養
    - 避免過度交易
    """
    
    def __init__(self):
        super().__init__()
        self.pnl_weight = 0.70      # 基本盈虧權重
        self.risk_weight = 0.20     # 風險控制權重
        self.frequency_weight = 0.10 # 交易頻率權重
        
    def forward(self, metrics: Dict[str, float]) -> RewardComponents:
        """計算階段一獎勵 (優化初期學習)"""
        components = RewardComponents()
        
        # 1. 基本盈虧 (權重70%) - 使用更溫和的映射
        pnl = metrics.get('pnl', 0.0)
        components.basic_pnl = np.tanh(pnl * 5) * self.pnl_weight  # 降低敏感度
        
        # 2. 簡單風險控制 (權重20%) - 更寬容的風險評估
        drawdown = metrics.get('drawdown', 0.0)
        if drawdown > 0.10:  # 放寬到回撤>10%
            components.risk_penalty = -2.0 * self.risk_weight  # 減少懲罰強度
        elif drawdown > 0.05:  # 中等回撤輕微懲罰
            components.risk_penalty = -0.5 * self.risk_weight
        else:
            components.risk_penalty = 0.5 * self.risk_weight  # 獎勵低回撤
            
        # 3. 交易頻率控制 (權重10%) - 鼓勵合理交易
        trade_freq = metrics.get('trade_frequency', 0.0)
        if trade_freq > 0.15:  # 放寬過度交易閾值
            components.trade_frequency = -1.0 * self.frequency_weight  # 減少懲罰
        elif trade_freq < 0.02:  # 懲罰過少交易
            components.trade_frequency = -0.5 * self.frequency_weight
        else:
            components.trade_frequency = 0.3 * self.frequency_weight  # 獎勵適中交易
            
        # 4. 新增學習進步獎勵 - 鼓勵任何形式的改善
        learning_bonus = 0.0
        if pnl > -0.02:  # 小虧損或盈利給予獎勵
            learning_bonus = 0.1
        if drawdown < 0.08:  # 良好的風險控制
            learning_bonus += 0.05
        
        components.total_reward = (components.basic_pnl + 
                                 components.risk_penalty + 
                                 components.trade_frequency +
                                 learning_bonus)
        
        return components


class Stage2IntermediateReward(nn.Module):
    """
    階段二：風險管理強化獎勵函數
    目標：引入更複雜的風險指標和績效評估
    
    學習重點：
    - 風險調整後收益優化
    - 勝率與盈虧比平衡
    - 穩定性追求
    """
    
    def __init__(self):
        super().__init__()
        self.base_reward = Stage1BasicReward()
        self.sharpe_weight = 0.25
        self.drawdown_weight = 0.20
        self.winrate_weight = 0.15
        
    def forward(self, metrics: Dict[str, float]) -> RewardComponents:
        """計算階段二獎勵"""
        # 繼承階段一基礎獎勵 (但降低權重)
        components = self.base_reward(metrics)
        
        # 調整基礎獎勵權重
        components.basic_pnl *= 0.7
        components.risk_penalty *= 0.7
        components.trade_frequency *= 0.7
        
        # 新增夏普比率獎勵
        sharpe = metrics.get('sharpe_ratio', 0.0)
        components.sharpe_ratio = np.tanh(sharpe) * self.sharpe_weight
        
        # 新增最大回撤控制
        max_drawdown = metrics.get('max_drawdown', 0.0)
        components.max_drawdown = -max_drawdown * 8.0 * self.drawdown_weight
        
        # 新增勝率激勵
        win_rate = metrics.get('win_rate', 0.0)
        if win_rate > 0.6:
            components.win_rate = 2.0 * self.winrate_weight
        elif win_rate < 0.4:
            components.win_rate = -1.0 * self.winrate_weight
        else:
            components.win_rate = 0.0
            
        # 重新計算總獎勵
        components.total_reward = (components.basic_pnl + 
                                 components.risk_penalty + 
                                 components.trade_frequency +
                                 components.sharpe_ratio +
                                 components.max_drawdown +
                                 components.win_rate)
        
        return components


class Stage3AdvancedReward(nn.Module):
    """
    階段三：複雜策略掌握獎勵函數
    目標：使用完整的複雜獎勵函數，發展高級策略
    
    學習重點：
    - 複雜市場環境適應
    - 多策略動態組合
    - 超人類策略創新
    """
    
    def __init__(self):
        super().__init__()
        self.stage2_reward = Stage2IntermediateReward()
        
        # 高級指標權重
        self.sortino_weight = 0.15
        self.var_weight = 0.12
        self.skewness_weight = 0.08
        self.kurtosis_weight = 0.08
        self.cost_weight = 0.05
        self.consistency_weight = 0.10
        
    def forward(self, metrics: Dict[str, float]) -> RewardComponents:
        """計算階段三獎勵"""
        # 繼承階段二獎勵 (但進一步調整權重)
        components = self.stage2_reward(metrics)
        
        # 調整前期獎勵權重
        components.basic_pnl *= 0.8
        components.risk_penalty *= 0.8
        components.trade_frequency *= 0.8
        components.sharpe_ratio *= 0.9
        components.max_drawdown *= 0.9
        components.win_rate *= 0.9
        
        # Sortino比率 (只考慮下行風險)
        sortino = metrics.get('sortino_ratio', 0.0)
        components.sortino_ratio = np.tanh(sortino) * self.sortino_weight
        
        # VaR風險 (Value at Risk)
        var_risk = metrics.get('var_risk', 0.0)
        components.var_risk = -abs(var_risk) * self.var_weight
        
        # 收益分佈偏度 (獎勵正偏)
        skewness = metrics.get('skewness', 0.0)
        components.skewness = skewness * self.skewness_weight
        
        # 收益分佈峰度 (懲罰極端峰度)
        kurtosis = metrics.get('kurtosis', 0.0)
        excess_kurtosis = kurtosis - 3.0  # 超額峰度
        components.kurtosis = -abs(excess_kurtosis) * 0.1 * self.kurtosis_weight
        
        # 交易成本
        transaction_costs = metrics.get('transaction_costs', 0.0)
        components.transaction_costs = -transaction_costs * self.cost_weight
        
        # 一致性獎勵 (獎勵穩定表現)
        consistency = metrics.get('consistency_score', 0.0)
        components.consistency_bonus = consistency * self.consistency_weight
        
        # 重新計算總獎勵
        components.total_reward = (components.basic_pnl + 
                                 components.risk_penalty + 
                                 components.trade_frequency +
                                 components.sharpe_ratio +
                                 components.max_drawdown +
                                 components.win_rate +
                                 components.sortino_ratio +
                                 components.var_risk +
                                 components.skewness +
                                 components.kurtosis +
                                 components.transaction_costs +
                                 components.consistency_bonus)
        
        return components


class ProgressiveLearningSystem:
    """
    漸進式學習系統主控制器
    基於實際交易表現管理三階段學習進程和獎勵函數切換
    """
    
    def __init__(self, 
                 min_stage_episodes: int = 50,      # 每階段最少回合數
                 performance_window: int = 20,       # 性能評估窗口
                 advancement_patience: int = 10,     # 進階耐心值
                 device: str = "cpu"):
        """
        初始化漸進式學習系統
        
        Args:
            min_stage_episodes: 每階段最少回合數
            performance_window: 性能評估窗口大小
            advancement_patience: 達到進階條件後的等待回合數
            device: 計算設備
        """
        self.min_stage_episodes = min_stage_episodes
        self.performance_window = performance_window
        self.advancement_patience = advancement_patience
        self.device = device
        
        # 當前狀態
        self.current_episode = 0
        self.current_stage = LearningStage.BASIC
        self.stage_episodes = 0
        self.force_advancement = False
        
        # 獎勵函數
        self.stage1_reward = Stage1BasicReward()
        self.stage2_reward = Stage2IntermediateReward()
        self.stage3_reward = Stage3AdvancedReward()
        
        # 性能追蹤
        self.stage_performance_history = {
            LearningStage.BASIC: [],
            LearningStage.INTERMEDIATE: [],
            LearningStage.ADVANCED: []
        }
        
        # 進階條件追蹤
        self.advancement_progress = {
            LearningStage.BASIC: 0,
            LearningStage.INTERMEDIATE: 0
        }
          # 進階條件定義 (基於實際交易表現，優化初期學習)
        self.advancement_criteria = {
            LearningStage.BASIC: {
                'min_episodes': min_stage_episodes,
                'min_avg_reward': -0.10,         # 降低初期獎勵要求 (從-0.25提升到-0.10)
                'win_rate_threshold': 0.35,      # 降低初期勝率要求 (從0.45到0.35)
                'max_drawdown_threshold': 0.15,  # 放寬初期回撤限制 (從0.10到0.15)
                'consistency_episodes': 5,       # 減少連續達標要求 (從10到5)
                'improvement_threshold': 0.05,   # 新增：獎勵改善要求
                'description': '基礎交易技能掌握'
            },
            LearningStage.INTERMEDIATE: {
                'min_episodes': min_stage_episodes,
                'min_avg_reward': 0.15,          # 中級階段合理獎勵要求
                'sharpe_threshold': 0.5,         # 降低夏普比率要求 (從1.0到0.5)
                'max_drawdown_threshold': 0.10,  # 回撤控制
                'win_rate_threshold': 0.45,      # 中級勝率要求 (從0.55到0.45)
                'consistency_episodes': 8,       # 中級連續達標要求 (從15到8)
                'improvement_threshold': 0.08,   # 新增：獎勵改善要求
                'description': '風險管理能力達標'
            }
        }
        
        logger.info(f"初始化漸進式學習系統:")
        logger.info(f"  當前階段: {self.current_stage.name}")
        logger.info(f"  每階段最少回合: {min_stage_episodes}")
        logger.info(f"  性能評估窗口: {performance_window}")
        logger.info(f"  進階耐心值: {advancement_patience}")
    
    def get_current_stage(self) -> LearningStage:
        """獲取當前學習階段"""
        return self.current_stage
    
    def get_current_reward_function(self):
        """獲取當前階段的獎勵函數"""
        if self.current_stage == LearningStage.BASIC:
            return self.stage1_reward
        elif self.current_stage == LearningStage.INTERMEDIATE:
            return self.stage2_reward
        else:
            return self.stage3_reward
    
    def calculate_reward(self, metrics: Dict[str, float]) -> Tuple[float, RewardComponents, LearningMetrics]:
        """
        計算當前階段獎勵並更新學習進度
        
        Args:
            metrics: 交易績效指標
            
        Returns:
            Tuple[總獎勵, 獎勵組件, 學習指標]
        """
        # 獲取當前獎勵函數
        reward_fn = self.get_current_reward_function()
        
        # 計算獎勵組件
        reward_components = reward_fn(metrics)
        total_reward = reward_components.total_reward
        
        # 更新性能歷史
        self.stage_performance_history[self.current_stage].append(total_reward)
        
        # 檢查是否應該進階
        should_advance, advancement_reason = self._should_advance_stage(metrics, total_reward)
        
        # 創建學習指標
        learning_metrics = self._create_learning_metrics(should_advance, advancement_reason)
        
        # 更新階段
        if should_advance:
            self._advance_to_next_stage(advancement_reason)
        
        # 更新計數器
        self.current_episode += 1
        self.stage_episodes += 1
        
        return total_reward, reward_components, learning_metrics
    
    def _should_advance_stage(self, metrics: Dict[str, float], current_reward: float) -> Tuple[bool, str]:
        """判斷是否應該進入下一階段"""
        if self.force_advancement:
            return True, "強制進階"
            
        if self.current_stage == LearningStage.ADVANCED:
            return False, ""  # 已經是最後階段
        
        # 檢查最小回合數
        if self.stage_episodes < self.min_stage_episodes:
            return False, f"回合數不足 ({self.stage_episodes}/{self.min_stage_episodes})"
        
        # 獲取當前階段標準
        criteria = self.advancement_criteria[self.current_stage]
        stage_history = self.stage_performance_history[self.current_stage]
          # 計算最近性能和改善趨勢
        recent_window = min(len(stage_history), self.performance_window)
        if recent_window < 5:  # 需要至少5個數據點
            return False, "數據點不足"
        
        recent_rewards = stage_history[-recent_window:]
        avg_reward = np.mean(recent_rewards)
        
        # 計算改善趨勢 (比較前半段和後半段)
        if len(stage_history) >= 10:
            early_rewards = stage_history[:len(stage_history)//2]
            late_rewards = stage_history[len(stage_history)//2:]
            improvement = np.mean(late_rewards) - np.mean(early_rewards)
        else:
            improvement = 0.0
        
        # 檢查基本性能要求 (結合絕對值和改善趨勢)
        min_reward_met = avg_reward >= criteria['min_avg_reward']
        improvement_met = improvement >= criteria.get('improvement_threshold', 0.0)
        
        if not (min_reward_met or improvement_met):
            return False, f"獎勵不足且無改善 (獎勵:{avg_reward:.3f}, 改善:{improvement:.3f})"
        
        # 階段特定檢查
        advancement_conditions = []
        
        if self.current_stage == LearningStage.BASIC:
            # 基礎階段檢查
            win_rate = metrics.get('win_rate', 0.0)
            max_drawdown = metrics.get('max_drawdown', 1.0)
            
            conditions_met = (
                win_rate >= criteria['win_rate_threshold'] and
                max_drawdown <= criteria['max_drawdown_threshold']
            )
            
            advancement_conditions = [
                f"勝率: {win_rate:.3f}>={criteria['win_rate_threshold']}",
                f"最大回撤: {max_drawdown:.3f}<={criteria['max_drawdown_threshold']}"
            ]
            
        elif self.current_stage == LearningStage.INTERMEDIATE:
            # 中級階段檢查
            sharpe = metrics.get('sharpe_ratio', 0.0)
            max_drawdown = metrics.get('max_drawdown', 1.0)
            win_rate = metrics.get('win_rate', 0.0)
            
            conditions_met = (
                sharpe >= criteria['sharpe_threshold'] and
                max_drawdown <= criteria['max_drawdown_threshold'] and
                win_rate >= criteria['win_rate_threshold']
            )
            
            advancement_conditions = [
                f"夏普比率: {sharpe:.3f}>={criteria['sharpe_threshold']}",
                f"最大回撤: {max_drawdown:.3f}<={criteria['max_drawdown_threshold']}",
                f"勝率: {win_rate:.3f}>={criteria['win_rate_threshold']}"
            ]
        
        else:
            conditions_met = False
            advancement_conditions = ["已達最高階段"]
          # 更新進階進度 (使用更靈活的標準)
        current_progress = self.advancement_progress[self.current_stage]
        
        if conditions_met or (min_reward_met and improvement_met):
            self.advancement_progress[self.current_stage] += 1
            progress_reason = "條件達標" if conditions_met else "獎勵改善顯著"
        else:
            # 如果有部分改善，保持部分進度，不完全重置
            if improvement > 0 or avg_reward > criteria['min_avg_reward'] * 0.8:
                self.advancement_progress[self.current_stage] = max(0, current_progress - 1)
            else:
                self.advancement_progress[self.current_stage] = 0  # 重置進度
        
        # 檢查是否滿足連續性要求 (使用動態標準)
        required_consistency = criteria['consistency_episodes']
        current_progress = self.advancement_progress[self.current_stage]
        
        # 如果學習進度良好，可以適當降低連續性要求
        if improvement > criteria.get('improvement_threshold', 0.0) * 2:
            required_consistency = max(3, required_consistency - 2)  # 最少3次
        
        if current_progress >= required_consistency:
            if conditions_met:
                reason = f"{criteria['description']} - 連續{current_progress}回合達標"
            else:
                reason = f"{criteria['description']} - 獎勵改善顯著 ({improvement:.3f})"
            return True, reason
        
        return False, f"進階進度: {current_progress}/{required_consistency} ({progress_reason if 'progress_reason' in locals() else '標準未達標'}) - " + ", ".join(advancement_conditions)
    
    def _advance_to_next_stage(self, reason: str):
        """進入下一學習階段"""
        old_stage = self.current_stage
        
        if self.current_stage == LearningStage.BASIC:
            self.current_stage = LearningStage.INTERMEDIATE
            logger.info(f"🎯 學習階段升級: 基礎學習 → 風險管理強化")
        elif self.current_stage == LearningStage.INTERMEDIATE:
            self.current_stage = LearningStage.ADVANCED
            logger.info(f"🚀 學習階段升級: 風險管理 → 複雜策略掌握")
        
        logger.info(f"   升級原因: {reason}")
        logger.info(f"   階段{old_stage.name}完成: {self.stage_episodes}回合")
        
        # 重置階段計數器
        self.stage_episodes = 0
        self.force_advancement = False
        
        # 重置進階進度
        if old_stage in self.advancement_progress:
            self.advancement_progress[old_stage] = 0
    
    def _create_learning_metrics(self, should_advance: bool, advancement_reason: str) -> LearningMetrics:
        """創建學習進度指標"""
        # 計算階段表現
        stage_history = self.stage_performance_history[self.current_stage]
        stage_performance = np.mean(stage_history[-10:]) if len(stage_history) >= 10 else np.mean(stage_history) if stage_history else 0.0
        
        # 計算階段進度 (基於性能改善)
        if len(stage_history) >= 20:
            early_performance = np.mean(stage_history[:10])
            recent_performance = np.mean(stage_history[-10:])
            stage_progress = min(1.0, max(0.0, (recent_performance - early_performance + 1) / 2))
        else:
            stage_progress = len(stage_history) / max(self.min_stage_episodes, 20)
        
        # 計算進階進度
        if self.current_stage in self.advancement_progress:
            required_consistency = self.advancement_criteria[self.current_stage]['consistency_episodes']
            current_progress = self.advancement_progress[self.current_stage]
            advancement_progress = current_progress / required_consistency
        else:
            advancement_progress = 1.0  # 最高階段
        
        return LearningMetrics(
            stage=self.current_stage,
            episode=self.current_episode,
            stage_episodes=self.stage_episodes,
            stage_progress=stage_progress,
            stage_performance=stage_performance,
            advancement_progress=advancement_progress,
            should_advance=should_advance,
            advancement_reason=advancement_reason,
            timestamp=datetime.now()
        )
    
    def force_stage_advancement(self):
        """強制進入下一階段"""
        if self.current_stage != LearningStage.ADVANCED:
            self.force_advancement = True
            logger.info(f"設置強制進階標誌: {self.current_stage.name}")
    
    def get_stage_criteria_status(self) -> Dict[str, Any]:
        """獲取當前階段的條件達成狀態"""
        if self.current_stage == LearningStage.ADVANCED:
            return {"stage": "ADVANCED", "status": "已達最高階段"}
        
        criteria = self.advancement_criteria[self.current_stage]
        stage_history = self.stage_performance_history[self.current_stage]
        
        if len(stage_history) < 5:
            return {"stage": self.current_stage.name, "status": "數據不足"}
        
        recent_rewards = stage_history[-self.performance_window:]
        avg_reward = np.mean(recent_rewards)
        
        current_progress = self.advancement_progress[self.current_stage]
        required_consistency = criteria['consistency_episodes']
        
        return {
            "stage": self.current_stage.name,
            "stage_episodes": self.stage_episodes,
            "min_episodes": criteria['min_episodes'],
            "avg_reward": avg_reward,
            "required_avg_reward": criteria['min_avg_reward'],
            "advancement_progress": current_progress,
            "required_consistency": required_consistency,
            "progress_percentage": (current_progress / required_consistency) * 100,
            "criteria": criteria,
            "ready_for_advancement": current_progress >= required_consistency
        }
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """獲取學習統計信息"""
        stats = {
            'current_stage': self.current_stage.name,
            'current_episode': self.current_episode,
            'stage_episodes': self.stage_episodes,
            'advancement_criteria_status': self.get_stage_criteria_status(),
            'stage_performances': {}
        }
        
        for stage, history in self.stage_performance_history.items():
            if history:
                stats['stage_performances'][stage.name] = {
                    'episodes': len(history),
                    'avg_reward': np.mean(history),
                    'max_reward': np.max(history),
                    'min_reward': np.min(history),
                    'recent_avg': np.mean(history[-10:]) if len(history) >= 10 else np.mean(history),
                    'performance_trend': self._calculate_trend(history)
                }
            else:
                stats['stage_performances'][stage.name] = {
                    'episodes': 0,
                    'avg_reward': 0.0,
                    'max_reward': 0.0,
                    'min_reward': 0.0,
                    'recent_avg': 0.0,
                    'performance_trend': 0.0
                }
        
        return stats
    
    def _calculate_trend(self, history: List[float]) -> float:
        """計算性能趨勢"""
        if len(history) < 10:
            return 0.0
        
        # 使用線性回歸計算趨勢
        x = np.arange(len(history))
        y = np.array(history)
        
        # 計算斜率
        n = len(history)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        
        return slope
    
    def reset_system(self):
        """重置學習系統"""
        self.current_episode = 0
        self.stage_episodes = 0
        self.current_stage = LearningStage.BASIC
        self.force_advancement = False
        
        for stage in self.stage_performance_history:
            self.stage_performance_history[stage].clear()
        
        for stage in self.advancement_progress:
            self.advancement_progress[stage] = 0
            
        logger.info("漸進式學習系統已重置")
    
    def save_checkpoint(self, filepath: str):
        """保存學習進度檢查點"""
        checkpoint = {
            'current_episode': self.current_episode,
            'stage_episodes': self.stage_episodes,
            'current_stage': self.current_stage.value,
            'stage_performance_history': {
                stage.value: history for stage, history in self.stage_performance_history.items()
            },
            'advancement_progress': {
                stage.value: progress for stage, progress in self.advancement_progress.items()
            },
            'min_stage_episodes': self.min_stage_episodes,
            'performance_window': self.performance_window,
            'advancement_patience': self.advancement_patience
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"學習進度檢查點已保存: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """載入學習進度檢查點"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.current_episode = checkpoint['current_episode']
            self.stage_episodes = checkpoint['stage_episodes']
            self.current_stage = LearningStage(checkpoint['current_stage'])
            
            # 載入性能歷史
            for stage_value, history in checkpoint['stage_performance_history'].items():
                stage = LearningStage(stage_value)
                self.stage_performance_history[stage] = history
            
            # 載入進階進度
            for stage_value, progress in checkpoint['advancement_progress'].items():
                stage = LearningStage(stage_value)
                self.advancement_progress[stage] = progress
            
            logger.info(f"學習進度檢查點已載入: {filepath}")
            logger.info(f"當前階段: {self.current_stage.name}, 總回合: {self.current_episode}, 階段回合: {self.stage_episodes}")
            
        except Exception as e:
            logger.error(f"載入檢查點失敗: {e}")
            raise
    
    def get_reward_function_description(self) -> Dict[str, str]:
        """獲取當前獎勵函數的描述"""
        descriptions = {
            LearningStage.BASIC: {
                "focus": "基礎交易原理學習",
                "components": "基本盈虧(70%) + 風險控制(20%) + 交易頻率(10%)",
                "objectives": "學習買賣時機、風險意識、避免過度交易"
            },
            LearningStage.INTERMEDIATE: {
                "focus": "風險管理強化",
                "components": "基礎獎勵 + 夏普比率(25%) + 回撤控制(20%) + 勝率(15%)",
                "objectives": "風險調整收益、勝率平衡、穩定性追求"
            },
            LearningStage.ADVANCED: {
                "focus": "複雜策略掌握",
                "components": "中級獎勵 + Sortino比率 + VaR + 偏度 + 峰度 + 成本 + 一致性",
                "objectives": "複雜環境適應、多策略組合、策略創新"
            }
        }
        
        return descriptions[self.current_stage]
    
    def __repr__(self):
        stats = self.get_learning_statistics()
        advancement_status = stats['advancement_criteria_status']
        
        return (f"ProgressiveLearningSystem("
                f"stage={stats['current_stage']}, "
                f"episode={stats['current_episode']}, "
                f"stage_progress={advancement_status.get('progress_percentage', 0):.1f}%)")


def test_progressive_learning_system():
    """測試漸進式學習系統"""
    logger.info("🧪 開始測試漸進式學習系統...")
    
    # 創建學習系統
    learning_system = ProgressiveLearningSystem(
        min_stage_episodes=20,
        performance_window=10,
        advancement_patience=5
    )
    
    logger.info(f"初始化: {learning_system}")
    
    # 模擬學習過程
    test_episodes = 100
    stage_transitions = []
    
    for episode in range(test_episodes):
        # 模擬交易指標 (隨著回合增加，性能逐漸提升)
        stage_factor = learning_system.current_stage.value
        base_performance = 0.1 + (episode / test_episodes) * 0.6 + stage_factor * 0.1
        noise = np.random.normal(0, 0.1)
        
        # 生成不同階段適合的指標
        mock_metrics = {
            'pnl': base_performance + noise,
            'drawdown': max(0, 0.15 - episode * 0.001 + abs(noise) * 0.3),
            'trade_frequency': 0.05 + np.random.uniform(-0.02, 0.02),
            'sharpe_ratio': 0.3 + episode * 0.015 + noise * 0.5,
            'max_drawdown': max(0, 0.20 - episode * 0.002),
            'win_rate': 0.35 + episode * 0.003 + noise * 0.1,
            'sortino_ratio': 0.4 + episode * 0.01 + noise * 0.3,
            'var_risk': -0.05 - abs(noise) * 0.02,
            'skewness': noise * 0.5,
            'kurtosis': 3.0 + abs(noise),
            'transaction_costs': 0.01 + np.random.uniform(0, 0.005),
            'consistency_score': min(1.0, 0.3 + episode * 0.005 + noise * 0.2)
        }
        
        # 計算獎勵
        total_reward, components, learning_metrics = learning_system.calculate_reward(mock_metrics)
        
        # 記錄階段轉換
        if learning_metrics.should_advance:
            stage_transitions.append({
                'episode': episode,
                'from_stage': learning_metrics.stage.name,
                'reason': learning_metrics.advancement_reason
            })
            logger.info(f"🔄 回合 {episode}: 階段升級!")
            logger.info(f"   原因: {learning_metrics.advancement_reason}")
        
        # 每20個回合輸出一次進度
        if episode % 20 == 0:
            criteria_status = learning_system.get_stage_criteria_status()
            logger.info(f"回合 {episode}: 階段={learning_metrics.stage.name}, "
                       f"獎勵={total_reward:.3f}, "
                       f"進階進度={criteria_status.get('progress_percentage', 0):.1f}%")
    
    # 輸出最終統計
    final_stats = learning_system.get_learning_statistics()
    logger.info("\n=== 📊 學習統計 ===")
    
    for stage_name, perf in final_stats['stage_performances'].items():
        if perf['episodes'] > 0:
            logger.info(f"{stage_name}: {perf['episodes']}回合, "
                       f"平均獎勵={perf['avg_reward']:.3f}, "
                       f"趨勢={perf['performance_trend']:.4f}")
    
    # 輸出階段轉換歷史
    logger.info(f"\n=== 🎯 階段轉換歷史 ===")
    for transition in stage_transitions:
        logger.info(f"回合 {transition['episode']}: {transition['from_stage']} - {transition['reason']}")
    
    # 輸出當前獎勵函數描述
    reward_desc = learning_system.get_reward_function_description()
    logger.info(f"\n=== 🎮 當前獎勵函數 ===")
    logger.info(f"專注領域: {reward_desc['focus']}")
    logger.info(f"組件: {reward_desc['components']}")
    logger.info(f"目標: {reward_desc['objectives']}")
    
    logger.info(f"\n最終狀態: {learning_system}")
    logger.info("✅ 漸進式學習系統測試完成!")
    
    return learning_system


if __name__ == "__main__":
    # 運行測試
    test_system = test_progressive_learning_system()