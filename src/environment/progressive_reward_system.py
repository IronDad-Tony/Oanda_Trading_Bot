# src/environment/progressive_reward_system.py
"""
漸進式獎勵系統 (Progressive Reward System)

此模組定義了用於強化學習環境的漸進式獎勵機制。
隨著智能體學習的進展，獎勵函數的複雜性會逐漸增加，
引導智能體從學習簡單的目標（如基本盈虧）到更複雜的目標（如風險調整後的收益、多維度優化）。
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional
import logging

# 配置日誌
logger = logging.getLogger(__name__)

class BaseRewardStrategy(ABC):
    """獎勵策略基類"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config is not None else {}
        logger.info(f"{self.__class__.__name__} initialized with config: {self.config}")

    @abstractmethod
    def calculate_reward(self, trade_info: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> float:
        """
        計算獎勵。

        Args:
            trade_info (Dict[str, Any]): 包含交易相關信息的字典，例如：
                - 'pnl': 盈虧 (Profit and Loss)
                - 'realized_pnl': 已實現盈虧
                - 'unrealized_pnl': 未實現盈虧
                - 'position_duration': 持倉時間 (單位：秒、分鐘、小時或天)
                - 'drawdown': 當前回撤
                - 'max_drawdown': 最大回撤
                - 'sharpe_ratio': 夏普比率 (可能是階段性的)
                - 'sortino_ratio': 索提諾比率 (可能是階段性的)
                - 'trade_cost': 交易成本
                - 'num_trades': 交易次數
                - 'winning_trades': 盈利交易次數
                - 'losing_trades': 虧損交易次數
                # ... 其他可能的交易相關指標
            market_data (Optional[Dict[str, Any]]): 包含市場相關信息的字典，例如：
                - 'volatility': 市場波動率
                - 'trend_strength': 趨勢強度
                - 'market_regime': 市場狀態 (如趨勢、震盪)
                # ... 其他可能的市場相關指標

        Returns:
            float: 計算得到的獎勵值。
        """
        pass

class SimpleReward(BaseRewardStrategy):
    """
    階段1：簡單獎勵（基本盈虧與風險懲罰）
    獎勵 = 實現盈虧 * profit_weight - 風險懲罰 * risk_weight
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.profit_weight = self.config.get('profit_weight', 0.8) # 盈虧權重
        self.risk_penalty_weight = self.config.get('risk_penalty_weight', 0.2) # 風險懲罰權重
        self.risk_metric = self.config.get('risk_metric', 'drawdown') # 用於風險懲罰的指標
        logger.info(f"SimpleReward initialized with profit_weight={self.profit_weight}, risk_penalty_weight={self.risk_penalty_weight}, risk_metric='{self.risk_metric}'")

    def calculate_reward(self, trade_info: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> float:
        """計算簡單獎勵"""
        realized_pnl = trade_info.get('realized_pnl', 0.0)
        
        risk_value = 0.0
        if self.risk_metric == 'drawdown':
            risk_value = trade_info.get('drawdown', 0.0)
        elif self.risk_metric == 'max_drawdown': # 可以考慮使用期間最大回撤
            risk_value = trade_info.get('max_drawdown', 0.0)
        # 可以擴展其他風險指標

        reward = realized_pnl * self.profit_weight - abs(risk_value) * self.risk_penalty_weight
        # logger.debug(f"SimpleReward: PnL={realized_pnl}, RiskValue ({self.risk_metric})={risk_value}, Reward={reward}")
        return reward

class IntermediateReward(BaseRewardStrategy):
    """
    階段2：中等複雜度獎勵（風險調整後的收益）
    包含夏普比率、回撤懲罰、交易成本等。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # 示例權重，可通過配置調整
        self.sharpe_weight = self.config.get('sharpe_weight', 0.5)
        self.pnl_weight = self.config.get('pnl_weight', 0.3)
        self.drawdown_penalty_weight = self.config.get('drawdown_penalty_weight', 0.15)
        self.cost_penalty_weight = self.config.get('cost_penalty_weight', 0.05)
        logger.info(f"IntermediateReward initialized with weights: sharpe={self.sharpe_weight}, pnl={self.pnl_weight}, drawdown={self.drawdown_penalty_weight}, cost={self.cost_penalty_weight}")

    def calculate_reward(self, trade_info: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> float:
        """計算中等複雜度獎勵"""
        sharpe_ratio = trade_info.get('sharpe_ratio', 0.0)
        realized_pnl = trade_info.get('realized_pnl', 0.0)
        drawdown = trade_info.get('drawdown', 0.0) # 通常是正值表示回撤幅度
        trade_cost = trade_info.get('trade_cost', 0.0)

        # 基礎獎勵組件
        reward_sharpe = sharpe_ratio * self.sharpe_weight
        reward_pnl = realized_pnl * self.pnl_weight
        
        # 懲罰項 (通常為負貢獻或減項)
        penalty_drawdown = abs(drawdown) * self.drawdown_penalty_weight
        penalty_cost = abs(trade_cost) * self.cost_penalty_weight

        total_reward = reward_sharpe + reward_pnl - penalty_drawdown - penalty_cost
        # logger.debug(f"IntermediateReward: Sharpe={sharpe_ratio}, PnL={realized_pnl}, Drawdown={drawdown}, Cost={trade_cost}, TotalReward={total_reward}")
        return total_reward

class ComplexReward(BaseRewardStrategy):
    """
    階段3：高複雜度獎勵（多維度優化）
    考慮更多市場狀態、交易行為一致性、目標達成度等。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # 示例權重，可通過配置調整
        self.sortino_weight = self.config.get('sortino_weight', 0.3)
        self.profit_factor_weight = self.config.get('profit_factor_weight', 0.2)
        self.win_rate_weight = self.config.get('win_rate_weight', 0.1)
        self.market_adaptability_weight = self.config.get('market_adaptability_weight', 0.2) # 需外部評估
        self.consistency_weight = self.config.get('consistency_weight', 0.1) # 需外部評估
        self.max_drawdown_penalty_weight = self.config.get('max_drawdown_penalty_weight', 0.1)
        logger.info(f"ComplexReward initialized with various weights including Sortino, ProfitFactor, etc.")

    def calculate_reward(self, trade_info: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> float:
        """計算高複雜度獎勵"""
        sortino_ratio = trade_info.get('sortino_ratio', 0.0)
        profit_factor = trade_info.get('profit_factor', 1.0) # 總盈利 / 總虧損，避免除零
        win_rate = trade_info.get('win_rate', 0.5) # 盈利交易次數 / 總交易次數
        max_drawdown = trade_info.get('max_drawdown', 0.0)

        # 模擬的外部評估指標
        market_adaptability_score = trade_info.get('market_adaptability_score', 0.0) # 假設由元學習或分析模組提供
        behavioral_consistency_score = trade_info.get('behavioral_consistency_score', 0.0) # 假設由元學習或分析模組提供

        reward = (
            sortino_ratio * self.sortino_weight +
            (profit_factor - 1) * self.profit_factor_weight + # profit_factor 大於1才有正貢獻
            (win_rate - 0.5) * self.win_rate_weight + # win_rate 大於0.5才有正貢獻
            market_adaptability_score * self.market_adaptability_weight +
            behavioral_consistency_score * self.consistency_weight -
            abs(max_drawdown) * self.max_drawdown_penalty_weight
        )
        # logger.debug(f"ComplexReward: Sortino={sortino_ratio}, ProfitFactor={profit_factor}, WinRate={win_rate}, MaxDrawdown={max_drawdown}, Adaptability={market_adaptability_score}, Consistency={behavioral_consistency_score}, TotalReward={reward}")
        return reward

class ProgressiveLearningSystem:
    """
    漸進式學習系統

    管理不同的獎勵階段，並根據預設的標準在它們之間轉換。
    """
    def __init__(self, stage_configs: Dict[int, Dict[str, Any]], initial_stage: int = 1):
        """
        初始化漸進式學習系統。

        Args:
            stage_configs (Dict[int, Dict[str, Any]]): 一個字典，鍵是階段編號 (從1開始)，
                值是該階段的配置字典。每個階段配置應包含：
                - 'reward_strategy_class': 獎勵策略類 (例如 SimpleReward, IntermediateReward)。
                - 'reward_config': 傳遞給獎勵策略構造函數的配置。
                - 'criteria_to_advance': (Callable[Dict[str, Any], bool], optional) 一個函數，
                    輸入為包含訓練統計數據的字典，返回布爾值指示是否應進入下一階段。
                    如果為 None，則階段轉換需要手動調用 advance_stage()。
                - 'max_episodes_or_steps': (int, optional) 在此階段允許的最大回合數或步數，
                    達到後如果 criteria_to_advance 未滿足，可以選擇停留或觸發其他 logique。
            initial_stage (int): 初始學習階段。
        """
        self.stage_configs = stage_configs
        self.current_stage_number = initial_stage
        self.current_reward_strategy: BaseRewardStrategy = None
        self.episode_in_current_stage = 0
        self.steps_in_current_stage = 0

        if not self.stage_configs:
            raise ValueError("Stage configurations cannot be empty.")
        
        self._setup_current_stage()
        logger.info(f"ProgressiveLearningSystem initialized. Starting at stage {self.current_stage_number} with {self.current_reward_strategy.__class__.__name__}.")

    def _setup_current_stage(self):
        """根據當前階段編號設置獎勵策略。"""
        if self.current_stage_number not in self.stage_configs:
            logger.error(f"Stage {self.current_stage_number} not defined in stage_configs. Available stages: {list(self.stage_configs.keys())}")
            # 可以選擇回退到最後一個有效階段或引發更嚴重的錯誤
            # 這裡我們嘗試使用最低編號的可用階段
            if not self.stage_configs: # Should be caught by init, but defensive
                raise ValueError("No stages configured.")
            self.current_stage_number = min(self.stage_configs.keys())
            logger.warning(f"Falling back to stage {self.current_stage_number}.")
        
        config = self.stage_configs[self.current_stage_number]
        reward_class = config.get('reward_strategy_class')
        reward_specific_config = config.get('reward_config', {})

        if not reward_class or not issubclass(reward_class, BaseRewardStrategy):
            raise ValueError(f"Invalid reward_strategy_class for stage {self.current_stage_number}: {reward_class}")

        self.current_reward_strategy = reward_class(config=reward_specific_config)
        self.episode_in_current_stage = 0 # 重置計數器
        self.steps_in_current_stage = 0
        logger.info(f"Transitioned to stage {self.current_stage_number}: {self.current_reward_strategy.__class__.__name__}")

    def get_current_reward_function(self) -> BaseRewardStrategy:
        """返回當前階段的獎勵策略實例."""
        return self.current_reward_strategy

    def calculate_reward(self, trade_info: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> float:
        """使用當前階段的獎勵策略計算獎勵."""
        if not self.current_reward_strategy:
            logger.error("No current reward strategy set. Cannot calculate reward.")
            return 0.0
        return self.current_reward_strategy.calculate_reward(trade_info, market_data)

    def record_step(self, training_stats: Optional[Dict[str, Any]] = None):
        """記錄一個訓練步驟，並檢查是否需要晉級。"""
        self.steps_in_current_stage += 1
        self.check_and_advance_stage(training_stats)

    def record_episode_end(self, training_stats: Optional[Dict[str, Any]] = None):
        """記錄一個回合結束，並檢查是否需要晉級。"""
        self.episode_in_current_stage += 1
        self.check_and_advance_stage(training_stats)

    def check_and_advance_stage(self, training_stats: Optional[Dict[str, Any]] = None):
        """
        檢查是否滿足進入下一階段的標準，如果滿足則晉級。

        Args:
            training_stats (Optional[Dict[str, Any]]): 當前的訓練統計數據，
                用於評估 criteria_to_advance。
        """
        current_config = self.stage_configs.get(self.current_stage_number, {})
        
        # 檢查是否達到最大回合數/步數限制
        max_duration = current_config.get('max_episodes_or_steps')
        duration_metric = 'episodes' # 或 'steps', 可配置
        current_duration = self.episode_in_current_stage if duration_metric == 'episodes' else self.steps_in_current_stage

        criteria_fn: Optional[Callable[[Dict[str, Any]], bool]] = current_config.get('criteria_to_advance')
        
        can_advance = False
        if criteria_fn and training_stats is not None:
            try:
                if criteria_fn(training_stats):
                    can_advance = True
                    logger.info(f"Stage {self.current_stage_number} advancement criteria met based on training_stats.")
            except Exception as e:
                logger.error(f"Error evaluating criteria_to_advance for stage {self.current_stage_number}: {e}")
        
        # 如果達到最大時長但標準未滿足，可以選擇停留或強制晉級 (取決於設計)
        if max_duration is not None and current_duration >= max_duration and not can_advance:
            logger.warning(f"Stage {self.current_stage_number} reached max duration ({max_duration} {duration_metric}) but criteria not met. Staying in current stage by default.")
            # 可在此處添加邏輯，例如：如果配置了 force_advance_on_max_duration=True，則 can_advance = True

        if can_advance:
            next_stage_number = self.current_stage_number + 1
            if next_stage_number in self.stage_configs:
                self.current_stage_number = next_stage_number
                self._setup_current_stage()
            else:
                logger.info(f"Already at the final stage ({self.current_stage_number}). No further advancement.")
        # else: logger.debug(f"Staying in stage {self.current_stage_number}. Criteria not met or no stats provided.")

    def advance_stage_manually(self) -> bool:
        """手動強制進入下一個階段."""
        next_stage_number = self.current_stage_number + 1
        if next_stage_number in self.stage_configs:
            self.current_stage_number = next_stage_number
            self._setup_current_stage()
            logger.info(f"Manually advanced to stage {self.current_stage_number}.")
            return True
        else:
            logger.warning(f"Cannot manually advance. Already at the final stage ({self.current_stage_number}) or next stage not configured.")
            return False

# 示例用法和測試
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger.info("----- Testing Progressive Reward System -----")

    # 1. 測試各個獎勵策略
    logger.info("\n--- Testing Individual Reward Strategies ---")
    simple_reward_config = {'profit_weight': 0.7, 'risk_penalty_weight': 0.3, 'risk_metric': 'drawdown'}
    simple_reward_strategy = SimpleReward(config=simple_reward_config)
    trade_info_1 = {'realized_pnl': 100, 'drawdown': 10}
    reward1 = simple_reward_strategy.calculate_reward(trade_info_1)
    logger.info(f"SimpleReward for {trade_info_1}: {reward1} (Expected: 100*0.7 - 10*0.3 = 67.0)")
    assert abs(reward1 - (100 * 0.7 - 10 * 0.3)) < 1e-6

    intermediate_reward_config = {
        'sharpe_weight': 0.4, 'pnl_weight': 0.3, 
        'drawdown_penalty_weight': 0.2, 'cost_penalty_weight': 0.1
    }
    intermediate_reward_strategy = IntermediateReward(config=intermediate_reward_config)
    trade_info_2 = {'realized_pnl': 150, 'sharpe_ratio': 1.5, 'drawdown': 20, 'trade_cost': 5}
    reward2 = intermediate_reward_strategy.calculate_reward(trade_info_2)
    expected_reward2 = 1.5*0.4 + 150*0.3 - 20*0.2 - 5*0.1 # 0.6 + 45 - 4 - 0.5 = 41.1
    logger.info(f"IntermediateReward for {trade_info_2}: {reward2} (Expected: {expected_reward2})")
    assert abs(reward2 - expected_reward2) < 1e-6

    complex_reward_strategy = ComplexReward()
    trade_info_3 = {
        'sortino_ratio': 2.0, 'profit_factor': 1.8, 'win_rate': 0.6,
        'max_drawdown': 25, 'market_adaptability_score': 0.7, 'behavioral_consistency_score': 0.6
    }
    reward3 = complex_reward_strategy.calculate_reward(trade_info_3)
    # Default weights: sortino=0.3, pf=0.2, wr=0.1, adapt=0.2, consist=0.1, mdd_penalty=0.1
    expected_reward3 = (2.0*0.3) + (1.8-1)*0.2 + (0.6-0.5)*0.1 + (0.7*0.2) + (0.6*0.1) - (25*0.1)
    # 0.6 + 0.8*0.2 + 0.1*0.1 + 0.14 + 0.06 - 2.5
    # 0.6 + 0.16 + 0.01 + 0.14 + 0.06 - 2.5 = 0.97 - 2.5 = -1.53
    logger.info(f"ComplexReward for {trade_info_3}: {reward3} (Expected: {expected_reward3})")
    assert abs(reward3 - expected_reward3) < 1e-6

    # 2. 測試 ProgressiveLearningSystem
    logger.info("\n--- Testing ProgressiveLearningSystem ---")
    
    # 定義晉級標準函數
    def stage1_criteria(stats: Dict[str, Any]) -> bool:
        return stats.get('avg_sharpe_ratio', 0) > 0.5 and stats.get('episodes_completed', 0) >= 10

    def stage2_criteria(stats: Dict[str, Any]) -> bool:
        return stats.get('avg_sortino_ratio', 0) > 0.8 and stats.get('stability_score', 0) > 0.7

    pls_configs = {
        1: {
            'reward_strategy_class': SimpleReward,
            'reward_config': simple_reward_config,
            'criteria_to_advance': stage1_criteria,
            'max_episodes_or_steps': 50
        },
        2: {
            'reward_strategy_class': IntermediateReward,
            'reward_config': intermediate_reward_config,
            'criteria_to_advance': stage2_criteria,
            'max_episodes_or_steps': 100
        },
        3: {
            'reward_strategy_class': ComplexReward,
            'reward_config': {},
            'criteria_to_advance': None, # 最後階段，無自動晉級
            'max_episodes_or_steps': None
        }
    }

    pls = ProgressiveLearningSystem(stage_configs=pls_configs, initial_stage=1)
    assert isinstance(pls.get_current_reward_function(), SimpleReward)
    logger.info(f"Initial stage: {pls.current_stage_number}, Reward type: {pls.get_current_reward_function().__class__.__name__}")

    # 模擬訓練過程
    mock_training_stats_stage1_fail = {'avg_sharpe_ratio': 0.2, 'episodes_completed': 5}
    pls.record_episode_end(mock_training_stats_stage1_fail)
    assert pls.current_stage_number == 1
    logger.info(f"After 5 episodes (fail criteria): Stage {pls.current_stage_number}")

    mock_training_stats_stage1_pass = {'avg_sharpe_ratio': 0.6, 'episodes_completed': 10}
    pls.record_episode_end(mock_training_stats_stage1_pass) # 應該晉級
    assert pls.current_stage_number == 2
    assert isinstance(pls.get_current_reward_function(), IntermediateReward)
    logger.info(f"After 10 episodes (pass criteria): Stage {pls.current_stage_number}, Reward type: {pls.get_current_reward_function().__class__.__name__}")

    # 測試手動晉級
    can_manual_advance = pls.advance_stage_manually() # 應晉級到第3階段
    assert can_manual_advance
    assert pls.current_stage_number == 3
    assert isinstance(pls.get_current_reward_function(), ComplexReward)
    logger.info(f"After manual advance: Stage {pls.current_stage_number}, Reward type: {pls.get_current_reward_function().__class__.__name__}")
    
    # 測試在最後階段手動晉級
    can_manual_advance_again = pls.advance_stage_manually() # 無法再晉級
    assert not can_manual_advance_again
    assert pls.current_stage_number == 3
    logger.info(f"Attempting manual advance from final stage: Stage {pls.current_stage_number}")

    # 測試計算獎勵
    current_reward_fn = pls.get_current_reward_function()
    reward_from_pls = pls.calculate_reward(trade_info_3) # 現在是 ComplexReward
    logger.info(f"Reward from PLS (Complex): {reward_from_pls} (Expected: {expected_reward3})")
    assert abs(reward_from_pls - expected_reward3) < 1e-6

    logger.info("----- Progressive Reward System Tests Passed -----")