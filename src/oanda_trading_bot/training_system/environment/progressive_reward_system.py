print("PROGRESSIVE REWARD SYSTEM MODULE LOADED", flush=True) # Top-level print
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
import pandas as pd

# Import market regime enums and identifier (assuming it will be passed or accessible)
from ..market_analysis.market_regime_identifier import VolatilityLevel, TrendStrength, MacroRegime, MarketRegimeIdentifier

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
        # 權重與單元測試 test_complex_reward_calculation (trade_info_missing) 的預期默認值對齊
        self.sortino_weight = self.config.get('sortino_weight', 0.3)
        self.profit_factor_weight = self.config.get('profit_factor_weight', 0.2)
        self.win_rate_weight = self.config.get('win_rate_weight', 0.1)
        self.market_adaptability_weight = self.config.get('market_adaptability_weight', 0.2) # 取消註釋並使用測試預期默認值
        self.consistency_weight = self.config.get('consistency_weight', 0.1) # 取消註釋並使用測試預期默認值
        self.max_drawdown_penalty_weight = self.config.get('max_drawdown_penalty_weight', 0.1)

        # New weights for market regime based rewards/penalties
        self.regime_multiplier_config = self.config.get('regime_multiplier_config', 
            {
                "default_multiplier": 1.0,
                "volatility": {
                    VolatilityLevel.HIGH.value: {"multiplier": 1.2, "pnl_threshold_factor": 0.5},
                    VolatilityLevel.LOW.value: {"multiplier": 0.8, "pnl_threshold_factor": 1.5}
                },
                "trend_strength": {
                    TrendStrength.STRONG_TREND.value: {"multiplier": 1.3, "pnl_threshold_factor": 0.7},
                    TrendStrength.NO_TREND.value: {"multiplier": 0.7, "pnl_threshold_factor": 1.3}
                },
                "macro_regime": {
                    MacroRegime.BULLISH.value: {"trend_bonus": 0.1, "ranging_penalty": 0.05},
                    MacroRegime.BEARISH.value: {"trend_bonus": 0.1, "ranging_penalty": 0.05},
                    MacroRegime.RANGING.value: {"ranging_bonus": 0.1, "trend_penalty": 0.05}
                }
            }
        )
        logger.info(f"ComplexReward initialized with various weights and regime multiplier config.")

    def _calculate_base_reward(self, trade_info: Dict[str, Any]) -> float:
        """Helper to calculate the non-regime part of the reward."""
        sortino_ratio = trade_info.get('sortino_ratio', 0.0)
        profit_factor = trade_info.get('profit_factor', 1.0) 
        win_rate = trade_info.get('win_rate', 0.5) 
        max_drawdown = trade_info.get('max_drawdown', 0.0)
        market_adaptability_score = trade_info.get('market_adaptability_score', 0.0) # 取消註釋
        behavioral_consistency_score = trade_info.get('behavioral_consistency_score', 0.0) # 取消註釋

        base_reward = (
            sortino_ratio * self.sortino_weight +
            (profit_factor - 1) * self.profit_factor_weight +
            (win_rate - 0.5) * self.win_rate_weight + # 確保與測試邏輯一致 (加法)
            market_adaptability_score * self.market_adaptability_weight + # 確保與測試邏輯一致 (加法)
            behavioral_consistency_score * self.consistency_weight - # 確保與測試邏輯一致 (加法後減去 MDD)
            abs(max_drawdown) * self.max_drawdown_penalty_weight
        )
        return base_reward

    def _apply_regime_modifiers(self, base_reward: float, trade_info: Dict[str, Any], market_data: Optional[Dict[str, Any]]) -> float:
        """Applies market regime based multipliers and bonuses/penalties."""
        if not market_data:
            return base_reward

        current_regime: Optional[Dict[str, Any]] = market_data.get('current_regime')
        if not current_regime:
            return base_reward

        pnl = trade_info.get('realized_pnl', trade_info.get('pnl', 0.0))
        modified_reward = base_reward
        multiplier = self.regime_multiplier_config.get("default_multiplier", 1.0)
        pnl_threshold_factor = 1.0 # Default, higher means harder to get bonus for positive PnL

        # Volatility modifier
        vol = current_regime.get('volatility_level')
        if vol and vol.value in self.regime_multiplier_config.get("volatility", {}):
            vol_config = self.regime_multiplier_config["volatility"][vol.value]
            multiplier *= vol_config.get("multiplier", 1.0)
            pnl_threshold_factor *= vol_config.get("pnl_threshold_factor", 1.0)
            logger.debug(f"Volatility {vol.value}: multiplier effect {vol_config.get('multiplier', 1.0)}, pnl_factor effect {vol_config.get('pnl_threshold_factor', 1.0)}")

        # Trend Strength modifier
        trend = current_regime.get('trend_strength')
        if trend and trend.value in self.regime_multiplier_config.get("trend_strength", {}):
            trend_config = self.regime_multiplier_config["trend_strength"][trend.value]
            multiplier *= trend_config.get("multiplier", 1.0)
            pnl_threshold_factor *= trend_config.get("pnl_threshold_factor", 1.0)
            logger.debug(f"Trend {trend.value}: multiplier effect {trend_config.get('multiplier', 1.0)}, pnl_factor effect {trend_config.get('pnl_threshold_factor', 1.0)}")

        # Apply multiplier: if PnL is positive and above a dynamic threshold, apply full multiplier.
        # If PnL is negative, the multiplier might amplify penalty (or be capped).
        # This is a simple heuristic, can be made more sophisticated.
        dynamic_pnl_threshold = trade_info.get('average_trade_pnl', 0) * pnl_threshold_factor 
        if pnl > dynamic_pnl_threshold: # Only apply full multiplier bonus if PnL is significantly positive
            modified_reward *= multiplier
        elif pnl < 0: # For negative PnL, ensure multiplier doesn't excessively reduce penalty (or can amplify it)
            modified_reward *= max(1.0, multiplier) # Example: if multiplier is <1, don't reduce penalty; if >1, amplify.
        else: # Neutral or slightly positive PnL, less impact from multiplier
            modified_reward *= (1.0 + multiplier) / 2 # Average effect

        logger.debug(f"After PnL-sensitive multiplier ({multiplier:.2f}, pnl_thresh_factor: {pnl_threshold_factor:.2f}, dyn_thresh: {dynamic_pnl_threshold:.2f}): {modified_reward:.4f}")

        # Macro Regime bonus/penalty (additive)
        macro = current_regime.get('macro_regime')
        if macro and macro.value in self.regime_multiplier_config.get("macro_regime", {}):
            macro_config = self.regime_multiplier_config["macro_regime"][macro.value]
            if pnl > 0: # Bonuses for profitable trades in favorable regimes
                if (macro == MacroRegime.BULLISH or macro == MacroRegime.BEARISH) and trend == TrendStrength.STRONG_TREND:
                    modified_reward += macro_config.get("trend_bonus", 0)
                    logger.debug(f"Macro {macro.value} with Strong Trend: adding trend_bonus {macro_config.get('trend_bonus', 0)}")
                elif macro == MacroRegime.RANGING and trend == TrendStrength.NO_TREND:
                    modified_reward += macro_config.get("ranging_bonus", 0)
                    logger.debug(f"Macro {macro.value} with No Trend: adding ranging_bonus {macro_config.get('ranging_bonus', 0)}")
            elif pnl < 0: # Penalties for losing trades in unfavorable regime mismatches
                if (macro == MacroRegime.BULLISH or macro == MacroRegime.BEARISH) and trend == TrendStrength.NO_TREND:
                    modified_reward -= macro_config.get("ranging_penalty", 0) # Penalize trying to trend in ranging
                    logger.debug(f"Macro {macro.value} with No Trend (loss): applying ranging_penalty {macro_config.get('ranging_penalty', 0)}")
                elif macro == MacroRegime.RANGING and trend == TrendStrength.STRONG_TREND:
                    modified_reward -= macro_config.get("trend_penalty", 0) # Penalize trying to range in trend
                    logger.debug(f"Macro {macro.value} with Strong Trend (loss): applying trend_penalty {macro_config.get('trend_penalty', 0)}")
        
        logger.debug(f"Final reward after regime modifiers: {modified_reward:.4f}")
        return modified_reward

    def calculate_reward(self, trade_info: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> float:
        """計算高複雜度獎勵，包含市場狀態調整"""
        
        base_reward = self._calculate_base_reward(trade_info)
        logger.debug(f"ComplexReward Base: {base_reward:.4f}")
        
        final_reward = self._apply_regime_modifiers(base_reward, trade_info, market_data)
        
        # logger.debug(f"ComplexReward: Base={base_reward:.4f}, Final={final_reward:.4f}, PnL={trade_info.get('realized_pnl', 0.0)}, MarketData provided: {market_data is not None}")
        return final_reward

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
    logger.info("\\n--- Testing Individual Reward Strategies ---")
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

    complex_reward_strategy = ComplexReward() # Uses default config
    trade_info_3 = {
        'sortino_ratio': 2.0, 'profit_factor': 1.8, 'win_rate': 0.6,
        'max_drawdown': 25, 
        'market_adaptability_score': 0.7, 
        'behavioral_consistency_score': 0.6 
    }
    reward3 = complex_reward_strategy.calculate_reward(trade_info_3)
    # 更新 expected_reward3 以反映 ComplexReward 的更改和新的默認權重
    # New default weights: 
    # sortino_weight = 0.3
    # profit_factor_weight = 0.2
    # win_rate_weight = 0.1
    # market_adaptability_weight = 0.2
    # consistency_weight = 0.1
    # max_drawdown_penalty_weight = 0.1
    
    expected_reward3 = (
        trade_info_3['sortino_ratio'] * complex_reward_strategy.sortino_weight +           # 2.0 * 0.3 = 0.6
        (trade_info_3['profit_factor'] - 1) * complex_reward_strategy.profit_factor_weight + # (1.8-1)*0.2 = 0.8*0.2 = 0.16
        (trade_info_3['win_rate'] - 0.5) * complex_reward_strategy.win_rate_weight +       # (0.6-0.5)*0.1 = 0.1*0.1 = 0.01
        trade_info_3['market_adaptability_score'] * complex_reward_strategy.market_adaptability_weight + # 0.7 * 0.2 = 0.14
        trade_info_3['behavioral_consistency_score'] * complex_reward_strategy.consistency_weight - # 0.6 * 0.1 = 0.06
        abs(trade_info_3['max_drawdown']) * complex_reward_strategy.max_drawdown_penalty_weight # 25 * 0.1 = 2.5
    )
    # Calculation: 0.6 + 0.16 + 0.01 + 0.14 + 0.06 - 2.5 = 0.97 - 2.5 = -1.53

    logger.info(f"ComplexReward for {trade_info_3} (no market data): {reward3:.4f} (Expected: {expected_reward3:.4f})")
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

    logger.info("\n--- Testing ComplexReward with MarketRegimeIdentifier Integration ---")
    # Setup MarketRegimeIdentifier
    # Assuming MarketRegimeIdentifier and its enums are imported at the top of the file
    # from oanda_trading_bot.training_system.market_analysis.market_regime_identifier import MarketRegimeIdentifier, VolatilityLevel, TrendStrength, MacroRegime
    
    mri_config = {
        "atr_period": 14,
        "atr_resample_freq": "1h",
        "atr_thresholds": {"low_to_medium": 0.002, "medium_to_high": 0.005},
        "adx_period": 14,
        "adx_resample_freq": "4h",
        "adx_thresholds": {"no_to_weak": 20, "weak_to_strong": 25}
    }
    try:
        market_regime_identifier = MarketRegimeIdentifier(config=mri_config)
    except Exception as e:
        logger.error(f"Failed to initialize MarketRegimeIdentifier for testing: {e}")
        market_regime_identifier = None

    # Sample S5 data (minimal, just to get some regime output)
    # For more realistic regime outputs, use more extensive data as in test_market_analysis.py
    periods = 2 * 24 * 60 * 12  # 2 days of 5s data
    rng = pd.date_range('2023-01-01', periods=periods, freq='5s')
    s5_data_for_regime = pd.DataFrame({
        'open': np.random.rand(periods) * 10 + 100,
        'high': np.random.rand(periods) * 10 + 100.5,
        'low': np.random.rand(periods) * 10 + 99.5,
        'close': np.random.rand(periods) * 10 + 100,
        'volume': np.random.randint(50, 200, periods)
    }, index=rng)
    s5_data_for_regime['high'] = s5_data_for_regime[['open', 'high', 'low', 'close']].max(axis=1)
    s5_data_for_regime['low'] = s5_data_for_regime[['open', 'high', 'low', 'close']].min(axis=1)

    current_regime_output = None
    if market_regime_identifier:
        try:
            current_regime_output = market_regime_identifier.get_current_regime(s5_data_for_regime)
            logger.info(f"Sample Current Regime for testing: {current_regime_output}")
        except Exception as e:
            logger.error(f"Error getting current regime for testing: {e}")
            current_regime_output = { # Fallback if MRI fails
                "macro_regime": MacroRegime.RANGING,
                "volatility_level": VolatilityLevel.MEDIUM,
                "trend_strength": TrendStrength.NO_TREND
            }
    else:
        current_regime_output = { # Fallback if MRI init fails
            "macro_regime": MacroRegime.RANGING,
            "volatility_level": VolatilityLevel.MEDIUM,
            "trend_strength": TrendStrength.NO_TREND
        }
        logger.warning("MarketRegimeIdentifier not available, using fallback regime for testing ComplexReward.")

    market_data_for_reward = {"current_regime": current_regime_output}

    # Define a ComplexReward instance with a specific or default regime_multiplier_config
    # Using default config as defined in ComplexReward class for this test
    complex_reward_strategy_for_regime_test = ComplexReward(config={})
    # Or define a custom one:
    # custom_regime_config = complex_reward_strategy_for_regime_test.regime_multiplier_config.copy()
    # custom_regime_config["volatility"][VolatilityLevel.HIGH.value]["multiplier"] = 1.5 
    # complex_reward_strategy_for_regime_test = ComplexReward(config={"regime_multiplier_config": custom_regime_config})

    test_scenarios = [
        {"name": "Positive PnL, Avg Trade PnL provided", "trade_info": {'realized_pnl': 100, 'average_trade_pnl': 50, 'sortino_ratio': 1.5, 'profit_factor': 2.0, 'win_rate': 0.6, 'max_drawdown': 10}},
        {"name": "Negative PnL, Avg Trade PnL provided", "trade_info": {'realized_pnl': -80, 'average_trade_pnl': 50, 'sortino_ratio': -0.5, 'profit_factor': 0.5, 'win_rate': 0.3, 'max_drawdown': 15}},
        {"name": "Small Positive PnL, below dynamic threshold", "trade_info": {'realized_pnl': 20, 'average_trade_pnl': 50, 'sortino_ratio': 0.8, 'profit_factor': 1.2, 'win_rate': 0.55, 'max_drawdown': 5}},
        {"name": "Positive PnL, NO Avg Trade PnL (threshold defaults to 0)", "trade_info": {'realized_pnl': 100, 'sortino_ratio': 1.5, 'profit_factor': 2.0, 'win_rate': 0.6, 'max_drawdown': 10}},
        {"name": "No Market Data (should use base reward only)", "trade_info": {'realized_pnl': 100, 'sortino_ratio': 1.5, 'profit_factor': 2.0, 'win_rate': 0.6, 'max_drawdown': 10}, "market_data": None}
    ]

    for scenario in test_scenarios:
        logger.info(f"\n--- Testing Scenario: {scenario['name']} ---")
        trade_info = scenario["trade_info"]
        market_data_input = scenario.get("market_data", market_data_for_reward) # Use default market_data_for_reward unless None is specified
        
        logger.info(f"Trade Info: {trade_info}")
        if market_data_input:
            logger.info(f"Market Data (Regime): {market_data_input['current_regime']}")
        else:
            logger.info("Market Data: None (testing base reward path)")

        # Calculate base reward for comparison
        base_reward = complex_reward_strategy_for_regime_test._calculate_base_reward(trade_info)
        logger.info(f"Calculated Base Reward: {base_reward:.4f}")

        # Calculate final reward with potential regime modification
        final_reward = complex_reward_strategy_for_regime_test.calculate_reward(trade_info, market_data_input)
        logger.info(f"Calculated Final Reward: {final_reward:.4f}")
        
        if market_data_input and market_data_input.get('current_regime'):
            if final_reward > base_reward:
                logger.info("Regime MODIFIER: Positive (reward increased or penalty reduced)")
            elif final_reward < base_reward:
                logger.info("Regime MODIFIER: Negative (reward reduced or penalty increased)")
            else:
                logger.info("Regime MODIFIER: Neutral (no change or effects cancelled out)")
        elif not market_data_input:
             assert abs(final_reward - base_reward) < 1e-6, "Final reward should be base reward if no market data"
             logger.info("Confirmed: Final reward equals base reward as no market data was provided.")

    logger.info("----- ComplexReward with MarketRegimeIdentifier Integration Tests Finished -----")

    logger.info("----- Progressive Reward System Tests Passed (including integration checks) -----")