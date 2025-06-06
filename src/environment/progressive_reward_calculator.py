"""
三階段漸進式獎勵計算器
設計理念：讓模型從基礎交易概念逐步演進到複雜的風險管理和收益優化
"""

from decimal import Decimal
from typing import Dict, List, Any, Optional
from collections import deque
import numpy as np
import logging
from .adaptive_reward_optimizer import AdaptiveRewardOptimizer
from .reward_normalizer import RewardNormalizer

logger = logging.getLogger(__name__)

class ProgressiveRewardCalculator:
    """
    三階段漸進式獎勵計算器
    
    階段1: 初始訓練階段 - 平均每筆交易期望值 < 0
    階段2: 進階訓練階段 - 平均每筆交易期望值 > 0，勝率 < 50%
    階段3: 最終訓練階段 - 平均每筆交易期望值 > 0，勝率 > 50%
    """
    
    def __init__(self, initial_capital: Decimal, config: Optional[Dict[str, Any]] = None):
        self.initial_capital = initial_capital
        
        # 階段判斷參數
        self.observation_window = 100  # 觀察窗口：過去100步
        self.stage_evaluation_interval = 50  # 每50步評估一次階段
        self.current_stage = 1  # 初始階段
        
        # 交易表現追蹤
        self.trade_history = deque(maxlen=self.observation_window)
        self.returns_history = deque(maxlen=self.observation_window)
        self.reward_components_history: List[Dict[str, Any]] = []
        
        # 新增：自適應獎勵優化器
        self.adaptive_optimizer = AdaptiveRewardOptimizer(config)
        
        # 新增：性能監控
        self.performance_metrics = {
            'episode_rewards': deque(maxlen=1000),
            'portfolio_values': deque(maxlen=1000),
            'sharpe_ratios': deque(maxlen=100),
            'max_drawdowns': deque(maxlen=100),
            'profit_factors': deque(maxlen=100)
        }
        
        # 階段特定參數
        self.stage_configs = {
            1: self._get_stage1_config(),  # 初始訓練階段
            2: self._get_stage2_config(),  # 進階訓練階段  
            3: self._get_stage3_config(),  # 最終訓練階段
        }
        
        # 全局設置        self.risk_free_rate = Decimal('0.02') / Decimal('252')  # 年化2%無風險利率
        self.last_portfolio_value = initial_capital
        self.step_count = 0
        self.stage_switch_cooldown = 0  # 防止頻繁切換階段
        
        # 新增：獎勵標準化器
        self.reward_normalizer = RewardNormalizer(
            target_range=(-100.0, 100.0),
            history_window=1000,
            adaptive_scaling=True
        )
        
        # 用戶自訂配置覆蓋
        if config:
            for stage in [1, 2, 3]:
                if f'stage_{stage}_config' in config:
                    # 確保所有配置值都轉換為 Decimal 類型
                    user_config = config[f'stage_{stage}_config']
                    for key, value in user_config.items():
                        if key in self.stage_configs[stage]:
                            if isinstance(value, (int, float, str)):
                                self.stage_configs[stage][key] = Decimal(str(value))
                            elif isinstance(value, Decimal):
                                self.stage_configs[stage][key] = value
    
    def _get_stage1_config(self) -> Dict[str, Decimal]:
        """
        階段1：初始訓練階段配置
        目標：讓模型快速學會基礎交易概念，鼓勵多交易多學習
        重點：多嘗試、多交易、理解獲利與虧損的基本原理
        
        教學理念：
        1. 正向強化 > 負向懲罰 (鼓勵實驗精神)
        2. 頻繁交易 = 快速學習 (增加訓練密度)
        3. 多樣性探索 = 概念理解 (避免過早收斂)
        4. 即時反饋 = 明確學習信號 (加速概念形成)
        """
        return {
            # 基礎獎勵權重（強化學習導向）- 比例 3:1 (獎勵:懲罰)
            "profit_reward_factor": Decimal('4.0'),        # 獲利獎勵係數（進一步加強正向反饋）
            "loss_penalty_factor": Decimal('0.7'),         # 虧損懲罰係數（再次降低，鼓勵大膽嘗試）
            "trade_frequency_bonus": Decimal('0.2'),       # 交易頻率獎勵（再次增強）
            "exploration_bonus": Decimal('0.8'),           # 探索獎勵（大幅增強，鼓勵多樣性）
            
            # 鼓勵交易學習（教學強化版）
            "min_trades_per_episode": 6,                   # 每集最少交易次數（進一步降低門檻）
            "trade_diversity_bonus": Decimal('0.4'),       # 交易多樣性獎勵（強化）
            "quick_decision_bonus": Decimal('0.2'),        # 快速決策獎勵（增強即時反應）
            "learning_progress_bonus": Decimal('0.5'),     # 學習進步獎勵（提高重要性）
            "concept_mastery_bonus": Decimal('0.35'),      # 概念掌握獎勵（加強基礎理解）
            
            # 基礎風險控制（寬鬆且教學導向）
            "max_single_loss_ratio": Decimal('0.1'),       # 單筆最大虧損比例：10%（適度放寬）
            "commission_penalty_factor": Decimal('0.2'),   # 手續費懲罰係數（進一步降低）
            "risk_awareness_threshold": Decimal('0.05'),   # 新增：風險意識門檻（5%）
            
            # 持倉管理（簡單但教學導向）
            "hold_profit_bonus": Decimal('0.4'),           # 持有獲利部位獎勵（增強讓利潤奔跑概念）
            "fast_stop_loss_bonus": Decimal('0.5'),        # 快速停損獎勵（強化風險控制概念）
            "position_experimentation_bonus": Decimal('0.3'), # 倉位實驗獎勵（鼓勵部位管理學習）
            
            # 教學導向獎勵（新增強化版）
            "first_trade_bonus": Decimal('0.6'),           # 首次交易獎勵（增強初始動機）
            "consecutive_learning_bonus": Decimal('0.25'),  # 連續學習獎勵（持續改進）
            "mistake_recovery_bonus": Decimal('0.4'),       # 錯誤恢復獎勵（從失敗中學習）
            "early_profit_multiplier": Decimal('1.5'),     # 新增：早期獲利倍數（前20筆交易）
            "learning_milestone_bonus": Decimal('0.3'),    # 新增：學習里程碑獎勵
            "concept_breakthrough_bonus": Decimal('0.8'),  # 新增：概念突破獎勵
        }
    
    def _get_stage2_config(self) -> Dict[str, Decimal]:
        """
        階段2：進階訓練階段配置  
        目標：引入風險概念，教導模型風險調整收益，優化停損停利
        """
        return {
            # 風險調整收益
            "sortino_ratio_factor": Decimal('1.5'),        # 索提諾比率係數
            "sharpe_ratio_factor": Decimal('1.0'),         # 夏普比率係數
            "calmar_ratio_factor": Decimal('0.8'),         # 卡爾瑪比率係數
            
            # 進階風險管理
            "drawdown_penalty_factor": Decimal('2.0'),     # 回撤懲罰係數
            "volatility_penalty_factor": Decimal('1.0'),   # 波動性懲罰係數
            "var_penalty_factor": Decimal('1.5'),          # VaR風險懲罰
            
            # 停損停利優化
            "profit_run_bonus": Decimal('1.0'),            # 讓利潤奔跑獎勵
            "quick_cut_loss_bonus": Decimal('0.8'),        # 快速截斷虧損獎勵
            "profit_loss_ratio_bonus": Decimal('0.6'),     # 盈虧比獎勵
            
            # 交易效率
            "win_rate_penalty": Decimal('0.5'),            # 低勝率懲罰（< 50%）
            "commission_efficiency": Decimal('1.0'),       # 手續費效率考量
            "trade_quality_bonus": Decimal('0.4'),         # 交易品質獎勵
            
            # 趨勢跟隨
            "trend_following_bonus": Decimal('0.6'),       # 趨勢跟隨獎勵
            "momentum_consistency": Decimal('0.3'),        # 動量一致性獎勵
        }
    
    def _get_stage3_config(self) -> Dict[str, Decimal]:
        """
        階段3：最終訓練階段配置
        目標：實現專業級交易表現，整合所有先進指標
        """
        return {
            # 高級績效指標
            "information_ratio_factor": Decimal('2.0'),    # 信息比率
            "treynor_ratio_factor": Decimal('1.5'),        # 特雷諾比率
            "omega_ratio_factor": Decimal('1.8'),          # Omega比率
            "sterling_ratio_factor": Decimal('1.2'),       # Sterling比率
            
            # Kelly準則與資金管理
            "kelly_criterion_bonus": Decimal('1.5'),       # Kelly準則獎勵
            "optimal_f_bonus": Decimal('1.0'),             # 最佳f值獎勵
            "position_sizing_efficiency": Decimal('0.8'),  # 倉位管理效率
            
            # 高階風險指標
            "max_drawdown_duration_penalty": Decimal('2.5'), # 最大回撤持續時間懲罰
            "tail_ratio_bonus": Decimal('1.0'),            # 尾部比率獎勵
            "skewness_preference": Decimal('0.5'),         # 收益偏度偏好
            "kurtosis_penalty": Decimal('0.3'),            # 峰度懲罰
            
            # 市場適應性
            "regime_adaptation_bonus": Decimal('1.2'),     # 市場狀態適應獎勵
            "volatility_timing_bonus": Decimal('0.8'),     # 波動率時機選擇
            "correlation_awareness": Decimal('0.6'),       # 相關性認知獎勵
            
            # 交易藝術與直覺
            "contrarian_timing_bonus": Decimal('0.4'),     # 逆向時機選擇
            "market_micro_structure": Decimal('0.3'),      # 市場微觀結構認知
            "behavioral_finance_bonus": Decimal('0.5'),    # 行為金融學獎勵
            
            # 組合效應
            "portfolio_diversification": Decimal('0.7'),   # 投資組合多樣化
            "cross_asset_momentum": Decimal('0.4'),        # 跨資產動量
            "factor_exposure_balance": Decimal('0.6'),     # 因子暴露平衡
        }
    
    def evaluate_current_stage(self) -> int:
        """
        評估當前訓練階段
        基於過去100步的交易表現決定階段
        """
        if len(self.trade_history) < 20:  # 數據不足
            return 1
        
        # 防止頻繁切換
        if self.stage_switch_cooldown > 0:
            self.stage_switch_cooldown -= 1
            return self.current_stage
        
        # 計算平均每筆交易期望值
        closed_trades = [t for t in self.trade_history if t.get('realized_pnl', 0) != 0]
        if len(closed_trades) < 10:
            return 1
            
        avg_trade_expectation = sum(t['realized_pnl'] for t in closed_trades) / len(closed_trades)
        
        # 計算勝率
        winning_trades = [t for t in closed_trades if t['realized_pnl'] > 0]
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        
        # 階段判斷邏輯
        new_stage = self.current_stage
        
        if avg_trade_expectation < 0:
            new_stage = 1  # 初始訓練階段
        elif avg_trade_expectation > 0 and win_rate < 0.5:
            new_stage = 2  # 進階訓練階段
        elif avg_trade_expectation > 0 and win_rate >= 0.5:
            new_stage = 3  # 最終訓練階段
          # 如果階段改變，設置冷卻期
        if new_stage != self.current_stage:
            self.stage_switch_cooldown = 25  # 25步冷卻期
            logger.info(f"獎勵系統階段切換: {self.current_stage} -> {new_stage}")
            logger.info(f"平均交易期望值: {avg_trade_expectation:.6f}, 勝率: {win_rate:.3f}")
        
        return new_stage
    
    def calculate_stage1_reward(self, 
                                current_portfolio_value: Decimal,
                                prev_portfolio_value: Decimal,
                                commission_this_step: Decimal,
                                trade_log: List[Dict[str, Any]],
                                positions_data: Dict[str, Any],
                                market_data: Dict[str, Any],
                                episode_step: int) -> Dict[str, Decimal]:
        """
        階段1獎勵計算：教學導向獎勵，鼓勵多交易多學習
        重點：讓模型多嘗試、多交易、理解獲利與虧損的基本原理
        """
        config = self.stage_configs[1]
        components = {}
        
        # 1. 基礎收益獎勵（強化正面學習體驗）
        step_return = current_portfolio_value - prev_portfolio_value
        if step_return > Decimal('0'):
            # 獲利時給予更大的鼓勵，讓模型明確知道這是好行為
            profit_reward = step_return / self.initial_capital * config["profit_reward_factor"]
            components['profit_reward'] = profit_reward
            
            # 新增：首次獲利額外獎勵
            if not hasattr(self, '_first_profit_achieved'):
                self._first_profit_achieved = True
                components['first_profit_milestone'] = config["first_trade_bonus"]
        else:
            # 虧損時給予較輕的懲罰，避免模型過度保守
            loss_penalty = abs(step_return) / self.initial_capital * config["loss_penalty_factor"]
            components['loss_penalty'] = -loss_penalty
        
        # 2. 增強交易頻率獎勵（鼓勵多嘗試）
        recent_trades = [t for t in trade_log[-10:] if t.get('realized_pnl', 0) != 0]
        if len(recent_trades) > 0:            # 使用非線性獎勵，鼓勵更多交易
            trade_count = len(recent_trades)
            frequency_multiplier = min(Decimal(str(trade_count)) / Decimal('5.0'), Decimal('2.0'))  # 5筆交易達到最大倍數
            trade_frequency_bonus = Decimal(str(trade_count)) * config["trade_frequency_bonus"] * frequency_multiplier / Decimal('10')
            components['trade_frequency'] = trade_frequency_bonus
            
            # 新增：交易里程碑獎勵
            total_trades = len([t for t in trade_log if t.get('realized_pnl', 0) != 0])
            if total_trades in [5, 10, 20, 50]:  # 里程碑交易數
                components['trade_milestone'] = config["learning_progress_bonus"]
        
        # 3. 增強探索獎勵（鼓勵嘗試不同策略）
        exploration_reward = self._calculate_enhanced_exploration_reward(trade_log, config)
        if exploration_reward > Decimal('0'):
            components['exploration'] = exploration_reward
        
        # 4. 教學導向的交易概念獎勵
        concept_reward = self._calculate_trading_concept_mastery(trade_log, positions_data, config)
        if concept_reward > Decimal('0'):
            components['concept_mastery'] = concept_reward
        
        # 5. 改進的快速停損獎勵（教學意圖明確）
        fast_stop_reward = self._calculate_educational_stop_loss_reward(trade_log, config)
        if fast_stop_reward > Decimal('0'):
            components['fast_stop_loss'] = fast_stop_reward
        
        # 6. 增強持有獲利部位獎勵（讓模型體驗"讓利潤奔跑"）
        hold_profit_reward = self._calculate_enhanced_hold_profit_reward(positions_data, config)
        if hold_profit_reward > Decimal('0'):
            components['hold_profit'] = hold_profit_reward
        
        # 7. 錯誤恢復獎勵（從虧損中學習）
        recovery_reward = self._calculate_mistake_recovery_reward(trade_log, config)
        if recovery_reward > Decimal('0'):
            components['mistake_recovery'] = recovery_reward
        
        # 8. 降低手續費懲罰（在學習階段不要讓手續費阻礙實驗）
        commission_penalty = commission_this_step / self.initial_capital * config["commission_penalty_factor"]
        components['commission'] = -commission_penalty
        
        # 9. 新增：學習進度獎勵
        progress_reward = self._calculate_learning_progress_reward(episode_step, trade_log, config)
        if progress_reward > Decimal('0'):
            components['learning_progress'] = progress_reward
        
        return components
    
    def calculate_stage2_reward(self,
                              current_portfolio_value: Decimal,
                              prev_portfolio_value: Decimal,
                              commission_this_step: Decimal,
                              trade_log: List[Dict[str, Any]],
                              positions_data: Dict[str, Any],
                              market_data: Dict[str, Any],
                              episode_step: int) -> Dict[str, Decimal]:
        """
        階段2獎勵計算：引入風險調整指標，教導風險概念
        """
        config = self.stage_configs[2]
        components = {}
        
        # 1. 索提諾比率（重點：下行風險）
        if len(self.returns_history) >= 20:
            sortino_ratio = self._calculate_sortino_ratio()
            components['sortino_ratio'] = sortino_ratio * config["sortino_ratio_factor"]
        
        # 2. 夏普比率（總體風險調整收益）
        if len(self.returns_history) >= 15:
            sharpe_ratio = self._calculate_sharpe_ratio()
            components['sharpe_ratio'] = sharpe_ratio * config["sharpe_ratio_factor"]
        
        # 3. 回撤控制（教導風險管理）
        current_drawdown = self._calculate_current_drawdown(current_portfolio_value)
        if current_drawdown > Decimal('0.02'):  # 2%以上回撤開始懲罰
            dd_penalty = current_drawdown * config["drawdown_penalty_factor"]
            components['drawdown_penalty'] = -dd_penalty
        
        # 4. 讓利潤奔跑獎勵
        profit_run_reward = self._calculate_profit_run_bonus(positions_data, config)
        if profit_run_reward > Decimal('0'):
            components['profit_run'] = profit_run_reward
        
        # 5. 快速截斷虧损
        quick_cut_reward = self._calculate_quick_cut_loss_bonus(trade_log, config)
        if quick_cut_reward > Decimal('0'):
            components['quick_cut_loss'] = quick_cut_reward
        
        # 6. 盈虧比獎勵
        profit_loss_ratio = self._calculate_profit_loss_ratio(trade_log)
        if profit_loss_ratio > Decimal('1.0'):
            components['profit_loss_ratio'] = (profit_loss_ratio - Decimal('1.0')) * config["profit_loss_ratio_bonus"]
        
        # 7. 勝率懲罰（< 50%時）
        win_rate = self._calculate_win_rate(trade_log)
        if win_rate < 0.5:
            win_rate_penalty = (Decimal('0.5') - Decimal(str(win_rate))) * config["win_rate_penalty"]
            components['win_rate_penalty'] = -win_rate_penalty
        
        # 8. 趨勢跟隨獎勵
        trend_bonus = self._calculate_trend_following_bonus(positions_data, market_data, config)
        if trend_bonus > Decimal('0'):
            components['trend_following'] = trend_bonus
        
        # 9. 手續費效率
        commission_efficiency = self._calculate_commission_efficiency(trade_log, commission_this_step)
        components['commission_efficiency'] = commission_efficiency * config["commission_efficiency"]
        
        return components
    
    def calculate_stage3_reward(self,
                              current_portfolio_value: Decimal,
                              prev_portfolio_value: Decimal,
                              commission_this_step: Decimal,
                              trade_log: List[Dict[str, Any]],
                              positions_data: Dict[str, Any],
                              market_data: Dict[str, Any],
                              episode_step: int) -> Dict[str, Decimal]:
        """
        階段3獎勵計算：專業級指標，追求卓越表現
        """
        config = self.stage_configs[3]
        components = {}
        
        # 1. 信息比率（超額收益的一致性）
        if len(self.returns_history) >= 30:
            info_ratio = self._calculate_information_ratio()
            components['information_ratio'] = info_ratio * config["information_ratio_factor"]
        
        # 2. Kelly準則倉位管理
        kelly_bonus = self._calculate_kelly_criterion_bonus(trade_log, positions_data, config)
        if kelly_bonus > Decimal('0'):
            components['kelly_criterion'] = kelly_bonus
        
        # 3. Omega比率（所有收益時刻的優化）
        if len(self.returns_history) >= 25:
            omega_ratio = self._calculate_omega_ratio()
            components['omega_ratio'] = omega_ratio * config["omega_ratio_factor"]
        
        # 4. 尾部比率（極端事件管理）
        tail_ratio = self._calculate_tail_ratio()
        if tail_ratio > Decimal('1.0'):
            components['tail_ratio'] = (tail_ratio - Decimal('1.0')) * config["tail_ratio_bonus"]
        
        # 5. 市場狀態適應性
        regime_adaptation = self._calculate_regime_adaptation_bonus(trade_log, market_data, config)
        if regime_adaptation > Decimal('0'):
            components['regime_adaptation'] = regime_adaptation
        
        # 6. 波動率時機選擇
        volatility_timing = self._calculate_volatility_timing_bonus(positions_data, market_data, config)
        if volatility_timing > Decimal('0'):
            components['volatility_timing'] = volatility_timing
        
        # 7. 行為金融學獎勵（逆向思維）
        behavioral_bonus = self._calculate_behavioral_finance_bonus(trade_log, market_data, config)
        if behavioral_bonus > Decimal('0'):
            components['behavioral_finance'] = behavioral_bonus
        
        # 8. 最大回撤持續時間懲罰
        dd_duration_penalty = self._calculate_drawdown_duration_penalty(config)
        if dd_duration_penalty > Decimal('0'):
            components['drawdown_duration_penalty'] = -dd_duration_penalty
        
        # 9. 收益分佈偏好（偏度和峰度）
        if len(self.returns_history) >= 30:
            skewness_reward = self._calculate_skewness_reward(config)
            kurtosis_penalty = self._calculate_kurtosis_penalty(config)
            components['skewness'] = skewness_reward
            components['kurtosis'] = kurtosis_penalty
        
        # 10. 超越人類直覺的策略獎勵
        unconventional_bonus = self._calculate_unconventional_strategy_bonus(trade_log, positions_data, config)
        if unconventional_bonus > Decimal('0'):
            components['unconventional_strategy'] = unconventional_bonus
        
        return components
    
    def calculate_reward(self, 
                        current_portfolio_value: Decimal,
                        prev_portfolio_value: Decimal,
                        commission_this_step: Decimal,
                        trade_log: List[Dict[str, Any]],
                        positions_data: Dict[str, Any],
                        market_data: Dict[str, Any],
                        episode_step: int) -> float:
        """
        主要獎勵計算函數
        """
        self.step_count += 1
        
        # 更新歷史數據
        self._update_history(current_portfolio_value, prev_portfolio_value, trade_log)
        
        # 每隔一定步數評估階段
        if self.step_count % self.stage_evaluation_interval == 0:
            self.current_stage = self.evaluate_current_stage()
        
        # 根據當前階段計算獎勵
        if self.current_stage == 1:
            reward_components = self.calculate_stage1_reward(
                current_portfolio_value, prev_portfolio_value, commission_this_step,
                trade_log, positions_data, market_data, episode_step
            )
        elif self.current_stage == 2:
            reward_components = self.calculate_stage2_reward(
                current_portfolio_value, prev_portfolio_value, commission_this_step,
                trade_log, positions_data, market_data, episode_step
            )
        else:  # stage 3
            reward_components = self.calculate_stage3_reward(
                current_portfolio_value, prev_portfolio_value, commission_this_step,
                trade_log, positions_data, market_data, episode_step
            )
        
        # 計算總獎勵
        total_reward = sum(reward_components.values())        # 記錄詳細組件（原始獎勵）
        raw_reward_info = {
            'step': episode_step,
            'stage': self.current_stage,
            'portfolio_value': float(current_portfolio_value),
            'components': {k: float(v) for k, v in reward_components.items()},
            'total_reward': float(total_reward)
        }
        
        # 應用獎勵標準化
        normalized_reward_info = self.reward_normalizer.normalize_reward(
            raw_reward_info, method='hybrid'
        )
        
        # 更新總獎勵為標準化後的值
        total_reward = normalized_reward_info['total_reward']
        
        # 保存原始和標準化的獎勵信息
        reward_info = normalized_reward_info.copy()
        reward_info['raw_reward_info'] = raw_reward_info
        
        self.reward_components_history.append(reward_info)
        self.last_portfolio_value = current_portfolio_value
        
        # 新增：更新性能監控數據
        self.performance_metrics['episode_rewards'].append(total_reward)
        self.performance_metrics['portfolio_values'].append(float(current_portfolio_value))
        
        if len(self.performance_metrics['sharpe_ratios']) >= 10:
            self.performance_metrics['sharpe_ratios'].append(float(self._calculate_sharpe_ratio()))
        
        if len(self.performance_metrics['max_drawdowns']) >= 10:
            self.performance_metrics['max_drawdowns'].append(float(self._calculate_current_drawdown(current_portfolio_value)))
        
        if len(self.performance_metrics['profit_factors']) >= 10:
            avg_profit = sum(v for v in self.performance_metrics['episode_rewards'][-10:] if v > 0)
            avg_loss = -sum(v for v in self.performance_metrics['episode_rewards'][-10:] if v < 0)
            profit_factor = avg_profit / avg_loss if avg_loss != 0 else Decimal('1.0')
            self.performance_metrics['profit_factors'].append(float(profit_factor))
        
        return reward_info
    
    # === 輔助計算函數 ===
    
    def _update_history(self, current_value: Decimal, prev_value: Decimal, trade_log: List[Dict[str, Any]]):
        """更新歷史數據"""
        # 更新收益歷史
        if prev_value > Decimal('0'):
            log_return = (current_value / prev_value).ln()
            self.returns_history.append(log_return)
        
        # 更新交易歷史
        for trade in trade_log[-5:]:  # 最近5筆交易
            if trade not in self.trade_history:
                self.trade_history.append(trade)
    
    def _is_new_trading_pattern(self, trade_log: List[Dict[str, Any]]) -> bool:
        """檢測是否出現新的交易模式"""
        if len(trade_log) < 5:
            return True
        
        recent_trades = trade_log[-5:]
        # 簡單檢測：如果最近的交易方向和之前不同
        directions = [1 if t.get('realized_pnl', 0) > 0 else -1 for t in recent_trades]
        return len(set(directions)) > 1
    
    def _calculate_sortino_ratio(self) -> Decimal:
        """計算索提諾比率"""
        if len(self.returns_history) < 10:
            return Decimal('0')
        
        returns = list(self.returns_history)
        mean_return = sum(returns) / len(returns)
        negative_returns = [r for r in returns if r < mean_return]
        
        if not negative_returns:
            return Decimal('2.0')  # 沒有負收益時給予高分
        
        downside_variance = sum((r - mean_return) ** 2 for r in negative_returns) / len(negative_returns)
        downside_std = downside_variance.sqrt()
        
        excess_return = mean_return - self.risk_free_rate
        sortino_ratio = excess_return / (downside_std + Decimal('1e-6'))
        
        return min(max(sortino_ratio, Decimal('-2.0')), Decimal('2.0'))
    
    def _calculate_sharpe_ratio(self) -> Decimal:
        """計算夏普比率"""
        if len(self.returns_history) < 10:
            return Decimal('0')
        
        returns = list(self.returns_history)
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_return = variance.sqrt()
        
        excess_return = mean_return - self.risk_free_rate
        sharpe_ratio = excess_return / (std_return + Decimal('1e-6'))
        
        return min(max(sharpe_ratio, Decimal('-2.0')), Decimal('2.0'))
    
    def _calculate_current_drawdown(self, current_value: Decimal) -> Decimal:
        """計算當前回撤"""
        if not hasattr(self, 'peak_value'):
            self.peak_value = current_value
        
        self.peak_value = max(self.peak_value, current_value)
        drawdown = (self.peak_value - current_value) / self.peak_value
        return max(drawdown, Decimal('0'))
    
    def _calculate_profit_run_bonus(self, positions_data: Dict[str, Any], config: Dict[str, Decimal]) -> Decimal:
        """計算讓利潤奔跑獎勵"""
        total_bonus = Decimal('0')
        positions = positions_data.get('positions', {})
        
        for symbol, position_info in positions.items():
            unrealized_pnl = position_info.get('unrealized_pnl', Decimal('0'))
            hold_duration = position_info.get('hold_duration', 0)
            
            if unrealized_pnl > Decimal('0') and hold_duration > 5:
                profit_ratio = unrealized_pnl / self.initial_capital
                duration_factor = min(Decimal(str(hold_duration)) / Decimal('20'), Decimal('2.0'))
                bonus = profit_ratio * duration_factor * config["profit_run_bonus"]
                total_bonus += bonus
        
        return total_bonus
    
    def _calculate_quick_cut_loss_bonus(self, trade_log: List[Dict[str, Any]], config: Dict[str, Decimal]) -> Decimal:
        """計算快速截斷虧損獎勵"""
        total_bonus = Decimal('0')
        
        for trade in trade_log[-10:]:
            if (trade.get('realized_pnl', 0) < 0 and 
                trade.get('hold_duration', 0) <= 3):
                loss_ratio = abs(trade['realized_pnl']) / self.initial_capital
                if loss_ratio <= Decimal('0.02'):  # 2%以內的快速停損
                    bonus = (Decimal('0.02') - loss_ratio) * config["quick_cut_loss_bonus"]
                    total_bonus += bonus
        
        return total_bonus
    
    def _calculate_profit_loss_ratio(self, trade_log: List[Dict[str, Any]]) -> Decimal:
        """計算盈虧比"""
        recent_trades = [t for t in trade_log[-20:] if t.get('realized_pnl', 0) != 0]
        
        if len(recent_trades) < 5:
            return Decimal('1.0')
        
        profits = [t['realized_pnl'] for t in recent_trades if t['realized_pnl'] > 0]
        losses = [abs(t['realized_pnl']) for t in recent_trades if t['realized_pnl'] < 0]
        
        if not profits or not losses:
            return Decimal('1.0')
        
        avg_profit = sum(profits) / len(profits)
        avg_loss = sum(losses) / len(losses)
        
        return avg_profit / max(avg_loss, Decimal('1e-6'))
    
    def _calculate_win_rate(self, trade_log: List[Dict[str, Any]]) -> float:
        """計算勝率"""
        recent_trades = [t for t in trade_log[-20:] if t.get('realized_pnl', 0) != 0]
        
        if len(recent_trades) < 5:
            return 0.5
        
        winning_trades = [t for t in recent_trades if t['realized_pnl'] > 0]
        return len(winning_trades) / len(recent_trades)
    
    def _calculate_trend_following_bonus(self, positions_data: Dict[str, Any], 
                                       market_data: Dict[str, Any], 
                                       config: Dict[str, Decimal]) -> Decimal:
        """計算趨勢跟隨獎勵"""
        # 簡化版趨勢跟隨邏輯
        total_bonus = Decimal('0')
        positions = positions_data.get('positions', {})
        
        for symbol, position_info in positions.items():
            units = position_info.get('units', Decimal('0'))
            if abs(units) > Decimal('1e-9'):
                # 假設有趨勢指標，這裡簡化處理
                trend_alignment = Decimal('0.1')  # 簡化假設
                total_bonus += trend_alignment * config["trend_following_bonus"]
        
        return total_bonus
    
    def _calculate_commission_efficiency(self, trade_log: List[Dict[str, Any]], 
                                       commission_this_step: Decimal) -> Decimal:
        """計算手續費效率"""
        recent_trades = [t for t in trade_log[-10:] if t.get('realized_pnl', 0) != 0]
        
        if not recent_trades:
            return Decimal('-1.0')
        
        total_profit = sum(max(t['realized_pnl'], 0) for t in recent_trades)
        total_commission = sum(t.get('commission', 0) for t in recent_trades)
        
        if total_commission <= Decimal('0'):
            return Decimal('0')
        
        efficiency = total_profit / total_commission
        return min(efficiency - Decimal('1.0'), Decimal('5.0'))  # 效率超過1的部分作為獎勵
    
    def _calculate_information_ratio(self) -> Decimal:
        """計算信息比率"""
        if len(self.returns_history) < 20:
            return Decimal('0')
        
        returns = list(self.returns_history)
        excess_returns = [r - self.risk_free_rate for r in returns]
        
        mean_excess = sum(excess_returns) / len(excess_returns)
        tracking_error = (sum((r - mean_excess) ** 2 for r in excess_returns) / len(excess_returns)).sqrt()
        
        info_ratio = mean_excess / (tracking_error + Decimal('1e-6'))
        return min(max(info_ratio, Decimal('-2.0')), Decimal('2.0'))
    
    def _calculate_kelly_criterion_bonus(self, trade_log: List[Dict[str, Any]], 
                                       positions_data: Dict[str, Any], 
                                       config: Dict[str, Decimal]) -> Decimal:
        """計算Kelly準則獎勵"""
        # Kelly準則：f* = (bp - q) / b
        # 其中 b = 賠率, p = 勝率, q = 敗率
        recent_trades = [t for t in trade_log[-30:] if t.get('realized_pnl', 0) != 0]
        
        if len(recent_trades) < 10:
            return Decimal('0')
        
        wins = [t for t in recent_trades if t['realized_pnl'] > 0]
        losses = [t for t in recent_trades if t['realized_pnl'] < 0]
        
        if not wins or not losses:
            return Decimal('0')
        
        win_rate = len(wins) / len(recent_trades)
        avg_win = sum(t['realized_pnl'] for t in wins) / len(wins)
        avg_loss = abs(sum(t['realized_pnl'] for t in losses)) / len(losses)
        
        odds_ratio = avg_win / avg_loss
        kelly_f = (odds_ratio * win_rate - (1 - win_rate)) / odds_ratio
        
        # 如果Kelly值合理（0-1之間），給予獎勵
        if Decimal('0') < kelly_f < Decimal('1'):
            return kelly_f * config["kelly_criterion_bonus"]
        
        return Decimal('0')
    
    def _calculate_omega_ratio(self, threshold: Decimal = None) -> Decimal:
        """計算Omega比率"""
        if threshold is None:
            threshold = self.risk_free_rate
        
        if len(self.returns_history) < 15:
            return Decimal('1.0')
        
        returns = list(self.returns_history)
        gains = sum(max(r - threshold, Decimal('0')) for r in returns)
        losses = sum(max(threshold - r, Decimal('0')) for r in returns)
        
        if losses <= Decimal('0'):
            return Decimal('3.0')  # 沒有損失時給高分
        
        omega_ratio = gains / losses
        return min(omega_ratio, Decimal('3.0'))
    
    def _calculate_tail_ratio(self) -> Decimal:
        """計算尾部比率（95分位數 / 5分位數的絕對值）"""
        if len(self.returns_history) < 20:
            return Decimal('1.0')
        
        returns = sorted(self.returns_history)
        n = len(returns)
        
        p95_idx = int(0.95 * n)
        p5_idx = int(0.05 * n)
        
        p95_return = returns[min(p95_idx, n-1)]
        p5_return = returns[max(p5_idx, 0)]
        
        if p5_return >= Decimal('0'):
            return Decimal('2.0')  # 沒有顯著負收益
        
        tail_ratio = p95_return / abs(p5_return)
        return min(max(tail_ratio, Decimal('0.1')), Decimal('5.0'))
    
    def _calculate_regime_adaptation_bonus(self, trade_log: List[Dict[str, Any]], 
                                         market_data: Dict[str, Any], 
                                         config: Dict[str, Decimal]) -> Decimal:
        """計算市場狀態適應獎勵"""
        # 簡化版市場狀態適應邏輯
        if len(trade_log) < 20:
            return Decimal('0')
        
        # 檢測市場波動率變化
        volatility_changes = self._detect_volatility_regime_changes()
        
        if volatility_changes > 0:
            # 如果模型在波動率變化後調整了策略，給予獎勵
            adaptation_score = min(Decimal(str(volatility_changes)), Decimal('2.0'))
            return adaptation_score * config["regime_adaptation_bonus"]
        
        return Decimal('0')
    
    def _calculate_volatility_timing_bonus(self, positions_data: Dict[str, Any], 
                                         market_data: Dict[str, Any], 
                                         config: Dict[str, Decimal]) -> Decimal:
        """計算波動率時機選擇獎勵"""
        # 簡化版波動率時機選擇
        current_volatility = self._estimate_current_volatility()
        
        positions = positions_data.get('positions', {})
        total_exposure = sum(abs(pos.get('units', Decimal('0'))) for pos in positions.values())
        
        # 高波動率時減少倉位，低波動率時增加倉位
        if current_volatility > Decimal('0.02'):  # 高波動
            if total_exposure < Decimal('0.5'):  # 減少倉位
                return config["volatility_timing_bonus"]
        elif current_volatility < Decimal('0.01'):  # 低波動
            if total_exposure > Decimal('0.8'):  # 增加倉位
                return config["volatility_timing_bonus"]
        
        return Decimal('0')
    
    def _calculate_behavioral_finance_bonus(self, trade_log: List[Dict[str, Any]], 
                                          market_data: Dict[str, Any], 
                                          config: Dict[str, Decimal]) -> Decimal:
        """計算行為金融學獎勵（逆向思維等）"""
        # 檢測是否有反向操作的跡象
        recent_trades = trade_log[-10:]
        
        contrarian_signals = 0
        for i in range(1, len(recent_trades)):
            prev_trade = recent_trades[i-1]
            curr_trade = recent_trades[i]
            
            # 如果前一筆虧損後立即調整策略
            if (prev_trade.get('realized_pnl', 0) < 0 and 
                curr_trade.get('realized_pnl', 0) > 0):
                contrarian_signals += 1
        
        if contrarian_signals > 0:
            return min(Decimal(str(contrarian_signals)), Decimal('3.0')) * config["behavioral_finance_bonus"]
        
        return Decimal('0')
    
    def _calculate_drawdown_duration_penalty(self, config: Dict[str, Decimal]) -> Decimal:
        """計算回撤持續時間懲罰"""
        if not hasattr(self, 'drawdown_start_step'):
            self.drawdown_start_step = None
        
        current_drawdown = self._calculate_current_drawdown(self.last_portfolio_value)
        
        if current_drawdown > Decimal('0.01'):  # 1%以上回撤
            if self.drawdown_start_step is None:
                self.drawdown_start_step = self.step_count
            
            duration = self.step_count - self.drawdown_start_step
            if duration > 20:  # 超過20步的回撤
                penalty = (Decimal(str(duration)) - Decimal('20')) * config["max_drawdown_duration_penalty"] / Decimal('100')
                return penalty
        else:
            self.drawdown_start_step = None
        
        return Decimal('0')
    
    def _calculate_skewness_reward(self, config: Dict[str, Decimal]) -> Decimal:
        """計算收益偏度獎勵（正偏度更好）"""
        if len(self.returns_history) < 30:
            return Decimal('0')
        
        returns = list(self.returns_history)
        mean_return = sum(returns) / len(returns)
        
        # 計算三階中心矩
        third_moment = sum((r - mean_return) ** 3 for r in returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = variance.sqrt()
        
        if std_dev <= Decimal('0'):
            return Decimal('0')
        
        skewness = third_moment / (std_dev ** 3)
        
        # 正偏度給獎勵
        if skewness > Decimal('0'):
            return min(skewness, Decimal('2.0')) * config["skewness_preference"]
        
        return Decimal('0')
    
    def _calculate_kurtosis_penalty(self, config: Dict[str, Decimal]) -> Decimal:
        """計算峰度懲罰（過高峰度表示極端風險）"""
        if len(self.returns_history) < 30:
            return Decimal('0')
        
        returns = list(self.returns_history)
        mean_return = sum(returns) / len(returns)
        
        # 計算四階中心矩
        fourth_moment = sum((r - mean_return) ** 4 for r in returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        
        if variance <= Decimal('0'):
            return Decimal('0')
        
        kurtosis = fourth_moment / (variance ** 2) - Decimal('3')  # 超額峰度
        
        # 過高峰度給懲罰
        if kurtosis > Decimal('1'):
            penalty = (kurtosis - Decimal('1')) * config["kurtosis_penalty"]
            return -min(penalty, Decimal('1.0'))
        
        return Decimal('0')
    
    def _calculate_unconventional_strategy_bonus(self, trade_log: List[Dict[str, Any]], 
                                               positions_data: Dict[str, Any], 
                                               config: Dict[str, Decimal]) -> Decimal:
        """計算超越人類直覺的策略獎勵"""
        # 檢測非傳統的交易模式
        unconventional_score = Decimal('0')
        
        # 1. 檢測逆向操作成功案例
        recent_trades = trade_log[-15:]
        for trade in recent_trades:
            # 如果在市場普遍下跌時做多並獲利，或在上漲時做空並獲利
            if trade.get('realized_pnl', 0) > 0:
                # 這裡需要更複雜的市場狀態檢測，簡化處理
                unconventional_score += Decimal('0.1')
        
        # 2. 檢測複雜的多資產相關性操作
        positions = positions_data.get('positions', {})
        if len(positions) > 1:
            # 多資產同時操作可能顯示複雜策略
            unconventional_score += Decimal('0.2')
        
        # 3. 檢測時機選擇的精準度
        profitable_trades = [t for t in recent_trades if t.get('realized_pnl', 0) > 0]
        if len(profitable_trades) > 0:
            avg_hold_duration = sum(t.get('hold_duration', 0) for t in profitable_trades) / len(profitable_trades)
            if 2 <= avg_hold_duration <= 8:  # 短期精準操作
                unconventional_score += Decimal('0.15')
        
        return min(unconventional_score, Decimal('1.0')) * config.get("behavioral_finance_bonus", Decimal('0.5'))
    
    def _detect_volatility_regime_changes(self) -> int:
        """檢測波動率狀態變化"""
        if len(self.returns_history) < 20:
            return 0
        
        returns = list(self.returns_history)
        recent_vol = self._calculate_volatility(returns[-10:])
        older_vol = self._calculate_volatility(returns[-20:-10])
        
        # 如果波動率變化超過50%
        if abs(recent_vol - older_vol) / older_vol > 0.5:
            return 1
        
        return 0
    
    def _calculate_volatility(self, returns: List[Decimal]) -> Decimal:
        """計算波動率"""
        if len(returns) < 2:
            return Decimal('0')
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return variance.sqrt()
    
    def _estimate_current_volatility(self) -> Decimal:
        """估計當前波動率"""
        if len(self.returns_history) < 10:
            return Decimal('0.02')  # 默認值
        
        recent_returns = list(self.returns_history)[-10:]
        return self._calculate_volatility(recent_returns)
    
    # === 階段1增強版輔助方法 ===
    
    def _calculate_enhanced_exploration_reward(self, trade_log: List[Dict[str, Any]], config: Dict[str, Decimal]) -> Decimal:
        """
        增強探索獎勵：鼓勵模型嘗試不同的交易策略和方向
        教學目標：讓模型理解多空雙向交易的概念
        """
        if len(trade_log) < 3:
            return config["exploration_bonus"]  # 早期給予基礎探索獎勵
        
        recent_trades = trade_log[-8:]  # 檢查最近8筆交易
        closed_trades = [t for t in recent_trades if t.get('realized_pnl', 0) != 0]
        
        if len(closed_trades) < 2:
            return Decimal('0')
        
        exploration_score = Decimal('0')
        
        # 1. 檢測交易方向多樣性（做多vs做空）
        long_trades = sum(1 for t in closed_trades if t.get('trade_type') == 'long' or 
                         (t.get('realized_pnl', 0) != 0 and t.get('units', 0) > 0))
        short_trades = len(closed_trades) - long_trades
        
        if long_trades > 0 and short_trades > 0:
            exploration_score += config["exploration_bonus"] * Decimal('0.5')  # 雙向交易獎勵
        
        # 2. 檢測不同持倉時間的嘗試
        hold_durations = [t.get('hold_duration', 0) for t in closed_trades]
        if len(set(hold_durations)) >= 3:  # 至少3種不同的持倉時間
            exploration_score += config["exploration_bonus"] * Decimal('0.3')
        
        # 3. 檢測交易規模的多樣性
        trade_sizes = [abs(t.get('units', 0)) for t in closed_trades]
        if len(set(trade_sizes)) >= 2:  # 至少2種不同的交易規模
            exploration_score += config["exploration_bonus"] * Decimal('0.2')
        
        return min(exploration_score, config["exploration_bonus"])  # 限制最大獎勵
    
    def _calculate_trading_concept_mastery(self, trade_log: List[Dict[str, Any]], 
                                         positions_data: Dict[str, Any], 
                                         config: Dict[str, Decimal]) -> Decimal:
        """
        交易概念掌握獎勵：獎勵模型掌握基本交易概念
        教學目標：讓模型理解獲利、虧損、做多、做空等基本概念
        """
        concept_score = Decimal('0')
        recent_trades = [t for t in trade_log[-10:] if t.get('realized_pnl', 0) != 0]
        
        if len(recent_trades) < 2:
            return Decimal('0')
        
        # 1. 掌握獲利概念：能夠實現獲利交易
        profitable_trades = [t for t in recent_trades if t.get('realized_pnl', 0) > 0]
        if len(profitable_trades) > 0:
            profit_ratio = len(profitable_trades) / len(recent_trades)
            concept_score += config["concept_mastery_bonus"] * Decimal(str(profit_ratio))
        
        # 2. 掌握止損概念：能夠及時止損
        loss_trades = [t for t in recent_trades if t.get('realized_pnl', 0) < 0]
        quick_stops = [t for t in loss_trades if t.get('hold_duration', 0) <= 5]
        if len(loss_trades) > 0 and len(quick_stops) > 0:
            stop_ratio = len(quick_stops) / len(loss_trades)
            concept_score += config["concept_mastery_bonus"] * Decimal(str(stop_ratio)) * Decimal('0.5')
        
        # 3. 掌握持倉概念：理解持有部位的意義
        positions = positions_data.get('positions', {})
        if len(positions) > 0:
            active_positions = sum(1 for pos in positions.values() if abs(pos.get('units', 0)) > 0)
            if active_positions > 0:
                concept_score += config["concept_mastery_bonus"] * Decimal('0.3')
        
        return concept_score
    
    def _calculate_educational_stop_loss_reward(self, trade_log: List[Dict[str, Any]], 
                                              config: Dict[str, Decimal]) -> Decimal:
        """
        教學導向停損獎勵：明確教導模型什麼是好的停損行為
        教學目標：培養風險意識，理解止損的重要性
        """
        stop_loss_reward = Decimal('0')
        recent_trades = [t for t in trade_log[-8:] if t.get('realized_pnl', 0) < 0]  # 只看虧損交易
        
        for trade in recent_trades:
            hold_duration = trade.get('hold_duration', 0)
            loss_ratio = abs(trade.get('realized_pnl', 0)) / self.initial_capital
              # 教學規則：快速止損小虧損 = 好行為
            if hold_duration <= 4 and loss_ratio <= config["max_single_loss_ratio"]:
                # 越快止損，越小虧損，獎勵越大
                time_factor = max(Decimal('0'), (Decimal('5') - Decimal(str(hold_duration))) / Decimal('5'))  # 1步止損=1.0，4步止損=0.2
                size_factor = max(Decimal('0'), (config["max_single_loss_ratio"] - loss_ratio) / config["max_single_loss_ratio"])
                
                bonus = config["fast_stop_loss_bonus"] * time_factor * size_factor
                stop_loss_reward += bonus
        
        return stop_loss_reward
    
    def _calculate_enhanced_hold_profit_reward(self, positions_data: Dict[str, Any], 
                                             config: Dict[str, Decimal]) -> Decimal:
        """
        增強持有獲利獎勵：教導模型"讓利潤奔跑"的概念
        教學目標：讓模型理解持有獲利部位可以獲得更多收益
        """
        hold_reward = Decimal('0')
        positions = positions_data.get('positions', {})
        
        for symbol, position_info in positions.items():
            unrealized_pnl = position_info.get('unrealized_pnl', Decimal('0'))
            hold_duration = position_info.get('hold_duration', 0)
            
            if unrealized_pnl > Decimal('0') and hold_duration > 1:  # 降低門檻
                profit_ratio = unrealized_pnl / self.initial_capital
                  # 教學導向：持有時間越長，獎勵遞增但有上限
                time_factor = min(Decimal(str(hold_duration)) / Decimal('8.0'), Decimal('1.5'))  # 8步達到最大1.5倍
                
                # 根據獲利幅度給予不同獎勵
                if profit_ratio > Decimal('0.02'):  # 2%以上獲利
                    bonus_multiplier = Decimal('1.5')
                elif profit_ratio > Decimal('0.01'):  # 1-2%獲利
                    bonus_multiplier = Decimal('1.2')
                else:  # 小於1%獲利
                    bonus_multiplier = Decimal('1.0')
                
                hold_bonus = config["hold_profit_bonus"] * profit_ratio * time_factor * bonus_multiplier
                hold_reward += hold_bonus
        
        return hold_reward
    
    def _calculate_mistake_recovery_reward(self, trade_log: List[Dict[str, Any]], 
                                         config: Dict[str, Decimal]) -> Decimal:
        """
        錯誤恢復獎勵：獎勵模型從虧損中快速恢復
        教學目標：讓模型理解虧損後的調整和恢復很重要
        """
        if len(trade_log) < 4:
            return Decimal('0')
        
        recovery_reward = Decimal('0')
        recent_trades = [t for t in trade_log[-6:] if t.get('realized_pnl', 0) != 0]
        
        # 檢測虧損後的快速恢復模式
        for i in range(len(recent_trades) - 1):
            current_trade = recent_trades[i]
            next_trade = recent_trades[i + 1]
            
            # 如果前一筆虧損，後一筆獲利
            if (current_trade.get('realized_pnl', 0) < 0 and 
                next_trade.get('realized_pnl', 0) > 0):
                
                # 計算恢復比例
                loss_amount = abs(current_trade.get('realized_pnl', 0))
                profit_amount = next_trade.get('realized_pnl', 0)
                
                if profit_amount >= loss_amount * 0.5:  # 至少恢復50%虧損
                    recovery_ratio = min(profit_amount / loss_amount, 2.0)  # 最大2倍恢復
                    recovery_reward += config["mistake_recovery_bonus"] * Decimal(str(recovery_ratio * 0.5))
        
        return recovery_reward
    
    def _calculate_learning_progress_reward(self, episode_step: int, 
                                          trade_log: List[Dict[str, Any]], 
                                          config: Dict[str, Decimal]) -> Decimal:
        """
        學習進度獎勵：根據學習階段給予適當獎勵
        教學目標：鼓勵持續學習和改進
        """
        progress_reward = Decimal('0')
        
        # 1. 早期交易鼓勵（前50步）
        if episode_step <= 50:
            closed_trades = [t for t in trade_log if t.get('realized_pnl', 0) != 0]
            if len(closed_trades) >= episode_step // 10:  # 每10步至少1筆交易
                progress_reward += config["learning_progress_bonus"] * Decimal('0.3')
        
        # 2. 交易品質改善獎勵
        if len(trade_log) >= 10:
            recent_5_trades = [t for t in trade_log[-5:] if t.get('realized_pnl', 0) != 0]
            previous_5_trades = [t for t in trade_log[-10:-5] if t.get('realized_pnl', 0) != 0]
            
            if len(recent_5_trades) >= 3 and len(previous_5_trades) >= 3:
                recent_avg = sum(t.get('realized_pnl', 0) for t in recent_5_trades) / len(recent_5_trades)
                previous_avg = sum(t.get('realized_pnl', 0) for t in previous_5_trades) / len(previous_5_trades)
                
                if recent_avg > previous_avg:  # 最近交易表現改善
                    improvement_ratio = (recent_avg - previous_avg) / max(abs(previous_avg), self.initial_capital * Decimal('0.001'))
                    progress_reward += config["learning_progress_bonus"] * min(improvement_ratio, Decimal('0.5'))
        
        return progress_reward

    def get_current_config(self) -> Dict[str, Any]:
        """獲取當前階段的配置"""
        return self.stage_configs.get(self.current_stage, {})
    
    def get_current_stage(self) -> int:
        """獲取當前階段"""
        return self.current_stage
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """獲取詳細的性能指標"""
        closed_trades = [t for t in self.trade_history if t.get('realized_pnl', 0) != 0]
        
        metrics = {
            'current_stage': self.current_stage,
            'total_trades': len(closed_trades),
            'avg_trade_expectation': sum(t['realized_pnl'] for t in closed_trades) / max(len(closed_trades), 1),
            'win_rate': self._calculate_win_rate(list(self.trade_history)),
            'step_count': self.step_count
        }
        
        if len(self.returns_history) >= 10:
            metrics.update({
                'sharpe_ratio': float(self._calculate_sharpe_ratio()),
                'sortino_ratio': float(self._calculate_sortino_ratio()),
                'current_drawdown': float(self._calculate_current_drawdown(self.last_portfolio_value))
            })
        
        return metrics
    
    def get_reward_normalization_stats(self) -> Dict[str, Any]:
        """獲取獎勵標準化統計信息"""
        return self.reward_normalizer.get_normalization_stats()
    
    def update_normalization_config(self, **kwargs):
        """更新標準化配置"""
        self.reward_normalizer.update_config(**kwargs)
    
    def reset_normalization_history(self):
        """重置標準化歷史數據"""
        self.reward_normalizer.reset_history()
