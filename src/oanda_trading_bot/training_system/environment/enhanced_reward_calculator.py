# src/environment/enhanced_reward_calculator.py
"""
增強版獎勵計算器 - 專注於提高勝率和期望值
"""

from decimal import Decimal, getcontext
from typing import List, Dict, Any, Optional
import numpy as np
from collections import deque
import pandas as pd

# 設置高精度計算
getcontext().prec = 28

class EnhancedRewardCalculator:
    """
    增強版獎勵計算器
    
    核心改進：
    1. 智能風險調整收益計算（索提諾比率）
    2. 動態手續費管理
    3. 多層級回撤控制
    4. 市場趨勢感知獎勵
    5. 勝率激勵機制
    """
    
    def __init__(self, initial_capital: Decimal, config: Optional[Dict[str, Any]] = None):
        self.initial_capital = initial_capital
        
        # 增強配置參數
        default_config = {
            "portfolio_log_return_factor": Decimal('0.8'),
            "risk_adjusted_return_factor": Decimal('1.2'),
            "max_drawdown_penalty_factor": Decimal('1.5'),
            "commission_penalty_factor": Decimal('0.8'),
            "margin_call_penalty": Decimal('-50.0'),
            "profit_target_bonus": Decimal('0.3'),
            "hold_penalty_factor": Decimal('0.0005'),
            "win_rate_incentive_factor": Decimal('1.0'),
            "trend_following_bonus": Decimal('0.5'),
            "quick_stop_loss_bonus": Decimal('0.1'),
            "compound_holding_factor": Decimal('0.2'),
        }
        
        if config:
            for key, value in config.items():
                if key in default_config:
                    default_config[key] = Decimal(str(value))
        
        self.reward_config = default_config
        
        # 歷史數據追蹤
        self.returns_history = deque(maxlen=50)  # 增加到50步滾動窗口
        self.reward_components_history: List[Dict[str, Any]] = []
        self.price_history: Dict[str, deque] = {}  # 用於趨勢分析
        self.trade_performance_cache: Dict[str, Any] = {}
        
        # 風險管理參數
        self.risk_free_rate = Decimal('0.02') / Decimal('252')  # 年化2%無風險利率
        self.atr_penalty_threshold = Decimal('0.025')  # 調整到2.5%
        self.volatility_lookback = 20
        
        # 勝率跟蹤
        self.recent_trades_window = 20
        self.min_trades_for_win_rate = 5
        
    def calculate_enhanced_reward(self, 
                                  current_portfolio_value: Decimal,
                                  prev_portfolio_value: Decimal,
                                  commission_this_step: Decimal,
                                  trade_log: List[Dict[str, Any]],
                                  positions_data: Dict[str, Any],
                                  market_data: Dict[str, Any],
                                  episode_step: int) -> float:
        """
        計算增強版獎勵函數
        
        Args:
            current_portfolio_value: 當前投資組合價值
            prev_portfolio_value: 前一步投資組合價值
            commission_this_step: 本步驟手續費
            trade_log: 交易記錄
            positions_data: 持倉數據
            market_data: 市場數據
            episode_step: 當前步驟
            
        Returns:
            總獎勵值
        """
        
        # 計算基礎收益
        log_return = self._calculate_log_return(current_portfolio_value, prev_portfolio_value)
        self.returns_history.append(log_return)
        
        reward_components = {}
        
        # 1. 增強的風險調整收益（權重40%）
        risk_adjusted_reward = self._calculate_enhanced_risk_adjusted_reward()
        reward_components['risk_adjusted'] = risk_adjusted_reward * Decimal('0.4')
        
        # 2. 智能手續費管理（權重10%）
        commission_penalty = self._calculate_adaptive_commission_penalty(
            commission_this_step, trade_log
        )
        reward_components['commission'] = -commission_penalty * Decimal('0.1')
        
        # 3. 動態回撤管理（權重20%）
        drawdown_penalty = self._calculate_dynamic_drawdown_penalty(
            current_portfolio_value, positions_data
        )
        reward_components['drawdown'] = -drawdown_penalty * Decimal('0.2')
        
        # 4. 增強持倉激勵（權重15%）
        position_holding_reward = self._calculate_enhanced_position_holding_reward(
            positions_data, episode_step
        )
        reward_components['position_holding'] = position_holding_reward * Decimal('0.15')
        
        # 5. 市場趨勢感知（權重10%）
        trend_reward = self._calculate_market_trend_awareness_reward(
            positions_data, market_data
        )
        reward_components['trend_awareness'] = trend_reward * Decimal('0.1')
        
        # 6. 勝率激勵機制（權重5%）
        win_rate_bonus = self._calculate_win_rate_bonus(trade_log)
        reward_components['win_rate_bonus'] = win_rate_bonus * Decimal('0.05')
        
        # 總獎勵計算
        total_reward = sum(reward_components.values())
        
        # 記錄詳細組件用於分析
        self.reward_components_history.append({
            'step': episode_step,
            'portfolio_value': float(current_portfolio_value),
            'log_return': float(log_return),
            'components': {k: float(v) for k, v in reward_components.items()},
            'total_reward': float(total_reward)
        })
        
        return float(total_reward)
    
    def _calculate_log_return(self, current_value: Decimal, prev_value: Decimal) -> Decimal:
        """計算對數收益率"""
        if prev_value <= Decimal('0'):
            return Decimal('0.0')
        
        return (current_value / prev_value).ln()
    
    def _calculate_enhanced_risk_adjusted_reward(self) -> Decimal:
        """
        增強的風險調整收益計算（使用索提諾比率）
        """
        if len(self.returns_history) < 10:
            # 數據不足時使用簡單對數收益
            if self.returns_history:
                return self.reward_config["portfolio_log_return_factor"] * self.returns_history[-1]
            return Decimal('0.0')
        
        returns_array = list(self.returns_history)
        mean_return = sum(returns_array) / len(returns_array)
        excess_return = mean_return - self.risk_free_rate
        
        # 計算下行標準差（索提諾比率的關鍵）
        negative_returns = [r for r in returns_array if r < mean_return]
        if negative_returns:
            downside_variance = sum((r - mean_return) ** 2 for r in negative_returns) / len(negative_returns)
            downside_std = downside_variance.sqrt()
        else:
            downside_std = Decimal('1e-6')
        
        # 索提諾比率
        sortino_ratio = excess_return / (downside_std + Decimal('1e-6'))
        
        # 動態係數調整（根據收益穩定性）
        volatility = self._calculate_returns_volatility(returns_array)
        stability_factor = Decimal('1.0') / (Decimal('1.0') + volatility * Decimal('10'))
        
        dynamic_factor = min(max(stability_factor, Decimal('0.5')), Decimal('2.0'))
        
        return self.reward_config["risk_adjusted_return_factor"] * sortino_ratio * dynamic_factor
    
    def _calculate_returns_volatility(self, returns: List[Decimal]) -> Decimal:
        """計算收益率波動性"""
        if len(returns) < 2:
            return Decimal('0.0')
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return variance.sqrt()
    
    def _calculate_adaptive_commission_penalty(self, 
                                               commission: Decimal, 
                                               trade_log: List[Dict[str, Any]]) -> Decimal:
        """
        智能手續費管理
        根據最近交易表現動態調整手續費懲罰
        """
        base_penalty = commission / self.initial_capital
        
        # 分析最近交易表現
        recent_trades = trade_log[-10:] if len(trade_log) >= 10 else trade_log
        
        if len(recent_trades) < 3:
            return self.reward_config["commission_penalty_factor"] * base_penalty
        
        # 計算交易效益指標
        profitable_trades = [t for t in recent_trades if t.get('realized_pnl_ac', 0) > 0]
        total_profit = sum(t.get('realized_pnl_ac', 0) for t in profitable_trades)
        total_commission = sum(t.get('commission_ac', 0) for t in recent_trades)
        
        if total_commission > 0:
            profit_to_commission_ratio = total_profit / total_commission
            win_rate = len(profitable_trades) / len(recent_trades)
            
            # 動態調整懲罰係數
            if profit_to_commission_ratio > 5.0 and win_rate > 0.6:
                # 高效益交易，減少懲罰
                penalty_factor = Decimal('0.3')
            elif profit_to_commission_ratio > 2.0 and win_rate > 0.5:
                penalty_factor = Decimal('0.6')
            elif win_rate < 0.3 or profit_to_commission_ratio < 1.0:
                # 低效交易，增加懲罰
                penalty_factor = Decimal('2.0')
            else:
                penalty_factor = Decimal('1.0')
        else:
            penalty_factor = Decimal('1.0')
        
        return self.reward_config["commission_penalty_factor"] * penalty_factor * base_penalty
    
    def _calculate_dynamic_drawdown_penalty(self, 
                                            current_portfolio_value: Decimal,
                                            positions_data: Dict[str, Any]) -> Decimal:
        """
        動態回撤管理
        多層級回撤懲罰 + 回撤恢復獎勵
        """
        peak_value = positions_data.get('peak_portfolio_value', self.initial_capital)
        current_dd = (peak_value - current_portfolio_value) / (peak_value + Decimal('1e-9'))
        
        # 多層級懲罰
        if current_dd <= Decimal('0.02'):  # 2%以內
            penalty_factor = Decimal('0.3')
        elif current_dd <= Decimal('0.05'):  # 2-5%
            penalty_factor = Decimal('1.0')
        elif current_dd <= Decimal('0.10'):  # 5-10%
            penalty_factor = Decimal('2.5')
        else:  # 10%以上
            penalty_factor = Decimal('6.0')
        
        base_penalty = self.reward_config["max_drawdown_penalty_factor"] * current_dd * penalty_factor
        
        # 回撤恢復獎勵
        recovery_bonus = Decimal('0.0')
        previous_dd = positions_data.get('previous_drawdown', current_dd)
        
        if current_dd < previous_dd and previous_dd > Decimal('0.01'):
            recovery_ratio = (previous_dd - current_dd) / previous_dd
            recovery_bonus = Decimal('0.3') * recovery_ratio
            
        # 更新前一個回撤值
        positions_data['previous_drawdown'] = current_dd
        
        return max(base_penalty - recovery_bonus, Decimal('0.0'))
    
    def _calculate_enhanced_position_holding_reward(self, 
                                                    positions_data: Dict[str, Any],
                                                    episode_step: int) -> Decimal:
        """
        增強持倉激勵機制
        實現'讓利潤奔跑，快速止損'
        """
        total_reward = Decimal('0.0')
        
        positions = positions_data.get('positions', {})
        
        for symbol, position_info in positions.items():
            units = position_info.get('units', Decimal('0.0'))
            unrealized_pnl = position_info.get('unrealized_pnl', Decimal('0.0'))
            last_trade_step = position_info.get('last_trade_step', -1)
            
            if abs(units) <= Decimal('1e-9') or last_trade_step < 0:
                continue
                
            hold_duration = episode_step - last_trade_step
            pnl_ratio = unrealized_pnl / self.initial_capital
            
            if unrealized_pnl > Decimal('0'):
                # 盈利持倉：複利激勵
                duration_factor = min(Decimal(str(hold_duration)) / Decimal('15'), Decimal('4.0'))
                
                # 複利效應獎勵（非線性增長）
                compound_factor = (Decimal('1.08') ** min(hold_duration, 25)) - Decimal('1.0')
                
                # 趨勢持續獎勵
                trend_bonus = self._calculate_trend_continuation_bonus(
                    symbol, position_info, hold_duration
                )
                
                position_reward = (
                    pnl_ratio * duration_factor * 
                    (Decimal('1.0') + compound_factor * self.reward_config["compound_holding_factor"]) +
                    trend_bonus
                ) * self.reward_config["profit_target_bonus"]
                
                total_reward += position_reward
                
            else:
                # 虧損持倉：快速止損激勵
                loss_ratio = abs(pnl_ratio)
                
                if hold_duration <= 3 and loss_ratio <= Decimal('0.008'):
                    # 快速止損獎勵
                    quick_stop_bonus = self.reward_config["quick_stop_loss_bonus"] * (
                        Decimal('0.008') - loss_ratio
                    )
                    total_reward += quick_stop_bonus
                    
                elif hold_duration > 8 and loss_ratio > Decimal('0.015'):
                    # 長期虧損重懲罰
                    long_loss_penalty = (
                        loss_ratio * Decimal(str(hold_duration)) * 
                        self.reward_config["hold_penalty_factor"] * Decimal('10')
                    )
                    total_reward -= long_loss_penalty
        
        return total_reward
    
    def _calculate_trend_continuation_bonus(self, 
                                            symbol: str, 
                                            position_info: Dict[str, Any],
                                            hold_duration: int) -> Decimal:
        """計算趨勢持續獎勵"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
            return Decimal('0.0')
        
        recent_prices = list(self.price_history[symbol])[-10:]
        if len(recent_prices) < 5:
            return Decimal('0.0')
        
        # 計算趨勢強度
        trend_slope = self._calculate_trend_slope(recent_prices)
        units = position_info.get('units', Decimal('0.0'))
        
        if abs(units) <= Decimal('1e-9'):
            return Decimal('0.0')
        
        position_direction = Decimal('1.0') if units > 0 else Decimal('-1.0')
        trend_consistency = position_direction * Decimal(str(trend_slope))
        
        if trend_consistency > Decimal('0.001') and hold_duration > 5:
            # 順勢且持有時間合理
            return min(trend_consistency * Decimal('2.0'), Decimal('0.1'))
        
        return Decimal('0.0')
    
    def _calculate_trend_slope(self, prices: List[float]) -> float:
        """計算標準化價格趨勢斜率 (每1000單位時間)"""
        if len(prices) < 2:
            return 0.0
        
        try:
            import numpy as np
        except ImportError:
            # 簡化實現備選方案
            n = len(prices)
            x = list(range(n))
            x_mean = sum(x) / n
            y_mean = sum(prices) / n
            numerator = sum((x[i] - x_mean) * (prices[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
            slope = numerator / denominator if denominator != 0 else 0.0
            return slope * 1000  # 標準化到1000單位時間尺度
        else:
            x = np.arange(len(prices))
            y = np.array(prices)
            slope = (np.mean(x*y) - np.mean(x)*np.mean(y)) / (np.mean(x**2) - np.mean(x)**2)
            return slope * 1000  # 標準化到1000單位時間尺度
    
    def _calculate_market_trend_awareness_reward(self, 
                                                 positions_data: Dict[str, Any],
                                                 market_data: Dict[str, Any]) -> Decimal:
        """
        市場趨勢感知獎勵
        獎勵與市場趨勢一致的持倉
        """
        trend_reward = Decimal('0.0')
        positions = positions_data.get('positions', {})
        
        for symbol, position_info in positions.items():
            units = position_info.get('units', Decimal('0.0'))
            
            if abs(units) <= Decimal('1e-9'):
                continue
            
            # 更新價格歷史
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=30)
            
            current_price = market_data.get(symbol, {}).get('mid_price', 0.0)
            if current_price > 0:
                self.price_history[symbol].append(current_price)
            
            if len(self.price_history[symbol]) >= 10:
                # 計算多時間框架趨勢
                short_term_trend = self._calculate_trend_slope(
                    list(self.price_history[symbol])[-5:]
                )
                long_term_trend = self._calculate_trend_slope(
                    list(self.price_history[symbol])[-15:]
                )
                
                position_direction = Decimal('1.0') if units > 0 else Decimal('-1.0')
                
                # 短期趨勢一致性
                short_consistency = position_direction * Decimal(str(short_term_trend))
                long_consistency = position_direction * Decimal(str(long_term_trend))
                
                # 綜合趨勢獎勵
                if short_consistency > Decimal('0.001') and long_consistency > Decimal('0.0005'):
                    # 雙重趨勢確認
                    trend_reward += self.reward_config["trend_following_bonus"] * (
                        short_consistency * Decimal('0.7') + long_consistency * Decimal('0.3')
                    )
                elif short_consistency > Decimal('0.001'):
                    # 僅短期趨勢
                    trend_reward += self.reward_config["trend_following_bonus"] * short_consistency * Decimal('0.5')
                elif short_consistency < Decimal('-0.001'):
                    # 逆勢懲罰
                    trend_reward -= self.reward_config["trend_following_bonus"] * abs(short_consistency) * Decimal('0.3')
        
        return trend_reward
    
    def _calculate_win_rate_bonus(self, trade_log: List[Dict[str, Any]]) -> Decimal:
        """
        勝率激勵機制
        根據最近交易勝率給予獎勵或懲罰
        """
        if len(trade_log) < self.min_trades_for_win_rate:
            return Decimal('0.0')
        
        # 分析最近的已平倉交易
        recent_trades = trade_log[-self.recent_trades_window:]
        closed_trades = [
            t for t in recent_trades 
            if t.get('trade_type') in ['CLOSE', 'CLOSE_AND_REVERSE'] and 
               t.get('realized_pnl_ac', 0) != 0
        ]
        
        if len(closed_trades) < self.min_trades_for_win_rate:
            return Decimal('0.0')
        
        # 計算勝率和平均盈虧比
        wins = [t for t in closed_trades if t['realized_pnl_ac'] > 0]
        losses = [t for t in closed_trades if t['realized_pnl_ac'] <= 0]
        
        win_rate = len(wins) / len(closed_trades)
        
        # 計算平均盈虧比
        avg_win = sum(t['realized_pnl_ac'] for t in wins) / max(1, len(wins))
        avg_loss = abs(sum(t['realized_pnl_ac'] for t in losses)) / max(1, len(losses))
        profit_loss_ratio = avg_win / max(avg_loss, 1e-9)
        
        # 綜合評分（勝率 + 盈虧比）
        if win_rate >= 0.65 and profit_loss_ratio >= 1.2:
            return self.reward_config["win_rate_incentive_factor"] * Decimal('1.0')
        elif win_rate >= 0.55 and profit_loss_ratio >= 1.0:
            return self.reward_config["win_rate_incentive_factor"] * Decimal('0.5')
        elif win_rate >= 0.45:
            return Decimal('0.0')
        elif win_rate >= 0.35:
            return self.reward_config["win_rate_incentive_factor"] * Decimal('-0.3')
        else:
            return self.reward_config["win_rate_incentive_factor"] * Decimal('-0.8')
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """獲取詳細的性能指標"""
        if not self.reward_components_history:
            return {}
        
        recent_history = self.reward_components_history[-100:]  # 最近100步
        
        metrics = {
            'avg_total_reward': np.mean([h['total_reward'] for h in recent_history]),
            'reward_volatility': np.std([h['total_reward'] for h in recent_history]),
            'avg_risk_adjusted_reward': np.mean([
                h['components'].get('risk_adjusted', 0) for h in recent_history
            ]),
            'avg_position_holding_reward': np.mean([
                h['components'].get('position_holding', 0) for h in recent_history
            ]),
            'avg_trend_awareness_reward': np.mean([
                h['components'].get('trend_awareness', 0) for h in recent_history
            ]),
        }
        
        if len(self.returns_history) > 10:
            returns_array = [float(r) for r in self.returns_history]
            metrics.update({
                'sharpe_ratio': np.mean(returns_array) / (np.std(returns_array) + 1e-9),
                'max_drawdown_period': self._calculate_max_drawdown_period(),
                'win_rate_trend': self._calculate_win_rate_trend(),
            })
        
        return metrics
    
    def _calculate_max_drawdown_period(self) -> int:
        """計算最大回撤期間"""
        if not self.reward_components_history:
            return 0
        
        portfolio_values = [h['portfolio_value'] for h in self.reward_components_history]
        peak = portfolio_values[0]
        max_dd_period = 0
        current_dd_period = 0
        
        for value in portfolio_values:
            if value >= peak:
                peak = value
                current_dd_period = 0
            else:
                current_dd_period += 1
                max_dd_period = max(max_dd_period, current_dd_period)
        
        return max_dd_period
    
    def _calculate_win_rate_trend(self) -> float:
        """計算勝率趨勢"""
        if len(self.reward_components_history) < 50:
            return 0.0
        
        recent_win_rates = []
        for i in range(len(self.reward_components_history) - 25, len(self.reward_components_history)):
            if i >= 25:
                win_rate_bonus = self.reward_components_history[i]['components'].get('win_rate_bonus', 0)
                recent_win_rates.append(win_rate_bonus)
        
        if len(recent_win_rates) < 10:
            return 0.0
        
        # 計算線性趨勢
        x = list(range(len(recent_win_rates)))
        y = recent_win_rates
        
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def calculate_reward(self, reward_data: Dict[str, Any]) -> float:
        """
        統一的獎勵計算接口，適配交易環境的數據格式
        
        這個方法接收來自交易環境的獎勵數據字典，並轉換為
        增強版獎勵計算器所需的格式，然後調用增強版計算邏輯
        
        Args:
            reward_data: 包含交易環境狀態的字典，包含：
                - current_portfolio_value: 當前投資組合價值
                - previous_portfolio_value: 前一步投資組合價值  
                - commission_this_step: 本步驟手續費
                - trade_log: 交易記錄列表
                - unrealized_pnl_per_slot: 未實現盈虧數組
                - current_positions: 當前持倉數組
                - current_step: 當前步驟數
                - 等等...
                
        Returns:
            float: 計算得出的獎勵值
        """
        
        try:
            # 提取基本數據
            current_portfolio_value = reward_data.get('current_portfolio_value', self.initial_capital)
            prev_portfolio_value = reward_data.get('previous_portfolio_value', self.initial_capital)
            commission_this_step = reward_data.get('commission_this_step', Decimal('0.0'))
            
            # 確保數據類型正確
            if not isinstance(current_portfolio_value, Decimal):
                current_portfolio_value = Decimal(str(current_portfolio_value))
            if not isinstance(prev_portfolio_value, Decimal):
                prev_portfolio_value = Decimal(str(prev_portfolio_value))
            if not isinstance(commission_this_step, Decimal):
                commission_this_step = Decimal(str(commission_this_step))
            
            # 構建持倉數據
            positions_data = {
                'unrealized_pnl_per_slot': reward_data.get('unrealized_pnl_per_slot', []),
                'current_positions': reward_data.get('current_positions', []),
                'last_trade_steps': reward_data.get('last_trade_steps', []),
                'equity': reward_data.get('equity', current_portfolio_value),
                'total_margin_used': reward_data.get('total_margin_used', Decimal('0.0')),
                'active_slot_indices': reward_data.get('active_slot_indices', []),
                'peak_portfolio_value': reward_data.get('peak_portfolio_value', current_portfolio_value),
                'max_drawdown': reward_data.get('max_drawdown', Decimal('0.0'))
            }
            
            # 構建市場數據
            market_data = {
                'atr_values': reward_data.get('atr_values', []),
                'atr_penalty_threshold': reward_data.get('atr_penalty_threshold', self.atr_penalty_threshold)
            }
            
            # 獲取交易記錄和其他信息
            trade_log = reward_data.get('trade_log', [])
            episode_step = reward_data.get('current_step', 0)
            
            # 更新收益歷史（如果提供）
            returns_history = reward_data.get('returns_history', [])
            if returns_history:
                # 清空並重新填充收益歷史
                self.returns_history.clear()
                for ret in returns_history[-self.returns_history.maxlen:]:
                    if isinstance(ret, (int, float, Decimal)):
                        self.returns_history.append(Decimal(str(ret)))
            
            # 調用增強版獎勵計算
            enhanced_reward = self.calculate_enhanced_reward(
                current_portfolio_value=current_portfolio_value,
                prev_portfolio_value=prev_portfolio_value,
                commission_this_step=commission_this_step,
                trade_log=trade_log,
                positions_data=positions_data,
                market_data=market_data,
                episode_step=episode_step
            )
            
            # 保存最後的獎勵組件用於分析
            if hasattr(self, 'reward_components_history') and self.reward_components_history:
                self.last_reward_components = self.reward_components_history[-1].get('components', {})
            else:
                self.last_reward_components = {}
            
            return enhanced_reward
            
        except Exception as e:
            # 如果增強版計算失敗，返回基礎對數收益作為後備
            try:
                log_return = Decimal('0.0')
                if prev_portfolio_value > Decimal('0'):
                    log_return = (current_portfolio_value / prev_portfolio_value).ln()
                return float(log_return * self.reward_config.get("portfolio_log_return_factor", Decimal('1.0')))
            except:
                return 0.0
