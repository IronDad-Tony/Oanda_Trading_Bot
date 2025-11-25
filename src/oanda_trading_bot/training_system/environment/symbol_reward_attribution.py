"""
Per-Symbol Reward Attribution - 品種級獎勵歸因系統
"""
from decimal import Decimal
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SymbolRewardAttribution:
    """品種級獎勵歸因"""
    
    def __init__(self, symbols: List[str], initial_capital: Decimal):
        """
        初始化
        
        Args:
            symbols: 交易品種列表
            initial_capital: 初始資金
        """
        self.symbols = symbols
        self.initial_capital = initial_capital
        
        # 記錄每個品種的貢獻歷史
        self.symbol_contributions: Dict[str, List[float]] = defaultdict(list)
        
        # 記錄每個品種的交易統計
        self.symbol_stats: Dict[str, Dict] = {
            symbol: {
                'total_trades': 0,
                'winning_trades': 0,
                'total_pnl': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0,
            }
            for symbol in symbols
        }
    
    def calculate_symbol_contributions(
        self,
        positions_data: Dict[str, Any],
        trade_log: List[Dict[str, Any]],
        current_step: int
    ) -> Dict[str, float]:
        """
        計算每個品種在當前步驟的貢獻分
        
        Args:
            positions_data: 持倉數據
            trade_log: 交易記錄
            current_step: 當前步驟
        
        Returns:
            {symbol: contribution_score}
        """
        contributions = {}
        
        for symbol in self.symbols:
            contribution = self._calculate_single_symbol_contribution(
                symbol=symbol,
                positions_data=positions_data,
                trade_log=trade_log,
                current_step=current_step
            )
            
            contributions[symbol] = contribution
            self.symbol_contributions[symbol].append(contribution)
        
        return contributions
    
    def _calculate_single_symbol_contribution(
        self,
        symbol: str,
        positions_data: Dict[str, Any],
        trade_log: List[Dict[str, Any]],
        current_step: int
    ) -> float:
        """
        計算單個品種的貢獻分
        
        貢獻分組成：
        1. 已實現P&L貢獻 (40%)
        2. 未實現P&L變化 (30%)  
        3. 風險控制質量 (20%)
        4. 交易效率 (10%)
        """
        contribution = 0.0
        
        # 1. 已實現P&L貢獻
        recent_trades = [
            t for t in trade_log
            if t.get('symbol') == symbol and t.get('step', 0) >= current_step - 10
        ]
        
        if recent_trades:
            realized_pnl = sum(float(t.get('pnl_ac', 0)) for t in recent_trades)
            realized_contribution = (realized_pnl / float(self.initial_capital)) * 100 * 0.4
            contribution += realized_contribution
            
            # 更新統計
            for trade in recent_trades:
                pnl = float(trade.get('pnl_ac', 0))
                self.symbol_stats[symbol]['total_trades'] += 1
                self.symbol_stats[symbol]['total_pnl'] += pnl
                if pnl > 0:
                    self.symbol_stats[symbol]['winning_trades'] += 1
                self.symbol_stats[symbol]['best_trade'] = max(
                    self.symbol_stats[symbol]['best_trade'], pnl
                )
                self.symbol_stats[symbol]['worst_trade'] = min(
                    self.symbol_stats[symbol]['worst_trade'], pnl
                )
        
        # 2. 未實現P&L變化
        position = positions_data.get(symbol)
        if position:
            current_unrealized = float(position.get('unrealized_pnl_ac', 0))
            prev_unrealized = float(position.get('prev_unrealized_pnl_ac', 0))
            unrealized_change = current_unrealized - prev_unrealized
            unrealized_contribution = (unrealized_change / float(self.initial_capital)) * 100 * 0.3
            contribution += unrealized_contribution
        
        # 3. 風險控制質量
        risk_score = self._calculate_risk_control_score(symbol, position)
        contribution += risk_score * 0.2
        
        # 4. 交易效率
        efficiency_score = self._calculate_efficiency_score(symbol, recent_trades)
        contribution += efficiency_score * 0.1
        
        return contribution
    
    def _calculate_risk_control_score(
        self,
        symbol: str,
        position: Optional[Dict]
    ) -> float:
        """
        計算風險控制評分
        
        Returns:
            0-10分
        """
        if not position:
            return 5.0  # 無持倉，中性分數
        
        score = 5.0
        
        # 檢查止損設置
        has_stop_loss = position.get('stop_loss') is not None
        if has_stop_loss:
            score += 2.0
            
            # 檢查止損距離是否合理
            current_price = float(position.get('current_price', 0))
            stop_loss = float(position.get('stop_loss', 0))
            entry_price = float(position.get('entry_price', 0))
            
            if current_price > 0 and stop_loss > 0:
                sl_distance = abs(current_price - stop_loss) / current_price
                
                # 理想止損距離：1-3%
                if 0.01 <= sl_distance <= 0.03:
                    score += 2.0
                elif sl_distance < 0.01:
                    score += 0.5  # 止損太緊
                elif sl_distance > 0.05:
                    score -= 1.0  # 止損太鬆
        
        # 檢查持倉時間
        time_in_position = position.get('time_in_position', 0)
        if 0 < time_in_position < 100:  # 合理的持倉時間
            score += 1.0
        elif time_in_position > 500:  # 持倉過久
            score -= 1.0
        
        return max(0, min(10, score))
    
    def _calculate_efficiency_score(
        self,
        symbol: str,
        recent_trades: List[Dict]
    ) -> float:
        """
        計算交易效率評分
        
        Returns:
            0-10分
        """
        if not recent_trades:
            return 5.0  # 無交易，中性分數
        
        score = 5.0
        
        # 勝率
        winning_trades = sum(1 for t in recent_trades if float(t.get('pnl_ac', 0)) > 0)
        win_rate = winning_trades / len(recent_trades) if recent_trades else 0
        
        if win_rate > 0.6:
            score += 3.0
        elif win_rate > 0.5:
            score += 2.0
        elif win_rate < 0.3:
            score -= 2.0
        
        # 盈虧比
        wins = [float(t.get('pnl_ac', 0)) for t in recent_trades if float(t.get('pnl_ac', 0)) > 0]
        losses = [abs(float(t.get('pnl_ac', 0))) for t in recent_trades if float(t.get('pnl_ac', 0)) < 0]
        
        if wins and losses:
            avg_win = np.mean(wins)
            avg_loss = np.mean(losses)
            profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            
            if profit_loss_ratio > 2.0:
                score += 2.0
            elif profit_loss_ratio > 1.5:
                score += 1.0
            elif profit_loss_ratio < 0.8:
                score -= 1.0
        
        return max(0, min(10, score))
    
    def get_symbol_feedback_features(
        self,
        contributions: Dict[str, float],
        window: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        生成品種反饋特徵（用於加入observation）
        
        Args:
            contributions: 當前貢獻分
            window: 歷史窗口大小
        
        Returns:
            {symbol: feedback_features}
        """
        feedback_features = {}
        
        for symbol in self.symbols:
            # 獲取最近的貢獻歷史
            history = self.symbol_contributions[symbol][-window:]
            
            if len(history) < window:
                # 填充0
                history = [0.0] * (window - len(history)) + history
            
            features = np.array([
                contributions.get(symbol, 0.0),  # 當前貢獻
                np.mean(history),  # 平均貢獻
                np.std(history),  # 貢獻波動性
                self.symbol_stats[symbol]['total_trades'],  # 總交易次數
                self.symbol_stats[symbol]['winning_trades'] / max(1, self.symbol_stats[symbol]['total_trades']),  # 勝率
            ], dtype=np.float32)
            
            feedback_features[symbol] = features
        
        return feedback_features
    
    def get_summary(self) -> Dict[str, Dict]:
        """獲取統計摘要"""
        summary = {}
        
        for symbol in self.symbols:
            stats = self.symbol_stats[symbol]
            contributions = self.symbol_contributions[symbol]
            
            summary[symbol] = {
                'total_trades': stats['total_trades'],
                'win_rate': stats['winning_trades'] / max(1, stats['total_trades']),
                'total_pnl': stats['total_pnl'],
                'avg_contribution': np.mean(contributions) if contributions else 0.0,
                'best_trade': stats['best_trade'],
                'worst_trade': stats['worst_trade'],
            }
        
        return summary
