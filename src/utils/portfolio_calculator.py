# src/utils/portfolio_calculator.py
"""
投資組合計算器
用於計算投資組合相關的指標和風險度量
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class PortfolioCalculator:
    """投資組合計算器類"""
    
    def __init__(self):
        """初始化投資組合計算器"""
        self.risk_free_rate = 0.02  # 無風險利率，預設2%
    
    def calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        """
        計算收益率
        
        Args:
            prices: 價格數組
            
        Returns:
            收益率數組
        """
        if len(prices) < 2:
            return np.array([])
        
        returns = np.diff(prices) / prices[:-1]
        return returns
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: Optional[float] = None) -> float:
        """
        計算夏普比率
        
        Args:
            returns: 收益率數組
            risk_free_rate: 無風險利率
            
        Returns:
            夏普比率
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # 日化無風險利率
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return float(sharpe)
    
    def calculate_max_drawdown(self, prices: np.ndarray) -> Dict[str, float]:
        """
        計算最大回撤
        
        Args:
            prices: 價格或淨值數組
            
        Returns:
            包含最大回撤信息的字典
        """
        if len(prices) == 0:
            return {'max_drawdown': 0.0, 'start_idx': 0, 'end_idx': 0}
        
        cumulative = np.cumprod(1 + self.calculate_returns(prices))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        max_drawdown = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        
        # 找到回撤開始的索引
        start_idx = 0
        for i in range(max_dd_idx, -1, -1):
            if drawdown[i] == 0:
                start_idx = i
                break
        
        return {
            'max_drawdown': float(abs(max_drawdown)),
            'start_idx': int(start_idx),
            'end_idx': int(max_dd_idx)
        }
    
    def calculate_volatility(self, returns: np.ndarray, annualized: bool = True) -> float:
        """
        計算波動率
        
        Args:
            returns: 收益率數組
            annualized: 是否年化
            
        Returns:
            波動率
        """
        if len(returns) == 0:
            return 0.0
        
        volatility = np.std(returns)
        
        if annualized:
            volatility *= np.sqrt(252)
        
        return float(volatility)
    
    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        計算在險價值(VaR)
        
        Args:
            returns: 收益率數組
            confidence_level: 置信水平
            
        Returns:
            VaR值
        """
        if len(returns) == 0:
            return 0.0
        
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return float(abs(var))
    
    def calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: Optional[float] = None) -> float:
        """
        計算索提諾比率
        
        Args:
            returns: 收益率數組
            risk_free_rate: 無風險利率
            
        Returns:
            索提諾比率
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        negative_returns = excess_returns[excess_returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')
        
        downside_deviation = np.sqrt(np.mean(negative_returns ** 2))
        
        if downside_deviation == 0:
            return 0.0
        
        sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(252)
        return float(sortino)
    
    def calculate_portfolio_metrics(self, prices: np.ndarray) -> Dict[str, float]:
        """
        計算投資組合的綜合指標
        
        Args:
            prices: 價格數組
            
        Returns:
            包含各種指標的字典
        """
        if len(prices) < 2:
            return {}
        
        returns = self.calculate_returns(prices)
        
        metrics = {
            'total_return': float((prices[-1] / prices[0] - 1) * 100),
            'annualized_return': float(np.mean(returns) * 252 * 100),
            'volatility': self.calculate_volatility(returns) * 100,
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(prices)['max_drawdown'] * 100,
            'var_95': self.calculate_var(returns) * 100,
            'calmar_ratio': 0.0
        }
        
        # 計算卡瑪比率
        if metrics['max_drawdown'] > 0:
            metrics['calmar_ratio'] = metrics['annualized_return'] / metrics['max_drawdown']
        
        return metrics
    
    def calculate_correlation_matrix(self, returns_matrix: np.ndarray) -> np.ndarray:
        """
        計算收益率相關性矩陣
        
        Args:
            returns_matrix: 收益率矩陣，每列代表一個資產
            
        Returns:
            相關性矩陣
        """
        if returns_matrix.size == 0:
            return np.array([])
        
        correlation_matrix = np.corrcoef(returns_matrix.T)
        return correlation_matrix
