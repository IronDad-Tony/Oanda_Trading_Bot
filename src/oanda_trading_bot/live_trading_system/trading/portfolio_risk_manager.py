"""
PortfolioRiskManager - 組合風險管理器
實現VaR、CVaR、相關性分析等專業級風險指標
"""
import logging
import numpy as np
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """風險指標"""
    var_95: float           # 95% VaR
    var_99: float           # 99% VaR
    cvar_95: float          # 95% CVaR (Expected Shortfall)
    cvar_99: float          # 99% CVaR
    max_drawdown: float     # 最大回撤
    current_drawdown: float # 當前回撤
    sharpe_ratio: float     # 夏普比率
    sortino_ratio: float    # 索提諾比率
    beta: float             # Beta係數
    correlation_to_benchmark: float  # 與基準相關性


@dataclass
class PositionRisk:
    """單一持倉風險"""
    symbol: str
    units: float
    current_price: float
    entry_price: float
    unrealized_pnl: float
    position_var: float
    weight: float  # 持倉權重


class PortfolioRiskManager:
    """
    專業級組合風險管理器
    
    功能：
    1. VaR/CVaR計算 (歷史模擬法 & 參數法)
    2. 相關性矩陣分析
    3. 風險預算分配
    4. 動態風險調整
    """
    
    def __init__(
        self,
        historical_returns: Optional[Dict[str, np.ndarray]] = None,
        lookback_days: int = 252,
        risk_free_rate: float = 0.02,
        confidence_levels: List[float] = None
    ):
        self.historical_returns = historical_returns or {}
        self.lookback_days = lookback_days
        self.risk_free_rate = risk_free_rate
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        
        self._returns_cache: Dict[str, np.ndarray] = {}
        self._correlation_matrix: Optional[np.ndarray] = None
        
    def update_returns(self, symbol: str, returns: np.ndarray):
        """更新歷史收益率"""
        self.historical_returns[symbol] = returns
        self._correlation_matrix = None  # 清除快取
        
    def calculate_portfolio_var(
        self,
        positions: List[PositionRisk],
        method: str = "historical",
        horizon_days: int = 1
    ) -> Tuple[float, float]:
        """
        計算組合VaR
        
        Args:
            positions: 持倉列表
            method: 計算方法 ("historical", "parametric", "monte_carlo")
            horizon_days: 持有期
            
        Returns:
            (VaR_95, VaR_99)
        """
        if not positions:
            return 0.0, 0.0
            
        if method == "historical":
            return self._historical_var(positions, horizon_days)
        elif method == "parametric":
            return self._parametric_var(positions, horizon_days)
        else:
            return self._monte_carlo_var(positions, horizon_days)
    
    def _historical_var(
        self, 
        positions: List[PositionRisk],
        horizon_days: int
    ) -> Tuple[float, float]:
        """歷史模擬法VaR"""
        # 獲取所有持倉的歷史收益率
        portfolio_returns = self._calculate_portfolio_returns(positions)
        
        if len(portfolio_returns) < 30:
            logger.warning("Insufficient data for VaR calculation")
            return 0.0, 0.0
        
        # 計算不同置信度的VaR
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(horizon_days)
        var_99 = np.percentile(portfolio_returns, 1) * np.sqrt(horizon_days)
        
        # 轉換為正值（損失）
        return abs(var_95), abs(var_99)
    
    def _parametric_var(
        self,
        positions: List[PositionRisk],
        horizon_days: int
    ) -> Tuple[float, float]:
        """參數法VaR (假設正態分佈)"""
        portfolio_returns = self._calculate_portfolio_returns(positions)
        
        mu = np.mean(portfolio_returns)
        sigma = np.std(portfolio_returns)
        
        # 使用正態分佈分位數
        from scipy import stats
        z_95 = stats.norm.ppf(0.05)
        z_99 = stats.norm.ppf(0.01)
        
        var_95 = abs((mu + z_95 * sigma) * np.sqrt(horizon_days))
        var_99 = abs((mu + z_99 * sigma) * np.sqrt(horizon_days))
        
        return var_95, var_99
    
    def _monte_carlo_var(
        self,
        positions: List[PositionRisk],
        horizon_days: int,
        n_simulations: int = 10000
    ) -> Tuple[float, float]:
        """蒙特卡洛模擬VaR"""
        portfolio_returns = self._calculate_portfolio_returns(positions)
        
        mu = np.mean(portfolio_returns)
        sigma = np.std(portfolio_returns)
        
        # 模擬
        simulated_returns = np.random.normal(
            mu * horizon_days, 
            sigma * np.sqrt(horizon_days), 
            n_simulations
        )
        
        var_95 = abs(np.percentile(simulated_returns, 5))
        var_99 = abs(np.percentile(simulated_returns, 1))
        
        return var_95, var_99
    
    def calculate_cvar(
        self,
        positions: List[PositionRisk],
        confidence: float = 0.95
    ) -> float:
        """
        計算條件VaR (Expected Shortfall)
        
        CVaR = 超出VaR的預期損失
        """
        portfolio_returns = self._calculate_portfolio_returns(positions)
        
        percentile = (1 - confidence) * 100
        var = np.percentile(portfolio_returns, percentile)
        
        # CVaR是低於VaR的所有收益的平均值
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        
        return abs(cvar)
    
    def calculate_correlation_matrix(
        self,
        symbols: List[str]
    ) -> np.ndarray:
        """計算相關性矩陣"""
        n = len(symbols)
        corr_matrix = np.eye(n)
        
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i < j:
                    if sym1 in self.historical_returns and sym2 in self.historical_returns:
                        ret1 = self.historical_returns[sym1]
                        ret2 = self.historical_returns[sym2]
                        
                        # 對齊長度
                        min_len = min(len(ret1), len(ret2))
                        corr = np.corrcoef(ret1[-min_len:], ret2[-min_len:])[0, 1]
                        
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
        
        return corr_matrix
    
    def calculate_risk_metrics(
        self,
        positions: List[PositionRisk],
        benchmark_returns: Optional[np.ndarray] = None
    ) -> RiskMetrics:
        """計算完整風險指標"""
        portfolio_returns = self._calculate_portfolio_returns(positions)
        
        # VaR
        var_95, var_99 = self.calculate_portfolio_var(positions)
        
        # CVaR
        cvar_95 = self.calculate_cvar(positions, 0.95)
        cvar_99 = self.calculate_cvar(positions, 0.99)
        
        # 回撤
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdowns.min())
        current_drawdown = abs(drawdowns[-1]) if len(drawdowns) > 0 else 0
        
        # 夏普比率
        excess_returns = portfolio_returns - self.risk_free_rate / 252
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        # 索提諾比率
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.001
        sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)
        
        # Beta
        beta = 1.0
        corr = 0.0
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            min_len = min(len(portfolio_returns), len(benchmark_returns))
            cov = np.cov(portfolio_returns[-min_len:], benchmark_returns[-min_len:])
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0
            corr = np.corrcoef(portfolio_returns[-min_len:], benchmark_returns[-min_len:])[0, 1]
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            beta=beta,
            correlation_to_benchmark=corr
        )
    
    def _calculate_portfolio_returns(
        self,
        positions: List[PositionRisk]
    ) -> np.ndarray:
        """計算組合收益率"""
        if not positions:
            return np.array([0.0])
        
        # 計算權重
        total_value = sum(abs(p.units * p.current_price) for p in positions)
        
        weighted_returns = None
        
        for pos in positions:
            if pos.symbol in self.historical_returns:
                weight = abs(pos.units * pos.current_price) / total_value
                returns = self.historical_returns[pos.symbol]
                
                if weighted_returns is None:
                    weighted_returns = weight * returns
                else:
                    min_len = min(len(weighted_returns), len(returns))
                    weighted_returns = weighted_returns[-min_len:] + weight * returns[-min_len:]
        
        return weighted_returns if weighted_returns is not None else np.array([0.0])
    
    def should_reduce_exposure(
        self,
        current_var: float,
        var_limit: float,
        current_drawdown: float,
        max_allowed_drawdown: float
    ) -> Tuple[bool, str]:
        """判斷是否需要減少敞口"""
        if current_var > var_limit:
            return True, f"VaR ({current_var:.2%}) exceeds limit ({var_limit:.2%})"
        
        if current_drawdown > max_allowed_drawdown:
            return True, f"Drawdown ({current_drawdown:.2%}) exceeds limit ({max_allowed_drawdown:.2%})"
        
        return False, ""
    
    def get_risk_budget_allocation(
        self,
        symbols: List[str],
        total_risk_budget: float,
        method: str = "equal_risk"
    ) -> Dict[str, float]:
        """風險預算分配"""
        if method == "equal_risk":
            # 風險平價
            n = len(symbols)
            return {sym: total_risk_budget / n for sym in symbols}
        
        # TODO: 實現更複雜的風險預算方法
        return {sym: total_risk_budget / len(symbols) for sym in symbols}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 測試
    manager = PortfolioRiskManager()
    
    # 模擬數據
    manager.update_returns("EUR_USD", np.random.randn(252) * 0.01)
    manager.update_returns("GBP_USD", np.random.randn(252) * 0.012)
    
    positions = [
        PositionRisk("EUR_USD", 10000, 1.0850, 1.0800, 50, 0, 0.6),
        PositionRisk("GBP_USD", 5000, 1.2650, 1.2600, 25, 0, 0.4),
    ]
    
    metrics = manager.calculate_risk_metrics(positions)
    print(f"VaR 95%: {metrics.var_95:.4f}")
    print(f"CVaR 95%: {metrics.cvar_95:.4f}")
    print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
