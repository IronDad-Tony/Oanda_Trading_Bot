# src/system/risk_management.py
"""
風險控制系統 (Risk Management System)

此模組實現了完整的風險控制功能，包括：
1. 實時風險監控
2. 動態倉位管理  
3. 緊急停損機制
4. VaR 計算和風險評估
5. 最大回撤控制

主要類：
- RiskManagementSystem: 主要風險管理系統
- RealTimeRiskMonitor: 實時風險監控器
- DynamicPositionManager: 動態倉位管理器
- EmergencyStopLoss: 緊急停損機制
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json

# 添加項目根目錄到路徑
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.common.logger_setup import logger
    from src.agent.risk_management_system import MarketStateAwareness, StressTester
except ImportError as e:
    # 基礎日誌設定
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 創建空的替代類
    class MarketStateAwareness:
        pass
    
    class StressTester:
        pass
    
    logger.warning(f"導入錯誤，使用基礎配置: {e}")


class RiskLevel(Enum):
    """風險等級枚舉"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """警報類型枚舉"""
    POSITION_SIZE = "position_size"
    VAR_BREACH = "var_breach"
    DRAWDOWN = "drawdown"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"


@dataclass
class RiskMetrics:
    """風險指標數據結構"""
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    portfolio_volatility: float = 0.0
    portfolio_beta: float = 0.0
    sharpe_ratio: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PositionLimits:
    """倉位限制數據結構"""
    max_position_size: float = 0.1  # 最大倉位大小（佔組合比例）
    max_sector_exposure: float = 0.3  # 最大行業暴露
    max_single_asset_exposure: float = 0.05  # 最大單一資產暴露
    max_leverage: float = 2.0  # 最大槓桿倍數
    max_correlation: float = 0.8  # 最大相關性
    var_limit: float = 0.02  # VaR 限制
    max_drawdown_limit: float = 0.1  # 最大回撤限制


@dataclass
class RiskAlert:
    """風險警報數據結構"""
    alert_type: AlertType
    severity: RiskLevel
    message: str
    value: float
    threshold: float
    timestamp: datetime = None
    action_required: bool = True
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class RealTimeRiskMonitor:
    """
    實時風險監控器
    
    監控當前倉位風險、市場波動等
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化實時風險監控器
        
        Args:
            config: 監控配置
        """
        self.config = config or self._get_default_config()
        self.position_limits = PositionLimits(**self.config.get('position_limits', {}))
        self.market_state_monitor = MarketStateAwareness()
        
        # 歷史數據緩存
        self.price_history: Dict[str, List[float]] = {}
        self.returns_history: Dict[str, List[float]] = {}
        self.risk_metrics_history: List[RiskMetrics] = []
        
        logger.info("實時風險監控器已初始化")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """獲取默認配置"""
        return {
            'var_confidence_levels': [0.95, 0.99],
            'var_lookback_days': 252,
            'volatility_window': 20,
            'correlation_window': 60,
            'risk_check_frequency': 60,  # 秒
            'position_limits': {}
        }
    
    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        計算 VaR (Value at Risk)
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            
        Returns:
            VaR 值
        """
        if len(returns) < 10:
            return 0.0
        
        # 使用歷史模擬法計算 VaR
        sorted_returns = np.sort(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        var = -sorted_returns[index] if index < len(sorted_returns) else 0.0
        
        return max(var, 0.0)
    
    def calculate_expected_shortfall(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        計算條件風險價值 (Expected Shortfall / CVaR)
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            
        Returns:
            Expected Shortfall 值
        """
        if len(returns) < 10:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level)
        # 計算超過 VaR 的損失的平均值
        tail_losses = returns[returns <= -var]
        
        if len(tail_losses) > 0:
            return -np.mean(tail_losses)
        else:
            return var
    
    def calculate_max_drawdown(self, portfolio_values: np.ndarray) -> Tuple[float, float]:
        """
        計算最大回撤和當前回撤
        
        Args:
            portfolio_values: 組合價值序列
            
        Returns:
            (最大回撤, 當前回撤)
        """
        if len(portfolio_values) < 2:
            return 0.0, 0.0
        
        # 計算累計最高值
        peak = np.maximum.accumulate(portfolio_values)
        
        # 計算回撤
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        current_drawdown = drawdown[-1]
        
        return abs(max_drawdown), abs(current_drawdown)
    
    def calculate_portfolio_volatility(self, returns_matrix: np.ndarray, weights: np.ndarray) -> float:
        """
        計算組合波動率
        
        Args:
            returns_matrix: 資產收益率矩陣
            weights: 資產權重
            
        Returns:
            組合波動率
        """
        if returns_matrix.shape[0] < 10 or len(weights) == 0:
            return 0.0
        
        # 計算協方差矩陣
        cov_matrix = np.cov(returns_matrix.T)
        
        # 計算組合波動率
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        return portfolio_volatility
    
    def assess_risk_level(self, metrics: RiskMetrics) -> RiskLevel:
        """
        評估風險等級
        
        Args:
            metrics: 風險指標
            
        Returns:
            風險等級
        """
        risk_score = 0
        
        # VaR 評分
        if metrics.var_95 > self.position_limits.var_limit * 1.5:
            risk_score += 3
        elif metrics.var_95 > self.position_limits.var_limit:
            risk_score += 2
        elif metrics.var_95 > self.position_limits.var_limit * 0.5:
            risk_score += 1
        
        # 回撤評分
        if metrics.current_drawdown > self.position_limits.max_drawdown_limit * 1.5:
            risk_score += 3
        elif metrics.current_drawdown > self.position_limits.max_drawdown_limit:
            risk_score += 2
        elif metrics.current_drawdown > self.position_limits.max_drawdown_limit * 0.5:
            risk_score += 1
        
        # 波動率評分
        if metrics.portfolio_volatility > 0.3:
            risk_score += 2
        elif metrics.portfolio_volatility > 0.2:
            risk_score += 1
        
        # 根據總分確定風險等級
        if risk_score >= 6:
            return RiskLevel.CRITICAL
        elif risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def monitor_risk(self, portfolio_data: Dict[str, Any]) -> Tuple[RiskMetrics, List[RiskAlert]]:
        """
        執行風險監控
        
        Args:
            portfolio_data: 組合數據
            
        Returns:
            (風險指標, 風險警報列表)
        """
        alerts = []
        
        try:
            # 提取數據
            returns = portfolio_data.get('returns', np.array([]))
            portfolio_values = portfolio_data.get('portfolio_values', np.array([]))
            positions = portfolio_data.get('positions', {})
            
            # 計算風險指標
            var_95 = self.calculate_var(returns, 0.95)
            var_99 = self.calculate_var(returns, 0.99)
            expected_shortfall = self.calculate_expected_shortfall(returns, 0.95)
            max_drawdown, current_drawdown = self.calculate_max_drawdown(portfolio_values)
            
            # 計算組合波動率
            returns_matrix = portfolio_data.get('returns_matrix', np.array([]))
            weights = portfolio_data.get('weights', np.array([]))
            portfolio_volatility = self.calculate_portfolio_volatility(returns_matrix, weights)
            
            # 計算 Sharpe 比率
            sharpe_ratio = 0.0
            if portfolio_volatility > 0 and len(returns) > 0:
                avg_return = np.mean(returns)
                sharpe_ratio = avg_return / portfolio_volatility
            
            # 創建風險指標對象
            metrics = RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                portfolio_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio
            )
            
            # 評估風險等級
            metrics.risk_level = self.assess_risk_level(metrics)
            
            # 檢查警報條件
            alerts.extend(self._check_alerts(metrics, positions))
            
            # 保存歷史記錄
            self.risk_metrics_history.append(metrics)
            
            return metrics, alerts
            
        except Exception as e:
            logger.error(f"風險監控執行失敗: {e}")
            return RiskMetrics(), []
    
    def _check_alerts(self, metrics: RiskMetrics, positions: Dict[str, Any]) -> List[RiskAlert]:
        """檢查警報條件"""
        alerts = []
        
        # VaR 超限警報
        if metrics.var_95 > self.position_limits.var_limit:
            alerts.append(RiskAlert(
                alert_type=AlertType.VAR_BREACH,
                severity=RiskLevel.HIGH if metrics.var_95 > self.position_limits.var_limit * 1.5 else RiskLevel.MEDIUM,
                message=f"VaR 95% 超限: {metrics.var_95:.4f} > {self.position_limits.var_limit:.4f}",
                value=metrics.var_95,
                threshold=self.position_limits.var_limit
            ))
        
        # 回撤超限警報
        if metrics.current_drawdown > self.position_limits.max_drawdown_limit:
            alerts.append(RiskAlert(
                alert_type=AlertType.DRAWDOWN,
                severity=RiskLevel.CRITICAL if metrics.current_drawdown > self.position_limits.max_drawdown_limit * 1.5 else RiskLevel.HIGH,
                message=f"當前回撤超限: {metrics.current_drawdown:.4f} > {self.position_limits.max_drawdown_limit:.4f}",
                value=metrics.current_drawdown,
                threshold=self.position_limits.max_drawdown_limit
            ))
        
        # 倉位大小警報
        for asset, position in positions.items():
            position_size = abs(position.get('size', 0))
            if position_size > self.position_limits.max_single_asset_exposure:
                alerts.append(RiskAlert(
                    alert_type=AlertType.POSITION_SIZE,
                    severity=RiskLevel.MEDIUM,
                    message=f"資產 {asset} 倉位過大: {position_size:.4f} > {self.position_limits.max_single_asset_exposure:.4f}",
                    value=position_size,
                    threshold=self.position_limits.max_single_asset_exposure
                ))
        
        return alerts


class DynamicPositionManager:
    """
    動態倉位管理器
    
    根據風險評估和市場狀況調整倉位大小
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化動態倉位管理器
        
        Args:
            config: 倉位管理配置
        """
        self.config = config or self._get_default_config()
        self.position_limits = PositionLimits(**self.config.get('position_limits', {}))
        
        # 倉位調整歷史
        self.adjustment_history: List[Dict[str, Any]] = []
        
        logger.info("動態倉位管理器已初始化")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """獲取默認配置"""
        return {
            'volatility_target': 0.15,
            'risk_budget': 0.02,
            'rebalance_threshold': 0.05,
            'max_adjustment_per_day': 0.02,
            'position_limits': {}
        }
    
    def calculate_optimal_position_size(self, 
                                      asset: str,
                                      expected_return: float,
                                      volatility: float,
                                      current_portfolio_risk: float,
                                      risk_budget: float) -> float:
        """
        計算最優倉位大小
        
        Args:
            asset: 資產名稱
            expected_return: 期望收益率
            volatility: 波動率
            current_portfolio_risk: 當前組合風險
            risk_budget: 風險預算
            
        Returns:
            最優倉位大小
        """
        try:
            # 使用凱利公式的修正版本
            if volatility <= 0:
                return 0.0
            
            # 基礎凱利比例
            kelly_fraction = expected_return / (volatility ** 2)
            
            # 風險調整
            risk_adjustment = max(0.1, 1 - current_portfolio_risk / risk_budget)
            adjusted_size = kelly_fraction * risk_adjustment
            
            # 應用倉位限制
            max_size = self.position_limits.max_single_asset_exposure
            optimal_size = np.clip(adjusted_size, -max_size, max_size)
            
            return optimal_size
            
        except Exception as e:
            logger.error(f"計算最優倉位大小失敗: {e}")
            return 0.0
    
    def adjust_positions(self, 
                        current_positions: Dict[str, float],
                        risk_metrics: RiskMetrics,
                        market_conditions: Dict[str, Any]) -> Dict[str, float]:
        """
        調整倉位
        
        Args:
            current_positions: 當前倉位
            risk_metrics: 風險指標
            market_conditions: 市場條件
            
        Returns:
            調整後的倉位
        """
        adjusted_positions = current_positions.copy()
        
        try:
            # 根據風險等級調整
            if risk_metrics.risk_level == RiskLevel.CRITICAL:
                # 關鍵風險：大幅減倉
                adjustment_factor = 0.5
            elif risk_metrics.risk_level == RiskLevel.HIGH:
                # 高風險：適度減倉
                adjustment_factor = 0.7
            elif risk_metrics.risk_level == RiskLevel.MEDIUM:
                # 中等風險：小幅減倉
                adjustment_factor = 0.9
            else:
                # 低風險：正常倉位或適度加倉
                adjustment_factor = 1.0
            
            # 根據市場波動率調整
            market_volatility = market_conditions.get('volatility', 0.2)
            volatility_adjustment = self.config['volatility_target'] / max(market_volatility, 0.01)
            volatility_adjustment = np.clip(volatility_adjustment, 0.5, 1.5)
            
            # 綜合調整因子
            final_adjustment = adjustment_factor * volatility_adjustment
            
            # 應用調整
            total_adjustment = 0.0
            for asset, current_size in current_positions.items():
                new_size = current_size * final_adjustment
                
                # 檢查調整幅度限制
                max_adjustment = self.config['max_adjustment_per_day']
                size_change = abs(new_size - current_size)
                
                if size_change > max_adjustment:
                    # 限制調整幅度
                    direction = 1 if new_size > current_size else -1
                    new_size = current_size + direction * max_adjustment
                
                adjusted_positions[asset] = new_size
                total_adjustment += abs(new_size - current_size)
            
            # 記錄調整歷史
            adjustment_record = {
                'timestamp': datetime.now(),
                'risk_level': risk_metrics.risk_level.value,
                'adjustment_factor': final_adjustment,
                'total_adjustment': total_adjustment,
                'positions_before': current_positions.copy(),
                'positions_after': adjusted_positions.copy()
            }
            self.adjustment_history.append(adjustment_record)
            
            logger.info(f"倉位已調整，調整因子: {final_adjustment:.3f}, 總調整量: {total_adjustment:.4f}")
            
            return adjusted_positions
            
        except Exception as e:
            logger.error(f"倉位調整失敗: {e}")
            return current_positions


class EmergencyStopLoss:
    """
    緊急停損機制
    
    當達到預設的虧損閾值或偵測到極端市場事件時觸發停損
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化緊急停損機制
        
        Args:
            config: 停損配置
        """
        self.config = config or self._get_default_config()
        self.is_active = False
        self.trigger_history: List[Dict[str, Any]] = []
        self.stress_tester = StressTester()
        
        logger.info("緊急停損機制已初始化")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """獲取默認配置"""
        return {
            'max_portfolio_loss': 0.15,  # 最大組合損失
            'max_daily_loss': 0.05,     # 最大日損失
            'var_multiplier': 3.0,      # VaR 倍數觸發
            'volatility_spike_threshold': 2.0,  # 波動率跳躍閾值
            'correlation_threshold': 0.9,       # 相關性閾值
            'cooldown_period': 3600,    # 冷卻期（秒）
            'partial_liquidation_threshold': 0.8  # 部分平倉閾值
        }
    
    def check_trigger_conditions(self, 
                                risk_metrics: RiskMetrics,
                                portfolio_data: Dict[str, Any],
                                market_conditions: Dict[str, Any]) -> Tuple[bool, str, str]:
        """
        檢查觸發條件
        
        Args:
            risk_metrics: 風險指標
            portfolio_data: 組合數據
            market_conditions: 市場條件
            
        Returns:
            (是否觸發, 觸發原因, 建議行動)
        """
        try:
            # 檢查組合總損失
            portfolio_loss = portfolio_data.get('total_loss', 0.0)
            if portfolio_loss > self.config['max_portfolio_loss']:
                return True, f"組合總損失超限: {portfolio_loss:.4f}", "全部平倉"
            
            # 檢查日損失
            daily_loss = portfolio_data.get('daily_loss', 0.0)
            if daily_loss > self.config['max_daily_loss']:
                return True, f"日損失超限: {daily_loss:.4f}", "部分平倉"
            
            # 檢查 VaR 倍數
            var_threshold = risk_metrics.var_95 * self.config['var_multiplier']
            if portfolio_loss > var_threshold:
                return True, f"損失超過 VaR {self.config['var_multiplier']} 倍", "部分平倉"
            
            # 檢查波動率跳躍
            current_volatility = market_conditions.get('volatility', 0.0)
            historical_volatility = market_conditions.get('historical_volatility', 0.0)
            if historical_volatility > 0:
                volatility_ratio = current_volatility / historical_volatility
                if volatility_ratio > self.config['volatility_spike_threshold']:
                    return True, f"波動率跳躍: {volatility_ratio:.2f}x", "部分平倉"
            
            # 檢查資產相關性
            correlation = market_conditions.get('avg_correlation', 0.0)
            if correlation > self.config['correlation_threshold']:
                return True, f"資產相關性過高: {correlation:.4f}", "分散風險"
            
            # 檢查極端市場事件
            market_stress = market_conditions.get('stress_indicators', {})
            if self._detect_extreme_event(market_stress):
                return True, "偵測到極端市場事件", "全部平倉"
            
            return False, "", ""
            
        except Exception as e:
            logger.error(f"檢查觸發條件失敗: {e}")
            return False, "", ""
    
    def _detect_extreme_event(self, stress_indicators: Dict[str, Any]) -> bool:
        """偵測極端市場事件"""
        try:
            # 檢查流動性危機
            liquidity_score = stress_indicators.get('liquidity_score', 1.0)
            if liquidity_score < 0.3:
                return True
            
            # 檢查市場恐慌指標
            fear_index = stress_indicators.get('fear_index', 0.0)
            if fear_index > 80:
                return True
            
            # 檢查系統性風險
            systemic_risk = stress_indicators.get('systemic_risk', 0.0)
            if systemic_risk > 0.8:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"極端事件偵測失敗: {e}")
            return False
    
    def execute_emergency_action(self, 
                                action_type: str,
                                current_positions: Dict[str, float],
                                trigger_reason: str) -> Dict[str, float]:
        """
        執行緊急行動
        
        Args:
            action_type: 行動類型（全部平倉/部分平倉/分散風險）
            current_positions: 當前倉位
            trigger_reason: 觸發原因
            
        Returns:
            調整後的倉位
        """
        emergency_positions = current_positions.copy()
        
        try:
            if action_type == "全部平倉":
                # 清空所有倉位
                emergency_positions = {asset: 0.0 for asset in current_positions.keys()}
                self.is_active = True
                
            elif action_type == "部分平倉":
                # 減少倉位至安全水平
                reduction_factor = self.config['partial_liquidation_threshold']
                for asset in emergency_positions:
                    emergency_positions[asset] *= reduction_factor
                
            elif action_type == "分散風險":
                # 平衡倉位以降低相關性
                total_exposure = sum(abs(pos) for pos in current_positions.values())
                if total_exposure > 0:
                    target_size = total_exposure / len(current_positions)
                    for asset in emergency_positions:
                        current_size = abs(emergency_positions[asset])
                        if current_size > target_size:
                            direction = 1 if emergency_positions[asset] > 0 else -1
                            emergency_positions[asset] = direction * target_size
            
            # 記錄觸發歷史
            trigger_record = {
                'timestamp': datetime.now(),
                'trigger_reason': trigger_reason,
                'action_type': action_type,
                'positions_before': current_positions.copy(),
                'positions_after': emergency_positions.copy()
            }
            self.trigger_history.append(trigger_record)
            
            logger.warning(f"緊急停損觸發: {trigger_reason}, 執行: {action_type}")
            
            return emergency_positions
            
        except Exception as e:
            logger.error(f"執行緊急行動失敗: {e}")
            return current_positions


class RiskManagementSystem:
    """
    主要風險管理系統
    
    整合實時監控、動態倉位管理和緊急停損機制
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化風險管理系統
        
        Args:
            config: 系統配置
        """
        self.config = config or self._get_default_config()
        
        # 初始化子系統
        self.risk_monitor = RealTimeRiskMonitor(self.config.get('monitor_config', {}))
        self.position_manager = DynamicPositionManager(self.config.get('position_config', {}))
        self.emergency_stop = EmergencyStopLoss(self.config.get('emergency_config', {}))
        
        # 系統狀態
        self.is_active = True
        self.last_check_time = datetime.now()
        self.system_alerts: List[RiskAlert] = []
        
        logger.info("風險管理系統已初始化")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """獲取默認配置"""
        return {
            'check_frequency': 60,  # 檢查頻率（秒）
            'enable_real_time_monitoring': True,
            'enable_dynamic_position_management': True,
            'enable_emergency_stop_loss': True,
            'monitor_config': {},
            'position_config': {},
            'emergency_config': {}
        }
    
    def comprehensive_risk_check(self, 
                                portfolio_data: Dict[str, Any],
                                market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        執行綜合風險檢查
        
        Args:
            portfolio_data: 組合數據
            market_data: 市場數據
            
        Returns:
            風險檢查結果
        """
        results = {
            'timestamp': datetime.now(),
            'risk_metrics': None,
            'alerts': [],
            'position_adjustments': {},
            'emergency_actions': {},
            'recommendations': []
        }
        
        try:
            # 1. 實時風險監控
            if self.config['enable_real_time_monitoring']:
                risk_metrics, monitor_alerts = self.risk_monitor.monitor_risk(portfolio_data)
                results['risk_metrics'] = risk_metrics
                results['alerts'].extend(monitor_alerts)
            
            # 2. 動態倉位管理
            if self.config['enable_dynamic_position_management'] and results['risk_metrics']:
                current_positions = portfolio_data.get('positions', {})
                adjusted_positions = self.position_manager.adjust_positions(
                    current_positions, 
                    results['risk_metrics'],
                    market_data
                )
                results['position_adjustments'] = adjusted_positions
            
            # 3. 緊急停損檢查
            if self.config['enable_emergency_stop_loss'] and results['risk_metrics']:
                should_trigger, reason, action = self.emergency_stop.check_trigger_conditions(
                    results['risk_metrics'],
                    portfolio_data,
                    market_data
                )
                
                if should_trigger:
                    current_positions = portfolio_data.get('positions', {})
                    emergency_positions = self.emergency_stop.execute_emergency_action(
                        action, current_positions, reason
                    )
                    results['emergency_actions'] = {
                        'triggered': True,
                        'reason': reason,
                        'action': action,
                        'new_positions': emergency_positions
                    }
                else:
                    results['emergency_actions'] = {'triggered': False}
            
            # 4. 生成建議
            results['recommendations'] = self._generate_recommendations(results)
            
            # 更新系統狀態
            self.last_check_time = datetime.now()
            self.system_alerts.extend(results['alerts'])
            
            return results
            
        except Exception as e:
            logger.error(f"綜合風險檢查失敗: {e}")
            results['error'] = str(e)
            return results
    
    def _generate_recommendations(self, check_results: Dict[str, Any]) -> List[str]:
        """生成風險管理建議"""
        recommendations = []
        
        try:
            risk_metrics = check_results.get('risk_metrics')
            alerts = check_results.get('alerts', [])
            emergency_actions = check_results.get('emergency_actions', {})
            
            if risk_metrics:
                # 基於風險等級的建議
                if risk_metrics.risk_level == RiskLevel.CRITICAL:
                    recommendations.append("風險等級危急，建議立即減倉或清倉")
                elif risk_metrics.risk_level == RiskLevel.HIGH:
                    recommendations.append("風險等級較高，建議減少倉位並加強監控")
                elif risk_metrics.risk_level == RiskLevel.MEDIUM:
                    recommendations.append("風險等級中等，建議密切關注市場變化")
                
                # 基於具體指標的建議
                if risk_metrics.var_95 > 0.02:
                    recommendations.append(f"VaR 95% 偏高 ({risk_metrics.var_95:.4f})，建議降低投資組合風險")
                
                if risk_metrics.current_drawdown > 0.1:
                    recommendations.append(f"當前回撤較大 ({risk_metrics.current_drawdown:.4f})，建議檢視投資策略")
                
                if risk_metrics.sharpe_ratio < 0.5:
                    recommendations.append(f"夏普比率偏低 ({risk_metrics.sharpe_ratio:.2f})，建議優化風險收益比")
            
            # 基於警報的建議
            for alert in alerts:
                if alert.alert_type == AlertType.VAR_BREACH:
                    recommendations.append("VaR 超限，建議立即調整倉位規模")
                elif alert.alert_type == AlertType.DRAWDOWN:
                    recommendations.append("回撤超限，建議檢討停損策略")
                elif alert.alert_type == AlertType.POSITION_SIZE:
                    recommendations.append("個別倉位過大，建議分散投資")
            
            # 基於緊急行動的建議
            if emergency_actions.get('triggered'):
                recommendations.append(f"緊急停損已觸發：{emergency_actions.get('reason')}")
                recommendations.append("建議暫停交易，重新評估風險管理策略")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"生成建議失敗: {e}")
            return ["系統生成建議時發生錯誤，請手動檢查風險狀況"]
    
    def get_system_status(self) -> Dict[str, Any]:
        """獲取系統狀態"""
        return {
            'is_active': self.is_active,
            'last_check_time': self.last_check_time,
            'total_alerts': len(self.system_alerts),
            'emergency_stop_active': self.emergency_stop.is_active,
            'recent_alerts': self.system_alerts[-5:] if self.system_alerts else [],
            'subsystem_status': {
                'risk_monitor': len(self.risk_monitor.risk_metrics_history),
                'position_manager': len(self.position_manager.adjustment_history),
                'emergency_stop': len(self.emergency_stop.trigger_history)
            }
        }


def test_risk_management_system():
    """測試風險管理系統"""
    print("測試風險管理系統...")
    
    # 創建測試配置
    config = {
        'check_frequency': 1,
        'monitor_config': {
            'var_confidence_levels': [0.95, 0.99],
            'position_limits': {
                'max_single_asset_exposure': 0.1,
                'var_limit': 0.02,
                'max_drawdown_limit': 0.15
            }
        }
    }
    
    # 初始化系統
    risk_system = RiskManagementSystem(config)
    
    # 創建測試數據
    portfolio_data = {
        'returns': np.random.normal(-0.001, 0.02, 100),  # 略微負的收益率
        'portfolio_values': np.cumprod(1 + np.random.normal(-0.001, 0.02, 100)) * 100000,
        'positions': {
            'EURUSD': 0.05,
            'GBPUSD': 0.08,
            'USDJPY': -0.03
        },
        'total_loss': 0.12,  # 模擬較大損失
        'daily_loss': 0.04,
        'returns_matrix': np.random.normal(0, 0.02, (100, 3)),
        'weights': np.array([0.4, 0.4, 0.2])
    }
    
    market_data = {
        'volatility': 0.25,
        'historical_volatility': 0.15,
        'avg_correlation': 0.75,
        'stress_indicators': {
            'liquidity_score': 0.6,
            'fear_index': 65,
            'systemic_risk': 0.4
        }
    }
    
    # 執行風險檢查
    results = risk_system.comprehensive_risk_check(portfolio_data, market_data)
    
    # 輸出結果
    print(f"風險等級: {results['risk_metrics'].risk_level.value}")
    print(f"VaR 95%: {results['risk_metrics'].var_95:.4f}")
    print(f"當前回撤: {results['risk_metrics'].current_drawdown:.4f}")
    print(f"警報數量: {len(results['alerts'])}")
    print(f"緊急行動觸發: {results['emergency_actions']['triggered']}")
    print(f"建議數量: {len(results['recommendations'])}")
    
    # 顯示具體建議
    print("\n風險管理建議:")
    for i, recommendation in enumerate(results['recommendations'][:3], 1):
        print(f"{i}. {recommendation}")
    
    print("\n風險管理系統測試完成!")


if __name__ == "__main__":
    test_risk_management_system()
