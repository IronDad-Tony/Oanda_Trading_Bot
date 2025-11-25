"""
Enhanced Simulation - 增強的交易模擬功能
包括隔夜利息計算和市場休市處理
"""
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Optional, Tuple
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class SwapCalculator:
    """隔夜利息計算器"""
    
    # OANDA隔夜利息結算時間（美東時間17:00）
    ROLLOVER_HOUR_EST = 17
    
    # 簡化的利率差（實際應從API獲取）
    # 格式: {symbol: (long_rate, short_rate)} 單位: 年化百分比
    DEFAULT_FINANCING_RATES = {
        'EUR_USD': (0.0002, -0.0005),  # 做多每天0.02%, 做空-0.05%
        'GBP_USD': (0.0003, -0.0006),
        'USD_JPY': (-0.0004, 0.0001),
        'XAU_USD': (-0.0010, -0.0008),  # 黃金通常雙向負利率
        'NAS100_USD': (-0.0008, -0.0006),
        'US30_USD': (-0.0008, -0.0006),
    }
    
    def __init__(self, financing_rates: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        初始化
        
        Args:
            financing_rates: 自定義利率差字典
        """
        self.financing_rates = financing_rates or self.DEFAULT_FINANCING_RATES
    
    def calculate_swap(
        self,
        symbol: str,
        position_type: str,  # 'long' or 'short'
        units: int,
        entry_price: Decimal,
        days_held: int
    ) -> Decimal:
        """
        計算隔夜利息
        
        Args:
            symbol: 交易對
            position_type: 持倉類型
            units: 持倉單位
            entry_price: 入場價格
            days_held: 持有天數
        
        Returns:
            隔夜利息（以quote currency計）
        """
        if days_held == 0:
            return Decimal('0')
        
        if symbol not in self.financing_rates:
            logger.warning(f"No financing rate found for {symbol}, using 0")
            return Decimal('0')
        
        long_rate, short_rate = self.financing_rates[symbol]
        rate = long_rate if position_type == 'long' else short_rate
        
        # Swap = units * entry_price * rate * days_held
        notional = Decimal(str(abs(units))) * entry_price
        swap = notional * Decimal(str(rate)) * Decimal(str(days_held))
        
        return swap
    
    @staticmethod
    def has_crossed_rollover(prev_time: datetime, current_time: datetime) -> int:
        """
        檢測是否跨越了隔夜利息結算時間
        
        Args:
            prev_time: 上一個時間點
            current_time: 當前時間點
        
        Returns:
            跨越的天數
        """
        # 簡化處理：如果日期變化，則認為跨越了一天
        prev_date = prev_time.date()
        current_date = current_time.date()
        
        days_diff = (current_date - prev_date).days
        
        return max(0, days_diff)


class MarketHoursManager:
    """市場交易時間管理器"""
    
    # 外匯市場：週一00:00 - 週六00:00（UTC）
    # 週末休市
    
    @staticmethod
    def is_market_open(timestamp: datetime) -> bool:
        """
        檢查市場是否開放
        
        Args:
            timestamp: 時間戳
        
        Returns:
            是否開放
        """
        weekday = timestamp.weekday()
        
        # 週六（5）和週日（6）休市
        if weekday == 5:  # Saturday
            return False
        if weekday == 6:  # Sunday
            # 週日晚上可能開市，這裡簡化為全天休市
            return False
        
        # 其他時間開市
        return True
    
    @staticmethod
    def detect_market_closure(
        prev_time: datetime,
        current_time: datetime
    ) -> Tuple[bool, Optional[timedelta]]:
        """
        檢測是否跨越了市場休市期
        
        Args:
            prev_time: 上一個時間點
            current_time: 當前時間點
        
        Returns:
            (是否跨越休市, 休市時長)
        """
        time_diff = current_time - prev_time
        
        # 如果時間跨度超過3小時，可能是休市
        if time_diff > timedelta(hours=3):
            return True, time_diff
        
        # 檢查是否跨越週末
        if prev_time.weekday() < 5 and current_time.weekday() >= 5:
            return True, time_diff
        
        return False, None
    
    @staticmethod
    def add_market_closure_markers(df: pd.DataFrame, time_column: str = 'time') -> pd.DataFrame:
        """
        為數據添加市場休市標記
        
        Args:
            df: 數據DataFrame
            time_column: 時間列名稱
        
        Returns:
            添加了'is_market_closed'列的DataFrame
        """
        df = df.copy()
        df['is_market_closed'] = False
        
        if len(df) < 2:
            return df
        
        # 計算時間差
        time_diff = df[time_column].diff()
        
        # 超過3小時的間隔標記為休市
        closure_threshold = pd.Timedelta(hours=3)
        df.loc[time_diff > closure_threshold, 'is_market_closed'] = True
        
        # 標記週末
        df.loc[df[time_column].dt.weekday.isin([5, 6]), 'is_market_closed'] = True
        
        return df


class EnhancedSimulation:
    """增強的交易模擬"""
    
    def __init__(self):
        self.swap_calculator = SwapCalculator()
        self.market_hours = MarketHoursManager()
        
        # 持倉跟踪（用於計算持有天數）
        self.position_entry_times: Dict[str, datetime] = {}
    
    def on_position_opened(self, symbol: str, entry_time: datetime):
        """記錄開倉時間"""
        self.position_entry_times[symbol] = entry_time
    
    def on_position_closed(self, symbol: str):
        """清除開倉記錄"""
        if symbol in self.position_entry_times:
            del self.position_entry_times[symbol]
    
    def calculate_step_swap(
        self,
        symbol: str,
        position_type: str,
        units: int,
        entry_price: Decimal,
        prev_time: datetime,
        current_time: datetime
    ) -> Decimal:
        """
        計算本步驟的隔夜利息
        
        Returns:
            隔夜利息
        """
        days_crossed = SwapCalculator.has_crossed_rollover(prev_time, current_time)
        
        if days_crossed == 0:
            return Decimal('0')
        
        swap = self.swap_calculator.calculate_swap(
            symbol=symbol,
            position_type=position_type,
            units=units,
            entry_price=entry_price,
            days_held=days_crossed
        )
        
        logger.debug(f"Swap for {symbol}: {swap} ({days_crossed} days)")
        return swap
    
    def should_skip_step(
        self,
        prev_time: datetime,
        current_time: datetime
    ) -> Tuple[bool, Optional[str]]:
        """
        檢查是否應該跳過當前步驟（市場休市）
        
        Returns:
            (是否跳過, 原因)
        """
        if not self.market_hours.is_market_open(current_time):
            return True, "Market closed (weekend)"
        
        is_closure, closure_duration = self.market_hours.detect_market_closure(
            prev_time, current_time
        )
        
        if is_closure:
            return True, f"Market closure detected ({closure_duration})"
        
        return False, None
