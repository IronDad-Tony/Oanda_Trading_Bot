"""
MarketHoursManager - 市場休市時間管理器
處理週末、假日、市場休市等情況
"""
import logging
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MarketSession(Enum):
    """交易時段"""
    SYDNEY = "sydney"
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"
    CLOSED = "closed"


@dataclass
class MarketHours:
    """市場交易時間 (UTC)"""
    session: MarketSession
    open_hour: int
    close_hour: int


class MarketHoursManager:
    """
    市場交易時間管理器
    
    外匯市場特點：
    - 週一至週五 24小時交易
    - 週末休市 (週五 22:00 UTC - 週日 22:00 UTC)
    - 主要交易時段: 悉尼、東京、倫敦、紐約
    """
    
    # 主要交易時段 (UTC)
    SESSIONS = {
        MarketSession.SYDNEY: MarketHours(MarketSession.SYDNEY, 22, 7),
        MarketSession.TOKYO: MarketHours(MarketSession.TOKYO, 0, 9),
        MarketSession.LONDON: MarketHours(MarketSession.LONDON, 8, 17),
        MarketSession.NEW_YORK: MarketHours(MarketSession.NEW_YORK, 13, 22),
    }
    
    # 主要假日 (每年更新)
    HOLIDAYS_2025 = [
        datetime(2025, 1, 1),   # New Year
        datetime(2025, 12, 25), # Christmas
        datetime(2025, 12, 26), # Boxing Day
    ]
    
    def __init__(self, holidays: Optional[List[datetime]] = None):
        self.holidays = holidays or self.HOLIDAYS_2025
        
    def is_market_open(self, dt: datetime) -> bool:
        """檢查市場是否開放"""
        # 週末休市
        if self._is_weekend(dt):
            return False
        
        # 假日休市
        if self._is_holiday(dt):
            return False
        
        return True
    
    def _is_weekend(self, dt: datetime) -> bool:
        """
        檢查是否為週末
        週末定義: 週五 22:00 UTC 到 週日 22:00 UTC
        """
        weekday = dt.weekday()
        hour = dt.hour
        
        # 週六全天
        if weekday == 5:
            return True
        
        # 週五 22:00 後
        if weekday == 4 and hour >= 22:
            return True
        
        # 週日 22:00 前
        if weekday == 6 and hour < 22:
            return True
        
        # 週日 22:00 後 = 市場開放
        return False
    
    def _is_holiday(self, dt: datetime) -> bool:
        """檢查是否為假日"""
        date_only = dt.date()
        for holiday in self.holidays:
            if holiday.date() == date_only:
                return True
        return False
    
    def get_current_session(self, dt: datetime) -> MarketSession:
        """獲取當前交易時段"""
        if not self.is_market_open(dt):
            return MarketSession.CLOSED
        
        hour = dt.hour
        
        # 檢查各時段 (可重疊)
        active_sessions = []
        for session, hours in self.SESSIONS.items():
            if self._is_in_session(hour, hours.open_hour, hours.close_hour):
                active_sessions.append(session)
        
        # 返回主要時段
        if MarketSession.LONDON in active_sessions:
            return MarketSession.LONDON
        elif MarketSession.NEW_YORK in active_sessions:
            return MarketSession.NEW_YORK
        elif MarketSession.TOKYO in active_sessions:
            return MarketSession.TOKYO
        elif MarketSession.SYDNEY in active_sessions:
            return MarketSession.SYDNEY
        
        return MarketSession.CLOSED
    
    def _is_in_session(self, current_hour: int, open_hour: int, close_hour: int) -> bool:
        """檢查是否在交易時段內"""
        if open_hour < close_hour:
            return open_hour <= current_hour < close_hour
        else:  # 跨日 (如悉尼: 22 - 7)
            return current_hour >= open_hour or current_hour < close_hour
    
    def time_to_market_open(self, dt: datetime) -> timedelta:
        """計算距離市場開放的時間"""
        if self.is_market_open(dt):
            return timedelta(0)
        
        # 找到下一個開市時間
        next_open = self._next_market_open(dt)
        return next_open - dt
    
    def _next_market_open(self, dt: datetime) -> datetime:
        """計算下一個開市時間"""
        check = dt
        
        while not self.is_market_open(check):
            check += timedelta(hours=1)
            check = check.replace(minute=0, second=0, microsecond=0)
        
        return check
    
    def time_to_market_close(self, dt: datetime) -> timedelta:
        """計算距離市場休市的時間"""
        if not self.is_market_open(dt):
            return timedelta(0)
        
        # 找到下一個休市時間
        next_close = self._next_market_close(dt)
        return next_close - dt
    
    def _next_market_close(self, dt: datetime) -> datetime:
        """計算下一個休市時間"""
        # 週五 22:00 UTC
        days_until_friday = (4 - dt.weekday()) % 7
        if days_until_friday == 0 and dt.hour >= 22:
            days_until_friday = 7
        
        next_friday = dt + timedelta(days=days_until_friday)
        next_close = next_friday.replace(hour=22, minute=0, second=0, microsecond=0)
        
        return next_close
    
    def get_session_volatility_factor(self, session: MarketSession) -> float:
        """
        獲取時段波動性因子
        用於調整風險參數
        """
        factors = {
            MarketSession.SYDNEY: 0.6,
            MarketSession.TOKYO: 0.8,
            MarketSession.LONDON: 1.2,
            MarketSession.NEW_YORK: 1.0,
            MarketSession.CLOSED: 0.0,
        }
        return factors.get(session, 1.0)
    
    def detect_gap_risk(self, current_time: datetime) -> Tuple[bool, str]:
        """
        檢測跳空風險
        週末前30分鐘警告
        """
        if not self.is_market_open(current_time):
            return False, ""
        
        time_to_close = self.time_to_market_close(current_time)
        
        if time_to_close < timedelta(minutes=30):
            return True, f"Market closes in {time_to_close}. Weekend gap risk!"
        
        return False, ""


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    manager = MarketHoursManager()
    
    # 測試
    test_times = [
        datetime(2025, 12, 5, 10, 0),  # 週五 10:00 UTC
        datetime(2025, 12, 5, 23, 0),  # 週五 23:00 UTC (休市)
        datetime(2025, 12, 6, 12, 0),  # 週六
        datetime(2025, 12, 7, 23, 0),  # 週日 23:00 UTC (開市)
    ]
    
    for t in test_times:
        is_open = manager.is_market_open(t)
        session = manager.get_current_session(t)
        print(f"{t}: Open={is_open}, Session={session.value}")
