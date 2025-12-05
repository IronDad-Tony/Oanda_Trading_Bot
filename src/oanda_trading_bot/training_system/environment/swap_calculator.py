"""
SwapCalculator - 隔夜利息計算器
模擬OANDA平台的Swap費用
"""
import logging
from decimal import Decimal
from datetime import datetime, time, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# OANDA結算時間 (美東時間 17:00 = UTC 22:00)
ROLLOVER_HOUR_UTC = 22


@dataclass
class SwapRate:
    """Swap費率"""
    symbol: str
    long_rate: float   # 做多隔夜利息 (年化)
    short_rate: float  # 做空隔夜利息 (年化)
    updated_at: datetime


class SwapCalculator:
    """
    隔夜利息計算器
    
    OANDA Swap計算規則:
    1. 每日美東時間17:00結算
    2. 週三計算3天（含週末）
    3. Swap = 持倉量 * 每日利率
    """
    
    # 默認Swap費率 (年化百分比)
    DEFAULT_SWAP_RATES = {
        "EUR_USD": SwapRate("EUR_USD", -0.0035, 0.0015, datetime.now()),
        "GBP_USD": SwapRate("GBP_USD", -0.0045, 0.0020, datetime.now()),
        "USD_JPY": SwapRate("USD_JPY", 0.0050, -0.0080, datetime.now()),
        "AUD_USD": SwapRate("AUD_USD", -0.0030, 0.0010, datetime.now()),
        "XAU_USD": SwapRate("XAU_USD", -0.0100, 0.0050, datetime.now()),
        "NAS100_USD": SwapRate("NAS100_USD", -0.0080, 0.0020, datetime.now()),
        "US30_USD": SwapRate("US30_USD", -0.0075, 0.0018, datetime.now()),
    }
    
    def __init__(self, swap_rates: Optional[Dict[str, SwapRate]] = None):
        self.swap_rates = swap_rates or self.DEFAULT_SWAP_RATES
        
    def calculate_swap(
        self,
        symbol: str,
        units: float,
        current_price: float,
        position_open_time: datetime,
        current_time: datetime,
        account_currency: str = "USD"
    ) -> Tuple[Decimal, int]:
        """
        計算隔夜利息
        
        Args:
            symbol: 交易品種
            units: 持倉量 (正=多, 負=空)
            current_price: 當前價格
            position_open_time: 開倉時間
            current_time: 當前時間
            account_currency: 帳戶貨幣
            
        Returns:
            (swap_amount, rollover_count)
        """
        # 計算經歷的結算次數
        rollover_count = self._count_rollovers(position_open_time, current_time)
        
        if rollover_count == 0:
            return Decimal("0"), 0
        
        # 獲取Swap費率
        swap_rate = self.swap_rates.get(symbol)
        if not swap_rate:
            logger.warning(f"No swap rate for {symbol}, using 0")
            return Decimal("0"), rollover_count
        
        # 選擇多/空費率
        is_long = units > 0
        annual_rate = swap_rate.long_rate if is_long else swap_rate.short_rate
        daily_rate = annual_rate / 365
        
        # 計算Swap
        position_value = abs(units) * current_price
        swap_per_day = position_value * daily_rate
        total_swap = swap_per_day * rollover_count
        
        # 轉換為帳戶貨幣 (簡化處理)
        swap_ac = Decimal(str(total_swap))
        
        logger.debug(
            f"Swap for {symbol}: {rollover_count} days, "
            f"rate={daily_rate:.6f}, amount={swap_ac:.2f}"
        )
        
        return swap_ac, rollover_count
    
    def _count_rollovers(
        self,
        open_time: datetime,
        current_time: datetime
    ) -> int:
        """計算經歷的結算次數"""
        if current_time <= open_time:
            return 0
        
        count = 0
        check_time = open_time
        
        while check_time < current_time:
            # 找到下一個結算時間
            next_rollover = self._next_rollover_time(check_time)
            
            if next_rollover <= current_time:
                # 週三計算3天
                if next_rollover.weekday() == 2:  # Wednesday
                    count += 3
                else:
                    count += 1
                check_time = next_rollover + timedelta(hours=1)
            else:
                break
        
        return count
    
    def _next_rollover_time(self, from_time: datetime) -> datetime:
        """計算下一個結算時間"""
        rollover_today = from_time.replace(
            hour=ROLLOVER_HOUR_UTC, 
            minute=0, 
            second=0, 
            microsecond=0
        )
        
        if from_time >= rollover_today:
            return rollover_today + timedelta(days=1)
        return rollover_today
    
    def update_swap_rates(self, rates: Dict[str, SwapRate]):
        """更新Swap費率"""
        self.swap_rates.update(rates)
        logger.info(f"Updated swap rates for {len(rates)} symbols")
    
    def get_daily_swap(
        self,
        symbol: str,
        units: float,
        current_price: float
    ) -> Decimal:
        """獲取每日Swap金額"""
        swap_rate = self.swap_rates.get(symbol)
        if not swap_rate:
            return Decimal("0")
        
        is_long = units > 0
        annual_rate = swap_rate.long_rate if is_long else swap_rate.short_rate
        daily_rate = annual_rate / 365
        
        position_value = abs(units) * current_price
        return Decimal(str(position_value * daily_rate))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    calc = SwapCalculator()
    
    # 測試：持有EUR_USD多頭3天
    open_time = datetime.now() - timedelta(days=3)
    swap, days = calc.calculate_swap(
        symbol="EUR_USD",
        units=10000,
        current_price=1.0850,
        position_open_time=open_time,
        current_time=datetime.now()
    )
    
    print(f"Swap: ${swap:.2f} for {days} days")
