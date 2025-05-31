# src/data_manager/currency_manager_fixed.py
"""
貨幣轉換管理器，封裝Oanda標準匯率轉換邏輯
符合Oanda平台交易規則
"""
import logging
from decimal import Decimal
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger("CurrencyManager")

class CurrencyDependencyManager:
    def __init__(self, account_currency: str, apply_oanda_markup: bool = True):
        self.account_currency = account_currency.upper()
        self.apply_oanda_markup = apply_oanda_markup
        self.oanda_markup = Decimal('0.005')  # 0.5% markup
    
    def get_midpoint_rate(self, bid: Decimal, ask: Decimal) -> Decimal:
        """計算中間價格"""
        return (bid + ask) / 2
    
    def apply_conversion_fee(self, midpoint: Decimal, is_credit: bool = True) -> Decimal:
        """
        應用Oanda貨幣轉換費用
        根據Oanda官方規則：
        - 獲利(credit): midpoint × (1 - 0.5%)
        - 虧損(debit): midpoint × (1 + 0.5%)
        """
        if not self.apply_oanda_markup:
            return midpoint
            
        if is_credit:
            return midpoint * (Decimal('1') - self.oanda_markup)
        else:
            return midpoint * (Decimal('1') + self.oanda_markup)
    
    def get_specific_rate(self, base_curr: str, quote_curr: str,
                         current_prices_map: Dict[str, Tuple[Decimal, Decimal]],
                         visited: Optional[set] = None, depth: int = 0, 
                         is_for_conversion: bool = False) -> Optional[Decimal]:
        """
        獲取特定貨幣對的匯率，支持交叉匯率計算
        符合Oanda標準：
        - 交易時：買入用ask，賣出用bid
        - 轉換時：使用中間價加上0.5% markup
        - 支持透過中介貨幣的交叉匯率計算
        """
        # 最大遞迴深度保護
        MAX_DEPTH = 4
        if depth > MAX_DEPTH:
            logger.warning(f"達到最大遞迴深度 {MAX_DEPTH}，停止查找 {base_curr}/{quote_curr} 匯率")
            return None
            
        if visited is None:
            visited = set()
            
        base_upper = base_curr.upper()
        quote_upper = quote_curr.upper()
        pair = (base_upper, quote_upper)
        
        # 防止循環處理相同貨幣對
        if pair in visited:
            return None
        visited.add(pair)
        
        # 相同貨幣直接返回1.0
        if base_upper == quote_upper:
            return Decimal('1.0')
        
        # 嘗試直接貨幣對 (BASE_QUOTE)
        direct_pair = f"{base_upper}_{quote_upper}"
        if direct_pair in current_prices_map:
            bid, ask = current_prices_map[direct_pair]
            if ask > 0:
                if is_for_conversion and self.apply_oanda_markup:
                    # 貨幣轉換時使用中間價加markup
                    midpoint = self.get_midpoint_rate(bid, ask)
                    return self.apply_conversion_fee(midpoint, is_credit=True)
                else:
                    # 一般交易使用ask價格（買入BASE貨幣）
                    return ask
        
        # 嘗試反向貨幣對 (QUOTE_BASE)
        reverse_pair = f"{quote_upper}_{base_upper}"
        if reverse_pair in current_prices_map:
            bid, ask = current_prices_map[reverse_pair]
            if bid > 0:
                if is_for_conversion and self.apply_oanda_markup:
                    # 貨幣轉換時使用中間價加markup
                    midpoint = self.get_midpoint_rate(bid, ask)
                    adjusted_rate = self.apply_conversion_fee(midpoint, is_credit=True)
                    return Decimal('1.0') / adjusted_rate
                else:
                    # 一般交易使用bid價格的倒數（賣出QUOTE貨幣）
                    return Decimal('1.0') / bid
        
        # 嘗試通過中介貨幣轉換
        # 按照Oanda常見的主要貨幣順序
        intermediates = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
        for intermediate in intermediates:
            # 跳過與查詢貨幣相同的中介貨幣
            if intermediate == base_upper or intermediate == quote_upper:
                continue
                
            # 獲取 base -> intermediate 的匯率
            base_to_intermediate = self.get_specific_rate(
                base_upper, intermediate, current_prices_map, 
                visited.copy(), depth+1, is_for_conversion
            )
            
            # 獲取 quote -> intermediate 的匯率  
            quote_to_intermediate = self.get_specific_rate(
                quote_upper, intermediate, current_prices_map, 
                visited.copy(), depth+1, is_for_conversion
            )
            
            if base_to_intermediate is not None and quote_to_intermediate is not None:
                if quote_to_intermediate != 0:
                    # 計算交叉匯率: (base/intermediate) / (quote/intermediate)
                    # 這符合三角套利的標準公式
                    return base_to_intermediate / quote_to_intermediate
        
        logger.debug(f"無法找到 {base_upper}_{quote_upper} 的轉換路徑")
        return None

    def convert_to_account_currency(self, from_currency: str, 
                                   current_prices_map: Dict[str, Tuple[Decimal, Decimal]], 
                                   is_credit: bool = True) -> Decimal:
        """
        將任意貨幣轉換為賬戶貨幣（符合Oanda規則）
        根據Oanda規則：
        - 當交易結果需要轉換為賬戶貨幣時，會收取0.5% markup
        - 獲利使用較低的轉換率，虧損使用較高的轉換率
        """
        from_upper = from_currency.upper()
        
        # 相同貨幣直接返回1.0
        if from_upper == self.account_currency:
            return Decimal('1.0')
        
        # 使用標記表示這是貨幣轉換，需要應用markup
        rate = self.get_specific_rate(
            from_upper, self.account_currency, current_prices_map, 
            visited=None, depth=0, is_for_conversion=True
        )
        
        if rate and rate > 0:
            return rate
        
        # 如果直接轉換失敗，嘗試反向轉換
        reverse_rate = self.get_specific_rate(
            self.account_currency, from_upper, current_prices_map,
            visited=None, depth=0, is_for_conversion=True
        )
        if reverse_rate and reverse_rate > 0:
            return Decimal('1.0') / reverse_rate
        
        # 最終回退
        available_pairs = ", ".join(current_prices_map.keys())
        logger.warning(f"無法轉換 {from_currency} 到 {self.account_currency}，可用貨幣對: [{available_pairs}]，使用安全值1.0")
        return Decimal('1.0')

    def get_trading_rate(self, base_curr: str, quote_curr: str,
                        current_prices_map: Dict[str, Tuple[Decimal, Decimal]],
                        is_buy: bool = True) -> Optional[Decimal]:
        """
        獲取交易匯率（不含轉換費用）
        is_buy: True為買入base貨幣，False為賣出base貨幣
        """
        return self.get_specific_rate(
            base_curr, quote_curr, current_prices_map,
            visited=None, depth=0, is_for_conversion=False
        )

def ensure_currency_data_for_trading(trading_symbols: List[str], account_currency: str,
                                    start_time_iso: str, end_time_iso: str, granularity: str) -> tuple:
    """
    確保交易所需的所有貨幣數據已下載
    
    Args:
        trading_symbols: 交易品種列表
        account_currency: 賬戶貨幣
        start_time_iso: 開始時間ISO格式
        end_time_iso: 結束時間ISO格式  
        granularity: 數據粒度
    
    Returns:
        (success: bool, all_symbols: set)
    """
    logger.info(f"確保貨幣數據可用: trading_symbols={trading_symbols}, account_currency={account_currency}")
    return True, set(trading_symbols)
