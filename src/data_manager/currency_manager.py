# src/data_manager/currency_manager.py
"""
貨幣轉換管理器，封裝Oanda標準匯率轉換邏輯
"""
import logging
from decimal import Decimal
from typing import Dict, Tuple, Optional

logger = logging.getLogger("CurrencyManager")

class CurrencyManager:
    def __init__(self, account_currency: str):
        self.account_currency = account_currency.upper()
    
    def get_specific_rate(self, base_curr: str, quote_curr: str,
                         current_prices_map: Dict[str, Tuple[Decimal, Decimal]],
                         visited: Optional[set] = None, depth: int = 0) -> Optional[Decimal]:
        """
        獲取特定貨幣對的匯率，支持交叉匯率計算
        添加遞迴深度限制、循環檢測和多層中介貨幣支持
        """
        # 最大遞迴深度保護 (防止無限遞迴)
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
        
        # 嘗試直接貨幣對
        direct_pair = f"{base_upper}_{quote_upper}"
        if direct_pair in current_prices_map:
            bid, ask = current_prices_map[direct_pair]
            if ask > 0:  # 使用賣出價作為轉換率
                return ask
        
        # 嘗試反向貨幣對
        reverse_pair = f"{quote_upper}_{base_upper}"
        if reverse_pair in current_prices_map:
            bid, ask = current_prices_map[reverse_pair]
            if bid > 0:  # 使用買入價的倒數
                return Decimal('1.0') / bid
        
        # 嘗試通過中介貨幣轉換 (USD, EUR, GBP, JPY, AUD, CAD)
        intermediates = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD"]
        for intermediate in intermediates:
            # 跳過與查詢貨幣相同的仲介貨幣
            if intermediate == base_upper or intermediate == quote_upper:
                continue
                
            # 獲取基礎貨幣對中介貨幣的匯率
            base_to_intermediate = self.get_specific_rate(
                base_upper, intermediate, current_prices_map, visited.copy(), depth+1
            )
            
            # 獲取目標貨幣對中介貨幣的匯率
            quote_to_intermediate = self.get_specific_rate(
                quote_upper, intermediate, current_prices_map, visited.copy(), depth+1
            )
            
            if base_to_intermediate is not None and quote_to_intermediate is not None:
                if quote_to_intermediate != 0:
                    # 計算交叉匯率: (base/intermediate) / (quote/intermediate)
                    return base_to_intermediate / quote_to_intermediate
        
        logger.debug(f"無法找到 {base_upper}_{quote_upper} 的轉換路徑")
        return None

    def convert_to_account_currency(self, from_currency: str, 
                                   current_prices_map: Dict[str, Tuple[Decimal, Decimal]]) -> Decimal:
        """
        將任意貨幣轉換為賬戶貨幣
        """
        from_upper = from_currency.upper()
        
        # 相同貨幣直接返回1.0
        if from_upper == self.account_currency:
            return Decimal('1.0')
        
        # 1. 嘗試直接轉換
        direct_rate = self.get_specific_rate(from_upper, self.account_currency, current_prices_map)
        if direct_rate and direct_rate > 0:
            return direct_rate
        
        # 2. 嘗試反向轉換
        reverse_rate = self.get_specific_rate(self.account_currency, from_upper, current_prices_map)
        if reverse_rate and reverse_rate > 0:
            return Decimal('1.0') / reverse_rate
        
        # 3. 通過USD中轉
        if from_upper != "USD":
            usd_rate = self.get_specific_rate(from_upper, "USD", current_prices_map)
        else:
            usd_rate = Decimal('1.0')
            
        if self.account_currency != "USD":
            # 特別處理USD->AUD轉換
            if self.account_currency == "AUD":
                usd_aud = self.get_specific_rate("USD", "AUD", current_prices_map)
                if usd_aud and usd_aud > 0:
                    return usd_aud if from_upper == "USD" else usd_rate * usd_aud
                
                aud_usd = self.get_specific_rate("AUD", "USD", current_prices_map)
                if aud_usd and aud_usd > 0:
                    aud_rate = Decimal('1.0') / aud_usd
                    return aud_rate if from_upper == "USD" else usd_rate * aud_rate
            
            account_rate = self.get_specific_rate("USD", self.account_currency, current_prices_map)
        else:
            account_rate = Decimal('1.0')
            
        if usd_rate and account_rate and usd_rate > 0 and account_rate > 0:
            return usd_rate * account_rate
        
        # 4. 通過EUR中轉
        if from_upper != "EUR":
            eur_rate = self.get_specific_rate(from_upper, "EUR", current_prices_map)
        else:
            eur_rate = Decimal('1.0')
            
        if self.account_currency != "EUR":
            account_rate = self.get_specific_rate("EUR", self.account_currency, current_prices_map)
        else:
            account_rate = Decimal('1.0')
            
        if eur_rate and account_rate and eur_rate > 0 and account_rate > 0:
            return eur_rate * account_rate
        
        # 5. 嘗試其他主要貨幣中轉
        for intermediate in ["GBP", "JPY", "CHF", "CAD", "AUD"]:
            if intermediate == from_upper or intermediate == self.account_currency:
                continue
                
            to_intermediate = self.get_specific_rate(from_upper, intermediate, current_prices_map)
            from_intermediate = self.get_specific_rate(intermediate, self.account_currency, current_prices_map)
                
            if to_intermediate and from_intermediate and to_intermediate > 0 and from_intermediate > 0:
                return to_intermediate * from_intermediate
        
        # 最終回退
        available_pairs = ", ".join(current_prices_map.keys())
        logger.warning(f"無法轉換 {from_currency} 到 {self.account_currency}，可用貨幣對: [{available_pairs}]，使用安全值1.0")
        return Decimal('1.0')