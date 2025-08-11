# src/data_manager/currency_manager.py
"""
貨幣轉換管理器，封裝Oanda標準匯率轉換邏輯
"""
import logging
from decimal import Decimal
from typing import Dict, Tuple, Optional, List, Set

logger = logging.getLogger("CurrencyManager")

class CurrencyDependencyManager:
    def __init__(self, account_currency: str, apply_oanda_markup: bool = True):
        self.account_currency = account_currency.upper()
        self.apply_oanda_markup = apply_oanda_markup
        self.oanda_markup = Decimal('0.005')  # 0.5% markup
    
    @staticmethod
    def normalize_symbol_format(symbol: str) -> str:
        """
        將 symbol 格式正規化，將 'EUR/USD' 轉為 'EUR_USD'，確保與 Oanda 標準一致。
        """
        if isinstance(symbol, str) and "/" in symbol:
            return symbol.replace("/", "_").upper()
        return symbol.upper() if isinstance(symbol, str) else symbol

    def get_midpoint_rate(self, bid: Decimal, ask: Decimal) -> Decimal:
        """計算中間價格"""
        # 確保輸入是 Decimal 類型
        bid_decimal = Decimal(str(bid))
        ask_decimal = Decimal(str(ask))
        return (bid_decimal + ask_decimal) / Decimal('2')
    
    def apply_conversion_fee(self, midpoint: Decimal, is_credit: bool = True) -> Decimal:
        """
        應用Oanda貨幣轉換費用
        is_credit: True為獲利(credit)，False為虧損(debit)
        """
        if not self.apply_oanda_markup:
            return midpoint
            
        if is_credit:
            # 獲利時：midpoint × (1 - 0.5%)
            return midpoint * (Decimal('1') - self.oanda_markup)
        else:
            # 虧損時：midpoint × (1 + 0.5%)
            return midpoint * (Decimal('1') + self.oanda_markup)
    
    def get_specific_rate(self, base_curr: str, quote_curr: str,
                         current_prices_map: Dict[str, Tuple[Decimal, Decimal]],
                         visited: Optional[set] = None, depth: int = 0, 
                         is_for_conversion: bool = False) -> Optional[Decimal]:
        """
        獲取特定貨幣對的匯率，支持交叉匯率計算
        添加遞迴深度限制、循環檢測和多層中介貨幣支持
        """
        # 先正規化 symbol 格式，確保都是 Oanda 標準
        base_curr = self.normalize_symbol_format(base_curr)
        quote_curr = self.normalize_symbol_format(quote_curr)
        
        # 最大遞迴深度保護 (防止無限遞迴)
        MAX_DEPTH = 4
        if depth > MAX_DEPTH:
            logger.warning(f"達到最大遞迴深度 {MAX_DEPTH}，停止查找 {base_curr}/{quote_curr} 匯率")
            return None
            
        if visited is None:
            visited = set()
            
        pair = (base_curr, quote_curr)
        
        # 防止循環處理相同貨幣對
        if pair in visited:
            return None
        visited.add(pair)
          # 相同貨幣直接返回1.0
        if base_curr == quote_curr:
            return Decimal('1.0')
        
        # 嘗試直接貨幣對
        direct_pair = f"{base_curr}_{quote_curr}"
        if direct_pair in current_prices_map:
            bid, ask = current_prices_map[direct_pair]
            if ask > 0:
                if is_for_conversion and self.apply_oanda_markup:
                    # 貨幣轉換時使用中間價加markup
                    midpoint = self.get_midpoint_rate(bid, ask)
                    return self.apply_conversion_fee(midpoint, is_credit=True)
                else:
                    # 一般交易使用ask價格
                    return ask
        
        # 嘗試反向貨幣對
        reverse_pair = f"{quote_curr}_{base_curr}"
        if reverse_pair in current_prices_map:
            bid, ask = current_prices_map[reverse_pair]
            if bid > 0:
                if is_for_conversion and self.apply_oanda_markup:
                    # 貨幣轉換時使用中間價加markup
                    midpoint = self.get_midpoint_rate(bid, ask)
                    adjusted_rate = self.apply_conversion_fee(midpoint, is_credit=True)
                    return Decimal('1.0') / adjusted_rate
                else:
                    # 一般交易使用bid價格的倒數
                    return Decimal('1.0') / bid
        
        # 嘗試通過中介貨幣轉換 (USD, EUR, GBP, JPY, AUD, CAD)
        intermediates = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD"]
        for intermediate in intermediates:            # 跳過與查詢貨幣相同的仲介貨幣
            if intermediate == base_curr or intermediate == quote_curr:
                continue
                
            # 獲取基礎貨幣對中介貨幣的匯率
            base_to_intermediate = self.get_specific_rate(
                base_curr, intermediate, current_prices_map, visited.copy(), depth+1, is_for_conversion
            )
            
            # 獲取目標貨幣對中介貨幣的匯率
            quote_to_intermediate = self.get_specific_rate(
                quote_curr, intermediate, current_prices_map, visited.copy(), depth+1, is_for_conversion
            )
            
            if base_to_intermediate is not None and quote_to_intermediate is not None:
                if quote_to_intermediate != 0:                    # 計算交叉匯率: (base/intermediate) / (quote/intermediate)
                    return base_to_intermediate / quote_to_intermediate
        
        logger.debug(f"無法找到 {base_curr}_{quote_curr} 的轉換路徑")
        return None

    def convert_to_account_currency(self, from_currency: str, 
                                   current_prices_map: Dict[str, Tuple[Decimal, Decimal]], 
                                   is_credit: bool = True) -> Decimal:
        """
        將任意貨幣轉換為賬戶貨幣（符合Oanda規則）
        is_credit: True為獲利轉換，False為虧損轉換
        """
        # 先正規化 symbol 格式
        from_currency = self.normalize_symbol_format(from_currency)
        
        # 相同貨幣直接返回1.0
        if from_currency == self.account_currency:
            return Decimal('1.0')
        
        # 使用標記表示這是貨幣轉換，需要應用markup
        rate = self.get_specific_rate(
            from_currency,
            self.account_currency,
            current_prices_map,
            visited=None,
            depth=0,
            is_for_conversion=True
        )
        
        if rate and rate > 0:
            return rate
        
        # 如果直接轉換失敗，嘗試反向轉換
        reverse_rate = self.get_specific_rate(self.account_currency, from_currency, current_prices_map,
                                            visited=None, depth=0, is_for_conversion=True)
        if reverse_rate and reverse_rate > 0:
            return Decimal('1.0') / reverse_rate
        
        # 最終回退 - 記錄警告並返回1.0
        available_pairs = ", ".join(current_prices_map.keys())
        logger.warning(f"無法轉換 {from_currency} 到 {self.account_currency}，可用貨幣對: [{available_pairs}]，使用安全值1.0")
        return Decimal('1.0')
    
    def get_trading_rate(self, base_curr: str, quote_curr: str,
                        current_prices_map: Dict[str, Tuple[Decimal, Decimal]],
                        is_buy: bool = True) -> Optional[Decimal]:
        """
        獲取交易匯率（不含轉換費用）
        專用於交易執行時的匯率計算
        
        Args:
            base_curr: 基礎貨幣
            quote_curr: 報價貨幣
            current_prices_map: 當前價格映射
            is_buy: True為買入base貨幣，False為賣出base貨幣
              Returns:
            交易匯率，如果無法獲取則返回None
        """
        # 先正規化 symbol 格式
        base_curr = self.normalize_symbol_format(base_curr)
        quote_curr = self.normalize_symbol_format(quote_curr)
        
        return self.get_specific_rate(
            base_curr, quote_curr, current_prices_map,
            visited=None, depth=0, is_for_conversion=False
        )


def get_required_conversion_pairs(symbols: List[str], account_currency: str, available_instruments: Set[str]) -> Set[str]:
    """获取进行货币转换所需的额外货币对，只生成有效的货币对"""
    required_pairs = set()
    account_currency = account_currency.upper()
    
    # 始终包含基础货币对（如果存在）
    if f"USD_{account_currency}" in available_instruments:
        required_pairs.add(f"USD_{account_currency}")
    if f"{account_currency}_USD" in available_instruments:
        required_pairs.add(f"{account_currency}_USD")
    
    # 收集所有涉及的货币
    currencies = set()
    for symbol in symbols:
        parts = symbol.split("_")
        if len(parts) == 2:
            currencies.add(parts[0])
            currencies.add(parts[1])
    
    # 添加账户货币相关的货币对
    for currency in currencies:
        if currency == account_currency:
            continue
            
        # 直接货币对
        direct_pair = f"{currency}_{account_currency}"
        inverse_pair = f"{account_currency}_{currency}"
        
        # 只添加有效的货币对
        if direct_pair in available_instruments:
            required_pairs.add(direct_pair)
        if inverse_pair in available_instruments:
            required_pairs.add(inverse_pair)
        
        # 添加通过USD中转的货币对（如果有效）
        if currency != "USD" and account_currency != "USD":
            usd_pair1 = f"{currency}_USD"
            usd_pair2 = f"USD_{currency}"
            if usd_pair1 in available_instruments:
                required_pairs.add(usd_pair1)
            if usd_pair2 in available_instruments:
                required_pairs.add(usd_pair2)
    
    # 添加常见中转货币对（如果有效）
    for intermediate in ["EUR", "GBP", "JPY"]:
        if intermediate != account_currency:
            pair1 = f"{intermediate}_{account_currency}"
            pair2 = f"{account_currency}_{intermediate}"
            if pair1 in available_instruments:
                required_pairs.add(pair1)
            if pair2 in available_instruments:
                required_pairs.add(pair2)
    
    return required_pairs - set(symbols)

def ensure_currency_data_for_trading(trading_symbols: List[str], account_currency: str,
                                    start_time_iso: str, end_time_iso: str, granularity: str) -> tuple:
    """
    确保交易所需的所有货币数据已下载，包括汇率转换所需的额外货币对
    
    所有额外貨幣對將按照訓練symbols的標準下載完整的價量信息（包括開盤價、最高價、最低價、收盤價、成交量等），
    並存儲到數據庫中。這樣如果這些貨幣對後續被選為訓練symbol，就不需要重新下載。
    
    返回:
        (success: bool, all_symbols: set)
    """
    # 获取所有可用的交易品种
    from oanda_trading_bot.training_system.data_manager.instrument_info_manager import InstrumentInfoManager
    instrument_info_manager = InstrumentInfoManager()
    available_instruments = set(instrument_info_manager.get_all_available_symbols())
    
    # 获取所有需要的货币对
    required_pairs = get_required_conversion_pairs(trading_symbols, account_currency, available_instruments)
    all_symbols = set(trading_symbols) | required_pairs
    
    # 记录详细信息
    logger.info(f"确保货币数据可用: 交易品种={trading_symbols}, 账户货币={account_currency}")
    if required_pairs:
        logger.info(f"额外添加 {len(required_pairs)} 个汇率转换货币对: {required_pairs}")
        logger.info("这些货币对将按照训练symbols标准下载完整的价量信息并存储到数据库")
        logger.info("如果这些货币对后续被选为训练symbol，将直接使用已下载数据，无需重新下载")
    
    # 实际实现中调用数据下载器
    try:
        from oanda_trading_bot.training_system.data_manager.oanda_downloader import manage_data_download_for_symbols
        manage_data_download_for_symbols(
            symbols=list(all_symbols),
            overall_start_str=start_time_iso,
            overall_end_str=end_time_iso,
            granularity=granularity
        )
        logger.info(f"已确保所有 {len(all_symbols)} 个货币对数据下载完成")
        return True, all_symbols
    except ImportError as e:
        logger.error(f"无法导入数据下载器: {e}")
        return False, set(trading_symbols)
    except Exception as e:
        logger.error(f"下载货币数据时出错: {e}")
        return False, set(trading_symbols)