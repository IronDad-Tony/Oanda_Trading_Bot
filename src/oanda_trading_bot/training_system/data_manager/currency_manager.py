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
    
    def _get_rate_via_usd(self, base_curr: str, quote_curr: str,
                           current_prices_map: Dict[str, Tuple[Decimal, Decimal]],
                           is_for_conversion: bool) -> Optional[Decimal]:
        """通过USD作为中介，计算交叉汇率"""
        if base_curr == "USD" or quote_curr == "USD":
            return None # Should be handled by direct/inverse lookup

        # 获取 base -> USD 的汇率
        base_to_usd_rate = self.get_specific_rate(
            base_curr, "USD", current_prices_map, is_for_conversion=is_for_conversion
        )
        if base_to_usd_rate is None:
            return None

        # 获取 USD -> quote 的汇率
        usd_to_quote_rate = self.get_specific_rate(
            "USD", quote_curr, current_prices_map, is_for_conversion=is_for_conversion
        )
        if usd_to_quote_rate is None:
            return None

        # 最终汇率是两者相乘
        return base_to_usd_rate * usd_to_quote_rate

    def get_specific_rate(self, base_curr: str, quote_curr: str,
                         current_prices_map: Dict[str, Tuple[Decimal, Decimal]],
                         is_for_conversion: bool = False) -> Optional[Decimal]:
        """
        获取特定货币对的汇率，支持直接、反向和通过USD的三角转换。
        """
        base_curr = self.normalize_symbol_format(base_curr)
        quote_curr = self.normalize_symbol_format(quote_curr)

        if base_curr == quote_curr:
            return Decimal('1.0')

        # 路径 1: 尝试直接货币对 (Base -> Quote)
        direct_pair = f"{base_curr}_{quote_curr}"
        if direct_pair in current_prices_map:
            bid, ask = current_prices_map[direct_pair]
            if is_for_conversion:
                midpoint = self.get_midpoint_rate(bid, ask)
                # 当我们卖出base_curr换取quote_curr时，我们得到较少的quote_curr
                return self.apply_conversion_fee(midpoint, is_credit=True) 
            else:
                return ask # From Oanda's perspective, this is the rate they offer

        # 路径 2: 尝试反向货币对 (Quote -> Base)
        reverse_pair = f"{quote_curr}_{base_curr}"
        if reverse_pair in current_prices_map:
            bid, ask = current_prices_map[reverse_pair]
            if bid > 0:
                if is_for_conversion:
                    midpoint = self.get_midpoint_rate(bid, ask)
                    # 当我们卖出quote_curr换取base_curr时，我们得到较少的base_curr
                    # 所以 1 / (rate) 会得到更多的 base_curr, 因此这里用 is_credit=False (debit)
                    adjusted_rate = self.apply_conversion_fee(midpoint, is_credit=False)
                    return Decimal('1.0') / adjusted_rate
                else:
                    return Decimal('1.0') / bid

        # 路径 3: 尝试通过USD进行三角转换
        triangular_rate = self._get_rate_via_usd(
            base_curr, quote_curr, current_prices_map, is_for_conversion
        )
        if triangular_rate is not None:
            return triangular_rate

        logger.debug(f"无法找到 {base_curr}_{quote_curr} 的转换路径")
        return None

    def convert_to_account_currency(self, from_currency: str, 
                                   current_prices_map: Dict[str, Tuple[Decimal, Decimal]], 
                                   is_credit: bool = True) -> Decimal:
        """
        将任意货币转换为账户货币（符合Oanda规则）。
        is_credit: True为获利转换，False为亏损转换。
        """
        from_currency = self.normalize_symbol_format(from_currency)
        
        if from_currency == self.account_currency:
            return Decimal('1.0')
        
        # is_for_conversion=True 会自动处理Oanda的markup
        rate = self.get_specific_rate(
            from_currency,
            self.account_currency,
            current_prices_map,
            is_for_conversion=True
        )
        
        if rate and rate > 0:
            # 在 get_specific_rate 中已经根据 is_for_conversion 应用了正确的 markup 逻辑
            # 此处无需再调用 apply_conversion_fee
            return rate
        
        available_pairs = ", ".join(current_prices_map.keys())
        logger.warning(f"无法转换 {from_currency} 到 {self.account_currency}，可用货币对: [{available_pairs}]，使用安全值1.0")
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
    """
    获取进行货币转换所需的额外货币对。
    该函数确保任何交易品种的任何货币都可以通过直接、反向或以USD为枢纽的三角转换路径，
    最终换算成账户本位币。
    """
    required_pairs = set()
    account_currency = account_currency.upper()
    
    # 收集所有涉及的独特货币
    currencies = set()
    for symbol in symbols:
        parts = symbol.split("_")
        if len(parts) == 2:
            currencies.add(parts[0])
            currencies.add(parts[1])

    # 对于每种货币，确保存在到账户货币的转换路径
    for currency in currencies:
        if currency == account_currency:
            continue

        # 路径 1 & 2: 直接或反向转换
        direct_pair = f"{currency}_{account_currency}"
        inverse_pair = f"{account_currency}_{currency}"

        if direct_pair in available_instruments:
            required_pairs.add(direct_pair)
            continue # 找到路径，无需进一步寻找
        if inverse_pair in available_instruments:
            required_pairs.add(inverse_pair)
            continue # 找到路径，无需进一步寻找

        # 路径 3: 三角转换 (通过 USD)
        # 如果货币本身不是USD，且账户货币也不是USD
        if currency != "USD" and account_currency != "USD":
            # 检查 Ccy -> USD 的路径
            path1_direct = f"{currency}_USD"
            path1_inverse = f"USD_{currency}"
            
            # 检查 USD -> AccountCurrency 的路径
            path2_direct = f"USD_{account_currency}"
            path2_inverse = f"{account_currency}_USD"

            # 确保两条路都通
            if (path1_direct in available_instruments or path1_inverse in available_instruments) and \
               (path2_direct in available_instruments or path2_inverse in available_instruments):
                
                # 将所有涉及的有效货币对加入列表
                if path1_direct in available_instruments:
                    required_pairs.add(path1_direct)
                if path1_inverse in available_instruments:
                    required_pairs.add(path1_inverse)
                if path2_direct in available_instruments:
                    required_pairs.add(path2_direct)
                if path2_inverse in available_instruments:
                    required_pairs.add(path2_inverse)
            else:
                logger.warning(f"無法為貨幣 {currency} 找到到帳戶貨幣 {account_currency} 的轉換路徑，即使通過USD也無法。")

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
    # Use the shared InstrumentInfoManager from common package
    from oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager
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
