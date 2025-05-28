# src/data_manager/instrument_info_manager.py
"""
交易品種信息管理器
負責從OANDA API獲取並緩存所有可交易品種的詳細信息，
包括保證金率、最小交易單位、點值信息、報價貨幣等。
"""
import requests
from typing import Dict, Optional, List, Union # <--- 添加 Union
from decimal import Decimal, ROUND_HALF_UP # <--- 添加 ROUND_HALF_UP
import time
from functools import lru_cache
import sys # 確保導入
from pathlib import Path # 確保導入

# Flag to prevent duplicate import logging
_import_logged = False

try:
    from common.config import (
        OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_BASE_URL,
        OANDA_API_TIMEOUT_SECONDS
    )
    from common.logger_setup import logger
except ImportError:
    # project_root_iim = Path(__file__).resolve().parent.parent.parent # 移除
    # src_path_iim = project_root_iim / "src" # 移除
    # if str(project_root_iim) not in sys.path: # 移除
    #     sys.path.insert(0, str(project_root_iim)) # 移除
    try:
        # 假設 PYTHONPATH 已設定，這些導入應該能工作
        from src.common.config import (
            OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_BASE_URL,
            OANDA_API_TIMEOUT_SECONDS
        )
        from src.common.logger_setup import logger
        if not _import_logged:
            logger.info("Direct run InstrumentInfoManager: Successfully re-imported common modules.")
            _import_logged = True
    except ImportError as e_retry_iim:
        import logging
        logger = logging.getLogger("instrument_info_fallback") # type: ignore
        logger.setLevel(logging.INFO) # 至少INFO
        _ch_iim = logging.StreamHandler(sys.stdout)
        _ch_iim.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        if not logger.handlers: logger.addHandler(_ch_iim)
        logger.error(f"Direct run InstrumentInfoManager: Critical import error: {e_retry_iim}", exc_info=True)
        OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_BASE_URL, OANDA_API_TIMEOUT_SECONDS = None, None, None, 20


class InstrumentDetails:
    def __init__(self, symbol: str, display_name: str, type: str, margin_rate: Decimal,
                 minimum_trade_size: Decimal, trade_units_precision: int, pip_location: int,
                 quote_currency: str, base_currency: str, tags: List[Dict[str, str]] = None,
                 max_trade_units: Optional[Decimal] = None,
                 financing_long_rate: Optional[Decimal] = None,
                 financing_short_rate: Optional[Decimal] = None,
                 is_forex: bool = True): # is_forex 已添加
        self.symbol = symbol; self.display_name = display_name; self.type = type
        self.margin_rate = margin_rate; self.minimum_trade_size = minimum_trade_size
        self.trade_units_precision = trade_units_precision; self.pip_location = pip_location
        self.quote_currency = quote_currency.upper(); self.base_currency = base_currency.upper()
        self.tags = tags if tags is not None else []; self.max_trade_units = max_trade_units
        self.financing_long_rate = financing_long_rate; self.financing_short_rate = financing_short_rate
        self.is_forex = is_forex # 存儲 is_forex
        self.pip_value_in_quote_currency_per_unit = Decimal(str(10**pip_location))

    def round_units(self, units: Union[float, Decimal, str]) -> Decimal: # Union 已導入
        units_decimal = Decimal(str(units)); min_unit_d = self.minimum_trade_size
        if min_unit_d > 0:
            units_decimal = (units_decimal / min_unit_d).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * min_unit_d # ROUND_HALF_UP 已導入
        quantizer = Decimal('1e-' + str(self.trade_units_precision))
        return units_decimal.quantize(quantizer, rounding=ROUND_HALF_UP) # ROUND_HALF_UP 已導入
    def __repr__(self): return (f"InstrumentDetails(symbol='{self.symbol}', type='{self.type}', margin_rate={self.margin_rate:.4f}, min_trade_size={self.minimum_trade_size}, qc='{self.quote_currency}')")

class InstrumentInfoManager:
    _instance = None; _instrument_cache: Dict[str, InstrumentDetails] = {}
    _last_fetch_time: Optional[float] = None; _cache_expiry_seconds: int = 3600 * 6
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(InstrumentInfoManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    def __init__(self, force_refresh: bool = False):
        if hasattr(self, '_initialized') and self._initialized and not force_refresh: return
        if not OANDA_API_KEY or not OANDA_ACCOUNT_ID:
            msg = "OANDA_API_KEY 或 OANDA_ACCOUNT_ID 未配置。InstrumentInfoManager 無法工作。"
            logger.critical(msg); self._instrument_cache = {}; self._initialized = True; return
        self._api_session = requests.Session()
        self._api_session.headers.update({"Authorization": f"Bearer {OANDA_API_KEY}", "Content-Type": "application/json"})
        if force_refresh or self._is_cache_expired(): self._fetch_all_instruments_details()
        self._initialized = True
        logger.info(f"InstrumentInfoManager 初始化完成。緩存中包含 {len(self._instrument_cache)} 個交易品種信息。")
    def _is_cache_expired(self) -> bool:
        if self._last_fetch_time is None: return True
        return (time.time() - self._last_fetch_time) > self._cache_expiry_seconds
    
    # _fetch_all_instruments_details 應該是實例方法，以便訪問 self._api_session
    # 但由於緩存的是類變量，這裡的設計可以討論，暫時保持為實例方法，但更新類變量
    # @lru_cache(maxsize=1) # lru_cache 用於實例方法時，如果實例本身變了，緩存可能不共享
    #                     # 對於Singleton，這通常沒問題，但對於類級別的緩存，要小心
    #                     # 這裡我們用時間戳來控制緩存刷新，lru_cache可能不是最必要的
    def _fetch_all_instruments_details(self) -> None:
        endpoint = f"{OANDA_BASE_URL}/accounts/{OANDA_ACCOUNT_ID}/instruments"
        logger.info(f"正在從OANDA API獲取所有交易品種詳細信息: {endpoint}")
        temp_instrument_cache: Dict[str, InstrumentDetails] = {}
        try:
            response = self._api_session.get(endpoint, timeout=OANDA_API_TIMEOUT_SECONDS)
            response.raise_for_status()
            instruments_data = response.json().get("instruments", [])
            if not instruments_data: logger.warning("OANDA API 未返回任何交易品種信息。"); return
            for साधन_data in instruments_data:
                try:
                    symbol = साधन_data.get("name"); type_str = साधन_data.get("type", "UNKNOWN")
                    if not symbol: logger.warning(f"API返回的品種缺少'name'字段: {साधन_data}"); continue
                    base_c, quote_c = symbol.split("_") if "_" in symbol and len(symbol.split("_")) == 2 else (symbol, "USD")
                    details = InstrumentDetails(
                        symbol=symbol, display_name=साधन_data.get("displayName", symbol), type=type_str,
                        margin_rate=Decimal(str(साधन_data.get("marginRate", "0.05"))),
                        minimum_trade_size=Decimal(str(साधन_data.get("minimumTradeSize", "1"))),
                        trade_units_precision=int(साधन_data.get("tradeUnitsPrecision", 0)),
                        pip_location=int(साधन_data.get("pipLocation", -4 if "JPY" not in symbol.upper() else -2)),
                        quote_currency=quote_c, base_currency=base_c,
                        tags=साधन_data.get("tags"),
                        max_trade_units=Decimal(str(साधन_data.get("maximumTradeUnits", "0"))) if साधन_data.get("maximumTradeUnits") else None,
                        is_forex = type_str == "CURRENCY"
                    )
                    temp_instrument_cache[symbol] = details
                except Exception as e_parse: logger.warning(f"解析品種 {साधन_data.get('name', 'UnknownSymbol')} 的詳細信息時出錯: {e_parse}")
            InstrumentInfoManager._instrument_cache = temp_instrument_cache # 更新類級別的緩存
            InstrumentInfoManager._last_fetch_time = time.time()
            logger.info(f"成功從OANDA API獲取並緩存了 {len(InstrumentInfoManager._instrument_cache)} 個交易品種的詳細信息。")
        except requests.exceptions.RequestException as e_req: logger.error(f"從OANDA API獲取交易品種信息失敗 (RequestException): {e_req}", exc_info=True)
        except Exception as e_gen: logger.error(f"獲取交易品種信息過程中發生未知錯誤: {e_gen}", exc_info=True)

    def get_details(self, symbol: str) -> Optional[InstrumentDetails]:
        if self._is_cache_expired() and OANDA_API_KEY and OANDA_ACCOUNT_ID :
            logger.info("Instrument details cache expired or empty, attempting to refresh.")
            self._fetch_all_instruments_details() # 調用實例方法
        details = InstrumentInfoManager._instrument_cache.get(symbol.upper())
        if details is None: logger.warning(f"未能找到交易品種 '{symbol}' 的詳細信息。")
        return details
    def get_all_available_symbols(self) -> List[str]:
        if self._is_cache_expired() and OANDA_API_KEY and OANDA_ACCOUNT_ID: self._fetch_all_instruments_details()
        return list(InstrumentInfoManager._instrument_cache.keys())
    def get_details_for_multiple_symbols(self, symbols: List[str]) -> Dict[str, Optional[InstrumentDetails]]:
        if self._is_cache_expired() and OANDA_API_KEY and OANDA_ACCOUNT_ID: self._fetch_all_instruments_details()
        result = {};
        for sym in symbols: result[sym.upper()] = self.get_details(sym.upper())
        return result

if __name__ == "__main__":
    logger.info("正在直接運行 InstrumentInfoManager.py 進行測試...")
    if not OANDA_API_KEY or not OANDA_ACCOUNT_ID: # OANDA_API_KEY, OANDA_ACCOUNT_ID 從頂部導入
        logger.error("OANDA_API_KEY 或 OANDA_ACCOUNT_ID 未配置，無法執行測試。")
        sys.exit(1) # sys 已在頂部導入
    manager = InstrumentInfoManager(force_refresh=True)
    eur_usd_details = manager.get_details("EUR_USD")
    if eur_usd_details:
        logger.info(f"EUR_USD Details: {eur_usd_details}")
        logger.info(f"  EUR_USD Pip Value in {eur_usd_details.quote_currency}: {eur_usd_details.pip_value_in_quote_currency_per_unit}")
        logger.info(f"  Rounding 1234.56 units for EUR_USD: {eur_usd_details.round_units(1234.56)}")
    else: logger.warning("未能獲取 EUR_USD 的詳細信息。")
    xau_usd_details = manager.get_details("XAU_USD")
    if xau_usd_details: logger.info(f"XAU_USD Details: {xau_usd_details}")
    else: logger.warning("未能獲取 XAU_USD 的詳細信息。")
    all_symbols = manager.get_all_available_symbols()
    logger.info(f"從API獲取到的總品種數量: {len(all_symbols)}")
    if len(all_symbols) > 5: logger.info(f"部分品種示例: {all_symbols[:5]}")
    logger.info("第二次調用 get_details (應使用緩存)...")
    eur_usd_details_cached = manager.get_details("EUR_USD")
    if eur_usd_details_cached: logger.info(f"Cached EUR_USD Display Name: {eur_usd_details_cached.display_name}")
    non_existent_details = manager.get_details("NON_EXISTENT_SYMBOL_XYZ")
    if non_existent_details is None: logger.info("成功測試：不存在的品種返回 None。")
    logger.info("InstrumentInfoManager.py 測試執行完畢。")