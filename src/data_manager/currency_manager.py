# src/data_manager/currency_manager.py
"""
貨幣管理器 - 智能匯率對依賴管理
根據交易symbols自動判斷並下載所需的匯率對數據，完全模擬OANDA V20帳戶的匯率轉換邏輯
"""

import logging
import sys
from pathlib import Path
from typing import List, Set, Dict, Optional, Tuple
from datetime import datetime, timezone

# 導入邏輯與其他模組相同
logger: logging.Logger = logging.getLogger("currency_manager_module_init")
_logger_initialized_by_common_cm = False

try:
    from common.logger_setup import logger as common_configured_logger
    logger = common_configured_logger
    _logger_initialized_by_common_cm = True
    logger.debug("currency_manager.py: Successfully imported logger from common.logger_setup.")
    
    from common.config import ACCOUNT_CURRENCY as _ACCOUNT_CURRENCY
    logger.info("currency_manager.py: Successfully imported common.config values.")
    
    from data_manager.oanda_downloader import manage_data_download_for_symbols, format_datetime_for_oanda
    from data_manager.instrument_info_manager import InstrumentInfoManager
    logger.info("currency_manager.py: Successfully imported other dependencies.")
    
except ImportError as e_initial_import_cm:
    logger_temp_cm = logging.getLogger("currency_manager_fallback_initial")
    logger_temp_cm.addHandler(logging.StreamHandler(sys.stdout))
    logger_temp_cm.setLevel(logging.DEBUG)
    logger = logger_temp_cm
    logger.warning(f"currency_manager.py: Initial import failed: {e_initial_import_cm}. Assuming PYTHONPATH is set correctly or this is a critical issue.")
    
    try:
        # 假設 PYTHONPATH 已設定，這些導入應該能工作
        from src.common.logger_setup import logger as common_logger_retry_cm
        logger = common_logger_retry_cm
        _logger_initialized_by_common_cm = True
        logger.info("currency_manager.py: Successfully re-imported common_logger after path adj.")
        
        from src.common.config import ACCOUNT_CURRENCY as _ACCOUNT_CURRENCY
        logger.info("currency_manager.py: Successfully re-imported common.config after path adjustment.")
        
        from src.data_manager.oanda_downloader import manage_data_download_for_symbols, format_datetime_for_oanda
        from src.data_manager.instrument_info_manager import InstrumentInfoManager
        logger.info("currency_manager.py: Successfully re-imported other dependencies after path adjustment.")
        
    except ImportError as e_retry_critical_cm:
        logger.error(f"currency_manager.py: Critical import error after path adjustment: {e_retry_critical_cm}", exc_info=True)
        logger.warning("currency_manager.py: Using fallback values for config (critical error during import).")
        _ACCOUNT_CURRENCY = "AUD"
        
        # 創建後備函數
        def manage_data_download_for_symbols(*args, **kwargs):
            logger.error("Downloader not available in fallback.")
        
        def format_datetime_for_oanda(dt):
            return dt.isoformat()
        
        class InstrumentInfoManager:
            def __init__(self, **kwargs):
                pass
            def get_details(self, symbol):
                return None

ACCOUNT_CURRENCY = _ACCOUNT_CURRENCY


class CurrencyDependencyManager:
    """
    貨幣依賴管理器
    
    根據OANDA V20的實際交易邏輯，智能判斷並管理匯率對依賴：
    1. 分析交易symbols的基礎貨幣和報價貨幣
    2. 根據帳戶貨幣確定需要的匯率對
    3. 自動下載缺失的匯率對數據
    4. 支援直接匯率和交叉匯率的轉換路徑
    """
    
    def __init__(self, account_currency: str = ACCOUNT_CURRENCY):
        self.account_currency = account_currency.upper()
        self.instrument_manager = InstrumentInfoManager(force_refresh=False)
        
        # OANDA主要貨幣列表 (用於優化匯率路徑 - 此處保留，儘管在 determine_required_currency_pairs 中直接使用它的部分被簡化)
        self.major_currencies = {
            'USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD'
        }
        
        # 常見的OANDA匯率對格式
        self.common_pairs_templates = [
            '{base}_{quote}',  # 直接對
            '{quote}_{base}'   # 反向對
        ]
        
        logger.info(f"CurrencyDependencyManager 初始化完成，帳戶貨幣: {self.account_currency}")
    
    def parse_symbol_currencies(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """
        解析交易symbol的基礎貨幣和報價貨幣
        
        Args:
            symbol: 交易對象名稱，如 'EUR_USD', 'XAU_USD' 等
            
        Returns:
            (base_currency, quote_currency) 或 (None, None) 如果無法解析
        """
        try:
            details = self.instrument_manager.get_details(symbol)
            if details and hasattr(details, 'baseCurrency') and hasattr(details, 'quoteCurrency'): # OANDA v20 API uses baseCurrency, quoteCurrency
                return details.baseCurrency.upper(), details.quoteCurrency.upper()
            elif details and hasattr(details, 'base_currency') and hasattr(details, 'quote_currency'): # Fallback for older/custom naming
                 return details.base_currency.upper(), details.quote_currency.upper()

            if '_' in symbol:
                parts = symbol.upper().split('_')
                if len(parts) == 2:
                    # Basic validation for common currency codes (3 letters)
                    if len(parts[0]) == 3 and len(parts[1]) == 3:
                        return parts[0], parts[1]
                    # Allow for metals like XAU, XAG etc.
                    if parts[0] in ['XAU', 'XAG', 'XPT', 'XPD'] and len(parts[1]) == 3:
                        return parts[0], parts[1]
                    if len(parts[0]) == 3 and parts[1] in ['XAU', 'XAG', 'XPT', 'XPD']:
                         return parts[0], parts[1]

            logger.warning(f"無法通過 InstrumentInfoManager 或標準格式解析symbol {symbol} 的貨幣信息。")
            return None, None
            
        except Exception as e:
            logger.error(f"解析symbol {symbol} 時發生錯誤: {e}", exc_info=True)
            return None, None
    
    def determine_required_currency_pairs(self, trading_symbols: List[str]) -> Set[str]:
        """
        根據交易symbols確定需要的匯率對
        
        Args:
            trading_symbols: 要交易的symbol列表
            
        Returns:
            需要的匯率對集合
        """
        required_pairs = set()
        involved_currencies = set()
        
        # 1. 收集所有涉及的貨幣 (只收集報價貨幣，因為我們關心的是將報價貨幣轉換為帳戶貨幣)
        #    以及基礎貨幣，以防帳戶貨幣是基礎貨幣之一 (例如交易 AUD/USD，帳戶是 AUD)
        for symbol in trading_symbols:
            base_curr, quote_curr = self.parse_symbol_currencies(symbol)
            if base_curr and quote_curr:
                involved_currencies.add(base_curr)
                involved_currencies.add(quote_curr)
                logger.debug(f"Symbol {symbol}: Base={base_curr}, Quote={quote_curr}")
        
        # 2. 添加帳戶貨幣，確保它在集合中
        involved_currencies.add(self.account_currency)
        
        logger.info(f"所有涉及的貨幣 (包含帳戶貨幣): {sorted(list(involved_currencies))}")
        
        # 3. 為每個非帳戶貨幣的報價貨幣，以及與帳戶貨幣不同的基礎貨幣，確定到帳戶貨幣的轉換路徑
        #    主要目標是將交易的價值轉換回帳戶貨幣
        currencies_to_convert = set()
        for symbol in trading_symbols:
            base_curr, quote_curr = self.parse_symbol_currencies(symbol)
            if base_curr and quote_curr:
                if quote_curr != self.account_currency:
                    currencies_to_convert.add(quote_curr)
                # 如果基礎貨幣不是帳戶貨幣，並且我們可能需要將其價值轉換為帳戶貨幣
                # (例如，保證金以基礎貨幣計價，但需要以帳戶貨幣顯示)
                # 這裡的邏輯是，任何與帳戶貨幣不同的貨幣都可能需要轉換路徑
                if base_curr != self.account_currency:
                    currencies_to_convert.add(base_curr)

        for currency in currencies_to_convert:
            if currency != self.account_currency: # Redundant check, but safe
                conversion_pairs = self._find_conversion_path(currency, self.account_currency)
                required_pairs.update(conversion_pairs)
        
        # 如果帳戶貨幣不是USD，確保所有主要貨幣（如果它們參與交易）到USD的轉換路徑也存在
        # 這有助於更廣泛的交叉匯率計算或標準化報告
        if self.account_currency != 'USD':
            for currency in involved_currencies:
                if currency != 'USD' and currency != self.account_currency and currency in self.major_currencies:
                    # We need a path from this major currency to USD
                    usd_conversion_pairs = self._find_conversion_path(currency, 'USD')
                    required_pairs.update(usd_conversion_pairs)

        logger.info(f"初步確定需要的匯率對: {sorted(list(required_pairs))}")
        return required_pairs
    
    def _find_conversion_path(self, from_currency: str, to_currency: str) -> Set[str]:
        """
        找到從一種貨幣到另一種貨幣的轉換路徑所需的匯率對。
        這包括直接對、反向對，以及通過USD的間接對。
        
        Args:
            from_currency: 源貨幣
            to_currency: 目標貨幣
            
        Returns:
            所需匯率對的集合 (例如 {"EUR_USD", "USD_JPY"})
        """
        if from_currency == to_currency:
            return set()
        
        possible_pairs = set()
        
        # 1. 嘗試直接匯率對 (FROM_TO, TO_FROM)
        for template in self.common_pairs_templates:
            pair = template.format(base=from_currency, quote=to_currency)
            possible_pairs.add(pair)
        
        # 2. 如果直接轉換不是到USD或從USD開始，則考慮通過USD的交叉匯率
        #    (FROM -> USD -> TO)
        if from_currency != 'USD' and to_currency != 'USD':
            # FROM -> USD
            for template in self.common_pairs_templates:
                pair1 = template.format(base=from_currency, quote='USD')
                possible_pairs.add(pair1)
            # USD -> TO
            for template in self.common_pairs_templates:
                pair2 = template.format(base='USD', quote=to_currency)
                possible_pairs.add(pair2)
        
        return possible_pairs
    
    def download_required_currency_data(self, trading_symbols: List[str], 
                                      start_time_iso: str, end_time_iso: str, 
                                      granularity: str = "S5") -> bool:
        """
        下載交易所需的匯率對數據
        
        Args:
            trading_symbols: 要交易的symbol列表
            start_time_iso: 開始時間 (ISO格式)
            end_time_iso: 結束時間 (ISO格式)
            granularity: 數據粒度
            
        Returns:
            是否成功下載所有必需的數據
        """
        try:
            required_candidate_pairs = self.determine_required_currency_pairs(trading_symbols)
            
            if not required_candidate_pairs:
                logger.info("根據交易symbols和帳戶貨幣，無需下載額外的匯率對。")
                return True

            valid_pairs_to_download = []
            for pair_candidate in required_candidate_pairs:
                # 檢查此貨幣對是否真實存在於OANDA
                details = self.instrument_manager.get_details(pair_candidate)
                if details and details.name == pair_candidate: # Ensure the name matches exactly
                    valid_pairs_to_download.append(pair_candidate)
                    logger.debug(f"驗證匯率對 {pair_candidate}: 有效，將加入下載列表。")
                else:
                    #嘗試反向對，因為 determine_required_currency_pairs 可能產生 A_B, B_A
                    if '_' in pair_candidate:
                        parts = pair_candidate.split('_')
                        if len(parts) == 2:
                            reverse_candidate = f"{parts[1]}_{parts[0]}"
                            details_reverse = self.instrument_manager.get_details(reverse_candidate)
                            if details_reverse and details_reverse.name == reverse_candidate:
                                valid_pairs_to_download.append(reverse_candidate)
                                logger.debug(f"驗證匯率對 {pair_candidate} 的反向對 {reverse_candidate}: 有效，將加入下載列表。")
                            else:
                                logger.debug(f"驗證匯率對 {pair_candidate} 及其反向均無效或不存在於OANDA。")
                        else:
                            logger.debug(f"驗證匯率對 {pair_candidate}: 無效或不存在於OANDA。")
                    else:
                        logger.debug(f"驗證匯率對 {pair_candidate}: 格式不正確，無法驗證。")
            
            # 去重
            final_download_list = sorted(list(set(valid_pairs_to_download)))

            if not final_download_list:
                logger.info("所有候選匯率對均無效或已滿足，無需下載。")
                return True
            
            logger.info(f"開始下載 {len(final_download_list)} 個經過驗證的匯率對的數據...")
            logger.info(f"匯率對列表: {final_download_list}")
            
            # 注意：manage_data_download_for_symbols 應該能處理重複下載或已存在的數據
            manage_data_download_for_symbols(
                symbols=final_download_list,
                overall_start_str=start_time_iso,
                overall_end_str=end_time_iso,
                granularity=granularity
            )
            
            logger.info("匯率對數據下載任務已提交/完成。")
            return True # Assuming manage_data_download_for_symbols handles its own success/failure logging
            
        except Exception as e:
            logger.error(f"下載匯率對數據時發生錯誤: {e}", exc_info=True)
            return False
    
    def validate_currency_coverage(self, trading_symbols: List[str], 
                                 dataset_symbols: List[str]) -> Tuple[bool, List[str]]:
        """
        驗證數據集是否包含所有必需的匯率對
        
        Args:
            trading_symbols: 要交易的symbol列表
            dataset_symbols: 數據集中已有的symbol列表 (通常是已下載的)
            
        Returns:
            (是否完整覆蓋, 缺失的匯率對列表)
        """
        required_pairs_candidates = self.determine_required_currency_pairs(trading_symbols)
        dataset_symbols_set = set(s.upper() for s in dataset_symbols) # Normalize to upper
        
        missing_pairs = []
        if not required_pairs_candidates:
            logger.info("無需額外匯率對，覆蓋驗證通過。")
            return True, []

        for pair_cand in required_pairs_candidates:
            # A_B or B_A must exist
            parts = pair_cand.split('_')
            if len(parts) != 2:
                logger.warning(f"候選對 {pair_cand} 格式不正確，跳過覆蓋檢查。")
                continue
            
            direct_match = f"{parts[0]}_{parts[1]}"
            reverse_match = f"{parts[1]}_{parts[0]}"

            if not (direct_match in dataset_symbols_set or reverse_match in dataset_symbols_set):
                # Before declaring missing, check if this pair is even valid on OANDA
                # This avoids flagging pairs like "XYZ_ABC" if they don't exist at all.
                # We only care about missing *valid* OANDA pairs.
                is_valid_on_oanda = False
                details_direct = self.instrument_manager.get_details(direct_match)
                if details_direct and details_direct.name == direct_match:
                    is_valid_on_oanda = True
                else:
                    details_reverse = self.instrument_manager.get_details(reverse_match)
                    if details_reverse and details_reverse.name == reverse_match:
                        is_valid_on_oanda = True
                
                if is_valid_on_oanda:
                    # It's a valid pair, but not in our dataset
                    missing_pairs.append(direct_match) # Report the canonical form or preferred form
                else:
                    logger.debug(f"候選對 {pair_cand} (或其反向) 並非有效的OANDA交易品種，不視為缺失。")


        is_complete = len(missing_pairs) == 0
        
        if is_complete:
            logger.info("匯率覆蓋驗證通過：所有必需的、有效的匯率對都可用於數據集。")
        else:
            logger.warning(f"匯率覆蓋不完整，數據集中缺失以下有效的OANDA匯率對: {sorted(list(set(missing_pairs)))}")
        
        return is_complete, sorted(list(set(missing_pairs)))
    
    def get_currency_conversion_info(self, trading_symbols: List[str]) -> Dict[str, Dict[str, str]]:
        """
        獲取每個交易symbol的貨幣轉換信息
        
        Args:
            trading_symbols: 交易symbol列表
            
        Returns:
            每個symbol的貨幣轉換信息字典
        """
        conversion_info = {}
        
        for symbol in trading_symbols:
            base_curr, quote_curr = self.parse_symbol_currencies(symbol)
            if base_curr and quote_curr:
                path_description = "無需轉換"
                needs_conversion = quote_curr != self.account_currency
                
                if needs_conversion:
                    path_description = self._describe_conversion_path(quote_curr, self.account_currency)
                
                conversion_info[symbol] = {
                    'base_currency': base_curr,
                    'quote_currency': quote_curr,
                    'account_currency': self.account_currency,
                    'needs_conversion_to_account_currency': needs_conversion,
                    'conversion_path_to_account_currency': path_description
                }
            else:
                conversion_info[symbol] = {
                    'base_currency': 'N/A',
                    'quote_currency': 'N/A',
                    'account_currency': self.account_currency,
                    'needs_conversion_to_account_currency': False,
                    'conversion_path_to_account_currency': '無法解析Symbol貨幣'
                }

        return conversion_info
    
    def _describe_conversion_path(self, from_currency: str, to_currency: str) -> str:
        """
        描述貨幣轉換路徑 (報價貨幣 -> 帳戶貨幣)
        
        Args:
            from_currency: 源貨幣 (通常是交易symbol的報價貨幣)
            to_currency: 目標貨幣 (通常是帳戶貨幣)
            
        Returns:
            轉換路徑描述
        """
        if from_currency == to_currency:
            return "無需轉換"
        
        # 檢查是否有直接匯率對 (FROM_TO)
        direct_pair = f"{from_currency}_{to_currency}"
        if self.instrument_manager.get_details(direct_pair):
            return f"直接轉換: 使用 {direct_pair}"
        
        # 檢查是否有反向匯率對 (TO_FROM)
        reverse_pair = f"{to_currency}_{from_currency}"
        if self.instrument_manager.get_details(reverse_pair):
            return f"反向轉換: 使用 1 / {reverse_pair}"
        
        # 如果沒有直接或反向，則檢查通過USD的交叉路徑
        if from_currency != 'USD' and to_currency != 'USD':
            from_usd_pair = f"{from_currency}_USD"
            usd_to_pair = f"USD_{to_currency}" # Corrected: USD is a string literal
            
            # Check for FROM_USD (or USD_FROM)
            from_usd_exists = False
            path_leg1 = "" # Initialize
            if self.instrument_manager.get_details(from_usd_pair):
                from_usd_exists = True
                path_leg1 = from_usd_pair
            elif self.instrument_manager.get_details(f"USD_{from_currency}"):
                from_usd_exists = True
                path_leg1 = f"1 / USD_{from_currency}"
            
            # Check for USD_TO (or TO_USD)
            usd_to_exists = False
            path_leg2 = "" # Initialize
            if self.instrument_manager.get_details(usd_to_pair):
                usd_to_exists = True
                path_leg2 = usd_to_pair
            elif self.instrument_manager.get_details(f"{to_currency}_USD"):
                usd_to_exists = True
                path_leg2 = f"1 / {to_currency}_USD"

            if from_usd_exists and usd_to_exists:
                return f"交叉轉換: {from_currency} -> USD -> {to_currency} (透過 {path_leg1} 和 {path_leg2})"
        
        return f"無法確定從 {from_currency} 到 {to_currency} 的明確轉換路徑 (可能需要下載相應貨幣對)"


# 便利函數
def ensure_currency_data_for_trading(trading_symbols: List[str], 
                                   start_time_iso: str, end_time_iso: str,
                                   granularity: str = "S5",
                                   account_currency: str = ACCOUNT_CURRENCY) -> bool:
    """
    確保交易所需的所有匯率數據都已下載
    
    Args:
        trading_symbols: 要交易的symbol列表
        start_time_iso: 開始時間 (ISO格式)
        end_time_iso: 結束時間 (ISO格式)
        granularity: 數據粒度
        account_currency: 帳戶貨幣
        
    Returns:
        是否成功確保所有數據
    """
    manager = CurrencyDependencyManager(account_currency)
    return manager.download_required_currency_data(
        trading_symbols, start_time_iso, end_time_iso, granularity
    )


if __name__ == "__main__":
    # 設置一個簡單的日誌記錄器用於測試
    if not _logger_initialized_by_common_cm : # Use basic logger if common_logger wasn't loaded
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s:%(lineno)d) - %(message)s')
        logger = logging.getLogger(__name__)
        logger.info("Using basicConfig for logging in __main__ of currency_manager.")


    logger.info("正在測試 CurrencyDependencyManager...")
    
    # 模擬 InstrumentInfoManager 的 get_details 方法
    class MockInstrumentInfoManager:
        def __init__(self, **kwargs):
            self.instruments = {
                "EUR_USD": type('obj', (object,), {'name': 'EUR_USD', 'baseCurrency': 'EUR', 'quoteCurrency': 'USD'}),
                "USD_JPY": type('obj', (object,), {'name': 'USD_JPY', 'baseCurrency': 'USD', 'quoteCurrency': 'JPY'}),
                "AUD_USD": type('obj', (object,), {'name': 'AUD_USD', 'baseCurrency': 'AUD', 'quoteCurrency': 'USD'}),
                "EUR_AUD": type('obj', (object,), {'name': 'EUR_AUD', 'baseCurrency': 'EUR', 'quoteCurrency': 'AUD'}),
                "XAU_USD": type('obj', (object,), {'name': 'XAU_USD', 'baseCurrency': 'XAU', 'quoteCurrency': 'USD'}),
                # "USD_AUD": type('obj', (object,), {'name': 'USD_AUD', 'baseCurrency': 'USD', 'quoteCurrency': 'AUD'}), # Test missing reverse
            }
        def get_details(self, symbol):
            return self.instruments.get(symbol.upper())

    # 替換真實的 InstrumentInfoManager 和下載函數進行測試
    original_iim = InstrumentInfoManager
    original_mdds = manage_data_download_for_symbols
    
    InstrumentInfoManager = MockInstrumentInfoManager
    def mock_manage_data_download_for_symbols(symbols, overall_start_str, overall_end_str, granularity):
        logger.info(f"[MOCK] 下載請求: {symbols} from {overall_start_str} to {overall_end_str} at {granularity}")
        return True
    manage_data_download_for_symbols = mock_manage_data_download_for_symbols
    
    _ACCOUNT_CURRENCY_TEST = "AUD"
    logger.info(f"測試時帳戶貨幣設定為: {_ACCOUNT_CURRENCY_TEST}")

    # 測試場景1: 交易 EUR/USD, 帳戶 AUD. 應需 AUD/USD (或 USD/AUD)
    logger.info("\n--- 測試場景 1: EUR_USD, 帳戶 AUD ---")
    manager_aud = CurrencyDependencyManager(account_currency=_ACCOUNT_CURRENCY_TEST)
    test_symbols_1 = ["EUR_USD"]
    
    logger.info(f"交易 symbols: {test_symbols_1}")
    required_pairs_1 = manager_aud.determine_required_currency_pairs(test_symbols_1)
    logger.info(f"判斷所需貨幣對: {required_pairs_1}") # 應包含 AUD_USD 或 USD_AUD
    
    conversion_info_1 = manager_aud.get_currency_conversion_info(test_symbols_1)
    for symbol, info in conversion_info_1.items(): logger.info(f"轉換資訊 ({symbol}): {info}")

    manager_aud.download_required_currency_data(test_symbols_1, "2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z")

    # 測試場景2: 交易 XAU/USD, 帳戶 AUD. 應需 AUD/USD (或 USD/AUD)
    logger.info("\n--- 測試場景 2: XAU_USD, 帳戶 AUD ---")
    test_symbols_2 = ["XAU_USD"]
    logger.info(f"交易 symbols: {test_symbols_2}")
    required_pairs_2 = manager_aud.determine_required_currency_pairs(test_symbols_2)
    logger.info(f"判斷所需貨幣對: {required_pairs_2}")
    conversion_info_2 = manager_aud.get_currency_conversion_info(test_symbols_2)
    for symbol, info in conversion_info_2.items(): logger.info(f"轉換資訊 ({symbol}): {info}")
    manager_aud.download_required_currency_data(test_symbols_2, "2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z")

    # 測試場景3: 交易 EUR/AUD, 帳戶 AUD. 無需額外貨幣對
    logger.info("\n--- 測試場景 3: EUR_AUD, 帳戶 AUD ---")
    test_symbols_3 = ["EUR_AUD"]
    logger.info(f"交易 symbols: {test_symbols_3}")
    required_pairs_3 = manager_aud.determine_required_currency_pairs(test_symbols_3)
    logger.info(f"判斷所需貨幣對: {required_pairs_3}") # 應為空或只包含 EUR_AUD 本身 (如果邏輯如此設計)
    conversion_info_3 = manager_aud.get_currency_conversion_info(test_symbols_3)
    for symbol, info in conversion_info_3.items(): logger.info(f"轉換資訊 ({symbol}): {info}")
    manager_aud.download_required_currency_data(test_symbols_3, "2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z")

    # 測試場景4: 帳戶 USD, 交易 EUR/AUD. 應需 EUR/USD, AUD/USD (或其反向)
    logger.info("\n--- 測試場景 4: EUR_AUD, 帳戶 USD ---")
    _ACCOUNT_CURRENCY_TEST_USD = "USD"
    manager_usd = CurrencyDependencyManager(account_currency=_ACCOUNT_CURRENCY_TEST_USD)
    test_symbols_4 = ["EUR_AUD"]
    logger.info(f"交易 symbols: {test_symbols_4}, 帳戶貨幣: {_ACCOUNT_CURRENCY_TEST_USD}")
    required_pairs_4 = manager_usd.determine_required_currency_pairs(test_symbols_4)
    logger.info(f"判斷所需貨幣對: {required_pairs_4}")
    conversion_info_4 = manager_usd.get_currency_conversion_info(test_symbols_4)
    for symbol, info in conversion_info_4.items(): logger.info(f"轉換資訊 ({symbol}): {info}")
    manager_usd.download_required_currency_data(test_symbols_4, "2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z")

    # 測試 validate_currency_coverage
    logger.info("\n--- 測試 validate_currency_coverage ---")
    # 假設我們需要 AUD_USD, EUR_USD
    # manager_aud (帳戶 AUD)
    # test_symbols_1 = ["EUR_USD"] -> 需要 AUD_USD (或 USD_AUD)
    # required_pairs_1 should be something like {'AUD_USD', 'USD_AUD'} after filtering for valid pairs
    
    # Case 1: All covered
    dataset_1 = ["AUD_USD", "EUR_USD", "USD_JPY"]
    is_complete, missing = manager_aud.validate_currency_coverage(test_symbols_1, dataset_1)
    logger.info(f"對於 {test_symbols_1}, 數據集 {dataset_1}: 完整性={is_complete}, 缺失={missing}")

    # Case 2: Missing AUD_USD (but USD_AUD exists in mock)
    # MockInstrumentInfoManager has AUD_USD, so if determine_required_currency_pairs correctly identifies it
    # and it's not in dataset, it should be missing.
    dataset_2 = ["EUR_USD", "USD_JPY"] # Missing AUD_USD
    is_complete, missing = manager_aud.validate_currency_coverage(test_symbols_1, dataset_2)
    logger.info(f"對於 {test_symbols_1}, 數據集 {dataset_2}: 完整性={is_complete}, 缺失={missing}") # Should be missing AUD_USD

    # Case 3: Required pair doesn't exist on OANDA (e.g. XYZ_AUD)
    # Add a symbol that would generate a non-existent pair
    manager_aud.instrument_manager.instruments["XYZ_LOL"] = type('obj', (object,), {'name': 'XYZ_LOL', 'baseCurrency': 'XYZ', 'quoteCurrency': 'LOL'})
    test_symbols_non_existent = ["XYZ_LOL"] # Assume this would require XYZ_AUD or LOL_AUD
    dataset_3 = ["EUR_USD"]
    is_complete, missing = manager_aud.validate_currency_coverage(test_symbols_non_existent, dataset_3)
    logger.info(f"對於 {test_symbols_non_existent}, 數據集 {dataset_3}: 完整性={is_complete}, 缺失={missing}") # Should be complete if XYZ_AUD etc. are not valid


    # 還原
    InstrumentInfoManager = original_iim
    manage_data_download_for_symbols = original_mdds
    logger.info("\nCurrencyDependencyManager 測試完成。")