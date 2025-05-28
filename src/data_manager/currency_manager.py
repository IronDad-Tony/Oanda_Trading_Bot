# src/data_manager/currency_manager.py
"""
貨幣管理器 - 智能匯率對依賴管理
根據交易symbols自動判斷並下載所需的匯率對數據，完全模擬OANDA V20帳戶的匯率轉換邏輯
"""

import logging
import sys
from pathlib import Path
from typing import List, Set, Dict, Optional, Tuple # Ensure Tuple is imported
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
        根據交易symbols確定需要的匯率對.
        會檢查直接對和反向對的有效性。
        
        Args:
            trading_symbols: 要交易的symbol列表
            
        Returns:
            實際有效的、需要的匯率對集合 (例如 {"EUR_USD", "USD_JPY"})
        """
        required_pairs = set()
        all_involved_currencies = set() # 包括交易對的基礎和報價貨幣，以及帳戶貨幣

        for symbol in trading_symbols:
            base_curr, quote_curr = self.parse_symbol_currencies(symbol)
            if base_curr and quote_curr:
                all_involved_currencies.add(base_curr)
                all_involved_currencies.add(quote_curr)
        all_involved_currencies.add(self.account_currency)
        
        logger.info(f"所有涉及的貨幣 (交易對 + 帳戶貨幣): {sorted(list(all_involved_currencies))}")

        # 確定需要轉換的路徑：任何非帳戶貨幣都需要一條到帳戶貨幣的路徑
        currencies_needing_conversion_to_account = all_involved_currencies - {self.account_currency}

        for currency_from in currencies_needing_conversion_to_account:
            # 路徑1: currency_from -> self.account_currency
            path1_pairs = self._find_and_validate_conversion_path_pairs(currency_from, self.account_currency)
            required_pairs.update(path1_pairs)

            # 路徑2 (如果帳戶貨幣非USD，且當前貨幣也非USD): currency_from -> USD
            # 這有助於建立更廣泛的交叉匯率基礎
            if self.account_currency != 'USD' and currency_from != 'USD':
                path2_pairs = self._find_and_validate_conversion_path_pairs(currency_from, 'USD')
                required_pairs.update(path2_pairs)
        
        # 如果帳戶貨幣不是USD，還需要確保USD到帳戶貨幣的路徑
        if self.account_currency != 'USD':
            usd_to_account_pairs = self._find_and_validate_conversion_path_pairs('USD', self.account_currency)
            required_pairs.update(usd_to_account_pairs)

        logger.info(f"最終確定的有效且必要的匯率對: {sorted(list(required_pairs))}")
        return required_pairs

    def _find_and_validate_conversion_path_pairs(self, from_currency: str, to_currency: str) -> Set[str]:
        """
        查找並驗證從 from_currency 到 to_currency 的轉換路徑中實際有效的OANDA貨幣對。
        """
        if from_currency == to_currency:
            return set()

        valid_pairs_for_path = set()

        # 1. 嘗試直接/反向對: FROM_TO 或 TO_FROM
        direct_pair = f"{from_currency}_{to_currency}"
        reverse_pair = f"{to_currency}_{from_currency}"
        
        details_direct = self.instrument_manager.get_details(direct_pair)
        if details_direct and details_direct.symbol == direct_pair:
            valid_pairs_for_path.add(direct_pair)
            logger.debug(f"找到有效直接對: {direct_pair} for {from_currency}->{to_currency}")
            return valid_pairs_for_path # 如果直接對存在，優先使用

        details_reverse = self.instrument_manager.get_details(reverse_pair)
        if details_reverse and details_reverse.symbol == reverse_pair:
            valid_pairs_for_path.add(reverse_pair)
            logger.debug(f"找到有效反向對: {reverse_pair} for {from_currency}->{to_currency}")
            return valid_pairs_for_path # 如果反向對存在（且直接對不存在），則使用

        # 2. 如果沒有直接/反向對，且涉及USD的轉換，則嘗試通過USD
        # (FROM -> USD -> TO)
        if from_currency != 'USD' and to_currency != 'USD':
            logger.debug(f"嘗試通過USD為 {from_currency}->{to_currency} 尋找路徑")
            # Leg 1: FROM -> USD
            from_to_usd_direct = f"{from_currency}_USD"
            usd_from_reverse = f"USD_{from_currency}"
            
            leg1_pair_found = None
            details_from_usd = self.instrument_manager.get_details(from_to_usd_direct)
            if details_from_usd and details_from_usd.symbol == from_to_usd_direct:
                leg1_pair_found = from_to_usd_direct
            else:
                details_usd_from = self.instrument_manager.get_details(usd_from_reverse)
                if details_usd_from and details_usd_from.symbol == usd_from_reverse:
                    leg1_pair_found = usd_from_reverse
            
            if leg1_pair_found:
                logger.debug(f"  找到USD路徑第一部分: {leg1_pair_found}")
            else:
                logger.debug(f"  未找到 {from_currency}<->USD 的有效對")


            # Leg 2: USD -> TO
            usd_to_to_direct = f"USD_{to_currency}"
            to_usd_reverse = f"{to_currency}_USD"

            leg2_pair_found = None
            details_usd_to = self.instrument_manager.get_details(usd_to_to_direct)
            if details_usd_to and details_usd_to.symbol == usd_to_to_direct:
                leg2_pair_found = usd_to_to_direct
            else:
                details_to_usd = self.instrument_manager.get_details(to_usd_reverse)
                if details_to_usd and details_to_usd.symbol == to_usd_reverse:
                    leg2_pair_found = to_usd_reverse
            
            if leg2_pair_found:
                logger.debug(f"  找到USD路徑第二部分: {leg2_pair_found}")
            else:
                logger.debug(f"  未找到 USD<->{to_currency} 的有效對")

            if leg1_pair_found and leg2_pair_found:
                valid_pairs_for_path.add(leg1_pair_found)
                valid_pairs_for_path.add(leg2_pair_found)
                logger.debug(f"找到通過USD的有效路徑對: {leg1_pair_found}, {leg2_pair_found} for {from_currency}->{to_currency}")
                return valid_pairs_for_path
            else:
                # Corrected f-string for missing legs
                missing_leg_info = []
                if not leg1_pair_found:
                    missing_leg_info.append("leg1")
                if not leg2_pair_found:
                    missing_leg_info.append("leg2")
                logger.warning(f"無法為 {from_currency} -> {to_currency} 找到通過USD的完整轉換路徑。缺失: {', '.join(missing_leg_info)}")
        
        if not valid_pairs_for_path:
             logger.warning(f"無法為 {from_currency} -> {to_currency} 找到任何直接、反向或通過USD的有效轉換貨幣對。")

        return valid_pairs_for_path

    def _find_conversion_path(self, from_currency: str, to_currency: str) -> Set[str]:
        # 此方法 (_find_conversion_path) 的原始邏輯是生成 *候選* 對，
        # 而新的 _find_and_validate_conversion_path_pairs 會直接查找並驗證。
        # 為了保持 determine_required_currency_pairs 的簡潔性，
        # 我們可以讓 determine_required_currency_pairs 直接調用 _find_and_validate_conversion_path_pairs。
        # 因此，這個舊的 _find_conversion_path 可能不再需要，或者可以被重構/移除。
        # 暫時保留，但標記為可能廢棄。
        logger.debug(f"[_find_conversion_path - DEPRECATED CANDIDATE] Called for {from_currency} to {to_currency}")
        if from_currency == to_currency:
            return set()
        
        possible_pairs = set()
        possible_pairs.add(f"{from_currency}_{to_currency}")
        possible_pairs.add(f"{to_currency}_{from_currency}")
        
        if from_currency != 'USD' and to_currency != 'USD':
            possible_pairs.add(f"{from_currency}_USD")
            possible_pairs.add(f"USD_{from_currency}")
            possible_pairs.add(f"USD_{to_currency}")
            possible_pairs.add(f"{to_currency}_USD")
        return possible_pairs

    def download_required_currency_data(self, trading_symbols: List[str], 
                                      start_time_iso: str, end_time_iso: str, 
                                      granularity: str = "S5") -> bool:
        """
        下載交易所需的、經過驗證有效的匯率對數據
        """
        try:
            # determine_required_currency_pairs 現在返回的是已經過驗證的有效貨幣對
            final_download_list = self.determine_required_currency_pairs(trading_symbols)
            
            if not final_download_list:
                logger.info("根據交易symbols和帳戶貨幣，無需下載額外的匯率對，或所有必需對均無效。")
                return True # No valid pairs means nothing to download

            logger.info(f"開始下載 {len(final_download_list)} 個經過驗證的匯率對的數據...")
            logger.info(f"匯率對列表: {sorted(list(final_download_list))}")
            
            manage_data_download_for_symbols(
                symbols=sorted(list(final_download_list)), # Pass the validated list
                overall_start_str=start_time_iso,
                overall_end_str=end_time_iso,
                granularity=granularity
            )
            
            logger.info("匯率對數據下載任務已提交/完成。")
            return True
            
        except Exception as e:
            logger.error(f"下載匯率對數據時發生錯誤: {e}", exc_info=True)
            return False
    
    def validate_currency_coverage(self, trading_symbols: List[str], 
                                 dataset_symbols: List[str]) -> Tuple[bool, List[str]]:
        """
        驗證數據集是否包含所有必需的、有效的匯率對
        """
        # determine_required_currency_pairs 返回的是OANDA上實際有效的對
        required_valid_pairs = self.determine_required_currency_pairs(trading_symbols)
        dataset_symbols_set = set(s.upper() for s in dataset_symbols)
        
        missing_valid_pairs = []
        if not required_valid_pairs:
            logger.info("無需額外匯率對，覆蓋驗證通過。")
            return True, []

        for valid_pair in required_valid_pairs:
            # 因為 determine_required_currency_pairs 返回的已經是OANDA上存在的symbol (可能是 A_B 或 B_A)
            # 所以我們只需要檢查這個 valid_pair 是否在 dataset_symbols_set 中
            if valid_pair not in dataset_symbols_set:
                # 這裡可以進一步確認，如果 valid_pair 是 X_Y，而數據庫裡有 Y_X，是否算滿足。
                # 但由於 determine_required_currency_pairs 已經做了智能選擇，
                # 它會優先選擇一個方向。如果那個方向不在 dataset_symbols_set，則認為缺失。
                # 或者，我們可以讓 determine_required_currency_pairs 返回一個 canonical form，
                # 然後檢查 canonical form 或其反向是否在 dataset_symbols_set 中。
                # 目前的邏輯是：如果 determine_required_currency_pairs 確定需要 X_Y，那麼 X_Y 就必須在數據集中。
                missing_valid_pairs.append(valid_pair)
        
        is_complete = len(missing_valid_pairs) == 0
        
        if is_complete:
            logger.info("匯率覆蓋驗證通過：所有必需的、有效的匯率對都可用於數據集。")
        else:
            logger.warning(f"匯率覆蓋不完整，數據集中缺失以下有效的OANDA匯率對: {sorted(list(set(missing_valid_pairs)))}")
            logger.warning(f"數據集中的可用symbols: {sorted(list(dataset_symbols_set))}")
        
        return is_complete, sorted(list(set(missing_valid_pairs)))


def ensure_currency_data_for_trading(
    trading_symbols: List[str],
    account_currency: str,
    start_time_iso: str,
    end_time_iso: str,
    granularity: str
) -> Tuple[bool, Set[str]]:
    """
    確保所有交易必需的數據（包括主要交易品種和其依賴的匯率轉換對）都已下載。

    Args:
        trading_symbols: 用戶選擇的主要交易品種列表。
        account_currency: 交易帳戶的貨幣。
        start_time_iso: 數據開始時間 (ISO格式字符串)。
        end_time_iso: 數據結束時間 (ISO格式字符串)。
        granularity: 數據的時間粒度 (例如 "S5", "M1")。

    Returns:
        Tuple[bool, Set[str]]: 一個元組，第一個元素表示操作是否成功，
                               第二個元素是包含所有被處理（嘗試下載）的品種的集合。
    """
    try:
        logger.info(f"開始確保交易數據完整性 for symbols: {trading_symbols}, account currency: {account_currency}")
        manager = CurrencyDependencyManager(account_currency=account_currency)
        
        # 1. 確定所有需要的匯率轉換對
        required_conversion_pairs = manager.determine_required_currency_pairs(trading_symbols)
        logger.info(f"根據交易品種 {trading_symbols} 和帳戶貨幣 {account_currency}，確定的必需匯率對: {required_conversion_pairs}")
        
        # 2. 合併主要交易品種和匯率轉換對
        all_symbols_to_process = set(s.upper() for s in trading_symbols)
        all_symbols_to_process.update(required_conversion_pairs) # required_conversion_pairs 已經是大寫
        
        if not all_symbols_to_process:
            logger.info("沒有識別出需要下載數據的品種 (主要交易品種和轉換對均為空)。")
            return True, set()

        logger.info(f"將為以下所有品種確保數據: {sorted(list(all_symbols_to_process))}")
        
        # 3. 為所有識別出的品種下載數據
        # manage_data_download_for_symbols 函數已在此文件頂部導入
        manage_data_download_for_symbols(
            symbols=sorted(list(all_symbols_to_process)),
            overall_start_str=start_time_iso,
            overall_end_str=end_time_iso,
            granularity=granularity
            # streamlit_progress_bar 和 streamlit_status_text 由 universal_trainer 控制，
            # manage_data_download_for_symbols 內部應使用 shared_data_manager 更新進度。
        )
        
        logger.info(f"已為品種 {sorted(list(all_symbols_to_process))} 提交/完成數據確保流程。")
        # 假設 manage_data_download_for_symbols 內部會處理錯誤並記錄
        # 此處返回True表示流程已執行，all_symbols_to_process 是嘗試處理的品種
        return True, all_symbols_to_process

    except Exception as e:
        logger.error(f"在 ensure_currency_data_for_trading 過程中發生嚴重錯誤: {e}", exc_info=True)
        return False, set()