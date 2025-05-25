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
    logger.warning(f"currency_manager.py: Initial import failed: {e_initial_import_cm}. Attempting path adjustment...")
    
    project_root_cm = Path(__file__).resolve().parent.parent.parent
    if str(project_root_cm) not in sys.path:
        sys.path.insert(0, str(project_root_cm))
        logger.info(f"currency_manager.py: Added project root to sys.path: {project_root_cm}")
    
    try:
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
        
        # OANDA主要貨幣列表 (用於優化匯率路徑)
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
            # 獲取instrument詳細信息
            details = self.instrument_manager.get_details(symbol)
            if details and hasattr(details, 'base_currency') and hasattr(details, 'quote_currency'):
                return details.base_currency.upper(), details.quote_currency.upper()
            
            # 後備解析邏輯 - 基於命名慣例
            if '_' in symbol:
                parts = symbol.upper().split('_')
                if len(parts) == 2:
                    return parts[0], parts[1]
            
            logger.warning(f"無法解析symbol {symbol} 的貨幣信息")
            return None, None
            
        except Exception as e:
            logger.error(f"解析symbol {symbol} 時發生錯誤: {e}")
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
        
        # 1. 收集所有涉及的貨幣
        for symbol in trading_symbols:
            base_curr, quote_curr = self.parse_symbol_currencies(symbol)
            if base_curr and quote_curr:
                involved_currencies.add(base_curr)
                involved_currencies.add(quote_curr)
                logger.debug(f"Symbol {symbol}: {base_curr}/{quote_curr}")
        
        # 2. 添加帳戶貨幣
        involved_currencies.add(self.account_currency)
        
        logger.info(f"涉及的貨幣: {sorted(involved_currencies)}")
        
        # 3. 為每個非帳戶貨幣確定到帳戶貨幣的轉換路徑
        for currency in involved_currencies:
            if currency != self.account_currency:
                conversion_pairs = self._find_conversion_path(currency, self.account_currency)
                required_pairs.update(conversion_pairs)
        
        # 4. 添加主要貨幣對USD的匯率 (用於交叉匯率計算)
        if self.account_currency != 'USD':
            for currency in involved_currencies:
                if currency != 'USD' and currency in self.major_currencies:
                    usd_pairs = self._find_conversion_path(currency, 'USD')
                    required_pairs.update(usd_pairs)
        
        logger.info(f"確定需要的匯率對: {sorted(required_pairs)}")
        return required_pairs
    
    def _find_conversion_path(self, from_currency: str, to_currency: str) -> Set[str]:
        """
        找到從一種貨幣到另一種貨幣的轉換路徑所需的匯率對
        
        Args:
            from_currency: 源貨幣
            to_currency: 目標貨幣
            
        Returns:
            所需匯率對的集合
        """
        if from_currency == to_currency:
            return set()
        
        possible_pairs = set()
        
        # 1. 嘗試直接匯率對
        for template in self.common_pairs_templates:
            pair = template.format(base=from_currency, quote=to_currency)
            possible_pairs.add(pair)
        
        # 2. 如果不是主要貨幣對，嘗試通過USD的交叉匯率
        if from_currency != 'USD' and to_currency != 'USD':
            # from_currency -> USD -> to_currency
            for template in self.common_pairs_templates:
                pair1 = template.format(base=from_currency, quote='USD')
                pair2 = template.format(base='USD', quote=to_currency)
                possible_pairs.add(pair1)
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
            # 1. 確定需要的匯率對
            required_pairs = self.determine_required_currency_pairs(trading_symbols)
            
            # 2. 過濾出實際存在的匯率對 (通過instrument manager驗證)
            valid_pairs = []
            for pair in required_pairs:
                details = self.instrument_manager.get_details(pair)
                if details:
                    valid_pairs.append(pair)
                    logger.debug(f"驗證匯率對 {pair}: 有效")
                else:
                    logger.debug(f"驗證匯率對 {pair}: 無效或不存在")
            
            if not valid_pairs:
                logger.warning("沒有找到有效的匯率對，可能所有交易symbol都已經是帳戶貨幣計價")
                return True
            
            # 3. 下載匯率對數據
            logger.info(f"開始下載 {len(valid_pairs)} 個匯率對的數據...")
            logger.info(f"匯率對列表: {valid_pairs}")
            
            manage_data_download_for_symbols(
                symbols=valid_pairs,
                overall_start_str=start_time_iso,
                overall_end_str=end_time_iso,
                granularity=granularity
            )
            
            logger.info("匯率對數據下載完成")
            return True
            
        except Exception as e:
            logger.error(f"下載匯率對數據時發生錯誤: {e}", exc_info=True)
            return False
    
    def validate_currency_coverage(self, trading_symbols: List[str], 
                                 dataset_symbols: List[str]) -> Tuple[bool, List[str]]:
        """
        驗證數據集是否包含所有必需的匯率對
        
        Args:
            trading_symbols: 要交易的symbol列表
            dataset_symbols: 數據集中已有的symbol列表
            
        Returns:
            (是否完整覆蓋, 缺失的匯率對列表)
        """
        required_pairs = self.determine_required_currency_pairs(trading_symbols)
        dataset_symbols_set = set(dataset_symbols)
        
        missing_pairs = []
        for pair in required_pairs:
            # 檢查直接匹配或反向匹配
            if pair not in dataset_symbols_set:
                # 檢查反向對
                if '_' in pair:
                    parts = pair.split('_')
                    reverse_pair = f"{parts[1]}_{parts[0]}"
                    if reverse_pair not in dataset_symbols_set:
                        missing_pairs.append(pair)
        
        is_complete = len(missing_pairs) == 0
        
        if is_complete:
            logger.info("匯率覆蓋驗證通過：所有必需的匯率對都可用")
        else:
            logger.warning(f"匯率覆蓋不完整，缺失: {missing_pairs}")
        
        return is_complete, missing_pairs
    
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
                conversion_info[symbol] = {
                    'base_currency': base_curr,
                    'quote_currency': quote_curr,
                    'account_currency': self.account_currency,
                    'needs_conversion': quote_curr != self.account_currency,
                    'conversion_path': self._describe_conversion_path(quote_curr, self.account_currency)
                }
        
        return conversion_info
    
    def _describe_conversion_path(self, from_currency: str, to_currency: str) -> str:
        """
        描述貨幣轉換路徑
        
        Args:
            from_currency: 源貨幣
            to_currency: 目標貨幣
            
        Returns:
            轉換路徑描述
        """
        if from_currency == to_currency:
            return "無需轉換"
        
        # 檢查是否有直接匯率對
        direct_pair = f"{from_currency}_{to_currency}"
        reverse_pair = f"{to_currency}_{from_currency}"
        
        details_direct = self.instrument_manager.get_details(direct_pair)
        details_reverse = self.instrument_manager.get_details(reverse_pair)
        
        if details_direct:
            return f"直接轉換: {direct_pair}"
        elif details_reverse:
            return f"反向轉換: {reverse_pair}"
        else:
            return f"交叉轉換: {from_currency} -> USD -> {to_currency}"


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
    # 測試代碼
    logger.info("正在測試 CurrencyDependencyManager...")
    
    # 測試場景1: EUR/USD 和 USD/JPY (需要 AUD/USD 匯率)
    test_symbols = ["EUR_USD", "USD_JPY"]
    manager = CurrencyDependencyManager("AUD")
    
    logger.info(f"測試symbols: {test_symbols}")
    logger.info(f"帳戶貨幣: AUD")
    
    # 分析貨幣轉換需求
    conversion_info = manager.get_currency_conversion_info(test_symbols)
    for symbol, info in conversion_info.items():
        logger.info(f"{symbol}: {info}")
    
    # 確定需要的匯率對
    required_pairs = manager.determine_required_currency_pairs(test_symbols)
    logger.info(f"需要的匯率對: {sorted(required_pairs)}")
    
    # 測試場景2: 已經是帳戶貨幣計價的情況
    test_symbols_aud = ["AUD_USD", "AUD_JPY"]
    logger.info(f"\n測試symbols (AUD計價): {test_symbols_aud}")
    
    conversion_info_aud = manager.get_currency_conversion_info(test_symbols_aud)
    for symbol, info in conversion_info_aud.items():
        logger.info(f"{symbol}: {info}")
    
    required_pairs_aud = manager.determine_required_currency_pairs(test_symbols_aud)
    logger.info(f"需要的匯率對: {sorted(required_pairs_aud)}")
    
    logger.info("CurrencyDependencyManager 測試完成")