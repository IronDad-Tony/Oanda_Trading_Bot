# live_trading_system/core/oanda_client.py
"""
Oanda API 封裝客戶端
"""
import os
import logging
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
import oandapyV20
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# 獲取日誌記錄器
logger = logging.getLogger("LiveTradingSystem")

# 載入環境變數
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

class OandaClient:
    """
    一個封裝了 Oanda V20 API 請求的客戶端，包含錯誤處理和重試機制。
    使用 oandapyV20 函式庫。
    """
    def __init__(self, api_key: str, account_id: str, environment: str = "practice"):
        self.account_id = account_id
        self.last_transaction_id: Optional[str] = None # <--- 新增：保存最後的交易ID
        if not api_key or not account_id:
            logger.critical("OANDA_ACCOUNT_ID 或 OANDA_API_KEY 未提供。")
            raise ValueError("API credentials must be provided")
        
        try:
            self.client = oandapyV20.API(access_token=api_key, environment=environment)
            self.initialize_transaction_id() # <--- 新增：初始化時獲取ID
        except Exception as e:
            logger.critical(f"無法初始化 Oanda API 客戶端: {e}")
            raise

    def initialize_transaction_id(self):
        """在啟動時獲取一次帳戶摘要，以初始化 last_transaction_id。"""
        logger.info("正在初始化 last_transaction_id...")
        summary = self.get_account_summary()
        if summary and 'account' in summary and 'lastTransactionID' in summary['account']:
            self.last_transaction_id = summary['account']['lastTransactionID']
            logger.info(f"last_transaction_id 初始化成功: {self.last_transaction_id}")
        else:
            logger.warning("無法在啟動時初始化 last_transaction_id。後續的 AccountChanges 請求可能會失敗。")

    @classmethod
    def from_env(cls):
        """從 .env 檔案創建一個客戶端實例。"""
        load_dotenv()
        api_key = os.getenv("OANDA_API_KEY")
        account_id = os.getenv("OANDA_ACCOUNT_ID")
        environment = os.getenv("OANDA_ENVIRONMENT", "practice")
        return cls(api_key, account_id, environment)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(V20Error),
        reraise=True
    )
    def _request(self, endpoint) -> Dict[str, Any]:
        """
        執行 API 請求並返回 JSON 響應。
        如果請求失敗，會根據 tenacity 配置進行重試。
        """
        try:
            endpoint_name = type(endpoint).__name__
            logger.debug(f"Executing OANDA request: {endpoint_name}")
            response = self.client.request(endpoint)
            logger.debug(f"OANDA response received for {endpoint_name}")
            return response
        except V20Error as err:
            endpoint_name = type(endpoint).__name__
            logger.error(f"OANDA API Error: {err}. Endpoint: {endpoint_name}, Params: {endpoint.params if hasattr(endpoint, 'params') else 'N/A'}")
            raise
        except Exception as e:
            logger.critical(f"An unexpected error occurred during API request: {e}")
            raise

    def get_account_summary(self) -> Optional[Dict[str, Any]]:
        """獲取帳戶摘要資訊，並更新 last_transaction_id。"""
        endpoint = accounts.AccountSummary(self.account_id)
        try:
            response = self._request(endpoint)
            if response and 'account' in response and 'lastTransactionID' in response['account']:
                self.last_transaction_id = response['account']['lastTransactionID']
                logger.debug(f"last_transaction_id 已更新為: {self.last_transaction_id}")
            return response
        except V20Error:
            return None

    def get_candles(self, instrument: str, count: int = 100, granularity: str = "S5", price: str = "M") -> Optional[List[Dict[str, Any]]]:
        """
        獲取最新的蠟燭圖數據。
        """
        params = {"granularity": granularity, "count": int(count), "price": price}
        endpoint = instruments.InstrumentsCandles(instrument=instrument, params=params)
        try:
            data = self._request(endpoint)
            return data.get("candles") if data else None
        except V20Error:
            return None

    def create_order(self, instrument: str, units: int, stop_loss_on_fill: Optional[Dict] = None, take_profit_on_fill: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """建立一個市價單。"""
        order_data = {
            "order": {
                "instrument": instrument,
                "units": str(units),
                "type": "MARKET",
                "timeInForce": "FOK",
                "positionFill": "DEFAULT"
            }
        }
        if stop_loss_on_fill:
            order_data["order"]["stopLossOnFill"] = stop_loss_on_fill
        if take_profit_on_fill:
            order_data["order"]["takeProfitOnFill"] = take_profit_on_fill
        
        endpoint = orders.OrderCreate(self.account_id, data=order_data)
        try:
            return self._request(endpoint)
        except V20Error as e:
            logger.error(f"Failed to create order for {instrument}: {e}")
            return None

    def get_open_positions(self) -> Optional[Dict[str, Any]]:
        """獲取所有未平倉部位。"""
        endpoint = positions.OpenPositions(accountID=self.account_id)
        try:
            return self._request(endpoint)
        except V20Error:
            return None
            
    def get_account_changes(self) -> Optional[Dict[str, Any]]: # <--- 原 get_equity_history, 更名並修改
        """獲取自上次查詢以來的帳戶變動。"""
        if not self.last_transaction_id:
            logger.warning("last_transaction_id 未設定，無法獲取 AccountChanges。請先調用 get_account_summary()。")
            # 或者，可以選擇在這裡調用一次 get_account_summary() 來初始化
            self.initialize_transaction_id()
            if not self.last_transaction_id:
                 return None

        params = {"sinceTransactionID": self.last_transaction_id}
        endpoint = accounts.AccountChanges(self.account_id, params=params)
        try:
            response = self._request(endpoint)
            # 在成功獲取後，更新 last_transaction_id
            if response and 'lastTransactionID' in response:
                new_last_id = response['lastTransactionID']
                if self.last_transaction_id != new_last_id:
                    self.last_transaction_id = new_last_id
                    logger.debug(f"AccountChanges 成功，last_transaction_id 更新為: {self.last_transaction_id}")
            return response
        except V20Error:
            return None

    def close_position(self, instrument: str, long_units: Optional[str] = None, short_units: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        關閉一個商品的未平倉部位。
        """
        data = {}
        if long_units:
            data["longUnits"] = long_units
        elif short_units:
            data["shortUnits"] = short_units
        else:
            logger.warning("No units specified for position close for instrument %s.", instrument)
            return None

        endpoint = positions.PositionClose(
            accountID=self.account_id,
            instrument=instrument,
            data=data
        )
        try:
            return self._request(endpoint)
        except V20Error as e:
            logger.error(f"Failed to close position for {instrument}: {e}")
            return None

    def get_bid_ask_candles_combined(self, instrument: str, count: int = 100, granularity: str = "S5") -> Optional[List[Dict[str, Any]]]:
        """Fetch bid and ask candles and merge into unified records for downstream processing.

        Each record contains: time, bid_open/high/low/close, ask_open/high/low/close, and volume if provided.
        """
        try:
            bid = self.get_candles(instrument, count=count, granularity=granularity, price="B") or []
            ask = self.get_candles(instrument, count=count, granularity=granularity, price="A") or []
            if not bid and not ask:
                return []
            bid_map = {c.get("time"): c for c in bid if c and c.get("complete", True)}
            ask_map = {c.get("time"): c for c in ask if c and c.get("complete", True)}
            times = sorted(set([t for t in bid_map.keys() if t]) | set([t for t in ask_map.keys() if t]))
            merged: List[Dict[str, Any]] = []
            for t in times:
                rec: Dict[str, Any] = {"time": t}
                if t in bid_map and "bid" in bid_map[t]:
                    b = bid_map[t]["bid"]
                    rec["bid_open"] = float(b.get("o", 0.0))
                    rec["bid_high"] = float(b.get("h", 0.0))
                    rec["bid_low"] = float(b.get("l", 0.0))
                    rec["bid_close"] = float(b.get("c", 0.0))
                    if "volume" in bid_map[t]:
                        rec["volume"] = int(bid_map[t]["volume"])
                if t in ask_map and "ask" in ask_map[t]:
                    a = ask_map[t]["ask"]
                    rec["ask_open"] = float(a.get("o", 0.0))
                    rec["ask_high"] = float(a.get("h", 0.0))
                    rec["ask_low"] = float(a.get("l", 0.0))
                    rec["ask_close"] = float(a.get("c", 0.0))
                    if "volume" in ask_map[t] and "volume" not in rec:
                        rec["volume"] = int(ask_map[t]["volume"])
                merged.append(rec)
            return merged
        except Exception as e:
            logger.error(f"Error fetching bid/ask candles for {instrument}: {e}")
            return None

    def create_order_v2(
        self,
        instrument: str,
        units: int,
        stop_loss_on_fill: Optional[Dict] = None,
        take_profit_on_fill: Optional[Dict] = None,
        price_bound: Optional[float] = None,
        client_extensions: Optional[Dict[str, Any]] = None,
        time_in_force: str = "FOK",
    ) -> Optional[Dict[str, Any]]:
        """Create a MARKET order with optional SL/TP, priceBound, and clientExtensions.

        This method preserves the original `create_order` for backwards compatibility
        and adds support for price protection and id tagging.
        """
        order_data: Dict[str, Any] = {
            "order": {
                "instrument": instrument,
                "units": str(units),
                "type": "MARKET",
                "timeInForce": time_in_force,
                "positionFill": "DEFAULT",
            }
        }
        if stop_loss_on_fill:
            order_data["order"]["stopLossOnFill"] = stop_loss_on_fill
        if take_profit_on_fill:
            order_data["order"]["takeProfitOnFill"] = take_profit_on_fill
        if price_bound is not None:
            order_data["order"]["priceBound"] = str(price_bound)
        if client_extensions:
            order_data["order"]["clientExtensions"] = client_extensions

        endpoint = orders.OrderCreate(self.account_id, data=order_data)
        try:
            return self._request(endpoint)
        except V20Error as e:
            logger.error(f"Failed to create order for {instrument}: {e}")
            return None

# --- 範例 ---
if __name__ == '__main__':
    logger.info("正在測試 OandaClient...")
    client = OandaClient.from_env()

    # 測試獲取帳戶資訊
    account_summary = client.get_account_summary()
    if account_summary:
        logger.info(f"帳戶資訊獲取成功: Balance = {account_summary['account']['balance']}")
        logger.info(f"Last Transaction ID: {client.last_transaction_id}")
    else:
        logger.error("帳戶資訊獲取失敗。")

    # 測試獲取帳戶變動
    changes = client.get_account_changes()
    if changes:
        logger.info(f"獲取到 {len(changes.get('changes', {}).get('ordersCreated', []))} 個新訂單。")
        logger.info(f"Last Transaction ID after changes: {client.last_transaction_id}")
    else:
        logger.warning("帳戶變動資訊獲取失敗 (可能是因為沒有變動)。")

    # 測試獲取蠟燭圖
    candles = client.get_candles("EUR_USD", granularity="M5", count=5)
    if candles:
        logger.info(f"獲取到 {len(candles)} 根 EUR_USD M5 蠟燭圖數據。")
    else:
        logger.error("蠟燭圖數據獲取失敗。")
