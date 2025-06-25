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
        if not api_key or not account_id:
            logger.critical("OANDA_ACCOUNT_ID 或 OANDA_API_KEY 未提供。")
            raise ValueError("API credentials must be provided")
        
        try:
            self.client = oandapyV20.API(access_token=api_key, environment=environment)
        except Exception as e:
            logger.critical(f"無法初始化 Oanda API 客戶端: {e}")
            raise

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
            # Use type(endpoint).__name__ to get the class name for logging
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
        """獲取帳戶摘要資訊。"""
        endpoint = accounts.AccountSummary(self.account_id)
        try:
            return self._request(endpoint)
        except V20Error:
            return None

    def get_candles(self, instrument: str, count: int = 100, granularity: str = "S5", price: str = "M") -> Optional[List[Dict[str, Any]]]:
        """
        獲取最新的蠟燭圖數據。 price: 'M' (Midpoint), 'B' (Bid), 'A' (Ask)
        修正參數順序與型別，確保 granularity 為字串、count 為整數。
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
                "timeInForce": "FOK",  # Fill or Kill
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

    def close_position(self, instrument: str, long_units: Optional[str] = None, short_units: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        關閉一個商品的未平倉部位。
        需明確指定關閉多頭或空頭部位。
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

# --- 範例 ---
if __name__ == '__main__':
    logger.info("正在測試 OandaClient...")
    client = OandaClient.from_env()

    # 測試獲取帳戶資訊
    account_summary = client.get_account_summary()
    if account_summary:
        logger.info(f"帳戶資訊獲取成功: Balance = {account_summary['account']['balance']}")
    else:
        logger.error("帳戶資訊獲取失敗。")

    # 測試獲取蠟燭圖
    candles = client.get_candles("EUR_USD", "M5", count=5)
    if candles:
        logger.info(f"獲取到 {len(candles)} 根 EUR_USD M5 蠟燭圖數據。")
        logger.info(f"最新一根蠟燭圖: {candles[-1]}")
    else:
        logger.error("蠟燭圖數據獲取失敗。")

    # 測試獲取倉位
    positions = client.get_open_positions()
    if positions is not None:
        if positions:
            logger.info(f"當前持有 {len(positions)} 個倉位。")
            for pos in positions:
                logger.info(f"  - {pos['instrument']}: {pos['long']['units']} (L) / {pos['short']['units']} (S)")
        else:
            logger.info("當前無任何倉位。")
    else:
        logger.error("倉位資訊獲取失敗。")
