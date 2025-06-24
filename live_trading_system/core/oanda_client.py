# live_trading_system/core/oanda_client.py
"""
Oanda API 封裝客戶端
"""
import os
import time
import requests
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List

import logging
from dotenv import load_dotenv

# 獲取日誌記錄器
logger = logging.getLogger("LiveTradingSystem")

# 載入環境變數
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

import oandapyV20
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.positions as positions
from tenacity import retry, stop_after_attempt, wait_fixed

from live_trading_system.core.system_state import SystemState

class OandaClient:
    """
    一個封裝了 Oanda V20 API 請求的客戶端，包含錯誤處理和重試機制。
    """
    def __init__(self):
        self.account_id = os.getenv("OANDA_ACCOUNT_ID")
        self.access_token = os.getenv("OANDA_ACCESS_TOKEN")
        self.base_url = "https://api-fxpractice.oanda.com/v3"  # 使用模擬盤 Practice URL
        
        if not self.account_id or not self.access_token:
            logger.critical("OANDA_ACCOUNT_ID 或 OANDA_ACCESS_TOKEN 未在 .env 檔案中設定。")
            raise ValueError("API credentials not found in .env file")

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339"
        })

    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None, data: Optional[Dict] = None, max_retries: int = 3, backoff_factor: float = 0.5) -> Optional[Dict[str, Any]]:
        """通用請求方法，包含重試邏輯。"""
        url = f"{self.base_url}{endpoint}"
        for attempt in range(max_retries):
            try:
                response = self.session.request(method, url, params=params, json=data)
                response.raise_for_status()  # 如果是 4xx 或 5xx 錯誤，會拋出 HTTPError
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"請求失敗 (嘗試 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(backoff_factor * (2 ** attempt))
                else:
                    logger.error(f"請求 {url} 在 {max_retries} 次嘗試後最終失敗。")
                    return None

    def get_account_summary(self) -> Optional[Dict[str, Any]]:
        """獲取帳戶摘要資訊。"""
        endpoint = f"/accounts/{self.account_id}/summary"
        return self._request("GET", endpoint)

    def get_candles(self, instrument: str, granularity: str, count: int = 100, price: str = "M") -> Optional[List[Dict[str, Any]]]:
        """獲取最新的蠟燭圖數據。 price: 'M' (Midpoint), 'B' (Bid), 'A' (Ask)"""
        endpoint = f"/instruments/{instrument}/candles"
        params = {
            "granularity": granularity,
            "count": count,
            "price": price
        }
        data = self._request("GET", endpoint, params=params)
        return data.get("candles") if data else None

    def create_order(self, instrument: str, units: int, stop_loss_on_fill: Optional[Dict] = None, take_profit_on_fill: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """建立一個市價單。"""
        endpoint = f"/accounts/{self.account_id}/orders"
        order_data = {
            "order": {
                "instrument": instrument,
                "units": str(units), # 單位必須是字串
                "type": "MARKET",
                "timeInForce": "FOK"  # Fill Or Kill
            }
        }
        if stop_loss_on_fill:
            order_data["order"]["stopLossOnFill"] = stop_loss_on_fill
        if take_profit_on_fill:
            order_data["order"]["takeProfitOnFill"] = take_profit_on_fill
            
        return self._request("POST", endpoint, data=order_data)

    def get_open_positions(self) -> Optional[List[Dict[str, Any]]]:
        """獲取所有未平倉部位。"""
        endpoint = f"/accounts/{self.account_id}/openPositions"
        data = self._request("GET", endpoint)
        return data.get("positions") if data else None

    def close_position(self, instrument: str, long_units: str = "ALL", short_units: str = "ALL") -> Optional[Dict[str, Any]]:
        """平掉一個交易對的所有倉位。"""
        endpoint = f"/accounts/{self.account_id}/positions/{instrument}/close"
        data = {
            "longUnits": long_units,
            "shortUnits": short_units
        }
        return self._request("PUT", endpoint, data=data)

# --- 範例 ---
if __name__ == '__main__':
    logger.info("正在測試 OandaClient...")
    client = OandaClient()

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
