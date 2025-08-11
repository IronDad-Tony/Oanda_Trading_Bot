# live_trading_system/core/system_state.py
"""
全局狀態管理器

提供一個單例模式的類來存儲和管理整個應用程序的共享狀態。
這有助於避免將狀態變量作為參數在多個模組之間傳遞。
"""
from typing import Dict, Any, Optional, List
import logging

# 嘗試導入 InstrumentInfoManager，以便快取標的資訊
try:
    from oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager
except ImportError:
    # 在測試或不同環境下可能無法導入，提供一個假類
    class InstrumentInfoManager:
        pass


class SystemState:
    """
    一個用於存儲應用程序全局狀態的單例类。
    
    屬性:
        is_running (bool): 標識交易主循環是否應該運行。
        ui_controls_enabled (bool): 控制UI上的啟動/停止按鈕是否可用。
        last_error (str | None): 存儲最近一次發生的嚴重錯誤訊息。
        status_message (str): 顯示在UI上的當前系統狀態訊息。
        selected_instruments (list[str]): 當前選擇的交易標的列表。
        current_model (str | None): 當前選擇的模型文件。
        instrument_info_manager (InstrumentInfoManager | None): 用於獲取和管理標的資訊的實例。
        categorized_instruments (Dict[str, Any]): 快取分類後的標的資訊。
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SystemState, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化所有狀態變量。"""
        self.is_running = False
        self.ui_controls_enabled = True
        self.last_error: str | None = None
        self.status_message = "系統已就緒，等待啟動。"
        self.selected_instruments: list[str] = ["EUR_USD"]
        self.current_model: str | None = "model_v1.pth"
        self.instrument_info_manager: Optional[InstrumentInfoManager] = None
        self.categorized_instruments: Dict[str, Any] = {}

    def get_instrument_manager(self) -> InstrumentInfoManager:
        """獲取 InstrumentInfoManager 的單例，並傳遞正確的憑證。"""
        # 正確獲取已配置的 logger
        logger = logging.getLogger("LiveTradingSystem")
        if self.instrument_info_manager is None:
            try:
                from dotenv import load_dotenv
                import os
                from oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager as ActualIIM

                # 從 live system 的 .env 檔案載入憑證
                dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
                if not os.path.exists(dotenv_path):
                    logger.error(f".env file not found at {dotenv_path}")
                load_dotenv(dotenv_path=dotenv_path)
                
                api_key = os.getenv("OANDA_API_KEY")
                account_id = os.getenv("OANDA_ACCOUNT_ID")
                environment = os.getenv("OANDA_ENVIRONMENT", "practice")
                base_url = "https://api-fxtrade.oanda.com/v3" if environment == "live" else "https://api-fxpractice.oanda.com/v3"

                if not api_key or not account_id:
                    logger.error("OANDA_API_KEY or OANDA_ACCOUNT_ID not found in .env file.")
                    raise ValueError("OANDA credentials not found.")

                self.instrument_info_manager = ActualIIM(
                    api_key=api_key,
                    account_id=account_id,
                    base_url=base_url,
                    force_refresh=True  # 首次啟動時強制刷新
                )
            except (ImportError, ValueError) as e:
                logger.error(f"Failed to import or initialize InstrumentInfoManager: {e}", exc_info=True)
                # 提供一個假的管理器以防止 UI 崩潰
                class FakeIIM:
                    def get_all_available_symbols(self): return ["EUR_USD", "GBP_USD"] # 提供假數據
                    def get_details(self, symbol): return None
                self.instrument_info_manager = FakeIIM()
        return self.instrument_info_manager

    def get_selected_instruments(self) -> list[str]:
        """獲取當前選擇的交易標的列表。"""
        return self.selected_instruments

    def set_selected_instruments(self, instruments: list[str]):
        """設置當前選擇的交易標的列表。"""
        self.selected_instruments = instruments
        self.status_message = f"Instrument selection changed to {instruments}"

    def get_current_model(self) -> str | None:
        """獲取當前加載的模型。"""
        return self.current_model

    def set_current_model(self, model_path: str):
        """設置當前要使用的模型。"""
        self.current_model = model_path
        self.status_message = f"Model changed to {model_path}"

    def start(self):
        """啟動系統。"""
        self.is_running = True
        self.status_message = "系統正在運行..."
        self.last_error = None

    def stop(self):
        """請求停止交易循環。"""
        self.is_running = False
        self.status_message = "系統正在停止..."

    def set_error(self, message: str):
        """記錄一個嚴重錯誤並停止系統。"""
        self.last_error = message
        self.status_message = f"錯誤: {message}"
        self.is_running = False

# 創建一個全局可用的實例
system_state = SystemState()

# --- 範例 ---
if __name__ == '__main__':
    # 由於是單例，這兩個變量指向同一個對象
    state1 = SystemState()
    state2 = SystemState()

    print(f"初始狀態: {state1.status_message}")
    print(f"state1 is state2: {state1 is state2}")

    state1.start()
    print(f"啟動後 (state1): {state1.status_message}")
    print(f"啟動後 (state2): {state2.status_message}")

    state2.set_error("Oanda API 連接失敗")
    print(f"發生錯誤後 (state1): {state1.status_message}")
    print(f"is_running (state2): {state2.is_running}")
