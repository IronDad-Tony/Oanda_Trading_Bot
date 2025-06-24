# live_trading_system/core/system_state.py
"""
全局狀態管理器

提供一個單例模式的類來存儲和管理整個應用程序的共享狀態。
這有助於避免將狀態變量作為參數在多個模組之間傳遞。
"""

class SystemState:
    """
    一個用於存儲應用程序全局狀態的單例類。
    
    屬性:
        is_running (bool): 標識交易主循環是否應該運行。
        ui_controls_enabled (bool): 控制UI上的啟動/停止按鈕是否可用。
        last_error (str | None): 存儲最近一次發生的嚴重錯誤訊息。
        status_message (str): 顯示在UI上的當前系統狀態訊息。
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
        self.selected_instruments: list[str] = [] # 新增

    def set_selected_instruments(self, instruments: list[str]):
        """設置當前選擇的交易標的。"""
        self.selected_instruments = instruments

    def get_selected_instruments(self) -> list[str]:
        """獲取當前選擇的交易標的。"""
        return self.selected_instruments

    def start(self):
        """請求啟動交易循環。"""
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
