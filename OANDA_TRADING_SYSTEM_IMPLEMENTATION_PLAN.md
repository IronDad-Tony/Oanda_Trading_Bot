# Oanda 量化交易執行系統：設計與實施計畫

## 1. 核心設計理念

1.  **獨立性 (Independence)**：建立一個全新的、與現有訓練系統完全隔離的專案目錄 (`live_trading_system`)。除了共享訓練好的模型權重檔案外，不共享任何程式碼或設定檔，避免任何潛在的互相干擾。
2.  **穩健性 (Robustness)**：系統的核心是穩定壓倒一切。必須包含全面的錯誤處理、重試機制、API 延遲偵測、以及與 Oanda 伺服器的狀態同步機制。
3.  **即時性 (Real-Time)**：所有帳戶狀態（如餘額、倉位、保證金）和市場數據都必須直接透過 Oanda API 取得，而不是基於本地計算的估算值。
4.  **可控性 (Controllability)**：提供一個清晰、直觀的 UI 介面，讓您可以隨時啟動/停止交易、監控系統狀態、手動干預，並能應對突發狀況。
5.  **可觀測性 (Observability)**：詳盡的日誌記錄是系統的黑盒子。記錄所有決策、API 請求/回應、訂單狀態變化和潛在錯誤，以便於事後分析和除錯。

## 2. 技術棧選擇

*   **後端/核心邏輯**：Python
*   **API 互動**：`oandapyV20` (推薦，專為 Oanda V20 API 設計)
*   **使用者介面 (UI)**：Streamlit (與您現有技術棧一致)
*   **數據庫**：SQLite (用於記錄交易歷史和系統事件)
*   **設定管理**：`.env` (儲存金鑰) + `live_config.json` (儲存策略與風險參數)

## 3. 專案結構規劃

```
live_trading_system/
├── .env                    # 環境變數 (API 金鑰, 帳戶 ID)
├── live_config.json        # 實盤交易設定檔
├── requirements.txt        # 專案依賴
├── main.py                 # 系統主啟動腳本
├── core/
│   ├── __init__.py
│   ├── logger_setup.py     # 日誌設定模組
│   ├── oanda_client.py     # Oanda API 封裝客戶端
│   └── system_state.py     # 全局狀態管理器
├── data/
│   ├── __init__.py
│   ├── data_preprocessor.py # 實時數據預處理器
│   └── instrument_monitor.py # 交易標的健康檢查器
├── model/
│   ├── __init__.py
│   └── prediction_service.py # 模型加載與預測服務
├── trading/
│   ├── __init__.py
│   ├── trading_logic.py    # 交易決策邏輯
│   ├── risk_manager.py     # 風險管理器
│   ├── order_manager.py      # 訂單管理器 (OMS)
│   └── position_manager.py # 倉位管理器
├── database/
│   ├── __init__.py
│   ├── database_manager.py # 數據庫管理
│   └── trading_history.db  # 交易歷史數據庫
├── ui/
│   ├── __init__.py
│   ├── app.py              # Streamlit 主應用
│   └── dashboard.py        # 儀表板介面元件
└── tests/
    ├── __init__.py
    ├── test_oanda_client.py
    ├── test_prediction_service.py
    └── test_risk_manager.py
```

---

## 4. 參數與依賴關係設計

為確保模組間的低耦合與高內聚，我們定義以下依賴注入與參數傳遞流程：

1.  **啟動點 (`main.py`)**: 
    *   初始化核心物件：`logger`, `system_state`, `oanda_client`, `live_config`。
    *   將這些核心物件作為參數，注入到需要它們的管理類 (Managers) 中。

2.  **管理類 (Managers)**:
    *   `PositionManager(client)`: 依賴 `oanda_client` 來同步倉位。
    *   `RiskManager(config, position_manager, account_summary)`: 依賴設定檔、倉位管理器和即時帳戶資訊。
    *   `OrderManager(client, risk_manager, position_manager, db_manager)`: 依賴客戶端、風控、倉位和數據庫管理器。
    *   `TradingLogic(order_manager, prediction_service, preprocessor, ...)`: 依賴訂單管理器和模型服務。

3.  **UI (`app.py`)**:
    *   UI 不直接控制後端邏輯，而是透過 `system_state` 來傳遞指令 (啟動/停止)。
    *   UI 透過定期調用各個管理器的 `get_...` 方法來獲取數據並刷新介面，實現與後端的解耦。

---

## 5. 高級功能設計

### 5.1. 動態標的與模型選擇

**目標**: UI 上應能讓使用者自由選擇要交易的標的，並自動載入對應的模型。

*   **`live_config.json` 設計**:
    ```json
    {
      "available_models": {
        "EUR_USD": {
          "model_path": "weights/eur_usd_transformer_v1.pth",
          "feature_config": "configs/feature_eur_usd.json"
        },
        "USD_JPY": {
          "model_path": "weights/usd_jpy_transformer_v1.pth",
          "feature_config": "configs/feature_usd_jpy.json"
        }
      },
      "risk_parameters": {
        "max_position_size_per_trade": 1000,
        "max_total_risk_pct": 2.0
      }
    }
    ```
*   **UI (`app.py`)**: 
    *   啟動時讀取 `live_config.json` 的 `available_models`，生成一個多選框 (`st.multiselect`) 讓使用者勾選要交易的標的。
    *   將使用者選擇的標的列表存入 `system_state`。
*   **`trading_logic.py`**: 
    *   不再讀取固定的交易對列表，而是從 `system_state` 獲取使用者選擇的標的列表。
    *   對於每一個被選中的標的，指示 `prediction_service` 載入其在 `live_config.json` 中對應的 `model_path`。

### 5.2. 全局緊急控制

**目標**: UI 提供一鍵式緊急操作，以應對市場極端波動或系統異常。

*   **`system_state.py` 設計**:
    *   新增狀態旗標: `is_paused` (bool, 暫停開新倉), `emergency_stop_activated` (bool, 緊急停止並清倉)。
*   **UI (`app.py`)**: 
    *   新增三個按鈕：
        1.  **[暫停交易]**: 點擊後，設置 `system_state.is_paused = True`。交易邏輯將不再產生新的開倉訊號，但會繼續管理現有倉位。
        2.  **[恢復交易]**: 點擊後，設置 `system_state.is_paused = False`。
        3.  **[!! 緊急平倉並停止 !!]**: (紅色按鈕) 點擊後，執行以下操作：
            *   設置 `system_state.emergency_stop_activated = True`。
            *   設置 `system_state.is_running = False`。
            *   直接調用 `order_manager.close_all_positions()`。
*   **`trading_logic.py`**: 
    *   在產生交易訊號前，檢查 `if system_state.is_paused or system_state.emergency_stop_activated: return`。
*   **`order_manager.py`**: 
    *   實作 `close_all_positions()` 函式，該函式會從 `position_manager` 獲取所有未平倉部位，並逐一發送平倉訂單。

---

## 6. 開發階段、待辦事項與腳本詳解

### 階段一：建立基礎設施與 Oanda API 核心連接 (已完成)

**目標**：建立新系統的骨架，並確保與 Oanda API 的通訊穩定可靠。

**已完成**: 目錄結構、環境設定、`logger_setup.py`, `oanda_client.py`, `system_state.py`。

**測試機制 (已完成)**: `tests/test_oanda_client.py` 驗證了 API 連線、帳戶資訊獲取和錯誤處理。

---

### 階段二：模型整合與決策引擎

**目標**：將訓練好的模型載入，並根據即時數據產生交易訊號。

- **狀態**: ✅ 已完成
- **任務**:
    - ✅ `data/data_preprocessor.py`: 建立 `LivePreprocessor` 類，負責將 Oanda API 的即時蠟燭圖數據轉換為模型所需的標準化特徵。
        - ✅ 實現一個方法，可以載入在模型訓練期間保存的 `StandardScaler` 參數（`mean_` 和 `scale_`）。
        - ✅ 實現一個方法，接收 Oanda 蠟燭圖數據列表，計算所有必要的技術指標（與訓練時完全一致），並應用加載的 scaler 進行標準化。
        - ✅ 確保輸出是一個 PyTorch Tensor，其形狀為 `(1, num_features)`，準備好輸入模型。
    - ✅ `model/prediction_service.py`: 建立 `PredictionService` 類，負責載入訓練好的 PyTorch 模型並產生預測。
        - ✅ 實現動態加載 `live_config.json` 中指定的模型檔案（`.pth`）。
        - ✅ 提供一個 `predict` 方法，接收 `LivePreprocessor` 產生的特徵 Tensor，並返回模型的原始輸出（預測信號）。
    - ✅ `trading/trading_logic.py`: 建立 `TradingLogic` 類，作為核心的決策引擎。
        - ✅ 整合 `OandaClient`, `SystemState`, `LivePreprocessor`, 和 `PredictionService`。
        - ✅ 實現一個 `run_logic_cycle` 方法，執行單次的交易邏輯：獲取數據 -> 預處理 -> 預測 -> （暫時）打印信號。
    - ✅ `trading/order_manager.py`: 建立一個**佔位 (Placeholder)** 的 `OrderManager`。
        - ✅ 在此階段，它只需要接收來自 `TradingLogic` 的信號並記錄下來，而**不執行任何真實的下單操作**。
    - ✅ `tests/test_prediction_service.py`: 撰寫單元測試，驗證從 `LivePreprocessor` 到 `PredictionService` 的流程是否正確。
        - ✅ Mock Oanda API 的蠟燭圖數據。
        - ✅ 驗證 `LivePreprocessor` 是否能正確處理數據並產生正確形狀和類型的 Tensor。
        - ✅ 驗證 `PredictionService` 是否能成功加載模型並根據預處理後的特徵產生預測。

---

### 階段三：訂單管理與交易執行 (OMS)

**目標**：將交易決策轉化為可管理、有風控的真實訂單。

**狀態**: ✅ 已完成
- **任務**:
    - ✅ `trading/position_manager.py`: 建立 `PositionManager` 類，用於內部追踪和管理所有開倉部位。
        - ✅ 包含一個 `Position` 資料類來表示單個倉位。
        - ✅ 實現 `update_position`, `close_position`, `get_position`, `get_all_positions` 等方法。
    - ✅ `trading/risk_manager.py`: 建立 `RiskManager` 類，在下單前強制執行風險管理規則。
        - ✅ 從 `live_config.json` 加載風險參數（如：最大倉位大小、默認停損/停利點數）。
        - ✅ 實現 `assess_trade` 方法，根據當前倉位、信號和價格，決定是否批准交易，並返回訂單參數（單位、停損價、停利價）或 `None`。
    - ✅ `trading/order_manager.py`: **升級** `OrderManager`，使其具備完整功能。
        - ✅ 整合 `RiskManager` 和 `PositionManager`。
        - ✅ `process_signal` 流程：首先調用 `RiskManager` 進行評估 -> 如果批准，則調用 `OandaClient` 創建真實訂單 -> 如果訂單成交，則調用 `PositionManager` 更新持倉狀態。
        - ✅ 實現 `close_all_positions` 緊急功能，能市價平掉所有持倉。
    - ✅ `tests/test_risk_management.py`: 為 `RiskManager` 的核心邏輯（如交易批准、拒絕）撰寫單元測試。
    - ✅ `tests/test_order_management.py`: 為 `OrderManager` 撰寫整合測試，驗證其與 `RiskManager` 和 `PositionManager` 的協同工作是否正確（需 Mock `OandaClient`）。

---

### 階段四：Streamlit UI 介面與主程序

**目標**：建立一個直觀、可控的監控儀表板，並整合為可執行的主程序。

**狀態**: ✅ 已完成
- **任務**:
    - ✅ `database/database_manager.py`: 建立 `DatabaseManager` 類，使用 SQLite 來儲存所有已執行交易、訂單歷史和關鍵系統日誌。
        - ✅ 實現 `save_trade`, `save_log` 等方法。
        - ✅ 實現 `get_trade_history` 方法供 UI 調用。
    - ✅ `ui/dashboard.py`: 建立 Streamlit 儀表板的獨立功能元件。
        - ✅ `display_system_status`: 顯示系統運行狀態、當前標的、模型。
        - ✅ `display_open_positions`: 以表格形式顯示 `PositionManager` 中的即時倉位。
        - ✅ `display_trade_history`: 以表格形式顯示從 `DatabaseManager` 讀取的歷史交易。
        - ✅ `create_control_panel`: 建立側邊欄，包含系統啟動/停止按鈕、標的和模型選擇下拉框。
    - ✅ `ui/app.py`: 建立主 UI 應用程式。
        - ✅ 整合 `dashboard.py` 中的所有元件。
        - ✅ 調用 `main.py` 中的 `initialize_system` 來獲取所有系統模組的實例。
        - ✅ 在背景執行緒中啟動交易主循環 (`trading_logic.run()`)，避免 UI 阻塞。
        - ✅ 管理 Streamlit 的 `session_state`，以在頁面刷新間保持系統狀態。
    - ✅ `main.py`: **重構** `main.py` 以支援 UI 啟動。
        - ✅ 建立一個 `initialize_system` 函數，負責實例化並組裝所有系統元件（Client, State, Managers, Services, Logic 等），並將它們作為一個字典返回。
        - ✅ 主函數 `main()` 現在的職責是使用 `os.system` 或 `subprocess` 來執行 `streamlit run ui/app.py` 命令，從而啟動整個應用。

---

### 階段五：高級功能與健壯性增強

**目標**：處理真實世界的邊界情況，讓系統能 7x24 小時無人值守地運行。

**狀態**: ✅ 已完成
- **任務**:
    - ✅ **[腳本開發] `data/instrument_monitor.py`**:
        -   ✅ **Class `InstrumentMonitor`**:
            -   `check_tradable(self, instrument: str) -> bool`: 檢查特定標的是否可交易 (已透過 `get_candles` 間接實現)。
    - ✅ **[功能增強] `data/data_preprocessor.py`**: 加入數據新鮮度檢查。如果獲取的最新 K 線時間戳距離當前時間過久，則發出警告並跳過此次決策。
    - ✅ **[功能增強] `main.py`**: 在主循環中加入對 `KeyboardInterrupt` (Ctrl+C) 的捕捉，並執行優雅停機程序（自動平倉）。

**測試機制:**

*   **模擬場景測試 (Demo 帳戶)**: 手動模擬交易對不可交易、數據延遲、按下 Ctrl+C 等情況，驗證系統的反應是否符合預期。
*   **自動化整合測試**: 撰寫 `tests/test_full_system.py` 腳本，自動化驗證從啟動、交易到優雅停地的完整流程。
