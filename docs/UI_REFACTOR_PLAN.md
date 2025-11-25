### Live Trading 系統 UI/UX 重新設計計劃

#### 1. 設計理念與核心原則

重新設計的目標是創建一個專業、高資訊密度、反應迅速且絕對安全的**交易指揮中心 (Trading Command Center)**。

*   **清晰度與資訊密度 (Clarity & Information Density):** 在單一螢幕上提供最重要的資訊，但避免混亂。使用清晰的視覺層次結構，讓交易員能一眼掃描到關鍵狀態。
*   **即時性 (Real-Time):** 所有關鍵數據（市場報價、倉位盈虧、帳戶淨值）必須是即時更新的，延遲應在毫秒級。
*   **可操作性與控制 (Actionability & Control):** UI 不僅僅是用於監控，更要提供強大、直觀的控制功能，特別是風險控制。
*   **模組化與可擴展性 (Modularity & Scalability):** 介面應由獨立的、可重用的組件構成，方便未來增加新的功能模組（例如：更多分析指標、新聞情緒分析等）。
*   **安全第一 (Safety First):** 所有具有風險的操作（如平倉、停止策略）都必須有清晰的標示和確認機制，防止誤操作。

#### 2. 儀表板 (Dashboard) 佈局與核心組件

建議採用多分欄、基於小工具 (Widget-based) 的佈局，允許用戶未來進行一定程度的自訂。

##### 組件 1: 全局狀態與緊急控制欄 (Global Status & Emergency Control Bar)

這是儀表板最頂層、永遠可見的區域。

*   **系統狀態 (System Status):**
    *   **OANDA 連線狀態:** `CONNECTED` (綠色) / `CONNECTING` (黃色) / `DISCONNECTED` (紅色)。
    *   **交易模型狀態:** `ACTIVE` (綠色) / `PAUSED` (黃色) / `STOPPED` (紅色) / `ERROR` (紅色)。
*   **帳戶核心指標 (Account Metrics):**
    *   **淨值 (Equity):** 帳戶即時總價值。
    *   **餘額 (Balance):** 已平倉的帳戶價值。
    *   **已用保證金 (Margin Used):** 即時顯示保證金佔用率。
    *   **今日盈虧 (Today's P/L):** 顯示金額和百分比。
*   **緊急控制按鈕 (Emergency Controls):**
    *   **緊急全平倉 (EMERGENCY LIQUIDATE ALL):** 一個非常醒目的紅色按鈕。點擊後會彈出確認對話框，要求輸入 "LIQUIDATE" 之類的文字才能執行，以防止誤觸。此操作會立即市價平掉所有倉位。
    *   **暫停新開倉 (PAUSE NEW TRADES):** 一個黃色按鈕，點擊後系統將不再根據模型信號開設新倉位，但會繼續管理現有倉位（例如：追蹤止損）。再次點擊可恢復。

##### 組件 2: 倉位管理 (Position Management)

以表格形式顯示所有當前持有的倉位。

*   **欄位:**
    *   `商品 (Instrument)`: 例如 EUR_USD。
    *   `方向 (Direction)`: `LONG` (綠色) / `SHORT` (紅色)。
    *   `數量 (Size)`: 持有單位。
    *   `開倉價格 (Entry Price)`:
    *   `當前價格 (Current Price)`: 即時跳動。
    *   `未實現盈虧 (Unrealized P/L)`: 以點數 (Pips) 和帳戶貨幣（例如 USD）兩種方式顯示，即時更新。
    *   `止盈/止損 (TP/SL)`: 顯示設定的價格。
*   **操作:**
    *   每一行倉位旁都有一個小的 `[X]` 按鈕，用於**單獨平倉**。同樣需要彈窗確認。

##### 組件 3: 互動式圖表 (Interactive Charting)

這是儀表板的核心視覺組件。

*   **技術:** 應使用專業的圖表庫（如 TradingView Lightweight Charts）。
*   **功能:**
    *   **商品選擇:** 下拉式選單或搜尋框，可快速切換商品。
    *   **時間週期:** M5, H1, D1 等多時間週期切換。
    *   **指標顯示:** 可疊加顯示模型使用的關鍵指標（如均線、RSI）。
    *   **倉位標示:** 在圖表上直接標示出當前倉位的開倉點、止盈線和止損線。

##### 組件 4: 模型性能分析 (Model Performance & Analytics)

顯示交易策略的長期和短期表現。

*   **關鍵績效指標 (KPIs):**
    *   `勝率 (Win Rate)`
    *   `盈虧比 (Profit Factor)`
    *   `夏普比率 (Sharpe Ratio)`
    *   `最大回撤 (Max Drawdown)`
    *   可選擇時間範圍：`今日 (Today)`, `本週 (This Week)`, `本月 (This Month)`, `全部 (All Time)`。
*   **淨值曲線 (Equity Curve):**
    *   一個顯示帳戶淨值隨時間增長的圖表，這是評估策略最重要的視覺工具。

##### 組件 5: 日誌與事件流 (Log & Event Stream)

以時間倒序顯示系統運行的所有關鍵日誌。

*   **日誌內容:**
    *   `[INFO]`: `Executing BUY order for EUR_USD at 1.0750.`
    *   `[SIGNAL]`: `Model prediction for EUR_USD: STRONG_BUY.`
    *   `[WARN]`: `High spread detected for GBP_JPY.`
    *   `[ERROR]`: `Failed to connect to OANDA API: Authentication failed.`
*   **功能:**
    *   可按日誌級別 (INFO, WARN, ERROR) 進行篩選。
    *   可搜尋關鍵字。

#### 3. 技術棧建議 (Tech Stack Recommendation)

為了實現高度互動和即時性的要求，建議採用現代前端框架。

*   **前端 (Frontend):**
    *   **框架:** **React** 或 **Vue.js** (使用 TypeScript 以增強代碼健壯性)。
    *   **狀態管理:** Redux Toolkit (React) 或 Pinia (Vue)，用於集中管理複雜的應用狀態。
    *   **UI 組件庫:** Ant Design 或 Material-UI，提供專業且一致的 UI 組件。
    *   **圖表庫:** **TradingView Lightweight Charts** (免費且功能強大)。
*   **後端 (Backend):**
    *   繼續使用 Python，因為交易邏輯和 OANDA API 的整合已經存在。
    *   需要建立一個 API 層來與前端通訊。
*   **通訊 (Communication):**
    *   **WebSockets:** 用於後端向前端**即時推送**數據（市場報價、帳戶更新、倉位變化、日誌）。Python 的 `FastAPI` 或 `Tornado` 很適合實現 WebSocket 伺服器。
    *   **REST API:** 用於前端向後端**發送指令**（緊急平倉、暫停/恢復、更新設定等）。

#### 4. 前後端 API 互動規範 (API Specification)

##### WebSocket 通道 (Server -> Client)

後端應建立以下通道，主動向前端推送數據：

*   `market_data`: 推送訂閱商品的即時報價。
    *   `{ "instrument": "EUR_USD", "bid": 1.0750, "ask": 1.0752, "timestamp": ... }`
*   `account_update`: 推送帳戶核心指標的變化。
    *   `{ "equity": 10500.50, "balance": 10480.00, "marginUsed": 50.20, ... }`
*   `positions_update`: 推送完整的當前倉位列表（任何倉位變化時觸發）。
    *   `[ { "instrument": "EUR_USD", "direction": "LONG", "size": 1000, ... }, ... ]`
*   `log_stream`: 推送新的日誌條目。
    *   `{ "level": "INFO", "message": "...", "timestamp": ... }`

##### REST API 端點 (Client -> Server)

前端透過這些端點向後端發送控制指令：

*   `POST /api/v1/actions/emergency_stop`: 觸發緊急全平倉。
*   `POST /api/v1/actions/pause_trading`: 暫停新交易。
*   `POST /api/v1/actions/resume_trading`: 恢復新交易。
*   `DELETE /api/v1/positions/{position_id}`: 關閉指定的單一倉位。
*   `GET /api/v1/config`: 獲取當前交易模型的參數配置。
*   `PUT /api/v1/config`: 更新交易模型的參數（例如：風險比例）。
