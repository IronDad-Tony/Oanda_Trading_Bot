Live Trading System v2 — Architecture Overview

Goals
- Robust live trading with dynamic symbol/model selection, cold start, and strong risk controls.
- Decoupled backend (Python) and frontend (your choice: Next.js/React or Streamlit Pro UI).
- Real‑time updates via WebSocket; auditable storage via SQLite.

High‑Level Components
- Live API (FastAPI): REST + WebSocket gateway over existing trading engine.
- Trading Engine: Reuse/refactor current modules (client, preprocessor, prediction, risk, orders, DB, logic).
- Model Registry: Discover models and introspect capacity (max symbols, lookback) on demand.
- Frontend UI: Modern dashboard, controls, and monitoring (stack to be decided).
- Storage: SQLite (trades, runs, metrics). CSV for quick export.

Key Data Flows
- UI selects: symbols -> Live API validates list = training set; returns compatible models.
- UI selects model + target capital + risk params -> Live API updates RiskManager + session config.
- Cold start: TradingLogic warmup fills buffers by lookback with history; engine begins cycles when full.
- Engine loop: fetch latest data -> preprocess -> predict -> risk assess -> send orders -> record to DB.
- Live telemetry: engine emits account/positions/orders/metrics over WebSocket to UI.

Risk Controls (runtime adjustable)
- Per‑trade risk % of equity; portfolio max exposure (USD); per‑symbol position limit; max concurrent positions.
- SL/TP via pips or ATR multipliers; slippage guard via priceBound; min interval between orders per symbol.
- Circuit breakers: daily loss limit, max drawdown for session, spread/volatility guards, trading windows.

Sessions
- Start/Stop endpoints manage a single active session; session status persists current config and last health.

Cold Start
- For each instrument: fetch `lookback_window` bid/ask candles; only when all buffers are full does trading start.
- If partial, continue attempting until timeout, then surface non‑ready instruments in status.

Observability
- DB: trades table; add runs, metrics (PnL, equity, DD) as needed.
- Logs: structured logging with subsystem tags.

Security
- Never expose API keys to frontend; server‑side only.
- Practice/live environment switch via .env; explicit confirmation before any live execution.

