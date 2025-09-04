Live Trading API — Draft Spec

Base
- REST: http://localhost:8000
- WS:   ws://localhost:8000/ws

Auth
- Local only for now. Optional token header later.

Endpoints
- GET /health
  - 200 { status: "ok", engine: "stopped|running" }

- GET /account/summary
  - 200 { accountId, balance, NAV, equity, pl, marginUsed, openPositionCount, lastTransactionID }

- GET /account/positions
  - 200 [ { instrument, units, side, avgPrice, unrealizedPL, ... } ]

- GET /account/trades?limit=100
  - 200 [ { id, instrument, units, entry_price, status, pnl, open_timestamp, close_timestamp, close_price } ]

- GET /symbols/training
  - 200 { symbols: ["EUR_USD", ...] }

- GET /symbols/oanda
  - 200 { instruments: [...] } (subset at first; used for info panels)

- GET /models?min_symbols=5
  - 200 [ { name, path, max_symbols, lookback_window, type: "SAC|Torch", device: "cpu|cuda?" } ]
  - Implementation notes: Introspect model on first request and cache results.

- GET /session/status
  - 200 { running: true|false, selected_symbols: [...], model: {...}, risk: {...}, warmup: { ready: true|false, missing: [..] } }

- POST /session/config
  - body: { symbols: [..], model_path, target_capital_usd, risk: {...} }
  - 200 { ok: true }
  - Validates: symbols in training set; model capacity >= symbols.length; target_capital <= account balance.

- POST /session/start
  - body: { cold_start_timeout_sec?: 120 }
  - 202 { started: true }

- POST /session/stop
  - 200 { stopped: true }

- POST /risk/update
  - body: { max_total_exposure_usd?, max_risk_per_trade_percent?, use_atr_sizing?, atr_period?, stop_loss_atr_multiplier?, take_profit_atr_multiplier?, stop_loss_pips?, take_profit_pips?, daily_loss_limit_pct?, max_concurrent_positions?, per_symbol_limit? }
  - 200 { ok: true }

WebSocket /ws
- Server pushes JSON frames:
  - { type: "status", data: {...} }
  - { type: "account", data: {...} }
  - { type: "positions", data: [...] }
  - { type: "order", data: {...} }
  - { type: "metric", data: { equity, pnl, drawdown, timestamp } }
  - { type: "log", data: { level, message, timestamp } }

Notes
- Practice/live switching remains server‑side; frontend never sees API keys.
- Long‑running operations (start/stop) return quickly; UI subscribes to /ws for progress.

