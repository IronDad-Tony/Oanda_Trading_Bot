import os
import threading
import time
import asyncio
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from oanda_trading_bot.live_trading_system.main import initialize_system, trading_loop


app = FastAPI(title="Live Trading API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AppState:
    def __init__(self):
        self.components: Optional[Dict[str, Any]] = None
        self.engine_thread: Optional[threading.Thread] = None
        self.ws_clients: List[WebSocket] = []
        self.ws_lock = threading.Lock()
        self.ws_task: Optional[asyncio.Task] = None

    def ensure_components(self):
        if self.components is None:
            self.components = initialize_system()
        return self.components


STATE = AppState()


# ----- Models -----
class SessionConfig(BaseModel):
    symbols: List[str]
    model_path: Optional[str] = None
    target_capital_usd: Optional[float] = None
    risk: Optional[Dict[str, Any]] = None


# ----- Helpers -----
def _scan_models(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    candidates: List[str] = []
    for d in ["trained_models", "weights"]:
        p = os.path.join(root, d)
        if os.path.isdir(p):
            for fn in os.listdir(p):
                if fn.lower().endswith((".zip", ".pth", ".pt")):
                    candidates.append(os.path.join(p, fn))
    out: List[Dict[str, Any]] = []
    max_symbols_fallback = config.get('max_symbols_allowed', 10) if isinstance(config, dict) else 10
    lookback = config.get('model_lookback_window', 128) if isinstance(config, dict) else 128
    for path in candidates:
        name = os.path.basename(path)
        max_symbols = max_symbols_fallback
        try:
            import re
            m = re.search(r"symbols(\d+)", name, re.IGNORECASE)
            if m:
                max_symbols = int(m.group(1))
        except Exception:
            pass
        out.append({
            "name": name,
            "path": path,
            "max_symbols": max_symbols,
            "lookback_window": lookback,
            "type": "SAC" if "sac" in name.lower() else "Torch",
        })
    return out


async def _broadcaster():
    while True:
        try:
            comps = STATE.components
            if comps:
                # Status frame
                ss = comps["system_state"]
                data = {
                    "running": getattr(ss, 'is_running', False),
                    "selected_symbols": ss.get_selected_instruments(),
                    "model": ss.get_current_model(),
                }
                await _ws_broadcast({"type": "status", "data": data})

                # Account frame
                try:
                    acc = comps["client"].get_account_summary() or {}
                    await _ws_broadcast({"type": "account", "data": acc})
                except Exception:
                    pass
        except Exception:
            pass
        await asyncio.sleep(2.0)


async def _ws_broadcast(obj: Dict[str, Any]):
    text = None
    try:
        import json
        text = json.dumps(obj)
    except Exception:
        return
    dead: List[WebSocket] = []
    with STATE.ws_lock:
        clients = list(STATE.ws_clients)
    for ws in clients:
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    if dead:
        with STATE.ws_lock:
            STATE.ws_clients = [w for w in STATE.ws_clients if w not in dead]


# ----- Routes -----
@app.on_event("startup")
async def _startup():
    comps = STATE.ensure_components()
    # Start broadcaster task once
    if STATE.ws_task is None:
        STATE.ws_task = asyncio.create_task(_broadcaster())


@app.get("/health")
def health():
    comps = STATE.ensure_components()
    running = getattr(comps["system_state"], 'is_running', False)
    return {"status": "ok", "engine": "running" if running else "stopped"}


@app.get("/account/summary")
def account_summary():
    comps = STATE.ensure_components()
    return comps["client"].get_account_summary() or {}


@app.get("/account/positions")
def positions():
    comps = STATE.ensure_components()
    data = comps["client"].get_open_positions() or {}
    out: List[Dict[str, Any]] = []
    for pos in data.get("positions", []):
        ins = pos.get("instrument")
        long_units = int(pos.get("long", {}).get("units", "0"))
        short_units = int(pos.get("short", {}).get("units", "0"))
        if long_units:
            out.append({
                "instrument": ins,
                "units": long_units,
                "side": "BUY",
                "avgPrice": float(pos.get("long", {}).get("averagePrice", 0) or 0),
                "unrealizedPL": float(pos.get("long", {}).get("unrealizedPL", 0) or 0),
            })
        if short_units:
            out.append({
                "instrument": ins,
                "units": short_units,
                "side": "SELL",
                "avgPrice": float(pos.get("short", {}).get("averagePrice", 0) or 0),
                "unrealizedPL": float(pos.get("short", {}).get("unrealizedPL", 0) or 0),
            })
    return out


@app.get("/account/trades")
def trades(limit: int = 100):
    comps = STATE.ensure_components()
    return comps["db_manager"].get_trade_history(limit=limit) or []


@app.get("/symbols/training")
def training_symbols():
    comps = STATE.ensure_components()
    cfg = comps["config"] or {}
    syms = cfg.get("trading_instruments", [])
    return {"symbols": syms}


@app.get("/models")
def models(min_symbols: int = 1):
    comps = STATE.ensure_components()
    cfg = comps["config"] or {}
    items = _scan_models(cfg)
    return [m for m in items if (m.get("max_symbols", 0) >= int(min_symbols))]


@app.get("/session/status")
def session_status():
    comps = STATE.ensure_components()
    ss = comps["system_state"]
    tl = comps["trading_logic"]
    instruments = ss.get_selected_instruments()
    lookback = int(comps["config"].get("model_lookback_window", 128))
    warm = True
    missing: List[str] = []
    for inst in instruments:
        buf = tl.data_buffers.get(inst)
        if not buf or len(buf) < lookback:
            warm = False
            missing.append(inst)
    risk = comps["risk_manager"].config if hasattr(comps["risk_manager"], 'config') else {}
    return {
        "running": getattr(ss, 'is_running', False),
        "selected_symbols": instruments,
        "model": ss.get_current_model(),
        "risk": risk,
        "warmup": {"ready": warm, "missing": missing}
    }


@app.post("/session/config")
def session_config(body: SessionConfig):
    comps = STATE.ensure_components()
    ss = comps["system_state"]
    tl = comps["trading_logic"]
    pred = comps["prediction_service"]
    risk = comps["risk_manager"]
    client = comps["client"]
    cfg = comps["config"]

    # Validate symbols
    training_syms = set(cfg.get("trading_instruments", []))
    if not set(body.symbols).issubset(training_syms):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Symbols not in training set")

    # Validate model capacity
    models = _scan_models(cfg)
    model_info = next((m for m in models if m["path"] == body.model_path), None) if body.model_path else None
    if body.model_path and (not model_info or model_info.get("max_symbols", 0) < len(body.symbols)):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Model cannot handle selected symbol count")

    # Validate target capital against equity (in account currency)
    if body.target_capital_usd is not None:
        acc = client.get_account_summary() or {}
        equity = float((((acc.get('account') or {}).get('NAV')) or ((acc.get('account') or {}).get('equity')) or 0))
        if float(body.target_capital_usd) > equity:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="Target capital exceeds equity")

    # Apply to system
    ss.set_selected_instruments(body.symbols)
    if body.model_path:
        pred.load_model(body.model_path, device=None)
        ss.set_current_model(body.model_path)
    # Risk updates (also map target capital to exposure cap)
    to_update = dict(body.risk or {})
    if body.target_capital_usd is not None:
        to_update['max_total_exposure_usd'] = float(body.target_capital_usd)
    risk.update_params(to_update)

    # Reset buffers to force clean warmup if instruments changed
    for sym in body.symbols:
        if sym not in tl.data_buffers:
            tl.data_buffers[sym] = __import__('collections').deque(maxlen=int(cfg.get("model_lookback_window", 128)))
    return {"ok": True}


@app.post("/session/start")
def session_start(cold_start_timeout_sec: Optional[int] = None):
    comps = STATE.ensure_components()
    ss = comps["system_state"]
    tl = comps["trading_logic"]
    cfg = comps["config"]

    # Warmup
    instruments = ss.get_selected_instruments()
    timeout = int(cold_start_timeout_sec or 120)
    tl.warmup_buffers(instruments, max_wait_seconds=timeout, sleep_seconds=1)

    # Start engine loop
    ss.start()
    if not STATE.engine_thread or not STATE.engine_thread.is_alive():
        STATE.engine_thread = threading.Thread(target=trading_loop, args=(comps,), daemon=True)
        STATE.engine_thread.start()
    return {"started": True}


@app.post("/session/stop")
def session_stop():
    comps = STATE.ensure_components()
    ss = comps["system_state"]
    ss.stop()
    th = STATE.engine_thread
    if th and th.is_alive():
        th.join(timeout=5)
    return {"stopped": True}


@app.post("/risk/update")
def risk_update(body: Dict[str, Any]):
    comps = STATE.ensure_components()
    comps["risk_manager"].update_params(body or {})
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    with STATE.ws_lock:
        STATE.ws_clients.append(ws)
    try:
        while True:
            # We don't expect incoming messages; keep alive by waiting for pings
            await ws.receive_text()
    except WebSocketDisconnect:
        with STATE.ws_lock:
            STATE.ws_clients = [w for w in STATE.ws_clients if w is not ws]


def main():
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

