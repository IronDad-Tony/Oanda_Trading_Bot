import os
import threading
import time
import asyncio
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from oanda_trading_bot.live_trading_system.main import initialize_system, trading_loop


app = FastAPI(title="Live Trading API", version="0.2.0")
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
def _to_int_units(val) -> int:
    try:
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            return int(val)
        s = str(val).strip()
        return int(float(s))
    except Exception:
        return 0

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
                    # Use global guard for suspended symbols
                    "suspended": __import__('oanda_trading_bot.live_trading_system.core.tradability_guard', fromlist=['get_all']).get_all(),
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

    # Mount static UI (served at "/")
    try:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui"))
        if os.path.isdir(root):
            # Only mount once
            mounted = any(hasattr(r, 'app') and isinstance(r.app, StaticFiles) for r in app.routes)
            if not mounted:
                app.mount("/", StaticFiles(directory=root, html=True), name="ui")
    except Exception:
        pass


@app.get("/ready")
def ready():
    # Lightweight readiness probe: do NOT initialize full components
    return {"ready": True}


@app.get("/health")
def health():
    # Keep health light; only report initialization status and engine state if available
    comps = STATE.components
    running = bool(comps and getattr(comps.get("system_state", object()), 'is_running', False))
    return {"status": "ok", "engine": "running" if running else "stopped", "initialized": bool(comps)}


@app.get("/account/summary")
def account_summary():
    comps = STATE.ensure_components()
    return comps["client"].get_account_summary() or {}


@app.get("/account/currency")
def account_currency():
    comps = STATE.ensure_components()
    acc = comps["client"].get_account_summary() or {}
    ccy = ((((acc.get('account') or {}).get('currency')) or '') or '').upper()
    return {"currency": ccy or (comps["config"].get("account_currency") or "USD")}


@app.get("/account/positions")
def positions():
    comps = STATE.ensure_components()
    data = comps["client"].get_open_positions() or {}
    out: List[Dict[str, Any]] = []
    for pos in data.get("positions", []):
        ins = pos.get("instrument")
        long_units = _to_int_units(pos.get("long", {}).get("units", "0"))
        short_units = _to_int_units(pos.get("short", {}).get("units", "0"))
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


@app.get("/symbols/available")
def available_symbols():
    """Return a comprehensive symbol list from InstrumentInfoManager cache (OANDA account)."""
    comps = STATE.ensure_components()
    # Reuse the RiskManager's InstrumentInfoManager if available
    iim = None
    try:
        iim = getattr(comps.get("risk_manager"), 'iim', None)
    except Exception:
        iim = None
    try:
        if iim is None:
            from oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager
            iim = InstrumentInfoManager()
        syms = iim.all_symbols if hasattr(iim, 'all_symbols') else (iim.get_all_available_symbols() or [])
        return {"symbols": syms}
    except Exception:
        # Fallback to training set
        cfg = comps["config"] or {}
        return {"symbols": cfg.get("trading_instruments", [])}


@app.get("/symbols/catalog")
def symbols_catalog():
    """Return categorized symbols mirroring the training system's grouping."""
    comps = STATE.ensure_components()
    try:
        from oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager
        iim = getattr(comps.get("risk_manager"), 'iim', None) or InstrumentInfoManager()
        all_symbols: List[str] = iim.all_symbols if hasattr(iim, 'all_symbols') else (iim.get_all_available_symbols() or [])

        # Categories matching training app
        out: Dict[str, List[str]] = {
            'Major Pairs': [],
            'Minor Pairs': [],
            'Precious Metals': [],
            'Indices': [],
            'Energy': [],
            'Commodities': [],
            'Crypto': [],
            'Bonds': [],
        }
        major_pairs = {'EUR_USD','USD_JPY','GBP_USD','AUD_USD','USD_CHF','USD_CAD','NZD_USD'}
        specific_classifications = {
            'NL25_EUR': 'Indices',
            'CH20_CHF': 'Indices',
            'ESPIX_EUR': 'Indices',
            'BCO_USD': 'Energy',
            'XCU_USD': 'Precious Metals',
            'US2000_USD': 'Indices',
            'CHINAH_HKD': 'Indices',
            'SG30_SGD': 'Indices',
            'FR40_EUR': 'Indices',
        }
        index_keywords = ["SPX","NAS","US30","UK100","DE30","JP225","HK33","AU200","FRA40","EU50","CN50","NL25","CH20","ESPIX","US2000","CHINAH","SG30","FR40"]
        energy_keywords = ["OIL","WTICO","BRENT","NATGAS","GAS","BCO"]
        metal_keywords = ["XAU","XAG","GOLD","SILVER","PLAT","PALL","XCU"]
        commodity_keywords = ["CORN","WHEAT","SOYBN","SUGAR","COFFEE","COCOA","COTTON"]
        crypto_keywords = ["BTC","ETH","LTC","BCH","XRP","ADA","DOGE","CRYPTO"]

        for sym in all_symbols:
            try:
                d = iim.get_details(sym)
                if d is None:
                    continue
                symbol = getattr(d, 'symbol', sym)
                t = (getattr(d, 'type', '') or '').upper()
                if symbol in specific_classifications:
                    cat = specific_classifications[symbol]
                elif symbol in major_pairs:
                    cat = 'Major Pairs'
                elif t == 'CURRENCY' and '_' in symbol and not (symbol.startswith('XAU') or symbol.startswith('XAG')):
                    cat = 'Minor Pairs'
                elif any(k in symbol or k in (getattr(d, 'display_name','') or '').upper() for k in metal_keywords):
                    cat = 'Precious Metals'
                elif any(k in symbol for k in index_keywords):
                    cat = 'Indices'
                elif any(k in symbol for k in energy_keywords):
                    cat = 'Energy'
                elif any(k in symbol for k in commodity_keywords):
                    cat = 'Commodities'
                elif t == 'CRYPTO' or any(k in symbol for k in crypto_keywords):
                    cat = 'Crypto'
                else:
                    cat = 'Bonds'
                out.setdefault(cat, []).append(symbol)
            except Exception:
                out.setdefault('Bonds', []).append(sym)

        # Deduplicate and sort
        out = {k: sorted(set(v)) for k, v in out.items() if v}
        return {"catalog": out}
    except Exception:
        cfg = comps["config"] or {}
        syms = cfg.get("trading_instruments", [])
        return {"catalog": {"Major Pairs": syms}}


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
    try:
        suspended = __import__('oanda_trading_bot.live_trading_system.core.tradability_guard', fromlist=['get_all']).get_all()
    except Exception:
        suspended = {}
    return {
        "running": getattr(ss, 'is_running', False),
        "selected_symbols": instruments,
        "model": ss.get_current_model(),
        "risk": risk,
        "warmup": {"ready": warm, "missing": missing},
        "suspended": suspended,
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

    # Validate symbols: allow any account-available instrument (not limited to training set)
    try:
        from oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager
        iim = getattr(risk, 'iim', None) or InstrumentInfoManager()
        available = set(iim.all_symbols if hasattr(iim, 'all_symbols') else (iim.get_all_available_symbols() or []))
        unknown = [s for s in body.symbols if s not in available]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown instruments: {unknown}")
    except HTTPException:
        raise
    except Exception:
        # If we cannot validate against OANDA now, accept and rely on execution-time checks
        pass

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
    pred = comps["prediction_service"]

    # Warmup
    instruments = ss.get_selected_instruments()
    timeout = int(cold_start_timeout_sec or 120)
    tl.warmup_buffers(instruments, max_wait_seconds=timeout, sleep_seconds=1)

    # Ensure a model is loaded
    try:
        model_cfg = (cfg or {}).get('model', {}) if isinstance(cfg, dict) else {}
        if getattr(pred, 'model', None) is None:
            mp = model_cfg.get('path')
            if mp:
                root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
                ap = os.path.join(root, mp)
                if os.path.exists(ap):
                    pred.load_model(ap, device=None)
        if getattr(pred, 'model', None) is None:
            raise HTTPException(status_code=400, detail="Model not loaded. Please select/load a model in Session settings.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to load model from configuration; please select a model.")

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


@app.post("/session/close_position")
def close_position(instrument: str):
    comps = STATE.ensure_components()
    client = comps["client"]
    pm = comps["position_manager"]
    resp_long = client.close_position(instrument, long_units="ALL")
    resp_short = client.close_position(instrument, short_units="ALL")
    # Update DB/PM if any closed
    closed_any = False
    for resp in (resp_long, resp_short):
        if not resp:
            continue
        fill_tx = resp.get('longOrderFillTransaction') or resp.get('shortOrderFillTransaction')
        if fill_tx and fill_tx.get('tradesClosed'):
            closed_any = True
            for closed_trade in fill_tx.get('tradesClosed', []):
                try:
                    trade_id_c = closed_trade.get("tradeID")
                    price_c = float(closed_trade.get("price", 0))
                    pnl_c = float(closed_trade.get("realizedPL", 0))
                    comps["db_manager"].update_trade_on_close(trade_id_c, price_c, pnl_c)
                except Exception:
                    pass
    if closed_any:
        try:
            pm.close_position(instrument)
        except Exception:
            pass
    return {"ok": True, "closed": closed_any}


@app.post("/session/emergency_stop")
def emergency_stop():
    comps = STATE.ensure_components()
    ss = comps["system_state"]
    client = comps["client"]
    pm = comps["position_manager"]
    # Stop engine
    try:
        ss.stop()
    except Exception:
        pass
    # Flatten all open positions
    try:
        pos = client.get_open_positions() or {}
        for p in pos.get("positions", []):
            ins = p.get("instrument")
            try:
                if _to_int_units(p.get("long", {}).get("units", 0)) != 0:
                    client.close_position(ins, long_units="ALL")
                if _to_int_units(p.get("short", {}).get("units", 0)) != 0:
                    client.close_position(ins, short_units="ALL")
                try:
                    pm.close_position(ins)
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        pass
    return {"ok": True}


@app.get("/config/risk")
def get_risk_config():
    comps = STATE.ensure_components()
    rm = comps["risk_manager"]
    return {
        "risk_management": getattr(rm, 'config', {})
    }


@app.post("/test/smoke_trade")
def smoke_trade_per_symbol(units: Optional[int] = 1):
    """
    Test utility: for each selected symbol, place a small BUY then SELL order using override_units.
    Intended for practice environment. No SL/TP attached.
    """
    comps = STATE.ensure_components()
    ss = comps["system_state"]
    tl = comps["trading_logic"]
    om = comps["order_manager"]
    cfg = comps["config"] or {}

    symbols = ss.get_selected_instruments()
    if not symbols:
        raise HTTPException(status_code=400, detail="No symbols selected in session")

    # Ensure buffers are warm for reading ref prices
    try:
        tl.warmup_buffers(symbols, max_wait_seconds=int(cfg.get('smoke_warmup_timeout', 60)))
    except Exception:
        pass

    placed: List[str] = []
    for ins in symbols:
        try:
            # Try to get latest
            latest = comps["client"].get_bid_ask_candles_combined(ins, count=1) or []
            price = 0.0
            if latest:
                last = latest[-1]
                price = float(last.get('ask_close') or last.get('bid_close') or 0.0)
            # BUY
            om.process_signal({"instrument": ins, "signal": 1, "price": price, "override_units": int(units), "no_sl_tp": True})
            time.sleep(1.5)
            # SELL
            latest2 = comps["client"].get_bid_ask_candles_combined(ins, count=1) or []
            price2 = price
            if latest2:
                last2 = latest2[-1]
                price2 = float(last2.get('ask_close') or last2.get('bid_close') or 0.0)
            sell_units = -abs(int(units))
            om.process_signal({"instrument": ins, "signal": -1, "price": price2, "override_units": sell_units, "no_sl_tp": True})
            placed.append(ins)
        except Exception:
            pass

    return {"ok": True, "symbols": placed}


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
