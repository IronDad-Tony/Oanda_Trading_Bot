import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_sys_path():
    root = project_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def load_env():
    from dotenv import load_dotenv
    dotenv_path = project_root() / ".env"
    load_dotenv(dotenv_path=str(dotenv_path))


def get_valid_symbols_from_iim(components: Dict[str, Any]) -> List[str]:
    iim = components["system_state"].get_instrument_manager()
    try:
        syms = getattr(iim, "all_symbols", None)
        if syms:
            return list(syms)
        return list(iim.get_all_available_symbols())
    except Exception:
        return []


def get_symbols_union_from_dataset() -> List[str]:
    root = project_root()
    mmap_dir = root / "data" / "mmap_s5_universal"
    out = set()
    try:
        if mmap_dir.is_dir():
            for sub in mmap_dir.iterdir():
                meta = sub / "dataset_metadata.json"
                if meta.is_file():
                    try:
                        with open(meta, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            for s in data.get("symbols", []) or []:
                                if isinstance(s, str):
                                    out.add(s)
                    except Exception:
                        pass
    except Exception:
        pass
    return sorted(out)


def select_five_fx_symbols(valid: List[str]) -> List[str]:
    # Prefer common FX majors available in most OANDA accounts
    preferred = ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "AUD_JPY", "USD_CAD", "USD_CHF", "NZD_USD"]
    sel = [s for s in preferred if s in valid]
    if len(sel) >= 5:
        return sel[:5]
    # Fall back to any FX pairs among valid list (exclude indices/metals/CFDs heuristically)
    fx = [s for s in valid if s.count('_') == 1 and s.endswith(("USD", "JPY", "EUR", "GBP", "AUD", "CAD", "CHF", "NZD"))]
    for s in fx:
        if s not in sel:
            sel.append(s)
            if len(sel) == 5:
                break
    return sel[:5]


def find_model_path() -> Optional[str]:
    root = project_root()
    cands = [root / "weights" / "sac_model_symbols5.zip",
             root / "weights" / "sac_model_symbols10.zip"]
    for p in cands:
        if p.exists():
            return str(p)
    # Fallback: first .zip in weights
    wdir = root / "weights"
    if wdir.is_dir():
        for p in wdir.iterdir():
            if p.suffix.lower() == ".zip":
                return str(p)
    return None


def latest_price_for_side(client, instrument: str, side: str, tl=None) -> Optional[float]:
    # BUY -> use ask, SELL -> use bid
    recs = client.get_bid_ask_candles_combined(instrument, count=1) or []
    if not recs:
        # fallback to buffer if available
        try:
            if tl is not None and instrument in getattr(tl, 'data_buffers', {}):
                last = tl.data_buffers[instrument][-1]
                if side.upper() == "BUY":
                    return float(last.get("ask_close") or last.get("ask") or 0) or None
                else:
                    return float(last.get("bid_close") or last.get("bid") or 0) or None
        except Exception:
            pass
        return None
    r = recs[-1]
    if side.upper() == "BUY":
        return float(r.get("ask_close") or r.get("ask") or r.get("ask_price") or 0) or None
    else:
        return float(r.get("bid_close") or r.get("bid") or r.get("bid_price") or 0) or None


def wait_for_trade_log(db_manager, instrument: str, timeout_s: int = 30) -> Optional[Dict[str, Any]]:
    deadline = time.time() + timeout_s
    last_seen = None
    while time.time() < deadline:
        try:
            row = db_manager.get_last_trade_for_instrument(instrument)
            if row and row != last_seen:
                return row
            last_seen = row
        except Exception:
            pass
        time.sleep(0.5)
    return None


def place_and_close(order_manager, db_manager, client, instrument: str, side: str, units: int = 1, tl=None) -> bool:
    # Place
    price = latest_price_for_side(client, instrument, side, tl=tl)
    if not price or price <= 0:
        print(f"[WARN] No price for {instrument} {side}")
        return False
    signal = 1 if side.upper() == "BUY" else -1
    order_manager.process_signal({
        "instrument": instrument,
        "signal": signal,
        "price": price,
        "override_units": units,
        "no_sl_tp": True,
    })
    row = wait_for_trade_log(db_manager, instrument, timeout_s=30)
    if not row:
        print(f"[WARN] No DB log observed for {instrument} {side}")
        return False
    # Close the just-opened position on that instrument (best-effort)
    try:
        open_pos = client.get_open_positions() or {}
        for pos in open_pos.get("positions", []) or []:
            if pos.get("instrument") == instrument:
                if side.upper() == "BUY" and pos.get("long", {}).get("units", "0") != "0":
                    client.close_position(instrument, long_units="ALL")
                elif side.upper() == "SELL" and pos.get("short", {}).get("units", "0") != "0":
                    client.close_position(instrument, short_units="ALL")
                break
    except Exception:
        pass
    return True


def main():
    ensure_sys_path()
    load_env()

    from oanda_trading_bot.live_trading_system.main import initialize_system

    comps = initialize_system()
    if not comps:
        print("[FATAL] initialize_system failed. Check .env and configs.")
        sys.exit(1)

    client = comps["client"]
    system_state = comps["system_state"]
    db_manager = comps["db_manager"]
    pred = comps["prediction_service"]
    tl = comps["trading_logic"]
    om = comps["order_manager"]

    # Relax exec constraints for testing fills
    try:
        om.use_price_bound = False
        om.min_interval_between_orders_ms = 0
    except Exception:
        pass

    # Choose five FX symbols
    valid = get_valid_symbols_from_iim(comps)
    if not valid:
        dataset_union = get_symbols_union_from_dataset()
        valid = dataset_union
    symbols = select_five_fx_symbols(valid)
    if len(symbols) < 5:
        print(f"[FATAL] Could not find 5 valid FX symbols. Got: {symbols}")
        sys.exit(1)
    print("Selected symbols:", symbols)

    # Load a model capable of 5+ symbols
    model_path = find_model_path()
    if model_path:
        try:
            pred.load_model(model_path, device=None)
            system_state.set_current_model(model_path)
        except Exception as e:
            print(f"[WARN] Failed to load model {model_path}: {e}")

    # Apply selection and warmup
    system_state.set_selected_instruments(symbols)
    lookback = int((comps.get("config") or {}).get("model_lookback_window", 128))
    from collections import deque
    for s in symbols:
        if s not in tl.data_buffers:
            tl.data_buffers[s] = deque(maxlen=lookback)
    print("Warming up buffers...")
    tl.warmup_buffers(symbols, max_wait_seconds=240, sleep_seconds=1)
    print("Warmup complete.")

    # Small unit size for safety (practice or live)
    test_units = 1

    # For each symbol: BUY then SELL with closure
    results: Dict[str, Dict[str, bool]] = {}
    for s in symbols:
        results[s] = {"buy": False, "sell": False}
        print(f"-- Testing {s} BUY...")
        results[s]["buy"] = place_and_close(om, db_manager, client, s, "BUY", units=test_units, tl=tl)
        time.sleep(0.5)
        print(f"-- Testing {s} SELL...")
        results[s]["sell"] = place_and_close(om, db_manager, client, s, "SELL", units=test_units, tl=tl)
        time.sleep(0.5)

    # Final flatten to be safe
    print("Flattening any remaining open positions...")
    try:
        om.close_all_positions()
    except Exception:
        pass

    # Summary
    print("Test summary:")
    for s, r in results.items():
        print(f"  {s}: BUY={'OK' if r['buy'] else 'FAIL'}, SELL={'OK' if r['sell'] else 'FAIL'}")

    # Account and positions snapshot
    try:
        acct = client.get_account_summary() or {}
        print("Account summary:", acct)
        pos = client.get_open_positions() or {}
        print("Open positions:", pos)
        hist = db_manager.get_trade_history(limit=50)
        print("Recent trade history (up to 50):", hist)
    except Exception as e:
        print(f"[WARN] Snapshot error: {e}")


if __name__ == "__main__":
    main()
