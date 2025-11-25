import os
import sys
import time
import json
import signal
import argparse
from collections import defaultdict
from typing import List, Dict, Any, Set

from dotenv import load_dotenv

# Wire project src into path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(REPO_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from oanda_trading_bot.live_trading_system.main import initialize_system, trading_loop, load_config
from oanda_trading_bot.live_trading_system.core.oanda_client import OandaClient


def pick_symbols(components: Dict[str, Any], desired: List[str], max_count: int) -> List[str]:
    ss = components["system_state"]
    try:
        iim = ss.get_instrument_manager()
        available = set(getattr(iim, 'all_symbols', None) or (iim.get_all_available_symbols() or []))
    except Exception:
        available = set()

    chosen: List[str] = []
    for sym in desired:
        if available and sym not in available:
            continue
        chosen.append(sym)
        if len(chosen) >= max_count:
            break
    # Fallback to config if we couldn't fill
    if len(chosen) == 0:
        cfg_syms = (components.get('config') or {}).get('trading_instruments', [])
        chosen = cfg_syms[:max_count]
    return chosen


def summarize_trades(trades: List[Dict[str, Any]]):
    # trades contain: oanda_trade_id, instrument, units, entry_price, signal, status, pnl, open_timestamp, close_timestamp, close_price
    by_id: Dict[str, Dict[str, Any]] = {}
    for t in trades:
        tid = t.get('oanda_trade_id') or t.get('id') or str(t.get('trade_id'))
        if not tid:
            continue
        by_id[tid] = t

    total = len(by_id)
    closed = [t for t in by_id.values() if (t.get('status') == 'CLOSED' and t.get('pnl') is not None)]
    open_trades = [t for t in by_id.values() if t.get('status') != 'CLOSED']
    realized = sum(float(t.get('pnl') or 0.0) for t in closed)
    by_sym = defaultdict(float)
    for t in closed:
        by_sym[str(t.get('instrument'))] += float(t.get('pnl') or 0.0)
    return {
        'total_trades': total,
        'closed_trades': len(closed),
        'open_trades': len(open_trades),
        'realized_pnl': realized,
        'pnl_by_symbol': dict(by_sym),
        'closed_details': closed,
        'open_details': open_trades,
    }


def main():
    parser = argparse.ArgumentParser(description="Headless run: warmup, trade via OANDA API, collect 10 fills, summarize.")
    parser.add_argument('--env', choices=['practice', 'live'], default='practice', help='OANDA environment to use (default: practice)')
    parser.add_argument('--max-trades', type=int, default=10, help='Stop after this many fills (default: 10)')
    parser.add_argument('--timeout-min', type=int, default=30, help='Max minutes to wait for fills (default: 30)')
    parser.add_argument('--symbols', type=str, nargs='*', default=None, help='Optional explicit symbols list')
    parser.add_argument('--model', type=str, default=None, help='Optional model path (relative to repo root)')
    parser.add_argument('--api-key', type=str, default=None, help='OANDA API key (overrides env)')
    parser.add_argument('--account-id', type=str, default=None, help='OANDA Account ID (overrides env)')
    args = parser.parse_args()

    load_dotenv(os.path.join(REPO_ROOT, 'src', 'oanda_trading_bot', 'live_trading_system', '.env'))
    # Allow command-line override without editing .env
    if args.api_key:
        os.environ['OANDA_API_KEY'] = args.api_key
    if args.account_id:
        os.environ['OANDA_ACCOUNT_ID'] = args.account_id
    api_key = os.getenv('OANDA_API_KEY')
    account_id = os.getenv('OANDA_ACCOUNT_ID')
    if not api_key or not account_id:
        print('ERROR: Missing OANDA_API_KEY or OANDA_ACCOUNT_ID in .env')
        sys.exit(1)

    # Force desired environment explicitly via mock client
    # Disable SL/TP attachments for higher fill probability in headless test
    os.environ['ATTACH_SL_TP_ON_FILL'] = '0'
    client = OandaClient(api_key=api_key, account_id=account_id, environment=args.env)
    components = initialize_system(mock_client=client)
    if not components:
        print('ERROR: initialize_system failed')
        sys.exit(2)

    # Select symbols (try to include different quote currencies to exercise conversion)
    cfg = components['config'] or {}
    model_cfg = cfg.get('model', {}) if isinstance(cfg, dict) else {}
    # Prefer user-specified model, else try weights/sac_model_symbols10.zip if present for broader coverage
    selected_model_rel = None
    if args.model:
        selected_model_rel = args.model
    else:
        candidate10 = os.path.join(REPO_ROOT, 'weights', 'sac_model_symbols10.zip')
        if os.path.exists(candidate10):
            selected_model_rel = os.path.relpath(candidate10, REPO_ROOT)
        else:
            selected_model_rel = model_cfg.get('path')

    # Determine max symbols from model name
    max_symbols = 5
    try:
        name = os.path.basename(selected_model_rel or '')
        import re
        m = re.search(r"symbols(\d+)", name, re.IGNORECASE)
        if m:
            max_symbols = int(m.group(1))
    except Exception:
        pass

    if args.symbols:
        desired = args.symbols
    else:
        # Build a diverse desired list; will be filtered by availability
        desired = [
            # Majors first to ensure always-on markets
            'EUR_USD','GBP_USD','USD_JPY','AUD_USD',
            # Metals
            'XAU_USD',
            # Indices
            'NAS100_USD','SPX500_USD','US30_USD','US2000_USD','DE30_EUR','EU50_EUR','UK100_GBP','JP225_USD','HK33_HKD','AU200_AUD',
        ]
    symbols = pick_symbols(components, desired, max_symbols)
    if not symbols:
        print('ERROR: No symbols selected')
        sys.exit(3)

    # Apply selections
    ss = components['system_state']
    ss.set_selected_instruments(symbols)
    # Ensure model is loaded (initialize_system attempted auto-load)
    pred = components['prediction_service']
    if getattr(pred, 'model', None) is None:
        # Try to load selected model path
        model_rel = selected_model_rel
        if model_rel:
            model_abs = os.path.join(REPO_ROOT, model_rel)
            if os.path.exists(model_abs):
                pred.load_model(model_abs, device=None)

    if getattr(pred, 'model', None) is None:
        print('ERROR: Model not loaded; please ensure weights exist.')
        sys.exit(4)

    # Warmup
    tl = components['trading_logic']
    print(f"Warmup starting for instruments: {symbols}")
    tl.warmup_buffers(symbols, max_wait_seconds=240, sleep_seconds=1)
    print("Warmup done or timeout reached.")

    # Exercise currency conversion path explicitly once
    try:
        rm = components['risk_manager']
        iim = rm.iim
        test_sym = None
        for cand in symbols:
            try:
                det = iim.get_details(cand)
                if det and getattr(det, 'quote_currency', None) and getattr(det, 'quote_currency').upper() != (rm.account_currency or '').upper():
                    test_sym = cand
                    break
            except Exception:
                continue
        if test_sym:
            price_map = rm._build_price_map(test_sym, components['client'])  # type: ignore[attr-defined]
            rate = float(rm.cur_mgr.convert_to_account_currency(getattr(iim.get_details(test_sym), 'quote_currency'), price_map))
            print(f"Currency conversion exercised on {test_sym}: quote->account rate ≈ {rate}")
        else:
            print("Currency conversion test: No suitable symbol found in selection; skipped.")
    except Exception as e:
        print(f"Currency conversion test error: {e}")

    # Start trading loop
    ss.start()
    stop_flag = False

    def _graceful_stop(signum=None, frame=None):
        nonlocal stop_flag
        stop_flag = True

    signal.signal(signal.SIGINT, _graceful_stop)
    signal.signal(signal.SIGTERM, _graceful_stop)

    th = None
    try:
        import threading
        th = threading.Thread(target=trading_loop, args=(components,), daemon=True)
        th.start()

        print("Trading loop started. Waiting for fills...")
        from datetime import datetime
        run_start_iso = datetime.utcnow().isoformat()
        deadline = time.time() + (args.timeout_min * 60)
        seen_ids: Set[str] = set()
        while time.time() < deadline and not stop_flag:
            trades = components['db_manager'].get_trade_history(limit=500) or []
            # consider only trades opened after this run started
            recent = [t for t in trades if (t.get('open_timestamp') or '') >= run_start_iso]
            for t in trades:
                tid = t.get('oanda_trade_id') or t.get('id') or str(t.get('trade_id'))
                if tid:
                    seen_ids.add(str(tid))
            # count only recent fills for threshold
            seen_recent_ids = set()
            for t in recent:
                tid = t.get('oanda_trade_id') or t.get('id') or str(t.get('trade_id'))
                if tid:
                    seen_recent_ids.add(str(tid))
            if len(seen_recent_ids) >= args.max_trades:
                break
            time.sleep(5)

    finally:
        try:
            ss.stop()
            if th and th.is_alive():
                th.join(timeout=10)
        except Exception:
            pass

    # Final summary
    trades_final = components['db_manager'].get_trade_history(limit=1000) or []
    # Final summary includes only trades from this run window
    from datetime import datetime
    run_start_cutoff = None
    try:
        # A very small chance run_start_iso is undefined if exception before it's set; guard below
        _ = run_start_iso  # type: ignore[name-defined]
        run_start_cutoff = run_start_iso
    except Exception:
        run_start_cutoff = None
    if run_start_cutoff:
        trades_final = [t for t in trades_final if (t.get('open_timestamp') or '') >= run_start_cutoff]
    summary = summarize_trades(trades_final)

    # Try to grab unrealized P/L snapshot
    unrealized = 0.0
    try:
        pos = components['client'].get_open_positions() or {}
        for p in pos.get('positions', []):
            try:
                up1 = float((p.get('long', {}) or {}).get('unrealizedPL') or 0.0)
                up2 = float((p.get('short', {}) or {}).get('unrealizedPL') or 0.0)
                unrealized += (up1 + up2)
            except Exception:
                continue
    except Exception:
        pass
    summary['unrealized_pl_snapshot'] = unrealized

    print("\n==== Headless Trading Summary ====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=================================")


if __name__ == '__main__':
    main()
