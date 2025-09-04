import os
import time
import json
from datetime import datetime

from oanda_trading_bot.live_trading_system.main import initialize_system


def run():
    components = initialize_system()
    if not components:
        raise SystemExit("Failed to initialize live system (check .env and config).")

    system_state = components['system_state']
    trading_logic = components['trading_logic']
    order_manager = components['order_manager']
    client = components['client']
    db_manager = components['db_manager']

    # Focus on a single liquid instrument for test
    system_state.set_selected_instruments(["EUR_USD"])

    # Warm-up data buffers
    try:
        trading_logic.warmup_buffers(["EUR_USD"], max_wait_seconds=60, sleep_seconds=2)
    except Exception:
        pass

    # Run several cycles to allow the model to act
    opened = False
    for _ in range(6):
        trading_logic.execute_trade_cycle()
        time.sleep(5)
        # Check DB for any open trade
        hist = db_manager.get_trade_history(limit=10) or []
        if any(t.get('status') == 'OPEN' for t in hist):
            opened = True
            break

    # Fallback: force a single BUY signal if model hasn’t opened anything
    if not opened:
        candles = client.get_bid_ask_candles_combined("EUR_USD", count=1) or []
        if candles:
            last = candles[-1]
            px = float(last.get('ask_close') or last.get('bid_close') or 0.0)
            # Request a small test order without SL/TP to minimize rejection chances
            order_manager.process_signal({
                "instrument": "EUR_USD",
                "signal": 1,
                "price": px,
                "timestamp": last.get('time'),
                "override_units": 10,
                "no_sl_tp": True,
            })
            time.sleep(5)

    # If still no open positions, directly place a minimal market order (no SL/TP)
    if not any((db_manager.get_trade_history(limit=5) or [])):
        candles = client.get_bid_ask_candles_combined("EUR_USD", count=1) or []
        if candles:
            last = candles[-1]
            ask = float(last.get('ask_close') or 0.0)
            pip = 0.0001
            resp = client.create_order_v2(
                instrument="EUR_USD",
                units=10,
                stop_loss_on_fill=None,
                take_profit_on_fill=None,
                price_bound=ask + 2 * pip,
                client_extensions={"id": f"practice-{int(time.time()*1000)}", "tag": "practice_test"},
                time_in_force="FOK"
            )
            if resp and 'orderFillTransaction' in resp:
                fill = resp['orderFillTransaction']
                trade_id = fill.get('tradeOpened', {}).get('tradeID')
                price = float(fill.get('price', 0.0))
                units = int(fill.get('units', 0))
                if trade_id and units:
                    # Reflect in local state + DB for summary
                    components['position_manager'].update_position("EUR_USD", units, price, trade_id)
                    db_manager.log_trade(trade_id, "EUR_USD", units, price, signal=1, status="OPEN")
                    time.sleep(3)

    # Close everything to end test
    order_manager.close_all_positions()

    # Summarize
    trades = db_manager.get_trade_history(limit=100)
    total_pnl = sum([t.get('pnl') or 0.0 for t in trades if t.get('pnl') is not None])
    open_count = len([t for t in trades if t.get('status') == 'OPEN'])
    closed_count = len([t for t in trades if t.get('status') == 'CLOSED'])
    summary = {
        'trades_logged': len(trades),
        'open_trades': open_count,
        'closed_trades': closed_count,
        'total_pnl': total_pnl,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }

    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    out_path = os.path.join(proj_root, 'reports', 'practice_run_summary.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run()
