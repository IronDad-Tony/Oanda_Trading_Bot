import os
import sys
import time

SRC_PATH = os.path.join(os.path.dirname(__file__), '..', 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from oanda_trading_bot.live_trading_system.main import initialize_system


def main():
    confirm = os.getenv('CONFIRM_LIVE_TRADING', 'NO').upper()
    if confirm != 'YES':
        print("Refusing to run live trades without CONFIRM_LIVE_TRADING=YES")
        return

    comps = initialize_system()
    if not comps:
        print("Failed to initialize live trading system.")
        return

    cfg = comps['config']
    client = comps['client']
    order_manager = comps['order_manager']
    ss = comps['system_state']

    instruments = cfg.get('trading_instruments', [])
    if not instruments:
        print("No instruments configured.")
        return

    ss.set_selected_instruments(instruments)

    # Place 10 minimal orders (BUY then SELL across instruments)
    placed = 0
    side = 1
    for inst in instruments:
        if placed >= 10:
            break
        # Fetch price to tag signals
        candles = client.get_bid_ask_candles_combined(inst, count=1, granularity=cfg.get('trading_granularity', 'S5')) or []
        px = None
        if candles:
            last = candles[-1]
            px = float(last.get('ask_close') or last.get('bid_close') or 0)

        for _ in range(2):
            if placed >= 10:
                break
            units = side * 1  # 1 unit per trade
            signal_info = {
                'instrument': inst,
                'signal': 'BUY' if side > 0 else 'SELL',
                'price': px,
                'override_units': units,
                'no_sl_tp': True,
            }
            try:
                order_manager.process_signal(signal_info)
                placed += 1
            except Exception as e:
                print(f"Failed to place order for {inst}: {e}")
            side *= -1
            time.sleep(0.5)

    print(f"Placed {placed} test orders. Waiting 5 seconds before closing positions...")
    time.sleep(5)

    try:
        order_manager.close_all_positions()
        print("Requested to close all positions.")
    except Exception as e:
        print(f"Failed to close positions: {e}")


if __name__ == '__main__':
    main()
