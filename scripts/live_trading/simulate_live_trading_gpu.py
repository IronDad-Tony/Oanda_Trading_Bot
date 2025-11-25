#!/usr/bin/env python3
"""
Simulate Live Trading on GPU with mocked market data.

Loads the trained SAC model (weights/sac_model_symbols5.zip), initializes the
Live Trading system with a mock Oanda client that feeds synthetic S5 candles for
5 symbols, and runs a few logic cycles to verify the model outputs actions on GPU.
"""
import os
import sys
import time
from datetime import datetime, timedelta, timezone
import random


def iso_ts(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


class MockOandaClient:
    def __init__(self, symbols, granularity="S5"):
        self.symbols = symbols
        self.granularity = granularity
        self._last_prices = {s: 1.0 + i * 0.1 for i, s in enumerate(symbols)}

    def _gen_candle(self, base_price):
        # small random walk
        delta = random.uniform(-0.001, 0.001)
        o = base_price
        c = max(0.0001, base_price + delta)
        h = max(o, c) + abs(delta) * 0.5
        l = min(o, c) - abs(delta) * 0.5
        v = random.randint(80, 200)
        return {
            'mid': {'o': str(o), 'h': str(h), 'l': str(l), 'c': str(c)},
            'time': None, # will be filled by caller if needed
            # Combined bid/ask structure used by training logic for price display
            'bid_open': o, 'bid_high': h, 'bid_low': l, 'bid_close': c,
            'ask_open': o, 'ask_high': h, 'ask_low': l, 'ask_close': c,
            'volume': v,
        }, c

    def get_bid_ask_candles_combined(self, instrument, lookback, granularity):
        # Generate lookback number of S5 candles ending now
        now = datetime.now(timezone.utc).replace(microsecond=0)
        step = timedelta(seconds=5)
        base = self._last_prices.get(instrument, 1.0)
        out = []
        t = now - step * lookback
        price = base
        for _ in range(lookback):
            cndl, price = self._gen_candle(price)
            cndl['time'] = iso_ts(t)
            out.append(cndl)
            t += step
        # Update last
        self._last_prices[instrument] = price
        return out

    # Optional: used by UI for charts; not required in this headless simulation
    def get_candles(self, instrument, count=100, granularity="S5"):
        return []


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    src_dir = os.path.join(project_root, 'src')
    for p in [project_root, src_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from oanda_trading_bot.live_trading_system.main import initialize_system
    from oanda_trading_bot.live_trading_system.trading.trading_logic import TradingLogic

    # Symbols consistent with training
    symbols = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "XAU_USD"]
    mock_client = MockOandaClient(symbols)

    # Initialize system (will auto-load model onto GPU if present)
    components = initialize_system(mock_client=mock_client)
    if not components:
        raise RuntimeError("Failed to initialize live trading system components.")

    # Limit instruments to our symbols
    components['system_state'].set_selected_instruments(symbols)

    logic: TradingLogic = components['trading_logic']

    # Run a few cycles
    cycles = 5
    for i in range(cycles):
        logic.execute_trade_cycle()
        time.sleep(0.2)

    print("Simulation completed.")


if __name__ == "__main__":
    main()

