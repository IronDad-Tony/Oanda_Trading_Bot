import os
import sys
import time
import json
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import numpy as np

from oanda_trading_bot.live_trading_system.main import initialize_system


class _DummyAPI:
    def request(self, endpoint):
        # Transactions stream compatibility: yield nothing (short test)
        if hasattr(endpoint, 'name') and 'transactions' in str(endpoint.name).lower():
            if False:
                yield {}
            return []
        return {}


class MockOandaClient:
    def __init__(self):
        self.account_id = 'MOCK-ACCOUNT'
        self.last_transaction_id = '0'
        self.client = _DummyAPI()
        self._base_prices: Dict[str, float] = {}
        self._candles: Dict[str, List[Dict[str, Any]]] = {}
        self._open_positions_units: Dict[str, int] = {}  # net signed units per instrument
        self._open_trades: Dict[str, Dict[str, Any]] = {}  # tradeID -> {instrument, units, price}
        self._trade_seq = 0
        self.position_manager = None  # filled by test

    def _gen_candles(self, instrument: str, count: int, granularity: str) -> List[Dict[str, Any]]:
        base = self._base_prices.get(instrument, 1.1000)
        out = []
        t0 = datetime.utcnow() - timedelta(seconds=count * 5)
        for i in range(count):
            t = t0 + timedelta(seconds=(i + 1) * 5)
            # Simple smooth trend + small oscillation
            drift = 0.00001 * i
            osc = 0.00005 * math.sin(i / 5.0)
            mid = base + drift + osc
            spread = 0.0001
            bid = mid - spread / 2
            ask = mid + spread / 2
            out.append({
                'time': t.isoformat() + 'Z',
                'bid_open': bid,
                'bid_high': bid + 0.00005,
                'bid_low': bid - 0.00005,
                'bid_close': bid,
                'ask_open': ask,
                'ask_high': ask + 0.00005,
                'ask_low': ask - 0.00005,
                'ask_close': ask,
                'volume': 100 + i,
                'complete': True,
            })
        self._base_prices[instrument] = out[-1]['ask_close']
        self._candles[instrument] = out
        return out

    def get_bid_ask_candles_combined(self, instrument: str, count: int = 100, granularity: str = "S5") -> Optional[List[Dict[str, Any]]]:
        return self._gen_candles(instrument, count, granularity)

    def get_candles(self, instrument: str, count: int = 100, granularity: str = "S5", price: str = "M"):
        # UI path compatibility
        recs = self._gen_candles(instrument, count, granularity)
        # Return mid-only like OANDA InstrumentsCandles
        out = []
        for r in recs:
            out.append({
                'time': r['time'],
                'mid': {
                    'o': (r['bid_open'] + r['ask_open']) / 2,
                    'h': (r['bid_high'] + r['ask_high']) / 2,
                    'l': (r['bid_low'] + r['ask_low']) / 2,
                    'c': (r['bid_close'] + r['ask_close']) / 2,
                },
                'volume': r['volume'],
                'complete': True,
            })
        return out

    def _new_trade_id(self) -> str:
        self._trade_seq += 1
        return str(self._trade_seq)

    def create_order_v2(self, instrument: str, units: int, stop_loss_on_fill=None, take_profit_on_fill=None, price_bound: Optional[float] = None, client_extensions: Optional[Dict[str, Any]] = None, time_in_force: str = "FOK"):
        # Fill immediately at bid/ask with tiny adverse slip within bound
        candles = self._candles.get(instrument) or self._gen_candles(instrument, 2, 'S5')
        last = candles[-1]
        bid = float(last['bid_close'])
        ask = float(last['ask_close'])
        is_buy = units > 0
        ref = ask if is_buy else bid
        slip = 0.0
        if price_bound is not None:
            # stay well within bound
            slip = min(0.2 * abs(price_bound - ref), 1e-6)
        price = ref + slip if is_buy else ref - slip
        trade_id = self._new_trade_id()
        # Update open positions and trades
        self._open_positions_units[instrument] = self._open_positions_units.get(instrument, 0) + int(units)
        self._open_trades[trade_id] = {"instrument": instrument, "units": int(units), "price": price}
        # Return OANDA-like fill payload
        return {
            "orderFillTransaction": {
                "type": "ORDER_FILL",
                "instrument": instrument,
                "units": str(units),
                "price": str(price),
                "time": datetime.utcnow().isoformat() + 'Z',
                "tradeOpened": {"tradeID": trade_id}
            },
            "lastTransactionID": trade_id
        }

    def get_open_positions(self):
        positions = []
        for inst, net in self._open_positions_units.items():
            if net == 0:
                continue
            positions.append({
                "instrument": inst,
                "long": {"units": str(net) if net > 0 else "0"},
                "short": {"units": str(-net) if net < 0 else "0"},
            })
        return {"positions": positions}

    def close_position(self, instrument: str, long_units: Optional[str] = None, short_units: Optional[str] = None):
        # Close all trades on instrument
        now = datetime.utcnow().isoformat() + 'Z'
        candles = self._candles.get(instrument) or self._gen_candles(instrument, 1, 'S5')
        last = candles[-1]
        bid = float(last['bid_close'])
        ask = float(last['ask_close'])
        trades_closed = []
        for tid, tr in list(self._open_trades.items()):
            if tr['instrument'] != instrument:
                continue
            units = tr['units']
            close_price = bid if units > 0 else ask
            realized = (close_price - tr['price']) * units
            trades_closed.append({"tradeID": tid, "price": str(close_price), "realizedPL": str(realized)})
            # update net units
            self._open_positions_units[instrument] = self._open_positions_units.get(instrument, 0) - units
            del self._open_trades[tid]
        key = 'longOrderFillTransaction' if (long_units or (self._open_positions_units.get(instrument, 0) <= 0)) else 'shortOrderFillTransaction'
        return {key: {"time": now, "tradesClosed": trades_closed}}

    # Compatibility stubs
    def get_account_summary(self):
        return {"account": {"balance": "100000", "equity": "100000", "lastTransactionID": self.last_transaction_id}}

    def get_account_changes(self):
        return {"lastTransactionID": self.last_transaction_id, "changes": {}}


class MockPredictionService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def load_model(self, path: str, device: Optional[str] = None):
        # No-op for mock
        return

    def predict(self, processed_data_map: Dict[str, np.ndarray]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for inst, arr in processed_data_map.items():
            try:
                # Simple momentum on ask_close
                series = arr[:, -1].astype(float)
                if len(series) >= 4:
                    slope = (series[-1] - series[-4])
                else:
                    slope = series[-1] - series[0]
                if slope > 0:
                    out[inst] = 1
                elif slope < 0:
                    out[inst] = -1
                else:
                    out[inst] = 0
            except Exception:
                out[inst] = 0
        return out


def run():
    logging.basicConfig(level=logging.INFO)
    mock_client = MockOandaClient()
    mock_pred = MockPredictionService({})

    components = initialize_system(mock_client=mock_client, mock_prediction_service=mock_pred)
    assert components is not None
    system_state = components['system_state']
    trading_logic = components['trading_logic']
    order_manager = components['order_manager']
    db_manager = components['db_manager']

    # Link position manager to mock for closing
    mock_client.position_manager = components['position_manager']

    # Select instruments and warmup
    system_state.set_selected_instruments(["EUR_USD"])
    try:
        trading_logic.warmup_buffers(["EUR_USD"], max_wait_seconds=5, sleep_seconds=1)
    except Exception:
        pass

    # Execute a few cycles
    cycles = 5
    for _ in range(cycles):
        trading_logic.execute_trade_cycle()
        time.sleep(0.1)

    # Close all positions
    order_manager.close_all_positions()

    # Build summary
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

    # Persist summary
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    reports_dir = os.path.join(proj_root, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, 'e2e_mock_live_test_summary.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run()

