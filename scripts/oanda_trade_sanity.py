import os
import sys
import time
from typing import Optional

from dotenv import load_dotenv
import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
from oandapyV20.exceptions import V20Error


def normalize_env(value: str) -> str:
    v = (value or "practice").strip().lower()
    if v in {"practice", "demo", "paper", "sandbox", "test"}:
        return "practice"
    if v in {"live", "prod", "production"}:
        return "live"
    return "practice"


def get_precision(api, account_id: str, instrument: str) -> int:
    try:
        ep = accounts.AccountInstruments(account_id, params={"instruments": instrument})
        data = api.request(ep)
        inst = (data.get("instruments") or [])[0]
        return int(inst.get("displayPrecision", 5))
    except Exception:
        return 5


def get_last_close(api, instrument: str) -> Optional[float]:
    try:
        ep = instruments.InstrumentsCandles(
            instrument=instrument,
            params={"granularity": "S5", "count": 2, "price": "M"},
        )
        data = api.request(ep)
        c = (data.get("candles") or [])[-1]
        mid = c.get("mid") or {}
        return float(mid.get("c"))
    except Exception:
        return None


def main() -> int:
    load_dotenv()
    api_key = os.getenv("OANDA_API_KEY")
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    env_raw = os.getenv("OANDA_ENVIRONMENT", "practice")
    environment = normalize_env(env_raw)

    if not api_key or not account_id:
        print("[error] Missing OANDA_API_KEY or OANDA_ACCOUNT_ID in .env")
        return 2

    if environment != "practice":
        print("[safe-guard] This trading sanity test only runs in practice.")
        print("             Current environment:", environment)
        return 5

    instrument = os.getenv("OANDA_TEST_INSTRUMENT", "EUR_USD")

    api = oandapyV20.API(access_token=api_key, environment=environment)

    # 1) Confirm account reachable
    try:
        _ = api.request(accounts.AccountSummary(account_id))
    except Exception as e:
        print(f"[error] Unable to reach account: {e}")
        return 3

    # 2) Fetch last close to derive a safe, non-executable LIMIT price
    last_close = get_last_close(api, instrument)
    if not last_close:
        print("[error] Could not fetch recent price for", instrument)
        return 4
    precision = get_precision(api, account_id, instrument)

    # Choose a BUY LIMIT 5% below last_close to avoid immediate execution
    buy_price = round(last_close * 0.95, precision)
    client_id = f"codex_test_{instrument}_{int(time.time())}"

    payload = {
        "order": {
            "type": "LIMIT",
            "instrument": instrument,
            "units": "1",
            "price": f"{buy_price:.{precision}f}",
            "timeInForce": "GTC",
            "positionFill": "DEFAULT",
            "clientExtensions": {"id": client_id},
        }
    }

    print(f"[info] Placing test LIMIT BUY order for {instrument} at {payload['order']['price']} (last_close={last_close})")

    try:
        create = orders.OrderCreate(accountID=account_id, data=payload)
        resp = api.request(create)
    except V20Error as e:
        print(f"[error] Order rejected by OANDA: {e}")
        return 6
    except Exception as e:
        print(f"[error] Unexpected while creating order: {e}")
        return 7

    # Extract order ID for cancellation
    order_id = None
    tx = resp.get("orderCreateTransaction") if isinstance(resp, dict) else None
    if tx:
        order_id = tx.get("orderID") or tx.get("id")

    if not order_id:
        print("[warn] Could not determine orderID from response; raw:")
        print(resp)
        return 8

    # 3) Immediately cancel the pending order
    try:
        cancel = orders.OrderCancel(accountID=account_id, orderID=order_id)
        cancel_resp = api.request(cancel)
    except V20Error as e:
        print(f"[error] Failed to cancel order {order_id}: {e}")
        return 9
    except Exception as e:
        print(f"[error] Unexpected while cancelling order {order_id}: {e}")
        return 10

    print("[ok] Trading API sanity test passed.")
    print("     - Order placed and cancelled successfully")
    print(f"     - Instrument: {instrument}")
    print(f"     - Order ID  : {order_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

