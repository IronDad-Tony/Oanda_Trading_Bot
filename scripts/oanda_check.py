import os
import sys
from typing import Tuple

from dotenv import load_dotenv
import oandapyV20
import oandapyV20.endpoints.accounts as accounts
from oandapyV20.exceptions import V20Error


def _normalize_env(value: str) -> Tuple[str, bool]:
    v = (value or "practice").strip().lower()
    if v in {"practice", "demo", "paper", "sandbox", "test"}:
        return "practice", v != "practice"
    if v in {"live", "prod", "production"}:
        return "live", v != "live"
    return "practice", True


def main() -> int:
    load_dotenv()

    api_key = os.getenv("OANDA_API_KEY")
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    env_raw = os.getenv("OANDA_ENVIRONMENT", "practice")
    env_norm, corrected = _normalize_env(env_raw)

    if corrected:
        print(f"[hint] OANDA_ENVIRONMENT='{env_raw}' treated as '{env_norm}'. Set exactly to 'practice' or 'live'.")

    missing = []
    if not api_key:
        missing.append("OANDA_API_KEY")
    if not account_id:
        missing.append("OANDA_ACCOUNT_ID")
    if missing:
        print("[error] Missing required env keys:", ", ".join(missing))
        print("        Add them to your .env (see .env.example).")
        return 2

    try:
        client = oandapyV20.API(access_token=api_key, environment=env_norm)
        ep = accounts.AccountSummary(account_id)
        data = client.request(ep)
    except V20Error as e:
        print(f"[error] OANDA API error: {e}")
        print("        Check token, account ID, and environment (practice vs live).")
        return 3
    except Exception as e:
        print(f"[error] Unexpected error: {e}")
        return 4

    acct = data.get("account", {}) if isinstance(data, dict) else {}
    print("[ok] Connected to OANDA v20")
    print(f"     Environment : {env_norm}")
    print(f"     Account ID  : {account_id}")
    if acct:
        alias = acct.get("alias") or "(no alias)"
        currency = acct.get("currency") or "?"
        balance = acct.get("balance") or "?"
        nav = acct.get("NAV") or "?"
        print(f"     Alias       : {alias}")
        print(f"     Currency    : {currency}")
        print(f"     Balance     : {balance}")
        print(f"     NAV         : {nav}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

