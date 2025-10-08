"""
Shared utilities for querying and caching account-level metadata from the
OANDA REST API.

This module centralises detection of the account currency so that both the
training and live trading stacks can rely on a single source of truth.
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
import oandapyV20
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.accounts as accounts


logger = logging.getLogger("AccountProfile")


@dataclass(frozen=True)
class AccountProfile:
    account_id: str
    currency: str
    environment: str
    alias: Optional[str]
    hedging_enabled: Optional[bool]


class AccountProfileError(RuntimeError):
    """Raised when the account profile cannot be retrieved from OANDA."""


def _build_api_client(api_key: str, environment: str) -> oandapyV20.API:
    return oandapyV20.API(access_token=api_key, environment=environment)


def _fetch_account_summary(api_key: str, account_id: str, environment: str) -> dict:
    client = _build_api_client(api_key=api_key, environment=environment)
    request = accounts.AccountSummary(account_id)
    return client.request(request)


def _detect_environment() -> str:
    env = os.getenv("OANDA_ENVIRONMENT", "practice").strip().lower()
    if env not in {"practice", "live"}:
        logger.warning("Unknown OANDA_ENVIRONMENT '%s'; defaulting to 'practice'.", env)
        env = "practice"
    return env


def _load_credentials() -> tuple[str, str, str]:
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    dotenv_path = os.path.join(project_root, ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
    else:
        load_dotenv()

    api_key = os.getenv("OANDA_API_KEY", "").strip()
    account_id = os.getenv("OANDA_ACCOUNT_ID", "").strip()
    environment = _detect_environment()

    if not api_key or not account_id:
        raise AccountProfileError(
            "OANDA_API_KEY and OANDA_ACCOUNT_ID must be provided in the environment."
        )
    return api_key, account_id, environment


@functools.lru_cache(maxsize=1)
def get_account_profile(refresh: bool = False) -> AccountProfile:
    """
    Retrieve the OANDA account profile, caching the result for subsequent calls.

    Args:
        refresh: bypass the cache and query the API again.

    Raises:
        AccountProfileError: if the API call fails or returns unexpected data.
    """
    if refresh:
        get_account_profile.cache_clear()  # type: ignore[attr-defined]

    try:
        api_key, account_id, environment = _load_credentials()
        summary = _fetch_account_summary(api_key, account_id, environment)
        account = summary.get("account", {})
        currency = str(account.get("currency", "")).strip().upper()
        if not currency:
            raise AccountProfileError("Unable to determine account currency from OANDA summary.")
        alias = account.get("alias")
        hedging_enabled = account.get("hedgingEnabled")
        profile = AccountProfile(
            account_id=account_id,
            currency=currency,
            environment=environment,
            alias=alias,
            hedging_enabled=hedging_enabled,
        )
        logger.info("Account profile resolved: %s", profile)
        return profile
    except V20Error as api_err:
        raise AccountProfileError(f"OANDA API error retrieving account profile: {api_err}") from api_err
    except Exception as exc:
        raise AccountProfileError(f"Failed to load account profile: {exc}") from exc


def require_account_currency() -> str:
    """
    Convenience helper returning the detected account currency.

    Raises:
        AccountProfileError: when the account profile cannot be loaded.
    """
    return get_account_profile().currency
