"""
Utility helpers for fetching and caching instrument metadata from OANDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from oanda_trading_bot.common.account_profile import (
    AccountProfileError,
    get_account_profile,
)

logger = logging.getLogger("InstrumentInfoManager")


@dataclass
class InstrumentDetails:
    symbol: str
    display_name: str
    instrument_type: str
    margin_rate: Decimal
    minimum_trade_size: Decimal
    trade_units_precision: int
    pip_location: int
    quote_currency: str
    base_currency: str
    max_trade_units: Optional[Decimal] = None
    financing_long_rate: Optional[Decimal] = None
    financing_short_rate: Optional[Decimal] = None

    def round_units(self, units: float | Decimal | str) -> Decimal:
        value = Decimal(str(units))
        quantizer = Decimal("1e-" + str(self.trade_units_precision))
        rounded = value.quantize(quantizer, rounding=ROUND_HALF_UP)
        minimum = self.minimum_trade_size
        if minimum > 0:
            multiplier = (rounded / minimum).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
            rounded = multiplier * minimum
        return rounded


class InstrumentInfoManager:
    """
    Lightweight cache of OANDA instrument metadata with automatic refresh.
    """

    _instrument_cache: Dict[str, InstrumentDetails] = {}
    _last_fetch_time: float = 0.0
    _cache_ttl_seconds: int = 1800

    def __init__(self, force_refresh: bool = False, cache_ttl_seconds: Optional[int] = None) -> None:
        if cache_ttl_seconds is not None:
            InstrumentInfoManager._cache_ttl_seconds = cache_ttl_seconds
        if force_refresh:
            self._refresh_cache()
        else:
            self._ensure_cache()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_all_available_symbols(self) -> List[str]:
        self._ensure_cache()
        return sorted(InstrumentInfoManager._instrument_cache.keys())

    def get_details(self, symbol: str) -> Optional[InstrumentDetails]:
        self._ensure_cache()
        return InstrumentInfoManager._instrument_cache.get(symbol.upper())

    def get_required_symbols_for_trading(self, trading_symbols: List[str], account_currency: str) -> List[str]:
        from oanda_trading_bot.training_system.data_manager.currency_route_planner import (
            compute_required_pairs_for_training,
        )

        required, _routes = compute_required_pairs_for_training(
            trading_symbols=[sym.upper() for sym in trading_symbols],
            account_currency=account_currency.upper(),
            instrument_info_manager=self,
        )
        required |= {sym.upper() for sym in trading_symbols}
        return sorted(required)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_cache(self) -> None:
        now = time.time()
        if (
            not InstrumentInfoManager._instrument_cache
            or now - InstrumentInfoManager._last_fetch_time > InstrumentInfoManager._cache_ttl_seconds
        ):
            self._refresh_cache()

    def _refresh_cache(self) -> None:
        try:
            profile = get_account_profile()
        except AccountProfileError as exc:
            raise RuntimeError(f"Cannot refresh instrument cache: {exc}") from exc

        api_key = os.getenv("OANDA_API_KEY")
        account_id = profile.account_id
        base_url = (
            "https://api-fxtrade.oanda.com/v3"
            if profile.environment == "live"
            else "https://api-fxpractice.oanda.com/v3"
        )
        url = f"{base_url}/accounts/{account_id}/instruments"
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {"accountId": account_id}
        response = requests.get(url, headers=headers, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        instruments = data.get("instruments", [])

        cache: Dict[str, InstrumentDetails] = {}
        for raw in instruments:
            try:
                details = self._parse_instrument(raw)
                cache[details.symbol] = details
            except Exception as exc:
                logger.debug("Skipping instrument due to parse error: %s", exc)

        if not cache:
            raise RuntimeError("Instrument list from OANDA was empty.")

        InstrumentInfoManager._instrument_cache = cache
        InstrumentInfoManager._last_fetch_time = time.time()
        logger.info("Instrument cache refreshed with %d entries.", len(cache))

    @staticmethod
    def _parse_instrument(raw: Dict[str, Any]) -> InstrumentDetails:
        symbol = str(raw["name"]).upper()
        display_name = str(raw.get("displayName", symbol))
        instrument_type = str(raw.get("type", "UNKNOWN"))
        margin_rate = Decimal(str(raw.get("marginRate", "0")))
        min_trade_size = Decimal(str(raw.get("minimumTradeSize", "1")))
        precision = int(raw.get("tradeUnitsPrecision", 0))
        pip_location = int(raw.get("pipLocation", -4))
        quote_currency = str(raw.get("quoteCurrency", "")).upper()
        base_currency = str(raw.get("baseCurrency", "")).upper()

        max_trade_units = raw.get("maximumOrderUnits")
        if max_trade_units is not None:
            max_trade_units = Decimal(str(max_trade_units))

        return InstrumentDetails(
            symbol=symbol,
            display_name=display_name,
            instrument_type=instrument_type,
            margin_rate=margin_rate,
            minimum_trade_size=min_trade_size,
            trade_units_precision=precision,
            pip_location=pip_location,
            quote_currency=quote_currency,
            base_currency=base_currency,
            max_trade_units=max_trade_units,
            financing_long_rate=_safe_decimal(raw.get("financing", {}).get("longRate")),
            financing_short_rate=_safe_decimal(raw.get("financing", {}).get("shortRate")),
        )


def _safe_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None

