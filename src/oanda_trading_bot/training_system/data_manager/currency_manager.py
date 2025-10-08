"""
Currency conversion utilities with explicit routing and OANDA-style markup.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import logging
from typing import Dict, Iterable, Mapping, Optional, Set, Tuple

from oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager

from .currency_route_planner import (
    ConversionRoute,
    compute_conversion_rate_along_route,
    compute_required_pairs_for_training,
    find_route_between_currencies,
)

logger = logging.getLogger("CurrencyManager")

OANDA_MARKUP = Decimal("0.005")


class ConversionRateError(RuntimeError):
    """Raised when a conversion route or price is missing."""


@dataclass(frozen=True)
class ConversionResult:
    rate: Decimal
    route: ConversionRoute


class CurrencyDependencyManager:
    """
    Maintains conversion routes from instrument quote currencies to the
    configured account currency and evaluates conversion rates using live prices.
    """

    def __init__(
        self,
        account_currency: str,
        apply_oanda_markup: bool = True,
        instrument_info_manager: Optional[InstrumentInfoManager] = None,
    ) -> None:
        self.account_currency = account_currency.upper()
        self.apply_oanda_markup = apply_oanda_markup
        self.instrument_info_manager = instrument_info_manager or InstrumentInfoManager()
        self._route_cache: Dict[str, ConversionRoute] = {}
        self._available_symbols: Optional[Set[str]] = None

    # ------------------------------------------------------------------ #
    # Route management
    # ------------------------------------------------------------------ #

    def _load_available_symbols(self) -> Set[str]:
        if self._available_symbols is None:
            symbols = self.instrument_info_manager.get_all_available_symbols()
            if not symbols:
                raise ConversionRateError("Unable to fetch tradable symbols from OANDA.")
            self._available_symbols = {str(sym).upper() for sym in symbols}
        return self._available_symbols

    def _cache_route(self, currency: str, route: ConversionRoute) -> None:
        self._route_cache[currency.upper()] = route

    def _route_for_currency(self, currency: str) -> ConversionRoute:
        currency = currency.upper()
        if currency == self.account_currency:
            return ConversionRoute(currency, currency, edges=[])

        if currency in self._route_cache:
            return self._route_cache[currency]

        available_symbols = self._load_available_symbols()
        route = find_route_between_currencies(
            from_currency=currency,
            to_currency=self.account_currency,
            available_symbols=available_symbols,
            instrument_info_manager=self.instrument_info_manager,
        )
        if route is None:
            raise ConversionRateError(
                f"No conversion route found from {currency} to {self.account_currency}."
            )
        self._cache_route(currency, route)
        return route

    def refresh_routes_for_symbols(self, trading_symbols: Iterable[str]) -> Set[str]:
        """
        Pre-compute routes for trading symbols' quote currencies and returns the
        extended symbol universe required for data downloads.
        """
        trading_symbols = [sym.upper() for sym in trading_symbols]
        required, routes = compute_required_pairs_for_training(
            trading_symbols=trading_symbols,
            account_currency=self.account_currency,
            instrument_info_manager=self.instrument_info_manager,
        )
        for route in routes.values():
            self._cache_route(route.from_ccy.upper(), route)
        return required

    # ------------------------------------------------------------------ #
    # Conversion evaluation
    # ------------------------------------------------------------------ #

    def convert_to_account_currency(
        self,
        from_currency: str,
        prices_map: Mapping[str, Tuple[Decimal | float, Decimal | float]],
        is_credit: bool = True,
    ) -> Decimal:
        """
        Return the conversion rate from `from_currency` into the account currency.

        Args:
            from_currency: currency code of the value being converted.
            prices_map: mapping of symbol -> (bid, ask) tuples.
            is_credit: True when the conversion credits the account (favourable
                markup), False when the account is debited (adverse markup).
        """
        route = self._route_for_currency(from_currency)
        if not route.edges:
            return Decimal("1")

        normalised_prices: Dict[str, Tuple[Decimal, Decimal]] = {}
        for sym, (bid, ask) in prices_map.items():
            try:
                normalised_prices[sym.upper()] = (Decimal(str(bid)), Decimal(str(ask)))
            except Exception as exc:
                raise ConversionRateError(f"Invalid price data for {sym}: {exc}") from exc

        rate = compute_conversion_rate_along_route(
            route=route,
            prices_map=normalised_prices,
            apply_oanda_markup=False,  # apply markup manually once we know the direction
        )
        if rate is None:
            raise ConversionRateError(
                f"Missing price data to convert {from_currency} to {self.account_currency}."
            )

        if self.apply_oanda_markup:
            adjustment = Decimal("1") - OANDA_MARKUP if is_credit else Decimal("1") + OANDA_MARKUP
            rate *= adjustment

        return rate

    def convert_amount(
        self,
        amount: Decimal | float,
        from_currency: str,
        prices_map: Mapping[str, Tuple[Decimal | float, Decimal | float]],
        is_credit: bool = True,
    ) -> Decimal:
        rate = self.convert_to_account_currency(from_currency, prices_map, is_credit=is_credit)
        return Decimal(str(amount)) * rate


# --------------------------------------------------------------------------- #
# Data preparation helpers
# --------------------------------------------------------------------------- #


def ensure_currency_data_for_trading(
    trading_symbols: Iterable[str],
    account_currency: str,
    start_time_iso: str,
    end_time_iso: str,
    granularity: str,
) -> Tuple[bool, Set[str]]:
    """
    Ensure that all currency pairs required for conversion are downloaded.
    """
    symbol_list = [sym.upper() for sym in trading_symbols]
    manager = InstrumentInfoManager()
    try:
        currency_manager = CurrencyDependencyManager(
            account_currency=account_currency,
            instrument_info_manager=manager,
        )
        required_symbols = currency_manager.refresh_routes_for_symbols(symbol_list)
    except ConversionRateError as exc:
        logger.error("Failed to prepare conversion routes: %s", exc)
        raise

    if not required_symbols:
        required_symbols = set(symbol_list)

    try:
        from oanda_trading_bot.training_system.data_manager.oanda_downloader import (
            manage_data_download_for_symbols,
        )

        manage_data_download_for_symbols(
            symbols=sorted(required_symbols),
            overall_start_str=start_time_iso,
            overall_end_str=end_time_iso,
            granularity=granularity,
        )
        logger.info(
            "Currency data ensured for %d symbols (base universe: %s).",
            len(required_symbols),
            ", ".join(sorted(required_symbols)),
        )
        return True, required_symbols
    except Exception as exc:
        logger.error("Failed to download required currency data: %s", exc, exc_info=True)
        raise


def enhance_currency_data_management() -> None:
    logger.info("Currency data management enhanced through deterministic routing.")

