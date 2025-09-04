from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from collections import deque, defaultdict
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteEdge:
    symbol: str            # e.g., "EUR_USD"
    from_ccy: str          # e.g., "EUR"
    to_ccy: str            # e.g., "USD"


@dataclass
class ConversionRoute:
    from_ccy: str
    to_ccy: str
    edges: List[RouteEdge]  # Ordered edges to traverse

    def as_pairs(self) -> Set[str]:
        return {edge.symbol for edge in self.edges}


def _build_currency_graph(instrument_details_map: Dict[str, object]) -> Dict[str, List[RouteEdge]]:
    """Build a directed graph of currencies from instrument details.

    instrument_details_map: symbol -> object having .base_currency and .quote_currency
    Returns adjacency list: currency -> list[RouteEdge]
    """
    graph: Dict[str, List[RouteEdge]] = defaultdict(list)
    for sym, det in instrument_details_map.items():
        if det is None:
            continue
        try:
            base = str(det.base_currency).upper()
            quote = str(det.quote_currency).upper()
            if not base or not quote or base == quote:
                continue
            # Add both directions (we will apply correct pricing/markup later)
            graph[base].append(RouteEdge(symbol=sym, from_ccy=base, to_ccy=quote))
            graph[quote].append(RouteEdge(symbol=sym, from_ccy=quote, to_ccy=base))
        except Exception as e:
            logger.debug(f"Skipping {sym} due to details parse error: {e}")
    return graph


def find_route_between_currencies(
    from_currency: str,
    to_currency: str,
    available_symbols: Set[str],
    instrument_info_manager,
) -> Optional[ConversionRoute]:
    """BFS to find a minimal-hop route from from_currency to to_currency.

    - available_symbols: the OANDA tradable instruments set
    - instrument_info_manager: instance with get_details(symbol)
    Returns ConversionRoute or None if not found.
    """
    from_ccy = from_currency.upper()
    to_ccy = to_currency.upper()
    if from_ccy == to_ccy:
        return ConversionRoute(from_ccy, to_ccy, edges=[])

    # Preload details for available symbols that look like pairs or carry clear currencies
    details_map: Dict[str, object] = {}
    for sym in available_symbols:
        det = instrument_info_manager.get_details(sym)
        if det is not None:
            details_map[sym] = det

    graph = _build_currency_graph(details_map)

    # Standard BFS
    q = deque([from_ccy])
    prev: Dict[str, Tuple[str, RouteEdge]] = {}  # currency -> (prev_currency, edge)
    seen = {from_ccy}
    found = False
    while q:
        cur = q.popleft()
        for edge in graph.get(cur, []):
            nxt = edge.to_ccy
            if nxt in seen:
                continue
            seen.add(nxt)
            prev[nxt] = (cur, edge)
            if nxt == to_ccy:
                found = True
                q.clear()
                break
            q.append(nxt)

    if not found:
        logger.debug(f"No route found: {from_ccy} -> {to_ccy}")
        return None

    # Reconstruct
    edges: List[RouteEdge] = []
    cur = to_ccy
    while cur != from_ccy:
        p, e = prev[cur]
        edges.append(e)
        cur = p
    edges.reverse()
    return ConversionRoute(from_ccy, to_ccy, edges)


def compute_required_pairs_for_training(
    trading_symbols: List[str],
    account_currency: str,
    instrument_info_manager,
) -> Tuple[Set[str], Dict[str, ConversionRoute]]:
    """For selected trading symbols, compute the union of conversion pairs needed
    to convert each symbol's quote currency to the account currency.

    Returns (required_symbols_set, route_map)
    where route_map maps trading_symbol -> ConversionRoute
    """
    available_symbols: Set[str] = set(instrument_info_manager.get_all_available_symbols())
    required: Set[str] = set(trading_symbols)
    routes: Dict[str, ConversionRoute] = {}
    for sym in trading_symbols:
        det = instrument_info_manager.get_details(sym)
        if det is None:
            logger.warning(f"Instrument details not found for {sym}; skipping route computation.")
            continue
        quote = str(det.quote_currency).upper()
        ac = account_currency.upper()
        if quote == ac:
            routes[sym] = ConversionRoute(quote, ac, edges=[])
            continue
        route = find_route_between_currencies(quote, ac, available_symbols, instrument_info_manager)
        if route is None:
            logger.warning(f"No conversion route found for {sym} ({quote}->{ac}).")
            continue
        routes[sym] = route
        required |= route.as_pairs()
    return required, routes


def compute_conversion_rate_along_route(
    route: ConversionRoute,
    prices_map: Dict[str, Tuple[Decimal, Decimal]],  # symbol -> (bid, ask)
    apply_oanda_markup: bool = True,
) -> Optional[Decimal]:
    """Compute a conversion rate along a precomputed route using midpoint and OANDA-like markup.

    The direction of each edge dictates debit/credit adjustment:
    - Step from instrument base->quote is treated as credit (markup -)
    - Step from quote->base is treated as debit (markup +), and inverted
    """
    from .currency_manager import CurrencyDependencyManager  # lazy import to avoid cycles
    mgr = CurrencyDependencyManager(account_currency=route.to_ccy, apply_oanda_markup=apply_oanda_markup)

    rate = Decimal('1')
    for edge in route.edges:
        bid, ask = prices_map.get(edge.symbol, (None, None))
        if bid is None or ask is None:
            logger.warning(f"Missing prices for {edge.symbol} while computing route {route.from_ccy}->{route.to_ccy}")
            return None
        midpoint = mgr.get_midpoint_rate(bid, ask)
        if edge.from_ccy + "_" + edge.to_ccy == edge.symbol:
            step = mgr.apply_conversion_fee(midpoint, is_credit=True)
            rate *= step
        else:
            step = mgr.apply_conversion_fee(midpoint, is_credit=False)
            if step == 0:
                return None
            rate *= (Decimal('1') / step)
    return rate

