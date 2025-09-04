#!/usr/bin/env python3
"""
Small-scale 1-day conversion route test.

What it does:
- Selects 5 symbols with varied quote currencies
- Ensures historical data for them and all required FX pairs for conversion
- Computes conversion routes (quote -> account currency) via BFS over available instruments
- Validates conversion at multiple timestamps by comparing:
    a) Precomputed-route conversion
    b) On-the-fly conversion via CurrencyDependencyManager

Run: python scripts/run_currency_conversion_route_test.py
Requires: .env with OANDA_API_KEY, OANDA_ACCOUNT_ID (practice or live) and optional ACCOUNT_CURRENCY
"""
import os
import sys
from datetime import datetime, timedelta, timezone
import random
import math


def fmt(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    src_dir = os.path.join(project_root, 'src')
    for p in [project_root, src_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Imports after path fix
    from oanda_trading_bot.training_system.common.logger_setup import logger
    from oanda_trading_bot.training_system.common.config import (
        ACCOUNT_CURRENCY, GRANULARITY,
    )
    from oanda_trading_bot.training_system.data_manager.currency_download_helper import ensure_currency_data_for_trading
    from oanda_trading_bot.training_system.data_manager.currency_route_planner import (
        compute_required_pairs_for_training, compute_conversion_rate_along_route,
    )
    from oanda_trading_bot.training_system.data_manager.currency_manager import CurrencyDependencyManager
    from oanda_trading_bot.training_system.data_manager.database_manager import query_historical_data
    from oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager
    from decimal import Decimal
    import pandas as pd

    # Choose symbols covering different quote currencies
    symbols = ["GBP_JPY", "EUR_GBP", "EUR_CAD", "XAU_USD", "USD_CHF"]

    # 1-day window aligned to recent business day
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    while end.weekday() >= 5:  # Sat/Sun
        end -= timedelta(days=1)
    start = end - timedelta(days=1)
    start_iso, end_iso = fmt(start), fmt(end)
    granularity = GRANULARITY or "S5"

    logger.info("=== Ensuring data (1 day) for test symbols and conversion pairs ===")
    ok, all_syms = ensure_currency_data_for_trading(
        trading_symbols=symbols,
        account_currency=ACCOUNT_CURRENCY,
        start_time_iso=start_iso,
        end_time_iso=end_iso,
        granularity=granularity,
        streamlit_progress_bar=None,
        streamlit_status_text=None,
        perform_download=True,
    )
    if not ok:
        logger.warning("ensure_currency_data_for_trading did not confirm success; proceeding with available DB data.")
    use_symbols = sorted(list(all_syms)) if all_syms else symbols
    logger.info(f"Symbols under test (including conversion pairs): {use_symbols}")

    # Compute conversion routes for info/logging
    iim = InstrumentInfoManager(force_refresh=False)
    required_set, routes = compute_required_pairs_for_training(symbols, ACCOUNT_CURRENCY, iim)
    for sym, route in routes.items():
        steps = " -> ".join([edge.from_ccy for edge in route.edges] + [route.to_ccy]) if route.edges else f"{route.from_ccy} -> {route.to_ccy} (direct)"
        logger.info(f"Route for {sym} ({route.from_ccy}->{route.to_ccy}): {steps}; pairs={list(route.as_pairs())}")

    # Load 1-day data for all involved symbols
    data_map = {}
    for s in use_symbols:
        df = query_historical_data(s, granularity, start_iso, end_iso, limit=None)
        if df.empty:
            logger.warning(f"No data for {s} in test window.")
        data_map[s] = df

    # Pick 6 evenly spaced timestamps within the day (based on the densest DataFrame)
    all_times = []
    for df in data_map.values():
        if not df.empty and 'time' in df.columns:
            all_times.extend(df['time'].tolist())
    if not all_times:
        logger.error("No data available to run conversion validation.")
        return
    all_times = sorted(list({t for t in all_times}))
    sample_count = 6
    idxs = [int(i * (len(all_times) - 1) / (sample_count - 1)) for i in range(sample_count)] if len(all_times) >= sample_count else list(range(len(all_times)))
    sample_times = [all_times[i] for i in idxs]
    logger.info(f"Sample timestamps: {[t.isoformat() for t in sample_times]}")

    # Helper: find last row <= ts
    def last_row_before_or_equal(df: pd.DataFrame, ts: pd.Timestamp):
        if df.empty:
            return None
        mask = df['time'] <= ts
        if not mask.any():
            return None
        return df[mask].iloc[-1]

    # Validate per symbol
    cur_mgr = CurrencyDependencyManager(ACCOUNT_CURRENCY, apply_oanda_markup=True)
    total_checks = 0
    mismatches = 0
    for sym in symbols:
        if sym not in routes:
            logger.warning(f"No route for {sym}; skipping.")
            continue
        route = routes[sym]
        det = iim.get_details(sym)
        if not det:
            logger.warning(f"No details for {sym}; skipping.")
            continue
        quote = det.quote_currency

        for ts in sample_times:
            # Build prices map for all symbols at ts from last known candle <= ts
            prices_map = {}
            for s, df in data_map.items():
                row = last_row_before_or_equal(df, ts)
                if row is not None:
                    try:
                        bid = Decimal(str(row.get('bid_close', row.get('bid_open', 0))))
                        ask = Decimal(str(row.get('ask_close', row.get('ask_open', 0))))
                        if bid and ask:
                            prices_map[s] = (bid, ask)
                    except Exception:
                        continue

            if len(prices_map) == 0:
                continue

            # a) Route-based conversion
            route_rate = compute_conversion_rate_along_route(route, prices_map, apply_oanda_markup=True)

            # b) On-the-fly conversion
            mgr_rate = cur_mgr.convert_to_account_currency(quote, prices_map, is_credit=True)

            if route_rate is None or mgr_rate is None:
                continue
            total_checks += 1
            # Compare within tolerance (markup is applied similarly, minor rounding diffs allowed)
            a, b = float(route_rate), float(mgr_rate)
            rel_err = abs(a - b) / max(1e-9, abs(b))
            if rel_err > 1e-3:  # 0.1% tolerance
                mismatches += 1
                logger.warning(f"Mismatch @ {ts.isoformat()} {sym} {quote}->{ACCOUNT_CURRENCY}: route={a:.6f}, mgr={b:.6f}, rel_err={rel_err:.6f}")

    if total_checks == 0:
        logger.warning("No comparable checks performed (insufficient data overlap).")
    else:
        logger.info(f"Validation checks: {total_checks}, mismatches(>0.1%): {mismatches}")
        if mismatches == 0:
            logger.info("All route-based conversions align with on-the-fly logic within tolerance.")

    logger.info("=== Conversion route test completed ===")


if __name__ == "__main__":
    main()

