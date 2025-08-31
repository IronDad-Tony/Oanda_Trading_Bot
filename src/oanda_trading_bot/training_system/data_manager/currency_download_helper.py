from typing import List, Set, Tuple
import logging

logger = logging.getLogger(__name__)


def ensure_currency_data_for_trading(
    trading_symbols: List[str],
    account_currency: str,
    start_time_iso: str,
    end_time_iso: str,
    granularity: str,
    streamlit_progress_bar=None,
    streamlit_status_text=None,
    perform_download: bool = True,
) -> Tuple[bool, Set[str]]:
    """
    Ensures historical data for both trading symbols and any required FX pairs for conversion
    to the account currency are present. Optionally downloads missing segments and reports
    progress via Streamlit UI elements.

    Returns (success: bool, all_symbols: set)
    """
    try:
        # Lazy imports to avoid circulars and keep module import cheap
        from oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager
        from oanda_trading_bot.training_system.data_manager.currency_manager import (
            get_required_conversion_pairs,
        )

        instrument_info_manager = InstrumentInfoManager()
        available_instruments = set(instrument_info_manager.get_all_available_symbols())

        required_pairs = get_required_conversion_pairs(
            trading_symbols, account_currency, available_instruments
        )
        all_symbols = set(trading_symbols) | required_pairs

        if perform_download:
            from oanda_trading_bot.training_system.data_manager.oanda_downloader import (
                manage_data_download_for_symbols,
            )

            manage_data_download_for_symbols(
                symbols=sorted(list(all_symbols)),
                overall_start_str=start_time_iso,
                overall_end_str=end_time_iso,
                granularity=granularity,
                streamlit_progress_bar=streamlit_progress_bar,
                streamlit_status_text=streamlit_status_text,
            )

        return True, all_symbols
    except Exception as e:
        logger.error(
            f"currency_download_helper.ensure_currency_data_for_trading failed: {e}",
            exc_info=True,
        )
        return False, set(trading_symbols)

