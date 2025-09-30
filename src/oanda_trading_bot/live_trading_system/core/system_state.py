"""Global system state for the live trading engine."""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

try:
    from oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager  # type: ignore
except ImportError:  # pragma: no cover - very early import fallback
    class InstrumentInfoManager:  # type: ignore[too-many-instance-attributes]
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            """Minimal fallback used when the real manager is unavailable."""

        def get_all_available_symbols(self) -> List[str]:
            return ["EUR_USD", "GBP_USD"]

        def get_details(self, symbol: str) -> Any:  # noqa: ANN401
            return None


class SystemState:
    """Singleton that tracks the live trading engine state and UI-facing metadata."""

    _instance: Optional["SystemState"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "SystemState":  # noqa: D401
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
        return cls._instance

    # ------------------------------------------------------------------
    # Internal initialisation helpers
    # ------------------------------------------------------------------
    def _initialize(self) -> None:
        self.logger = logging.getLogger("LiveTradingSystem")
        self.is_running: bool = False
        self.ui_controls_enabled: bool = True
        self.last_error: Optional[str] = None
        self.status_message: str = "Idle"

        # Session configuration state
        self.selected_instruments: List[str] = ["EUR_USD"]
        self.current_model: Optional[str] = None
        self.current_primary_instrument: Optional[str] = None
        self.target_capital: Optional[float] = None
        self.session_start_equity: Optional[float] = None
        self.session_peak_equity: Optional[float] = None
        self.session_hard_stop_reason: Optional[str] = None

        # Instrument metadata cache
        self.instrument_info_manager: Optional[InstrumentInfoManager] = None
        self.categorized_instruments: Dict[str, List[str]] = {}
        self._suspended: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Accessors and mutators used across the engine
    # ------------------------------------------------------------------
    def get_instrument_manager(self) -> InstrumentInfoManager:
        """Return a cached InstrumentInfoManager, creating one if needed."""
        if self.instrument_info_manager is not None:
            return self.instrument_info_manager

        try:
            from dotenv import load_dotenv

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            dotenv_path = os.path.join(project_root, ".env")
            if os.path.exists(dotenv_path):
                load_dotenv(dotenv_path=dotenv_path)
            else:
                load_dotenv()

            api_key = os.getenv("OANDA_API_KEY")
            account_id = os.getenv("OANDA_ACCOUNT_ID")

            if not api_key or not account_id:
                raise ValueError("OANDA credentials not found in environment variables.")

            self.instrument_info_manager = InstrumentInfoManager(
                api_key=api_key,
                account_id=account_id,
                base_url=None,
                force_refresh=False,
            )
        except Exception as exc:  # pragma: no cover - best-effort fallback
            self.logger.error("Failed to initialise InstrumentInfoManager: %s", exc, exc_info=True)

            class _FallbackInstrumentInfoManager(InstrumentInfoManager):  # type: ignore[misc]
                def get_all_available_symbols(self) -> List[str]:
                    return ["EUR_USD", "GBP_USD", "USD_JPY"]

                def get_details(self, symbol: str) -> Any:  # noqa: ANN401
                    return None

            self.instrument_info_manager = _FallbackInstrumentInfoManager()

        return self.instrument_info_manager

    # Selected symbols -------------------------------------------------
    def get_selected_instruments(self) -> List[str]:
        return list(self.selected_instruments)

    def set_selected_instruments(self, instruments: List[str]) -> None:
        self.selected_instruments = [sym.upper() for sym in instruments]
        self.status_message = f"Symbols updated: {', '.join(self.selected_instruments)}"

    # Current model ----------------------------------------------------
    def get_current_model(self) -> Optional[str]:
        return self.current_model

    def set_current_model(self, model_path: Optional[str]) -> None:
        self.current_model = model_path
        if model_path:
            self.status_message = f"Model loaded: {model_path}"

    # Primary instrument toggle (used by monitor) ----------------------
    def get_current_instrument(self) -> Optional[str]:
        return self.current_primary_instrument

    def set_current_instrument(self, instrument: Optional[str]) -> None:
        self.current_primary_instrument = instrument.upper() if instrument else None

    # Session lifecycle ------------------------------------------------
    def start(self) -> None:
        self.is_running = True
        self.status_message = "Engine running"
        self.session_hard_stop_reason = None

    def stop(self) -> None:
        self.is_running = False
        self.status_message = "Engine stopped"

    def set_error(self, message: str) -> None:
        self.last_error = message
        self.status_message = f"Error: {message}"
        self.is_running = False

    # Capital management -----------------------------------------------
    def set_target_capital(self, amount: Optional[float]) -> None:
        self.target_capital = float(amount) if amount is not None else None

    def get_target_capital(self) -> Optional[float]:
        return self.target_capital

    def mark_session_start_equity(self, equity: Optional[float]) -> None:
        self.session_start_equity = float(equity) if equity is not None else None
        self.session_peak_equity = self.session_start_equity

    def update_peak_equity(self, equity: Optional[float]) -> None:
        if equity is None:
            return
        val = float(equity)
        if self.session_peak_equity is None or val > self.session_peak_equity:
            self.session_peak_equity = val

    def record_hard_stop(self, reason: str) -> None:
        self.session_hard_stop_reason = reason
        self.status_message = reason
        self.is_running = False

    def clear_hard_stop(self) -> None:
        self.session_hard_stop_reason = None

    # Snapshot for UI --------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "running": self.is_running,
            "status_message": self.status_message,
            "last_error": self.last_error,
            "selected_instruments": list(self.selected_instruments),
            "current_model": self.current_model,
            "target_capital": self.target_capital,
            "session_start_equity": self.session_start_equity,
            "session_peak_equity": self.session_peak_equity,
            "hard_stop_reason": self.session_hard_stop_reason,
        }


# Singleton instance exposed for convenience
system_state = SystemState()
