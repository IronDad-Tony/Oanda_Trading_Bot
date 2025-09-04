import logging
import os
import threading
import time
from typing import Dict, Any, Optional

from ..core.oanda_client import OandaClient
from .position_manager import PositionManager
from ..core.system_state import SystemState
from ..database.database_manager import DatabaseManager
from .risk_manager import RiskManager
from ..core.execution_metrics import ExecutionMetricsLogger


class OrderManager:
    """
    Manages the full lifecycle of an order: risk assessment, execution,
    and position tracking.
    """

    def __init__(
        self,
        client: OandaClient,
        system_state: SystemState,
        position_manager: PositionManager,
        risk_manager: RiskManager,
        db_manager: DatabaseManager,
        config: Optional[Dict[str, Any]] = None,
    ):
        # 添加参数验证
        if not all([client, system_state, position_manager, risk_manager, db_manager]):
            raise ValueError("OrderManager missing required dependencies")
        
        self.logger = logging.getLogger(__name__)
        self.client = client
        self.system_state = system_state
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.db_manager = db_manager
        # Execution control
        self.exec_cfg = (config or {}).get("execution", {}) if isinstance(config, dict) else {}
        self.order_fill_timeout_seconds: float = float(self.exec_cfg.get("order_fill_timeout_seconds", 3))
        self.min_interval_between_orders_ms: int = int(self.exec_cfg.get("min_interval_between_orders_ms", 250))
        self.use_price_bound: bool = bool(self.exec_cfg.get("use_price_bound", True))
        self.max_slippage_pips: float = float(self.exec_cfg.get("max_slippage_pips", 1.0))
        self.use_transactions_stream: bool = bool(self.exec_cfg.get("use_transactions_stream", False))

        self._symbol_locks: Dict[str, threading.Lock] = {}
        self._last_order_ts_ms: Dict[str, float] = {}
        self._inflight_events: Dict[str, threading.Event] = {}

        # Metrics logger for calibration
        try:
            project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            metrics_path = os.path.join(project_root_path, 'logs', 'execution_metrics.csv')
            self.metrics = ExecutionMetricsLogger(metrics_path)
        except Exception:
            self.metrics = None

        self.logger.info("OrderManager initialized with per-symbol sequencing and slippage protection.")

    def _get_lock(self, instrument: str) -> threading.Lock:
        if instrument not in self._symbol_locks:
            self._symbol_locks[instrument] = threading.Lock()
        return self._symbol_locks[instrument]

    def _pip_size(self, instrument: str) -> float:
        try:
            iim = self.system_state.get_instrument_manager()
            details = iim.get_details(instrument)
            if details and hasattr(details, 'pip_location'):
                return 10 ** details.pip_location
        except Exception:
            pass
        # Fallback for common FX
        return 0.0001

    def _build_price_bound(self, instrument: str, side_units: int, ref_price: Optional[float]) -> Optional[float]:
        if not self.use_price_bound or ref_price is None:
            return None
        pip = self._pip_size(instrument)
        bound = self.max_slippage_pips * pip
        # BUY: cap max price; SELL: cap min price
        if side_units > 0:
            return float(ref_price + bound)
        else:
            return float(ref_price - bound)

    def process_signal(self, signal_info: Dict[str, Any]):
        """
        Processes a trading signal, assesses risk, and places an order if approved.

        Args:
            signal_info (Dict[str, Any]): Signal details from TradingLogic.
        """
        instrument = signal_info.get("instrument")
        signal = signal_info.get("signal")
        price = signal_info.get("price")
        override_units = signal_info.get("override_units")
        no_sl_tp = bool(signal_info.get("no_sl_tp", False))

        self.logger.info(f"Processing signal for {instrument} at price {price} with signal {signal}.")

        # 0. Per-symbol sequencing and simple rate limit to avoid API spam
        lock = self._get_lock(instrument)
        if not lock.acquire(blocking=False):
            self.logger.info(f"Instrument {instrument} has an in-flight order. Skipping new signal until fill/cancel.")
            return
        try:
            now_ms = time.time() * 1000.0
            last_ms = self._last_order_ts_ms.get(instrument, 0)
            if now_ms - last_ms < self.min_interval_between_orders_ms:
                self.logger.info(f"Instrument {instrument}: throttled. Last order {now_ms - last_ms:.0f}ms ago.")
                return

            # 1. Assess the trade with the Risk Manager
            trade_details = self.risk_manager.assess_trade(instrument, signal, price)

            if not trade_details:
                if override_units is not None:
                    self.logger.warning(f"RiskManager declined trade for {instrument}; proceeding with override_units={override_units} (test mode).")
                    trade_details = {"units": int(override_units)}
                else:
                    self.logger.warning(f"Trade for {instrument} was not approved by RiskManager.")
                    return

            # 1.5 If opposite position exists, close it first before proceeding
            try:
                existing = self.position_manager.get_position(instrument)
                if existing:
                    if (signal == 1 and existing.position_type == 'short') or (signal == -1 and existing.position_type == 'long'):
                        self.logger.info(f"Opposite position detected on {instrument} (existing={existing.position_type}). Closing before new order.")
                        resp = None
                        if existing.position_type == 'long':
                            resp = self.client.close_position(instrument, long_units="ALL")
                        else:
                            resp = self.client.close_position(instrument, short_units="ALL")
                        if resp:
                            fill_tx = resp.get('longOrderFillTransaction') or resp.get('shortOrderFillTransaction')
                            if fill_tx and fill_tx.get('tradesClosed'):
                                for closed_trade in fill_tx.get('tradesClosed', []):
                                    try:
                                        trade_id_c = closed_trade.get("tradeID")
                                        price_c = float(closed_trade.get("price", 0))
                                        pnl_c = float(closed_trade.get("realizedPL", 0))
                                        self.db_manager.update_trade_on_close(trade_id_c, price_c, pnl_c)
                                    except Exception:
                                        pass
                                self.position_manager.close_position(instrument)
                        else:
                            self.logger.warning(f"Close position API returned no response for {instrument}.")
            except Exception as e:
                self.logger.error(f"Failed to close opposite position for {instrument}: {e}")

            # 2. If trade is approved, create the order
            self.logger.info(f"Trade approved. Preparing to place order for {instrument}.")
            
            # Map risk manager output to OANDA API format
            if override_units is not None:
                trade_details["units"] = int(override_units)
            stop_loss_on_fill = None if no_sl_tp else ({"price": str(trade_details["stop_loss"])} if trade_details.get("stop_loss") else None)
            take_profit_on_fill = None if no_sl_tp else ({"price": str(trade_details["take_profit"])} if trade_details.get("take_profit") else None)

            # Optional priceBound to protect against large slippage
            ref_price = float(price) if price is not None else None
            price_bound = self._build_price_bound(instrument, int(trade_details["units"]), ref_price)

            # Tag the order for idempotency/debug
            client_extensions = {
                "id": f"{instrument}-{int(time.time()*1000)}",
                "tag": "live_exec",
            }

            def _submit_order(units: int, price_bound_val: Optional[float]) -> Dict[str, Any] | None:
                req_ms = int(time.time() * 1000)
                if hasattr(self.client, 'create_order_v2'):
                    resp_local = self.client.create_order_v2(
                        instrument=instrument,
                        units=units,
                        stop_loss_on_fill=stop_loss_on_fill,
                        take_profit_on_fill=take_profit_on_fill,
                        price_bound=price_bound_val,
                        client_extensions=client_extensions,
                        time_in_force="FOK",
                    )
                else:
                    resp_local = self.client.create_order(
                        instrument=instrument,
                        units=units,
                        stop_loss_on_fill=stop_loss_on_fill,
                        take_profit_on_fill=take_profit_on_fill,
                    )
                return resp_local, req_ms, int(time.time() * 1000)

            # First attempt
            order_response, request_ts_ms, response_ts_ms = _submit_order(trade_details["units"], price_bound)
            attempted_retry = False

            # 3. If order is filled, update the Position Manager and log to DB
            if order_response and "orderFillTransaction" in order_response:
                fill_details = order_response["orderFillTransaction"]
                self.logger.debug(f"Full orderFillTransaction response: {fill_details}")

                trade_id = None
                # Correctly parse the tradeID from the 'tradeOpened' dictionary
                if "tradeOpened" in fill_details and isinstance(fill_details["tradeOpened"], dict):
                    trade_id = fill_details["tradeOpened"].get("tradeID")

                if not trade_id:
                    self.logger.error("CRITICAL: Could not extract tradeID from orderFillTransaction.")
                    # Depending on strictness, we might want to stop or handle this error
                    return 

                executed_price = float(fill_details.get("price", 0))
                executed_units = int(fill_details.get("units", 0))
                
                self.logger.info(f"Order for {instrument} filled. Trade ID: {trade_id}, Price: {executed_price}, Units: {executed_units}.")
                
                # Update internal state
                self.position_manager.update_position(
                    instrument=instrument,
                    units=executed_units,
                    price=executed_price,
                    trade_id=trade_id
                )
                
                # Log trade to database
                self.db_manager.log_trade(
                    trade_id=trade_id,
                    instrument=instrument,
                    units=executed_units,
                    entry_price=executed_price,
                    signal=signal,
                    status="OPEN"
                )
                self._last_order_ts_ms[instrument] = time.time() * 1000.0
                # Metrics
                if self.metrics:
                    try:
                        self.metrics.log({
                            'instrument': instrument,
                            'client_id': client_extensions.get('id') if client_extensions else None,
                            'side': 'BUY' if executed_units > 0 else 'SELL',
                            'units': executed_units,
                            'request_ts_ms': request_ts_ms,
                            'response_ts_ms': response_ts_ms,
                            'server_time': fill_details.get('time'),
                            'ref_price': ref_price,
                            'price_bound': price_bound,
                            'fill_price': executed_price,
                            'status': 'FILLED',
                            'reason': fill_details.get('reason'),
                        })
                    except Exception:
                        pass

            elif order_response and "orderCancelTransaction" in order_response:
                reason = order_response['orderCancelTransaction'].get('reason')
                self.logger.warning(f"Order for {instrument} was cancelled. Reason: {reason}")
                if self.metrics:
                    try:
                        self.metrics.log({
                            'instrument': instrument,
                            'client_id': client_extensions.get('id') if client_extensions else None,
                            'side': 'BUY' if trade_details["units"] > 0 else 'SELL',
                            'units': int(trade_details["units"]),
                            'request_ts_ms': request_ts_ms,
                            'response_ts_ms': response_ts_ms,
                            'server_time': order_response['orderCancelTransaction'].get('time'),
                            'ref_price': ref_price,
                            'price_bound': price_bound,
                            'fill_price': None,
                            'status': 'CANCEL',
                            'reason': reason,
                        })
                    except Exception:
                        pass
                # Optional single retry on common transient reasons with refreshed price
                try:
                    if (not attempted_retry) and str(reason).upper() in ("PRICE_BOUND", "PRICE_BOUND_EXCEEDED", "INSUFFICIENT_LIQUIDITY"):
                        self.logger.info(f"Retrying order once for {instrument} after cancel reason={reason}.")
                        latest = self.client.get_bid_ask_candles_combined(instrument, count=1) or []
                        if latest:
                            last = latest[-1]
                            new_ref = float(last.get('ask_close') or last.get('bid_close') or 0.0)
                        else:
                            new_ref = ref_price
                        new_bound = self._build_price_bound(instrument, int(trade_details["units"]), new_ref)
                        order_response2, request_ts_ms2, response_ts_ms2 = _submit_order(trade_details["units"], new_bound)
                        attempted_retry = True
                        # Promote retry response to main flow handling
                        order_response = order_response2 or order_response
                        request_ts_ms = request_ts_ms2 or request_ts_ms
                        response_ts_ms = response_ts_ms2 or response_ts_ms
                        # fall through to handling below with updated response
                except Exception as e:
                    self.logger.error(f"Retry submission failed for {instrument}: {e}")
            else:
                self.logger.error(f"Order execution failed or no fill details received for {instrument}. Response: {order_response}")
                if self.metrics:
                    try:
                        self.metrics.log({
                            'instrument': instrument,
                            'client_id': client_extensions.get('id') if client_extensions else None,
                            'side': 'BUY' if trade_details["units"] > 0 else 'SELL',
                            'units': int(trade_details["units"]),
                            'request_ts_ms': request_ts_ms,
                            'response_ts_ms': response_ts_ms,
                            'server_time': None,
                            'ref_price': ref_price,
                            'price_bound': price_bound,
                            'fill_price': None,
                            'status': 'ERROR',
                            'reason': 'NO_FILL_DETAILS',
                        })
                    except Exception:
                        pass
            
            # If using transactions stream, wait for definitive event or timeout before releasing lock
            if self.use_transactions_stream:
                evt = self._inflight_events.get(instrument)
                if not evt:
                    evt = threading.Event()
                    self._inflight_events[instrument] = evt
                evt.wait(timeout=self.order_fill_timeout_seconds)
        finally:
            try:
                lock.release()
            except Exception:
                pass

    def on_transaction_event(self, tx: Dict[str, Any]):
        """Called by the transaction stream worker upon receiving a transaction event.

        We only use it for sequencing release and optional logging. Position/DB updates are
        handled on synchronous responses already.
        """
        try:
            ttype = tx.get('type') or ''
            # Attempt to extract instrument
            instrument = tx.get('instrument') or ''
            if not instrument:
                # Some transaction payloads nest data; best-effort scan
                instrument = tx.get('orderFillTransaction', {}).get('instrument', '')
            if not instrument:
                return
            if ttype in ("ORDER_FILL", "ORDER_CANCEL", "ORDER_REJECT"):
                self._last_order_ts_ms[instrument] = time.time() * 1000.0
                # Signal any waiter
                evt = self._inflight_events.get(instrument)
                if evt:
                    evt.set()
        except Exception as e:
            self.logger.debug(f"on_transaction_event error: {e}")

    def close_all_positions(self):
        """
        Closes all open positions and updates the database with the final PnL.
        """
        self.logger.warning("EMERGENCY: Closing all open positions.")
        open_positions_data = self.client.get_open_positions()
        
        if not open_positions_data or not open_positions_data.get("positions"):
            self.logger.info("No open positions found on OANDA to close.")
            return

        for position in open_positions_data["positions"]:
            instrument = position["instrument"]
            internal_pos = self.position_manager.get_position(instrument)
            if not internal_pos:
                self.logger.warning(f"Found position for {instrument} on OANDA but not in internal state. Skipping DB update for this one.")
                # Decide if you still want to close it without internal tracking
                # For safety, we will proceed to close it.

            long_units = position.get("long", {}).get("units", "0")
            short_units = position.get("short", {}).get("units", "0")

            response = None
            if long_units != "0":
                self.logger.info(f"Closing long position for {instrument} ({long_units} units)...")
                response = self.client.close_position(instrument, long_units="ALL")
            elif short_units != "0":
                self.logger.info(f"Closing short position for {instrument} ({short_units} units)...")
                response = self.client.close_position(instrument, short_units="ALL")

            if response:
                # Extract the closing transaction details
                fill_transaction = response.get('longOrderFillTransaction') or response.get('shortOrderFillTransaction')
                if fill_transaction and fill_transaction.get('tradesClosed'):
                    self.logger.info(f"Successfully submitted close order for {instrument}.")
                    
                    # Process each closed trade within the transaction
                    for closed_trade in fill_transaction.get('tradesClosed', []):
                        trade_id = closed_trade.get("tradeID")
                        price = float(closed_trade.get("price", 0))
                        pnl = float(closed_trade.get("realizedPL", 0))
                        
                        self.logger.info(f"Trade {trade_id} closed. Close Price: {price}, PnL: {pnl}")
                        self.db_manager.update_trade_on_close(trade_id, price, pnl)

                    # Remove from internal state after processing all parts of the closure
                    self.position_manager.close_position(instrument)
                else:
                    self.logger.error(f"Failed to get fill details when closing {instrument}. Response: {response}")
            else:
                self.logger.error(f"Failed to receive response when closing position for {instrument}.")
