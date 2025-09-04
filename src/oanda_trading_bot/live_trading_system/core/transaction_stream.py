import logging
import threading
import time
from typing import Optional, Callable, Dict, Any

from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.transactions as transactions


class TransactionStreamWorker:
    """
    Background thread to consume OANDA TransactionsStream and dispatch events.

    For each incoming transaction, calls the provided callback with the raw message.
    """

    def __init__(
        self,
        api_client,  # oandapyV20.API instance
        account_id: str,
        last_transaction_id_getter: Optional[Callable[[], Optional[str]]] = None,
        on_transaction: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.logger = logging.getLogger("LiveTradingSystem")
        self.client = api_client
        self.account_id = account_id
        self.last_transaction_id_getter = last_transaction_id_getter
        self.on_transaction = on_transaction
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self):
        if self._thread and self._thread.is_alive():
            self.logger.warning("TransactionStreamWorker is already running.")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="OANDA-TransactionsStream", daemon=True)
        self._thread.start()
        self.logger.info("TransactionStreamWorker started.")

    def stop(self, timeout: float = 5.0):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self.logger.info("TransactionStreamWorker stopped.")

    def _run(self):
        backoff = 1.0
        while not self._stop.is_set():
            since_id = None
            try:
                if self.last_transaction_id_getter:
                    since_id = self.last_transaction_id_getter() or None
            except Exception:
                since_id = None

            params = {}
            if since_id:
                params["sinceTransactionID"] = since_id

            try:
                r = transactions.TransactionsStream(accountID=self.account_id, params=params)
                for msg in self.client.request(r):
                    if self._stop.is_set():
                        break
                    if not msg:
                        continue
                    # Normalize: Transactions stream messages typically contain 'type'
                    if self.on_transaction:
                        try:
                            self.on_transaction(msg)
                        except Exception as cb_e:
                            self.logger.error(f"Transaction callback error: {cb_e}", exc_info=True)
                # Reset backoff after a successful loop
                backoff = 1.0
            except V20Error as e:
                self.logger.error(f"TransactionsStream V20Error: {e}")
            except Exception as e:
                self.logger.error(f"TransactionsStream unexpected error: {e}", exc_info=True)

            # Backoff and retry
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 60.0)

