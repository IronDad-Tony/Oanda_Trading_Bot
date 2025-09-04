import csv
import os
import threading
import time
from typing import Optional, Dict, Any


class ExecutionMetricsLogger:
    """Append-only CSV logger for execution timing and slippage calibration."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._lock = threading.Lock()
        # Ensure header exists
        if not os.path.exists(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    'ts_ms', 'instrument', 'client_id', 'side', 'units',
                    'request_ts_ms', 'response_ts_ms', 'server_time',
                    'ref_price', 'price_bound', 'fill_price', 'status', 'reason'
                ])

    def log(self, row: Dict[str, Any]):
        with self._lock:
            with open(self.log_path, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    int(time.time() * 1000),
                    row.get('instrument'),
                    row.get('client_id'),
                    row.get('side'),
                    row.get('units'),
                    row.get('request_ts_ms'),
                    row.get('response_ts_ms'),
                    row.get('server_time'),
                    row.get('ref_price'),
                    row.get('price_bound'),
                    row.get('fill_price'),
                    row.get('status'),
                    row.get('reason'),
                ])

