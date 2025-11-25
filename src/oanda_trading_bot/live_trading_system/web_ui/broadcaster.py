import asyncio
import logging
from typing import Dict, Any

from ..web_ui.main import broadcast as broadcast_to_clients
from ..core.oanda_client import OandaClient

logger = logging.getLogger(__name__)

class WebsocketBroadcaster:
    """
    Handles broadcasting of live trading data to WebSocket clients.
    """
    def __init__(self, client: OandaClient):
        self.client = client

    async def broadcast_account_update(self):
        """
        Fetches the latest account summary and broadcasts it.
        """
        try:
            account_summary = self.client.get_account_summary()
            if account_summary:
                # According to our UI plan, we need:
                # Equity, Balance, Margin Used, P/L
                payload = {
                    "type": "account_update",
                    "data": {
                        "equity": float(account_summary.get("NAV", 0.0)), # NAV is Net Asset Value, same as Equity
                        "balance": float(account_summary.get("balance", 0.0)),
                        "marginUsed": float(account_summary.get("marginUsed", 0.0)),
                        "unrealizedPL": float(account_summary.get("unrealizedPL", 0.0)),
                        "realizedPL": float(account_summary.get("pl", 0.0)) # 'pl' is profit/loss for the day
                    }
                }
                import json
                await broadcast_to_clients(json.dumps(payload))
                logger.debug("Broadcasted account update to clients.")
            else:
                logger.warning("Could not get account summary for broadcasting.")
        except Exception as e:
            logger.error(f"Error broadcasting account update: {e}", exc_info=True)

    async def broadcast_positions_update(self):
        """
        Fetches the current open positions and broadcasts them.
        """
        try:
            open_positions = self.client.get_open_positions()
            # The UI plan expects a list of position objects
            payload = {
                "type": "positions_update",
                "data": open_positions
            }
            import json
            await broadcast_to_clients(json.dumps(payload))
            logger.debug("Broadcasted positions update to clients.")
        except Exception as e:
            logger.error(f"Error broadcasting positions update: {e}", exc_info=True)

    async def broadcast_log(self, level: str, message: str):
        """
        Broadcasts a log message.
        """
        try:
            payload = {
                "type": "log_stream",
                "data": {
                    "level": level.upper(),
                    "message": message
                }
            }
            import json
            await broadcast_to_clients(json.dumps(payload))
        except Exception as e:
            # Avoid a broadcast loop if logging itself fails
            print(f"CRITICAL: Could not broadcast log message. Error: {e}")

# Global broadcaster instance to be initialized
broadcaster_instance: WebsocketBroadcaster = None

def initialize_broadcaster(client: OandaClient):
    """Initializes the global broadcaster instance."""
    global broadcaster_instance
    if broadcaster_instance is None:
        broadcaster_instance = WebsocketBroadcaster(client)
        logger.info("WebsocketBroadcaster initialized.")

def get_broadcaster() -> WebsocketBroadcaster:
    """Returns the global broadcaster instance."""
    return broadcaster_instance
