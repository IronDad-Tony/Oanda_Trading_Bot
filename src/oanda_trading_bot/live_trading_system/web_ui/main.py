from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
import threading
import json
from typing import Dict, Any, Optional

# Add project root to be able to import the live_trading_system
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from src.oanda_trading_bot.live_trading_system.main import initialize_system, async_trading_loop

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow CORS for the frontend development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Global variable to hold the initialized system components
trading_components: Optional[Dict[str, Any]] = None

@app.on_event("startup")
def startup_event():
    """
    On startup, initialize the trading system and run the trading loop in a background thread.
    """
    global trading_components
    logger.info("FastAPI server starting up. Initializing trading system...")
    
    trading_components = initialize_system()
    
    if trading_components is None:
        logger.critical("System initialization failed. Trading loop will not start.")
        return
        
    system_state = trading_components.get("system_state")
    if not system_state:
        logger.critical("SystemState component not found. Trading loop will not start.")
        return

    # Set the system to running
    system_state.is_running = True
    
    # Manually set some instruments to trade for testing purposes
    system_state.select_instrument("EUR_USD")
    system_state.select_instrument("GBP_USD")
    logger.info(f"Default instruments selected: {system_state.get_selected_instruments()}")

    def run_loop():
        asyncio.run(async_trading_loop(trading_components))

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    logger.info("Asynchronous trading loop started in a background thread.")


# --- API Endpoints for Control ---

@app.post("/api/v1/actions/pause")
async def pause_trading():
    if not trading_components or 'system_state' not in trading_components:
        raise HTTPException(status_code=503, detail="System not initialized")
    system_state = trading_components['system_state']
    system_state.is_paused = True
    logger.warning("Trading has been PAUSED via API request.")
    return {"status": "ok", "message": "New trades have been paused."}

@app.post("/api/v1/actions/resume")
async def resume_trading():
    if not trading_components or 'system_state' not in trading_components:
        raise HTTPException(status_code=503, detail="System not initialized")
    system_state = trading_components['system_state']
    system_state.is_paused = False
    logger.warning("Trading has been RESUMED via API request.")
    return {"status": "ok", "message": "Trading has been resumed."}

@app.post("/api/v1/actions/emergency_stop")
async def emergency_stop():
    if not trading_components or 'system_state' not in trading_components or 'order_manager' not in trading_components:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    system_state = trading_components['system_state']
    order_manager = trading_components['order_manager']
    
    logger.critical("EMERGENCY STOP initiated via API request.")
    
    # 1. Stop the bot from running further cycles
    system_state.is_running = False
    logger.critical("Trading loop has been stopped.")

    # 2. Liquidate all positions
    try:
        # This method needs to be created in OrderManager
        result = order_manager.close_all_positions()
        logger.critical(f"Liquidation result: {result}")
        return {"status": "ok", "message": "Emergency stop initiated. All positions are being liquidated.", "details": result}
    except Exception as e:
        logger.critical(f"Error during emergency liquidation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during liquidation: {e}")

@app.delete("/api/v1/positions/{instrument}")
async def close_position(instrument: str):
    if not trading_components or 'order_manager' not in trading_components:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    order_manager = trading_components['order_manager']
    logger.info(f"Received API request to close position for {instrument}.")
    
    try:
        result = order_manager.close_specific_position(instrument)
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result)
        return result
    except Exception as e:
        logger.error(f"Error closing position for {instrument}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error closing position: {e}")


# --- WebSocket and Status Endpoints ---

@app.get("/")
async def get():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
        <head>
            <title>OANDA Trading Bot - Live Dashboard</title>
            <style>
                body { font-family: monospace; background-color: #111; color: #eee; }
                h1 { color: #0f0; }
                #messages { list-style-type: none; padding: 0; }
                #messages li { background-color: #222; margin-bottom: 5px; padding: 8px; border-left: 3px solid #0f0; }
                #messages li.account_update { border-left-color: #0af; }
                #messages li.positions_update { border-left-color: #fa0; }
            </style>
        </head>
        <body>
            <h1>Live Data Stream</h1>
            <ul id='messages'>
            </ul>
            <script>
                var ws = new WebSocket("ws://localhost:8000/ws");
                ws.onmessage = function(event) {
                    var messages = document.getElementById('messages');
                    var message = document.createElement('li');
                    var data = JSON.parse(event.data);
                    
                    if (data.type) {
                        message.classList.add(data.type);
                    }
                    
                    var content = document.createTextNode(JSON.stringify(data, null, 2));
                    var pre = document.createElement('pre');
                    pre.appendChild(content);
                    message.appendChild(pre);
                    messages.insertBefore(message, messages.firstChild);
                };
            </script>
        </body>
    </html>
    """)

@app.get("/api/v1/status")
async def get_status():
    if not trading_components:
        return {"status": "initializing", "trading_loop_running": False, "is_paused": None}
    
    system_state = trading_components['system_state']
    return {
        "status": "ok", 
        "trading_loop_running": system_state.is_running,
        "is_paused": getattr(system_state, 'is_paused', False)
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"New client connected. Total clients: {len(active_connections)}")
    try:
        await websocket.send_text(json.dumps({"type": "system", "data": "Welcome to the OANDA Trading Bot WebSocket!"}))
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received message from client: {data}")
            await websocket.send_text(json.dumps({"type": "echo", "data": data}))
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(active_connections)}")
    except Exception as e:
        logger.error(f"An error occurred in WebSocket: {e}", exc_info=True)
        if websocket in active_connections:
            active_connections.remove(websocket)

active_connections: list[WebSocket] = []
async def broadcast(message: str):
    for connection in active_connections[:]:
        try:
            await connection.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send message to a client: {e}")
            if connection in active_connections:
                active_connections.remove(connection)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for Web UI backend...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
