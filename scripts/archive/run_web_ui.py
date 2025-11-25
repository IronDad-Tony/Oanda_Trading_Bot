import uvicorn
import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.oanda_trading_bot.live_trading_system.web_ui.main import app

if __name__ == "__main__":
    print("Starting Live Trading Web UI server...")
    print("Access the test page at http://localhost:8000")
    print("API status endpoint at http://localhost:8000/api/v1/status")
    uvicorn.run(app, host="0.0.0.0", port=8000)
