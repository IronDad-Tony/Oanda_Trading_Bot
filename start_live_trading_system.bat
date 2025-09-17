@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

echo ==============================================
echo        Live Trading Control Center Launcher
echo ==============================================
echo API endpoint : http://localhost:8000/
echo UI dashboard : http://localhost:8000/
echo ----------------------------------------------
echo Press Ctrl+C in this window to stop the server.
echo ==============================================

echo Starting API server...
start "LiveTradingAPI" /MIN cmd /c python -c "from oanda_trading_bot.live_trading_system.api.server import main; main()"

timeout /t 2 >nul

echo Opening control center in default browser...
start "" http://localhost:8000/

echo Ready.
exit /b 0
