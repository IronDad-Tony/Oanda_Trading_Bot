@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

echo ==========================================================
echo   Live Trading Control Center - UI Launcher
echo ==========================================================
echo API service  : http://localhost:8000/
echo UI dashboard : http://localhost:8000/
echo ----------------------------------------------------------
echo This launcher keeps your existing environment intact.
echo Press Ctrl+C in this window to stop the API service.
echo ==========================================================

echo Starting live trading API server in the background...
start "LiveTradingAPI" /MIN cmd /c python -c "from oanda_trading_bot.live_trading_system.api.server import main; main()"

timeout /t 2 >nul

echo Opening the redesigned trading dashboard...
start "" http://localhost:8000/

echo Ready. Happy trading!
exit /b 0
