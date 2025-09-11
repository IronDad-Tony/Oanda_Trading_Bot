@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

echo ========================================
echo      OANDA Live Trading System
echo ========================================
echo - UI: http://localhost:8000/
echo ========================================

REM Start FastAPI server in a new minimized window
start "LiveTradingAPI" /MIN cmd /c python -c "from oanda_trading_bot.live_trading_system.api.server import main; main()"

REM Give server a moment to start
timeout /t 2 >nul

REM Open UI in default browser
start "" http://localhost:8000/

exit /b 0
