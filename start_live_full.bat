@echo off
chcp 65001 >nul
echo ========================================
echo   Live Trading Backend + Next UI
echo ========================================
cd /d "%~dp0"

REM Set live environment for this session (does not modify .env)
set OANDA_ENVIRONMENT=live
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

REM Start backend (FastAPI via uvicorn) in new window
start "Live Backend" cmd /k python -m oanda_trading_bot.live_trading_system.api.server

REM Start Next.js UI (dev) in new window
start "Next UI" cmd /k ui\start_next_ui.bat

REM Open UI in browser
start "" "http://localhost:3000"

echo Launched backend and UI. Press any key to close this launcher.
pause >nul

