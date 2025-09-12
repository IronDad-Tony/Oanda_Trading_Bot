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

REM Wait for API to become ready before opening the browser (max ~60s)
set "READY_URL=http://localhost:8000/ready"
for /L %%i in (1,1,120) do (
  powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri '%READY_URL%' -TimeoutSec 1; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1 && goto :READY
  timeout /t 1 >nul
)

:READY
REM Open UI in default browser once server responds OK
start "" http://localhost:8000/

exit /b 0
