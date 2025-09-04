@echo off
chcp 65001 >nul
echo ========================================
echo      OANDA Live Trading UI (Minimal)
echo ========================================
echo This launcher does not check or install any dependencies.
echo If dependencies are missing, the app may fail to start.
echo ----------------------------------------

cd /d "%~dp0"

REM Launch the Python UI bootstrap which starts Streamlit
python src\oanda_trading_bot\live_trading_system\main.py

echo.
echo UI process exited.
pause

