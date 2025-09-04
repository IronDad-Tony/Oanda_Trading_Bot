@echo off
chcp 65001 >nul
echo ========================================
echo      OANDA Live Trading System (LIVE)
echo ========================================
echo.
echo Starts the live trading system and opens the UI in your browser.
echo - UI: http://localhost:8501
echo.
echo Press Ctrl+C in the terminal to stop the service.
echo ========================================
echo.

cd /d "%~dp0"

REM Minimal check: Python available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not found. Please install Python and add it to your PATH.
    pause
    exit /b 1
)

REM Force OANDA environment to LIVE for this session (do not edit .env)
set OANDA_ENVIRONMENT=live

REM Ensure Python can import the 'src' layout package
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

echo Starting the system...
echo.

REM Launch the main script which in turn starts the Streamlit UI
echo Launching Live Trading System...
echo Main application file: src\oanda_trading_bot\live_trading_system\main.py
echo.

REM Open browser to the UI (non-blocking)
start "" "http://localhost:8501"

REM Run package entry (blocking in this window)
python -m oanda_trading_bot.live_trading_system.main

echo.
echo System has been stopped.
pause
