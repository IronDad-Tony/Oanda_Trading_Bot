@echo off
chcp 65001 >nul
echo ========================================
echo      OANDA Live Trading System
echo ========================================
echo.
echo Starts the live trading system UI in your browser.
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

REM Use OANDA_ENVIRONMENT from .env (no forced override here)

REM Ensure Python can import the 'src' layout package
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

echo Starting the system...
echo.

REM Launch the new Streamlit UI directly
echo Launching Live Trading UI...
echo UI application file: src\oanda_trading_bot\live_ui\app.py
echo.

REM Run Streamlit (blocking in this window) on a fixed port
python -m streamlit run src\oanda_trading_bot\live_ui\app.py --server.port 8501

echo.
echo System has been stopped.
pause
