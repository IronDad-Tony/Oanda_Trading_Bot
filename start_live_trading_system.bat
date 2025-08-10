@echo off
chcp 65001 >nul
echo ========================================
echo      OANDA Live Trading System
echo ========================================
echo.
echo This script will start the live trading system and its UI.
echo - Streamlit UI Interface will be available at: http://localhost:8501
echo.
echo Press Ctrl+C in the terminal to stop the service.
echo ========================================
echo.

cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not found. Please install Python and add it to your PATH.
    pause
    exit /b 1
)

REM Define the new path to requirements.txt
set "REQUIREMENTS_PATH=src\oanda_trading_bot\live_trading_system\requirements.txt"

REM Check if requirements.txt exists at the new location
if not exist "%REQUIREMENTS_PATH%" (
    echo WARNING: "%REQUIREMENTS_PATH%" not found.
    echo Dependency installation will be skipped. The application might fail.
) else (
    echo Checking and installing required packages from %REQUIREMENTS_PATH%...
    pip install -r "%REQUIREMENTS_PATH%"
)

echo.
echo All checks passed. Starting the system...
echo.

REM Wait a moment before launching
timeout /t 2 /nobreak >nul

REM Launch the main script which in turn starts the Streamlit UI
echo Launching Live Trading System...
echo Main application file: src\oanda_trading_bot\live_trading_system\main.py
echo.

python src/oanda_trading_bot/live_trading_system/main.py

echo.
echo System has been stopped.
pause
