@echo off
chcp 65001 >nul
echo ========================================
echo      OANDA Live Trading System UI
echo ========================================
echo.
echo This script will start the live trading user interface.
echo - Streamlit UI Interface: http://localhost:8501
echo.
echo Press Ctrl+C to stop the service.
echo ========================================
echo.

cd /d "%~dp0"

REM Set the PYTHONPATH environment variable to include the project root
set "PYTHONPATH=%cd%;%PYTHONPATH%"
echo PYTHONPATH set to: %PYTHONPATH%
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not found. Please install Python and add it to your PATH.
    pause
    exit /b 1
)

REM Check if requirements.txt exists
if not exist "live_trading_system\requirements.txt" (
    echo WARNING: "live_trading_system\requirements.txt" not found.
    echo Dependency installation will be skipped. The application might fail.
) else (
    echo Checking and installing required packages from requirements.txt...
    pip install -r "live_trading_system\requirements.txt"
)

echo.
echo All checks passed. Starting the system...
echo.

REM Wait a moment before launching
timeout /t 2 /nobreak >nul

REM Launch the Streamlit UI
echo Launching Streamlit UI...
echo Main application file: live_trading_system\ui\app.py
echo.

REM Manually open the default browser to avoid issues with Streamlit's auto-open
echo Launching browser at http://localhost:8501
start "" "http://localhost:8501"

REM Start Streamlit in headless mode (won't try to open a browser itself)
echo Starting Streamlit server...
streamlit run live_trading_system/ui/app.py --server.headless=true

echo.
echo System has been stopped.
pause
