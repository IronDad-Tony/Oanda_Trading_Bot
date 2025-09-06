@echo off
chcp 65001 >nul
echo ========================================
echo      OANDA AI Training Model
echo ========================================
echo.
echo This script will start the AI model training interface.
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

echo All checks passed. Starting the training UI...
echo.

REM Wait a moment before launching
timeout /t 2 /nobreak >nul

REM Launch the Streamlit UI for the training system
echo Launching Training System UI...
echo Main application file: src\oanda_trading_bot\training_system\app.py
echo.

REM Ensure Python can import the 'src' package layout
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

streamlit run src/oanda_trading_bot/training_system/app.py

echo.
echo System has been stopped.
pause
