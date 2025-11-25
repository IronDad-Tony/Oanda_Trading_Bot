@echo off
REM Quick Start Script for OANDA Trading Bot v2.0
REM 快速啟動腳本

echo ============================================================
echo OANDA AI Trading Bot v2.0 - Quick Start
echo ============================================================
echo.

:menu
echo Please select an option:
echo.
echo [1] Start API Server (Port 8000)
echo [2] Start Training System (Streamlit)
echo [3] Start Live Trading System (Streamlit)  
echo [4] Run Smoke Test
echo [5] Train Mini Model (5000 steps)
echo [6] View API Documentation
echo [7] Check System Status
echo [0] Exit
echo.

set /p choice="Enter your choice (0-7): "

if "%choice%"=="0" goto :end
if "%choice%"=="1" goto :api
if "%choice%"=="2" goto :training
if "%choice%"=="3" goto :live
if "%choice%"=="4" goto :smoke_test
if "%choice%"=="5" goto :mini_train
if "%choice%"=="6" goto :api_docs
if "%choice%"=="7" goto :status

echo Invalid choice. Please try again.
echo.
goto :menu

:api
echo.
echo Starting API Server...
echo Access API docs at: http://localhost:8000/docs
echo Press Ctrl+C to stop
echo.
cd src\oanda_trading_bot\api
python main.py
goto :end

:training
echo.
echo Starting Training System UI...
echo Access at: http://localhost:8501
echo.
streamlit run src\oanda_trading_bot\training_system\app.py
goto :end

:live
echo.
echo Starting Live Trading System UI...
echo Access at: http://localhost:8502
echo.
cd src\oanda_trading_bot\live_trading_system\ui
streamlit run app.py --server.port 8502
goto :end

:smoke_test
echo.
echo Running Smoke Test...
echo.
python scripts\testing\complete_smoke_test.py
echo.
pause
goto :menu

:mini_train
echo.
echo Training Mini Model (5000 steps)...
echo This will take a few minutes...
echo.
python scripts\training\train_quick_sac.py
echo.
pause
goto :menu

:api_docs
echo.
echo Opening API Documentation...
start http://localhost:8000/docs
echo.
echo If API server is not running, please select option [1] first.
pause
goto :menu

:status
echo.
echo ============================================================
echo System Status Check
echo ============================================================
echo.

REM Check if API server is running
echo Checking API Server (Port 8000)...
powershell -Command "$response = try { Invoke-WebRequest -Uri 'http://localhost:8000/api/health' -TimeoutSec 2 } catch { $null }; if ($response.StatusCode -eq 200) { Write-Host '✓ API Server: RUNNING' -ForegroundColor Green } else { Write-Host '✗ API Server: NOT RUNNING' -ForegroundColor Red }"

echo.
echo Checking Python Environment...
python --version
echo.

echo Checking Required Packages...
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>nul || echo ✗ PyTorch not found
python -c "import stable_baselines3; print('✓ Stable-Baselines3')" 2>nul || echo ✗ Stable-Baselines3 not found
python -c "import fastapi; print('✓ FastAPI')" 2>nul || echo ✗ FastAPI not found
python -c "import streamlit; print('✓ Streamlit')" 2>nul || echo ✗ Streamlit not found

echo.
echo Checking Model Registry...
if exist "model_registry\index.json" (
    echo ✓ Model Registry initialized
) else (
    echo ✗ Model Registry not found
)

echo.
echo Checking Upgrades...
if exist "src\oanda_trading_bot\common\model_registry.py" (
    echo ✓ Model Registry System
) else (
    echo ✗ Model Registry System not found
)

if exist "src\oanda_trading_bot\training_system\environment\enhanced_simulation.py" (
    echo ✓ Enhanced Simulation
) else (
    echo ✗ Enhanced Simulation not found
)

if exist "src\oanda_trading_bot\training_system\environment\symbol_reward_attribution.py" (
    echo ✓ Symbol Reward Attribution
) else (
    echo ✗ Symbol Reward Attribution not found
)

if exist "src\oanda_trading_bot\api\main.py" (
    echo ✓ Unified API Backend
) else (
    echo ✗ Unified API Backend not found
)

echo.
echo ============================================================
pause
goto :menu

:end
echo.
echo Thank you for using OANDA AI Trading Bot!
echo.
pause
