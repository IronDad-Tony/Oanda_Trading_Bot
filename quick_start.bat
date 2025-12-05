@echo off
REM Quick Start Script for OANDA Trading Bot v2.0 (Fixed)
REM Updated: 2025-11-25

echo ============================================================
echo OANDA AI Trading Bot v2.0 - Quick Start
echo ============================================================
echo.

:menu
echo Please select an option:
echo.
echo [1] Start Unified API Backend (Port 8000)
echo [2] Start Training System UI (Streamlit - Port 8501)
echo [3] Start Live Trading Web UI (FastAPI - Port 8003)
echo [4] Start React Frontend (Port 3000) [DEV]
echo [5] Run Smoke Test
echo [6] Train Mini Model (5000 steps)
echo [7] Check System Status
echo [0] Exit
echo.

set /p choice="Enter your choice (0-7): "

if "%choice%"=="0" goto :end
if "%choice%"=="1" goto :api
if "%choice%"=="2" goto :training
if "%choice%"=="3" goto :live
if "%choice%"=="4" goto :frontend
if "%choice%"=="5" goto :smoke_test
if "%choice%"=="6" goto :mini_train
if "%choice%"=="7" goto :status

echo Invalid choice. Please try again.
echo.
goto :menu

:api
echo.
echo ============================================================
echo Starting Unified API Backend...
echo ============================================================
echo.
echo Access API docs at: http://localhost:8000/docs
echo WebSocket: ws://localhost:8000/ws
echo            ws://localhost:8000/ws/training
echo            ws://localhost:8000/ws/live
echo.
echo Press Ctrl+C to stop
echo.
set PYTHONPATH=%CD%\src
python src\oanda_trading_bot\api\main.py
goto :end

:training
echo.
echo ============================================================
echo Starting Training System UI (Streamlit)...
echo ============================================================
echo.
echo Access at: http://localhost:8501
echo.
echo Press Ctrl+C to stop
echo.
streamlit run src\oanda_trading_bot\training_system\app.py
goto :end

:live
echo.
echo ============================================================
echo Starting Live Trading Web UI (FastAPI)...
echo ============================================================
echo.
echo Access at: http://localhost:8003
echo.
echo Note: This UI runs the actual trading system!
echo       Make sure your .env is configured correctly.
echo.
echo Press Ctrl+C to stop
echo.
REM Run on port 8003 to avoid conflict with unified API
cd src\oanda_trading_bot\live_trading_system\web_ui
uvicorn main:app --port 8003 --reload
goto :end

:frontend
echo.
echo ============================================================
echo Starting React Frontend (Development Mode)...
echo ============================================================
echo.
echo Access at: http://localhost:3000
echo.
echo Note: Make sure you have run 'npm install' first!
echo       This connects to the Unified API at port 8000.
echo.
echo Press Ctrl+C to stop
echo.
cd frontend
if not exist "node_modules" (
    echo Installing dependencies...
    call npm install
)
call npm start
goto :end

:smoke_test
echo.
echo ============================================================
echo Running Smoke Test...
echo ============================================================
echo.
python scripts\testing\complete_smoke_test.py
echo.
pause
goto :menu

:mini_train
echo.
echo ============================================================
echo Training Mini Model (5000 steps)...
echo ============================================================
echo.
echo This will take a few minutes...
echo.
python scripts\training\train_quick_sac.py
echo.
pause
goto :menu

:status
echo.
echo ============================================================
echo System Status Check
echo ============================================================
echo.

REM Check Python
echo Checking Python Environment...
python --version
echo.

REM Check Required Packages
echo Checking Required Packages...
python -c "import torch; print(f'[OK] PyTorch {torch.__version__}')  " 2>nul || echo [X] PyTorch not found
python -c "import stable_baselines3; print('[OK] Stable-Baselines3')" 2>nul || echo [X] Stable-Baselines3 not found
python -c "import fastapi; print('[OK] FastAPI')" 2>nul || echo [X] FastAPI not found
python -c "import streamlit; print('[OK] Streamlit')" 2>nul || echo [X] Streamlit not found
python -c "import uvicorn; print('[OK] Uvicorn')" 2>nul || echo [X] Uvicorn not found
echo.

REM Check API server
echo Checking API Servers...
powershell -Command "$response = try { Invoke-WebRequest -Uri 'http://localhost:8000/api/health' -TimeoutSec 1 } catch { $null }; if ($response.StatusCode -eq 200) { Write-Host '[OK] Unified API (8000): RUNNING' -ForegroundColor Green } else { Write-Host '[X] Unified API (8000): NOT RUNNING' -ForegroundColor Yellow }"
powershell -Command "$response = try { Invoke-WebRequest -Uri 'http://localhost:8501' -TimeoutSec 1 } catch { $null }; if ($response) { Write-Host '[OK] Training UI (8501): RUNNING' -ForegroundColor Green } else { Write-Host '[X] Training UI (8501): NOT RUNNING' -ForegroundColor Yellow }"
powershell -Command "$response = try { Invoke-WebRequest -Uri 'http://localhost:8003' -TimeoutSec 1 } catch { $null }; if ($response) { Write-Host '[OK] Live Web UI (8003): RUNNING' -ForegroundColor Green } else { Write-Host '[X] Live Web UI (8003): NOT RUNNING' -ForegroundColor Yellow }"
powershell -Command "$response = try { Invoke-WebRequest -Uri 'http://localhost:3000' -TimeoutSec 1 } catch { $null }; if ($response) { Write-Host '[OK] React Frontend (3000): RUNNING' -ForegroundColor Green } else { Write-Host '[X] React Frontend (3000): NOT RUNNING' -ForegroundColor Yellow }"
echo.

REM Check Files
echo Checking Core Files...
if exist "src\oanda_trading_bot\common\model_registry.py" (
    echo [OK] Model Registry System
) else (
    echo [X] Model Registry System not found
)

if exist "src\oanda_trading_bot\training_system\environment\enhanced_simulation.py" (
    echo [OK] Enhanced Simulation
) else (
    echo [X] Enhanced Simulation not found
)

if exist "src\oanda_trading_bot\training_system\environment\symbol_reward_attribution.py" (
    echo [OK] Symbol Reward Attribution
) else (
    echo [X] Symbol Reward Attribution not found
)

if exist "src\oanda_trading_bot\api\main.py" (
    echo [OK] Unified API Backend
) else (
    echo [X] Unified API Backend not found
)

if exist "src\oanda_trading_bot\training_system\app.py" (
    echo [OK] Training System UI
) else (
    echo [X] Training System UI not found
)

if exist "src\oanda_trading_bot\live_trading_system\web_ui\main.py" (
    echo [OK] Live Trading Web UI
) else (
    echo [X] Live Trading Web UI not found
)

if exist "frontend\package.json" (
    echo [OK] React Frontend
) else (
    echo [X] React Frontend not found
)

echo.

REM Check Model Registry
echo Checking Model Registry...
if exist "model_registry\index.json" (
    echo [OK] Model Registry initialized
) else (
    echo [X] Model Registry index not found
)

echo.
echo ============================================================
echo.
echo UI Systems Available:
echo [1] Unified API Backend - Port 8000 (FastAPI)
echo [2] Training System UI - Port 8501 (Streamlit)
echo [3] Live Trading Web UI - Port 8003 (FastAPI)
echo [4] React Frontend - Port 3000 (Under Development)
echo.
echo Technology Stack:
echo - Training UI: Python + Streamlit
echo - Live Web UI: Python + FastAPI + vanilla HTML/JS
echo - React Frontend: TypeScript + Material-UI (NOT Next.js/Tailwind)
echo - API Backend: Python + FastAPI + WebSocket
echo.
echo ============================================================
pause
goto :menu

:end
echo.
echo Thank you for using OANDA AI Trading Bot!
echo.
pause
