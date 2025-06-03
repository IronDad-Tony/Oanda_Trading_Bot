@echo off
chcp 65001 >nul
echo ========================================
echo    OANDA AI 交易系統 - 完整監控系統
echo ========================================
echo.
echo 正在啟動完整監控系統...
echo - Streamlit UI 介面: http://localhost:8501
echo - TensorBoard 監控: http://localhost:6006
echo.
echo 功能說明:
echo - Streamlit: 訓練控制、實時監控、系統資源
echo - TensorBoard: 詳細訓練指標、模型圖表
echo - 自動啟動兩個服務並在瀏覽器中打開
echo.
echo 按 Ctrl+C 停止所有服務
echo ========================================
echo.

cd /d "%~dp0"

REM 設置 Python 路徑環境變量
set PYTHONPATH=%cd%;%PYTHONPATH%
echo 已設置 PYTHONPATH: %PYTHONPATH%

REM 設置 PyTorch 兼容性環境變量
set "TORCH_CPP_LOG_LEVEL=ERROR"
set "TORCH_DISTRIBUTED_DEBUG=OFF"
set "TORCH_SHOW_CPP_STACKTRACES=0"
set "PYTORCH_DISABLE_PER_OP_PROFILING=1"
echo 已設置 PyTorch 兼容性環境變量
echo.

REM 檢查 Python 是否安裝
python --version >nul 2>&1
if errorlevel 1 (
    echo 錯誤: 未找到 Python，請先安裝 Python
    pause
    exit /b 1
)

REM 檢查必要套件是否安裝
echo 檢查必要套件...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo 正在安裝 Streamlit...
    pip install streamlit
)

python -c "import tensorboard" >nul 2>&1
if errorlevel 1 (
    echo 正在安裝 TensorBoard...
    pip install tensorboard
)

REM 確保日誌目錄存在
if not exist "logs" mkdir logs
if not exist "logs\tensorboard" mkdir logs\tensorboard

echo.
echo 正在啟動服務...
echo.

REM 啟動 TensorBoard (在背景執行) - 使用統一的目錄
echo 啟動 TensorBoard...
start "TensorBoard" cmd /c "tensorboard --logdir=logs\tensorboard --port=6006 --host=localhost"

REM 等待 TensorBoard 啟動
timeout /t 3 /nobreak >nul

REM 啟動 Streamlit (使用修復完成的版本)
echo 啟動 Streamlit UI...
echo 使用檔案: streamlit_app_complete.py
echo.

REM 等待一下再啟動瀏覽器
timeout /t 2 /nobreak >nul

REM 在背景啟動瀏覽器（只開 TensorBoard，Streamlit 讓其自動開啟即可）
start "" "http://localhost:6006"

REM 啟動 Streamlit (前景執行，這樣可以看到日誌)
python streamlit_launcher.py

echo.
echo 系統已停止
pause