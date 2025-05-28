@echo off
chcp 65001 >nul
title OANDA AI交易系統 - 系統診斷

echo.
echo ========================================
echo    🔧 OANDA AI交易系統 - 系統診斷
echo ========================================
echo.

REM 獲取批次文件所在目錄
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

REM 設置 PYTHONPATH
set PYTHONPATH=%cd%;%PYTHONPATH%
echo 已設置 PYTHONPATH: %PYTHONPATH%

echo 📁 項目目錄: %PROJECT_DIR%
echo.

echo 🔍 開始系統診斷...
echo.

REM 1. 檢查Python環境
echo ========================================
echo 🐍 Python環境檢查
echo ========================================
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python未安裝或未加入PATH
    echo 💡 請從 https://www.python.org/downloads/ 下載並安裝Python 3.8+
) else (
    echo ✅ Python已安裝: 
    python --version
    echo 📍 Python路徑: 
    where python
)
echo.

REM 2. 檢查pip
echo ========================================
echo 📦 pip包管理器檢查
echo ========================================
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip未找到
    echo 💡 請重新安裝Python並確保包含pip
) else (
    echo ✅ pip已安裝: 
    pip --version
)
echo.

REM 3. 檢查核心依賴
echo ========================================
echo 🔧 核心依賴檢查
echo ========================================

echo 🔍 檢查PyTorch...
python -c "import torch; print('✅ PyTorch版本:', torch.__version__)" 2>nul
if errorlevel 1 (
    echo ❌ PyTorch未安裝
    echo 💡 安裝命令: pip install torch
)

echo 🔍 檢查Streamlit...
python -c "import streamlit; print('✅ Streamlit版本:', streamlit.__version__)" 2>nul
if errorlevel 1 (
    echo ❌ Streamlit未安裝
    echo 💡 安裝命令: pip install streamlit
)

echo 🔍 檢查Pandas...
python -c "import pandas; print('✅ Pandas版本:', pandas.__version__)" 2>nul
if errorlevel 1 (
    echo ❌ Pandas未安裝
    echo 💡 安裝命令: pip install pandas
)

echo 🔍 檢查NumPy...
python -c "import numpy; print('✅ NumPy版本:', numpy.__version__)" 2>nul
if errorlevel 1 (
    echo ❌ NumPy未安裝
    echo 💡 安裝命令: pip install numpy
)

echo 🔍 檢查其他依賴...
python -c "import plotly, requests, python_dotenv; print('✅ 其他依賴正常')" 2>nul
if errorlevel 1 (
    echo ⚠️  部分依賴缺失
    echo 💡 安裝命令: pip install plotly requests python-dotenv
)

echo.

REM 4. 檢查GPU支援
echo ========================================
echo 🎮 GPU支援檢查
echo ========================================
python -c "import torch; print('GPU可用:', torch.cuda.is_available()); print('GPU數量:', torch.cuda.device_count()); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('將使用CPU進行訓練')" 2>nul
echo.

REM 5. 檢查系統文件
echo ========================================
echo 📁 系統文件檢查
echo ========================================

if exist "start_training_english.py" (
    echo ✅ 訓練腳本: start_training_english.py
) else (
    echo ❌ 缺少: start_training_english.py
)

if exist "streamlit_app_complete.py" (
    echo ✅ 監控應用: streamlit_app_complete.py
) else (
    echo ❌ 缺少: streamlit_app_complete.py
)

if exist "requirements.txt" (
    echo ✅ 依賴清單: requirements.txt
) else (
    echo ❌ 缺少: requirements.txt
)

if exist ".env" (
    echo ✅ 配置文件: .env
) else (
    echo ⚠️  配置文件: .env (未找到，但可選)
)

if exist "src\" (
    echo ✅ 源碼目錄: src/
) else (
    echo ❌ 缺少: src/ 目錄
)

echo.

REM 6. 檢查目錄結構
echo ========================================
echo 📂 目錄結構檢查
echo ========================================

for %%d in (src logs data weights) do (
    if exist "%%d\" (
        echo ✅ 目錄存在: %%d/
    ) else (
        echo ⚠️  目錄缺失: %%d/ (將自動創建)
        mkdir "%%d" 2>nul
    )
)

echo.

REM 7. 檢查磁碟空間
echo ========================================
echo 💾 磁碟空間檢查
echo ========================================
for /f "tokens=3" %%a in ('dir /-c "%PROJECT_DIR%" ^| find "bytes free"') do set free_space=%%a
echo 💾 可用空間: %free_space% bytes
echo 💡 建議至少保留5GB空間用於訓練數據和模型

echo.

REM 8. 網絡連接檢查
echo ========================================
echo 🌐 網絡連接檢查
echo ========================================
ping -n 1 8.8.8.8 >nul 2>&1
if errorlevel 1 (
    echo ❌ 網絡連接異常
    echo 💡 請檢查網絡設置，某些功能可能需要網絡連接
) else (
    echo ✅ 網絡連接正常
)

echo.

REM 9. 端口檢查
echo ========================================
echo 🔌 端口可用性檢查
echo ========================================
netstat -an | find "8501" >nul 2>&1
if errorlevel 1 (
    echo ✅ 端口8501可用 (Streamlit默認端口)
) else (
    echo ⚠️  端口8501被占用，可能需要使用其他端口
)

echo.

REM 10. 運行整合測試
echo ========================================
echo 🧪 整合測試
echo ========================================
if exist "test_imports.py" (
    echo 🔍 運行模組導入測試...
    python test_imports.py
    if errorlevel 1 (
        echo ❌ 模組導入測試失敗
        echo 💡 請檢查上述診斷結果並修復問題
    ) else (
        echo ✅ 模組導入測試通過
    )
) else (
    echo ⚠️  測試文件不存在，跳過測試
)

echo.

REM 診斷總結
echo ========================================
echo 📋 診斷總結
echo ========================================
echo.
echo 💡 常見問題解決方案:
echo.
echo 🔧 如果依賴缺失:
echo    pip install -r requirements.txt
echo.
echo 🔧 如果GPU不可用但有NVIDIA顯卡:
echo    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo.
echo 🔧 如果端口被占用:
echo    streamlit run streamlit_app_complete.py --server.port 8502
echo.
echo 🔧 如果訓練失敗:
echo    1. 檢查磁碟空間是否充足
echo    2. 確認所有依賴已正確安裝
echo    3. 查看 training.log 日誌文件
echo.
echo 🔧 如果需要重新安裝所有依賴:
echo    pip uninstall -r requirements.txt -y
echo    pip install -r requirements.txt
echo.

echo ========================================
echo 🔧 系統診斷完成
echo ========================================
echo.
pause