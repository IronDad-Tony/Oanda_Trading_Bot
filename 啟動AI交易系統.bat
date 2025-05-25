@echo off
chcp 65001 >nul
title OANDA AI交易系統啟動器

echo.
echo ========================================
echo    🚀 OANDA AI交易系統啟動器
echo ========================================
echo.

REM 獲取批次文件所在目錄
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

echo 📁 項目目錄: %PROJECT_DIR%
echo.

REM 檢查Python是否安裝
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 錯誤: 未找到Python，請先安裝Python
    echo 📥 下載地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python已安裝
python --version

REM 檢查streamlit是否安裝
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ⚠️  Streamlit未安裝，正在安裝依賴...
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ 依賴安裝失敗，請檢查網絡連接
        pause
        exit /b 1
    )
)

echo ✅ 依賴檢查完成
echo.

REM 檢查.env文件
if not exist ".env" (
    echo ⚠️  警告: 未找到.env文件
    echo 📝 請確保已設置OANDA API密鑰
    echo.
    echo 創建.env文件範例:
    echo OANDA_API_KEY=your_api_key_here
    echo OANDA_ACCOUNT_ID=your_account_id_here
    echo.
    set /p "continue=是否繼續啟動？(y/N): "
    if /i not "%continue%"=="y" (
        echo 已取消啟動
        pause
        exit /b 0
    )
)

echo 🚀 正在啟動Streamlit應用...
echo.
echo 📱 應用將在瀏覽器中自動打開
echo 🌐 如果沒有自動打開，請手動訪問: http://localhost:8501
echo.
echo ⏹️  要停止應用，請按 Ctrl+C 或關閉此視窗
echo.
echo ========================================
echo.

REM 啟動Streamlit應用
streamlit run streamlit_app.py

REM 如果Streamlit退出，顯示訊息
echo.
echo 📴 AI交易系統已關閉
pause