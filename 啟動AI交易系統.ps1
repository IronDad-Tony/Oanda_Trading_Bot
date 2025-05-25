# OANDA AI交易系統 PowerShell啟動腳本
# 設置執行策略以允許腳本運行
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force

# 設置控制台編碼為UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "OANDA AI交易系統啟動器"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "    🚀 OANDA AI交易系統啟動器" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 獲取腳本所在目錄
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

Write-Host "📁 項目目錄: $ProjectDir" -ForegroundColor Green
Write-Host ""

# 檢查Python是否安裝
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python已安裝: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ 錯誤: 未找到Python，請先安裝Python" -ForegroundColor Red
    Write-Host "📥 下載地址: https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "按Enter鍵退出"
    exit 1
}

# 檢查streamlit是否安裝
try {
    python -c "import streamlit" 2>$null
    Write-Host "✅ Streamlit已安裝" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "⚠️  Streamlit未安裝，正在安裝依賴..." -ForegroundColor Yellow
    Write-Host ""
    
    try {
        pip install -r requirements.txt
        Write-Host "✅ 依賴安裝完成" -ForegroundColor Green
    } catch {
        Write-Host "❌ 依賴安裝失敗，請檢查網絡連接" -ForegroundColor Red
        Read-Host "按Enter鍵退出"
        exit 1
    }
}

Write-Host ""

# 檢查.env文件
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  警告: 未找到.env文件" -ForegroundColor Yellow
    Write-Host "📝 請確保已設置OANDA API密鑰" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "創建.env文件範例:" -ForegroundColor Cyan
    Write-Host "OANDA_API_KEY=your_api_key_here" -ForegroundColor Gray
    Write-Host "OANDA_ACCOUNT_ID=your_account_id_here" -ForegroundColor Gray
    Write-Host ""
    
    $continue = Read-Host "是否繼續啟動？(y/N)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        Write-Host "已取消啟動" -ForegroundColor Yellow
        Read-Host "按Enter鍵退出"
        exit 0
    }
}

Write-Host "🚀 正在啟動Streamlit應用..." -ForegroundColor Green
Write-Host ""
Write-Host "📱 應用將在瀏覽器中自動打開" -ForegroundColor Cyan
Write-Host "🌐 如果沒有自動打開，請手動訪問: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "⏹️  要停止應用，請按 Ctrl+C 或關閉此視窗" -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 啟動Streamlit應用
try {
    streamlit run streamlit_app.py
} catch {
    Write-Host ""
    Write-Host "❌ 啟動失敗: $_" -ForegroundColor Red
} finally {
    Write-Host ""
    Write-Host "📴 AI交易系統已關閉" -ForegroundColor Yellow
    Read-Host "按Enter鍵退出"
}