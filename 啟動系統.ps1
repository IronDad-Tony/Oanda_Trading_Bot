# OANDA Trading Bot - PowerShell 啟動腳本
# 解決 PyTorch 2.7.0+ 與 Streamlit 的兼容性問題

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   OANDA AI 交易系統 - 完整監控系統     " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 設置工作目錄
$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptPath

Write-Host "專案目錄: $ScriptPath" -ForegroundColor Green

# 設置環境變量
$env:PYTHONPATH = "$ScriptPath;$env:PYTHONPATH"
$env:TORCH_CPP_LOG_LEVEL = "ERROR"
$env:TORCH_DISTRIBUTED_DEBUG = "OFF"
$env:TORCH_SHOW_CPP_STACKTRACES = "0"
$env:PYTORCH_DISABLE_PER_OP_PROFILING = "1"

Write-Host "✓ 環境變量設置完成" -ForegroundColor Green

# 檢查 Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python 版本: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ 未找到 Python，請先安裝 Python" -ForegroundColor Red
    Read-Host "按 Enter 鍵退出"
    exit 1
}

# 檢查必要套件
Write-Host "檢查必要套件..." -ForegroundColor Yellow

$packages = @("streamlit", "torch", "pandas", "numpy", "plotly")
$missingPackages = @()

foreach ($package in $packages) {
    try {
        python -c "import $package" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ $package" -ForegroundColor Green
        } else {
            Write-Host "✗ $package (缺失)" -ForegroundColor Red
            $missingPackages += $package
        }
    } catch {
        Write-Host "✗ $package (檢查失敗)" -ForegroundColor Red
        $missingPackages += $package
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host ""
    Write-Host "發現缺失套件，正在安裝..." -ForegroundColor Yellow
    pip install streamlit torch pandas numpy plotly GPUtil psutil python-dotenv
}

Write-Host ""
Write-Host "正在啟動 Streamlit 應用..." -ForegroundColor Yellow
Write-Host "瀏覽器將自動打開: http://localhost:8501" -ForegroundColor Cyan
Write-Host "按 Ctrl+C 停止服務" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

try {
    # 使用改進的啟動器
    python streamlit_launcher.py
} catch {
    Write-Host ""
    Write-Host "使用 Python 啟動器失敗，嘗試直接啟動..." -ForegroundColor Yellow
    
    try {
        streamlit run streamlit_app_complete.py --server.port=8501 --server.fileWatcherType=none
    } catch {
        Write-Host "直接啟動也失敗，請檢查環境配置" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "應用已停止" -ForegroundColor Yellow
Read-Host "按 Enter 鍵退出"
