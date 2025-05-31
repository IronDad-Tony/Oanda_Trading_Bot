#!/usr/bin/env python3
"""
啟動 Streamlit 應用的兼容性包裝器
解決 PyTorch 2.7.0+ 與 Streamlit 的兼容性問題
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """設置環境變量以解決兼容性問題"""
    # 設置 PyTorch 環境變量
    os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'
    os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'
    
    # 禁用一些可能導致問題的 PyTorch 功能
    os.environ['PYTORCH_DISABLE_PER_OP_PROFILING'] = '1'
    
    # 設置 TensorFlow 環境變量以抑制警告
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 抑制 INFO 和 WARNING
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用 oneDNN 優化警告
    
    # 設置項目路徑
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    print(f"環境設置完成，項目根目錄: {project_root}")

def fix_torch_classes():
    """修復 torch._classes 模組的路徑問題"""
    try:
        import torch
        # 確保 torch._classes 不會被 Streamlit 的文件監視器檢查
        if hasattr(torch, '_classes'):
            if hasattr(torch._classes, '__path__'):
                torch._classes.__path__ = []
        print("PyTorch 類別模組修復完成")
    except Exception as e:
        print(f"PyTorch 修復時出現警告: {e}")

def main():
    """主啟動函數"""
    print("正在啟動 OANDA Trading Bot Streamlit 應用...")
    print("=" * 50)
    
    # 設置環境
    setup_environment()
    
    # 修復 PyTorch 兼容性
    fix_torch_classes()
    
    # 檢查依賴（使用pkg_resources避免導入streamlit）
    print("檢查依賴套件...")
    try:
        import pkg_resources
        required_packages = {'streamlit', 'torch', 'pandas', 'numpy'}
        installed_packages = {pkg.key for pkg in pkg_resources.working_set}
        missing = required_packages - installed_packages
        
        if missing:
            print(f"✗ 缺少必要套件: {', '.join(missing)}")
            print("請運行: pip install -r docs/requirements.txt")
            return
        else:
            print("✓ 所有必要套件已安裝")
    except Exception as e:
        print(f"✗ 檢查依賴時出錯: {e}")
        return
    
    # 導入並啟動 Streamlit
    try:
        print("正在啟動 Streamlit 服務器...")
        print("瀏覽器將自動打開: http://localhost:8501")
        print("按 Ctrl+C 停止服務")
        print("=" * 50)
        
        import streamlit.web.cli as stcli
        
        # 設置 Streamlit 參數
        app_file = Path(__file__).parent / "streamlit_app_complete.py"
        
        # 啟動 Streamlit
        sys.argv = [
            "streamlit",
            "run",
            str(app_file),
            "--server.port=8501",
            "--server.address=localhost",
            "--server.fileWatcherType=none",
            "--logger.level=error",
            "--browser.gatherUsageStats=false"
        ]
        
        stcli.main()
        
    except KeyboardInterrupt:
        print("\n\n應用已停止")
    except Exception as e:
        print(f"✗ 啟動 Streamlit 時出現錯誤: {e}")
        print("嘗試使用備用方法...")
        
        # 備用方法：使用系統命令
        try:
            import subprocess
            cmd = [
                "streamlit", "run", "streamlit_app_complete.py",
                "--server.port=8501",
                "--server.fileWatcherType=none"
            ]
            subprocess.run(cmd)
        except Exception as e2:
            print(f"✗ 備用方法也失敗: {e2}")
            print("請手動運行: streamlit run streamlit_app_complete.py")

if __name__ == "__main__":
    main()
