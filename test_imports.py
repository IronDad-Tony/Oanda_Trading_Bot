#!/usr/bin/env python3
# test_imports.py
"""
測試系統模組導入
用於診斷和修復模組導入問題
"""

import sys
import os
from pathlib import Path

def setup_project_path():
    """設置項目的 Python 路徑"""
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 設置 PYTHONPATH 環境變量
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if str(project_root) not in current_pythonpath:
        if current_pythonpath:
            os.environ['PYTHONPATH'] = f"{project_root}{os.pathsep}{current_pythonpath}"
        else:
            os.environ['PYTHONPATH'] = str(project_root)
    
    return project_root

def test_imports():
    """測試各個模組的導入"""
    project_root = setup_project_path()
    
    print("=" * 60)
    print("測試系統模組導入")
    print("=" * 60)
    print(f"項目根目錄: {project_root}")
    print(f"當前工作目錄: {os.getcwd()}")
    print(f"Python 版本: {sys.version}")
    print()
    
    # 測試導入列表
    import_tests = [
        ("src.common.config", "測試配置模組"),
        ("src.common.logger_setup", "測試日誌設置"),
        ("src.common.shared_data_manager", "測試共享數據管理器"),
        ("src.data_manager.database_manager", "測試數據庫管理器"),
        ("src.data_manager.oanda_downloader", "測試OANDA下載器"),
        ("src.data_manager.currency_manager", "測試貨幣管理器"),
        ("src.data_manager.instrument_info_manager", "測試工具信息管理器"),
        ("src.data_manager.mmap_dataset", "測試內存映射數據集"),
        ("src.environment.trading_env", "測試交易環境"),
        ("src.models.transformer_model", "測試Transformer模型"),
        ("src.agent.feature_extractors", "測試特徵提取器"),
        ("src.agent.sac_agent_wrapper", "測試SAC代理包裝器"),
        ("src.trainer.universal_trainer", "測試通用訓練器"),
        ("src.trainer.callbacks", "測試訓練回調"),
    ]
    
    success_count = 0
    total_count = len(import_tests)
    
    for module_name, description in import_tests:
        try:
            __import__(module_name)
            print(f"✅ {description}: {module_name}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {description}: {module_name}")
            print(f"   錯誤: {e}")
        except Exception as e:
            print(f"⚠️  {description}: {module_name}")
            print(f"   其他錯誤: {e}")
    
    print()
    print("=" * 60)
    print(f"導入測試結果: {success_count}/{total_count} 成功")
    
    if success_count == total_count:
        print("🎉 所有模組導入成功！")
        return True
    else:
        print(f"⚠️  有 {total_count - success_count} 個模組導入失敗")
        return False

def test_specific_configs():
    """測試特定配置是否正確載入"""
    print()
    print("=" * 60)
    print("測試配置載入")
    print("=" * 60)
    
    try:
        from src.common.config import (
            OANDA_API_KEY, OANDA_ACCOUNT_ID, ACCOUNT_CURRENCY,
            DATABASE_PATH, WEIGHTS_DIR, LOGS_DIR, DEVICE
        )
        
        print(f"✅ OANDA_API_KEY 已設置: {'是' if OANDA_API_KEY else '否'}")
        print(f"✅ OANDA_ACCOUNT_ID 已設置: {'是' if OANDA_ACCOUNT_ID else '否'}")
        print(f"✅ 帳戶貨幣: {ACCOUNT_CURRENCY}")
        print(f"✅ 數據庫路徑: {DATABASE_PATH}")
        print(f"✅ 權重目錄: {WEIGHTS_DIR}")
        print(f"✅ 日誌目錄: {LOGS_DIR}")
        print(f"✅ 設備: {DEVICE}")
        
        return True
    except Exception as e:
        print(f"❌ 配置載入失敗: {e}")
        return False

def main():
    """主函數"""
    print("開始診斷...")
    
    # 測試導入
    imports_ok = test_imports()
    
    # 測試配置
    config_ok = test_specific_configs()
    
    print()
    print("=" * 60)
    print("診斷結果")
    print("=" * 60)
    
    if imports_ok and config_ok:
        print("🎉 系統診斷成功！所有模組都可以正常導入和運行。")
        return 0
    else:
        print("⚠️  系統診斷發現問題，請檢查上述錯誤信息。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    print()
    input("按 Enter 鍵結束...")
    sys.exit(exit_code)
