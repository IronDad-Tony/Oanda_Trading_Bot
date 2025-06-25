import os
import shutil
from pathlib import Path
from live_trading_system.core.oanda_client import OandaClient
from live_trading_system.trading.position_manager import PositionManager

def cleanup():
    """執行清理操作：平倉、刪除臨時文件、重置環境"""
    # 初始化OANDA客戶端
    config_path = Path(__file__).resolve().parent.parent.parent / 'live_config.json'
    client = OandaClient.load_config(str(config_path))
    position_manager = PositionManager(client, config)
    
    try:
        print("🛟 開始清理測試環境...")
        
        # 平倉所有測試倉位
        position_manager.close_all_positions()
        print("✅ 所有測試倉位已平倉")
        
        # 刪除臨時目錄
        temp_dirs = [
            Path("temp_data"),
            Path("debug_logs"),
            Path("test_artifacts")
        ]
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                print(f"🧹 已刪除臨時目錄: {temp_dir}")
        
        # 重置環境變量
        os.environ.pop('OANDA_ENV', None)
        os.environ.pop('OANDA_DEMO_API_KEY', None)
        print("🔄 環境變量已重置")
        
        print("✨ 清理完成")
    except Exception as e:
        print(f"❌ 清理過程中出錯: {str(e)}")
    finally:
        client.close()

if __name__ == "__main__":
    cleanup()