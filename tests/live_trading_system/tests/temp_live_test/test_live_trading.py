import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import subprocess
from pathlib import Path

# 添加src目錄到系統路徑
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / 'src'))

# 導入必要的模塊
from agent.sac_agent_wrapper import SACAgentWrapper
from common.config import load_config
from environment.trading_env import TradingEnv
from live_trading_system.core.oanda_client import OandaClient
from live_trading_system.trading.order_manager import OrderManager
from live_trading_system.trading.position_manager import PositionManager

# 初始化沙盒環境
def setup_sandbox():
    """初始化OANDA沙盒環境並設置API金鑰"""
    os.environ['OANDA_ENV'] = 'practice'
    os.environ['OANDA_DEMO_API_KEY'] = 'your_demo_api_key_here'  # 實際使用時替換為真實DEMO KEY
    # 获取配置文件绝对路径
    config_path = Path(__file__).resolve().parent.parent.parent / 'live_config.json'
    config = load_config(str(config_path))
    return OandaClient(config), config

# 模型加載驗證
def test_model_loading():
    """驗證SAC模型加載是否正確"""
    # 初始化沙盒環境
    client, config = setup_sandbox()
    
    try:
        # 加載模型
        model = SACAgentWrapper(config)
        assert model is not None, "模型加載失敗"
        
        # 創建模擬輸入數據
        mock_state = MagicMock()
        
        # 獲取預測結果
        action = model.predict(mock_state)
        assert action is not None, "模型預測返回None"
        
        print("✅ 模型加載驗證通過")
    except Exception as e:
        pytest.fail(f"模型加載驗證失敗: {str(e)}")
    finally:
        # 清理環境
        client.close()

# 交易循環測試
def test_trading_cycle():
    """測試完整交易循環（限價單+市價單）"""
    client, config = setup_sandbox()
    # 初始化交易管理器和环境
    order_manager = OrderManager(client, config)
    position_manager = PositionManager(client, config)
    
    # 初始化交易环境需要更多参数，这里使用mock简化
    with patch('live_trading_system.trading.trading_logic.TradingLogic') as mock_trading_logic:
        env = TradingEnv(config, mock_trading_logic)
    
    try:
        # 市價單測試
        market_order = {
            'instrument': 'EUR_USD',
            'units': 1000,
            'type': 'MARKET',
            'side': 'BUY'
        }
        market_result = order_manager.execute_order(market_order)
        assert market_result['status'] == 'FILLED', "市價單執行失敗"
        
        # 限價單測試
        limit_order = {
            'instrument': 'EUR_USD',
            'units': 1000,
            'type': 'LIMIT',
            'price': 1.0800,
            'side': 'SELL'
        }
        limit_result = order_manager.execute_order(limit_order)
        assert limit_result['status'] == 'PENDING', "限價單創建失敗"
        
        print("✅ 交易循環測試通過")
    except Exception as e:
        pytest.fail(f"交易循環測試失敗: {str(e)}")
    finally:
        # 清理倉位
        position_manager.close_all_positions()
        client.close()

# 異常處理測試
def test_exception_handling():
    """測試交易異常處理機制"""
    client, config = setup_sandbox()
    order_manager = OrderManager(client, config)
    
    try:
        # 故意發送無效訂單
        invalid_order = {
            'instrument': 'INVALID_PAIR',
            'units': 1000,
            'type': 'MARKET'
        }
        
        with pytest.raises(Exception):
            order_manager.execute_order(invalid_order)
            
        print("✅ 異常處理測試通過")
    finally:
        client.close()

# 主測試函數
def test_live_trading():
    """主測試流程"""
    try:
        test_model_loading()
        test_trading_cycle()
        test_exception_handling()
    finally:
        # 確保執行清理腳本
        cleanup_script = Path(__file__).parent / 'cleanup_test.py'
        if cleanup_script.exists():
            os.system(f'python {cleanup_script}')
        else:
            print("⚠️ 清理腳本不存在")

if __name__ == "__main__":
    test_live_trading()