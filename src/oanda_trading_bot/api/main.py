"""
Unified Trading Platform API
統一交易平台後端API
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional, Any
from datetime import datetime
import asyncio
import logging
import json
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OANDA AI Trading Platform API",
    description="統一管理訓練和實盤交易系統",
    version="2.0.0"
)

# CORS設置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket連接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {
            'training': [],
            'live': [],
            'general': []
        }
    
    async def connect(self, websocket: WebSocket, channel: str = 'general'):
        await websocket.accept()
        if channel not in self.active_connections:
            self.active_connections[channel] = []
        self.active_connections[channel].append(websocket)
        logger.info(f"WebSocket connected to {channel} channel")
    
    def disconnect(self, websocket: WebSocket, channel: str = 'general'):
        if channel in self.active_connections:
            if websocket in self.active_connections[channel]:
                self.active_connections[channel].remove(websocket)
        logger.info(f"WebSocket disconnected from {channel} channel")
    
    async def broadcast(self, message: dict, channel: str = 'general'):
        """廣播消息到指定頻道"""
        if channel not in self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections[channel]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
                disconnected.append(connection)
        
        # 清理斷開的連接
        for conn in disconnected:
            self.disconnect(conn, channel)

manager = ConnectionManager()

# 全局狀態（實際應用中應使用數據庫或緩存）
training_state = {
    'status': 'idle',  # idle, running, paused, completed, error
    'progress': 0,
    'current_episode': 0,
    'total_episodes': 0,
    'current_stage': 1,
    'latest_metrics': {},
    'error': None
}

live_trading_state = {
    'active': False,
    'status': 'stopped',  # stopped, running, paused
    'account': {},
    'positions': [],
    'recent_trades': [],
    'performance': {}
}

# ==================== Health Check ====================

@app.get("/")
async def root():
    """API根路徑"""
    return {
        "message": "OANDA AI Trading Platform API",
        "version": "2.0.0",
        "status": "running"
    }

@app.get("/api/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# ==================== System Status ====================

@app.get("/api/status")
async def get_system_status():
    """獲取整體系統狀態"""
    try:
        # 嘗試從training system獲取real status
        from oanda_trading_bot.training_system.common.shared_data_manager import get_shared_data_manager
        training_mgr = get_shared_data_manager()
        training_status_real = training_mgr.get_current_status()
        
        # 更新global state
        training_state.update({
            'status': training_status_real.get('status', 'idle'),
            'progress': training_status_real.get('progress', 0),
        })
    except Exception as e:
        logger.warning(f"Failed to get training status: {e}")
    
    return {
        "training": training_state,
        "live_trading": live_trading_state,
        "timestamp": datetime.now().isoformat()
    }

# ==================== Model Registry ====================

@app.get("/api/models")
async def list_models(
    status: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = 50
):
    """列出所有模型"""
    try:
        from oanda_trading_bot.common.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        filters = {}
        if status:
            filters['status'] = status
        if tag:
            filters['tag'] = tag
        
        models = registry.list_models(filters=filters, limit=limit)
        
        return {
            "models": models,
            "total": len(models)
        }
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/{model_id}")
async def get_model_details(model_id: str):
    """獲取模型詳細信息"""
    try:
        from oanda_trading_bot.common.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        model_metadata = registry.get_model(model_id)
        
        if not model_metadata:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return model_metadata.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/{model_id}/deploy")
async def deploy_model(model_id: str, environment: str = 'production'):
    """部署模型"""
    try:
        from oanda_trading_bot.common.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        registry.deploy_model(model_id, environment)
        
        return {
            "message": f"Model {model_id} deployed to {environment}",
            "model_id": model_id,
            "environment": environment
        }
    except Exception as e:
        logger.error(f"Failed to deploy model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/compare")
async def compare_models(model_ids: List[str]):
    """對比模型"""
    try:
        from oanda_trading_bot.common.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        comparison = registry.compare_models(model_ids)
        
        return comparison
    except Exception as e:
        logger.error(f"Failed to compare models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Training System ====================

@app.post("/api/training/start")
async def start_training(config: Dict, background_tasks: BackgroundTasks):
    """開始訓練"""
    try:
        if training_state['status'] == 'running':
            raise HTTPException(status_code=400, detail="Training already running")
        
        # 更新狀態
        training_state['status'] = 'running'
        training_state['progress'] = 0
        training_state['error'] = None
        
        # 在後台執行訓練
        def run_training_task():
            try:
                from oanda_trading_bot.training_system.trainer.universal_trainer import UniversalTrainer
                
                # 從config提取參數
                symbols = config.get('symbols', ['EUR_USD', 'GBP_USD'])
                total_timesteps = config.get('total_timesteps', 100000)
                
                logger.info(f"Starting training with {len(symbols)} symbols, {total_timesteps} timesteps")
                
                # 創建trainer並訓練
                trainer = UniversalTrainer(
                    symbols=symbols,
                    total_timesteps=total_timesteps,
                    # 其他參數...
                )
                
                trainer.train()
                
                training_state['status'] = 'completed'
                logger.info("Training completed successfully")
                
            except Exception as e:
                logger.error(f"Training error: {e}", exc_info=True)
                training_state['status'] = 'error'
                training_state['error'] = str(e)
        
        # 使用BackgroundTasks
        background_tasks.add_task(run_training_task)
        
        return {
            "message": "Training started",
            "status": "running",
            "config": config
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        training_state['status'] = 'error'
        training_state['error'] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/training/stop")
async def stop_training():
    """停止訓練"""
    try:
        from oanda_trading_bot.training_system.common.shared_data_manager import get_shared_data_manager
        
        mgr = get_shared_data_manager()
        mgr.request_stop()
        
        training_state['status'] = 'stopped'
        
        return {
            "message": "Stop requested",
            "status": "stopping"
        }
    except Exception as e:
        logger.error(f"Failed to stop training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/metrics")
async def get_training_metrics(limit: int = 100):
    """獲取訓練指標"""
    try:
        from oanda_trading_bot.training_system.common.shared_data_manager import get_shared_data_manager
        
        mgr = get_shared_data_manager()
        metrics = mgr.get_latest_metrics(count=limit)
        
        return {
            "metrics": metrics,
            "count": len(metrics)
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Live Trading System ====================

@app.get("/api/live/account")
async def get_account_info():
    """獲取賬戶信息"""
    try:
        # TODO: 從live trading system獲取
        return live_trading_state.get('account', {
            "balance": 0,
            "equity": 0,
            "margin_used": 0,
            "free_margin": 0,
            "currency": "AUD"
        })
    except Exception as e:
        logger.error(f"Failed to get account info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/live/positions")
async def get_open_positions():
    """獲取持倉"""
    try:
        # TODO: 從position manager獲取
        return {
            "positions": live_trading_state.get('positions', []),
            "count": len(live_trading_state.get('positions', []))
        }
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/live/trades")
async def get_recent_trades(limit: int = 50):
    """獲取最近交易"""
    try:
        return {
            "trades": live_trading_state.get('recent_trades', [])[:limit],
            "count": len(live_trading_state.get('recent_trades', []))
        }
    except Exception as e:
        logger.error(f"Failed to get trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/live/start")
async def start_live_trading(config: Dict):
    """啟動實盤交易"""
    try:
        if live_trading_state['active']:
            raise HTTPException(status_code=400, detail="Live trading already active")
        
        # TODO: 啟動live trading system
        live_trading_state['active'] = True
        live_trading_state['status'] = 'running'
        
        return {
            "message": "Live trading started",
            "status": "running"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start live trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/live/stop")
async def stop_live_trading():
    """停止實盤交易"""
    try:
        live_trading_state['active'] = False
        live_trading_state['status'] = 'stopped'
        
        return {
            "message": "Live trading stopped",
            "status": "stopped"
        }
    except Exception as e:
        logger.error(f"Failed to stop live trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WebSocket Endpoints ====================

@app.websocket("/ws/training")
async def training_websocket(websocket: WebSocket):
    """訓練進度實時推送"""
    await manager.connect(websocket, 'training')
    try:
        from oanda_trading_bot.training_system.common.shared_data_manager import get_shared_data_manager
        mgr = get_shared_data_manager()
        
        while True:
            await asyncio.sleep(1)
            
            # 獲取最新狀態
            status = mgr.get_current_status()
            latest_metrics = mgr.get_latest_metrics(count=1)
            
            # 發送更新
            await websocket.send_json({
                "type": "training_update",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "status": status,
                    "latest_metrics": latest_metrics
                }
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket, 'training')
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, 'training')

@app.websocket("/ws/live")
async def live_trading_websocket(websocket: WebSocket):
    """實盤交易實時推送"""
    await manager.connect(websocket, 'live')
    try:
        while True:
            await asyncio.sleep(1)
            
            # 發送實盤更新
            await websocket.send_json({
                "type": "live_update",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "positions": live_trading_state.get('positions', []),
                    "account": live_trading_state.get('account', {}),
                    "performance": live_trading_state.get('performance', {})
                }
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket, 'live')
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, 'live')

# ==================== Startup & Shutdown ====================

@app.on_event("startup")
async def startup_event():
    """應用啟動時執行"""
    logger.info("OANDA AI Trading Platform API started")
    logger.info("API documentation available at http://localhost:8000/docs")

@app.on_event("shutdown")
async def shutdown_event():
    """應用關閉時執行"""
    logger.info("OANDA AI Trading Platform API shutting down")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
