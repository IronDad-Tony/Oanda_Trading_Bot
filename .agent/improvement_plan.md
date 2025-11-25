# OANDA交易機器人 - 詳細改進實施方案

## 改進方案總覽

本文檔提供所有改進項目的詳細技術設計和實施步驟。

---

## 方案 1: 統一管理平台 UI

### 1.1 技術架構

```
┌─────────────────────────────────────────┐
│         React Frontend (Port 3000)       │
│  - Dashboard                             │
│  - Training Center                       │
│  - Live Trading Monitor                  │
│  - Model Registry                        │
│  - Backtesting Lab                       │
└────────────┬────────────────────────────┘
             │ HTTP + WebSocket
┌────────────▼────────────────────────────┐
│      FastAPI Backend (Port 8000)         │
│  - REST API Endpoints                    │
│  - WebSocket Handlers                    │
│  - Authentication                        │
└────────────┬────────────────────────────┘
             │
   ┌─────────┼─────────┐
   │         │         │
   ▼         ▼         ▼
┌──────┐ ┌──────┐ ┌──────┐
│Training│Live   │Model  │
│System │Trading│Registry│
└──────┘ └──────┘ └──────┘
```

### 1.2 Backend API設計

創建 `src/oanda_trading_bot/api/main.py`:

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from typing import List, Dict
import json

app = FastAPI(title="OANDA Trading Platform API")

# CORS設置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket連接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

# ========== REST API Endpoints ==========

@app.get("/api/status")
async def get_system_status():
    """獲取整體系統狀態"""
    from oanda_trading_bot.training_system.common.shared_data_manager import get_shared_data_manager
    from oanda_trading_bot.live_trading_system.core.system_state import SystemState
    
    training_mgr = get_shared_data_manager()
    training_status = training_mgr.get_current_status()
    
    # TODO: 獲取live trading狀態
    
    return {
        "training": {
            "status": training_status['status'],
            "progress": training_status['progress'],
            "current_stage": training_status.get('current_stage'),
        },
        "live_trading": {
            "active": False,  # TODO
            "positions": 0,
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/models")
async def list_models(
    status: Optional[str] = None,
    limit: int = 50
):
    """列出所有模型"""
    from oanda_trading_bot.common.model_registry import ModelRegistry
    
    registry = ModelRegistry()
    models = registry.list_models(
        filters={'status': status} if status else {},
        limit=limit
    )
    return models

@app.get("/api/models/{model_id}")
async def get_model_details(model_id: str):
    """獲取模型詳細信息"""
    from oanda_trading_bot.common.model_registry import ModelRegistry
    
    registry = ModelRegistry()
    model_info = registry.get_model(model_id)
    
    if not model_info:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return model_info

@app.post("/api/training/start")
async def start_training(config: Dict):
    """開始訓練"""
    from oanda_trading_bot.training_system.trainer.universal_trainer import UniversalTrainer
    import threading
    
    def run_training():
        trainer = UniversalTrainer(**config)
        trainer.train()
    
    # 在後台線程中運行
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()
    
    return {"message": "Training started", "status": "running"}

@app.post("/api/training/stop")
async def stop_training():
    """停止訓練"""
    from oanda_trading_bot.training_system.common.shared_data_manager import get_shared_data_manager
    
    mgr = get_shared_data_manager()
    mgr.request_stop()
    
    return {"message": "Stop requested"}

@app.get("/api/live/account")
async def get_account_info():
    """獲取賬戶信息"""
    # TODO: 從live trading system獲取
    return {
        "balance": 10000,
        "equity": 10234.56,
        "margin_used": 500,
        "free_margin": 9734.56
    }

@app.get("/api/live/positions")
async def get_open_positions():
    """獲取持倉"""
    # TODO: 從position manager獲取
    return []

# ========== WebSocket Endpoints ==========

@app.websocket("/ws/training")
async def training_websocket(websocket: WebSocket):
    """訓練進度實時推送"""
    await manager.connect(websocket)
    try:
        from oanda_trading_bot.training_system.common.shared_data_manager import get_shared_data_manager
        mgr = get_shared_data_manager()
        
        while True:
            # 每秒推送一次
            await asyncio.sleep(1)
            
            status = mgr.get_current_status()
            latest_metrics = mgr.get_latest_metrics(count=1)
            
            await websocket.send_json({
                "type": "training_update",
                "data": {
                    "status": status,
                    "latest_metrics": latest_metrics
                }
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/ws/live")
async def live_trading_websocket(websocket: WebSocket):
    """實盤交易實時推送"""
    await manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(1)
            
            # TODO: 推送實盤數據
            await websocket.send_json({
                "type": "live_update",
                "data": {
                    "timestamp": datetime.now().isoformat(),
                    "positions": [],  # TODO
                    "pnl": 0  # TODO
                }
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/ws/candles/{symbol}")
async def candles_websocket(websocket: WebSocket, symbol: str):
    """K線數據實時推送"""
    await manager.connect(websocket)
    try:
        from oanda_trading_bot.live_trading_system.core.oanda_client import OandaClient
        
        client = OandaClient()
        
        while True:
            # 獲取最新K線
            candles = client.get_bid_ask_candles_combined(symbol, count=1)
            if candles:
                await websocket.send_json({
                    "symbol": symbol,
                    "candle": candles[-1]
                })
            
            await asyncio.sleep(5)  # 5秒更新一次
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 1.3 Frontend結構

創建新的React應用：

```bash
cd frontend
npm create vite@latest . -- --template react-ts
npm install @mui/material @emotion/react @emotion/styled
npm install @reduxjs/toolkit react-redux
npm install lightweight-charts
npm install plotly.js react-plotly.js
npm install socket.io-client
npm install recharts
```

**項目結構：**
```
frontend/
├── src/
│   ├── app/
│   │   ├── store.ts                 # Redux store
│   │   └── hooks.ts                 # Typed hooks
│   ├── features/
│   │   ├── training/
│   │   │   ├── TrainingCenter.tsx
│   │   │   ├── trainingSlice.ts
│   │   │   └── TrainingMonitor.tsx
│   │   ├── live/
│   │   │   ├── LiveDashboard.tsx
│   │   │   ├── liveSlice.ts
│   │   │   └── TradingChart.tsx
│   │   ├── models/
│   │   │   ├── ModelRegistry.tsx
│   │   │   ├── modelsSlice.ts
│   │   │   └── ModelCard.tsx
│   │   └── dashboard/
│   │       └── UnifiedDashboard.tsx
│   ├── components/
│   │   ├── charts/
│   │   │   ├── EquityCurve.tsx
│   │   │   ├── TradingChart.tsx
│   │   │   └── MetricsChart.tsx
│   │   ├── common/
│   │   │   ├── MetricCard.tsx
│   │   │   ├── StatusBadge.tsx
│   │   │   └── LoadingSpinner.tsx
│   │   └── layout/
│   │       ├── Sidebar.tsx
│   │       ├── Header.tsx
│   │       └── MainLayout.tsx
│   ├── services/
│   │   ├── api.ts                   # API calls
│   │   └── websocket.ts             # WebSocket manager
│   ├── types/
│   │   ├── training.ts
│   │   ├── live.ts
│   │   └── models.ts
│   └── App.tsx
└── package.json
```

**關鍵組件實現：**

`src/services/websocket.ts`:
```typescript
import io, { Socket } from 'socket.io-client';

class WebSocketService {
  private socket: Socket | null = null;
  private callbacks: Map<string, Function[]> = new Map();

  connect(url: string = 'http://localhost:8000') {
    this.socket = io(url, {
      transports: ['websocket'],
    });

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
    });

    this.socket.on('disconnect', () => {
      console.log('WebSocket disconnected');
    });

    // 設置通用消息處理
    this.socket.onAny((event, data) => {
      const handlers = this.callbacks.get(event) || [];
      handlers.forEach(handler => handler(data));
    });
  }

  on(event: string, callback: Function) {
    if (!this.callbacks.has(event)) {
      this.callbacks.set(event, []);
    }
    this.callbacks.get(event)!.push(callback);
  }

  emit(event: string, data: any) {
    if (this.socket) {
      this.socket.emit(event, data);
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }
}

export const wsService = new WebSocketService();
```

`src/features/training/TrainingCenter.tsx`:
```typescript
import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Button,
  LinearProgress,
} from '@mui/material';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { wsService } from '../../services/websocket';
import { updateTrainingStatus } from './trainingSlice';
import EquityCurve from '../../components/charts/EquityCurve';

export const TrainingCenter: React.FC = () => {
  const dispatch = useAppDispatch();
  const trainingStatus = useAppSelector((state) => state.training.status);
  const metrics = useAppSelector((state) => state.training.metrics);

  useEffect(() => {
    // 訂閱WebSocket訓練更新
    wsService.on('training_update', (data: any) => {
      dispatch(updateTrainingStatus(data));
    });

    return () => {
      // 清理
    };
  }, [dispatch]);

  const handleStartTraining = async () => {
    const config = {
      symbols: ['EUR_USD', 'GBP_USD'],
      total_timesteps: 2000000,
      // ... 其他配置
    };

    const response = await fetch('http://localhost:8000/api/training/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });

    const result = await response.json();
    console.log(result);
  };

  const handleStopTraining = async () => {
    await fetch('http://localhost:8000/api/training/stop', {
      method: 'POST',
    });
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Training Center
      </Typography>

      <Grid container spacing={3}>
        {/* 左側：配置 */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Configuration
            </Typography>
            
            {/* TODO: 配置表單 */}
            
            <Box sx={{ mt: 2 }}>
              <Button
                variant="contained"
                color="primary"
                fullWidth
                onClick={handleStartTraining}
                disabled={trainingStatus === 'running'}
              >
                Start Training
              </Button>
              
              <Button
                variant="outlined"
                color="secondary"
                fullWidth
                sx={{ mt: 1 }}
                onClick={handleStopTraining}
                disabled={trainingStatus !== 'running'}
              >
                Stop Training
              </Button>
            </Box>
          </Paper>
        </Grid>

        {/* 右側：監控 */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Training Monitor
            </Typography>

            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Status: {trainingStatus}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Progress: {metrics.progress}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={metrics.progress}
                sx={{ mt: 1 }}
              />
            </Box>

            {/* 獎勵曲線 */}
            <EquityCurve data={metrics.rewardHistory} />

            {/* 當前指標 */}
            <Grid container spacing={2} sx={{ mt: 2 }}>
              <Grid item xs={6} md={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="h6">{metrics.episodeReward?.toFixed(2)}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Episode Reward
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={6} md={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="h6">{metrics.sharpeRatio?.toFixed(2)}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Sharpe Ratio
                  </Typography>
                </Paper>
              </Grid>
              {/* 更多指標... */}
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};
```

---

## 方案 2: 模型註冊與版本控制

### 2.1 Model Registry實現

創建 `src/oanda_trading_bot/common/model_registry.py`:

```python
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import shutil
import logging

logger = logging.getLogger(__name__)

class ModelMetadata:
    """模型元數據"""
    def __init__(self, **kwargs):
        self.model_id: str = kwargs.get('model_id', str(uuid.uuid4()))
        self.model_name: str = kwargs.get('model_name', '')
        self.version: str = kwargs.get('version', '1.0.0')
        self.created_at: str = kwargs.get('created_at', datetime.now().isoformat())
        self.created_by: str = kwargs.get('created_by', 'system')
        self.git_commit: Optional[str] = kwargs.get('git_commit')
        
        # 訓練配置
        self.training_config: Dict = kwargs.get('training_config', {})
        
        # 模型架構
        self.model_architecture: Dict = kwargs.get('model_architecture', {})
        
        # 訓練指標
        self.training_metrics: Dict = kwargs.get('training_metrics', {})
        
        # 驗證結果
        self.validation_results: Dict = kwargs.get('validation_results', {})
        
        # 部署狀態
        self.deployment_status: str = kwargs.get('deployment_status', 'development')
        
        # 標籤
        self.tags: List[str] = kwargs.get('tags', [])
        
        # 備註
        self.notes: str = kwargs.get('notes', '')
    
    def to_dict(self) -> Dict:
        return {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'version': self.version,
            'created_at': self.created_at,
            'created_by': self.created_by,
            'git_commit': self.git_commit,
            'training_config': self.training_config,
            'model_architecture': self.model_architecture,
            'training_metrics': self.training_metrics,
            'validation_results': self.validation_results,
            'deployment_status': self.deployment_status,
            'tags': self.tags,
            'notes': self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


class ModelRegistry:
    """模型倉庫管理器"""
    
    def __init__(self, registry_dir: Optional[Path] = None):
        if registry_dir is None:
            from oanda_trading_bot.training_system.common.config import BASE_DIR
            registry_dir = BASE_DIR / 'model_registry'
        
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # 索引文件
        self.index_file = self.registry_dir / 'index.json'
        self._load_index()
    
    def _load_index(self):
        """加載索引"""
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
        else:
            self.index = {'models': []}
            self._save_index()
    
    def _save_index(self):
        """保存索引"""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)
    
    def save_model(
        self,
        model_path: Path,
        metadata: ModelMetadata,
        overwrite: bool = False
    ) -> str:
        """
        保存模型到倉庫
        
        Returns:
            model_id
        """
        model_id = metadata.model_id
        model_dir = self.registry_dir / model_id
        
        if model_dir.exists() and not overwrite:
            raise ValueError(f"Model {model_id} already exists")
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 複製模型文件
        model_file = model_dir / 'model.zip'
        shutil.copy(model_path, model_file)
        
        # 保存元數據
        metadata_file = model_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 更新索引
        # 移除舊條目（如果存在）
        self.index['models'] = [
            m for m in self.index['models']
            if m['model_id'] != model_id
        ]
        # 添加新條目
        self.index['models'].append({
            'model_id': model_id,
            'model_name': metadata.model_name,
            'version': metadata.version,
            'created_at': metadata.created_at,
            'deployment_status': metadata.deployment_status,
            'tags': metadata.tags,
        })
        self._save_index()
        
        logger.info(f"Model {model_id} saved to registry")
        return model_id
    
    def load_model(self, model_id: str) -> Optional[Path]:
        """加載模型文件路徑"""
        model_dir = self.registry_dir / model_id
        model_file = model_dir / 'model.zip'
        
        if not model_file.exists():
            logger.error(f"Model file not found: {model_file}")
            return None
        
        return model_file
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """獲取模型元數據"""
        model_dir = self.registry_dir / model_id
        metadata_file = model_dir / 'metadata.json'
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return ModelMetadata.from_dict(data)
    
    def list_models(
        self,
        filters: Optional[Dict] = None,
        limit: int = 100
    ) -> List[Dict]:
        """列出模型"""
        models = self.index['models']
        
        # 應用過濾器
        if filters:
            if 'status' in filters:
                models = [
                    m for m in models
                    if m.get('deployment_status') == filters['status']
                ]
            if 'tag' in filters:
                models = [
                    m for m in models
                    if filters['tag'] in m.get('tags', [])
                ]
        
        # 按創建時間倒序排序
        models = sorted(
            models,
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )
        
        return models[:limit]
    
    def deploy_model(self, model_id: str, environment: str = 'production'):
        """部署模型"""
        metadata = self.get_model(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found")
        
        # 更新部署狀態
        metadata.deployment_status = environment
        
        # 保存更新後的元數據
        model_dir = self.registry_dir / model_id
        metadata_file = model_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 更新索引
        for m in self.index['models']:
            if m['model_id'] == model_id:
                m['deployment_status'] = environment
                break
        self._save_index()
        
        logger.info(f"Model {model_id} deployed to {environment}")
    
    def compare_models(self, model_ids: List[str]) -> Dict:
        """對比多個模型"""
        comparison = {
            'models': [],
            'metrics_comparison': {},
        }
        
        for model_id in model_ids:
            metadata = self.get_model(model_id)
            if metadata:
                comparison['models'].append(metadata.to_dict())
        
        # 提取共同的指標進行對比
        if comparison['models']:
            # 獲取所有可用的指標名稱
            all_metrics = set()
            for model in comparison['models']:
                all_metrics.update(model.get('training_metrics', {}).keys())
            
            # 對每個指標進行對比
            for metric in all_metrics:
                values = {}
                for model in comparison['models']:
                    model_id = model['model_id']
                    value = model.get('training_metrics', {}).get(metric)
                    if value is not None:
                        values[model_id] = value
                comparison['metrics_comparison'][metric] = values
        
        return comparison
```

### 2.2 集成到訓練流程

修改 `src/oanda_trading_bot/training_system/trainer/universal_trainer.py`:

```python
from oanda_trading_bot.common.model_registry import ModelRegistry, ModelMetadata
import subprocess

class UniversalTrainer:
    def __init__(self, ...):
        # ... 現有代碼 ...
        self.model_registry = ModelRegistry()
    
    def train(self):
        # ... 訓練代碼 ...
        
        # 訓練完成後保存到registry
        self._save_to_registry()
    
    def _get_git_commit(self) -> Optional[str]:
        """獲取當前Git commit hash"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except Exception:
            return None
    
    def _save_to_registry(self):
        """保存模型到registry"""
        model_path = self.checkpoint_path / 'final_model.zip'
        
        if not model_path.exists():
            logger.warning("Model file not found, skipping registry save")
            return
        
        # 創建元數據
        metadata = ModelMetadata(
            model_name=f"sac_transformer_{len(self.symbols)}symbols",
            version="1.0.0",
            git_commit=self._get_git_commit(),
            training_config={
                'symbols': self.symbols,
                'total_timesteps': self.total_timesteps,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                # ... 其他配置
            },
            model_architecture={
                'type': 'QuantumEnhancedSAC',
                'feature_extractor': 'EnhancedTransformer',
                'transformer_layers': 32,
                'hidden_dim': 1024,
                # ... 其他架構信息
            },
            training_metrics={
                'final_episode_reward': self.final_reward,
                'sharpe_ratio': self.final_sharpe,
                'max_drawdown': self.final_max_dd,
                'win_rate': self.final_win_rate,
                # ... 其他指標
            },
            deployment_status='development',
            tags=['transformer', 'sac', 'multi-asset'],
            notes='Trained with progressive reward system'
        )
        
        # 保存到registry
        model_id = self.model_registry.save_model(model_path, metadata)
        logger.info(f"Model saved to registry with ID: {model_id}")
```

---

## 方案 3: 獨立Backtester實現

創建 `src/oanda_trading_bot/backtester/backtester.py`:

```python
from typing import Protocol, Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import pandas as pd
import numpy as np
from enum import Enum

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

@dataclass
class Order:
    symbol: str
    order_type: OrderType
    side: str  # 'buy' or 'sell'
    units: int
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: Optional[datetime] = None

@dataclass
class Trade:
    trade_id: str
    symbol: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    units: int
    side: str
    pnl: float
    pnl_percent: float
    commission: float
    
class Strategy(Protocol):
    """策略接口"""
    def on_data(self, timestamp: datetime, data: Dict[str, pd.DataFrame]) -> List[Order]:
        """
        接收市場數據，返回訂單列表
        
        Args:
            timestamp: 當前時間
            data: {symbol: DataFrame} 格式的市場數據
        
        Returns:
            訂單列表
        """
        ...

class UniversalBacktester:
    """通用回測引擎"""
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        initial_capital: float = 10000,
        commission_rate: float = 0.0001,
        slippage_pips: float = 0.5,
    ):
        self.data = data
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_pips = slippage_pips
        
        # 回測狀態
        self.cash = initial_capital
        self.equity = initial_capital
        self.positions: Dict[str, Dict] = {}  # {symbol: {units, entry_price, ...}}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        
        # 時間軸（所有symbol的時間戳合併）
        self.timeline = self._create_timeline()
    
    def _create_timeline(self) -> pd.DatetimeIndex:
        """創建統一的時間軸"""
        all_times = []
        for symbol, df in self.data.items():
            all_times.extend(df.index.tolist())
        
        return pd.DatetimeIndex(sorted(set(all_times)))
    
    def run(self, strategy: Strategy) -> 'BacktestResult':
        """運行回測"""
        for timestamp in self.timeline:
            # 獲取當前時刻的數據
            current_data = self._get_data_at(timestamp)
            
            # 策略決策
            orders = strategy.on_data(timestamp, current_data)
            
            # 執行訂單
            for order in orders:
                self._execute_order(order, current_data)
            
            # 更新權益
            self._update_equity(timestamp, current_data)
            
            # 記錄權益曲線
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self.equity,
                'cash': self.cash,
                'positions_value': self.equity - self.cash
            })
        
        # 生成報告
        return self._generate_result()
    
    def _get_data_at(self, timestamp: datetime) -> Dict[str, pd.Series]:
        """獲取指定時間的數據"""
        result = {}
        for symbol, df in self.data.items():
            if timestamp in df.index:
                result[symbol] = df.loc[timestamp]
        return result
    
    def _execute_order(self, order: Order, current_data: Dict[str, pd.Series]):
        """執行訂單"""
        if order.symbol not in current_data:
            return  # 數據不可用
        
        current_price = current_data[order.symbol]['close']
        
        # 應用滑價
        if order.side == 'buy':
            execution_price = current_price * (1 + self.slippage_pips / 10000)
        else:
            execution_price = current_price * (1 - self.slippage_pips / 10000)
        
        # 計算手續費
        commission = abs(order.units) * execution_price * self.commission_rate
        
        # 檢查是否有足夠的資金
        required_cash = abs(order.units) * execution_price + commission
        if order.side == 'buy' and self.cash < required_cash:
            return  # 資金不足
        
        # 執行
        if order.side == 'buy':
            # 開多倉或平空倉
            if order.symbol in self.positions and self.positions[order.symbol]['side'] == 'short':
                # 平空倉
                self._close_position(order.symbol, execution_price, order.timestamp or datetime.now())
            else:
                # 開多倉
                self.positions[order.symbol] = {
                    'side': 'long',
                    'units': order.units,
                    'entry_price': execution_price,
                    'entry_time': order.timestamp or datetime.now(),
                    'stop_loss': order.stop_loss,
                    'take_profit': order.take_profit,
                }
                self.cash -= required_cash
        
        else:  # sell
            # 開空倉或平多倉
            if order.symbol in self.positions and self.positions[order.symbol]['side'] == 'long':
                # 平多倉
                self._close_position(order.symbol, execution_price, order.timestamp or datetime.now())
            else:
                # 開空倉
                self.positions[order.symbol] = {
                    'side': 'short',
                    'units': abs(order.units),
                    'entry_price': execution_price,
                    'entry_time': order.timestamp or datetime.now(),
                    'stop_loss': order.stop_loss,
                    'take_profit': order.take_profit,
                }
                self.cash += abs(order.units) * execution_price - commission
    
    def _close_position(self, symbol: str, exit_price: float, exit_time: datetime):
        """平倉"""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        
        # 計算P&L
        if pos['side'] == 'long':
            pnl = (exit_price - pos['entry_price']) * pos['units']
        else:  # short
            pnl = (pos['entry_price'] - exit_price) * pos['units']
        
        # 扣除手續費
        commission = pos['units'] * exit_price * self.commission_rate
        pnl -= commission
        
        # 記錄交易
        trade = Trade(
            trade_id=f"{symbol}_{len(self.trades)}",
            symbol=symbol,
            entry_time=pos['entry_time'],
            entry_price=pos['entry_price'],
            exit_time=exit_time,
            exit_price=exit_price,
            units=pos['units'],
            side=pos['side'],
            pnl=pnl,
            pnl_percent=(pnl / (pos['entry_price'] * pos['units'])) * 100,
            commission=commission
        )
        self.trades.append(trade)
        
        # 更新現金
        self.cash += pnl
        if pos['side'] == 'long':
            self.cash += pos['units'] * exit_price
        
        # 移除倉位
        del self.positions[symbol]
    
    def _update_equity(self, timestamp: datetime, current_data: Dict[str, pd.Series]):
        """更新權益"""
        positions_value = 0.0
        
        for symbol, pos in self.positions.items():
            if symbol not in current_data:
                continue
            
            current_price = current_data[symbol]['close']
            
            if pos['side'] == 'long':
                unrealized_pnl = (current_price - pos['entry_price']) * pos['units']
            else:  # short
                unrealized_pnl = (pos['entry_price'] - current_price) * pos['units']
            
            positions_value += pos['entry_price'] * pos['units'] + unrealized_pnl
        
        self.equity = self.cash + positions_value
    
    def _generate_result(self) -> 'BacktestResult':
        """生成回測結果"""
        return BacktestResult(
            trades=self.trades,
            equity_curve=pd.DataFrame(self.equity_curve),
            initial_capital=self.initial_capital,
            final_equity=self.equity,
        )


@dataclass
class BacktestResult:
    """回測結果"""
    trades: List[Trade]
    equity_curve: pd.DataFrame
    initial_capital: float
    final_equity: float
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """計算績效指標"""
        if not self.trades:
            return {}
        
        # 基本指標
        total_return = (self.final_equity - self.initial_capital) / self.initial_capital
        total_trades = len(self.trades)
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Sharpe Ratio
        returns = self.equity_curve['equity'].pct_change().dropna()
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Max Drawdown
        cummax = self.equity_curve['equity'].cummax()
        drawdown = (self.equity_curve['equity'] - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        sortino = np.sqrt(252) * returns.mean() / downside_returns.std() if len(downside_returns) > 0 else 0
        
        # Calmar Ratio
        calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'total_return_percent': total_return * 100,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'max_drawdown_percent': max_drawdown * 100,
            'calmar_ratio': calmar,
        }
    
    def plot_equity_curve(self):
        """繪製權益曲線"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve['timestamp'], self.equity_curve['equity'])
        plt.title('Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.show()
```

**使用示例：**

```python
# 創建一個簡單的MA策略進行回測
class MAStrategy:
    def __init__(self, fast_period=10, slow_period=30):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.positions = {}
    
    def on_data(self, timestamp, data):
        orders = []
        
        for symbol, df in data.items():
            # 計算MA
            fast_ma = df['close'].rolling(self.fast_period).mean()
            slow_ma = df['close'].rolling(self.slow_period).mean()
            
            # 生成信號
            if fast_ma.iloc[-1] > slow_ma.iloc[-1] and fast_ma.iloc[-2] <= slow_ma.iloc[-2]:
                # 金叉 - 買入
                if symbol not in self.positions:
                    orders.append(Order(
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        side='buy',
                        units=1000
                    ))
                    self.positions[symbol] = 'long'
            
            elif fast_ma.iloc[-1] < slow_ma.iloc[-1] and fast_ma.iloc[-2] >= slow_ma.iloc[-2]:
                # 死叉 - 賣出
                if symbol in self.positions:
                    orders.append(Order(
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        side='sell',
                        units=1000
                    ))
                    del self.positions[symbol]
        
        return orders

# 運行回測
backtest_data = {
    'EUR_USD': eur_usd_df,
    'GBP_USD': gbp_usd_df,
}

backtester = UniversalBacktester(
    data=backtest_data,
    initial_capital=10000,
    commission_rate=0.0001
)

strategy = MAStrategy(fast_period=10, slow_period=30)
result = backtester.run(strategy)

# 查看結果
metrics = result.calculate_metrics()
print(metrics)
result.plot_equity_curve()
```

---

## 總結

以上方案提供了：

1. **統一管理UI** - 完整的前後端架構
2. **模型註冊系統** - 版本控制和元數據管理
3. **獨立Backtester** - 支持傳統策略的快速驗證

這些改進將使系統達到工業級水準。
