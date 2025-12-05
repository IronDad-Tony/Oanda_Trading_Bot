# OANDA Trading Bot - 升級實施計劃 (最終版)

## 📊 實施進度總覽

| 項目 | 狀態 | 完成日期 | 優先級 |
|------|------|----------|--------|
| **Phase 1: Core Infrastructure** | | | |
| WarmupManager | ✅ 完成 | 2025-12-05 | 高 |
| PortfolioRiskManager | ✅ 完成 | 2025-12-05 | 高 |
| AlertManager | ✅ 完成 | 2025-12-05 | 高 |
| GPUOptimizer | ✅ 完成 | 2025-12-05 | 高 |
| **Phase 2: Data Management** | | | |
| DataCacheManager | ✅ 完成 | 2025-12-05 | 高 |
| DataQualityChecker | ✅ 完成 | 2025-12-05 | 中 |
| SwapCalculator | ✅ 完成 | 2025-12-05 | 中 |
| MarketHoursManager | ✅ 完成 | 2025-12-05 | 中 |
| **Phase 3: Optimization** | | | |
| HyperparameterOptimizer | ✅ 完成 | 2025-12-05 | 中 |
| CUDA Diagnostics | ✅ 完成 | 2025-12-05 | 中 |
| **Phase 4: UI/Dashboard** | | | |
| UI系統分析報告 | ✅ 完成 | 2025-12-05 | 中 |
| ProfessionalDashboard | ✅ 設計完成 | 2025-12-05 | 中 |
| **Testing** | | | |
| Complete Smoke Test | ✅ 完成 | 2025-12-05 | 高 |
| 配置常量修復 | ✅ 完成 | 2025-12-05 | 高 |

---

## 💾 數據管理策略

### 推薦方案: 智能緩存策略

```
┌─────────────────────────────────────────────────────────────────┐
│                    數據管理流程                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  訓練啟動時:                                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 1. 清理不完整/損壞的緩存數據                                    ││
│  │ 2. 檢查本地緩存是否有需要的時間範圍                              ││
│  │ 3. 只下載缺失的增量數據                                        ││
│  │ 4. 驗證數據完整性                                             ││
│  │ 5. 開始訓練                                                   ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  訓練結束時:                                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 1. 標記數據為「完整」                                          ││
│  │ 2. 更新緩存元數據                                             ││
│  │ 3. 執行過期數據清理 (保留30天內)                                ││
│  │ 4. 強制執行大小限制 (最大10GB)                                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### DataCacheManager 配置

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `max_cache_days` | 30 | 最多保留30天的數據 |
| `max_cache_size_gb` | 10.0 | 最大緩存 10GB |
| `auto_clean_on_startup` | True | 啟動時自動清理 |
| `clean_incomplete_data` | True | 清理不完整的數據 |
| `incremental_download` | True | 增量下載 |

### 使用方式

```python
from oanda_trading_bot.training_system.data_manager.data_cache_manager import get_cache_manager

# 獲取緩存管理器
cache = get_cache_manager()

# 啟動清理
cache.cleanup_startup()

# 檢查缺失範圍
missing = cache.find_missing_ranges(
    symbol="EUR_USD",
    granularity="S5", 
    requested_start=start_dt,
    requested_end=end_dt
)

# 只下載缺失的數據
for start, end in missing:
    download_data(symbol, start, end)
```

---

## 🎮 RTX 5090 優化配置

### 診斷結果
| 項目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 5090 |
| 顯存 | 34.2 GB |
| 架構 | Blackwell (sm_120) |
| PyTorch | 2.8.0+cu129 |
| cuDNN | 9.1.0.02 |
| FP16 TFLOPS | 194.5 |

### 優化配置
```python
{
    "device": "cuda",
    "use_amp": True,
    "batch_size": 256,        # 大批次
    "hidden_dim": 1024,
    "num_layers": 24,
    "buffer_size_per_symbol": 20000,
    "compile_model": True     # torch.compile
}
```

### 預估性能
- 訓練速度: **434 steps/s**
- 樣本吞吐: **27,806 samples/s**
- 每小時步數: **156萬**

---

## 🖥️ UI系統分析

### 現狀
- **Streamlit** (訓練UI): 2,210 行，功能完整但樣式基礎
- **React** (實盤UI): 7個組件，但使用Mock數據
- **問題**: UI分離、無實時連接、API重複

### 解決方案
已創建 `ProfessionalDashboard.tsx`:
- 統一的專業級交易終端
- 實時 WebSocket 連接
- 響應式設計
- 深色主題優化
- 整合訓練、實盤、模型管理

---

## 📁 新增模組總覽

### 核心基礎設施
| 模組 | 路徑 | 說明 |
|------|------|------|
| WarmupManager | `live_trading_system/core/warmup_manager.py` | 模型預熱 |
| AlertManager | `live_trading_system/core/alert_manager.py` | 多渠道告警 |
| PortfolioRiskManager | `live_trading_system/trading/portfolio_risk_manager.py` | VaR/CVaR |
| GPUOptimizer | `training_system/common/gpu_optimizer.py` | GPU自動優化 |

### 數據管理
| 模組 | 路徑 | 說明 |
|------|------|------|
| DataCacheManager | `training_system/data_manager/data_cache_manager.py` | 智能緩存 |
| DataQualityChecker | `training_system/data_manager/data_quality_checker.py` | 數據品質 |

### 環境模擬
| 模組 | 路徑 | 說明 |
|------|------|------|
| SwapCalculator | `training_system/environment/swap_calculator.py` | 隔夜利息 |
| MarketHoursManager | `training_system/environment/market_hours_manager.py` | 市場休市 |

### 訓練優化
| 模組 | 路徑 | 說明 |
|------|------|------|
| HyperparameterOptimizer | `training_system/trainer/hyperparameter_optimizer.py` | Optuna HPO |

### UI
| 模組 | 路徑 | 說明 |
|------|------|------|
| ProfessionalDashboard | `frontend/src/components/ProfessionalDashboard.tsx` | 專業級Dashboard |

### 測試
| 模組 | 路徑 | 說明 |
|------|------|------|
| cuda_diagnostics.py | `scripts/testing/cuda_diagnostics.py` | GPU完整診斷 |
| complete_smoke_test.py | `scripts/testing/complete_smoke_test.py` | 系統煙霧測試 |

---

## 📝 待辦事項

### 高優先級
- [ ] 整合 DataCacheManager 到訓練流程
- [ ] 連接 React 前端到 WebSocket
- [ ] 實現 TradingView 圖表組件

### 中優先級
- [ ] 合併 API 層 (main.py + server.py)
- [ ] 添加更多告警規則
- [ ] 實現模型版本對比功能

### 低優先級
- [ ] 添加多語言支持
- [ ] 移動端適配
- [ ] 分布式訓練支持

---

*最後更新: 2025-12-05 09:20*
