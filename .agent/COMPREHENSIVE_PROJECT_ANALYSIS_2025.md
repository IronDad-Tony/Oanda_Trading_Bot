# OANDA量化交易系統 - 全面專案分析報告 (2025年12月)

## 📋 執行摘要

本報告對 **OANDA AI Trading Bot v2.0** 進行全面深度分析，涵蓋系統架構、參數傳遞、配置項、運作流程，以及與業界標準的差距分析和改進建議。

---

## 🗂️ 專案結構總覽

```
Oanda_Trading_Bot/
├── .agent/                    # AI分析與設計文檔 (12個檔案)
├── .env                       # 環境變數配置
├── configs/                   # 配置文件目錄
│   ├── enhanced_transformer_config.json    # 主要模型配置
│   ├── quantum_strategy_config.json        # 量子策略配置
│   ├── live/
│   │   └── live_config.json               # 實盤交易配置
│   └── training/
│       ├── enhanced_model_config.json     # 詳細模型參數
│       ├── execution_sim_config.json      # 執行模擬配置
│       └── test_strategy_config.json      # 測試策略配置
├── data/                      # 數據目錄
├── docs/                      # 文檔 (28個檔案)
├── frontend/                  # React前端 (開發中)
├── logs/                      # 日誌目錄
├── model_registry/            # 模型倉庫
├── reports/                   # 測試報告 (57個檔案)
├── scripts/                   # 腳本目錄 (31個檔案)
├── src/oanda_trading_bot/     # 主要源代碼
│   ├── common/                # 共用模組
│   ├── training_system/       # 訓練系統 (118個檔案)
│   └── live_trading_system/   # 實盤交易系統 (31個檔案)
├── tests/                     # 測試檔案 (65個檔案)
├── weights/                   # 訓練權重
├── requirements.txt           # Python依賴
└── quick_start.bat           # 快速啟動腳本
```

---

## 📊 第一部分：核心組件詳細分析

### 1.1 訓練系統 (Training System)

#### 📁 數據管理模組
| 模組 | 檔案 | 功能描述 |
|------|------|----------|
| `oanda_downloader.py` | 27KB | OANDA API數據下載，支持多時間週期 |
| `mmap_dataset.py` | 36KB | Memory-Mapped數據集，高效處理大規模數據 |
| `currency_manager.py` | 8KB | 貨幣依賴管理，處理匯率轉換路徑 |
| `database_manager.py` | 25KB | SQLite數據庫管理，歷史數據持久化 |

**參數傳遞流程：**
```
OandaDownloader → SQLite Database → MemoryMappedDataset → TradingEnv
                      ↓
            時間標準化 (ISO8601)
            價格類型: [bid_open, bid_high, bid_low, bid_close,
                      ask_open, ask_high, ask_low, ask_close, volume]
```

#### 📁 交易環境模組
| 模組 | 檔案 | 行數 | 功能描述 |
|------|------|------|----------|
| `trading_env.py` | 81KB | 1366行 | 核心交易環境，符合Gymnasium接口 |
| `progressive_reward_calculator.py` | 58KB | ~1200行 | 三階段漸進式獎勵系統 |
| `enhanced_simulation.py` | 8KB | 交易執行模擬（滑點、延遲） |
| `reward_normalizer.py` | 21KB | 獎勵正規化處理 |

**交易環境參數：**
```python
# 關鍵初始化參數
initial_capital: float = 100000.0          # 初始資金
max_position_size_percentage: float = 5.0  # 最大持倉比例
account_currency: str = "AUD"              # 帳戶貨幣
leverage: float = 20.0                     # 槓桿倍數
timesteps_history: int = 128               # 歷史時間步數

# 執行模擬參數 (execution_sim_config.json)
simulate_order_latency: bool = true        # 模擬訂單延遲
execution_delay_mean_steps: int = 2        # 平均延遲步數
execution_delay_jitter_steps: int = 1      # 延遲抖動
max_slippage_pips: float = 0.8            # 最大滑點 (pips)
slippage_sigma_pips: float = 0.3          # 滑點標準差
reject_if_slippage_exceeds_bound: bool = true  # 超滑點拒單
```

#### 📁 模型架構模組
| 模組 | 檔案 | 功能描述 |
|------|------|----------|
| `enhanced_transformer.py` | 75KB | 核心Transformer模型，支持Fourier/Wavelet特徵 |
| `sac_agent_wrapper.py` | 49KB | SAC智能體封裝，整合量子策略層 |
| `sac_policy.py` | 45KB | 自定義SAC策略網路 |
| `enhanced_feature_extractor.py` | 28KB | 特徵提取器 |

**模型配置參數 (enhanced_model_config.json)：**
```json
{
  "hidden_dim": 1024,
  "num_layers": 24,
  "num_heads": 16,
  "intermediate_dim": 4096,
  "dropout_rate": 0.1,
  "max_sequence_length": 512,
  "use_fourier_features": false,
  "use_wavelet_features": false,
  "use_msfe": true,                      // Multi-Scale Feature Extraction
  "msfe_scales": [3, 5, 7, 11],
  "use_cts_fusion": true,                // Cross-Time-Scale Fusion
  "use_graph_attn": true,                // Graph Attention
  "use_regularization": true,
  "regularization_params": {
    "l2_lambda": 0.0001,
    "norm_clip_value": 1.0
  }
}
```

### 1.2 實盤交易系統 (Live Trading System)

#### 📁 核心模組
| 模組 | 檔案 | 功能描述 |
|------|------|----------|
| `oanda_client.py` | 13KB | OANDA V20 API客戶端封裝 |
| `risk_manager.py` | 25KB | 風險管理器 |
| `order_manager.py` | 27KB | 訂單管理器 |
| `trading_logic.py` | 13KB | 交易邏輯引擎 |

**實盤配置參數 (live_config.json)：**
```json
{
  "trading_instruments": ["EUR_USD", "GBP_USD", "XAU_USD", "NAS100_USD", "US30_USD"],
  "trading_granularity": "S5",
  "model_lookback_window": 512,
  "max_position_size_per_trade": 1000,
  "max_total_risk_pct": 2.0,
  "daily_loss_limit_pct": 5.0,
  "account_currency": "AUD",
  "trading_loop_interval_seconds": 5,
  "risk_management": {
    "max_total_exposure_usd": 200,
    "max_risk_per_trade_percent": 0.1,
    "use_atr_sizing": true,
    "atr_period": 14,
    "stop_loss_atr_multiplier": 2.0,
    "take_profit_atr_multiplier": 3.0
  },
  "execution": {
    "per_symbol_sequencing": true,
    "order_fill_timeout_seconds": 3,
    "min_interval_between_orders_ms": 250,
    "use_price_bound": true,
    "max_slippage_pips": 1.0,
    "use_transactions_stream": true
  }
}
```

---

## 📊 第二部分：參數傳遞與配置系統

### 2.1 配置層級架構

```
Level 1: 環境變數 (.env)
    ├─ OANDA_API_KEY
    ├─ OANDA_ACCOUNT_ID
    ├─ OANDA_ENVIRONMENT (practice/live)
    └─ INITIAL_CAPITAL

Level 2: 全局配置 (common/config.py)
    ├─ ACCOUNT_CURRENCY (動態從API獲取)
    ├─ DEVICE (cuda/cpu 自動偵測)
    ├─ USE_AMP (混合精度訓練)
    ├─ SAC超參數 (學習率、batch_size等)
    └─ 路徑配置 (DATABASE_DIR, WEIGHTS_DIR等)

Level 3: 訓練配置 (configs/training/*.json)
    ├─ 模型架構參數
    ├─ 訓練循環參數
    └─ 漸進式學習配置

Level 4: 實盤配置 (configs/live/live_config.json)
    ├─ 交易品種設定
    ├─ 風險管理參數
    └─ 執行參數
```

### 2.2 核心參數一覽表

#### 訓練相關參數
| 參數名稱 | 來源 | 預設值 | 說明 |
|----------|------|--------|------|
| `INITIAL_CAPITAL` | config.py | 100,000 | 訓練初始資金 |
| `SAC_LEARNING_RATE` | config.py | 0.0003 | SAC學習率 |
| `SAC_BATCH_SIZE` | config.py | 64 | 批次大小 |
| `SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR` | config.py | 5000 | 每個品種的緩衝區因子 |
| `SAC_GAMMA` | config.py | 0.99 | 折扣因子 |
| `TIMESTEPS` | config.py | 100,000 | 訓練總步數 |
| `MAX_SYMBOLS_ALLOWED` | config.py | 10 | 最大交易品種數 |

#### 風險管理參數
| 參數名稱 | 來源 | 預設值 | 說明 |
|----------|------|--------|------|
| `max_risk_per_trade_percent` | live_config | 0.1% | 單筆交易風險 |
| `max_total_exposure_usd` | live_config | $200 | 總敞口限制 |
| `stop_loss_atr_multiplier` | live_config | 2.0 | ATR止損倍數 |
| `take_profit_atr_multiplier` | live_config | 3.0 | ATR止盈倍數 |
| `daily_loss_limit_pct` | live_config | 5.0% | 日損限制 |

---

## 📊 第三部分：運作流程分析

### 3.1 訓練流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING WORKFLOW                              │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. DATA PREPARATION                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │ OANDA API   │───►│ SQLite DB   │───►│ MemoryMappedDataset     │  │
│  │ Downloader  │    │ (S5 Candles)│    │ (Preprocessed Features) │  │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. ENVIRONMENT SETUP                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ UniversalTradingEnvV4                                           ││
│  │  ├─ InstrumentInfoManager (margin rates, pip location)         ││
│  │  ├─ CurrencyDependencyManager (exchange rates)                 ││
│  │  ├─ ProgressiveRewardCalculator (3-stage rewards)              ││
│  │  └─ ExecutionSimulator (slippage, latency)                     ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. AGENT & MODEL                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ QuantumEnhancedSAC                                              ││
│  │  ├─ CustomSACPolicy                                             ││
│  │  │   └─ EnhancedTransformerFeatureExtractor                     ││
│  │  │       ├─ MultiScaleFeatureExtractor (Conv1D: 3,5,7,11)       ││
│  │  │       ├─ FourierFeatureBlock (frequency domain)              ││
│  │  │       ├─ WaveletFeatureBlock (time-frequency)                ││
│  │  │       └─ CrossAssetAttention (inter-symbol relations)        ││
│  │  └─ QuantumStrategyLayer (策略選擇)                              ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. TRAINING LOOP                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ for step in total_timesteps:                                    ││
│  │   obs = env.reset() or env.step(action)                         ││
│  │   action = agent.predict(obs)                                   ││
│  │   reward = progressive_reward_calculator.calculate()             ││
│  │   agent.learn()                                                 ││
│  │   if step % eval_freq == 0: evaluate()                         ││
│  │   if step % save_freq == 0: save_checkpoint()                  ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. MODEL SAVING                                                      │
│  ┌─────────────┐    ┌─────────────────────┐                         │
│  │ Model.zip   │───►│ ModelRegistry       │                         │
│  │ (weights)   │    │ (metadata + version)│                         │
│  └─────────────┘    └─────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 實盤交易流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                      LIVE TRADING WORKFLOW                            │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. SYSTEM INITIALIZATION                                             │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ initialize_system()                                              ││
│  │  ├─ Load .env (API credentials)                                 ││
│  │  ├─ Resolve AccountProfile (currency, environment)              ││
│  │  ├─ Initialize OandaClient                                      ││
│  │  ├─ Initialize SystemState                                      ││
│  │  ├─ Initialize DatabaseManager                                  ││
│  │  ├─ Initialize InstrumentMonitor                                ││
│  │  ├─ Initialize LivePreprocessor                                 ││
│  │  ├─ Initialize PredictionService + Load Model                   ││
│  │  ├─ Initialize PositionManager                                  ││
│  │  ├─ Initialize RiskManager                                      ││
│  │  ├─ Initialize OrderManager                                     ││
│  │  └─ Initialize TradingLogic                                     ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. TRADING LOOP (每5秒)                                              │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ async_trading_loop():                                           ││
│  │   ├─ trading_logic.execute_trade_cycle()                        ││
│  │   │   ├─ Fetch latest candles                                   ││
│  │   │   ├─ Preprocess data                                        ││
│  │   │   ├─ Generate prediction (model inference)                  ││
│  │   │   ├─ Risk assessment                                        ││
│  │   │   └─ Execute orders                                         ││
│  │   └─ broadcaster.broadcast_updates()                            ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. ORDER EXECUTION FLOW                                              │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ OrderManager.execute_signal():                                  ││
│  │  ├─ Validate tradability (market hours, halted)                ││
│  │  ├─ Risk check (exposure limits, drawdown)                      ││
│  │  ├─ Calculate position size (ATR-based)                         ││
│  │  ├─ Set stop-loss and take-profit                              ││
│  │  ├─ Execute via OandaClient.create_order_v2()                  ││
│  │  └─ Record trade to database                                    ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 第四部分：與業界標準的差距分析

### 4.1 業界標準量化交易系統功能對比

| 功能類別 | 業界標準 | 本系統現狀 | 差距評估 |
|----------|----------|-----------|----------|
| **數據處理** | | | |
| 多時間週期數據 | ✅ 必備 | ⚠️ 部分支援 | 需增強MTF融合 |
| 數據品質檢查 | ✅ 必備 | ❌ 缺失 | 需新增 |
| 異常值處理 | ✅ 必備 | ⚠️ 基礎 | 需增強 |
| **特徵工程** | | | |
| 技術指標庫 | ✅ 必備 | ✅ 有 | 良好 |
| 因子標準化 | ✅ 必備 | ✅ 有 | 良好 |
| 特徵選擇/重要性 | ✅ 推薦 | ❌ 缺失 | 需新增 |
| **模型設計** | | | |
| 模型架構 | 多樣化 | ✅ Transformer+SAC | 先進 |
| 模型壓縮/量化 | ✅ 推薦 | ❌ 缺失 | 需新增 |
| 模型解釋性 | ✅ 推薦 | ⚠️ 有但未完整 | 需增強 |
| **模型訓練** | | | |
| 分布式訓練 | ✅ 推薦 | ❌ 缺失 | 可選 |
| 超參數優化 | ✅ 必備 | ❌ 缺失 | 需新增 |
| 實驗追蹤 | ✅ 必備 | ⚠️ 基礎 | 需增強 |
| **模型部署** | | | |
| 模型版本控制 | ✅ 必備 | ✅ 有ModelRegistry | 良好 |
| A/B測試 | ✅ 推薦 | ❌ 缺失 | 需新增 |
| 熱更新/無停機更新 | ✅ 推薦 | ❌ 缺失 | 需新增 |
| **預熱系統** | | | |
| 模型預加載 | ✅ 必備 | ❌ 缺失 | 需新增 |
| 推理預熱 | ✅ 必備 | ❌ 缺失 | 需新增 |
| 連接池預熱 | ✅ 推薦 | ❌ 缺失 | 需新增 |
| **實時推論** | | | |
| 低延遲推論 | ✅ 必備 | ⚠️ 未優化 | 需優化 |
| 批次推論 | ✅ 推薦 | ❌ 缺失 | 可選 |
| GPU推論加速 | ✅ 推薦 | ✅ 支援 | 良好 |
| **風險控管** | | | |
| 單筆風險控制 | ✅ 必備 | ✅ 有 | 良好 |
| 組合風險 (VaR/CVaR) | ✅ 必備 | ❌ 缺失 | 需新增 |
| 動態風險調整 | ✅ 推薦 | ⚠️ 基礎 | 需增強 |
| 斷路器機制 | ✅ 必備 | ✅ 有 | 良好 |
| **Dashboard** | | | |
| 實時監控 | ✅ 必備 | ⚠️ 分散 | 需整合 |
| 歷史分析 | ✅ 必備 | ⚠️ 基礎 | 需增強 |
| 告警系統 | ✅ 必備 | ❌ 缺失 | 需新增 |

### 4.2 關鍵缺失項詳細分析

#### 🔴 高優先級缺失

1. **預熱系統 (Warm-up System)**
   - 問題：模型首次載入慢，可能錯過交易機會
   - 解決方案：實現模型預載入、GPU記憶體預分配、JIT編譯預熱

2. **組合風險管理 (Portfolio Risk)**
   - 問題：只有單一品種風控，缺乏系統性風險評估
   - 解決方案：實現VaR/CVaR計算、相關性矩陣、風險預算

3. **告警系統 (Alert System)**
   - 問題：無法即時通知異常情況
   - 解決方案：整合Telegram/Email/Webhook告警

4. **統一Dashboard**
   - 問題：訓練UI(Streamlit)和實盤UI(React)分離
   - 解決方案：建立統一的專業級交易終端

#### 🟡 中優先級缺失

5. **超參數優化 (HPO)**
   - 問題：手動調參效率低
   - 解決方案：整合Optuna/Ray Tune

6. **實驗追蹤**
   - 問題：訓練實驗難以對比
   - 解決方案：整合MLflow/Weights & Biases

7. **數據品質檢查**
   - 問題：可能使用異常數據訓練
   - 解決方案：實現數據驗證pipeline

---

## 📊 第五部分：模擬真實交易的完整性分析

### 5.1 已實現的模擬機制

| 模擬項目 | 實現狀態 | 位置 | 說明 |
|----------|----------|------|------|
| 滑點 (Slippage) | ✅ 已實現 | `execution_sim_config.json` | 正態分佈滑點，max 0.8 pips |
| 執行延遲 | ✅ 已實現 | `execution_sim_config.json` | 平均2步延遲 |
| 保證金計算 | ✅ 已實現 | `trading_env.py` | 使用真實margin rate |
| 強制平倉 | ✅ 已實現 | `trading_env.py` | 50% margin closeout |
| 貨幣轉換 | ✅ 已實現 | `CurrencyDependencyManager` | 自動路徑規劃 |
| 點差 (Spread) | ✅ 已實現 | `trading_env.py` | 使用bid/ask價格 |

### 5.2 缺失的模擬機制

| 模擬項目 | 業界需求 | 當前狀態 | 影響程度 |
|----------|----------|----------|----------|
| 隔夜利息 (Swap) | ✅ 必備 | ❌ 缺失 | 高 - 影響長期持倉策略 |
| 市場休市 | ✅ 必備 | ❌ 缺失 | 高 - 週末跳空風險 |
| 流動性變化 | ✅ 推薦 | ❌ 缺失 | 中 - 重大事件期間 |
| 訂單拒絕率 | ✅ 推薦 | ⚠️ 部分 | 低 - 有價格邊界檢查 |
| 部分成交 | ✅ 推薦 | ❌ 缺失 | 低 - 主要影響大單 |

---

## 📊 第六部分：報告與測試腳本清點

### 6.1 需要清理的舊報告 (reports/)

**建議刪除的檔案：** (共35個舊報告)
```
reports/phase_tests/                # 22個舊的階段測試指標 (2025年8月)
reports/live_small_run_20250906_*.md       # 舊的實盤測試
reports/test_run_20250906_*.md             # 舊的測試運行
reports/smoke_test_20251125_*.md           # 可保留最新的
reports/*.md                        # 保留有意義的分析報告
reports/*.json                      # 刪除舊的JSON報告
```

### 6.2 需要整理的測試腳本 (scripts/)

**現有腳本結構：**
```
scripts/
├── archive/       # 6個歸檔腳本 (可保留作參考)
├── live_trading/  # 7個實盤相關腳本
├── setup/         # 5個環境設置腳本
├── testing/       # 6個測試腳本
└── training/      # 6個訓練腳本
```

**建議整合：**
- 合併冗餘腳本
- 標準化命名規範
- 添加統一的入口腳本

---

## 📋 現狀總結

### ✅ 優勢

1. **模組化架構設計優秀** - 訓練系統和實盤系統完全解耦
2. **創新的三階段漸進式獎勵系統** - 符合人類學習曲線
3. **高度擬真的交易環境模擬** - 包含滑點、延遲、保證金
4. **專業的風險管理機制** - ATR動態止損、敞口限制
5. **完整的貨幣轉換** - 自動處理多幣種
6. **先進的模型架構** - Transformer + Fourier/Wavelet特徵

### ⚠️ 待改進

1. **UI分散** - Streamlit訓練UI + React前端未整合
2. **缺少預熱系統** - 模型載入未優化
3. **缺少組合風險管理** - 無VaR/CVaR
4. **缺少告警系統** - 無即時通知
5. **缺少超參數優化** - 手動調參
6. **缺少完整的Smoke測試** - 測試覆蓋不足

---

*報告生成時間：2025-12-05*
*版本：v2.0 分析報告*
