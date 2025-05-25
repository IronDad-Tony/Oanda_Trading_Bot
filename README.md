# 🚀 OANDA 通用自動交易模型訓練系統

一個基於強化學習的通用量化交易模型，使用先進的Transformer架構和頻域分析技術，能夠同時操作多個交易品種。

## ✨ 主要特點

### 🧠 先進的AI架構
- **Transformer模型**: 處理時序數據的最新架構
- **頻域分析**: 傅立葉變換捕捉週期性模式
- **小波分析**: 多尺度時頻分析
- **跨資產注意力**: 學習不同品種間的關聯性
- **SAC強化學習**: 連續動作空間的最優策略

### 📊 智能數據管理
- **自動數據下載**: 從OANDA API獲取歷史數據
- **智能貨幣管理**: 自動分析並下載所需匯率對
- **高效存儲**: SQLite數據庫 + 內存映射文件
- **實時預處理**: 動態特徵工程和標準化

### 🏦 真實交易模擬
- **多資產支持**: 最多20個交易品種同時操作
- **保證金模擬**: 完全模擬OANDA V20帳戶規則
- **貨幣轉換**: 精確的跨貨幣計算
- **風險管理**: ATR止損、保證金水平監控

### 🔧 完整訓練流程
- **自動化流程**: 一鍵完成數據準備到模型訓練
- **檢查點保存**: 定期保存，支持斷點續練
- **實時監控**: TensorBoard可視化
- **早停機制**: 防止過擬合

## 🚀 快速開始

### 1. 環境準備

```bash
# 安裝依賴
pip install -r requirements.txt

# 設置OANDA API密鑰
# 創建 .env 文件並添加：
OANDA_API_KEY=your_api_key_here
OANDA_ACCOUNT_ID=your_account_id_here
```

### 2. 開始訓練

```bash
# 使用預設配置開始訓練
python start_training.py

# 或者運行系統測試
python test_system.py
```

### 3. 監控訓練進度

```bash
# 啟動TensorBoard
tensorboard --logdir=logs/

# 在瀏覽器中打開 http://localhost:6006
```

## 📁 項目結構

```
Oanda_Trading_Bot/
├── src/
│   ├── common/              # 通用配置和日誌
│   ├── data_manager/        # 數據管理模組
│   ├── environment/         # 交易環境
│   ├── models/             # AI模型架構
│   ├── agent/              # 強化學習智能體
│   └── trainer/            # 訓練流程
├── data/                   # 數據存儲
├── logs/                   # 訓練日誌和模型
├── weights/               # 模型權重
├── start_training.py      # 主訓練腳本
├── test_system.py         # 系統測試
└── README.md             # 本文件
```

## ⚙️ 配置說明

主要配置在 `src/common/config.py` 中：

```python
# 交易配置
MAX_SYMBOLS_ALLOWED = 20        # 最大交易品種數
INITIAL_CAPITAL = 10000         # 初始資金
ACCOUNT_CURRENCY = "AUD"        # 帳戶貨幣

# 模型配置
TIMESTEPS = 128                 # 歷史時間步數
TRANSFORMER_MODEL_DIM = 512     # Transformer維度
TRANSFORMER_NUM_HEADS = 8       # 注意力頭數

# 訓練配置
LEARNING_RATE = 3e-4           # 學習率
BUFFER_SIZE_MULTIPLIER = 64000  # 經驗回放緩衝區
```

## 🎯 使用示例

### 基本訓練

```python
from src.trainer.enhanced_trainer import EnhancedUniversalTrainer
from datetime import datetime, timezone, timedelta

# 設置交易品種
symbols = ["EUR_USD", "USD_JPY", "GBP_USD"]

# 設置時間範圍（最近30天）
end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(days=30)

# 創建訓練器
trainer = EnhancedUniversalTrainer(
    trading_symbols=symbols,
    start_time=start_time,
    end_time=end_time,
    total_timesteps=50000
)

# 執行訓練
trainer.run_full_training_pipeline()
```

### 斷點續練

```python
# 從已保存的模型繼續訓練
trainer.run_full_training_pipeline(
    load_model_path="logs/sac_universal_trader_latest.zip"
)
```

## 📊 監控指標

系統會記錄以下關鍵指標：

- **訓練獎勵**: 每步的獎勵值
- **投資組合價值**: 帳戶總淨值變化
- **保證金水平**: 風險管理指標
- **交易頻率**: 模型交易活躍度
- **Transformer範數**: 模型複雜度監控

## 🔧 高級功能

### 自定義交易品種

```python
# 支持所有OANDA可交易品種
custom_symbols = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
    "USD_CAD", "USD_CHF", "NZD_USD", "EUR_GBP",
    "EUR_JPY", "GBP_JPY", "AUD_JPY", "XAU_USD"
]
```

### 調整訓練參數

```python
trainer = EnhancedUniversalTrainer(
    trading_symbols=symbols,
    start_time=start_time,
    end_time=end_time,
    total_timesteps=100000,     # 更長訓練
    save_freq=1000,             # 更頻繁保存
    eval_freq=2000,             # 更頻繁評估
)
```

## 🚨 注意事項

1. **API限制**: OANDA API有請求頻率限制，大量數據下載可能需要時間
2. **計算資源**: Transformer模型需要較多GPU/CPU資源
3. **數據存儲**: 長期歷史數據會佔用較多磁盤空間
4. **風險警告**: 這是研究用途的模型，實盤交易需要充分測試

## 🛠️ 故障排除

### 常見問題

1. **導入錯誤**: 確保Python路徑正確，運行 `python test_system.py` 檢查
2. **API錯誤**: 檢查 `.env` 文件中的API密鑰是否正確
3. **內存不足**: 減少 `TIMESTEPS` 或 `MAX_SYMBOLS_ALLOWED`
4. **訓練緩慢**: 考慮使用GPU或減少模型複雜度

### 日誌檢查

```bash
# 查看詳細日誌
tail -f logs/trading_system.log

# 檢查錯誤
grep ERROR logs/trading_system.log
```

## 🔮 未來計劃

- [ ] 實盤交易接口
- [ ] 更多技術指標集成
- [ ] 多時間框架分析
- [ ] 風險管理優化
- [ ] Web界面開發
- [ ] 回測引擎完善

## 📞 支持

如有問題或建議，請查看日誌文件或檢查配置設置。

---

**⚠️ 風險提示**: 本系統僅供研究和學習使用。任何實盤交易都存在資金損失風險，請謹慎使用。