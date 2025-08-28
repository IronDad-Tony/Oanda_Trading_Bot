# 通用模型重構與驗證計畫（Transformer + SAC + 量子策略池）

目標
- 建立在 MAX_SYMBOLS_ALLOWED 下的通用多資產交易模型，觀察空間與動作空間固定，透過 padding + masking 維持通用性。
- Transformer 擔任「超級分析師」：跨資產注意力抽取高維特徵；SAC 作為大腦透過強化學習整合顧問團（量子策略池）。
- 顧問團（策略池）完全模組化、可插拔；ML 類策略消耗 Transformer 輸出，高維抽象特徵；傳統策略消耗預處理後的原始價量特徵。
- 端到端可訓練：SAC 前向計算與 ESS（Enhanced Strategy Superposition）層對 Transformer、ML 策略層回傳梯度。
- 單一 MAX_SYMBOLS_ALLOWED 對應單一模型家族（模型名稱包含 MS 值）；更動後自動建立新家族，避免硬編碼。

設計總覽
- 固定維度：
  - 觀察空間：Dict，含 `market_features [B,N,F]`、`features_from_dataset [B,N,T,F_raw]`、`context_features [B,N,C]`、`symbol_id [B,N]`、`padding_mask [B,N]`；`N = MAX_SYMBOLS_ALLOWED`。
  - 動作空間：Box `[-1,1]^N`，對 dummy slots 由環境忽略（mask）。
- Transformer（EnhancedTransformerFeatureExtractor）：
  - 輸入：`market_features` 與 `symbol_id`（具 padding_idx 的 embedding）；使用 `src_key_padding_mask` 忽略 dummy symbols。
  - 輸出：
    - `per_symbol_outputs [B,N,D]`（每 symbol 的高維特徵）。
    - `pooled [B,D]` 以 masked mean 池化作為投組級特徵（提供 ESS 的注意力/加權控制訊號）。
- 顧問團（EnhancedStrategySuperposition, ESS）：
  - 傳統策略：輸入 `raw [B,N,T,F_raw]` 中各資產序列。
  - ML 策略：輸入 `transformer_per_symbol [B,N,D]` 的各資產向量（以 `seq_len=1` 包裝為 [B,1,D]）。
  - 權重機制：Gumbel-Softmax/Softmax，支援自適應偏置與注意力（以 `pooled [B,D]` 作為 state）。
  - 端到端梯度：策略前向與 ESS 組合損失對 Transformer、策略參數反向傳遞。
- SAC（CustomSACPolicy）：
  - 特徵抽取器：改為 EnhancedTransformerFeatureExtractor（帶 masking 與符號 embedding）。
  - Actor `get_action_dist_params`：
    1) 以 FE 取得 `pooled` 作為 `market_state_features`；
    2) 直接調用 FE 內部 Transformer 取得 `per_symbol_outputs`；
    3) 將 `raw [B,N,T,F_raw]` 與 `per_symbol_outputs [B,N,D]` 同時交給 ESS 做加權融合得到 `mean_actions [B,N]`；
    4) 返回 `mean, log_std`。

資料流與特徵路由
- Padding/Masking：
  - `env.padding_mask` 為 1 表示 active, 0 表示 dummy；Transformer 內部使用相反的 `src_key_padding_mask`（True=padding），靈活忽略 dummy。
  - SAC 執行動作時，環境僅對 active slots 生效；dummy 動作被忽略。
- ML 策略與傳統策略特徵分配：
  - ML 策略（ml_strategies.py 下的 ReinforcementLearningStrategy, EnsembleLearningStrategy, TransferLearningStrategy）：使用 Transformer `per_symbol_outputs`。
  - 傳統策略（trend/stat-arb/risk/other）：使用原始預處理 `features_from_dataset`。
  - ESS 於初始化時為 ML 與傳統策略分別配置 input_dim（`ml_strategy_input_dim = D`、`classical_strategy_input_dim = F_raw`）。

命名與通用性
- 模型命名包含 `MS{MAX_SYMBOLS_ALLOWED}` 與粒度標記（如 `S5`），例如：`sac_universal_MS10_S5_...`。
- 若更改 `MAX_SYMBOLS_ALLOWED`，自動建立新模型家族（觀察/動作空間變更）。

實務依據（高層指引）
- 多資產視作 token 的序列模型：以 Transformer 跨資產注意力，對 dummy 以 key padding mask 處理，保持固定序列長度。
- 混合專家（MoE）/顧問團式路由：以 learned gating（Softmax 或 Gumbel-Softmax）融合多策略輸出，並可由投組級狀態引導權重。
- 強化學習整合多模塊：將顧問團作為 actor 的一部分，讓梯度能對齊策略與特徵抽取器參數（端到端學習）。

落地變更（代辦）
1) SAC 改用增強型 FE（EnhancedTransformerFeatureExtractor）
   - [ ] 切換預設 `features_extractor_class`
   - [ ] 以 `configs/training/enhanced_model_config.py: ModelConfig` 餵入 FE（避免 JSON 結構不符）
   - [ ] 確認 `symbol_id`、`padding_id`、mask 傳遞正確

2) ESS 維度與路由強化
   - [ ] 在 `CustomSACPolicy.__init__` 依 `observation_space` 推導：`F_raw` 與 `D`，並注入到 `ess_config`
   - [ ] `EnhancedStrategySuperposition` 增加 `ml_strategy_input_dim`、`classical_strategy_input_dim` 參數
   - [ ] 初始化每個策略時自動分派 input_dim（ML -> D，傳統 -> F_raw）
   - [ ] `forward` 支援 `transformer_per_symbol_features_batch`，並為 ML/傳統策略選擇對應特徵

3) 策略設定檔與預設
   - [ ] 新增 `configs/quantum_strategy_config.json`（包含策略名單與可選參數）
   - [ ] 若現有設定缺失，仍可從動態註冊載入策略（登記在 strategies/__init__.py）

4) 驗證腳本（可本地 smoke 測試即可）
   - [ ] 新增 `scripts/validate_feature_pipeline.py`：
        - 構造與環境一致的 `Dict` 觀察空間與張量
        - 建立 `CustomSACPolicy`（使用增強 FE + ESS）
        - 執行一次 `get_action_dist_params` 檢查維度、面罩與路由是否正常

5) 文件更新與清理
   - [ ] 在 README/PLAN 中補上資料流圖與關鍵 API 說明
   - [ ] 若 `configs/*` 中有過時 JSON，遷移/合併至新檔後移除（保守執行，避免破壞相依）

測試要點（Smoke）
- FE：`pooled.shape == [B,D]`、遮罩後 dummy 對輸出影響弱化。
- ESS：
  - `asset_features_batch.shape == [B,N,T,F_raw]`
  - `transformer_per_symbol_features_batch.shape == [B,N,D]`
  - 動作輸出 `mean.shape == [B,N]`、`log_std.shape == [B,N]`
- 兩種策略群的 input_dim 設定正確（ML=D，傳統=F_raw）。

風險與回退
- 若增強 FE 引發相容性問題，可暫時 fallback 舊 `TransformerFeatureExtractor`，但保留 ESS 路由改造。
- 如策略維度不匹配，ESS 會以零輸出容錯並記錄警示；但我們將在初始化階段即校正 input_dim 以降低風險。

里程碑
- M1：切換 FE + ESS 維度注入完成（此 PR）
- M2：新增/調整策略設定檔 + 驗證腳本
- M3：文件更新 + 清理過時檔案

變更記錄
- v1: 初版計畫與落地待辦，開始實作。

