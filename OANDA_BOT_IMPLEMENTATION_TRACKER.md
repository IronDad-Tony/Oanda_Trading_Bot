# Oanda Trading Bot - 實施進度追蹤

本文件旨在追蹤 Oanda Trading Bot 根據《模型增強實施藍圖》的開發進度。

## 總體結構與組織

- [x] **標準化專案結構**
    - [x] 創建 `scripts/` 目錄，並將現有的訓練腳本 (如 `train_universal_trader_rewritten.py`, `train_universal_trader.py`) 遷移至此，並可能重命名為 `training_pipeline.py`。
    - [x] 創建頂層 `tests/` 目錄。
    - [x] 在 `tests/` 下創建 `unit_tests/`, `integration_tests/`, `performance_tests/` 子目錄。
    - [x] 評估並將現有的根目錄測試檔案 (如 `comprehensive_integration_test.py`, `test_critical_issue_fixes.py` 等) 遷移到 `tests/` 下的相應子目錄。
    - **測試**:
        - [x] 確認所有腳本在遷移後仍可正常執行。
        - [x] 確認測試套件在新的結構下能被發現並執行。

## 階段一：核心架構增強

### 1.1 增強 Transformer 模型 (`src/models/enhanced_transformer.py`)
*現狀：`enhanced_transformer.py` 檔案已創建。以下為內部組件的實現任務。*

- [x] **實施多尺度特徵提取器**
    - [x] 實現並行卷積層 (kernel_size=[3,5,7,11])。
    - [x] 加入不同時間窗口的特徵融合邏輯。
    - [x] 實現自適應池化層。
    - **測試**:
        - **單元測試**: [x] 驗證每個卷積核的輸出形狀和內容，[x] 測試特徵融合邏輯，[x] 驗證池化層的行為。
        - **整合測試**: [x] 與 Transformer 主模型整合後，檢查輸入輸出維度是否正確。

- [x] **實現增強 Transformer 模型主體架構**
    - [x] 實現 `EnhancedTransformer` 類的基本骨架 (輸入投射, 可選MSFE, 位置編碼, Transformer層堆疊, 可選CTS融合, 輸出投射)。
    - [x] 整合小波濾波器 (Wavelet Filter) 作為可選的早期特徵處理階段。
    - [x] 整合傅立葉頻段機率分析 (Fourier Analysis Frequency Band Probability) 作為可選的早期特徵處理階段。
    - [x] 實現符號嵌入 (Symbol Embedding) 以處理多個交易對輸入。
    - [x] 實現位置嵌入 (Position Embedding) - *注意: 此處的位置嵌入是指交易對在輸入序列中的順序，而非時間序列中的位置，後者由 `PositionalEncoding` 處理。*
    - [x] 處理 `src_key_padding_mask` 以適應可變數量的活躍交易對 (padding/masking)。
    - **測試**:
        - **單元測試**: [x] 驗證模型骨架的輸入輸出形狀，[x] 測試小波濾波器整合 (初步通過 `if __name__ == \'__main__\'` 腳本, 單元測試通過)，[x] 測試傅立葉分析整合 (初步通過 `if __name__ == \'__main__\'` 腳本, 單元測試通過)，[x] 測試符號與位置嵌入的整合 (初步通過 `if __name__ == \'__main__\'` 腳本, 單元測試通過)，[x] 測試 padding/masking 機制 (初步通過 `if __name__ == \'__main__\'` 腳本, 單元測試通過)。
        - **整合測試**: [x] 結合 MSFE 進行測試。

- [x] **添加自適應注意力機制**
    - [x] 實現市場狀態感知的注意力機制。
    - [x] 加入動態注意力權重調整邏輯。
    - [x] 整合長短期記憶融合機制。
    - **測試**:
        - **單元測試**: [x] 驗證注意力權重是否根據模擬的市場狀態動態調整。
        - **整合測試**: [x] 在模型中驗證注意力機制是否能有效處理不同市場模式。

- [x] **實現跨時間尺度融合**
    - [x] 實現多時間框架資訊整合邏輯。
    - [ ] 開發分層時間建模組件。
    - [ ] 加入時間一致性約束。
    - **測試**:
        - **單元測試**: [x] 驗證不同時間尺度數據的融合結果。
        - **整合測試**: [x] 測試模型在處理多時間尺度輸入時的表現。

- [x] **整合市場狀態檢測**
    - [x] 開發用於檢測市場狀態（如趨勢、震盪）的模組。 (初步實現 GMMMarketStateDetector)
    - [x] 將此模組整合到 Transformer 模型中，以影響其決策。 <!-- MODIFIED -->
    - **測試**:
        - [x] **單元測試**: 驗證市場狀態檢測模組在不同市場數據下的準確性。 (已創建 `test_market_state_detector.py` 並加入初步測試架構, 所有測試已通過, 包括 EWM ATR 調整和 save/load 功能)
        - [x] **整合測試**: 觀察模型行為是否根據檢測到的市場狀態做出相應調整。 (已在 `test_enhanced_models.py` 中添加 GMM 整合測試，包括成功案例和多種回退情境的驗證) <!-- MODIFIED -->

### 1.2 擴展量子策略層
- [x] **創建 `src/agent/enhanced_quantum_strategy_layer.py` 檔案**
    - [x] 根據藍圖定義 `DynamicStrategyGenerator` 類的基本結構。
    - **測試**:
        - [x] **單元測試**: 檔案可被 Python 解譯器成功導入。

- [x] **實施 15+ 預定義策略** <!-- MODIFIED -->
    - [x] 趨勢策略 (動量、突破、趨勢跟隨、反轉) - *All strategies fully implemented.*
    - [x] 統計套利 (均值回歸、協整、統計配對、波動率套利) - *All strategies fully implemented.*
    - [x] 機器學習策略 (強化學習、深度學習預測、集成學習、遷移學習) - *All strategies fully implemented.*
    - [x] 風險管理策略 (動態對沖、風險平價、VaR控制、最大回撤控制) - *All strategies fully implemented.*
    - [x] **Refactor Strategy File Structure for Improved Organization:** <!-- MODIFIED -->
        - [x] Move `BaseStrategy` to `src/agent/strategies/base_strategy.py`. <!-- MODIFIED -->
        - [x] Create `src/agent/strategies/trend_strategies.py` and migrate relevant strategies. <!-- MODIFIED -->
        - [x] Create `src/agent/strategies/statistical_arbitrage_strategies.py` and migrate relevant strategies. <!-- MODIFIED -->
        - [x] Create `src/agent/strategies/ml_strategies.py` and migrate relevant strategies. <!-- MODIFIED -->
        - [x] Create `src/agent/strategies/risk_management_strategies.py` and migrate relevant strategies. <!-- MODIFIED -->
        - [x] Create `src/agent/strategies/other_strategies.py` and migrate relevant strategies. <!-- MODIFIED -->
        - [x] Update imports in `src/agent/enhanced_quantum_strategy_layer.py` to reflect new strategy locations. <!-- MODIFIED -->
        - [x] Ensure `src/agent/strategies/__init__.py` correctly exports all strategies for easy import. <!-- MODIFIED -->
    - [x] **Implement Detailed Logic for All Refactored Strategies:** <!-- MODIFIED -->
        - [x] Implement logic for strategies in `trend_strategies.py`. <!-- MODIFIED -->
        - [x] Implement logic for strategies in `statistical_arbitrage_strategies.py`. <!-- MODIFIED -->
        - [x] Implement logic for strategies in `ml_strategies.py`. <!-- MODIFIED -->
        - [x] Implement logic for strategies in `risk_management_strategies.py`. <!-- MODIFIED -->
        - [x] Implement logic for strategies in `other_strategies.py`. <!-- MODIFIED -->
    - *Overall: All 28 predefined strategies have their logic fully implemented in their respective categorized files.* <!-- MODIFIED -->
    - **測試**:
        - [x] **單元測試**: 對每個策略進行單獨測試，驗證其在模擬數據上的邏輯正確性和預期行為。 <!-- MODIFIED -->

- [ ] **添加動態策略生成**
    - [ ] 在 `DynamicStrategyGenerator` 中實現 `generate_new_strategy` 方法。
    - [ ] 整合基因演算法 (`GeneticOptimizer`) 和神經架構搜索 (`NeuralArchitectureSearch`) 的初步框架。
    - **測試**:
        - **單元測試**: 驗證 `generate_new_strategy` 方法能夠基於輸入的市場條件生成（或選擇）策略。

- [ ] **創建策略組合機制**
    - [ ] 設計並實現在 `enhanced_quantum_strategy_layer.py` 中組合不同策略的邏輯。
    - **測試**:
        - **單元測試**: 驗證策略組合的權重分配和執行順序。

- [ ] **實現策略權重自適應**
    - [ ] 開發根據策略表現和市場狀況動態調整策略權重的機制。
    - **測試**:
        - **單元測試**: 模擬不同市場條件和策略表現，驗證權重調整是否符合預期。

## 階段二：學習系統重構

### 2.1 漸進式獎勵系統
- [ ] **創建 `src/environment/progressive_reward_system.py` 檔案**
    - [ ] 根據藍圖定義 `ProgressiveLearningSystem` 類的基本結構。
    - [ ] 實現 `SimpleReward`, `IntermediateReward`, `ComplexReward` 類的基本框架。
    - **測試**:
        - **單元測試**: 檔案可導入，類可實例化。

- [ ] **階段1：簡單獎勵（基本盈虧）**
    - [ ] 在 `SimpleReward` 中實現 `profit_loss * 0.8 + risk_penalty * 0.2` 的獎勵計算。
    - **測試**:
        - **單元測試**: 給定模擬的盈虧和風險數據，驗證獎勵計算的正確性。

- [ ] **階段2：中等複雜度（風險調整）**
    - [ ] 在 `IntermediateReward` 中實現藍圖中定義的中等複雜度獎勵計算。
    - **測試**:
        - **單元測試**: 驗證包含夏普比率、回撤懲罰、交易成本的獎勵計算。

- [ ] **階段3：高複雜度（多維優化）**
    - [ ] 在 `ComplexReward` 中實現藍圖中定義的高複雜度獎勵計算（多因子加權組合）。
    - **測試**:
        - **單元測試**: 驗證包含多個市場指標的複雜獎勵計算。

- [ ] **整合 `ProgressiveLearningSystem`**
    - [ ] 實現 `get_current_reward_function` 方法，使其能根據 `current_stage` 返回對應的獎勵函數實例。
    - [ ] 實現階段轉換邏輯 (基於 `stage_criteria`)。
    - **測試**:
        - **單元測試**: 驗證系統能否正確返回當前階段的獎勵函數，以及能否根據標準正確轉換階段。

### 2.2 元學習機制
- [ ] **創建 `src/agent/meta_learning_system.py` 檔案**
    - [ ] 根據藍圖定義 `MetaLearningSystem` 類的基本結構。
    - [ ] 實現 `MarketKnowledgeBase` 類的基本框架。
    - **測試**:
        - **單元測試**: 檔案可導入，類可實例化。

- [ ] **策略表現評估**
    - [ ] 在 `MetaLearningSystem` 中實現 `evaluate_strategy_performance` 方法。
    - [ ] 實現計算回報、風險、一致性、適應性的邏輯。
    - **測試**:
        - **單元測試**: 使用模擬的策略交易歷史，驗證各項評估指標計算的準確性。

- [ ] **自動策略調整**
    - [ ] 在 `MetaLearningSystem` 中實現 `adapt_strategies` 方法。
    - [ ] 設計基於市場狀態和策略評估結果調整策略的邏輯。
    - **測試**:
        - **單元測試**: 模擬不同市場狀態和策略表現，驗證策略調整的邏輯是否符合預期。

- [ ] **跨市場知識遷移**
    - [ ] 在 `MarketKnowledgeBase` 中設計存儲和檢索跨市場知識的機制。
    - [ ] 整合到 `MetaLearningSystem` 中，用於輔助策略調整或生成。
    - **測試**:
        - **單元測試**: 驗證知識庫的存取功能，以及知識遷移邏輯的初步有效性。

## 階段三：高級功能實現

### 3.1 策略創新系統
- [ ] **創建 `src/agent/strategy_innovation_engine.py` 檔案**
    - [ ] 定義 `StrategyInnovationEngine` 類的基本結構。
    - **測試**:
        - [ ] **單元測試**: 檔案可導入，類可實例化。

- [ ] **基因算法策略進化**
    - [ ] 在引擎中整合或實現一個基因演算法模組，用於策略參數或結構的進化。
    - **測試**:
        - **單元測試**: 驗證基因演算法的基本操作（選擇、交叉、突變）是否按預期工作。

- [ ] **神經架構搜索**
    - [ ] 在引擎中整合或實現一個神經架構搜索模組，用於探索新的策略模型結構。
    - **測試**:
        - **單元測試**: 驗證搜索演算法能否生成有效的模型架構描述。

- [ ] **自動特徵工程**
    - [ ] 在引擎中整合或實現自動特徵生成與選擇的模組。
    - **測試**:
        - **單元測試**: 驗證特徵生成和選擇的邏輯。

### 3.2 風險控制系統
*註：藍圖未指定特定檔案，這些功能可能整合到現有 agent 或新模組中。*

- [ ] **設計風險控制模組** (例如 `src/system/risk_management.py`)
    - [ ] 創建檔案並定義相關類和函數。

- [ ] **實時風險監控**
    - [ ] 實現監控當前倉位風險、市場波動等的邏輯。
    - **測試**:
        - **單元測試**: 驗證風險指標（如VaR、敞口）計算的準確性。

- [ ] **動態倉位管理**
    - [ ] 實現根據風險評估和市場狀況調整倉位大小的機制。
    - **測試**:
        - **單元測試**: 驗證倉位調整邏輯是否符合預設規則。

- [ ] **緊急停損機制**
    - [ ] 實現當達到預設的虧損閾值或偵測到極端市場事件時觸發的停損邏輯。
    - **測試**:
        - **單元測試**: 模擬觸發條件，驗證停損機制是否能被正確激活。

## 階段四：測試與驗證 (檔案創建與結構)

### 4.1 單元測試
- [ ] **創建 `tests/unit_tests/` 目錄** (如果尚未按總體結構完成)。
- [ ] **創建 `tests/test_enhanced_models.py`**
    - [ ] 針對 `enhanced_transformer.py` 中的各組件編寫單元測試。
- [ ] **為 `enhanced_quantum_strategy_layer.py` 編寫單元測試**
    - [ ] 在 `tests/unit_tests/` 下創建如 `test_quantum_strategies.py`。
    - [ ] 測試每個預定義策略和動態生成邏輯。
- [ ] **為 `progressive_reward_system.py` 編寫單元測試**
    - [ ] 在 `tests/unit_tests/` 下創建如 `test_reward_system.py`。
    - [ ] 測試各階段獎勵計算和階段轉換邏輯。
- [ ] **為 `meta_learning_system.py` 編寫單元測試**
    - [ ] 在 `tests/unit_tests/` 下創建如 `test_meta_learning.py`。
    - [ ] 測試策略評估、調整和知識遷移。
- [ ] **為 `strategy_innovation_engine.py` 編寫單元測試**
    - [ ] 在 `tests/unit_tests/` 下創建如 `test_strategy_innovation.py`。
- [ ] **為風險控制模組編寫單元測試**
    - [ ] 在 `tests/unit_tests/` 下創建如 `test_risk_management.py`。

### 4.2 集成測試
- [ ] **創建 `tests/integration_tests/` 目錄** (如果尚未按總體結構完成)。
- [ ] **創建/重構 `tests/test_integration.py` (或 `tests/integration_tests/test_end_to_end.py`)**
    - [ ] 根據藍圖要求，編寫端到端流程測試。
    - [ ] 測試模型訓練、策略執行、獎勵計算、元學習調整的整體流程。
    - [ ] 考慮將現有的 `comprehensive_integration_test.py` 和 `comprehensive_integration_test_final.py` 重構並遷移到此處。

### 4.3 性能測試
- [ ] **創建 `tests/performance_tests/` 目錄** (如果尚未按總體結構完成)。
- [ ] **編寫性能基準測試腳本**
    - [ ] 測試模型推理延遲。
    - [ ] 測試內存使用情況。
    - [ ] 測試訓練收斂速度。

## 實施檢查清單 (檔案創建)

- [x] `src/models/enhanced_transformer.py` (已存在，需實現內部組件)
- [x] `src/agent/enhanced_quantum_strategy_layer.py`
- [ ] `src/environment/progressive_reward_system.py`
- [ ] `src/agent/meta_learning_system.py`
- [ ] `src/agent/strategy_innovation_engine.py`
- [ ] `src/utils/model_integration.py` (用於整合各模組的輔助函數)
    - **任務**: 創建此檔案並根據需要填充輔助整合函數。
    - **測試**: 單元測試其包含的各個輔助函數。
- [ ] `tests/test_enhanced_models.py`
- [ ] `tests/test_integration.py`
- [ ] `scripts/training_pipeline.py`
    - **任務**: 創建此檔案，可能通過重構現有的 `train_universal_trader*.py` 腳本。
    - **測試**: 驗證訓練流程可以完整執行並產出模型。
- [ ] `scripts/model_validation.py`
    - **任務**: 創建此檔案，用於對訓練好的模型進行驗證。
    - **測試**: 驗證模型驗證流程可以執行並輸出評估指標。

## 配置文件

- [x] **`configs/enhanced_model_config.py`**
    - **任務**: 創建此 Python 檔案。將現有 `enhanced_transformer_config.json` 的內容轉換為 Python 字典 `ModelConfig` 並存儲於此。
    ```python
    ModelConfig = {
        'hidden_dim': 512,
        'num_layers': 12,
        'num_heads': 16,
        'intermediate_dim': 2048,
        'dropout_rate': 0.1,
        'max_sequence_length': 1000
    }
    ```
    - **測試**: 應用程式能正確讀取並使用此 Python 配置。

- [ ] **`configs/training_config.py`**
    - **任務**: 創建此檔案，用於存放訓練相關的配置 (如學習率、批次大小、訓練週期等)。
    - **測試**: 訓練流程能正確讀取並使用此配置。

- [ ] **`configs/strategy_config.py`**
    - **任務**: 創建此檔案，用於存放策略相關的配置 (如預定義策略的參數、策略組合規則等)。
    - **測試**: 策略層能正確讀取並使用此配置。

---
**下一步行動建議**：
1.  優先完成「總體結構與組織」中的目錄和檔案遷移任務，以建立清晰的開發環境。
2.  接著，按照階段順序，從「階段一：核心架構增強」開始，逐項完成檔案創建和功能實現。
3.  每完成一個模組或主要功能，立即編寫並執行其單元測試。
4.  在關鍵節點執行整合測試，確保各模組協同工作正常。

請定期更新此檔案的複選框狀態，以追蹤整體進度。
