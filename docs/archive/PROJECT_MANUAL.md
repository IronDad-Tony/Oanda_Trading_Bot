# Oanda Trading Bot - Project Manual

## 1. Project Overview
This project is a sophisticated algorithmic trading system designed for the OANDA platform. It utilizes a hybrid approach combining **Deep Reinforcement Learning (SAC)** with an **Enhanced Transformer** architecture for feature extraction. The system is divided into two main subsystems:
1.  **Training System**: A robust pipeline for downloading data, simulating trading environments, and training the RL agent.
2.  **Live Trading System**: A real-time engine that executes trades based on model predictions, manages risk, and interfaces with the OANDA API.
3.  **Frontend UI**: A React-based dashboard for monitoring and controlling the trading bot.

---

## 2. System Architecture

### 2.1 Directory Structure
-   `src/oanda_trading_bot/`: Main Python package.
    -   `live_trading_system/`: Real-time trading logic.
    -   `training_system/`: Model training pipeline and RL environment.
    -   `common/`: Shared utilities (logger, config, instrument info).
-   `configs/`: Configuration files (JSON).
-   `frontend/`: React application source code.
-   `data/`: Storage for historical data, scalers, and database.
-   `reports/`: Training logs and performance reports.

### 2.2 Data Flow
1.  **Data Ingestion**: Raw candle data (OHLCV) is fetched from OANDA API.
2.  **Preprocessing**: Data is cleaned, features are engineered (Log Returns, MTF, Fourier/Wavelet), and scaled.
3.  **Model Inference**: Processed features are fed into the `EnhancedTransformer` (wrapped in SAC policy) to generate trade signals.
4.  **Execution**: Signals are validated by `RiskManager` and executed via `OrderManager`.
5.  **Feedback**: Trade results and account state are broadcast to the Frontend UI.

---

## 3. Functional Modules

### 3.1 Live Trading System
-   **Entry Point**: `src/oanda_trading_bot/live_trading_system/main.py`
    -   Initializes `OandaClient`, `DatabaseManager`, `SystemState`, and `Broadcaster`.
    -   Runs the main event loop (async).
-   **Trading Logic**: `src/oanda_trading_bot/live_trading_system/trading/trading_logic.py`
    -   Manages data buffers for each instrument.
    -   Orchestrates the cycle: Fetch Data -> Preprocess -> Predict -> Execute.
    -   Handles "Cold Start" (pre-filling buffers).
-   **Prediction Service**: `src/oanda_trading_bot/live_trading_system/model/prediction_service.py`
    -   Loads trained models (PyTorch `.pt` or Stable Baselines3 `.zip`).
    -   Handles batch inference and data padding/masking.
-   **Data Preprocessor**: `src/oanda_trading_bot/live_trading_system/data/data_preprocessor.py`
    -   **Features**: Log returns, Log volume, Multi-Timeframe (1T, 15T, 1H) stats, Fourier/Wavelet components.
    -   **Scaling**: Standardizes features using pre-computed scalers (`scalers.json`).

### 3.2 Training System
-   **Trainer**: `src/oanda_trading_bot/training_system/trainer/universal_trainer.py`
    -   **Pipeline**:
        1.  **Data Prep**: Downloads historical data and creates memory-mapped datasets.
        2.  **Env Setup**: Initializes `UniversalTradingEnvV4`.
        3.  **Agent Setup**: Configures `QuantumEnhancedSAC` with `EnhancedTransformerFeatureExtractor`.
        4.  **Training**: Runs the training loop with checkpoints and evaluation.
-   **Environment**: `src/oanda_trading_bot/training_system/environment/trading_env.py`
    -   **Type**: `gym.Env` implementation.
    -   **Simulation**: Simulates latency, slippage, and commission.
    -   **Observation Space**: Complex dictionary including `market_features`, `features_from_dataset` (sequence), `context_features` (positions, PnL), and masks.
    -   **Reward System**: Supports `ProgressiveRewardCalculator` (curriculum learning) and `EnhancedRewardCalculator`.

---

## 4. Model Architecture

### 4.1 Enhanced Transformer
**File**: `src/oanda_trading_bot/training_system/models/enhanced_transformer.py`

A state-of-the-art Transformer designed for time-series financial data. Key components:
1.  **Multi-Scale Feature Extractor (MSFE)**: Convolutions with different kernel sizes (3, 5, 7, 11) to capture patterns at various frequencies.
2.  **Adaptive Attention**:
    -   Uses a `MarketStateDetector` (GMM or heuristic) to identify regimes (Trend, Volatility, etc.).
    -   Modulates attention weights based on the detected market state.
3.  **Cross-Time-Scale Fusion (CTS)**: Fuses information from different time scales (e.g., 1min, 5min, 15min).
4.  **Positional Encoding**: Supports Learnable or Sinusoidal encodings.
5.  **Output**: Feeds into the SAC Agent's Actor and Critic networks.

### 4.2 Reinforcement Learning (SAC)
-   **Algorithm**: Soft Actor-Critic (SAC).
-   **Policy**: `QuantumEnhancedSAC` (Custom wrapper around SB3's SAC).
-   **Features**:
    -   **Discrete/Continuous Actions**: Maps continuous model output to discrete trade actions (Long, Short, Flat).
    -   **Entropy Regularization**: Encourages exploration.

---

## 5. Configuration & Parameters

### 5.1 Live Configuration (`configs/live/live_config.json`)
| Parameter | Description | Example |
| :--- | :--- | :--- |
| `trading_instruments` | List of symbols to trade | `["EUR_USD", "XAU_USD"]` |
| `trading_granularity` | Candle timeframe | `"S5"` (5 seconds) |
| `model_lookback_window` | Input sequence length | `512` |
| `risk_management` | Risk settings (Stop Loss, Take Profit) | `{"max_risk_per_trade_percent": 0.1}` |
| `execution` | Order execution settings | `{"max_slippage_pips": 1.0}` |

### 5.2 Model Configuration (`configs/training/enhanced_model_config.json`)
| Parameter | Description | Example |
| :--- | :--- | :--- |
| `hidden_dim` | Transformer model dimension | `1024` |
| `num_layers` | Number of Transformer layers | `24` |
| `num_heads` | Number of Attention heads | `16` |
| `use_adaptive_attention` | Enable state-aware attention | `true` |
| `use_msfe` | Enable Multi-Scale Feature Extractor | `true` |
| `msfe_scales` | Kernel sizes for MSFE | `[3, 5, 7, 11]` |

---

## 6. Frontend UI Design

### 6.1 Technology Stack
-   **Framework**: React (TypeScript)
-   **UI Library**: Material UI (MUI) v5
-   **Theme**: Dark Mode (Professional Trading Terminal aesthetic)

### 6.2 Key Components
-   **Dashboard (`Dashboard.tsx`)**: The main layout container.
-   **Global Controls**: Buttons to Start/Stop/Pause the bot and Emergency Close All.
-   **Global Status**: Displays connection status, current time, and system health.
-   **Account Summary**: Real-time view of Balance, Equity, Margin, and Unrealized PnL.
-   **Positions Table**: Detailed list of open positions with PnL and action buttons.
-   **Interactive Chart**: Visualizes price data and trade markers (likely using `lightweight-charts` or `recharts`).
-   **Log Stream**: Real-time scrolling log of system events.

---

## 7. Operation Logic

### 7.1 Training Workflow
1.  **Configuration**: Edit `configs/training/enhanced_model_config.json` and `universal_trainer.py` parameters.
2.  **Execution**: Run `start_training_system.bat`.
3.  **Process**:
    -   System downloads missing data.
    -   Initializes the Environment.
    -   Trains the Agent for `total_timesteps`.
    -   Saves checkpoints to `weights/`.

### 7.2 Live Trading Workflow
1.  **Configuration**: Edit `configs/live/live_config.json` to select instruments and risk settings.
2.  **Execution**: Run `start_live_trading_system.bat`.
3.  **Process**:
    -   System loads the latest model from `weights/`.
    -   Connects to OANDA stream.
    -   Warms up data buffers (fetches history).
    -   Starts the async trading loop.
    -   Broadcasts updates to the Web UI (localhost).

### 7.3 Error Handling
-   **Data Stale**: If data stops arriving (`market_stale_threshold`), the system suspends trading for that instrument.
-   **API Errors**: Exponential backoff for connection retries.
-   **Critical Failures**: `SystemState` triggers a shutdown or pause to protect capital.
