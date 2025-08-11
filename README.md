# OANDA AI Trading Bot

## 1. Project Overview

This project is a comprehensive, AI-powered trading bot designed for the OANDA platform. It consists of two primary, independent systems:

1.  **Training System**: A sophisticated environment for training and evaluating reinforcement learning models for trading. It features a Streamlit-based UI for real-time monitoring of training progress, model performance, and system resources.
2.  **Live Trading System**: A robust system for deploying the trained models to a live (or paper) OANDA account. It also includes a Streamlit-based dashboard for monitoring live trades, account status, and system health.

This codebase has been professionally refactored to follow modern software engineering best practices, ensuring modularity, maintainability, and ease of use.


## 2. Directory Structure

The project has been organized into a clean, standard Python project structure:

```
oanda-trading-bot/
├── configs/
│   ├── live/
│   │   └── live_config.json
│   └── training/
│       └── ... (training-related configurations)
├── docs/
│   └── ... (all project documentation)
├── src/
│   └── oanda_trading_bot/
│       ├── __init__.py
│       ├── common/
│       │   └── ... (code shared between both systems)
│       ├── live_trading_system/
│       │   ├── __init__.py
│       │   ├── app.py          (Live Trading UI)
│       │   ├── main.py         (Live Trading entry point)
│       │   └── ... (core, data, trading modules)
│       └── training_system/
│           ├── __init__.py
│           ├── app.py          (Training UI)
│           └── ... (agent, environment, models, trainer modules)
├── tests/
│   ├── live_trading_system/
│   └── training_system/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── start_live_trading_system.bat
├── start_training_system.bat
└── verify_imports.py
```

-   **`configs/`**: Contains all configuration files, separated by system.
-   **`docs/`**: Contains all markdown documentation.
-   **`src/oanda_trading_bot/`**: The main source code package.
    -   **`common/`**: Modules shared by both the training and live systems.
    -   **`live_trading_system/`**: All code related to the live trading bot.
    -   **`training_system/`**: All code related to model training.
-   **`tests/`**: Contains all unit and integration tests, separated by system.
-   **`.bat` files**: Convenient scripts for launching the UIs on Windows.
-   **`verify_imports.py`**: A script to test that all modules can be imported correctly after setup.


## 3. Installation & Setup

Follow these steps to set up the project environment.

### Step 1: Clone the Repository
Clone this repository to your local machine.

### Step 2: Create a `.env` File
The system uses a `.env` file in the project root to manage sensitive credentials. Create a file named `.env` and add your OANDA credentials:

```
OANDA_API_KEY="YOUR_API_KEY_HERE"
OANDA_ACCOUNT_ID="YOUR_ACCOUNT_ID_HERE"
OANDA_ENVIRONMENT="practice" # or "live"
```

### Step 3: Install Dependencies
All required Python packages are listed in `requirements.txt`. Install them using pip:

```sh
pip install -r requirements.txt
```
**Note:** Due to the size of some packages (especially PyTorch), this step may take some time and requires a significant amount of disk space.


## 4. Execution Flow

You can run each system independently using the provided batch scripts or by running the Python scripts directly.

### Training System

The training system provides a UI to configure, run, and monitor the training of your AI models.

**To run (Windows):**
Simply double-click the `start_training_system.bat` file.

**To run (Manual):**
Open a terminal in the project root and run the following command:
```sh
streamlit run src/oanda_trading_bot/training_system/app.py
```
This will launch the Streamlit web server and open the UI in your browser.

### Live Trading System

The live trading system deploys a trained model to trade on your OANDA account.

**To run (Windows):**
Simply double-click the `start_live_trading_system.bat` file.

**To run (Manual):**
Open a terminal in the project root and run the following command:
```sh
python src/oanda_trading_bot/live_trading_system/main.py
```
This script will first initialize all backend components and then automatically launch the Streamlit UI for monitoring.


## 5. System Verification

After installation, you can verify that all modules are correctly structured and importable by running the `verify_imports.py` script:

```sh
python verify_imports.py
```
If the script runs without any errors and prints a success message, your environment is set up correctly.


## 6. Spec Sheet / Key Components

This project is built from several key components that work together.

### Training System Components
-   **`UniversalTrainer`**: The main orchestrator for the training pipeline. It manages data preparation, environment setup, agent training, and model saving.
-   **`UniversalTradingEnvV4`**: The custom OpenAI Gym-compatible environment where the agent learns. It simulates trading, calculates portfolio value, and provides rewards.
-   **`QuantumEnhancedSAC`**: The core reinforcement learning agent, based on Soft-Actor-Critic (SAC), with custom enhancements for trading.
-   **`EnhancedTransformer`**: The deep learning model used by the agent to analyze market data. It uses multiple attention layers to understand market dynamics.
-   **`ProgressiveRewardSystem`**: A sophisticated reward system that adapts as the model learns, guiding it through different stages of training.
-   **`UniversalMemoryMappedDataset`**: An efficient data loader that uses memory-mapped files to handle large historical datasets without consuming excessive RAM.

### Live Trading System Components
-   **`initialize_system` (`main.py`)**: The entry function that loads configurations and instantiates all live system components.
-   **`trading_loop` (`main.py`)**: The main loop that runs in the background, executing the trading logic at regular intervals.
-   **`OandaClient`**: A robust client for interacting with the OANDA V20 API, including error handling and retries.
-   **`PositionManager`**: An in-memory store that tracks all currently open positions.
-   **`OrderManager`**: Responsible for creating and submitting orders to the OANDA API.
-   **`RiskManager`**: Enforces risk management rules, such as stop-loss and position sizing.
-   **`DatabaseManager`**: A simple SQLite-based manager for logging all trade history.

### Shared Components
-   **`InstrumentInfoManager`**: A utility for fetching and caching instrument details (like pip size, margin rates) from OANDA.
-   **`logger_setup`**: Centralized logging configuration for consistent logs across the application.
-   **`shared_data_manager`**: (In Training System) A manager to safely share data between the training thread and the Streamlit UI thread.

