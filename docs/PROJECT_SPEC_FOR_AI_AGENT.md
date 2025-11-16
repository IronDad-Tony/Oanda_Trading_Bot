# OANDA Trading Bot - Technical Specification for AI Agents

## Project Overview

This is a comprehensive AI-powered trading bot for the OANDA platform, consisting of two independent systems:

1. **Training System**: Reinforcement learning model training with Streamlit UI
2. **Live Trading System**: Deployed model execution with FastAPI + Web Dashboard

## System Architecture

### Training System Components

```
UniversalTrainer
├── Data Preparation
│   ├── OandaDownloader: Downloads historical data from OANDA API
│   ├── CurrencyManager: Handles currency conversion
│   ├── CurrencyRoutePlanner: Plans conversion routes
│   └── UniversalMemoryMappedDataset: Memory-mapped dataset for efficient I/O
├── Environment
│   └── UniversalTradingEnvV4: Gymnasium-compatible trading environment
├── Agent
│   ├── QuantumEnhancedSAC: SAC agent wrapper
│   ├── EnhancedTransformerFeatureExtractor: Transformer-based feature extractor
│   └── QuantumStrategyLayer: Quantum-inspired strategy layer
└── Trainer
    └── UniversalTrainer: Orchestrates training pipeline
```

### Live Trading System Components

```
TradingLogic
├── OandaClient: OANDA V20 API wrapper
├── LivePreprocessor: Real-time data preprocessing
├── PredictionService: Model inference service
└── OrderManager
    ├── RiskManager: Risk assessment and position sizing
    ├── PositionManager: Position tracking
    └── DatabaseManager: SQLite database operations
```

## Key Configuration Files

### Environment Variables (.env)

```env
OANDA_API_KEY=<required>
OANDA_ACCOUNT_ID=<required>
OANDA_ENVIRONMENT=practice|live
TRAINING_GRANULARITY=S5
MAX_SYMBOLS_ALLOWED=10
INITIAL_CAPITAL=100000.0
USE_AMP=true
SAC_LEARNING_RATE=0.0003
SAC_BATCH_SIZE=64
```

### Live Trading Config (configs/live/live_config.json)

Key parameters:
- `trading_instruments`: List of instruments to trade
- `model_lookback_window`: Number of historical candles (default: 512)
- `trading_loop_interval_seconds`: Main loop sleep duration (default: 5)
- `risk_management`: Risk parameters (exposure limits, ATR settings, stop loss/take profit)
- `execution`: Order execution settings (timeouts, slippage protection)

### Model Config (configs/training/enhanced_model_config.json)

Key parameters:
- `hidden_dim`: Transformer hidden dimension (default: 1024)
- `num_layers`: Number of transformer layers (default: 24)
- `num_heads`: Number of attention heads (default: 16)
- `max_sequence_length`: Maximum input sequence length (default: 512)
- Feature flags: `use_msfe`, `use_cts_fusion`, `use_graph_attn`, etc.

## Data Flow

### Training System Data Flow

```
1. User selects symbols and time range in UI
2. UniversalTrainer downloads data via OandaDownloader
3. CurrencyRoutePlanner determines required conversion pairs
4. CurrencyManager downloads conversion data
5. Data is processed and stored in MMAP format
6. UniversalMemoryMappedDataset loads data for training
7. UniversalTradingEnvV4 provides observations to agent
8. Agent generates actions
9. Environment executes actions and returns rewards
10. Agent updates policy using SAC algorithm
```

### Live Trading System Data Flow

```
1. System initializes and loads model
2. User selects symbols via UI/API
3. TradingLogic warms up data buffers (fills lookback_window)
4. Main loop (trading_loop):
   a. Fetch latest candles for each instrument
   b. Update data buffers
   c. Preprocess data (LivePreprocessor)
   d. Get predictions (PredictionService)
   e. Generate trading signals
   f. Process signals (OrderManager):
      - Risk assessment (RiskManager)
      - Position sizing
      - Order execution (OandaClient)
      - Position tracking (PositionManager)
      - Database logging (DatabaseManager)
5. Risk monitoring (continuous)
6. WebSocket broadcasts status updates
```

## Database Schema

### trades table

```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    oanda_trade_id TEXT UNIQUE NOT NULL,
    instrument TEXT NOT NULL,
    units INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    signal TEXT,
    status TEXT NOT NULL,  -- 'OPEN' or 'CLOSED'
    pnl REAL,
    open_timestamp TEXT NOT NULL,
    close_timestamp TEXT,
    close_price REAL
)
```

## API Endpoints (Live Trading System)

### REST API (FastAPI)

Base URL: `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ready` | GET | System readiness check |
| `/health` | GET | Health check |
| `/account/summary` | GET | Account summary |
| `/account/positions` | GET | Open positions |
| `/account/trades` | GET | Trade history |
| `/symbols/available` | GET | Available instruments |
| `/symbols/catalog` | GET | Categorized instruments |
| `/models` | GET | Available models |
| `/session/status` | GET | Current session status |
| `/session/config` | POST | Configure session |
| `/session/start` | POST | Start trading |
| `/session/stop` | POST | Stop trading |
| `/session/close_position` | POST | Close position for instrument |
| `/session/emergency_stop` | POST | Emergency stop and flatten |
| `/config/risk` | GET | Risk configuration |
| `/risk/update` | POST | Update risk parameters |
| `/analytics/performance` | GET | Performance metrics |
| `/ws | WebSocket | Real-time status updates |

### WebSocket Protocol

Messages are JSON objects with `type` and `data` fields:

```json
{
  "type": "status",
  "data": {
    "running": true,
    "selected_symbols": ["EUR_USD", "GBP_USD"],
    "model": "weights/sac_model_symbols5.zip",
    "suspended": {}
  }
}
```

## Model Architecture

### Enhanced Transformer Feature Extractor

Input: `(batch_size, num_symbols, sequence_length, num_features)`

Processing:
1. Symbol embedding (padding_idx = num_symbols)
2. Multi-scale feature extraction (MSFE) with scales [3, 5, 7, 11]
3. Transformer encoder (24 layers, 16 heads, 1024 dim)
4. Cross-time-scale fusion (CTS) with scales [1, 3, 5]
5. Graph attention layers (2 layers, 4 heads, topk=5)
6. Pooling (ignores padding)

Output: `(batch_size, output_dim)` where output_dim=256

### Quantum Enhanced SAC

- Policy network: Takes transformer features, outputs action distribution
- Q networks: Two independent Q networks for double Q-learning
- V network: State value network
- Experience replay: Buffer size = 5000 * num_symbols
- Learning: SAC algorithm with automatic entropy tuning

## Risk Management

### Position Sizing

Two modes:

1. **ATR-based** (`use_atr_sizing = true`):
   - Calculate ATR over `atr_period` candles
   - Stop loss distance = ATR * `stop_loss_atr_multiplier`
   - Position size = Risk amount / (Stop loss distance * pip value)

2. **Fixed pips** (`use_atr_sizing = false`):
   - Stop loss distance = `stop_loss_pips` * pip size
   - Position size = Risk amount / (Stop loss distance * pip value)

### Exposure Limits

- `max_total_exposure_usd`: Absolute USD limit
- `max_total_exposure_percent`: Percentage of target capital
- `max_position_exposure_percent`: Per-instrument limit

### Drawdown Protection

- `hard_stop_loss_percent`: Stop if drawdown from session start exceeds threshold
- `trailing_stop_loss_percent`: Stop if drawdown from peak exceeds threshold
- `daily_loss_limit_percent`: Stop if daily loss exceeds threshold

## Key Classes and Methods

### UniversalTrainer

```python
class UniversalTrainer:
    def __init__(self, trading_symbols, start_time, end_time, ...):
        # Initialize trainer with symbols and time range
    
    def train(self):
        # Main training loop
        # - Prepare data
        # - Initialize environment
        # - Initialize agent
        # - Run training loop
        # - Save model
```

### TradingLogic

```python
class TradingLogic:
    def execute_trade_cycle(self):
        # Main trading cycle
        # - For each selected instrument:
        #   - Fetch latest candles
        #   - Update buffer
        #   - Preprocess features
        #   - Get prediction
        #   - Process signal
    
    def warmup_buffers(self, instruments, max_wait_seconds=120):
        # Fill data buffers before trading starts
```

### RiskManager

```python
class RiskManager:
    def assess_trade(self, instrument, signal, price):
        # Assess trade and return:
        # - units: Position size
        # - stop_loss: Stop loss price
        # - take_profit: Take profit price
    
    def should_halt_trading(self):
        # Check if trading should halt due to drawdown
        # Returns: (should_halt: bool, reason: str)
```

### OrderManager

```python
class OrderManager:
    def process_signal(self, signal_info):
        # Process trading signal:
        # - Risk assessment
        # - Order creation
        # - Position tracking
        # - Database logging
```

## File Structure

```
src/oanda_trading_bot/
├── common/
│   ├── account_profile.py      # Account profile detection
│   ├── instrument_info_manager.py  # Instrument metadata cache
│   └── mtf_features.py          # Multi-timeframe features
├── training_system/
│   ├── app.py                  # Streamlit UI
│   ├── trainer/
│   │   └── universal_trainer.py
│   ├── environment/
│   │   └── trading_env.py      # UniversalTradingEnvV4
│   ├── agent/
│   │   ├── sac_agent_wrapper.py
│   │   └── enhanced_feature_extractor.py
│   └── data_manager/
│       ├── oanda_downloader.py
│       ├── currency_manager.py
│       └── mmap_dataset.py
└── live_trading_system/
    ├── main.py                 # System initialization
    ├── api/
    │   └── server.py           # FastAPI server
    ├── core/
    │   ├── oanda_client.py
    │   ├── system_state.py
    │   └── tradability_guard.py
    ├── trading/
    │   ├── trading_logic.py
    │   ├── order_manager.py
    │   ├── risk_manager.py
    │   └── position_manager.py
    ├── model/
    │   └── prediction_service.py
    ├── data/
    │   └── data_preprocessor.py
    ├── database/
    │   └── database_manager.py
    └── ui/                     # Web dashboard
        ├── index.html
        ├── main.js
        └── styles.css
```

## Common Patterns

### Account Currency Detection

Both systems use `get_account_profile()` from `common.account_profile`:

```python
from oanda_trading_bot.common.account_profile import get_account_profile

profile = get_account_profile()
account_currency = profile.currency  # e.g., "AUD"
environment = profile.environment    # "practice" or "live"
```

### Instrument Information

Use `InstrumentInfoManager` for instrument metadata:

```python
from oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager

iim = InstrumentInfoManager()
details = iim.get_details("EUR_USD")
pip_location = details.pip_location
minimum_trade_size = details.minimum_trade_size
```

### Currency Conversion

Use `CurrencyDependencyManager` for currency conversion:

```python
from oanda_trading_bot.training_system.data_manager.currency_manager import CurrencyDependencyManager

cur_mgr = CurrencyDependencyManager(account_currency, apply_oanda_markup=True)
rate = cur_mgr.convert_to_account_currency(quote_currency, price_map)
```

## Error Handling

### OANDA API Errors

- Uses `tenacity` for retry logic
- Retries 3 times with 2-second intervals
- Only retries on `V20Error` exceptions

### Model Loading Errors

- Checks if model file exists
- Validates model compatibility
- Falls back gracefully if model unavailable

### Data Errors

- Validates data shape and columns
- Handles missing data gracefully
- Logs warnings for data issues

## Testing

### Unit Tests

Located in `tests/`:
- `tests/training_system/unit_tests/`: Unit tests for training components
- `tests/live_trading_system/tests/`: Unit tests for live trading components

### Integration Tests

- `tests/training_system/integration_tests/`: End-to-end training tests
- `tests/live_trading_system/tests/e2e_*.py`: End-to-end live trading tests

### Test Execution

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/training_system/unit_tests/test_sac_agent.py

# Run with coverage
pytest --cov=src/oanda_trading_bot
```

## Logging

### Log Configuration

- Training system: `src/oanda_trading_bot/training_system/common/logger_setup.py`
- Live trading system: `src/oanda_trading_bot/live_trading_system/core/logger_setup.py`

### Log Levels

- `DEBUG`: Detailed debugging information
- `INFO`: General informational messages
- `WARNING`: Warning messages
- `ERROR`: Error messages
- `CRITICAL`: Critical errors

### Log Locations

- Training logs: `logs/training_*.log`
- Live trading logs: `logs/live_trading_*.log`
- Execution metrics: `logs/execution_metrics.csv`

## Performance Considerations

### Memory Management

- Uses memory-mapped datasets for large historical data
- Clears GPU cache periodically during training
- Limits buffer sizes to prevent memory overflow

### GPU Usage

- Automatically detects CUDA availability
- Uses GPU for model inference and training
- Falls back to CPU if GPU unavailable

### API Rate Limiting

- Implements per-symbol sequencing to avoid API spam
- Minimum interval between orders: 250ms (configurable)
- Uses transaction stream to reduce polling

## Security

### Credentials

- Stored in `.env` file (not committed to git)
- Loaded via `python-dotenv`
- Never logged or exposed in UI

### API Security

- Uses OANDA API keys for authentication
- Supports both practice and live environments
- Validates account permissions before trading

## Deployment

### Training System

```bash
# Windows
start_training_system.bat

# Manual
streamlit run src/oanda_trading_bot/training_system/app.py
```

### Live Trading System

```bash
# Windows
start_live_trading_system.bat

# Manual
python -c "from oanda_trading_bot.live_trading_system.api.server import main; main()"
```

### Headless Mode

```bash
# Quick test run (practice mode, 3 cycles)
python scripts/run_small_live_cycle.py --cycles 3
```

## Common Issues and Solutions

### Issue: Model not loading

- Check model path in config
- Verify model file exists
- Check model compatibility (SAC vs PyTorch)

### Issue: Data buffer not ready

- Increase warmup timeout
- Check OANDA API connectivity
- Verify instrument names are correct

### Issue: Currency conversion errors

- Ensure required conversion pairs are downloaded
- Check CurrencyRoutePlanner for conversion routes
- Verify account currency is correct

### Issue: Risk manager rejecting trades

- Check exposure limits
- Verify account equity
- Review risk parameters in config

## Future Enhancements

Potential areas for improvement:

1. **Multi-timeframe support**: Currently uses single timeframe (S5)
2. **Portfolio optimization**: Advanced portfolio allocation strategies
3. **Backtesting framework**: Historical performance testing
4. **Strategy backtesting**: Test strategies before live deployment
5. **Advanced risk models**: VaR, CVaR, stress testing
6. **Machine learning features**: Additional feature engineering
7. **Real-time monitoring**: Enhanced monitoring and alerting

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-XX  
**For**: AI Agent Reference

