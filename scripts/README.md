# Scripts Directory

Organized collection of utility scripts for the OANDA Trading Bot project.

## Directory Structure

```
scripts/
├── setup/           # Environment and data preparation
├── training/        # Model training scripts
├── testing/         # Test and validation scripts
├── live_trading/    # Live trading utilities
└── archive/         # Deprecated scripts (for reference)
```

## Quick Reference

### Setup Scripts (`setup/`)

| Script | Purpose | Usage |
|--------|---------|-------|
| `check_cuda.py` | Check CUDA/GPU availability | `python scripts/setup/check_cuda.py` |
| `env_check.py` | Verify environment setup | `python scripts/setup/env_check.py` |
| `build_scalers_from_oanda.py` | Build data scalers | `python scripts/setup/build_scalers_from_oanda.py` |
| `debug_imports.py` | Debug import issues | `python scripts/setup/debug_imports.py` |

### Training Scripts (`training/`)

| Script | Purpose | Steps | Duration |
|--------|---------|-------|----------|
| `train_quick_sac.py` | Quick SAC training | 5K | ~1 min |
| `run_short_training.py` | Short training session | 50K | ~5 min |
| `run_small_training_1day_5symbols.py` | Small training (1 day, 5 symbols) | 100K | ~10 min |
| `run_small_training_1day_5symbols_with_summary.py` | Small training with report | 100K | ~10 min |
| `run_small_training_1day_indices_with_summary.py` | Indices training with report | 100K | ~10 min |
| `training_pipeline.py` | Full training pipeline | Custom | Variable |

**Recommended for beginners**: Start with `train_quick_sac.py`

### Testing Scripts (`testing/`)

| Script | Purpose | What it tests |
|--------|---------|---------------|
| `simple_e2e_test.py` | **Main E2E test** | Training → Model → API → Trading |
| `complete_smoke_test.py` | **Main smoke test** | All upgraded components |
| `comprehensive_smoke_test.py` | Comprehensive validation | Full system check |
| `run_complete_smoke_test.py` | Extended smoke test | Detailed validation |
| `phase_test_runner.py` | Phase-by-phase testing | Staged validation |

**Recommended**: Use `simple_e2e_test.py` for quick validation

### Live Trading Scripts (`live_trading/`)

| Script | Purpose | Mode |
|--------|---------|------|
| `run_small_live_cycle.py` | Small live trading cycle | Practice |
| `headless_api_trading.py` | Headless API trading | Practice/Live |
| `simulate_live_trading_gpu.py` | GPU-accelerated simulation | Simulation |
| `live_test_bypass_ui.py` | Live test without UI | Practice |
| `start_live_trading_headless.py` | Start headless trading | Practice/Live |
| `live_trade_test.py` | Live trading test | Practice |

**Warning**: Always test in Practice mode first!

### Archive (`archive/`)

Deprecated scripts kept for reference. Do not use in production.

## Common Workflows

### 1. Quick System Check
```bash
# Check environment
python scripts/setup/env_check.py
python scripts/setup/check_cuda.py

# Run smoke test
python scripts/testing/complete_smoke_test.py
```

### 2. Train a Model
```bash
# Quick training (1 minute)
python scripts/training/train_quick_sac.py

# Small training (10 minutes)
python scripts/training/run_small_training_1day_5symbols_with_summary.py
```

### 3. End-to-End Test
```bash
# Complete E2E test (training + trading)
python scripts/testing/simple_e2e_test.py
```

### 4. Live Trading (Practice)
```bash
# Small trading cycle
python scripts/live_trading/run_small_live_cycle.py
```

## Best Practices

1. **Always use Practice mode first**
   - Set `OANDA_ENVIRONMENT=practice` in `.env`
   - Verify with small amounts

2. **Run tests before deployment**
   ```bash
   python scripts/testing/complete_smoke_test.py
   python scripts/testing/simple_e2e_test.py
   ```

3. **Check GPU availability for training**
   ```bash
   python scripts/setup/check_cuda.py
   ```

4. **Use smaller scripts for testing**
   - Faster iteration
   - Lower cost
   - Easier debugging

## Script Naming Convention

- `check_*.py` - Environment checks
- `build_*.py` - Data/model preparation
- `train_*.py` - Training utilities
- `run_*.py` - Execution scripts
- `*_test.py` - Testing scripts

## Getting Help

For detailed usage of any script:
```bash
python scripts/<category>/<script>.py --help
```

Or check the script's docstring:
```python
python -c "import scripts.<category>.<script>; help(scripts.<category>.<script>)"
```

---

**Note**: Scripts in `archive/` are deprecated and may not work with the current version.
