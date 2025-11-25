# Project Navigation Map

Quick reference guide to navigate the OANDA Trading Bot project.

## 🎯 Start Here

**New to the project?**
1. Read [README.md](../README.md) - Project overview
2. Run `python scripts/setup/env_check.py` - Verify environment
3. Try `python scripts/training/train_quick_sac.py` - Quick test
4. Check [scripts/README.md](../scripts/README.md) - Script guide

**Want to understand the system?**
1. [.agent/project_analysis_report.md](../.agent/project_analysis_report.md) - Complete analysis
2. [.agent/upgrade_complete_report.md](../.agent/upgrade_complete_report.md) - v2.0 changes
3. [.agent/upgrade_visualization.md](../.agent/upgrade_visualization.md) - Architecture diagrams

---

## 📂 Key Directories

| Directory | Purpose | Start With |
|-----------|---------|------------|
| `scripts/` | Utility scripts | [scripts/README.md](../scripts/README.md) |
| `src/oanda_trading_bot/` | Source code | `src/oanda_trading_bot/__init__.py` |
| `configs/` | Configuration | `configs/enhanced_transformer_config.json` |
| `docs/` | Documentation | This file |
| `.agent/` | Design docs | `.agent/project_analysis_report.md` |
| `tests/` | Test suites | `tests/` |
| `reports/` | Test reports | Latest `*_test_*.md` |
| `model_registry/` | Model storage | `model_registry/index.json` |
| `weights/` | Trained models | Latest `.zip` files |

---

##  Quick Task Guide

### I want to...

#### Train a Model
→ Start: [scripts/training/](../scripts/training/)
- Quick: `train_quick_sac.py` (1 min)
- Small: `run_small_training_1day_5symbols.py` (10 min)
- Full: Use Streamlit UI

#### Test the System
→ Start: [scripts/testing/](../scripts/testing/)
- Quick: `complete_smoke_test.py`
- Full: `simple_e2e_test.py`

#### Run Live Trading
→ Start: [scripts/live_trading/](../scripts/live_trading/)
- Safe: `run_small_live_cycle.py` (Practice mode)
- API: `headless_api_trading.py`

#### Understand the Code
→ Start: Source code walkthrough
1. `src/oanda_trading_bot/common/model_registry.py` - Model management
2. `src/oanda_trading_bot/training_system/environment/` - Trading environment
3. `src/oanda_trading_bot/live_trading_system/core/` - Live trading core

#### Configure Settings
→ Start: [configs/](../configs/)
- Training: `enhanced_transformer_config.json`
- Live trading: `live/live_config.json`
- Credentials: `.env` (create from `.env.example`)

#### Check Logs
→ Start: [logs/](../logs/)
- Recent logs: Latest `*.log` files
- Reports: [reports/](../reports/)

#### Use API
→ Start: API server
1. `cd src/oanda_trading_bot/api`
2. `python main.py`
3. Visit: http://localhost:8000/docs

---

## 🔍 File Finder

### Configuration Files
- **Environment**: `.env`
- **Training config**: `configs/enhanced_transformer_config.json`
- **Live config**: `configs/live/live_config.json`
- **Model registry**: `model_registry/index.json`

### Key Source Files
- **Model Registry**: `src/oanda_trading_bot/common/model_registry.py`
- **Enhanced Simulation**: `src/oanda_trading_bot/training_system/environment/enhanced_simulation.py`
- **Reward Attribution**: `src/oanda_trading_bot/training_system/environment/symbol_reward_attribution.py`
- **API Main**: `src/oanda_trading_bot/api/main.py`
- **Training Environment**: `src/oanda_trading_bot/training_system/environment/trading_env.py`
- **Live Trading**: `src/oanda_trading_bot/live_trading_system/core/oanda_client.py`

### Documentation
- **Main README**: `README.md`
- **Scripts Guide**: `scripts/README.md`
- **This Navigation**: `docs/NAVIGATION.md`
- **Project Analysis**: `.agent/project_analysis_report.md`
- **Improvement Plan**: `.agent/improvement_plan.md`
- **Upgrade Report**: `.agent/upgrade_complete_report.md`
- **Cleanup Plan**: `.agent/CLEANUP_PLAN.md`

---

## 📊 Source Code Structure

```
src/oanda_trading_bot/
├── common/                    # Shared utilities
│   ├── model_registry.py     # Model version control (NEW v2.0)
│   ├── config.py             # Configuration management
│   └── utils.py              # Common utilities
│
├── training_system/           # Training pipeline
│   ├── environment/
│   │   ├── trading_env.py                  # Main trading env
│   │   ├── enhanced_simulation.py          # Swap + Market hours (NEW)
│   │   ├── symbol_reward_attribution.py    # Per-symbol rewards (NEW)
│   │   └── progressive_reward_calculator.py
│   ├── models/
│   │   └── enhanced_transformer.py         # Transformer model
│   ├── agent/
│   │   └── sac_agent_wrapper.py           # SAC algorithm
│   ├── trainer/
│   │   └── universal_trainer.py           # Training orchestration
│   └── app.py                              # Streamlit UI
│
├── live_trading_system/       # Live trading
│   ├── core/
│   │   ├── oanda_client.py                # OANDA API client
│   │   └── system_state.py                # System state management
│   ├── trading/
│   │   ├── position_manager.py            # Position management
│   │   ├── risk_manager.py                # Risk control
│   │   └── executor.py                    # Order execution
│   ├── ui/
│   │   └── app.py                         # Streamlit UI
│   └── main.py                             # Main entry point
│
├── api/                       # Unified API (NEW v2.0)
│   └── main.py                # FastAPI backend
│
└── backtester/                # Future: Independent backtester
    └── backtester.py          # Planned

```

**Legend**:
- (NEW v2.0) = Added in version 2.0
- (Planned) = In development

---

## 🚦 Common Workflows

### Workflow 1: First Time Setup
1. Clone repo: `git clone <url>`
2. Install: `pip install -r requirements.txt`
3. Configure: Edit `.env` with OANDA credentials
4. Verify: `python scripts/setup/env_check.py`
5. Test: `python scripts/testing/complete_smoke_test.py`

### Workflow 2: Train and Test
1. Train: `python scripts/training/train_quick_sac.py`
2. Check model: `ls weights/`
3. Test E2E: `python scripts/testing/simple_e2e_test.py`
4. View report: `cat reports/E2E_TEST_COMPLETE_REPORT.md`

### Workflow 3: Deploy to Live
1. Train model in Practice mode
2. Save to registry (automatic)
3. Test thoroughly: `python scripts/testing/simple_e2e_test.py`
4. Deploy: `registry.deploy_model(model_id, 'production')`
5. Start trading: `python scripts/live_trading/headless_api_trading.py`

### Workflow 4: Development
1. Make changes to `src/`
2. Run tests: `python scripts/testing/complete_smoke_test.py`
3. Check logs: `cat logs/*.log`
4. Commit: `git add -A && git commit -m "message"`

---

## 🎓 Learning Path

### Beginner
1. Read `README.md`
2. Run setup scripts
3. Train quick model
4. Read scripts README

### Intermediate
1. Study `project_analysis_report.md`
2. Understand trading environment
3. Modify configurations
4. Run full training

### Advanced
1. Read source code
2. Implement custom features
3. Extend API
4. Contribute improvements

---

## 📝 Quick Reference Commands

```bash
# Setup
python scripts/setup/env_check.py
python scripts/setup/check_cuda.py

# Training
python scripts/training/train_quick_sac.py
python scripts/training/run_small_training_1day_5symbols.py

# Testing
python scripts/testing/complete_smoke_test.py
python scripts/testing/simple_e2e_test.py

# Live Trading (Practice)
python scripts/live_trading/run_small_live_cycle.py

# API Server
cd src/oanda_trading_bot/api && python main.py

# Training UI
streamlit run src/oanda_trading_bot/training_system/app.py

# Live UI
cd src/oanda_trading_bot/live_trading_system/ui
streamlit run app.py --server.port 8502

# Quick Start (Windows)
quick_start.bat
```

---

## 🆘 Help Resources

**Getting Started Issues?**
→ Check [README.md - Troubleshooting](../README.md#-troubleshooting)

**Script Usage Questions?**
→ See [scripts/README.md](../scripts/README.md)

**System Design Questions?**
→ Read [.agent/project_analysis_report.md](../.agent/project_analysis_report.md)

**v2.0 Changes?**
→ See [.agent/upgrade_complete_report.md](../.agent/upgrade_complete_report.md)

**API Documentation?**
→ Start API server, visit http://localhost:8000/docs

---

## 🗺️ Site Map

```
Project Root
├── README.md (Start here!)
├── .env (Your credentials)
├── quick_start.bat (Quick launcher)
│
├── scripts/ (All utilities)
│   ├── README.md (Scripts guide)
│   ├── setup/ (Environment)
│   ├── training/ (Train models)
│   ├── testing/ (Tests)
│   └── live_trading/ (Trading)
│
├── docs/ (Documentation)
│   ├── NAVIGATION.md (This file)
│   └── archive/ (Old docs)
│
├── .agent/ (Design docs)
│   ├── project_analysis_report.md (Analysis)
│   ├── improvement_plan.md (Plans)
│   ├── upgrade_complete_report.md (v2.0)
│   └── CLEANUP_PLAN.md (Cleanup)
│
├── src/ (Source code)
│   └── oanda_trading_bot/
│       ├── common/ (Shared)
│       ├── training_system/ (Training)
│       ├── live_trading_system/ (Live)
│       └── api/ (API Server)
│
├── configs/ (Settings)
├── tests/ (Test suite)
├── reports/ (Test reports)
├── logs/ (Log files)
├── weights/ (Models)
└── model_registry/ (Model store)
```

---

**💡 Tip**: Bookmark this file for quick navigation!

---

*Last updated: 2025-11-25*
