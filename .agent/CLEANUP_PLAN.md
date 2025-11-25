# Project Cleanup and Reorganization Plan

Date: 2025-11-25
Version: 2.0 Cleanup

## Current Issues Identified

### 1. Duplicate/Legacy Files
- `README.md` vs `README_v2.md` - Need to consolidate
- `PROJECT_MANUAL.md` vs docs in `.agent/` - Redundant documentation
- Multiple training scripts in `scripts/` with overlapping functionality
- Chinese filename: `詳細的改善報告.md` - Should use English

### 2. Bloated Files
- `scripts/end_to_end_test.py` - 44MB (DELETED - corrupted)
- Multiple log files in root directory

### 3. Scattered Documentation
- `.agent/` - Analysis reports
- `docs/` - 24 files
- Root level markdown files
- Need centralized documentation structure

### 4. Script Redundancy
28 scripts in `scripts/` folder - many overlapping:
- Multiple smoke test versions
- Multiple training pipeline scripts
- Legacy runners

## Reorganization Strategy

### Phase 1: Clean Root Directory

#### Keep
- `.env`, `.gitignore`, `pytest.ini`
- `requirements.txt`
- `README.md` (consolidated version)
- `quick_start.bat`

#### Move to Archive
- `README_v2.md` → docs/archive/
- `PROJECT_MANUAL.md` → docs/archive/
- `詳細的改善報告.md` → docs/archive/legacy/
- `*.log` files → logs/

#### Delete
- `tmp_prefix.txt`
- `start_*.bat` (use quick_start.bat instead)

### Phase 2: Reorganize Documentation

New structure:
```
docs/
├── README.md (index)
├── user_guide/
│   ├── quick_start.md
│   ├── installation.md
│   ├── configuration.md
│   └── api_reference.md
├── architecture/
│   ├── system_design.md
│   ├── model_architecture.md
│   └── data_flow.md
├── development/
│   ├── contributing.md
│   ├── testing.md
│   └── deployment.md
└── archive/
    └── (old docs)
```

### Phase 3: Consolidate Scripts

New `scripts/` structure:
```
scripts/
├── README.md (script index)
├── setup/
│   ├── check_environment.py
│   └── build_scalers.py
├── training/
│   ├── train_mini.py (5k steps)
│   ├── train_small.py (100k steps)
│   └── train_full.py (2M steps)
├── testing/
│   ├── smoke_test.py (main)
│   └── e2e_test.py (simplified version)
├── live_trading/
│   ├── run_live_trading.py
│   └── headless_trading.py
└── archive/
    └── (deprecated scripts)
```

#### Scripts to Keep (Consolidated)
1. `simple_e2e_test.py` → `testing/e2e_test.py`
2. `complete_smoke_test.py` → `testing/smoke_test.py`
3. `check_cuda.py` → `setup/check_environment.py`
4. `build_scalers_from_oanda.py` → `setup/build_scalers.py`
5. `run_small_training_*` → Consolidate to `training/train_small.py`
6. `headless_api_trading.py` → `live_trading/run_trading.py`

#### Scripts to Archive
- `comprehensive_smoke_test.py` (duplicate)
- `run_complete_smoke_test.py` (duplicate)
- `legacy_training_pipeline.py` (legacy)
- `training_pipeline.py` (use new version)
- All test runners (consolidated)

### Phase 4: Reorganize Source Code

Keep current structure but clarify:
```
src/oanda_trading_bot/
├── __init__.py
├── common/          # Shared utilities
│   ├── model_registry.py (NEW)
│   └── ...
├── training_system/
│   ├── environment/
│   │   ├── enhanced_simulation.py (NEW)
│   │   └── symbol_reward_attribution.py (NEW)
│   └── ...
├── live_trading_system/
├── api/            # NEW: Unified API
└── backtester/     # Future
```

### Phase 5: Clean Up Tests

Current: 65 files in `tests/`
- Remove duplicates
- Organize by module
- Keep only functional tests

### Phase 6: Standardize Reports

```
reports/
├── README.md
├── smoke_tests/
├── e2e_tests/
├── training_results/
└── trading_logs/
```

## Implementation Steps

### Step 1: Backup
```bash
git add -A
git commit -m "Backup before reorganization"
git tag v2.0-pre-cleanup
```

### Step 2: Clean Root
- Move old READMEs
- Move logs
- Delete temp files
- Consolidate batch scripts

### Step 3: Reorganize docs/
- Create new structure
- Move files
- Create indexes

### Step 4: Consolidate scripts/
- Create subdirectories
- Move/rename scripts
- Archive old scripts
- Create script index

### Step 5: Update imports
- Fix all import paths
- Update documentation
- Update quick_start.bat

### Step 6: Test everything
- Run smoke test
- Verify all scripts
- Check documentation links

## Expected Benefits

1. **Clarity**: Clear folder structure
2. **Maintainability**: No duplicate files
3. **AI-Friendly**: Better structure for LLM understanding
4. **Professional**: Industry-standard organization
5. **Newbie-Friendly**: Clear entry points

## File Reduction Target

Current:
- Scripts: 28 → ~10 (65% reduction)
- Docs: Scattered → Organized
- Root files: 14 → 6 (57% reduction)

## Priority Order

1. Delete corrupted/temp files (DONE: end_to_end_test.py)
2. Move logs to logs/ directory
3. Consolidate README files
4. Reorganize scripts/
5. Reorganize docs/
6. Update all documentation
7. Final testing

---

Next: Execute cleanup plan
