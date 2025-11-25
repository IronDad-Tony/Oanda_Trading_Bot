"""
Complete Path Verification Script
Checks all paths used in the project after reorganization
"""
import sys
from pathlib import Path
import logging

# Setup - scripts/setup/verify_paths.py is 2 levels deep
project_root = Path(__file__).resolve().parents[2]
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def check_path(path_str, description):
    """Check if a path exists"""
    path = project_root / path_str
    exists = path.exists()
    status = "[OK]" if exists else "[ERROR]"
    logger.info(f"{status} {description:<50} {path_str}")
    return exists

def main():
    logger.info("="*80)
    logger.info("PATH VERIFICATION REPORT")
    logger.info("="*80)
    logger.info(f"Project Root: {project_root}\n")
    
    all_pass = True
    
    # Core Files
    logger.info("=" * 80)
    logger.info("CORE FILES")
    logger.info("=" * 80)
    all_pass &= check_path("README.md", "Main README")
    all_pass &= check_path("requirements.txt", "Requirements")
    all_pass &= check_path(".env", ".env file (credentials)")
    all_pass &= check_path("quick_start.bat", "Quick Start Script")
    
    # Scripts
    logger.info("\n" + "=" * 80)
    logger.info("SCRIPTS (Reorganized)")
    logger.info("=" * 80)
    all_pass &= check_path("scripts/README.md", "Scripts Index")
    all_pass &= check_path("scripts/setup/env_check.py", "Environment Check")
    all_pass &= check_path("scripts/setup/check_cuda.py", "CUDA Check")
    all_pass &= check_path("scripts/testing/complete_smoke_test.py", "Smoke Test")
    all_pass &= check_path("scripts/testing/simple_e2e_test.py", "E2E Test")
    all_pass &= check_path("scripts/training/train_quick_sac.py", "Quick Training")
    all_pass &= check_path("scripts/live_trading/run_small_live_cycle.py", "Small Live Cycle")
    
    # Documentation
    logger.info("\n" + "=" * 80)
    logger.info("DOCUMENTATION")
    logger.info("=" * 80)
    all_pass &= check_path("docs/NAVIGATION.md", "Navigation Guide")
    all_pass &= check_path(".agent/project_analysis_report.md", "Project Analysis")
    all_pass &= check_path(".agent/improvement_plan.md", "Improvement Plan")
    all_pass &= check_path(".agent/upgrade_complete_report.md", "Upgrade Report")
    all_pass &= check_path(".agent/UI_ARCHITECTURE_REPORT.md", "UI Architecture")
    
    # Source Code - Core
    logger.info("\n" + "=" * 80)
    logger.info("SOURCE CODE - CORE SYSTEMS")
    logger.info("=" * 80)
    all_pass &= check_path("src/oanda_trading_bot/__init__.py", "Main Package Init")
    all_pass &= check_path("src/oanda_trading_bot/common/model_registry.py", "Model Registry (v2.0)")
    all_pass &= check_path("src/oanda_trading_bot/training_system/environment/enhanced_simulation.py", "Enhanced Simulation (v2.0)")
    all_pass &= check_path("src/oanda_trading_bot/training_system/environment/symbol_reward_attribution.py", "Symbol Reward Attribution (v2.0)")
    
    # Source Code - APIs & UIs
    logger.info("\n" + "=" * 80)
    logger.info("SOURCE CODE - APIs & UIs")
    logger.info("=" * 80)
    all_pass &= check_path("src/oanda_trading_bot/api/main.py", "Unified API Backend (v2.0)")
    all_pass &= check_path("src/oanda_trading_bot/training_system/app.py", "Training UI (Streamlit)")
    all_pass &= check_path("src/oanda_trading_bot/live_trading_system/web_ui/main.py", "Live Web UI (FastAPI)")
    all_pass &= check_path("src/oanda_trading_bot/live_trading_system/ui/index.html", "Live UI - HTML")
    all_pass &= check_path("src/oanda_trading_bot/live_trading_system/main.py", "Live Trading Main")
    
    # Frontend
    logger.info("\n" + "=" * 80)
    logger.info("FRONTEND (React)")
    logger.info("=" * 80)
    all_pass &= check_path("frontend/package.json", "Frontend Package Config")
    all_pass &= check_path("frontend/tsconfig.json", "TypeScript Config")
    all_pass &= check_path("frontend/src", "Frontend Source Directory")
    
    # Directories
    logger.info("\n" + "=" * 80)
    logger.info("DIRECTORIES")
    logger.info("=" * 80)
    all_pass &= check_path("logs", "Logs Directory")
    all_pass &= check_path("reports", "Reports Directory")
    all_pass &= check_path("model_registry", "Model Registry Directory")
    all_pass &= check_path("weights", "Weights Directory")
    all_pass &= check_path("configs", "Configs Directory")
    
    # Test Paths Used in quick_start.bat
    logger.info("\n" + "=" * 80)
    logger.info("QUICK_START.BAT PATHS")
    logger.info("=" * 80)
    
    paths_in_bat = {
        "src\\oanda_trading_bot\\api": "API Backend Path",
        "src\\oanda_trading_bot\\training_system\\app.py": "Training UI Path",
        "src\\oanda_trading_bot\\live_trading_system\\web_ui": "Live Web UI Path",
        "frontend": "Frontend Path",
        "scripts\\testing\\complete_smoke_test.py": "Smoke Test Path",
        "scripts\\training\\train_quick_sac.py": "Mini Train Path",
    }
    
    for path_str, desc in paths_in_bat.items():
        all_pass &= check_path(path_str.replace("\\", "/"), desc)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    if all_pass:
        logger.info("[SUCCESS] All paths verified successfully!")
        logger.info("\nThe project structure is correct and all files are in place.")
        return 0
    else:
        logger.warning("[WARNING] Some paths are missing!")
        logger.warning("\nPlease check the [ERROR] items above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
