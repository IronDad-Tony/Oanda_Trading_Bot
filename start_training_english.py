#!/usr/bin/env python3
"""
OANDA Universal Trading Model - Integrated Startup Script
Unified launcher with all fixes and optimizations applied
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
import logging

# Ensure src modules can be found
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def check_system_requirements():
    """Check system requirements"""
    logger.info("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    # Check required directories
    required_dirs = ['src', 'logs', 'data', 'weights']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Detected {gpu_count} GPU(s): {gpu_name}")
        else:
            logger.info("No GPU detected, will use CPU training")
    except ImportError:
        logger.warning("PyTorch not installed, please check dependencies")
        return False
    
    logger.info("System requirements check passed")
    return True

def cleanup_mmap_files():
    """Clean up old mmap files"""
    logger.info("Cleaning up old mmap files...")
    
    try:
        # Clean mmap files in data directory
        data_dir = Path('data')
        if data_dir.exists():
            mmap_files = list(data_dir.glob('*.mmap')) + list(data_dir.glob('*.dat'))
            for mmap_file in mmap_files:
                try:
                    mmap_file.unlink()
                    logger.info(f"Deleted mmap file: {mmap_file}")
                except Exception as e:
                    logger.warning(f"Could not delete {mmap_file}: {e}")
        
        # Clean temporary files
        temp_files = list(Path('.').glob('*.tmp')) + list(Path('.').glob('*.temp'))
        for temp_file in temp_files:
            try:
                temp_file.unlink()
                logger.info(f"Deleted temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Could not delete {temp_file}: {e}")
        
        logger.info("Mmap file cleanup completed")
        
    except Exception as e:
        logger.warning(f"Error during mmap file cleanup: {e}")

def main():
    """Main function"""
    
    print("=" * 60)
    print("OANDA Universal Trading Model Training System - Integrated")
    print("=" * 60)
    
    # Check system requirements
    if not check_system_requirements():
        print("System requirements check failed, please check environment")
        return False
    
    # Clean old files
    cleanup_mmap_files()
    
    try:
        # Import trainer
        from src.trainer.enhanced_trainer_complete import EnhancedUniversalTrainer, create_training_time_range
        from src.common.logger_setup import logger as common_logger
        from src.common.shared_data_manager import get_shared_data_manager
        
        logger.info("All modules imported successfully")
        
        # Configure training parameters
        trading_symbols = [
            "EUR_USD",    # Euro/US Dollar
            "USD_JPY",    # US Dollar/Japanese Yen
            "GBP_USD",    # British Pound/US Dollar
            "AUD_USD",    # Australian Dollar/US Dollar
            "USD_CAD",    # US Dollar/Canadian Dollar
        ]
        
        # Use last 30 days of data for training
        start_time, end_time = create_training_time_range(days_back=30)
        
        print(f"Training Configuration:")
        print(f"   Trading Symbols: {', '.join(trading_symbols)}")
        print(f"   Data Period: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Data Granularity: S5 (5 seconds)")
        print(f"   Training Steps: 50,000")
        print(f"   Save Frequency: Every 2,000 steps")
        print(f"   Eval Frequency: Every 5,000 steps")
        print()
        
        # Ask user confirmation
        response = input("Start training? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Training cancelled")
            return False
        
        # Initialize shared data manager
        shared_manager = get_shared_data_manager()
        shared_manager.clear_data()
        
        # Create trainer
        trainer = EnhancedUniversalTrainer(
            trading_symbols=trading_symbols,
            start_time=start_time,
            end_time=end_time,
            granularity="S5",
            total_timesteps=50000,      # 50K steps, about 30-60 minutes
            save_freq=2000,             # Save every 2K steps
            eval_freq=5000,             # Evaluate every 5K steps
            model_name_prefix="sac_universal_trader"
        )
        
        print("\nStarting complete training pipeline...")
        
        # Execute complete training pipeline
        success = trainer.run_full_training_pipeline()
        
        if success:
            print("\n" + "=" * 60)
            print("Training completed successfully!")
            print("=" * 60)
            print("Model files saved in: weights/ directory")
            print("TensorBoard logs: logs/sac_tensorboard_logs_*/")
            print("View training progress: tensorboard --logdir=logs/")
            print("Launch Streamlit UI: streamlit run streamlit_app_complete.py")
            print("=" * 60)
            return True
        else:
            print("\n" + "=" * 60)
            print("Training did not complete successfully")
            print("=" * 60)
            print("Please check log files for details")
            print("Run integration test: python integration_test.py")
            return False
            
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Training interrupted by user")
        print("=" * 60)
        print("Model automatically saved")
        return False
        
    except ImportError as e:
        print("\n" + "=" * 60)
        print(f"Module import error: {e}")
        print("=" * 60)
        print("Suggested solutions:")
        print("   1. Check all dependencies installed: pip install -r requirements.txt")
        print("   2. Run integration test: python integration_test.py")
        print("   3. Check Python path configuration")
        return False
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"Error during training: {e}")
        print("=" * 60)
        logger.error(f"Training error: {e}", exc_info=True)
        print("Suggested solutions:")
        print("   1. Run integration test: python integration_test.py")
        print("   2. Check log file: training.log")
        print("   3. Ensure sufficient disk space and memory")
        return False

def show_help():
    """Show help information"""
    print("OANDA AI Trading Bot - Integrated Startup Script")
    print()
    print("Usage:")
    print("  python start_training_english.py        # Start training")
    print("  python start_training_english.py --help # Show this help")
    print()
    print("Related commands:")
    print("  python integration_test.py              # Run integration test")
    print("  streamlit run streamlit_app_complete.py # Launch Web UI")
    print("  tensorboard --logdir=logs/              # View training progress")
    print()
    print("Troubleshooting:")
    print("  1. Ensure all dependencies installed: pip install -r requirements.txt")
    print("  2. Check Python version >= 3.8")
    print("  3. Ensure sufficient disk space (at least 5GB)")
    print("  4. If using GPU, ensure CUDA drivers properly installed")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        show_help()
        sys.exit(0)
    
    success = main()
    sys.exit(0 if success else 1)