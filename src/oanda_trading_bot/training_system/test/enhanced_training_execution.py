#!/usr/bin/env python3
"""
Enhanced Training Execution Script - Phase 5 Real Implementation
=====================================

This script executes real historical data training with all enhanced components:
- Enhanced Transformer (768 dim, 16 layers, 24 heads)
- Enhanced Quantum Strategy Layer (15 complete strategies)
- Progressive Learning System
- Meta-Learning Optimizer
- Real Historical Data Training

Target Performance Metrics:
- Sharpe Ratio: >2.0
- Max Drawdown: <5%
- Win Rate: >65%
"""

import os
import sys
import torch
import logging
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Optional

# Ensure logs directory exists
project_root = Path(__file__).parent
logs_dir = project_root / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

# Setup logging
log_filename = logs_dir / f'enhanced_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_training_execution")

def check_system_requirements():
    """Check if all enhanced components are available"""
    logger.info("🔍 Checking enhanced component availability...")
    
    try:
        # Check Enhanced Transformer
        from oanda_trading_bot.training_system.models.enhanced_transformer import EnhancedTransformer as EnhancedUniversalTradingTransformer
        logger.info("✅ Enhanced Transformer - Available")
        
        # Check Enhanced Quantum Strategy Layer
        from oanda_trading_bot.training_system.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition
        logger.info("✅ Enhanced Quantum Strategy Layer (15 strategies) - Available")
        
        # Check Enhanced Feature Extractor
        from oanda_trading_bot.training_system.agent.enhanced_feature_extractor import EnhancedTransformerFeatureExtractor
        logger.info("✅ Enhanced Feature Extractor - Available")
        
        # Check Progressive Learning System
        from oanda_trading_bot.training_system.environment.progressive_learning_system import ProgressiveLearningSystem
        logger.info("✅ Progressive Learning System - Available")
        
        # Check Meta-Learning Optimizer
        from oanda_trading_bot.training_system.agent.meta_learning_optimizer import MetaLearningOptimizer
        logger.info("✅ Meta-Learning Optimizer - Available")
        
        # Check Universal Trainer
        from oanda_trading_bot.training_system.trainer.universal_trainer import UniversalTrainer
        logger.info("✅ Universal Trainer - Available")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Missing enhanced component: {e}")
        return False

def create_enhanced_training_configuration():
    """Create comprehensive training configuration for enhanced model"""
      # Enhanced trading symbols - diverse currency pairs (limited to 5 max)
    trading_symbols = [
        "EUR_USD",  # Major - Euro/US Dollar
        "USD_JPY",  # Major - US Dollar/Japanese Yen
        "GBP_USD",  # Major - British Pound/US Dollar
        "AUD_USD",  # Major - Australian Dollar/US Dollar
        "USD_CAD",  # Major - US Dollar/Canadian Dollar
    ]    # Get current time and find the most recent weekday with market data
    current_time = datetime.now(timezone.utc)
    
    # Find the most recent trading day (weekday)
    # If today is weekend (Saturday=5, Sunday=6), go back to Friday
    # If today is Monday and markets may not be fully open, use previous Friday
    weekday = current_time.weekday()
    
    if weekday == 5:  # Saturday
        days_back = 1  # Go to Friday
    elif weekday == 6:  # Sunday
        days_back = 2  # Go to Friday
    elif weekday == 0 and current_time.hour < 12:  # Early Monday
        days_back = 3  # Go to Friday to ensure data availability
    else:
        days_back = 0  # Use current day if it's a weekday
    
    # Calculate end time (use previous trading day if needed)
    end_time = current_time - timedelta(days=days_back)
    
    # Ensure we're using trading hours (around market close time)
    end_time = end_time.replace(hour=21, minute=0, second=0, microsecond=0)  # 9 PM UTC
    
    # 6 hours of weekday data for comprehensive but quick testing
    start_time = end_time - timedelta(hours=6)
      # Enhanced training parameters (optimized for half-day S5 data)
    training_config = {
        'trading_symbols': trading_symbols,
        'start_time': start_time,
        'end_time': end_time,
        'granularity': "S5",  # 5-second granularity for detailed learning
        'total_timesteps': 100_000,  # Reduced for half-day S5 data testing
        'save_freq': 10_000,  # More frequent saves
        'eval_freq': 5_000,   # More frequent evaluation
        'initial_capital': 100_000.0,  # Substantial capital for realistic testing
        'risk_percentage': 2.0,  # Conservative risk management
        'atr_stop_loss_multiplier': 2.5,  # Adaptive stop loss
        'max_position_percentage': 15.0,  # Reasonable position sizing
    }
    
    logger.info("📋 Enhanced Training Configuration:")
    logger.info(f"   • Symbols: {len(trading_symbols)} pairs")
    logger.info(f"   • Time Range: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
    logger.info(f"   • Training Steps: {training_config['total_timesteps']:,}")
    logger.info(f"   • Initial Capital: ${training_config['initial_capital']:,.2f}")
    
    return training_config

def execute_enhanced_training():
    """Execute enhanced training with comprehensive monitoring"""
    logger.info("🚀 Starting Enhanced Training Execution...")
    logger.info("=" * 80)
    
    try:
        # Import enhanced trainer
        from oanda_trading_bot.training_system.trainer.universal_trainer import UniversalTrainer
        
        # Get training configuration
        config = create_enhanced_training_configuration()
        
        # Create enhanced trainer with all parameters
        logger.info("🏗️ Creating Enhanced Universal Trainer...")
        trainer = UniversalTrainer(
            trading_symbols=config['trading_symbols'],
            start_time=config['start_time'],
            end_time=config['end_time'],
            granularity=config['granularity'],
            total_timesteps=config['total_timesteps'],
            save_freq=config['save_freq'],
            eval_freq=config['eval_freq'],
            model_name_prefix="enhanced_sac_trader",
            initial_capital=config['initial_capital'],
            risk_percentage=config['risk_percentage'],
            atr_stop_loss_multiplier=config['atr_stop_loss_multiplier'],
            max_position_percentage=config['max_position_percentage']
        )
        
        logger.info("✅ Enhanced Universal Trainer created successfully")
        
        # Display enhanced model information
        logger.info("🧠 Enhanced Model Specifications:")
        logger.info("   • Transformer: 768 dimensions, 16 layers, 24 attention heads")
        logger.info("   • Quantum Strategies: 15 complete trading strategies")
        logger.info("   • Progressive Learning: 3-stage advancement system")
        logger.info("   • Meta-Learning: MAML-based adaptation")
        logger.info("   • Feature Extraction: Multi-scale temporal analysis")
        
        # Execute full training pipeline
        logger.info("🎯 Executing Enhanced Training Pipeline...")
        logger.info("=" * 80)
        
        training_start_time = datetime.now(timezone.utc)
        success = trainer.run_full_training_pipeline()
        training_end_time = datetime.now(timezone.utc)
        
        training_duration = training_end_time - training_start_time
        
        # Report training results
        logger.info("=" * 80)
        if success:
            logger.info("🎉 ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
            logger.info(f"   • Duration: {training_duration}")
            logger.info(f"   • Total Steps: {config['total_timesteps']:,}")
            logger.info(f"   • Symbols Trained: {len(config['trading_symbols'])}")
            
            # Performance validation
            logger.info("📊 Validating Performance Metrics...")
            validate_performance_metrics(trainer)
            
        else:
            logger.error("❌ Enhanced training failed")
            logger.error("   • Check logs for detailed error information")
            
        return success
        
    except Exception as e:
        logger.error(f"💥 Critical error during enhanced training: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return False

def validate_performance_metrics(trainer):
    """Validate that the training achieved target performance metrics"""
    logger.info("🎯 Performance Validation:")
    
    try:
        # Get shared data manager for metrics
        shared_manager = trainer.shared_data_manager
        training_summary = shared_manager.get_training_summary()
        
        if training_summary:
            # Extract key metrics
            total_return = training_summary.get('total_return', 0)
            max_drawdown = training_summary.get('max_drawdown', 0)
            volatility = training_summary.get('volatility', 0)
            
            # Calculate approximate Sharpe ratio (simplified)
            if volatility > 0:
                sharpe_ratio = total_return / volatility
            else:
                sharpe_ratio = 0
            
            # Calculate win rate (simplified from available data)
            all_trades = shared_manager.get_all_trades()
            if all_trades:
                profitable_trades = sum(1 for trade in all_trades if trade.get('profit_loss', 0) > 0)
                win_rate = (profitable_trades / len(all_trades)) * 100
            else:
                win_rate = 0
            
            # Log performance metrics
            logger.info(f"   • Total Return: {total_return:.2f}%")
            logger.info(f"   • Sharpe Ratio: {sharpe_ratio:.2f} (Target: >2.0)")
            logger.info(f"   • Max Drawdown: {max_drawdown:.2f}% (Target: <5%)")
            logger.info(f"   • Win Rate: {win_rate:.2f}% (Target: >65%)")
            logger.info(f"   • Volatility: {volatility:.2f}%")
            
            # Performance assessment
            targets_met = 0
            total_targets = 3
            
            if sharpe_ratio > 2.0:
                logger.info("   ✅ Sharpe Ratio Target MET")
                targets_met += 1
            else:
                logger.warning("   ⚠️  Sharpe Ratio Target NOT MET")
            
            if abs(max_drawdown) < 5.0:
                logger.info("   ✅ Max Drawdown Target MET")
                targets_met += 1
            else:
                logger.warning("   ⚠️  Max Drawdown Target NOT MET")
            
            if win_rate > 65.0:
                logger.info("   ✅ Win Rate Target MET")
                targets_met += 1
            else:
                logger.warning("   ⚠️  Win Rate Target NOT MET")
            
            # Overall assessment
            performance_score = (targets_met / total_targets) * 100
            logger.info(f"   📈 Overall Performance Score: {performance_score:.1f}%")
            
            if targets_met == total_targets:
                logger.info("   🏆 ALL PERFORMANCE TARGETS ACHIEVED!")
            elif targets_met >= 2:
                logger.info("   🎯 MAJORITY OF TARGETS ACHIEVED - Good Performance")
            else:
                logger.info("   📚 ADDITIONAL TRAINING RECOMMENDED")
                
        else:
            logger.warning("   ⚠️  No training summary available for validation")
            
    except Exception as e:
        logger.error(f"   ❌ Error during performance validation: {e}")

def main():
    """Main execution function"""
    logger.info("🚀 Enhanced Oanda Trading Bot - Real Training Execution")
    logger.info("=" * 80)
    logger.info("Phase 5: Complete Implementation with Real Historical Data")
    logger.info(f"Start Time: {datetime.now()}")
    logger.info("=" * 80)
    
    # Check system requirements
    if not check_system_requirements():
        logger.error("❌ System requirements not met. Please ensure all enhanced components are available.")
        return False
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"🎮 GPU Available: {torch.cuda.get_device_name()}")
        logger.info(f"   • Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("💻 Using CPU for training")
    
    # Execute enhanced training
    success = execute_enhanced_training()
    
    # Final report
    logger.info("=" * 80)
    if success:
        logger.info("🎉 ENHANCED TRAINING EXECUTION COMPLETED SUCCESSFULLY!")
        logger.info("📊 Review the performance metrics above")
        logger.info("💾 Model weights saved in weights/ directory")
        logger.info("📋 Training logs available in logs/ directory")
    else:
        logger.error("❌ Enhanced training execution failed")
        logger.error("🔍 Check the error logs for troubleshooting")
    
    logger.info(f"End Time: {datetime.now()}")
    logger.info("=" * 80)
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("⏹️  Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        sys.exit(1)
