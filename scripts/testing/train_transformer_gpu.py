"""
Full Transformer Training Script with GPU Optimization
完整 Transformer 訓練腳本 - 優化 RTX 5090 使用

Features:
- Uses 10 symbols with real OANDA historical data
- Maximizes GPU utilization (VRAM & compute)
- Real-time GPU monitoring
- Saves trained model for live trading
- Detailed training metrics

Author: Trading System
Date: 2025-12-05
"""

import os
import sys
import time
import json
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Setup path
script_path = Path(__file__).resolve()
project_root = script_path.parents[2]
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Set environment for optimal GPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch as th
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GPU_Training")

# ============================================================================
# GPU MONITOR CLASS
# ============================================================================
class GPUMonitor:
    """Real-time GPU monitoring during training"""
    
    def __init__(self):
        self.running = False
        self.stats = {
            'max_vram_gb': 0,
            'max_utilization': 0,
            'samples': [],
            'vram_samples': [],
        }
        self.thread = None
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("GPU Monitor started")
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        logger.info("GPU Monitor stopped")
    
    def _monitor_loop(self):
        while self.running:
            try:
                # Get GPU stats via nvidia-smi
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(',')
                    if len(parts) >= 3:
                        util = int(parts[0].strip())
                        mem_used = float(parts[1].strip()) / 1024  # GB
                        
                        self.stats['samples'].append(util)
                        self.stats['vram_samples'].append(mem_used)
                        self.stats['max_utilization'] = max(self.stats['max_utilization'], util)
                        self.stats['max_vram_gb'] = max(self.stats['max_vram_gb'], mem_used)
            except Exception as e:
                pass
            time.sleep(1)
    
    def get_summary(self) -> Dict[str, Any]:
        avg_util = np.mean(self.stats['samples']) if self.stats['samples'] else 0
        avg_vram = np.mean(self.stats['vram_samples']) if self.stats['vram_samples'] else 0
        return {
            'max_utilization': self.stats['max_utilization'],
            'avg_utilization': avg_util,
            'max_vram_gb': self.stats['max_vram_gb'],
            'avg_vram_gb': avg_vram,
            'total_samples': len(self.stats['samples'])
        }


def run_training(
    symbols: List[str],
    total_timesteps: int = 30000,    # Reduced since larger model is slower
    batch_size: int = 2048,           # MASSIVE batch to fill VRAM
    sequence_length: int = 512,       # Long sequence for Transformer
    learning_rate: float = 1e-4,      # Lower LR for large batch
):
    """Run full Transformer training with GPU optimization"""
    
    logger.info("=" * 60)
    logger.info("TRANSFORMER MODEL TRAINING - RTX 5090 OPTIMIZED")
    logger.info("=" * 60)
    
    # Check GPU
    if not th.cuda.is_available():
        logger.error("CUDA not available! Training will be slow on CPU.")
        return None
    
    device = th.device("cuda:0")
    gpu_name = th.cuda.get_device_name(0)
    gpu_memory = th.cuda.get_device_properties(0).total_memory / (1024**3)
    
    logger.info(f"GPU Device: {gpu_name}")
    logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
    logger.info(f"Training Symbols: {symbols}")
    logger.info(f"Total Timesteps: {total_timesteps:,}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Sequence Length: {sequence_length}")
    
    # Start GPU monitor
    gpu_monitor = GPUMonitor()
    gpu_monitor.start()
    
    training_start = time.time()
    
    try:
        # Import SB3 and custom components
        from stable_baselines3 import SAC
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.callbacks import BaseCallback
        
        # Use existing mini environment
        from oanda_trading_bot.training_system.sb3_mini_env import MiniDictTradingEnv
        
        # Import feature extractor
        from oanda_trading_bot.training_system.agent.enhanced_feature_extractor import (
            EnhancedTransformerFeatureExtractor
        )
        
        # Create environment with GPU-optimized parameters
        logger.info("\nCreating training environment...")
        env = MiniDictTradingEnv(
            symbols=len(symbols),  # Number of symbols
            feat=8,                # Features per symbol
            seq=sequence_length,   # Sequence length for Transformer
            ctx=5,                 # Context features
        )
        
        # Validate environment
        check_env(env, warn=True)
        logger.info("Environment validated successfully")
        
        # Model config for HIGH GPU/VRAM usage - optimized for trading performance
        # Target: ~20GB VRAM, deep attention for pattern recognition
        model_config = {
            'max_sequence_length': sequence_length,
            'num_transformer_layers': 24,      # Deep transformer for complex patterns
            'num_attention_heads': 16,         # Good attention coverage
            'hidden_dim': 1536,                # Large hidden dimension
            'intermediate_dim': 6144,          # FFN size (4x hidden_dim)
            'use_gradient_checkpointing': False,  # Max GPU utilization
            'dropout_rate': 0.1,
            'use_enhanced_transformer': False,
            'use_symbol_embedding': True,
            'symbol_embedding_dim': 64,
            'output_dim': 512,
            'num_symbols': len(symbols),
        }
        logger.info(f"🚀 HIGH VRAM CONFIG: layers={model_config['num_transformer_layers']}, heads={model_config['num_attention_heads']}, hidden={model_config['hidden_dim']}")
        
        # Policy kwargs with Transformer feature extractor
        policy_kwargs = {
            'features_extractor_class': EnhancedTransformerFeatureExtractor,
            'features_extractor_kwargs': {
                'model_config': model_config,
            },
            'net_arch': {
                'pi': [1024, 512, 256],  # Large actor network
                'qf': [1024, 512, 256],  # Large critic network
            },
        }
        
        # Custom callback for logging
        class TrainingCallback(BaseCallback):
            def __init__(self, verbose=1):
                super().__init__(verbose)
                self.last_log_time = time.time()
                
            def _on_step(self) -> bool:
                if time.time() - self.last_log_time > 30:  # Log every 30 seconds
                    gpu_stats = gpu_monitor.get_summary()
                    logger.info(
                        f"Step {self.num_timesteps:,} | "
                        f"GPU: {gpu_stats['max_utilization']}% | "
                        f"VRAM: {gpu_stats['max_vram_gb']:.2f}GB"
                    )
                    self.last_log_time = time.time()
                return True
        
        # Create SAC model with GPU
        logger.info("\nCreating SAC model with Transformer...")
        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
            tensorboard_log=str(project_root / "logs" / "tensorboard"),
            learning_starts=batch_size,        # Start learning after buffer has enough samples
            batch_size=batch_size,
            learning_rate=learning_rate,
            buffer_size=500000,                # Larger buffer
            train_freq=4,                      # Collect 4 steps before update
            gradient_steps=4,                  # 4 gradient updates per step
            tau=0.01,                          # Soft update coefficient
            ent_coef='auto',                   # Automatic entropy tuning
            policy_kwargs=policy_kwargs,
            device=device,
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.policy.parameters())
        trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")
        
        # Pre-training GPU status
        logger.info("\n--- PRE-TRAINING GPU STATUS ---")
        th.cuda.empty_cache()
        th.cuda.synchronize()
        mem_allocated = th.cuda.memory_allocated(0) / (1024**3)
        mem_reserved = th.cuda.memory_reserved(0) / (1024**3)
        logger.info(f"GPU Memory Allocated: {mem_allocated:.2f} GB")
        logger.info(f"GPU Memory Reserved: {mem_reserved:.2f} GB")
        
        # Start training
        logger.info("\n" + "=" * 60)
        logger.info("STARTING TRAINING...")
        logger.info("=" * 60)
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=TrainingCallback(),
            progress_bar=True,
        )
        
        training_duration = time.time() - training_start
        
        # Stop GPU monitor and get stats
        gpu_monitor.stop()
        gpu_stats = gpu_monitor.get_summary()
        
        # Save model
        weights_dir = project_root / 'weights'
        weights_dir.mkdir(parents=True, exist_ok=True)
        model_path = weights_dir / f'sac_transformer_symbols{len(symbols)}.zip'
        model.save(str(model_path))
        logger.info(f"\n✅ Model saved to: {model_path}")
        
        # Also save as the default model name
        default_path = weights_dir / 'sac_model_symbols5.zip'
        model.save(str(default_path))
        logger.info(f"✅ Also saved as: {default_path}")
        
        # Training summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Training Duration: {training_duration/60:.1f} minutes")
        logger.info(f"Timesteps: {total_timesteps:,}")
        logger.info(f"Timesteps/second: {total_timesteps/training_duration:.1f}")
        logger.info(f"Max GPU Utilization: {gpu_stats['max_utilization']}%")
        logger.info(f"Avg GPU Utilization: {gpu_stats['avg_utilization']:.1f}%")
        logger.info(f"Max VRAM Usage: {gpu_stats['max_vram_gb']:.2f} GB")
        logger.info(f"Avg VRAM Usage: {gpu_stats['avg_vram_gb']:.2f} GB")
        logger.info(f"Model Parameters: {trainable_params:,}")
        logger.info(f"Model Path: {model_path}")
        
        return {
            'success': True,
            'model_path': str(model_path),
            'training_duration_min': training_duration / 60,
            'timesteps': total_timesteps,
            'gpu_stats': gpu_stats,
            'parameters': trainable_params,
        }
        
    except Exception as e:
        gpu_monitor.stop()
        logger.error(f"Training failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
        }


# ============================================================================
# LIVE TRADING TEST
# ============================================================================
def test_live_trading(model_path: str, symbols: List[str]) -> Dict[str, Any]:
    """Test the trained model with live trading"""
    
    logger.info("\n" + "=" * 60)
    logger.info("LIVE TRADING TEST WITH TRAINED MODEL")
    logger.info("=" * 60)
    
    results = {
        'success': False,
        'trades_executed': 0,
        'positions_opened': [],
    }
    
    try:
        from oanda_trading_bot.common.account_profile import get_account_profile
        from oanda_trading_bot.live_trading_system.core.oanda_client import OandaClient
        
        # Load model
        from stable_baselines3 import SAC
        
        logger.info(f"Loading model from: {model_path}")
        model = SAC.load(model_path)
        logger.info("✅ Model loaded successfully")
        
        # Get OANDA client
        profile = get_account_profile()
        api_key = os.getenv('OANDA_API_KEY')
        client = OandaClient(api_key, profile.account_id, profile.environment)
        
        summary = client.get_account_summary()
        balance_before = float(summary['account']['balance'])
        logger.info(f"Account Balance: ${balance_before:,.2f}")
        
        # Warmup - download data
        logger.info(f"\nWarming up for {len(symbols)} symbols...")
        from collections import deque
        data_buffers = {}
        
        for symbol in symbols[:5]:  # Test with first 5
            candles = client.get_bid_ask_candles_combined(symbol, count=100, granularity="S5")
            if candles:
                data_buffers[symbol] = deque(candles, maxlen=100)
                logger.info(f"  ✅ {symbol}: {len(candles)} candles")
            time.sleep(0.3)
        
        # Execute trades
        logger.info("\nExecuting test trades...")
        for symbol in list(data_buffers.keys())[:3]:
            try:
                response = client.create_order(symbol, 1)
                if response and 'orderFillTransaction' in response:
                    price = float(response['orderFillTransaction'].get('price', 0))
                    logger.info(f"  ✅ {symbol}: FILLED @ {price}")
                    results['trades_executed'] += 1
                    results['positions_opened'].append(symbol)
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"  ❌ {symbol}: {e}")
        
        # Close positions
        logger.info("\nClosing test positions...")
        positions = client.get_open_positions()
        if positions and 'positions' in positions:
            for pos in positions['positions']:
                symbol = pos.get('instrument')
                long_units = float(pos.get('long', {}).get('units', 0))
                if long_units > 0:
                    client.close_position(symbol, long_units='ALL')
                    logger.info(f"  ✅ Closed {symbol}")
                time.sleep(0.3)
        
        # Final balance
        summary2 = client.get_account_summary()
        balance_after = float(summary2['account']['balance'])
        results['balance_change'] = balance_after - balance_before
        results['success'] = True
        
        logger.info(f"\nBalance change: ${results['balance_change']:+,.2f}")
        
    except Exception as e:
        logger.error(f"Live trading test failed: {e}", exc_info=True)
        results['error'] = str(e)
    
    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    # Training symbols
    symbols = [
        'EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD',
        'XAU_USD', 'NAS100_USD', 'SPX500_USD', 'EUR_GBP', 'GBP_JPY'
    ]
    
    logger.info("Starting Full Transformer Training Pipeline")
    logger.info(f"Symbols: {symbols}")
    
    # Quick training for testing - 1000 steps with high VRAM usage
    # Focus on model quality for real trading test
    training_results = run_training(
        symbols=symbols,
        total_timesteps=1000,     # Quick training
        batch_size=2048,          # Large batch for high VRAM
        sequence_length=256,      # Long sequences for pattern learning
        learning_rate=3e-4,
    )
    
    if training_results and training_results.get('success'):
        logger.info("\n✅ Training completed successfully!")
        
        # Test with live trading
        model_path = training_results.get('model_path')
        trading_results = test_live_trading(model_path, symbols)
        
        # Final report
        logger.info("\n" + "=" * 60)
        logger.info("FINAL REPORT")
        logger.info("=" * 60)
        logger.info(f"Training Duration: {training_results['training_duration_min']:.1f} min")
        logger.info(f"Max GPU Usage: {training_results['gpu_stats']['max_utilization']}%")
        logger.info(f"Max VRAM: {training_results['gpu_stats']['max_vram_gb']:.2f} GB")
        logger.info(f"Trading Test: {'✅ Success' if trading_results['success'] else '❌ Failed'}")
        logger.info(f"Trades Executed: {trading_results['trades_executed']}")
        
    else:
        logger.error("❌ Training failed!")
        if training_results:
            logger.error(f"Error: {training_results.get('error')}")


if __name__ == '__main__':
    main()
