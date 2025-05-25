# src/trainer/enhanced_trainer_complete.py
"""
Enhanced Universal Trainer - Complete Implementation
Integrated intelligent currency management and automatic data download
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import numpy as np
import time
import torch
import gc
import os

# Import logic same as other modules
logger: logging.Logger = logging.getLogger("enhanced_trainer_module_init")
_logger_initialized_by_common_et = False

try:
    from common.logger_setup import logger as common_configured_logger
    logger = common_configured_logger
    _logger_initialized_by_common_et = True
    logger.debug("enhanced_trainer_complete.py: Successfully imported logger from common.logger_setup.")
    
    from common.config import (
        TIMESTEPS, MAX_SYMBOLS_ALLOWED, ACCOUNT_CURRENCY, INITIAL_CAPITAL,
        OANDA_MARGIN_CLOSEOUT_LEVEL, TRADE_COMMISSION_PERCENTAGE, OANDA_API_KEY,
        ATR_PERIOD, STOP_LOSS_ATR_MULTIPLIER, MAX_ACCOUNT_RISK_PERCENTAGE,
        LOGS_DIR, DEVICE, USE_AMP
    )
    logger.info("enhanced_trainer_complete.py: Successfully imported common.config values.")
    
    # Import shared data manager
    from common.shared_data_manager import get_shared_data_manager
    logger.info("enhanced_trainer_complete.py: Successfully imported shared data manager.")
    
    # Import required modules
    from data_manager.currency_manager import CurrencyDependencyManager, ensure_currency_data_for_trading
    from data_manager.mmap_dataset import UniversalMemoryMappedDataset
    from data_manager.instrument_info_manager import InstrumentInfoManager
    from data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols
    from environment.trading_env import UniversalTradingEnvV4
    from agent.sac_agent_wrapper import SACAgentWrapper
    from trainer.callbacks import UniversalCheckpointCallback
    logger.info("enhanced_trainer_complete.py: Successfully imported other dependencies.")
    
except ImportError as e_initial_import_et:
    logger_temp_et = logging.getLogger("enhanced_trainer_fallback_initial")
    logger_temp_et.addHandler(logging.StreamHandler(sys.stdout))
    logger_temp_et.setLevel(logging.DEBUG)
    logger = logger_temp_et
    logger.warning(f"enhanced_trainer_complete.py: Initial import failed: {e_initial_import_et}. Assuming PYTHONPATH is set correctly or this is a critical issue.")
    
    try:
        # Assume PYTHONPATH is set, these imports should work
        from src.common.logger_setup import logger as common_logger_retry_et
        logger = common_logger_retry_et
        _logger_initialized_by_common_et = True
        logger.info("enhanced_trainer_complete.py: Successfully re-imported common_logger after path adj.")
        
        from src.common.config import (
            TIMESTEPS, MAX_SYMBOLS_ALLOWED, ACCOUNT_CURRENCY, INITIAL_CAPITAL,
            OANDA_MARGIN_CLOSEOUT_LEVEL, TRADE_COMMISSION_PERCENTAGE, OANDA_API_KEY,
            ATR_PERIOD, STOP_LOSS_ATR_MULTIPLIER, MAX_ACCOUNT_RISK_PERCENTAGE,
            LOGS_DIR, DEVICE, USE_AMP
        )
        logger.info("enhanced_trainer_complete.py: Successfully re-imported common.config after path adjustment.")
        
        # Import shared data manager
        from src.common.shared_data_manager import get_shared_data_manager
        logger.info("enhanced_trainer_complete.py: Successfully re-imported shared data manager.")
        
        from src.data_manager.currency_manager import CurrencyDependencyManager, ensure_currency_data_for_trading
        from src.data_manager.mmap_dataset import UniversalMemoryMappedDataset
        from src.data_manager.instrument_info_manager import InstrumentInfoManager
        from src.data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols
        from src.environment.trading_env import UniversalTradingEnvV4
        from src.agent.sac_agent_wrapper import SACAgentWrapper
        from src.trainer.callbacks import UniversalCheckpointCallback
        logger.info("enhanced_trainer_complete.py: Successfully re-imported other dependencies after path adjustment.")
        
    except ImportError as e_retry_critical_et:
        logger.error(f"enhanced_trainer_complete.py: Critical import error after path adjustment: {e_retry_critical_et}", exc_info=True)
        logger.warning("enhanced_trainer_complete.py: Using fallback mode - some features may not work.")
        
        # Create fallback classes
        class CurrencyDependencyManager:
            def __init__(self, **kwargs):
                pass
            def download_required_currency_data(self, *args, **kwargs):
                return True
        
        def ensure_currency_data_for_trading(*args, **kwargs):
            return True
        
        def get_shared_data_manager():
            """Fallback shared data manager"""
            class FallbackManager:
                def update_training_status(self, *args, **kwargs): pass
                def is_stop_requested(self): return False
                def request_stop(self): pass
                def add_training_metric(self, *args, **kwargs): pass
                def add_trade_record(self, *args, **kwargs): pass
            return FallbackManager()


class EnhancedUniversalTrainer:
    """
    Enhanced Universal Trading Model Trainer
    
    Features:
    1. Intelligent currency dependency management - automatically determine and download required currency pairs
    2. Automatic data preparation - ensure all necessary data is complete
    3. Complete training pipeline - one-stop from data to model
    4. Real-time monitoring and callbacks - support checkpoint resumption
    """
    
    def __init__(self,
                 trading_symbols: List[str],
                 start_time: datetime,
                 end_time: datetime,
                 granularity: str = "S5",
                 timesteps_history: int = TIMESTEPS,
                 account_currency: str = ACCOUNT_CURRENCY,
                 initial_capital: float = float(INITIAL_CAPITAL),
                 max_episode_steps: Optional[int] = None,
                 total_timesteps: int = 10000,
                 save_freq: int = 1000,
                 eval_freq: int = 2000,
                 model_name_prefix: str = "sac_universal_trader",
                 streamlit_session_state=None):
        
        self.trading_symbols = sorted(list(set(trading_symbols)))
        self.start_time = start_time
        self.end_time = end_time
        self.granularity = granularity
        self.timesteps_history = timesteps_history
        self.account_currency = account_currency.upper()
        self.initial_capital = initial_capital
        self.max_episode_steps = max_episode_steps
        self.total_timesteps = total_timesteps
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.model_name_prefix = model_name_prefix
        self.streamlit_session_state = streamlit_session_state  # For updating Streamlit UI
        
        # Initialize shared data manager
        self.shared_data_manager = get_shared_data_manager()
        logger.info("Connected to shared data manager")
        
        # Setup GPU optimization
        self._setup_gpu_optimization()
        
        # Generate parameter-based model name
        self.model_identifier = self._generate_model_identifier()
        self.existing_model_path = self._find_existing_model()
        
        # Initialize components
        self.currency_manager = CurrencyDependencyManager(account_currency)
        self.instrument_manager = InstrumentInfoManager(force_refresh=False)
        
        # Will be initialized in prepare_data
        self.dataset = None
        self.env = None
        self.agent = None
        self.callback = None
        self._stop_training = False  # Training stop flag
        
        # Training data collection
        self.training_start_time = None
        
        logger.info(f"EnhancedUniversalTrainer initialization complete")
        logger.info(f"Trading symbols: {self.trading_symbols}")
        logger.info(f"Time range: {self.start_time} to {self.end_time}")
        logger.info(f"Account currency: {self.account_currency}")
        logger.info(f"Model identifier: {self.model_identifier}")
        if self.existing_model_path:
            logger.info(f"Found existing model: {self.existing_model_path}")
        else:
            logger.info("No existing model found, will create new model")
    
    def _setup_gpu_optimization(self):
        """Setup GPU optimization configuration"""
        try:
            if torch.cuda.is_available():
                # Check GPU information
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
                
                logger.info(f"Detected {gpu_count} GPU device(s)")
                logger.info(f"Currently using GPU {current_device}: {gpu_name} ({gpu_memory:.1f}GB)")
                
                # Clear GPU memory
                torch.cuda.empty_cache()
                gc.collect()
                
                # Setup GPU memory management
                torch.cuda.set_per_process_memory_fraction(0.85)  # Use 85% of GPU memory
                
                # Enable cuDNN optimization
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                
                # Enable TensorFloat-32 (TF32) for improved Ampere architecture GPU performance
                if hasattr(torch.backends.cuda, 'matmul'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                
                # Set environment variables to optimize GPU memory allocation
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
                
                # If mixed precision training is enabled
                if USE_AMP:
                    logger.info("Mixed precision training enabled")
                    # Check if GPU supports mixed precision
                    if torch.cuda.get_device_capability(current_device)[0] >= 7:  # Volta architecture or newer
                        logger.info("GPU supports mixed precision training")
                    else:
                        logger.warning("GPU may not fully support mixed precision training, but will still attempt to use it")
                
                logger.info("GPU optimization setup complete")
                
            else:
                logger.info("CUDA not detected, will use CPU training")
                # CPU optimization settings
                torch.set_num_threads(min(8, torch.get_num_threads()))  # Limit CPU thread count
                logger.info(f"CPU thread count set to: {torch.get_num_threads()}")
                
        except Exception as e:
            logger.warning(f"Error occurred during GPU optimization setup: {e}")
    
    def _generate_model_identifier(self) -> str:
        """
        Generate model identifier based on key parameters
        
        Returns:
            Model identifier string
        """
        # Use number of trading symbols and timesteps as main parameters
        max_symbols = len(self.trading_symbols)
        timestep = self.timesteps_history
        
        # Generate identifier: sac_model_symbols{count}_timestep{steps}
        identifier = f"sac_model_symbols{max_symbols}_timestep{timestep}"
        return identifier
    
    def _find_existing_model(self) -> Optional[str]:
        """
        Find if a model with same parameters exists
        
        Returns:
            Existing model path, or None if not found
        """
        try:
            # Check multiple possible locations
            search_paths = [
                Path("weights"),
                Path("logs"),
                Path(LOGS_DIR) if LOGS_DIR else None
            ]
            
            for search_path in search_paths:
                if search_path is None or not search_path.exists():
                    continue
                
                # Find matching model files
                pattern = f"{self.model_identifier}*.zip"
                matching_files = list(search_path.glob(pattern))
                
                if matching_files:
                    # Return the newest file
                    latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
                    return str(latest_file)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error occurred while finding existing model: {e}")
            return None
    
    def get_model_save_path(self, suffix: str = "") -> Path:
        """
        Get model save path
        
        Args:
            suffix: Filename suffix
            
        Returns:
            Model save path
        """
        if suffix:
            filename = f"{self.model_identifier}_{suffix}.zip"
        else:
            filename = f"{self.model_identifier}.zip"
        
        # Modified: Store models in /weights folder instead of /logs
        save_dir = Path("weights")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        return save_dir / filename
    
    def prepare_data(self) -> bool:
        """
        Prepare training data
        
        Returns:
            Whether data preparation was successful
        """
        try:
            logger.info("Starting data preparation...")
            
            # 1. Download required currency data
            logger.info("Downloading required currency data...")
            success = self.currency_manager.download_required_currency_data(
                self.trading_symbols,
                self.start_time,
                self.end_time,
                self.granularity
            )
            
            if not success:
                logger.error("Failed to download required currency data")
                return False
            
            # 2. Ensure currency data for trading
            logger.info("Ensuring currency data for trading...")
            success = ensure_currency_data_for_trading(
                self.trading_symbols,
                self.account_currency,
                self.start_time,
                self.end_time,
                self.granularity
            )
            
            if not success:
                logger.error("Failed to ensure currency data for trading")
                return False
            
            # 3. Create dataset
            logger.info("Creating memory-mapped dataset...")
            self.dataset = UniversalMemoryMappedDataset(
                symbols=self.trading_symbols,
                start_time=self.start_time,
                end_time=self.end_time,
                granularity=self.granularity,
                timesteps=self.timesteps_history
            )
            
            if not self.dataset.is_valid():
                logger.error("Dataset creation failed or is invalid")
                return False
            
            logger.info(f"Dataset created successfully with {len(self.dataset)} samples")
            logger.info("Data preparation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}", exc_info=True)
            return False
    
    def setup_environment(self) -> bool:
        """
        Setup trading environment
        
        Returns:
            Whether environment setup was successful
        """
        try:
            if self.dataset is None:
                logger.error("Dataset not prepared, please call prepare_data() first")
                return False
            
            logger.info("Setting up trading environment...")
            
            # Create trading environment
            self.env = UniversalTradingEnvV4(
                dataset=self.dataset,
                initial_capital=self.initial_capital,
                max_episode_steps=self.max_episode_steps,
                account_currency=self.account_currency,
                commission_percentage=TRADE_COMMISSION_PERCENTAGE,
                margin_closeout_level=OANDA_MARGIN_CLOSEOUT_LEVEL,
                atr_period=ATR_PERIOD,
                stop_loss_atr_multiplier=STOP_LOSS_ATR_MULTIPLIER,
                max_account_risk_percentage=MAX_ACCOUNT_RISK_PERCENTAGE
            )
            
            logger.info("Trading environment setup successful")
            logger.info(f"Environment observation space: {self.env.observation_space}")
            logger.info(f"Environment action space: {self.env.action_space}")
            
            return True
            
        except Exception as e:
            logger.error(f"Environment setup failed: {e}", exc_info=True)
            return False
    
    def setup_agent(self, load_model_path: Optional[str] = None) -> bool:
        """
        Setup SAC agent
        
        Args:
            load_model_path: Optional model loading path (for checkpoint resumption)
            
        Returns:
            Whether agent setup was successful
        """
        try:
            if self.env is None:
                logger.error("Environment not setup, please call setup_environment() first")
                return False
            
            logger.info("Setting up SAC agent...")
            
            # Determine model loading path
            model_path_to_load = load_model_path or self.existing_model_path
            
            # Create SAC agent
            self.agent = SACAgentWrapper(
                env=self.env,
                device=DEVICE,
                use_amp=USE_AMP,
                verbose=1
            )
            
            # Load existing model if available
            if model_path_to_load and Path(model_path_to_load).exists():
                logger.info(f"Loading existing model: {model_path_to_load}")
                try:
                    self.agent.load(model_path_to_load)
                    logger.info("Model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load model, will create new one: {e}")
            else:
                logger.info("Creating new model")
            
            logger.info("SAC agent setup successful")
            return True
            
        except Exception as e:
            logger.error(f"Agent setup failed: {e}", exc_info=True)
            return False
    
    def setup_callbacks(self) -> bool:
        """
        Setup training callbacks
        
        Returns:
            Whether callback setup was successful
        """
        try:
            if self.agent is None:
                logger.error("Agent not setup, please call setup_agent() first")
                return False
            
            logger.info("Setting up training callbacks...")
            
            # Use model identifier as save path to ensure same parameters use same file
            # Modified: Store models in /weights folder instead of /logs
            save_dir = Path("weights")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Create checkpoint callback
            self.callback = UniversalCheckpointCallback(
                save_freq=self.save_freq,
                save_path=save_dir,
                name_prefix=self.model_identifier,  # Use model identifier instead of generic prefix
                eval_freq=self.eval_freq,
                n_eval_episodes=5,
                deterministic_eval=True,
                verbose=1,
                streamlit_session_state=self.streamlit_session_state
            )
            
            logger.info("Training callback setup successful")
            return True
            
        except Exception as e:
            logger.error(f"Callback setup failed: {e}", exc_info=True)
            return False
    
    def train(self) -> bool:
        """
        Execute training process with enhanced monitoring
        
        Returns:
            Whether training was successful
        """
        try:
            if self.agent is None or self.callback is None:
                logger.error("Agent or callback not setup")
                return False
            
            logger.info("=" * 60)
            logger.info("Starting SAC model training")
            logger.info("=" * 60)
            
            self.training_start_time = datetime.now(timezone.utc)
            
            # Update shared data manager status
            self.shared_data_manager.update_training_status('running', 0)
            
            # Start training with enhanced monitoring
            try:
                self.agent.learn(
                    total_timesteps=self.total_timesteps,
                    callback=self.callback,
                    log_interval=100,
                    eval_env=None,
                    eval_freq=self.eval_freq,
                    n_eval_episodes=5,
                    tb_log_name="sac_training",
                    eval_log_path=None,
                    reset_num_timesteps=False
                )
                
                # Training completed successfully
                self.shared_data_manager.update_training_status('completed', 100)
                
                # Save final model
                final_model_path = self.get_model_save_path("final")
                self.agent.save(str(final_model_path))
                logger.info(f"Final model saved: {final_model_path}")
                
                training_duration = datetime.now(timezone.utc) - self.training_start_time
                logger.info("=" * 60)
                logger.info("Training completed successfully!")
                logger.info(f"Training duration: {training_duration}")
                logger.info(f"Total timesteps: {self.total_timesteps}")
                logger.info(f"Final model saved: {final_model_path}")
                logger.info("=" * 60)
                
                return True
                
            except KeyboardInterrupt:
                logger.info("Training interrupted by user")
                self.shared_data_manager.update_training_status('idle')
                return False
                
            except Exception as e:
                logger.error(f"Training process error: {e}", exc_info=True)
                self.shared_data_manager.update_training_status('error', error=str(e))
                return False
            
        except Exception as e:
            logger.error(f"Training setup error: {e}", exc_info=True)
            self.shared_data_manager.update_training_status('error', error=str(e))
            return False
    
    def stop(self):
        """Stop training process"""
        self._stop_training = True
        self.shared_data_manager.request_stop()
        logger.info("Training stop requested")
    
    def save_current_model(self):
        """Save current training progress"""
        if self.agent:
            try:
                checkpoint_path = self.get_model_save_path("checkpoint")
                self.agent.save(str(checkpoint_path))
                logger.info(f"Current model saved: {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to save current model: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            logger.info("Cleaning up training resources...")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("GPU memory cleared")
            
            # Close environment
            if self.env:
                try:
                    self.env.close()
                    logger.info("Environment closed")
                except:
                    pass
            
            # Clear references
            self.agent = None
            self.env = None
            self.dataset = None
            self.callback = None
            
            logger.info("Resource cleanup complete")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def run_full_training_pipeline(self, load_model_path: Optional[str] = None) -> bool:
        """
        Execute complete training pipeline
        
        Args:
            load_model_path: Optional model loading path (for checkpoint resumption)
            
        Returns:
            Whether the entire pipeline completed successfully
        """
        logger.info("=" * 60)
        logger.info("Starting complete training pipeline")
        logger.info("=" * 60)
        
        try:
            # Initialize shared data manager status
            self.shared_data_manager.update_training_status('running', 0)
            
            # 1. Prepare data
            if not self.prepare_data():
                logger.error("Data preparation failed, terminating training")
                self.shared_data_manager.update_training_status('error', error="Data preparation failed")
                return False
            
            # 2. Setup environment
            if not self.setup_environment():
                logger.error("Environment setup failed, terminating training")
                self.shared_data_manager.update_training_status('error', error="Environment setup failed")
                return False
            
            # 3. Setup agent
            if not self.setup_agent(load_model_path):
                logger.error("Agent setup failed, terminating training")
                self.shared_data_manager.update_training_status('error', error="Agent setup failed")
                return False
            
            # 4. Setup callbacks
            if not self.setup_callbacks():
                logger.error("Callback setup failed, terminating training")
                self.shared_data_manager.update_training_status('error', error="Callback setup failed")
                return False
            
            # 5. Execute training
            success = self.train()
            
        except Exception as e:
            logger.error(f"Complete training pipeline error: {e}", exc_info=True)
            self.shared_data_manager.update_training_status('error', error=str(e))
            success = False
        finally:
            # 6. Clean up resources
            self.cleanup()
            
            if success:
                logger.info("=" * 60)
                logger.info("Complete training pipeline completed successfully!")
                logger.info("=" * 60)
            else:
                logger.warning("=" * 60)
                logger.warning("Training pipeline did not complete successfully")
                logger.warning("=" * 60)
            
            return success


def create_training_time_range(days_back: int = 30) -> tuple[datetime, datetime]:
    """
    Create training time range
    
    Args:
        days_back: Number of days back from current time
        
    Returns:
        Tuple of (start_time, end_time)
    """
    end_time = datetime.now(timezone.utc) - timedelta(days=1)  # Yesterday
    start_time = end_time - timedelta(days=days_back)
    
    return start_time, end_time


# Example usage
if __name__ == "__main__":
    # Example configuration
    symbols = ["EUR_USD", "USD_JPY", "GBP_USD"]
    start_time, end_time = create_training_time_range(30)
    
    trainer = EnhancedUniversalTrainer(
        trading_symbols=symbols,
        start_time=start_time,
        end_time=end_time,
        total_timesteps=50000,
        save_freq=2000,
        eval_freq=5000
    )
    
    # Execute complete training pipeline
    success = trainer.run_full_training_pipeline()
    
    if success:
        print("Training completed successfully!")
    else:
        print("Training failed!")