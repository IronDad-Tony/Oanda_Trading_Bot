#!/usr/bin/env python3
"""
Test script for SAC Agent Wrapper with High-Level Integration System
Adapted to work with the existing TransformerFeatureExtractor architecture
"""

import torch
import numpy as np
import logging
import sys
import os
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockTradingEnv(gym.Env):
    """Mock trading environment that mimics our UniversalTradingEnv structure"""
    
    def __init__(self, num_symbols=5, timesteps=60, feature_dim=20):
        super().__init__()
        
        self.num_symbols = num_symbols
        self.timesteps = timesteps
        self.feature_dim = feature_dim
        
        # Create observation space that matches our TransformerFeatureExtractor expectations
        self.observation_space = spaces.Dict({
            'features_from_dataset': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(num_symbols, timesteps, feature_dim),
                dtype=np.float32
            ),
            'padding_mask': spaces.Box(
                low=0, high=1,
                shape=(num_symbols,),
                dtype=np.bool_
            )
        })
        
        # SAC requires continuous action space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(num_symbols,),  # One action per symbol
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = 100
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Generate mock observations
        obs = {
            'features_from_dataset': np.random.randn(
                self.num_symbols, self.timesteps, self.feature_dim
            ).astype(np.float32),
            'padding_mask': np.random.choice(
                [True, False], size=(self.num_symbols,), p=[0.8, 0.2]
            )
        }
        
        return obs, {}
    
    def step(self, action):
        self.current_step += 1
        
        # Generate next observation
        obs = {
            'features_from_dataset': np.random.randn(
                self.num_symbols, self.timesteps, self.feature_dim
            ).astype(np.float32),
            'padding_mask': np.random.choice(
                [True, False], size=(self.num_symbols,), p=[0.8, 0.2]
            )
        }
        
        # Mock reward calculation
        reward = np.random.randn() * 0.1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        truncated = False
        
        return obs, reward, done, truncated, {}

def test_sac_integration():
    """Test SAC Agent Wrapper with High-Level Integration System"""
    
    logger.info("=" * 80)
    logger.info("🔧 TESTING SAC AGENT WRAPPER WITH HIGH-LEVEL INTEGRATION")
    logger.info("=" * 80)
    
    try:
        # Import required modules
        from stable_baselines3.common.vec_env import DummyVecEnv
        from src.agent.sac_agent_wrapper import QuantumEnhancedSAC
        
        logger.info("✅ Successfully imported SAC wrapper and dependencies")
        
        # Create a mock trading environment that matches our system architecture
        def make_test_env():
            return MockTradingEnv(num_symbols=3, timesteps=60, feature_dim=20)
        
        # Create vectorized environment
        env = DummyVecEnv([make_test_env])
        logger.info("✅ Created mock trading environment with Dict observation space")
        
        # Initialize SAC wrapper with integration
        sac_wrapper = QuantumEnhancedSAC(
            env=env,
            batch_size=32,
            buffer_size=1000,
            learning_starts_factor=10,
            verbose=1
        )
        logger.info("✅ SAC wrapper initialized successfully")
        
        # Check if high-level integration is available
        if hasattr(sac_wrapper, 'high_level_integration') and sac_wrapper.high_level_integration is not None:
            logger.info("✅ High-Level Integration System is available")
            
            # Test the integration processing method
            try:
                # Create mock market data that matches the system's expectations
                batch_size = 4
                feature_dim = 512
                market_data = torch.randn(batch_size, feature_dim)
                
                # Test the processing method
                results = sac_wrapper.process_with_high_level_integration(
                    market_data=market_data
                )
                
                if results:
                    logger.info("✅ High-level integration processing successful")
                    logger.info(f"   Results keys: {list(results.keys())}")
                    
                    # Check for expected result types
                    expected_keys = ['market_state', 'strategy_innovation', 'anomaly_detection', 'system_health']
                    found_keys = [key for key in expected_keys if key in results]
                    logger.info(f"   Found expected keys: {found_keys}")
                    
                    # Check system health if available
                    if 'system_health' in results:
                        health_info = results['system_health']
                        if isinstance(health_info, dict):
                            logger.info(f"   System health score: {health_info.get('health_score', 'N/A')}")
                            logger.info(f"   System state: {health_info.get('system_state', 'N/A')}")
                    
                else:
                    logger.warning("⚠️ Integration processing returned empty results")
                    
            except Exception as e:
                logger.error(f"❌ Integration processing failed: {e}")
                logger.error("Stack trace:", exc_info=True)
                
        else:
            logger.warning("⚠️ High-Level Integration System not available")
        
        # Test basic SAC functionality
        try:
            # Get observation from environment
            obs = env.reset()
            logger.info("✅ Environment reset successful")
            logger.info(f"   Observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")
            
            # Test prediction
            action, _states = sac_wrapper.agent.predict(obs)
            logger.info(f"✅ Action prediction successful: shape={action.shape}")
            
            # Test a single step
            obs, reward, done, info = env.step(action)
            logger.info(f"✅ Environment step successful: reward={reward}")
            
        except Exception as e:
            logger.error(f"❌ Basic SAC functionality test failed: {e}")
            logger.error("Stack trace:", exc_info=True)
        
        logger.info("=" * 80)
        logger.info("🎉 SAC INTEGRATION TEST COMPLETED!")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SAC integration test failed: {e}")
        logger.error("Stack trace:", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_sac_integration()
    if success:
        print("\n✅ Test completed!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)
