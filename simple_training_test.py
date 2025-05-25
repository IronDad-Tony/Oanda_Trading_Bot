#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple training test script to verify the system works
"""

import sys
import os
import locale

# Set UTF-8 encoding for Windows console
if sys.platform.startswith('win'):
    try:
        # Try to set UTF-8 encoding
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        # Fallback for older Python versions
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.append('src')

def test_basic_imports():
    """Test basic imports work"""
    print("Testing basic imports...")
    
    try:
        from common.config import DEFAULT_SYMBOLS, TIMESTEPS, MAX_SYMBOLS_ALLOWED
        print(f"Config loaded: symbols={DEFAULT_SYMBOLS[:3]}..., timesteps={TIMESTEPS}, max_symbols={MAX_SYMBOLS_ALLOWED}")
    except Exception as e:
        print(f"Config import failed: {e}")
        return False
    
    try:
        from environment.trading_env import UniversalTradingEnvV4
        print("Trading environment imported successfully")
    except Exception as e:
        print(f"Trading environment import failed: {e}")
        return False
    
    try:
        from agent.sac_agent_wrapper import SACAgentWrapper
        print("SAC agent wrapper imported successfully")
    except Exception as e:
        print(f"SAC agent wrapper import failed: {e}")
        return False
    
    return True

def test_environment_creation():
    """Test environment creation"""
    print("\nTesting environment creation...")
    
    try:
        from environment.trading_env import UniversalTradingEnvV4
        from data_manager.mmap_dataset import UniversalMemoryMappedDataset
        from data_manager.instrument_info_manager import InstrumentInfoManager
        from common.config import DEFAULT_SYMBOLS, TIMESTEPS, MAX_SYMBOLS_ALLOWED
        
        # Create a small dataset for testing
        test_symbols = DEFAULT_SYMBOLS[:2]  # Just use first 2 symbols
        print(f"Creating test dataset with symbols: {test_symbols}")
        
        # Create instrument info manager
        instrument_manager = InstrumentInfoManager()
        print("Instrument info manager created")
        
        # Create dataset (this might take a moment)
        dataset = UniversalMemoryMappedDataset(
            symbols=test_symbols,
            start_time_iso="2024-05-20T10:00:00Z",
            end_time_iso="2024-05-20T10:30:00Z",  # Just 30 minutes for testing
            timesteps_history=TIMESTEPS,
            force_reload=False
        )
        print(f"Dataset created with {len(dataset)} samples")
        
        if len(dataset) == 0:
            print("Warning: Dataset is empty, this might indicate missing data")
            return False
        
        # Create environment
        env = UniversalTradingEnvV4(
            dataset=dataset,
            instrument_info_manager=instrument_manager,
            active_symbols_for_episode=test_symbols
        )
        print("Environment created successfully")
        
        # Test reset
        obs, info = env.reset()
        print(f"Environment reset successful, observation keys: {list(obs.keys())}")
        
        return True
        
    except Exception as e:
        print(f"Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_creation():
    """Test agent creation"""
    print("\nTesting agent creation...")
    
    try:
        from agent.sac_agent_wrapper import SACAgentWrapper
        from environment.trading_env import UniversalTradingEnvV4
        from data_manager.mmap_dataset import UniversalMemoryMappedDataset
        from data_manager.instrument_info_manager import InstrumentInfoManager
        from common.config import DEFAULT_SYMBOLS, TIMESTEPS, MAX_SYMBOLS_ALLOWED
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        # Create minimal environment for agent
        test_symbols = DEFAULT_SYMBOLS[:1]  # Just 1 symbol for speed
        instrument_manager = InstrumentInfoManager()
        
        dataset = UniversalMemoryMappedDataset(
            symbols=test_symbols,
            start_time_iso="2024-05-20T10:00:00Z",
            end_time_iso="2024-05-20T10:15:00Z",  # Just 15 minutes for speed
            timesteps_history=TIMESTEPS,
            force_reload=False
        )
        
        if len(dataset) == 0:
            print("Skipping agent test due to empty dataset")
            return True
        
        env = UniversalTradingEnvV4(
            dataset=dataset,
            instrument_info_manager=instrument_manager,
            active_symbols_for_episode=test_symbols
        )
        
        # Wrap environment in DummyVecEnv for SB3 compatibility
        vec_env = DummyVecEnv([lambda: env])
        
        # Create agent
        agent = SACAgentWrapper(vec_env)
        print("Agent created successfully")
        
        # Test a few steps
        obs = vec_env.reset()
        for i in range(3):
            action, _ = agent.predict(obs, deterministic=False)
            step_result = vec_env.step(action)
            
            # Handle different return formats from vec_env.step()
            if len(step_result) == 4:
                obs, reward, done, info = step_result
                terminated = done
                truncated = [False] * len(done)
            else:
                obs, reward, terminated, truncated, info = step_result
            
            print(f"Step {i+1}: reward={reward[0]:.6f}, terminated={terminated[0]}, truncated={truncated[0] if hasattr(truncated, '__getitem__') else truncated}")
            
            if terminated[0] or (hasattr(truncated, '__getitem__') and truncated[0]):
                obs = vec_env.reset()
        
        print("Agent test completed successfully")
        return True
        
    except Exception as e:
        print(f"Agent creation/testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=== Simple Training System Test ===\n")
    
    # Test 1: Basic imports
    if not test_basic_imports():
        print("\nBasic imports failed. Cannot proceed.")
        return False
    
    # Test 2: Environment creation
    if not test_environment_creation():
        print("\nEnvironment creation failed. Cannot proceed.")
        return False
    
    # Test 3: Agent creation
    if not test_agent_creation():
        print("\nAgent creation failed.")
        return False
    
    print("\n=== All Tests Passed! ===")
    print("The training system appears to be working correctly.")
    print("You can now proceed with full training using the Streamlit GUI.")
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\nTest completed successfully!")
        else:
            print("\nTest failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)