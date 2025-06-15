import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

class TestBasicConfig(unittest.TestCase):
    """Test basic configuration functionality"""
    
    def test_config_import(self):
        """Test configuration module can be imported normally"""
        try:
            from src.common.config import DEVICE, DEFAULT_SYMBOLS
            self.assertIsNotNone(DEVICE)
            self.assertIsNotNone(DEFAULT_SYMBOLS)
            print("CONFIG Import successful")
        except ImportError as e:
            self.fail(f"Configuration module import failed: {e}")
    
    def test_basic_constants(self):
        """Test basic constant settings"""
        try:
            from src.common.config import (
                INITIAL_CAPITAL, SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR, 
                DEFAULT_TRAIN_START_ISO, DEFAULT_TRAIN_END_ISO
            )
            self.assertGreater(INITIAL_CAPITAL, 0)
            self.assertGreater(SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR, 0)
            self.assertIsInstance(DEFAULT_TRAIN_START_ISO, str)
            self.assertIsInstance(DEFAULT_TRAIN_END_ISO, str)
            print("CONFIG Basic constants correct")
        except ImportError as e:
            self.fail(f"Basic constants import failed: {e}")

if __name__ == '__main__':
    unittest.main()
