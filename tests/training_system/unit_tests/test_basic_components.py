import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

class TestBasicComponents(unittest.TestCase):
    """Test basic components"""
    
    def test_import_basic_modules(self):
        """Test basic module imports"""
        try:
            import torch
            import numpy as np
            import pandas as pd
            print("BASIC Module imports successful")
        except ImportError as e:
            self.fail(f"Basic module import failed: {e}")

if __name__ == '__main__':
    unittest.main()
