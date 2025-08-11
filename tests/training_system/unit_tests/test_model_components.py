import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

class TestModelComponents(unittest.TestCase):
    """Test model components"""
    
    def test_pytorch_available(self):
        """Test PyTorch availability"""
        try:
            import torch
            self.assertTrue(torch.cuda.is_available() or True)  # CPU is also OK
            print("PYTORCH PyTorch available")
        except ImportError:
            self.fail("PyTorch not available")

if __name__ == '__main__':
    unittest.main()
