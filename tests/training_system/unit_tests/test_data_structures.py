import unittest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

class TestDataStructures(unittest.TestCase):
    """Test basic data structures"""
    
    def test_numpy_operations(self):
        """Test basic numpy operations"""
        arr = np.array([1, 2, 3, 4, 5])
        self.assertEqual(arr.mean(), 3.0)
        self.assertEqual(arr.std(), np.std([1, 2, 3, 4, 5]))
        print("DATA Numpy operations normal")
    
    def test_dict_operations(self):
        """Test dictionary operations"""
        test_dict = {'a': 1, 'b': 2, 'c': 3}
        self.assertEqual(len(test_dict), 3)
        self.assertEqual(test_dict['a'], 1)
        print("DATA Dictionary operations normal")

if __name__ == '__main__':
    unittest.main()
