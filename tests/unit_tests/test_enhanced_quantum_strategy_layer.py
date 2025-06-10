import unittest
import sys
import os

# Add src directory to Python path
# This is a common way to ensure that modules in src can be imported in tests
# Adjust the number of '..' based on the test file's location relative to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

class TestEnhancedQuantumStrategyLayerImport(unittest.TestCase):

    def test_import_dynamic_strategy_generator(self):
        """
        Tests if DynamicStrategyGenerator can be imported from the
        enhanced_quantum_strategy_layer module.
        """
        try:
            from src.agent.enhanced_quantum_strategy_layer import DynamicStrategyGenerator
            self.assertTrue(True, "Successfully imported DynamicStrategyGenerator.")
            # Optionally, try to instantiate it
            dsg = DynamicStrategyGenerator()
            self.assertIsNotNone(dsg, "DynamicStrategyGenerator instance should not be None.")
        except ImportError as e:
            self.fail(f"Failed to import DynamicStrategyGenerator: {e}")
        except Exception as e:
            self.fail(f"An error occurred during import or instantiation: {e}")

if __name__ == '__main__':
    unittest.main()
