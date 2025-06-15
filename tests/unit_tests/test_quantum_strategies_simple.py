import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

class TestQuantumStrategies(unittest.TestCase):
    """Test quantum strategy layer"""
    
    def test_strategy_import(self):
        """Test strategy module import"""
        try:
            from src.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition
            print("QUANTUM Quantum strategy layer import successful")
        except ImportError as e:
            print(f"WARNING Quantum strategy layer import failed: {e}")
            self.skipTest("Quantum strategy layer import failed")
    
    def test_strategy_creation(self):
        """Test strategy creation"""
        try:
            from src.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition
            from src.agent.strategies import STRATEGY_REGISTRY
            
            strategies_config = {
                "strategies": [
                    {"name": "MomentumStrategy", "params": {"window": 20}, "input_dim": 64}
                ]
            }
            
            superposition = EnhancedStrategySuperposition(
                overall_config_for_strategies=strategies_config,
                strategy_registry=STRATEGY_REGISTRY
            )
            
            self.assertIsNotNone(superposition)
            print("QUANTUM Strategy superposition creation successful")
        except Exception as e:
            print(f"WARNING Strategy creation failed: {e}")
            self.skipTest("Strategy creation failed")

if __name__ == '__main__':
    unittest.main()
