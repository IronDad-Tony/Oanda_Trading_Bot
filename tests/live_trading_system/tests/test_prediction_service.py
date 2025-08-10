
import unittest
import os
import json
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Add project root to path to allow direct imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_preprocessor import LivePreprocessor
from model.prediction_service import PredictionService

class TestPredictionService(unittest.TestCase):
    """
    Tests the integration between LivePreprocessor and PredictionService.
    """

    def setUp(self):
        """Set up the test environment."""
        self.test_dir = os.path.dirname(__file__)
        self.project_root = os.path.abspath(os.path.join(self.test_dir, '..'))

        # --- Create a dummy model and scaler info ---
        self.model_path = os.path.join(self.test_dir, 'dummy_model.pth')
        self.scaler_path = os.path.join(self.test_dir, 'dummy_scaler.json')

        # Create a simple Linear model for testing
        self.model = torch.nn.Linear(10, 1) # Assumes 10 input features
        torch.save(self.model.state_dict(), self.model_path)

        # Create a dummy scaler file
        scaler_info = {
            'mean_': list(np.random.rand(10)),
            'scale_': list(np.random.rand(10) + 0.5)
        }
        with open(self.scaler_path, 'w') as f:
            json.dump(scaler_info, f)

        # --- Create dummy config and services ---
        self.config = {
            "model_path": self.model_path,
            "scaler_path": self.scaler_path,
            "feature_columns": [f"feature_{i}" for i in range(10)]
        }

        self.preprocessor = LivePreprocessor(self.config)
        self.prediction_service = PredictionService(self.config)

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.scaler_path):
            os.remove(self.scaler_path)

    def _create_dummy_candles(self, num_candles=50):
        """Generates a list of dummy candle data."""
        candles = []
        for i in range(num_candles):
            candle = {
                'time': f'2025-01-01T00:{i:02d}:00.000000000Z',
                'volume': 100 + i,
                'mid': {
                    'o': str(1.1 + i * 0.001),
                    'h': str(1.1 + i * 0.001 + 0.0005),
                    'l': str(1.1 + i * 0.001 - 0.0005),
                    'c': str(1.1 + i * 0.001)
                }
            }
            candles.append(candle)
        return candles

    def test_preprocessing_and_prediction_flow(self):
        """Test the complete flow from raw candles to a model prediction."""
        # 1. Generate dummy data
        dummy_candles = self._create_dummy_candles()

        # 2. Preprocess the data
        features_tensor = self.preprocessor.preprocess_live_data(dummy_candles)

        # Assert that preprocessing was successful
        self.assertIsNotNone(features_tensor)
        self.assertIsInstance(features_tensor, torch.Tensor)
        # Shape should be [1, num_features]
        self.assertEqual(features_tensor.shape, (1, len(self.config['feature_columns'])))

        # 3. Get a prediction
        prediction = self.prediction_service.predict(features_tensor)

        # Assert that prediction was successful
        self.assertIsNotNone(prediction)
        self.assertIsInstance(prediction, torch.Tensor)
        self.assertEqual(prediction.shape, (1, 1))

if __name__ == '__main__':
    unittest.main()
