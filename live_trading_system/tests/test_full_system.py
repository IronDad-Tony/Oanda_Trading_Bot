import sys
import os
import unittest
from unittest.mock import patch, MagicMock, call

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from live_trading_system.main import initialize_system, trading_loop
from live_trading_system.core.system_state import SystemState

class TestFullSystemFlow(unittest.TestCase):
    """
    End-to-end integration test for the full trading system flow.
    """

    @patch('live_trading_system.main.load_config')
    @patch('live_trading_system.database.database_manager.DatabaseManager')
    @patch('live_trading_system.model.prediction_service.PredictionService')
    @patch('live_trading_system.core.oanda_client.OandaClient')
    def setUp(self, MockOandaClient, MockPredictionService, MockDatabaseManager, mock_load_config):
        """
        Set up the mock environment before each test.
        """
        print("Setting up test environment...")

        # --- Mock Config ---
        dummy_scaler_path = os.path.join(project_root, 'live_trading_system', 'tests', 'dummy_scalers.json')
        self.config = {
            "oanda": {
                "api_key": "test_key",
                "account_id": "test_account",
                "environment": "practice"
            },
            "preprocessor": {
                "scalers_path": dummy_scaler_path
            },
            "trading_instruments": ["EUR_USD"],
            "risk_management": {
                "max_total_exposure_usd": 10000,
                "max_risk_per_trade_percent": 1.0,
                "stop_loss_pips": 15,
                "take_profit_pips": 30,
                "default_trade_size_units": 1000
            },
            "database_path": ":memory:",
            "trading_loop_interval_seconds": 1
        }
        mock_load_config.return_value = self.config

        # --- Mock Core Components ---
        self.mock_oanda_client = MockOandaClient()
        self.mock_prediction_service = MockPredictionService()
        self.mock_db_manager = MockDatabaseManager()

        # Mock OandaClient behavior
        self.mock_oanda_client.get_account_summary.return_value = {'balance': 100000, 'pl': 0, 'marginUsed': 0}
        self.mock_oanda_client.get_candles.return_value = [{'time': '2025-06-24T10:00:00Z', 'mid': {'c': '1.1000'}}] * 100
        self.mock_oanda_client.create_order.return_value = {'orderFillTransaction': {'id': '12345', 'price': '1.1005', 'units': '1000', 'instrument': 'EUR_USD', 'reason': 'CLIENT_ORDER'}}
        self.mock_oanda_client.get_open_positions.return_value = []

        # Mock PredictionService behavior
        self.mock_prediction_service.predict.side_effect = [1, 0, 0, 0, 0, 0]

        # --- Initialize System with Mocks ---
        self.system_components = initialize_system(
            mock_client=self.mock_oanda_client,
            mock_prediction_service=self.mock_prediction_service,
            mock_db_manager=self.mock_db_manager
        )

    def test_end_to_end_trading_cycle(self):
        """
        Test a single, complete trading cycle.
        """
        print("Starting end-to-end test...")

        self.assertIsNotNone(self.system_components, "System components should be initialized")

        # Get components from the initialized system
        system_state = self.system_components['system_state']
        position_manager = self.system_components['position_manager']
        trading_logic = self.system_components['trading_logic']

        # --- Run one cycle of the trading loop ---
        system_state.start() # Corrected method call
        
        # We call the trading_logic's method directly instead of the global trading_loop
        # to have more control in the test.
        trading_logic.execute_trade_cycle()

        # --- Assertions ---
        # 1. Verify prediction service was called
        self.mock_prediction_service.predict.assert_called()

        # 2. Verify an order was created
        self.mock_oanda_client.create_order.assert_called_once()
        args, kwargs = self.mock_oanda_client.create_order.call_args
        self.assertEqual(kwargs['instrument'], 'EUR_USD')
        self.assertEqual(kwargs['units'], 1000)

        # 3. Verify a position is now open
        self.assertIsNotNone(position_manager.get_position('EUR_USD'))
        self.assertEqual(position_manager.get_position('EUR_USD').units, 1000)

        # 4. Verify the trade was saved to the database
        self.mock_db_manager.save_trade.assert_called_once()

        # --- Test graceful shutdown ---
        system_state.stop() # Corrected method call
        self.assertFalse(system_state.is_running())

        print("End-to-end test finished successfully.")

if __name__ == '__main__':
    # This allows running the test script directly
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
