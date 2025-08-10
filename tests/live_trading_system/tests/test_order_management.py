
import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add project root to path to allow direct imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading.order_manager import OrderManager
from trading.risk_manager import RiskManager
from trading.position_manager import PositionManager
from core.oanda_client import OandaClient
from core.system_state import SystemState

class TestOrderManager(unittest.TestCase):

    def setUp(self):
        """Set up the test environment for OrderManager."""
        self.mock_client = MagicMock(spec=OandaClient)
        self.mock_system_state = MagicMock(spec=SystemState)
        
        # Real instances of PositionManager and RiskManager for integration testing
        self.position_manager = PositionManager()
        
        # Mock config for RiskManager
        risk_config = {
            "risk_management": {
                "default_trade_size_units": 100,
                "stop_loss_pips": 20,
                "take_profit_pips": 40
            }
        }
        self.risk_manager = RiskManager(risk_config, self.mock_system_state, self.position_manager)

        self.order_manager = OrderManager(
            client=self.mock_client,
            system_state=self.mock_system_state,
            position_manager=self.position_manager,
            risk_manager=self.risk_manager
        )

    def test_process_signal_approved_trade_and_filled_order(self):
        """Test the full flow: signal -> risk approval -> order -> position update."""
        # 1. Setup
        instrument = "EUR_USD"
        signal_info = {"instrument": instrument, "signal": 1, "price": 1.20000}
        
        # Mock OandaClient response for a successful order fill
        mock_order_fill_response = {
            "orderFillTransaction": {
                "instrument": instrument,
                "units": "100",
                "price": "1.20010",
                "reason": "MARKET_ORDER"
            }
        }
        self.mock_client.create_order.return_value = mock_order_fill_response

        # 2. Action
        self.order_manager.process_signal(signal_info)

        # 3. Assertions
        # a. RiskManager was called (implicitly tested by create_order call)
        # b. OandaClient.create_order was called with correct parameters
        self.mock_client.create_order.assert_called_once()
        call_args = self.mock_client.create_order.call_args[1]
        self.assertEqual(call_args['instrument'], instrument)
        self.assertEqual(call_args['units'], 100)
        self.assertAlmostEqual(call_args['stop_loss'], 1.19800) # 1.20000 - 20 pips
        self.assertAlmostEqual(call_args['take_profit'], 1.20400) # 1.20000 + 40 pips

        # c. PositionManager was updated
        position = self.position_manager.get_position(instrument)
        self.assertIsNotNone(position)
        self.assertEqual(position.instrument, instrument)
        self.assertEqual(position.units, 100)
        self.assertEqual(position.entry_price, 1.20010)
        self.assertEqual(position.position_type, 'long')

    def test_process_signal_trade_not_approved(self):
        """Test that no order is placed if RiskManager rejects the trade."""
        # 1. Setup
        instrument = "GBP_USD"
        signal_info = {"instrument": instrument, "signal": 1, "price": 1.40000}
        
        # Make RiskManager reject the trade (e.g., by creating an existing position)
        self.position_manager.update_position(instrument, 100, 1.39000)

        # 2. Action
        self.order_manager.process_signal(signal_info)

        # 3. Assertions
        # OandaClient.create_order should NOT have been called
        self.mock_client.create_order.assert_not_called()

    def test_process_signal_order_creation_fails(self):
        """Test that PositionManager is not updated if the order fails."""
        # 1. Setup
        instrument = "USD_JPY"
        signal_info = {"instrument": instrument, "signal": -1, "price": 110.00}
        
        # Mock OandaClient to return a failure response
        self.mock_client.create_order.return_value = {"errorMessage": "Market is closed"}

        # 2. Action
        self.order_manager.process_signal(signal_info)

        # 3. Assertions
        # a. OandaClient.create_order was called
        self.mock_client.create_order.assert_called_once()

        # b. PositionManager should NOT have been updated
        position = self.position_manager.get_position(instrument)
        self.assertIsNone(position)

    def test_close_all_positions(self):
        """Test the emergency close_all_positions function."""
        # 1. Setup: Create a few open positions
        self.position_manager.update_position("EUR_USD", 100, 1.2)
        self.position_manager.update_position("USD_JPY", -200, 110.5) # A short position

        # Mock the client response for closing orders
        self.mock_client.create_order.return_value = {"orderFillTransaction": {"reason": "MARKET_ORDER"}}

        # 2. Action
        self.order_manager.close_all_positions()

        # 3. Assertions
        # a. create_order should be called for each position
        self.assertEqual(self.mock_client.create_order.call_count, 2)

        # b. Check the units for the closing orders
        calls = self.mock_client.create_order.call_args_list
        self.assertEqual(calls[0][1]['instrument'], "EUR_USD")
        self.assertEqual(calls[0][1]['units'], -100) # Close long
        self.assertEqual(calls[1][1]['instrument'], "USD_JPY")
        self.assertEqual(calls[1][1]['units'], 200) # Close short

        # c. All positions should be removed from PositionManager
        self.assertEqual(len(self.position_manager.get_all_positions()), 0)

if __name__ == '__main__':
    unittest.main()
