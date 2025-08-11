
import unittest
from unittest.mock import MagicMock
import os
import sys

# Add project root to path to allow direct imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading.risk_manager import RiskManager
from trading.position_manager import PositionManager, Position
from core.system_state import SystemState

class TestRiskManager(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.config = {
            "risk_management": {
                "max_total_exposure_usd": 10000,
                "max_risk_per_trade_percent": 1.0,
                "stop_loss_pips": 50,
                "take_profit_pips": 100,
                "default_trade_size_units": 1000
            }
        }
        self.mock_system_state = MagicMock(spec=SystemState)
        self.mock_position_manager = MagicMock(spec=PositionManager)
        
        self.risk_manager = RiskManager(
            config=self.config,
            system_state=self.mock_system_state,
            position_manager=self.mock_position_manager
        )

    def test_assess_trade_no_existing_position_buy_signal(self):
        """Test assessing a buy trade with no existing position."""
        self.mock_position_manager.get_position.return_value = None
        
        instrument = "EUR_USD"
        signal = 1  # Buy
        price = 1.12000

        trade_details = self.risk_manager.assess_trade(instrument, signal, price)

        self.assertIsNotNone(trade_details)
        self.assertEqual(trade_details['units'], 1000)
        self.assertAlmostEqual(trade_details['stop_loss'], 1.11500) # 1.12000 - 50 * 0.0001
        self.assertAlmostEqual(trade_details['take_profit'], 1.13000) # 1.12000 + 100 * 0.0001
        self.mock_position_manager.get_position.assert_called_once_with(instrument)

    def test_assess_trade_no_existing_position_sell_signal(self):
        """Test assessing a sell trade with no existing position."""
        self.mock_position_manager.get_position.return_value = None

        instrument = "EUR_USD"
        signal = -1  # Sell
        price = 1.12000

        trade_details = self.risk_manager.assess_trade(instrument, signal, price)

        self.assertIsNotNone(trade_details)
        self.assertEqual(trade_details['units'], -1000)
        self.assertAlmostEqual(trade_details['stop_loss'], 1.12500) # 1.12000 + 50 * 0.0001
        self.assertAlmostEqual(trade_details['take_profit'], 1.11000) # 1.12000 - 100 * 0.0001
        self.mock_position_manager.get_position.assert_called_once_with(instrument)

    def test_assess_trade_with_existing_long_position_same_signal(self):
        """Test assessing a buy trade when a long position already exists."""
        instrument = "EUR_USD"
        existing_position = Position(instrument, 'long', 1000, 1.11000)
        self.mock_position_manager.get_position.return_value = existing_position

        signal = 1  # Buy
        price = 1.12000

        trade_details = self.risk_manager.assess_trade(instrument, signal, price)

        self.assertIsNone(trade_details)
        self.mock_position_manager.get_position.assert_called_once_with(instrument)

    def test_assess_trade_with_existing_short_position_same_signal(self):
        """Test assessing a sell trade when a short position already exists."""
        instrument = "EUR_USD"
        existing_position = Position(instrument, 'short', 1000, 1.13000)
        self.mock_position_manager.get_position.return_value = existing_position

        signal = -1  # Sell
        price = 1.12000

        trade_details = self.risk_manager.assess_trade(instrument, signal, price)

        self.assertIsNone(trade_details)
        self.mock_position_manager.get_position.assert_called_once_with(instrument)
        
    def test_assess_trade_hold_signal(self):
        """Test assessing a hold trade signal."""
        self.mock_position_manager.get_position.return_value = None
        
        instrument = "EUR_USD"
        signal = 0 # Hold
        price = 1.12000
        
        trade_details = self.risk_manager.assess_trade(instrument, signal, price)
        
        self.assertIsNone(trade_details)

if __name__ == '__main__':
    unittest.main()
