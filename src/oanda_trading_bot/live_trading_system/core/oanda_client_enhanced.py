# Enhanced OANDA Client - Complete Implementation
# Adds all missing critical features based on v20 API docs

"""
OANDA v20 API完整客戶端
包含所有交易、風險管理和數據獲取所需的功能
"""

import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.trades as trades

# Add to existing OandaClient class:

class OandaClientEnhanced(OandaClient):
    """Enhanced client with all critical features"""
    
    # ============================================================================
    # CRITICAL: Instrument Details
    # ============================================================================
    
    def get_instrument_details(self, instruments: List[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get detailed specifications for instruments
        
        Returns margin rate, minimum trade size, financing rates, etc.
        Critical for training environment to replicate OANDA exactly.
        
        Response includes:
        - marginRate: e.g. "0.0333" for 30:1 leverage
        - minimumTradeSize: e.g. "1" 
        - maximumOrderUnits: e.g. "100000000"
        - financing.longRate / shortRate: Swap rates
        - pipLocation, displayPrecision, etc.
        """
        params = {}
        if instruments:
            params["instruments"] = ",".join(instruments)
        
        endpoint = accounts.AccountInstruments(self.account_id, params=params)
        try:
            response = self._request(endpoint)
            logger.info(f"Retrieved details for {len(response.get('instruments', []))} instruments")
            return response
        except V20Error as e:
            logger.error(f"Failed to get instrument details: {e}")
            return None
    
    # ============================================================================
    # CRITICAL: Full Account Details (includes conversion factors)
    # ============================================================================
    
    def get_account_details(self) -> Optional[Dict[str, Any]]:
        """
        Get complete account details including:
        - Home currency
        - Margin rates
        - Margin closeout levels
        - Conversion factors
        - All positions and trades
        
        More complete than get_account_summary()
        """
        endpoint = accounts.AccountDetails(self.account_id)
        try:
            response = self._request(endpoint)
            if response and 'account' in response:
                # Update last transaction ID
                if 'lastTransactionID' in response['account']:
                    self.last_transaction_id = response['account']['lastTransactionID']
                    logger.debug(f"last_transaction_id updated: {self.last_transaction_id}")
            return response
        except V20Error as e:
            logger.error(f"Failed to get account details: {e}")
            return None
    
    # ============================================================================
    # IMPORTANT: Current Pricing (better than candles for live prices)
    # ============================================================================
    
    def get_pricing(self, instruments: List[str]) -> Optional[Dict[str, Any]]:
        """
        Get current tradeable prices for instruments
        
        Returns:
        - Current bid/ask spreads
        - Tradeable status
        - Quote home conversion factors
        - Units available for trading
        
        More accurate than candles for current prices.
        Lower latency.
        """
        params = {"instruments": ",".join(instruments)}
        endpoint = pricing.PricingInfo(self.account_id, params=params)
        try:
            response = self._request(endpoint)
            logger.debug(f"Retrieved pricing for {len(response.get('prices', []))} instruments")
            return response
        except V20Error as e:
            logger.error(f"Failed to get pricing: {e}")
            return None
    
    def get_home_conversion_factors(self, instrument: str) -> Optional[Dict[str, Any]]:
        """
        Get conversion factors to convert P&L to home currency
        
        Returns quote home conversion factors from pricing data.
        Critical for accurate P&L calculation in training.
        """
        pricing_data = self.get_pricing([instrument])
        if not pricing_data or 'prices' not in pricing_data:
            return None
        
        for price in pricing_data['prices']:
            if price['instrument'] == instrument:
                return price.get('quoteHomeConversionFactors')
        
        return None
    
    # ============================================================================
    # IMPORTANT: Trade Management
    # ============================================================================
    
    def get_open_trades(self) -> Optional[Dict[str, Any]]:
        """
        List all open trades
        
        Returns details of all currently open trades including:
        - Trade ID
        - Instrument
        - Current units
        - Price
        - Unrealized P&L
        - Margin used
        """
        endpoint = trades.OpenTrades(self.account_id)
        try:
            response = self._request(endpoint)
            logger.debug(f"Retrieved {len(response.get('trades', []))} open trades")
            return response
        except V20Error as e:
            logger.error(f"Failed to get open trades: {e}")
            return None
    
    def get_trade_details(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific trade
        
        Includes complete trade history, financing charges, etc.
        """
        endpoint = trades.TradeDetails(self.account_id, trade_id)
        try:
            return self._request(endpoint)
        except V20Error as e:
            logger.error(f"Failed to get trade {trade_id} details: {e}")
            return None
    
    def close_trade(self, trade_id: str, units: str = "ALL") -> Optional[Dict[str, Any]]:
        """
        Close a specific trade (all or partial)
        
        Args:
            trade_id: The trade ID to close
            units: "ALL" or specific number of units
        """
        data = {"units": units}
        endpoint = trades.TradeClose(self.account_id, trade_id, data=data)
        try:
            response = self._request(endpoint)
            logger.info(f"Closed trade {trade_id}")
            return response
        except V20Error as e:
            logger.error(f"Failed to close trade {trade_id}: {e}")
            return None
    
    # ============================================================================
    # HELPER: Get Current Mid Price
    # ============================================================================
    
    def get_current_mid_price(self, instrument: str) -> Optional[float]:
        """
        Get current mid price for an instrument
        
        Uses pricing API for most accurate current price.
        Falls back to candles if pricing unavailable.
        """
        # Try pricing API first (faster, more accurate)
        pricing_data = self.get_pricing([instrument])
        if pricing_data and 'prices' in pricing_data:
            for price_data in pricing_data['prices']:
                if price_data['instrument'] == instrument and price_data.get('tradeable'):
                    bid = float(price_data['bids'][0]['price'])
                    ask = float(price_data['asks'][0]['price'])
                    return (bid + ask) / 2.0
        
        # Fallback to candles
        candles = self.get_candles(instrument, count=1, price="M")
        if candles and len(candles) > 0:
            last_candle = candles[-1]
            if 'mid' in last_candle:
                return float(last_candle['mid'].get('c', 0))
        
        return None
    
    # ============================================================================
    # HELPER: Validate Instrument Availability
    # ============================================================================
    
    def is_instrument_tradeable(self, instrument: str) -> bool:
        """
        Check if an instrument is currently tradeable
        
        Returns True only if instrument is available and tradeable.
        """
        pricing_data = self.get_pricing([instrument])
        if not pricing_data or 'prices' not in pricing_data:
            return False
        
        for price_data in pricing_data['prices']:
            if price_data['instrument'] == instrument:
                return price_data.get('tradeable', False)
        
        return False
    
    # ============================================================================
    # HELPER: Get Available Units from Pricing
    # ============================================================================
    
    def get_available_units(self, instrument: str) -> Optional[Dict[str, Any]]:
        """
        Get available units that can be traded
        
        Returns units available for:
        - default: Normal trading
        - openOnly: Only opening new positions
        - reduceFirst: Reducing then opening
        - reduceOnly: Only reducing existing
        """
        pricing_data = self.get_pricing([instrument])
        if not pricing_data or 'prices' not in pricing_data:
            return None
        
        for price_data in pricing_data['prices']:
            if price_data['instrument'] == instrument:
                return price_data.get('unitsAvailable')
        
        return None


# Example usage
if __name__ == '__main__':
    #... existing code ...
    
    # Test new features
    client_enhanced = OandaClientEnhanced.from_env()
    
    # Get instrument details
    instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY']
    inst_details = client_enhanced.get_instrument_details(instruments)
    if inst_details:
        for inst in inst_details['instruments']:
            logger.info(f"{inst['name']}: margin={inst['marginRate']}, "
                       f"min_size={inst['minimumTradeSize']}, "
                       f"financing_long={inst['financing']['longRate']}")
    
    # Get account details with conversion factors
    acc_details = client_enhanced.get_account_details()
    if acc_details:
        acc = acc_details['account']
        logger.info(f"Account currency: {acc['currency']}")
        logger.info(f"Margin available: {acc['marginAvailable']}")
        logger.info(f"Margin closeout%%: {acc['marginCloseoutPercent']}")
    
    # Get current pricing
    pricing_info = client_enhanced.get_pricing(['EUR_USD'])
    if pricing_info:
        for price in pricing_info['prices']:
            logger.info(f"{price['instrument']}: "
                       f"bid={price['closeoutBid']}, "
                       f"ask={price['closeoutAsk']}, "
                       f"tradeable={price['tradeable']}")
    
    # Check if tradeable
    is_tradeable = client_enhanced.is_instrument_tradeable('EUR_USD')
    logger.info(f"EUR_USD tradeable: {is_tradeable}")
    
    # Get available units
    units = client_enhanced.get_available_units('EUR_USD')
    if units:
        logger.info(f"EUR_USD units available: "
                   f"long={units['default']['long']}, "
                   f"short={units['default']['short']}")
