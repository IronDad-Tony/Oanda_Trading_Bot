class MarketStateAwareness:
    def monitor_risk_indicators(self, market_data):
        """
        監控波動率、流動性、相關性等風險指標
        返回風險等級 (0-1, 1表示最高風險)
        """
        # 計算波動率 (使用標準差)
        volatility = market_data['close'].std()
        
        # 計算流動性 (交易量變化率)
        liquidity = market_data['volume'].pct_change().abs().mean()
        
        # 計算相關性 (多資產價格變動相關性)
        correlation = market_data[['asset1', 'asset2']].corr().iloc[0,1]
        
        # 綜合風險評分 (範例算法)
        risk_score = (volatility * 0.4) + (liquidity * 0.3) + ((1 - abs(correlation)) * 0.3)
        return min(max(risk_score, 0), 1)  # 確保在0-1範圍內

    def dynamic_stoploss(self, position, risk_level):
        """
        基於風險等級的動態停損計算
        position: 當前持倉信息 (dict)
        risk_level: 風險等級 (0-1)
        返回調整後的停損價
        """
        base_stop = position['entry_price'] * (1 - position['base_stop_loss'])
        
        # 風險等級越高，停損越緊急
        risk_adjustment = 1 - (risk_level * 0.5)  # 風險最高時停損距離縮小50%
        new_stop = position['entry_price'] - (
            (position['entry_price'] - base_stop) * risk_adjustment
        )
        return new_stop

class StressTester:
    def __init__(self):
        """
        初始化壓力測試器，加載自定義場景
        """
        self.custom_scenarios = {}
        # 嘗試加載預設的自定義場景
        try:
            self.add_custom_scenario("live_trading/crisis_scenarios.json")
        except FileNotFoundError:
            print("警告：預設場景配置文件未找到，將使用空場景")

    def simulate_black_swan(self, historical_event):
        """
        模擬歷史黑天鵝事件場景
        historical_event: 歷史事件名稱 (str)
        返回壓力測試結果 (dict)
        
        新增場景：
        - 流動性突然枯竭
        - 跨市場傳染效應
        - 極端波動率跳躍
        """
        # 檢查是否為預定義場景
        if historical_event in self.custom_scenarios:
            return self.custom_scenarios[historical_event]
            
        # 內置事件數據
        event_data = {
            '2020_COVID': {'max_drawdown': -0.34, 'recovery_days': 45},
            '2008_Lehman': {'max_drawdown': -0.52, 'recovery_days': 120},
            # 新增場景
            '流動性突然枯竭': self.liquidity_crisis_test({'severity': 0.9}),
            '跨市場傳染效應': {
                'asset_correlation': 0.95,
                'contagion_effect': 0.85,
                'max_drawdown': -0.45
            },
            '極端波動率跳躍': {
                'volatility_spike': 0.35,
                'impact_duration': 7,
                'max_drawdown': -0.38
            }
        }
        return event_data.get(historical_event, {})

    def liquidity_crisis_test(self, scenario_params):
        """
        流動性危機壓力測試
        scenario_params: 場景參數 (dict)
        返回流動性指標變化 (dict)
        """
        severity = scenario_params.get('severity', 0.5)
        # 模擬流動性枯竭程度
        liquidity_shock = {
            'bid_ask_spread': 0.001 * (1 + severity * 10),  # 點差擴大
            'order_execution_time': 5 * (1 + severity * 3),  # 執行延遲
            'fill_rate': max(0.7 - severity * 0.5, 0.1),  # 成交率下降
            'severity_level': severity
        }
        return liquidity_shock
        
    def add_custom_scenario(self, scenario_config):
        """
        新增方法：從JSON加載自定義場景
        scenario_config: JSON配置檔案路徑
        """
        import json
        try:
            with open(scenario_config, 'r') as f:
                scenarios = json.load(f)
                
            # 將場景轉換為統一的格式
            for event in scenarios.get('black_swan_events', []):
                self.custom_scenarios[event['name']] = {
                    'type': 'black_swan',
                    'description': event['description'],
                    'date_range': event['date_range']
                }
                
            for crisis in scenarios.get('liquidity_crises', []):
                self.custom_scenarios[crisis['name']] = {
                    'type': 'liquidity_crisis',
                    'description': crisis['description'],
                    'severity': crisis['severity'],
                    'impact': self.liquidity_crisis_test({'severity': crisis['severity']})
                }
                
            print(f"成功加載 {len(self.custom_scenarios)} 個自定義場景")
            return True
        except Exception as e:
            print(f"加載自定義場景失敗: {e}")
            return False