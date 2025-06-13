# src/agent/strategies/__init__.py

# Base Strategy and Config
from .base_strategy import BaseStrategy, StrategyConfig

# Trend Strategies
from .trend_strategies import (
    MomentumStrategy,
    BreakoutStrategy,
    TrendFollowingStrategy,
    ReversalStrategy
)

# Statistical Arbitrage Strategies
from .statistical_arbitrage_strategies import (
    StatisticalArbitrageStrategy,
    MeanReversionStrategy,
    CointegrationStrategy,
    VolatilityBreakoutStrategy,
    PairsTradeStrategy
)

# Machine Learning Strategies
from .ml_strategies import (
    ReinforcementLearningStrategy,
    DeepLearningPredictionStrategy,
    EnsembleLearningStrategy,
    TransferLearningStrategy
)

# Risk Management Strategies
from .risk_management_strategies import (
    DynamicHedgingStrategy,
    RiskParityStrategy,
    VaRControlStrategy,
    MaxDrawdownControlStrategy
)

# Other Strategies
from .other_strategies import (
    OptionFlowStrategy,
    MicrostructureStrategy,
    CarryTradeStrategy,
    MacroEconomicStrategy,
    EventDrivenStrategy,
    SentimentStrategy,
    QuantitativeStrategy,
    MarketMakingStrategy,
    HighFrequencyStrategy,
    AlgorithmicStrategy
)

# Dynamically create STRATEGY_REGISTRY from imported strategy classes
STRATEGY_REGISTRY = {
    # Trend Strategies
    "MomentumStrategy": MomentumStrategy,
    "BreakoutStrategy": BreakoutStrategy,
    "TrendFollowingStrategy": TrendFollowingStrategy,
    "ReversalStrategy": ReversalStrategy,
    # Statistical Arbitrage Strategies
    "StatisticalArbitrageStrategy": StatisticalArbitrageStrategy,
    "MeanReversionStrategy": MeanReversionStrategy,
    "CointegrationStrategy": CointegrationStrategy,
    "VolatilityBreakoutStrategy": VolatilityBreakoutStrategy,
    "PairsTradeStrategy": PairsTradeStrategy,
    # Machine Learning Strategies
    "ReinforcementLearningStrategy": ReinforcementLearningStrategy,
    "DeepLearningPredictionStrategy": DeepLearningPredictionStrategy,
    "EnsembleLearningStrategy": EnsembleLearningStrategy,
    "TransferLearningStrategy": TransferLearningStrategy,
    # Risk Management Strategies
    "DynamicHedgingStrategy": DynamicHedgingStrategy,
    "RiskParityStrategy": RiskParityStrategy,
    "VaRControlStrategy": VaRControlStrategy,
    "MaxDrawdownControlStrategy": MaxDrawdownControlStrategy,
    # Other Strategies
    "OptionFlowStrategy": OptionFlowStrategy,
    "MicrostructureStrategy": MicrostructureStrategy,
    "CarryTradeStrategy": CarryTradeStrategy,
    "MacroEconomicStrategy": MacroEconomicStrategy,
    "EventDrivenStrategy": EventDrivenStrategy,
    "SentimentStrategy": SentimentStrategy,
    "QuantitativeStrategy": QuantitativeStrategy,
    "MarketMakingStrategy": MarketMakingStrategy,
    "HighFrequencyStrategy": HighFrequencyStrategy,
    "AlgorithmicStrategy": AlgorithmicStrategy,
}


__all__ = [
    # Base
    'BaseStrategy',
    'StrategyConfig',
    # Trend
    'MomentumStrategy',
    'BreakoutStrategy',
    'TrendFollowingStrategy',
    'ReversalStrategy',    # Statistical Arbitrage
    'StatisticalArbitrageStrategy',
    'MeanReversionStrategy',
    'CointegrationStrategy',
    'VolatilityBreakoutStrategy',
    'PairsTradeStrategy',
    # ML
    'ReinforcementLearningStrategy',
    'DeepLearningPredictionStrategy',
    'EnsembleLearningStrategy',
    'TransferLearningStrategy',
    # Risk Management
    'DynamicHedgingStrategy',
    'RiskParityStrategy',
    'VaRControlStrategy',
    'MaxDrawdownControlStrategy',
    # Other
    'OptionFlowStrategy',
    'MicrostructureStrategy',
    'CarryTradeStrategy',
    'MacroEconomicStrategy',
    'EventDrivenStrategy',
    'SentimentStrategy',
    'QuantitativeStrategy',
    'MarketMakingStrategy',
    'HighFrequencyStrategy',
    'AlgorithmicStrategy',
    # Add STRATEGY_REGISTRY to __all__ if it\'s intended to be importable via *
    'STRATEGY_REGISTRY'
]
