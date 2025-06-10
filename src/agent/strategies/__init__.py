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
    VolatilityArbitrageStrategy,
    VolatilityBreakoutStrategy
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
    'VolatilityArbitrageStrategy',
    'VolatilityBreakoutStrategy',
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
    'AlgorithmicStrategy'
]
