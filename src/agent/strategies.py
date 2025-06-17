# This file previously contained placeholder strategies and has been cleared.
# Actual strategies are located in the 'strategies' sub-directory.
# The registration and loading should be handled by 'src/agent/strategies/__init__.py'
# and the EnhancedStrategySuperposition layer.

STRATEGY_REGISTRY = {}

def register_strategy(name):
    def decorator(cls):
        STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator

