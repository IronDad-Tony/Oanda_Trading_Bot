# src/agent/strategies/ml_strategies.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sys # Added for dynamic module loading
from .base_strategy import BaseStrategy, StrategyConfig
from typing import Dict, List, Any

# Activation functions that might be used by internal PyTorch models
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class ReinforcementLearningStrategy(BaseStrategy):
    """
    Reinforcement Learning Strategy.
    Note: The core RL logic (agent training, environment interaction, policy updates)
    typically happens outside the direct `forward` and `generate_signals` flow
    of this BaseStrategy. These methods are adapted to fit the framework
    but the primary RL operations are more extensive.
    This strategy, when used in EnhancedStrategySuperposition, might represent
    a pre-trained policy or a component whose parameters are tuned by an RL process.
    """
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        # Example: Define a simple policy network.
        # Dimensions would come from config.default_params
        self.input_dim = int(self.effective_params.get('input_dim', 10)) # Example dimension
        self.action_dim = int(self.effective_params.get('action_dim', 3)) # Example: Buy, Hold, Sell
        
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
            # Output might be logits for actions, or continuous action values
        )
        # print(f"{self.config.strategy_id}: Initialized. Model expects input_dim={self.input_dim}. RL training/inference is specialized.")

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context=None) -> Dict[str, pd.DataFrame]:
        """
        For an RL strategy, 'forward' might prepare state representations from market data.
        However, the direct application of a policy usually happens in generate_signals or an external loop.
        This method will focus on feature engineering if specified by RL state design.
        """
        # print(f"{self.config.strategy_id}: forward() called. RL state preparation would occur here if designed.")
        # For now, pass through or create minimal features.
        # This depends heavily on the specific RL agent's state definition.
        # Example: create a dummy feature set.
        processed_data = {}
        for asset, df in market_data_dict.items():
            if 'close' in df.columns and len(df) >= self.input_dim:
                # Create some placeholder features of size self.input_dim
                # In a real scenario, these would be meaningful features (lags, indicators, etc.)
                # Taking the last 'input_dim' close prices as a feature vector
                features = df['close'].tail(self.input_dim).values.reshape(1, -1)
                # Create a DataFrame for these features, aligning index with the last timestamp
                feature_df = pd.DataFrame(features, index=[df.index[-1]], columns=[f'feature_{i}' for i in range(self.input_dim)])
                processed_data[asset] = feature_df
            # else:
                # print(f"{self.config.strategy_id}: Insufficient data or no 'close' column for {asset} to create RL state features.")
        return processed_data

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context=None) -> pd.DataFrame:
        """
        Generates signals using the RL policy model.
        The 'processed_data_dict' should contain the state features.
        """
        # print(f"{self.config.name}: generate_signals() called. RL policy inference would occur here.")
        all_signals_list = []

        for asset, features_df in processed_data_dict.items():
            if not features_df.empty and features_df.shape[1] == self.input_dim:
                current_state_np = features_df.iloc[-1].values.astype(np.float32)
                current_state_tensor = torch.from_numpy(current_state_np).unsqueeze(0) # Add batch dimension
                
                with torch.no_grad():
                    action_logits = self.model(current_state_tensor)
                    # Assuming action_logits correspond to [Sell, Hold, Buy] or similar
                    # For discrete actions, take argmax or sample.
                    # Example: argmax to get the action index
                    action_idx = torch.argmax(action_logits, dim=1).item() 
                    # Map action_idx to signal: 0 -> -1 (Sell), 1 -> 0 (Hold), 2 -> 1 (Buy)
                    signal_val = action_idx - 1 # Adjust if action_dim or mapping is different
                
                signal_time = features_df.index[-1]
                all_signals_list.append({'timestamp': signal_time, 'asset': asset, 'signal': signal_val})
            # else:
                # print(f"{self.config.strategy_id}: No or incompatible features for {asset} in processed_data_dict.")

        if not all_signals_list:
            return pd.DataFrame()

        signals_df = pd.DataFrame(all_signals_list)
        if signals_df.empty:
            return pd.DataFrame()
            
        signals_df = signals_df.pivot(index='timestamp', columns='asset', values='signal').fillna(0).astype(int)
        return signals_df

class DeepLearningPredictionStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.lookback_window = int(self.effective_params.get('lookback_window', 20))
        self.output_dim = int(self.effective_params.get('output_dim', 1)) # e.g., 1 for regression (price change), 3 for classification (sell/hold/buy)
        
        # Example: Simple MLP model. In reality, could be LSTM, GRU, Transformer, etc.
        self.model = nn.Sequential(
            nn.Linear(self.lookback_window, 128),
            Mish(), # Using Mish activation
            nn.Linear(128, 64),
            Swish(),  # Using Swish activation
            nn.Linear(64, self.output_dim)
        )
        # print(f"{self.config.strategy_id}: Initialized with lookback={self.lookback_window}, output_dim={self.output_dim}.")

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context=None) -> Dict[str, pd.DataFrame]:
        processed_data = {}
        for asset, df in market_data_dict.items():
            if 'close' not in df.columns:
                # print(f"{self.config.strategy_id}: 'close' column missing for {asset}. Skipping.")
                continue
            
            if len(df) < self.lookback_window:
                # print(f"{self.config.strategy_id}: Insufficient data for {asset} ({len(df)} points) for lookback {self.lookback_window}. Skipping.")
                continue
            
            # Create features: use rolling windows of 'close' prices as input features
            # Each row in the output DataFrame will be a feature vector for a given timestamp
            features_list = []
            for i in range(self.lookback_window -1, len(df)):
                feature_vector = df['close'].iloc[i - self.lookback_window + 1 : i + 1].values
                features_list.append(feature_vector)
            
            if not features_list:
                continue

            feature_df = pd.DataFrame(
                features_list, 
                index=df.index[self.lookback_window-1:], 
                columns=[f'lag_{i}' for i in range(self.lookback_window)]
            )
            processed_data[asset] = feature_df
        return processed_data

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context=None) -> pd.DataFrame:
        all_asset_signals_dfs = []

        for asset, features_df in processed_data_dict.items():
            if features_df.empty or features_df.shape[1] != self.lookback_window:
                # print(f"{self.config.strategy_id}: No or incompatible features for {asset}. Expected {self.lookback_window} features.")
                continue

            # Convert features to tensor
            features_tensor = torch.from_numpy(features_df.values.astype(np.float32))
            
            asset_signals = pd.Series(index=features_df.index, dtype=int)

            with torch.no_grad():
                predictions = self.model(features_tensor) # (num_samples, output_dim)

            # Interpret predictions to signals
            if self.output_dim == 1: # Regression: predict price change or value
                # Example: if prediction > threshold, buy; if < -threshold, sell
                # This is a simplified interpretation.
                signals_raw = predictions.squeeze().numpy()
                asset_signals = np.sign(signals_raw).astype(int) # Simple sign of predicted change
            elif self.output_dim == 3: # Classification: [sell, hold, buy]
                action_indices = torch.argmax(predictions, dim=1).numpy()
                # Map: 0 -> -1 (sell), 1 -> 0 (hold), 2 -> 1 (buy)
                signal_map = {-1: -1, 0: -1, 1:0, 2:1} # Adjust if class order is different
                asset_signals = np.array([signal_map.get(idx,0) for idx in action_indices])
            else: # Default to no signal if output_dim is not handled
                asset_signals.values[:] = 0
            
            signals_for_asset_df = pd.DataFrame(asset_signals, index=features_df.index, columns=[asset])
            all_asset_signals_dfs.append(signals_for_asset_df)

        if not all_asset_signals_dfs:
            return pd.DataFrame()
            
        final_signals_df = pd.concat(all_asset_signals_dfs, axis=1)
        final_signals_df = final_signals_df.fillna(method='ffill').fillna(0)
        final_signals_df = final_signals_df.astype(int)
        return final_signals_df

class EnsembleLearningStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.base_strategy_configs: List[Dict[str, Any]] = self.effective_params.get('base_strategy_configs', [])
        self.base_strategies: List[BaseStrategy] = []
        
        print(f"[{self.config.name}] Initializing EnsembleLearningStrategy. Found {len(self.base_strategy_configs)} base strategy configurations.")

        try:
            # Assuming __package__ is 'src.agent.strategies' or similar,
            # giving access to all classes imported in src/agent/strategies/__init__.py
            strategies_module = sys.modules[__package__]
        except KeyError:
            print(f"[{self.config.name}] CRITICAL: Could not determine package for dynamic strategy loading. __package__ is not set. Ensure this module is run as part of a package.")
            strategies_module = None 
            # Attempt to import the strategies module directly if __package__ is None (e.g. running script directly)
            # This is a fallback and might not always work as expected depending on PYTHONPATH.
            if __name__ == '__main__' or __package__ is None: # Check if running as script or package context is missing
                try:
                    import src.agent.strategies as strategies_fallback_module
                    strategies_module = strategies_fallback_module
                    print(f"[{self.config.name}] Fallback: Attempting to use 'src.agent.strategies' module directly.")
                except ImportError:
                    print(f"[{self.config.name}] CRITICAL: Fallback import of 'src.agent.strategies' failed. Dynamic loading of base strategies will not work.")
                    strategies_module = None


        if strategies_module:
            for i, bs_conf_item in enumerate(self.base_strategy_configs):
                if not isinstance(bs_conf_item, dict):
                    print(f"[{self.config.name}] Warning: Base strategy config item #{i} is not a dict: {bs_conf_item}. Skipping.")
                    continue

                strategy_class_name = bs_conf_item.get('strategy_class_name')
                strategy_name = bs_conf_item.get('name') # Changed from strategy_id
                params = bs_conf_item.get('default_params', {}) # Changed from params
                # Allow specifying assets per base strategy, defaults to ensemble's assets or empty list
                assets = bs_conf_item.get('applicable_assets', self.config.applicable_assets or []) # Changed from assets

                if not strategy_class_name or not strategy_name:
                    print(f"[{self.config.name}] Warning: Base strategy config item #{i} missing 'strategy_class_name' or 'name': {bs_conf_item}. Skipping.")
                    continue

                print(f"[{self.config.name}] Attempting to load base strategy: Name='{strategy_name}', Class='{strategy_class_name}'")
                if hasattr(strategies_module, strategy_class_name):
                    strategy_class = getattr(strategies_module, strategy_class_name)
                    try:
                        base_config = StrategyConfig(
                            name=strategy_name, # Changed from strategy_id
                            default_params=params, # Changed from params
                            applicable_assets=assets # Changed from assets
                        )
                        self.base_strategies.append(strategy_class(config=base_config))
                        print(f"[{self.config.name}] Successfully instantiated base strategy: {strategy_name} ({strategy_class_name})")
                    except Exception as e:
                        print(f"[{self.config.name}] Error instantiating base strategy {strategy_class_name} with name {strategy_name}: {e}")
                else:
                    print(f"[{self.config.name}] Warning: Base strategy class '{strategy_class_name}' not found in strategies module '{__package__}'. Skipping.")
        else:
            print(f"[{self.config.name}] Warning: Strategies module not available. Cannot instantiate base strategies for ensemble.")
        
        print(f"[{self.config.name}] Initialized. Actual base strategies count: {len(self.base_strategies)}.")


    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context=None) -> Dict[str, pd.DataFrame]:
        # print(f"[{self.config.name}] Ensemble: forward() called.")
        aggregated_processed_data = {}
        
        if not self.base_strategies:
             print(f"  [{self.config.name}] Ensemble: No base strategies configured or loaded, passing through market_data_dict.")
             return market_data_dict 
             
        for base_strategy in self.base_strategies:
            try:
                # print(f"  [{self.config.name}] Ensemble: Calling forward for {base_strategy.config.name}")
                processed_data_child = base_strategy.forward(market_data_dict, portfolio_context)
                # Prefix keys to avoid collision if strategies produce same asset keys but different features
                for asset, df_child in processed_data_child.items():
                    aggregated_processed_data[f"{base_strategy.config.name}_{asset}"] = df_child
            except Exception as e:
                print(f"  [{self.config.name}] Error in forward pass of base strategy {base_strategy.config.name}: {e}")
                # Continue with other strategies
                pass
        
        return aggregated_processed_data


    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context=None) -> pd.DataFrame:
        # print(f"[{self.config.name}] Ensemble: generate_signals() called.")
        if not self.base_strategies:
            print(f"[{self.config.name}] Ensemble: No base strategies to generate signals from.")
            return pd.DataFrame()

        all_child_signals_dfs: List[pd.DataFrame] = []
        for base_strategy in self.base_strategies:
            try:
                # print(f"  [{self.config.name}] Ensemble: Calling generate_signals for {base_strategy.config.name}")
                
                child_specific_processed_data = {}
                # Extract data relevant to this child strategy from the aggregated processed_data_dict
                # The keys in processed_data_dict are prefixed like "child_strategy_name_asset_name"
                for key, value_df in processed_data_dict.items():
                    prefix = f"{base_strategy.config.name}_"
                    if key.startswith(prefix):
                        asset_name = key[len(prefix):] # Get original asset name
                        child_specific_processed_data[asset_name] = value_df
                
                # If no specific data was prepared for this child by the ensemble's forward pass
                # (e.g., child's forward failed or produced nothing), child_specific_processed_data will be empty.
                # The child strategy should handle an empty input dict gracefully.
                if not child_specific_processed_data:
                    print(f"  [{self.config.name}] Ensemble: No specific processed data found for child {base_strategy.config.name} from ensemble's forward pass. Passing empty dict.")

                child_signals_df = base_strategy.generate_signals(child_specific_processed_data, portfolio_context)
                
                if not child_signals_df.empty:
                    all_child_signals_dfs.append(child_signals_df)
                # else:
                #     print(f"  [{self.config.name}] Ensemble: Child strategy {base_strategy.config.name} produced no signals.")

            except Exception as e:
                print(f"  [{self.config.name}] Error in generate_signals of base strategy {base_strategy.config.name}: {e}")
                # Continue with other strategies
                pass
        
        if not all_child_signals_dfs:
            print(f"[{self.config.name}] Ensemble: No signals generated by any base strategy.")
            return pd.DataFrame()

        # Combine signals:
        # Concatenate along columns. If strategies signal on the same assets, columns will be like asset, asset_2, etc.
        # Or, if they use unique column names (assets), they will just be added.
        # Pandas handles alignment by index (timestamp).
        try:
            combined_df = pd.concat(all_child_signals_dfs, axis=1)
        except Exception as e:
            print(f"[{self.config.name}] Ensemble: Error during pd.concat of child signals: {e}")
            return pd.DataFrame() # Return empty if combination fails

        # Sum signals for the same asset from different strategies.
        # groupby(level=0, axis=1) groups by column names. If multiple strategies output signals
        # for the same asset (e.g., 'EUR_USD'), their signals for 'EUR_USD' will be summed.
        final_signals_df = combined_df.groupby(level=0, axis=1).sum()

        # Normalize/finalize signals (e.g., sign of sum for a voting-like mechanism)
        final_signals_df = final_signals_df.apply(np.sign).astype(int)
        # Fill any NaNs that might arise from non-overlapping signals or issues.
        final_signals_df = final_signals_df.fillna(0) 
        
        # print(f"[{self.config.name}] Ensemble: Generated final signals shape: {final_signals_df.shape}")
        return final_signals_df

class TransferLearningStrategy(BaseStrategy):
    """
    Transfer Learning Strategy.
    Uses a pre-trained model for feature extraction and a new head for the specific task.
    """
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.lookback_window = int(self.effective_params.get('lookback_window', 50)) # Features for pre-trained model
        self.num_extracted_features = int(self.effective_params.get('num_extracted_features', 64)) # Output of feature extractor
        self.output_dim = int(self.effective_params.get('output_dim', 1)) # Output of the new head
        self.pretrained_model_path = self.effective_params.get('pretrained_model_path', None)

        # Define a placeholder feature extractor (e.g., part of a larger pre-trained model)
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.lookback_window, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_extracted_features),
            nn.ReLU()
        )
        
        # Define a new head for the specific trading task
        self.head = nn.Sequential(
            nn.Linear(self.num_extracted_features, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_dim)
        )

        if self.pretrained_model_path:
            self._load_pretrained_model()
        
        # print(f"{self.config.strategy_id}: Initialized. Lookback={self.lookback_window}, ExtractedFeatures={self.num_extracted_features}, OutputDim={self.output_dim}.")

    def _load_pretrained_model(self):
        # Placeholder for loading weights into self.feature_extractor
        # This typically involves torch.load() and model.load_state_dict()
        # print(f"{self.config.name}: Attempting to load pre-trained model from {self.pretrained_model_path}. (Placeholder logic)")
        try:
            # Example: self.feature_extractor.load_state_dict(torch.load(self.pretrained_model_path))
            # This requires the saved model to match the feature_extractor architecture.
            # For fine-tuning, some layers might be frozen:
            # for param in self.feature_extractor.parameters():
            #     param.requires_grad = False
            pass # Actual loading logic here
        except Exception as e:
            # print(f"Error loading pre-trained model for {self.config.name}: {e}. Using initialized weights.")
            pass

    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context=None) -> Dict[str, pd.DataFrame]:
        # Similar to DeepLearningPredictionStrategy: prepare input features for the model
        processed_data = {}
        for asset, df in market_data_dict.items():
            if 'close' not in df.columns or len(df) < self.lookback_window:
                continue
            
            features_list = []
            for i in range(self.lookback_window - 1, len(df)):
                feature_vector = df['close'].iloc[i - self.lookback_window + 1 : i + 1].values
                features_list.append(feature_vector)
            
            if not features_list:
                continue

            feature_df = pd.DataFrame(
                features_list, 
                index=df.index[self.lookback_window-1:], 
                columns=[f'tl_lag_{i}' for i in range(self.lookback_window)]
            )
            processed_data[asset] = feature_df
        return processed_data

    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context=None) -> pd.DataFrame:
        all_asset_signals_dfs = []

        for asset, features_df in processed_data_dict.items():
            if features_df.empty or features_df.shape[1] != self.lookback_window:
                continue

            features_tensor = torch.from_numpy(features_df.values.astype(np.float32))
            asset_signals = pd.Series(index=features_df.index, dtype=int)

            with torch.no_grad():
                extracted_features = self.feature_extractor(features_tensor)
                predictions = self.head(extracted_features)

            if self.output_dim == 1:
                signals_raw = predictions.squeeze().numpy()
                asset_signals = np.sign(signals_raw).astype(int)
            elif self.output_dim == 3: # Example for classification
                action_indices = torch.argmax(predictions, dim=1).numpy()
                signal_map = {0: -1, 1: 0, 2: 1} # sell, hold, buy
                asset_signals = np.array([signal_map.get(idx, 0) for idx in action_indices])
            else:
                asset_signals.values[:] = 0
            
            signals_for_asset_df = pd.DataFrame(asset_signals, index=features_df.index, columns=[asset])
            all_asset_signals_dfs.append(signals_for_asset_df)

        if not all_asset_signals_dfs:
            return pd.DataFrame()
            
        final_signals_df = pd.concat(all_asset_signals_dfs, axis=1)
        final_signals_df = final_signals_df.fillna(method='ffill').fillna(0)
        final_signals_df = final_signals_df.astype(int)
        return final_signals_df
