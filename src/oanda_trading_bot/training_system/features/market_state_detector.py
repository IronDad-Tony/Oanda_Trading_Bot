import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
import joblib # Added for saving/loading model
from typing import Union, List

class GMMMarketStateDetector:
    def __init__(self, n_states=3, gmm_config=None, feature_config=None, random_state=None):
        """
        Initializes the GMMMarketStateDetector.

        Args:
            n_states (int): Number of market states to detect.
            gmm_config (dict, optional): Configuration for GaussianMixture.
                                           Example: {'covariance_type': 'full'}
            feature_config (dict, optional): Configuration for input feature calculation.
                                           Example: {'return_window': 5, 'vol_window': 20}
            random_state (int, optional): Random state for GMM for reproducibility.
        """
        self.n_states = n_states
        self.gmm_config = gmm_config if gmm_config else {'covariance_type': 'diag'} # Keep diag as a robust default
        self.feature_config = feature_config if feature_config else self._default_feature_config()
        self.random_state = random_state
        
        self.gmm_model = GaussianMixture(n_components=self.n_states,
                                         random_state=self.random_state,
                                         **self.gmm_config)
        self.scaler = None 
        self.fitted = False

    def _default_feature_config(self):
        return {
            'return_window': 5, 
            'vol_window': 20, 
            'ma_short_window': 10, 
            'ma_long_window': 30,
            'atr_window': 14
        }

    def _calculate_features(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates basic statistical features for GMM input.
        Args:
            ohlcv_data (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume' columns.
                                       Requires 'Close', and 'High', 'Low' if ATR is to be calculated.
        Returns:
            pd.DataFrame: DataFrame with features. Rows with NaNs are dropped.
        """
        if not isinstance(ohlcv_data, pd.DataFrame):
            raise ValueError("ohlcv_data must be a pandas DataFrame.")
        required_cols = ['Close']
        if 'atr_window' in self.feature_config: # ATR requires High, Low
             required_cols.extend(['High', 'Low'])
        
        for col in required_cols:
            if col not in ohlcv_data.columns:
                # If ATR is configured but H/L are missing, it will be NaN and handled.
                # Only 'Close' is strictly essential for other features.
                if col == 'Close':
                    raise ValueError(f"ohlcv_data must contain '{col}' column.")

        features = pd.DataFrame(index=ohlcv_data.index)
        safe_close = ohlcv_data['Close'].replace(0, np.nan).bfill().ffill()
        
        features['log_return'] = np.log(safe_close / safe_close.shift(1))
        
        min_p_ret = max(1, self.feature_config["return_window"] // 2)
        features[f'return_mean_{self.feature_config["return_window"]}'] = \
            features['log_return'].rolling(window=self.feature_config["return_window"], min_periods=min_p_ret).mean()

        min_p_vol = max(1, self.feature_config["vol_window"] // 2)
        features[f'volatility_{self.feature_config["vol_window"]}'] = \
            features['log_return'].rolling(window=self.feature_config["vol_window"], min_periods=min_p_vol).std() * np.sqrt(252) # Annualized

        min_p_ma_short = max(1, self.feature_config['ma_short_window'] // 2)
        ma_short = safe_close.rolling(window=self.feature_config['ma_short_window'], min_periods=min_p_ma_short).mean()
        
        min_p_ma_long = max(1, self.feature_config['ma_long_window'] // 2)
        ma_long = safe_close.rolling(window=self.feature_config['ma_long_window'], min_periods=min_p_ma_long).mean()
        
        features['price_ma_short_ratio'] = safe_close / ma_short.replace(0, np.nan) 
        features['ma_short_long_ratio'] = ma_short.replace(0, np.nan) / ma_long.replace(0, np.nan)
        
        atr_window_val = self.feature_config.get('atr_window')
        if atr_window_val and 'High' in ohlcv_data.columns and 'Low' in ohlcv_data.columns:
            prev_close_safe = safe_close.shift(1)
            if not prev_close_safe.isnull().all():
                tr1 = ohlcv_data['High'] - ohlcv_data['Low']
                tr2 = np.abs(ohlcv_data['High'] - prev_close_safe)
                tr3 = np.abs(ohlcv_data['Low'] - prev_close_safe)
                tr_df = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}, index=ohlcv_data.index)
                true_range = tr_df.max(axis=1)
                
                # Using EWM for ATR (Wilder's smoothing style: span = 2*N - 1, or alpha = 1/N)
                # Pandas ewm with span=N is alpha = 2/(N+1).
                # For Wilder's N, use ewm(alpha=1/N, adjust=False)
                # Or, a common approximation: ewm(span=N, adjust=False, min_periods=N)
                features['atr'] = true_range.ewm(span=atr_window_val, adjust=False, min_periods=atr_window_val).mean()
            else:
                features['atr'] = np.nan
        else:
            features['atr'] = np.nan

        # Replace infs and drop all rows with any NaNs
        return features.replace([np.inf, -np.inf], np.nan).dropna()


    def fit(self, ohlcv_data_history: Union[pd.DataFrame, List[pd.DataFrame]]):
        """
        Trains the GMM model on historical data.
        Args:
            ohlcv_data_history (Union[pd.DataFrame, List[pd.DataFrame]]): 
                Historical OHLCV data for training. Can be a single DataFrame
                or a list of DataFrames (e.g., for multiple symbols or periods).
        """
        self.fitted = False # Reset fitted status
        self.scaler = None  # Reset scaler

        all_features_list = []
        if isinstance(ohlcv_data_history, pd.DataFrame):
            features = self._calculate_features(ohlcv_data_history)
            if not features.empty:
                all_features_list.append(features)
        elif isinstance(ohlcv_data_history, list):
            for df in ohlcv_data_history:
                if not isinstance(df, pd.DataFrame):
                    warnings.warn(f"Item in ohlcv_data_history is not a DataFrame. Skipping. Type: {type(df)}", UserWarning)
                    continue
                features = self._calculate_features(df)
                if not features.empty:
                    all_features_list.append(features)
        else:
            raise ValueError("ohlcv_data_history must be a pandas DataFrame or a list of pandas DataFrames.")

        if not all_features_list:
            warnings.warn("No features could be calculated from the provided ohlcv_data_history. GMM not trained.", UserWarning)
            return
            
        features_history_df = pd.concat(all_features_list, ignore_index=True)
        
        if features_history_df.empty or len(features_history_df) < self.n_states:
            warnings.warn(f"Not enough data or features to train GMM. Need at least {self.n_states} valid samples after feature calculation. Got {len(features_history_df)} from {len(all_features_list)} dataframes. GMM not trained.", UserWarning)
            return

        self.scaler = StandardScaler()
        try:
            scaled_features = self.scaler.fit_transform(features_history_df)
        except ValueError as e:
            warnings.warn(f"Error during scaling of features: {e}. Cannot train GMM.", UserWarning)
            self.scaler = None # Invalidate scaler
            return
        except Exception as e: # Catch any other error during scaling
            warnings.warn(f"An unexpected error occurred during feature scaling: {e}. Cannot train GMM.", UserWarning)
            self.scaler = None
            return


        # self.gmm_model is already instantiated in __init__
        try:
            self.gmm_model.fit(scaled_features)
            # print(f"GMMMarketStateDetector trained with {self.n_states} states.")
            self.fitted = True # Mark as successfully fitted
        except ValueError as e:
            warnings.warn(f"Error training GMM: {e}. This might be due to insufficient or collinear data after scaling. GMM not fitted.", UserWarning)
            self.fitted = False 
            self.scaler = None # Invalidate scaler if GMM fit failed, as it's tied to this fitting attempt.
        except Exception as e: # Generic catch for other potential issues during fit
            warnings.warn(f"An unexpected error occurred during GMM fitting: {e}. GMM not fitted.", UserWarning)
            self.fitted = False
            self.scaler = None


    def predict_state_probabilities(self, current_ohlcv_data_segment: pd.DataFrame) -> np.ndarray:
        """
        Predicts the probability of being in each market state for the latest data point.
        Args:
            current_ohlcv_data_segment (pd.DataFrame): Recent segment of OHLCV data,
                                                       long enough to calculate all features.
        Returns:
            np.ndarray: Probabilities for each state for the latest data point (1D array), 
                        or a uniform distribution if prediction fails or model not ready.
        """
        if not self.fitted or self.scaler is None: 
            # warnings.warn("GMM model not successfully trained or scaler not available. Returning uniform probabilities.", UserWarning)
            return np.full(self.n_states, 1.0 / self.n_states) 
        
        features = self._calculate_features(current_ohlcv_data_segment)
        
        if features.empty:
            # warnings.warn("Not enough data in segment to calculate features for prediction. Returning uniform probabilities.", UserWarning)
            return np.full(self.n_states, 1.0 / self.n_states)

        latest_features_unscaled = features.iloc[-1:] 
        
        # Check for NaNs again, although dropna() in _calculate_features should handle it.
        if latest_features_unscaled.isnull().any().any():
            # warnings.warn("Latest features for prediction contain NaNs after calculation. Returning uniform probabilities.", UserWarning)
            return np.full(self.n_states, 1.0 / self.n_states)

        try:
            scaled_latest_features = self.scaler.transform(latest_features_unscaled)
        except Exception as e: # Could be NotFittedError if scaler wasn't properly fitted, or ValueError
            # warnings.warn(f"Error scaling features for prediction: {e}. Returning uniform probabilities.", UserWarning)
            return np.full(self.n_states, 1.0 / self.n_states)
        
        try:
            # GMM's predict_proba returns a 2D array (n_samples, n_components)
            # Since we pass one sample, we take the first row.
            return self.gmm_model.predict_proba(scaled_latest_features)[0]
        except Exception as e:
            # warnings.warn(f"Error during GMM predict_proba: {e}. Returning uniform probabilities.", UserWarning)
            return np.full(self.n_states, 1.0 / self.n_states)


    def predict_state(self, current_ohlcv_data_segment: pd.DataFrame) -> int:
        """
        Predicts the most likely market state for the latest data point.
        Returns:
            int: The most likely state index, or -1 if prediction fails (represented by argmax of uniform).
        """
        probabilities = self.predict_state_probabilities(current_ohlcv_data_segment)
        # If probabilities are uniform (e.g., due to an issue or model not ready),
        # np.argmax will return 0. This is an acceptable default.
        return int(np.argmax(probabilities))

    def save_model(self, path: str):
        """
        Saves the GMMMarketStateDetector instance to a file.
        Args:
            path (str): Path to save the model file.
        """
        if not self.fitted:
            warnings.warn("Attempting to save a model that has not been successfully fitted. "
                          "The saved model may not be usable for predictions.", UserWarning)
        
        model_data = {
            'gmm_model': self.gmm_model,
            'scaler': self.scaler,
            'n_states': self.n_states,
            'gmm_config': self.gmm_config,
            'feature_config': self.feature_config,
            'random_state': self.random_state,
            'fitted': self.fitted
        }
        try:
            joblib.dump(model_data, path)
            # print(f"Model saved to {path}")
        except Exception as e:
            warnings.warn(f"Error saving model to {path}: {e}", UserWarning)
            raise # Re-raise the exception after warning

    @staticmethod
    def load_model(path: str) -> 'GMMMarketStateDetector':
        """
        Loads a GMMMarketStateDetector instance from a file.
        Args:
            path (str): Path to the model file.
        Returns:
            GMMMarketStateDetector: The loaded instance.
        """
        try:
            model_data = joblib.load(path)
        except Exception as e:
            warnings.warn(f"Error loading model from {path}: {e}", UserWarning)
            raise # Re-raise

        # Create an instance and populate it
        # Need to handle the case where gmm_config might not exist in older saved models for robustness
        # However, current save_model includes it.
        loaded_detector = GMMMarketStateDetector(
            n_states=model_data['n_states'],
            gmm_config=model_data.get('gmm_config', {'covariance_type': 'diag'}), # Default if missing
            feature_config=model_data.get('feature_config', GMMMarketStateDetector()._default_feature_config()), # Default if missing
            random_state=model_data.get('random_state') 
        )
        
        loaded_detector.gmm_model = model_data['gmm_model']
        loaded_detector.scaler = model_data['scaler']
        loaded_detector.fitted = model_data['fitted']
        
        # Ensure the loaded GMM model's n_components and random_state are consistent if they were part of its direct attributes
        # This is more of a sanity check, as they should be set by the initial GMM instantiation.
        if hasattr(loaded_detector.gmm_model, 'n_components') and loaded_detector.gmm_model.n_components != loaded_detector.n_states:
             warnings.warn(f"Loaded GMM model n_components ({loaded_detector.gmm_model.n_components}) "
                           f"differs from detector n_states ({loaded_detector.n_states}). Check model integrity.", UserWarning)

        # print(f"Model loaded from {path}")
        return loaded_detector

# Example Usage (for testing purposes within the module)
if __name__ == '__main__':
    periods = 500
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    
    # Generate more structured data for better GMM differentiation
    close_prices = np.zeros(periods)
    high_prices = np.zeros(periods)
    low_prices = np.zeros(periods)
    open_prices = np.zeros(periods) # Add Open for completeness
    volume = np.random.randint(1000, 5000, size=periods)


    close_prices[0] = 100
    regime = 0
    for i in range(1, periods):
        if i % 100 == 0: # Switch regime every 100 days
            regime = (regime + 1) % 3
        
        if regime == 0: # Trending up, low vol
            change = np.random.normal(0.1, 0.5)
            close_prices[i] = close_prices[i-1] + change
        elif regime == 1: # Ranging, high vol
            change = np.random.normal(0, 2.0)
            close_prices[i] = close_prices[i-1] + change
        else: # Trending down, mid vol
            change = np.random.normal(-0.05, 1.0)
            close_prices[i] = close_prices[i-1] + change
        
        close_prices[i] = max(close_prices[i], 1) # Ensure price is positive
        open_prices[i] = close_prices[i-1] # Simplistic Open
        high_prices[i] = max(close_prices[i], open_prices[i]) + np.random.rand() * 0.5 # Simplistic High
        low_prices[i] = min(close_prices[i], open_prices[i]) - np.random.rand() * 0.5   # Simplistic Low
        low_prices[i] = max(low_prices[i], 0.1) # Ensure low is positive

    sample_data = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    }, index=dates)
    
    sample_data.iloc[0, sample_data.columns.get_loc('Open')] = sample_data.iloc[0, sample_data.columns.get_loc('Close')] # Set first Open to first Close


    print("Sample Data Head:")
    print(sample_data.head())

    # Using default feature config which now includes EWM for ATR
    detector = GMMMarketStateDetector(n_states=3, random_state=42) 
    
    print("\\nCalculating features for the full dataset...")
    features_full = detector._calculate_features(sample_data)
    print(f"Features calculated. Shape: {features_full.shape}")
    if not features_full.empty:
        print("Features Head:")
        print(features_full.head())
        print("Features Tail:")
        print(features_full.tail())
        print(f"NaNs in features: {features_full.isnull().sum().sum()}")
        print(f"Infs in features: {features_full.isin([np.inf, -np.inf]).sum().sum()}")
    else:
        print("No features were generated. Check data and feature calculation logic.")

    print("\\nFitting detector...")
    fit_data_length = max(100, detector.n_states * 20) # Ensure enough data for GMM (e.g. 20 samples per state)
    fit_data_length = min(fit_data_length, len(sample_data))
    
    # Determine min_ohlcv_rows_for_one_feature_row for the current detector config
    # This is a bit complex to calculate perfectly here without replicating test logic
    # For default config with EWM ATR(14, min_p=14), MA_long(30, min_p=15)
    # Max initial NaN index is 15 (from MA_long or its ratio). So 16 rows lost.
    # Min data for 1 feature row = 16 + 1 = 17
    # Let's use a simpler heuristic for the example:
    max_config_window = 0
    temp_fc = detector.feature_config
    windows_to_check = ['return_window', 'vol_window', 'ma_short_window', 'ma_long_window', 'atr_window']
    for k in windows_to_check:
        if k in temp_fc:
            max_config_window = max(max_config_window, temp_fc[k])
    
    min_data_for_calc_heuristic = max_config_window + 2 # A loose lower bound

    if fit_data_length < min_data_for_calc_heuristic:
        print(f"Adjusting fit_data_length from {fit_data_length} to {min_data_for_calc_heuristic} for feature calculation.")
        fit_data_length = min_data_for_calc_heuristic
    
    if len(sample_data) < fit_data_length:
        print(f"Warning: Sample data length ({len(sample_data)}) is less than desired fit_data_length ({fit_data_length}). Using all available data for fitting.")
        fit_data_segment = sample_data
    else:
        fit_data_segment = sample_data.head(fit_data_length)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        detector.fit(fit_data_segment)
        if w:
            print("Warnings during fit:")
            for warning_message in w:
                print(warning_message.message)

    if detector.fitted:
        print("Detector fitted successfully.")
        # ... (GMM parameter printing can be verbose, kept commented)

        # --- Test Save and Load ---
        model_path = "gmm_market_detector_test_model.joblib"
        print(f"\\nSaving model to {model_path}...")
        detector.save_model(model_path)

        print(f"Loading model from {model_path}...")
        loaded_detector = GMMMarketStateDetector.load_model(model_path)
        print("Model loaded successfully.")

        if loaded_detector.fitted:
            print("Loaded detector is marked as fitted.")
            if loaded_detector.scaler:
                 print(f"Loaded Scaler mean: {loaded_detector.scaler.mean_}")
            # Verify some properties
            assert loaded_detector.n_states == detector.n_states
            assert loaded_detector.random_state == detector.random_state
            # It's hard to compare GMM models directly beyond basic properties without deep comparison
            # For example, compare predict_proba output on same data.
        # --- End Test Save and Load ---


        print("\\nPredicting for a segment using original detector...")
        # Prediction segment needs to be long enough for feature calculation
        # Use the same heuristic for prediction segment length
        prediction_segment_len = min_data_for_calc_heuristic + 20 
        if len(sample_data) >= prediction_segment_len:
            prediction_segment = sample_data.tail(prediction_segment_len)
            
            print(f"Using prediction segment of length: {len(prediction_segment)}")
            
            # ... (feature printing for segment commented out for brevity)

            with warnings.catch_warnings(record=True) as w_pred:
                warnings.simplefilter("always")
                probabilities = detector.predict_state_probabilities(prediction_segment)
                state = detector.predict_state(prediction_segment)
                # ... (warnings printing commented out)
            
            print(f"Original detector - Predicted probabilities: {probabilities}")
            print(f"Original detector - Predicted state: {state}")

            if loaded_detector and loaded_detector.fitted:
                print("\\nPredicting for the same segment using loaded detector...")
                probs_loaded = loaded_detector.predict_state_probabilities(prediction_segment)
                state_loaded = loaded_detector.predict_state(prediction_segment)
                print(f"Loaded detector - Predicted probabilities: {probs_loaded}")
                print(f"Loaded detector - Predicted state: {state_loaded}")
                assert np.allclose(probabilities, probs_loaded)
                assert state == state_loaded
                print("Predictions from original and loaded model match.")

        else:
            print(f"Sample data too short for a valid prediction segment (needs at least {prediction_segment_len}).")

    else:
        print("Detector fitting failed or was skipped due to insufficient data.")

    # ... (unfitted detector test can remain)
    print("\\nTesting predict_state_probabilities before fit (new instance):")
    detector_unfitted = GMMMarketStateDetector(n_states=3, random_state=42)
    # Use a segment length that is definitely enough for feature calculation
    unfitted_pred_segment_len = min_data_for_calc_heuristic + 5 
    if len(sample_data) >= unfitted_pred_segment_len:
        probs_unfitted = detector_unfitted.predict_state_probabilities(sample_data.tail(unfitted_pred_segment_len))
        print(f"Probabilities from unfitted model: {probs_unfitted}")
        assert np.allclose(probs_unfitted, 1.0/detector_unfitted.n_states)
        state_unfitted = detector_unfitted.predict_state(sample_data.tail(unfitted_pred_segment_len))
        print(f"State from unfitted model: {state_unfitted}")
        assert state_unfitted == 0 
    else:
        print("Sample data too short for unfitted detector test segment.")

    print("\\nDone with example usage.")
