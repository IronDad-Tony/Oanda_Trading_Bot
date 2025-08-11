import unittest
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from src.features.market_state_detector import GMMMarketStateDetector
import os # Added for save/load tests
import tempfile # Added for save/load tests
import joblib # Added for save/load tests

class TestGMMMarketStateDetector(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.n_states = 3
        self.default_feature_config = GMMMarketStateDetector()._default_feature_config()
        self.detector = GMMMarketStateDetector(n_states=self.n_states, random_state=42)
        
        n_samples = 200 
        data = {
            'Open': np.random.rand(n_samples) * 100 + 100,
            'High': np.random.rand(n_samples) * 10 + 100, 
            'Low': 100 - np.random.rand(n_samples) * 10,   
            'Close': np.random.rand(n_samples) * 100 + 100,
            'Volume': np.random.randint(1000, 10000, size=n_samples)
        }
        self.sample_ohlcv_data = pd.DataFrame(data)
        self.sample_ohlcv_data['High'] = self.sample_ohlcv_data[['Open', 'Close', 'High']].max(axis=1)
        self.sample_ohlcv_data['Low'] = self.sample_ohlcv_data[['Open', 'Close', 'Low']].min(axis=1)
        
        fc = self.default_feature_config
        self.max_window_param = max(
            fc['return_window'], 
            fc['vol_window'], 
            fc['ma_short_window'], 
            fc['ma_long_window'], 
            fc.get('atr_window', 14) 
        )
        # For default config: max(10, 20, 5, 20, 14) = 20
        self.min_rows_for_general_testing = self.max_window_param + 1 # 20 + 1 = 21

        # Max index (0-based) of a row in the input OHLCV data that will result
        # in a feature vector containing at least one NaN due to windowing calculations
        # before dropna(). This determines how many initial rows are effectively lost.
        # For default config, ATR(14) with min_periods=14 on TR (NaN at index 0)
        # makes features NaN up to index 13.
        self.default_max_initial_nan_idx = 13 
        # Min OHLCV rows needed to produce ONE valid feature row:
        # (default_max_initial_nan_idx + 1) rows to cover NaNs from feature calculation,
        # plus 1 more row for the actual features.
        # So, 13 + 1 + 1 = 15 rows.
        self.min_ohlcv_rows_for_one_feature_row = self.default_max_initial_nan_idx + 1 + 1 

        self.prediction_segment = self.sample_ohlcv_data.tail(self.min_ohlcv_rows_for_one_feature_row + 20)

    def test_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.n_states, self.n_states)
        self.assertIsNotNone(self.detector.gmm_model)
        self.assertIsInstance(self.detector.gmm_model, GaussianMixture)
        self.assertEqual(self.detector.gmm_model.n_components, self.n_states)
        self.assertIsNone(self.detector.scaler)
        self.assertFalse(self.detector.fitted) # Check fitted flag
        # weights_ does not exist before fit, checking self.fitted is sufficient

    def test_calculate_features_basic(self):
        """Test feature calculation with good data."""
        test_data_rows = self.min_rows_for_general_testing + 50 # Default: 21 + 50 = 71 rows
        test_data = self.sample_ohlcv_data.head(test_data_rows)
        features = self.detector._calculate_features(test_data)
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(features.shape[1], 6) 
        self.assertFalse(features.isnull().any().any(), "Features should not contain NaNs after internal dropna.")
        self.assertFalse(features.isin([np.inf, -np.inf]).any().any(), "Features should not contain Infs after internal handling.")
        # Expected rows = input_rows - (max_initial_nan_idx + 1)
        expected_feature_rows = test_data_rows - (self.default_max_initial_nan_idx + 1) # 71 - (13 + 1) = 57
        self.assertEqual(features.shape[0], expected_feature_rows)

    def test_calculate_features_min_data_for_one_row(self):
        """Test feature calculation with minimal data just enough for one row of features."""
        min_data_input = self.sample_ohlcv_data.head(self.min_ohlcv_rows_for_one_feature_row) # 15 rows for default
        features_min_data = self.detector._calculate_features(min_data_input)
        self.assertIsInstance(features_min_data, pd.DataFrame)
        self.assertEqual(features_min_data.shape[0], 1, f"Should produce one row of features with {self.min_ohlcv_rows_for_one_feature_row} input rows.")
        self.assertFalse(features_min_data.isnull().any().any(), "Single feature row should be complete.")

    def test_calculate_features_insufficient_data(self):
        """Test _calculate_features with data insufficient to produce any feature rows."""
        insufficient_data_count = self.min_ohlcv_rows_for_one_feature_row - 1 # 14 rows for default
        insufficient_data = self.sample_ohlcv_data.head(insufficient_data_count)
        features = self.detector._calculate_features(insufficient_data)
        self.assertTrue(features.empty, f"Should return empty DataFrame for {insufficient_data_count} rows of data.")

    def test_calculate_features_zero_prices(self):
        """Test feature calculation with zero prices in Close."""
        data_with_zeros = self.sample_ohlcv_data.head(self.min_rows_for_general_testing + 20).copy()
        data_with_zeros.loc[data_with_zeros.index[5:10], 'Close'] = 0
        data_with_zeros['High'] = data_with_zeros[['Open', 'Close', 'High']].max(axis=1)
        data_with_zeros['Low'] = data_with_zeros[['Open', 'Close', 'Low']].min(axis=1)

        features = self.detector._calculate_features(data_with_zeros)
        self.assertFalse(features.isin([np.inf, -np.inf]).any().any(), "Features should handle zeros in Close and not produce Infs.")
        self.assertFalse(features.isnull().any().any(), "Features should not contain NaNs after handling zeros and dropna.")

    def test_calculate_features_constant_prices(self):
        """Test _calculate_features with constant price data."""
        constant_data_rows = self.min_rows_for_general_testing + 20
        constant_price_data = pd.DataFrame({
            'Open': [100] * constant_data_rows, 
            'High': [100] * constant_data_rows, 
            'Low': [100] * constant_data_rows, 
            'Close': [100] * constant_data_rows, 
            'Volume': [1000] * constant_data_rows
        }, index=pd.date_range(start='2023-01-01', periods=constant_data_rows))
        
        features = self.detector._calculate_features(constant_price_data)
        self.assertFalse(features.empty)
        self.assertFalse(features.isnull().any().any())
        self.assertTrue(np.allclose(features['log_return'], 0, atol=1e-9))
        self.assertTrue(np.allclose(features[f'volatility_{self.default_feature_config["vol_window"]}'], 0, atol=1e-9))
        if 'atr' in features.columns: # atr column name is just 'atr'
            self.assertTrue(np.allclose(features['atr'], 0, atol=1e-9))
        self.assertTrue(np.allclose(features['price_ma_short_ratio'], 1, atol=1e-9))
        self.assertTrue(np.allclose(features['ma_short_long_ratio'], 1, atol=1e-9))

    def test_fit(self):
        """Test fitting the GMM model."""
        fitting_data = self.sample_ohlcv_data.head(self.min_rows_for_general_testing + 50) 
        self.detector.fit(fitting_data)
        self.assertTrue(self.detector.fitted, "Fitted flag should be True after successful fit.")
        self.assertIsNotNone(self.detector.scaler, "Scaler should be initialized after fit.")
        self.assertIsNotNone(self.detector.gmm_model.weights_, "GMM model should be fitted and have weights.")
        self.assertEqual(len(self.detector.gmm_model.weights_), self.n_states)

    def test_fit_not_enough_data_for_gmm(self):
        """Test fitting with insufficient data for GMM after feature calculation."""
        # We want n_states - 1 feature rows to trigger the warning.
        # For n_states = 3, this is 2 feature rows.
        num_feature_rows_needed = self.n_states - 1 # 2 feature rows
        # Input OHLCV rows = (rows_lost_to_nan_features) + num_feature_rows_needed
        # rows_lost_to_nan_features = default_max_initial_nan_idx + 1 = 13 + 1 = 14
        insufficient_data_count = (self.default_max_initial_nan_idx + 1) + num_feature_rows_needed # 14 + 2 = 16 rows
        insufficient_data = self.sample_ohlcv_data.head(insufficient_data_count)

        with self.assertWarns(UserWarning) as cm:
            self.detector.fit(insufficient_data)

        expected_features_available_for_gmm = num_feature_rows_needed # Should be 2
        # Updated expected message to match the actual warning format more closely.
        # The key part is that the number of samples is correctly identified.
        self.assertTrue(f"Need at least {self.n_states} valid samples after feature calculation" in str(cm.warning))
        self.assertTrue(f"Got {expected_features_available_for_gmm}" in str(cm.warning))
        self.assertFalse(self.detector.fitted, "Fitted flag should be False when GMM training is skipped.")
        self.assertIsNone(self.detector.scaler, "Scaler should be None when GMM training is skipped.")

    def test_predict_state_probabilities(self):
        """Test predicting state probabilities."""
        self.detector.fit(self.sample_ohlcv_data) 
        
        # Use the pre-defined self.prediction_segment which is tailored to be long enough
        probabilities = self.detector.predict_state_probabilities(self.prediction_segment)
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(probabilities.shape, (self.n_states,)) # predict_state_probabilities now returns 1D
        self.assertAlmostEqual(np.sum(probabilities), 1.0, places=5)
        self.assertTrue(all(0 <= p <= 1 for p in probabilities))

    def test_predict_state(self):
        """Test predicting the most likely state."""
        self.detector.fit(self.sample_ohlcv_data)
        state = self.detector.predict_state(self.prediction_segment)
        self.assertIsInstance(state, int) # predict_state casts to int
        self.assertTrue(0 <= state < self.n_states)

    def test_predict_before_fit(self):
        """Test prediction methods before fitting the model."""
        unfitted_detector = GMMMarketStateDetector(n_states=self.n_states, random_state=42)
        
        # predict_state_probabilities should return uniform if not fitted
        probs = unfitted_detector.predict_state_probabilities(self.prediction_segment)
        self.assertTrue(np.allclose(probs, np.full(self.n_states, 1.0 / self.n_states)))

        # predict_state will then return argmax of uniform, typically 0
        state = unfitted_detector.predict_state(self.prediction_segment)
        self.assertEqual(state, 0) # or handle as a specific "unknown" state if desired

    def test_predict_with_insufficient_segment_data(self):
        """Test prediction when segment is too short for feature calculation."""
        self.detector.fit(self.sample_ohlcv_data)
        # Segment too short to produce any features (e.g., 1 row for default config)
        short_segment = self.sample_ohlcv_data.tail(self.min_ohlcv_rows_for_one_feature_row - 1) 
        
        probs = self.detector.predict_state_probabilities(short_segment)
        self.assertTrue(np.allclose(probs, np.full(self.n_states, 1.0 / self.n_states)),
                        "Should return uniform probabilities for very short segment.")
        
        state = self.detector.predict_state(short_segment)
        # Argmax of uniform probabilities will be 0
        self.assertEqual(state, 0, "State should be 0 for uniform probabilities from short segment.")

    def test_custom_config(self):
        """Test initialization with custom GMM and feature configurations."""
        custom_gmm_config = {'covariance_type': 'tied', 'n_init': 5}
        custom_feature_config = {
            'return_window': 2,
            'vol_window': 25, # Max window here
            'ma_short_window': 7,
            'ma_long_window': 25,
            'atr_window': 10
        }
        detector = GMMMarketStateDetector(
            n_states=4, 
            gmm_config=custom_gmm_config,
            feature_config=custom_feature_config,
            random_state=123
        )
        self.assertEqual(detector.n_states, 4)
        self.assertEqual(detector.gmm_model.n_components, 4)
        self.assertEqual(detector.gmm_model.covariance_type, 'tied')
        self.assertEqual(detector.gmm_model.n_init, 5)
        self.assertEqual(detector.gmm_model.random_state, 123) # Check GMM's random_state
        self.assertEqual(detector.feature_config['vol_window'], 25)
        self.assertEqual(detector.feature_config['atr_window'], 10)

        # Test that _calculate_features respects new window sizes
        # For this custom config:
        # log_return: NaN idx 0.
        # return_mean_2 (of log_return, min_p 1): non-NaN from idx 1. NaN idx 0.
        # volatility_25 (of log_return, min_p 12): non-NaN from idx 12. NaN idx 0-11.
        # price_ma_short_ratio (ma_short_window=7, min_p 3): MA non-NaN from idx 2. Ratio non-NaN from idx 2. NaN idx 0-1.
        # ma_short_long_ratio (sw=7 min_p 3; lw=25 min_p 12): ma_long non-NaN idx 11. Ratio non-NaN idx 11. NaN idx 0-10.
        # atr_10 (min_p 5 for EWM of TR. TR NaN idx 0): non-NaN from idx 5. NaN idx 0-4.
        # Max NaN index for custom config: max(0, 0, 11, 1, 10, 4) = 11.
        # Min input rows for one feature row = 11 + 1 + 1 = 13.
        custom_max_initial_nan_idx = 11
        min_input_for_custom_one_row = custom_max_initial_nan_idx + 1 + 1 # 13 rows
        
        # current_max_window_param = max(custom_feature_config.values()) # 25
        # fitting_data needs to be long enough to produce > n_states (4) feature rows
        # num_feature_rows = num_input - (custom_max_initial_nan_idx + 1)
        # We need num_input - 12 > 4  => num_input > 16.
        # Let's use a safe amount, e.g., 50 + min_input_for_custom_one_row
        fitting_data_rows = min_input_for_custom_one_row + 50 # 13 + 50 = 63 rows
        fitting_data = self.sample_ohlcv_data.head(fitting_data_rows)
        
        features = detector._calculate_features(fitting_data)
        self.assertFalse(features.empty)
        # Expected features: 63 - (11+1) = 51 rows
        self.assertEqual(features.shape[0], fitting_data_rows - (custom_max_initial_nan_idx+1))
        
        detector.fit(fitting_data)
        self.assertTrue(detector.fitted, "Detector should be fitted with custom config and sufficient data.")
        self.assertIsNotNone(detector.gmm_model.weights_, "GMM weights should exist after successful fit.")
        
        # Prediction segment needs to be long enough for the custom config
        # min_input_for_custom_one_row = 13 rows
        pred_segment = fitting_data.tail(min_input_for_custom_one_row + 10) # 13 + 10 = 23 rows
        probs = detector.predict_state_probabilities(pred_segment)
        self.assertEqual(probs.shape, (4,)) # n_states = 4

    def test_save_and_load_fitted_model(self):
        """Test saving a fitted model and then loading it."""
        # Fit the model
        fitting_data = self.sample_ohlcv_data.head(self.min_rows_for_general_testing + 50)
        self.detector.fit(fitting_data)
        self.assertTrue(self.detector.fitted)

        original_probs = self.detector.predict_state_probabilities(self.prediction_segment)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmpfile:
            model_path = tmpfile.name
        
        try:
            self.detector.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))

            loaded_detector = GMMMarketStateDetector.load_model(model_path)

            self.assertIsInstance(loaded_detector, GMMMarketStateDetector)
            self.assertEqual(loaded_detector.n_states, self.detector.n_states)
            self.assertEqual(loaded_detector.random_state, self.detector.random_state)
            self.assertEqual(loaded_detector.gmm_config, self.detector.gmm_config)
            self.assertEqual(loaded_detector.feature_config, self.detector.feature_config)
            self.assertEqual(loaded_detector.fitted, self.detector.fitted)

            self.assertIsNotNone(loaded_detector.scaler)
            self.assertIsNotNone(loaded_detector.gmm_model.means_) # Check if GMM is fitted

            # Compare scaler attributes
            np.testing.assert_array_almost_equal(loaded_detector.scaler.mean_, self.detector.scaler.mean_)
            np.testing.assert_array_almost_equal(loaded_detector.scaler.scale_, self.detector.scaler.scale_)

            # Compare GMM model parameters (weights, means, covariances)
            np.testing.assert_array_almost_equal(loaded_detector.gmm_model.weights_, self.detector.gmm_model.weights_)
            np.testing.assert_array_almost_equal(loaded_detector.gmm_model.means_, self.detector.gmm_model.means_)
            np.testing.assert_array_almost_equal(loaded_detector.gmm_model.covariances_, self.detector.gmm_model.covariances_)

            # Verify prediction consistency
            loaded_probs = loaded_detector.predict_state_probabilities(self.prediction_segment)
            np.testing.assert_array_almost_equal(loaded_probs, original_probs, decimal=6)
            
            loaded_state = loaded_detector.predict_state(self.prediction_segment)
            original_state = self.detector.predict_state(self.prediction_segment)
            self.assertEqual(loaded_state, original_state)

        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_save_unfitted_model_and_load(self):
        """Test saving an unfitted model and then loading it."""
        unfitted_detector = GMMMarketStateDetector(n_states=self.n_states, random_state=11)
        self.assertFalse(unfitted_detector.fitted)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmpfile:
            model_path = tmpfile.name

        try:
            # Expect a warning when saving an unfitted model
            with self.assertWarns(UserWarning) as cm:
                unfitted_detector.save_model(model_path)
            self.assertIn("Attempting to save a model that has not been successfully fitted", str(cm.warning))
            
            self.assertTrue(os.path.exists(model_path))

            loaded_detector = GMMMarketStateDetector.load_model(model_path)

            self.assertIsInstance(loaded_detector, GMMMarketStateDetector)
            self.assertEqual(loaded_detector.n_states, unfitted_detector.n_states)
            self.assertEqual(loaded_detector.random_state, unfitted_detector.random_state)
            self.assertFalse(loaded_detector.fitted)
            self.assertIsNone(loaded_detector.scaler) # Scaler should be None for unfitted model
            # GMM model attributes like means_ won't exist for an unfitted model
            with self.assertRaises(AttributeError): # Or check if it's None if that's the behavior
                _ = loaded_detector.gmm_model.means_


            # Predictions should be uniform for an unfitted model
            probs = loaded_detector.predict_state_probabilities(self.prediction_segment)
            expected_probs = np.full(loaded_detector.n_states, 1.0 / loaded_detector.n_states)
            np.testing.assert_array_almost_equal(probs, expected_probs)

        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_load_model_file_not_found(self):
        """Test loading a model from a non-existent file path."""
        non_existent_path = "non_existent_model.joblib"
        with self.assertWarns(UserWarning) as cm_warn: # The method itself warns
            with self.assertRaises(FileNotFoundError): # joblib.load raises FileNotFoundError
                GMMMarketStateDetector.load_model(non_existent_path)
        self.assertIn(f"Error loading model from {non_existent_path}", str(cm_warn.warning))


    def test_load_invalid_model_file(self):
        """Test loading an invalid (e.g., corrupted or not a joblib) model file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            invalid_model_path = tmpfile.name
            tmpfile.write(b"This is not a valid joblib model file.")

        with self.assertWarns(UserWarning) as cm_warn:
            # joblib.load can raise various errors for invalid files (e.g., ValueError, pickle.UnpicklingError)
            # We catch a general Exception here as joblib's specific error can vary.
            with self.assertRaises(Exception) as cm_exc: 
                GMMMarketStateDetector.load_model(invalid_model_path)
        
        self.assertIn(f"Error loading model from {invalid_model_path}", str(cm_warn.warning))
        # Check that the raised exception is not FileNotFoundError, but something related to unpickling/format
        self.assertNotIsInstance(cm_exc.exception, FileNotFoundError)

        os.remove(invalid_model_path)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
