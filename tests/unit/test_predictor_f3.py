import sys
import os
import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from intelligence.price.predictor import predict_future, TIME_STEP

class TestPredictorF3(unittest.TestCase):
    @patch('intelligence.price.predictor.load_model')
    @patch('intelligence.price.predictor.MinMaxScaler')
    def test_single_step_prediction(self, mock_scaler, mock_load_model):
        """Verify predict_future calls model.predict exactly once."""
        
        # Mock dependencies
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        # Mock Scaler
        scaler_instance = MagicMock()
        mock_scaler.return_value = scaler_instance
        scaler_instance.fit_transform.return_value = np.zeros((100, 1)) # Scalar dummy data
        scaler_instance.inverse_transform.return_value = [[150.0]] # Mock prediction result
        
        # Mock Model Output
        mock_model.predict.return_value = np.array([[0.5]]) 
        
        # Mock Data
        dates = pd.date_range("2024-01-01", periods=100)
        df = pd.DataFrame({'Close': np.random.rand(100) * 100}, index=dates)
        
        # Call Predictor
        days_to_predict = 30
        include_last_n = 14
        final_dates, final_prices = predict_future(mock_model, df, scaler_instance, days_to_predict, include_last_n)
        
        # ASSERTIONS
        
        # 1. Model predict called EXACTLY ONCE
        mock_model.predict.assert_called_once()
        
        # 2. Output length correct (last 14 real + 30 future = 44)
        expected_len = 14 + days_to_predict
        self.assertEqual(len(final_prices), expected_len)
        
        # 3. Flat projection check
        # The last 30 prices should all be identical (150.0)
        future_part = final_prices[-30:]
        self.assertTrue(np.all(future_part == 150.0), "Future prices should be projected flat")

if __name__ == '__main__':
    unittest.main()
