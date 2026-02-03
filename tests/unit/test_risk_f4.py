import sys
import os
import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from intelligence.price.predictor import monte_carlo_heston, monte_carlo_heston_stats_only

class TestRiskF4(unittest.TestCase):
    
    @patch('intelligence.price.predictor.safe_download')
    def test_monte_carlo_gbm_output(self, mock_download):
        """Verify GBM simulation returns correct structure and labels."""
        
        # Mock Data (Flat)
        df = pd.DataFrame({'Close': [100.0] * 61})
        mock_download.return_value = df
        
        # Call function
        fig, stats = monte_carlo_heston("AAPL", 100.0, 105.0, days=14, n_simulations=100)
        
        # 1. Verify No Heston Labels
        self.assertIn("Volatility Cone Simulation (GBM)", fig.layout.title.text)
        self.assertNotIn("Heston", fig.layout.title.text)
        
        # 2. Verify Stats String
        self.assertIn("Simulated final day", stats)
        
        # 3. Verify Bounds
        trace_names = [t.name for t in fig.data]
        self.assertIn("Upper Vol Cone (97.5%)", trace_names)
        self.assertIn("Mean Projection", trace_names)

    @patch('intelligence.price.predictor.safe_download')
    def test_risk_score_validity(self, mock_download):
        """Verify risk score is normalized 0-1."""
        df = pd.DataFrame({'Close': [100.0] * 61})
        mock_download.return_value = df
        
        score = monte_carlo_heston_stats_only("AAPL", 100.0, 90.0, days=30)
        
        self.assertTrue(0.0 <= score <= 1.0, f"Risk score {score} out of bounds")
        self.assertIsInstance(score, float)

if __name__ == '__main__':
    unittest.main()
