import sys
import os
import unittest
from unittest.mock import MagicMock, call

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from kg.kg_schema import NODE_COMPANY, REL_ISSUED
from kg.kg_writer import KGWriter
from kg.state_adapter import State, PriceState, SentimentState
from recall_engine.query_generator import QueryGenerator

class TestSchemaUnification(unittest.TestCase):
    def test_writer_creates_company(self):
        """Verify KGWriter merges Company node and links to Stock."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_tx = MagicMock()
        
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.begin_transaction.return_value.__enter__.return_value = mock_tx
        
        writer = KGWriter(mock_driver)
        state = State(
            symbol="TEST", 
            as_of="2024-01-01", 
            price=PriceState("Bullish", 0.1, 0.05),
            sentiment=SentimentState(0.8, "Positive", {})
        )
        
        writer.write_snapshot(state)
        
        # Check if we created the Company node
        found_company_merge = False
        for call_args in mock_tx.run.call_args_list:
            query = call_args[0][0]
            if f"MERGE (c:{NODE_COMPANY}" in query and f"MERGE (c)-[:{REL_ISSUED}]->(s)" in query:
                found_company_merge = True
                break
        
        self.assertTrue(found_company_merge, "KGWriter did not merge Company node or link it to Stock")

    def test_reader_schema_exposure(self):
        """Verify QueryGenerator exposes the correct Stock/Signal schema."""
        mock_model = MagicMock()
        generator = QueryGenerator(mock_model)
        
        schema = generator.db_schema
        
        self.assertIn("Stock", schema)
        self.assertIn("Signal", schema)
        self.assertIn("Company", schema)
        self.assertIn("ISSUED", schema)
        self.assertNotIn("MetricValue", schema, "Old hallucinated schema elements found")

if __name__ == '__main__':
    unittest.main()
