import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from recall_engine.database_manager import DatabaseManager
from recall_engine.llm_query_generator import LLMQueryGenerator

class TestSecurityF2(unittest.TestCase):
    def test_database_manager_blocks_unsafe(self):
        """Verify DatabaseManager blocks queries with unsafe keywords."""
        mock_driver = MagicMock()
        db = DatabaseManager(mock_driver)
        
        unsafe_queries = [
            "MATCH (n) DELETE n",
            "CREATE (n:Person {name: 'Admin'})",
            "MATCH (n) SET n.prop = 1",
            "DROP INDEX index_name"
        ]
        
        for q in unsafe_queries:
            result = db.execute_query(q)
            self.assertEqual(result, [], f"DatabaseManager failed to block: {q}")
            # Ensure session.run was NOT called
            mock_driver.session.assert_not_called()

    @patch('recall_engine.llm_query_generator.chatbot_no_context')
    def test_llm_generator_uses_template(self, mock_chatbot):
        """Verify LLM generator maps intent to a known template."""
        mock_model = MagicMock()
        # Mock LLM responding with a valid template ID
        mock_chatbot.return_value = '{"template_id": "LATEST_SIGNALS", "parameters": {"symbol": "AAPL"}}'
        
        generator = LLMQueryGenerator(mock_model)
        query, valid, msg = generator.generate_and_validate_query({}, {})
        
        self.assertTrue(valid)
        self.assertIn("MATCH (c:Company {symbol: 'AAPL'})", query, "Template parameters were not bound correctly")
        self.assertIn("sig:Signal", query, "Correct template logic missing")

    @patch('recall_engine.llm_query_generator.chatbot_no_context')
    def test_llm_generator_fails_unknown_id(self, mock_chatbot):
        """Verify LLM generator fails safely on hallucinated ID."""
        mock_model = MagicMock()
        mock_chatbot.return_value = '{"template_id": "HAL_9000_KILL_ALL", "parameters": {}}'
        
        generator = LLMQueryGenerator(mock_model)
        query, valid, msg = generator.generate_and_validate_query({}, {})
        
        self.assertFalse(valid)
        self.assertIn("Unknown Template ID", msg)
        self.assertEqual(query, "")

if __name__ == '__main__':
    unittest.main()
