from typing import List, Dict, Any
from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, driver):
        self.driver = driver

    def execute_query(self, query: str, params: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                logger.debug(f"Neo4j query: {query}")
                logger.debug(f"Neo4j result: {result}")
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            return []

    def get_all_data(self) -> List[Dict[str, Any]]:
        query = """
        MATCH (c:Company)
        OPTIONAL MATCH (c)-[:HAS_METRIC]->(mv:MetricValue)-[:OF_METRIC]->(m:Metric)
        RETURN c.name AS CompanyName, m.name AS MetricName, mv.value AS Value
        """
        return self.execute_query(query)

    def database_is_empty(self) -> bool:
        query = "MATCH (n) RETURN COUNT(n) AS count"
        result = self.execute_query(query)
        return result[0]['count'] == 0 if result else True

    def get_database_stats(self) -> Dict[str, int]:
        queries = {
            "companies": "MATCH (c:Company) RETURN COUNT(c) AS count",
            "metrics": "MATCH (m:Metric) RETURN COUNT(m) AS count",
            "metric_values": "MATCH (mv:MetricValue) RETURN COUNT(mv) AS count"
        }
        stats = {}
        for key, query in queries.items():
            result = self.execute_query(query)
            stats[key] = result[0]['count'] if result else 0
        return stats