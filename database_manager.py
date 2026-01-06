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

    def clear_database(self):
        query = "MATCH (n) DETACH DELETE n"
        self.execute_query(query)

    def add_company(self, company_name: str):
        query = "MERGE (c:Company {name: $name})"
        self.execute_query(query, {"name": company_name})

    def add_metric(self, company_name: str, metric_name: str, value: Any):
        query = """
        MATCH (c:Company {name: $company_name})
        MERGE (m:Metric {name: $metric_name})
        MERGE (mv:MetricValue {value: $value})
        MERGE (c)-[:HAS_METRIC]->(mv)-[:OF_METRIC]->(m)
        """
        self.execute_query(query, {"company_name": company_name, "metric_name": metric_name, "value": value})

    def populate_sample_data(self):
        companies = ["TCS", "Infosys", "Wipro", "HCL Technologies"]
        metrics = [
            ("Revenue", [22917, 17944, 10136, 11787]),
            ("Net Income", [4633, 3442, 1889, 1938]),
            ("EPS", [12.77, 8.31, 3.55, 7.10])
        ]
        
        for company in companies:
            self.add_company(company)
            for metric_name, values in metrics:
                value = values[companies.index(company)]
                self.add_metric(company, metric_name, value)

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