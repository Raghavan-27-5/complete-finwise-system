from typing import Dict, Any, List, Tuple
from recall_engine.chatbot import chatbot_no_context
import json
import logging
import re

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LLMQueryGenerator:
    def __init__(self, model, db_schema: str): 
        self.model = model 
        self.db_schema = db_schema

    def generate_and_validate_query(self, intent: Dict[str, Any], entities: Dict[str, List[str]]) -> Tuple[str, bool, str]:
        query = self.generate_query(intent, entities)
        is_valid, explanation = self.validate_query(query)
        return query, is_valid, explanation

    def generate_query(self, intent: Dict[str, Any], entities: Dict[str, List[str]]) -> str:
        prompt = self._create_prompt(intent, entities)
        logger.debug(f"Generated Prompt for Query: {prompt}")
        response = chatbot_no_context(prompt, self.model)
        logger.debug(f"Chatbot Response for Query: {response}")
        return self._extract_query(response)

    def validate_query(self, query: str) -> Tuple[bool, str]:
        prompt = f"""
        Validate the following Cypher query for Neo4j:

        {query}

        Check for the following:
        1. Syntax errors
        2. Potential security issues (e.g., injection vulnerabilities)
        3. Performance concerns (e.g., unbounded queries)
        4. Correctness in relation to the Neo4j database structure
        5. Proper use of parameters

        Database schema:
        {self.db_schema}"""+\
        """
        Return a JSON object with the following fields:
        1. "is_valid": boolean indicating whether the query is valid and safe to run
        2. "explanation": string explaining the validation result, including any issues found
        3. "suggested_fix": string containing a suggested fix if the query is not valid

        Example response:
        {{
            "is_valid": false,
            "explanation": "The query lacks proper parameter usage for the 'companyName', which could lead to injection vulnerabilities.",
            "suggested_fix": "MATCH (c:Company {{name: $companyName}}) RETURN c"
        }}
        """
        logger.debug(f"Validation Prompt: {prompt}")
        response = chatbot_no_context(prompt, self.model)
        logger.debug(f"Chatbot Response for Validation: {response}")

        try:
            result = json.loads(response)
            is_valid = result.get("is_valid", False)
            explanation = result.get("explanation", "No explanation provided.")
            suggested_fix = result.get("suggested_fix", "")
            if not is_valid and suggested_fix:
                explanation += f"\nSuggested fix: {suggested_fix}"
            return is_valid, explanation
        except json.JSONDecodeError:
            logger.error("Failed to parse validation response as JSON.")
            return False, "Error parsing validation result."

    def _create_prompt(self, intent: Dict[str, Any], entities: Dict[str, List[str]]) -> str:
        prompt = f"""
        Generate a parameterized Cypher query for a Neo4j database based on the following intent and entities:

        Intent: {json.dumps(intent)}
        Entities: {json.dumps(entities)}

        Database schema:
        {self.db_schema}

        Guidelines:
        1. Use only parameters present in the Entities section.
        2. Ensure the query is efficient and follows Neo4j best practices.
        3. Use appropriate indexes and constraints where applicable.
        4. Handle potential null values and empty lists in parameters.
        5. Limit results when appropriate to prevent performance issues.
        6. Use CASE statements for complex conditional logic.
        7. Utilize appropriate aggregation functions when dealing with multiple records."""+\
        """
        Example queries:

        1. Retrieve the latest quarterly report for a company:
        MATCH (c:Company {{name: $companyName}})-[:HAS_REPORT]->(r:Report {{type: "quarterly"}})
        WITH c, r ORDER BY r.date DESC LIMIT 1
        RETURN c.name AS Company, r.date AS ReportDate, r.content AS ReportContent

        2. Compare multiple metrics for several companies:
        MATCH (c:Company)
        WHERE c.name IN $companyNames
        OPTIONAL MATCH (c)-[:HAS_METRIC]->(mv:MetricValue)-[:OF_METRIC]->(m:Metric)
        WHERE m.name IN $metricNames
        WITH c, m, mv ORDER BY mv.date DESC
        WITH c, m, COLLECT(mv)[0] AS latestValue
        RETURN c.name AS Company, 
               m.name AS Metric, 
               latestValue.value AS Value, 
               latestValue.date AS Date

        3. Analyze trend of a specific metric for a company over time:
        MATCH (c:Company {{name: $companyName}})-[:HAS_METRIC]->(mv:MetricValue)-[:OF_METRIC]->(m:Metric {{name: $metricName}})
        WHERE mv.date >= $startDate AND mv.date <= $endDate
        WITH c, m, mv ORDER BY mv.date
        RETURN c.name AS Company, 
               m.name AS Metric, 
               COLLECT({{date: mv.date, value: mv.value}}) AS Trend

        4. Find top N companies by a specific metric:
        MATCH (c:Company)-[:HAS_METRIC]->(mv:MetricValue)-[:OF_METRIC]->(m:Metric {{name: $metricName}})
        WITH c, mv WHERE mv.date = $date
        ORDER BY mv.value DESC
        LIMIT $limit
        RETURN c.name AS Company, mv.value AS MetricValue

        5. Compare companies within the same industry:
        MATCH (c:Company)
        WHERE c.industry = $industry
        OPTIONAL MATCH (c)-[:HAS_METRIC]->(mv:MetricValue)-[:OF_METRIC]->(m:Metric)
        WHERE m.name IN $metricNames
        WITH c, m, mv ORDER BY mv.date DESC
        WITH c, m, COLLECT(mv)[0] AS latestValue
        RETURN c.name AS Company, 
               c.industry AS Industry,
               m.name AS Metric, 
               latestValue.value AS Value

        Generate a parameterized Cypher query that retrieves the relevant information based on the intent and entities.
        Ensure the query is efficient, follows Neo4j best practices, and is safe from injection vulnerabilities.
        Return only the Cypher query without any explanation or additional text.
        """
        logger.debug(f"Created Prompt: {prompt}")
        return prompt

    def _extract_query(self, response: str) -> str:
        query = response.strip()
        logger.debug(f"Extracted Query: {query}")
        return query
    
    def extract_parameters_from_query(self, query: str) -> List[str]:
        pattern = r'\$([a-zA-Z_][a-zA-Z0-9_]*)'
        parameters = re.findall(pattern, query)
        logger.debug(f"Extracted Parameters: {parameters}")
        return parameters