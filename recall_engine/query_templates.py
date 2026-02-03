from typing import Dict, List

# Template Registry
# Defines all allowable Cypher queries. 
# KEYS are the IDs the LLM must select.

TEMPLATE_REGISTRY = {
    "LATEST_SIGNALS": {
        "description": "Get latest signals (price/sentiment) for a company",
        "parameters": ["symbol"],
        "cypher": """
        MATCH (c:Company {symbol: $symbol})-[:ISSUED]->(s:Stock)-[:HAS_SNAPSHOT]->(sn:Snapshot)
        WITH sn ORDER BY sn.as_of DESC LIMIT 1
        MATCH (sn)-[:HAS_SIGNAL]->(sig:Signal)
        RETURN sn.as_of as Date, sig.name as SignalName, sig.value as Value, sig.direction as Direction
        """
    },
    "COMPARE_TREND": {
        "description": "Compare 'trend' signal across multiple companies",
        "parameters": ["symbols"],
        "cypher": """
        MATCH (c:Company)-[:ISSUED]->(s:Stock)-[:HAS_SNAPSHOT]->(sn:Snapshot)-[:HAS_SIGNAL]->(sig:Signal {name: "trend"})
        WHERE c.symbol IN $symbols
        WITH c, sn, sig ORDER BY sn.as_of DESC
        WITH c, COLLECT(sig)[0] as latestSig, COLLECT(sn)[0] as latestSn
        RETURN c.symbol as Symbol, latestSn.as_of as Date, latestSig.value as Trend
        """
    },
    "RISK_ANALYSIS": {
        "description": "Get specific downside_risk signal for a company",
        "parameters": ["symbol"],
        "cypher": """
        MATCH (c:Company {symbol: $symbol})-[:ISSUED]->(s:Stock)-[:HAS_SNAPSHOT]->(sn:Snapshot)-[:HAS_SIGNAL]->(sig:Signal {name: "downside_risk"})
        WITH sn, sig ORDER BY sn.as_of DESC LIMIT 1
        RETURN sn.as_of as Date, sig.value as DownsideRisk
        """
    },
    "COMPANY_LOOKUP": {
        "description": "List all tracked companies",
        "parameters": [],
        "cypher": """
        MATCH (c:Company)
        RETURN c.symbol as Symbol
        ORDER BY c.symbol
        """
    }
}
