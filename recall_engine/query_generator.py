from typing import Dict, Any, List, Tuple
from recall_engine.llm_query_generator import LLMQueryGenerator

class QueryGenerator:
    def __init__(self, model):
        self.db_schema = self.generate_db_schema(self)
        self.llm_generator = LLMQueryGenerator(model, self.db_schema)

    def generate_and_validate_query(self, intent: Dict[str, Any], entities: Dict[str, List[str]]) -> Tuple[str, bool, str]:
        return self.llm_generator.generate_and_validate_query(intent, entities)

    @staticmethod
    def generate_db_schema(self) -> str:
        return """
            {
        Stock: {
            type: "node",
            labels: ["Stock"],
            properties: {
                symbol: { type: "STRING", unique: true }
            },
            relationships: {
                HAS_SNAPSHOT: { direction: "out", labels: ["Snapshot"] },
                ISSUED: { direction: "in", labels: ["Company"] }
            }
        },
        Company: {
            type: "node",
            labels: ["Company"],
            properties: {
                symbol: { type: "STRING", unique: true }
            },
            relationships: {
                ISSUED: { direction: "out", labels: ["Stock"] }
            }
        },
        Snapshot: {
            type: "node",
            labels: ["Snapshot"],
            properties: {
                symbol: { type: "STRING", unique: true },
                as_of: { type: "STRING", unique: true }
            },
            relationships: {
                HAS_SIGNAL: { direction: "out", labels: ["Signal"] },
                HAS_SNAPSHOT: { direction: "in", labels: ["Stock"] }
            }
        },
        Signal: {
            type: "node",
            labels: ["Signal"],
            properties: {
                name: { type: "STRING" },
                value: { type: "FLOAT" },
                direction: { type: "STRING" },
                as_of: { type: "STRING" }
            },
            relationships: {
                HAS_SIGNAL: { direction: "in", labels: ["Snapshot"] },
                OF_ASPECT: { direction: "out", labels: ["Aspect"] }
            }
        },
        Aspect: {
            type: "node",
            labels: ["Aspect"],
            properties: {
                name: { type: "STRING", unique: true }
            },
            relationships: {
                OF_ASPECT: { direction: "in", labels: ["Signal"] }
            }
        }
        }
    """
    def extract_parameters_from_query(self, query: str) -> List[str]:
        return self.llm_generator.extract_parameters_from_query(query)