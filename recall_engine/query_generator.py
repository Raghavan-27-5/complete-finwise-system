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
        HAS_REPORT: {
            count: 10,
            properties: {},
            type: "relationship"
        },
        HAS_VALUE: {
            count: 390,
            properties: {},
            type: "relationship"
        },
        MetricValue: {
            count: 390,
            labels: [],
            properties: {
            value: {
                unique: false,
                indexed: false,
                type: "FLOAT",
                existence: false
            },
            date: {
                unique: false,
                indexed: false,
                type: "STRING",
                existence: false
            }
            },
            type: "node",
            relationships: {
            HAS_VALUE: {
                count: 390,
                direction: "in",
                labels: ["Metric"],
                properties: {}
            },
            OF_METRIC: {
                count: 0,
                direction: "out",
                labels: ["Metric"],
                properties: {}
            }
            }
        },
        Metric: {
            count: 6,
            labels: [],
            properties: {
            unit: {
                unique: false,
                indexed: false,
                type: "STRING",
                existence: false
            },
            description: {
                unique: false,
                indexed: false,
                type: "STRING",
                existence: false
            },
            name: {
                unique: true,
                indexed: true,
                type: "STRING",
                existence: false
            }
            },
            type: "node",
            relationships: {
            HAS_VALUE: {
                count: 0,
                direction: "out",
                labels: ["MetricValue"],
                properties: {}
            },
            HAS_METRIC: {
                count: 390,
                direction: "in",
                labels: ["Company"],
                properties: {}
            },
            OF_METRIC: {
                count: 390,
                direction: "in",
                labels: ["MetricValue"],
                properties: {}
            }
            }
        },
        HAS_METRIC: {
            count: 390,
            properties: {},
            type: "relationship"
        },
        Report: {
            count: 10,
            labels: [],
            properties: {
            id: {
                unique: true,
                indexed: true,
                type: "STRING",
                existence: false
            },
            content: {
                unique: false,
                indexed: false,
                type: "STRING",
                existence: false
            },
            date: {
                unique: false,
                indexed: false,
                type: "STRING",
                existence: false
            },
            type: {
                unique: false,
                indexed: false,
                type: "STRING",
                existence: false
            }
            },
            type: "node",
            relationships: {
            HAS_REPORT: {
                count: 10,
                direction: "in",
                labels: ["Company"],
                properties: {}
            }
            }
        },
        Company: {
            count: 5,
            labels: [],
            properties: {
            location: {
                unique: false,
                indexed: false,
                type: "STRING",
                existence: false
            },
            name: {
                unique: true,
                indexed: true,
                type: "STRING",
                existence: false
            },
            industry: {
                unique: false,
                indexed: false,
                type: "STRING",
                existence: false
            },
            revenue: {
                unique: false,
                indexed: false,
                type: "INTEGER",
                existence: false
            },
            employees: {
                unique: false,
                indexed: false,
                type: "INTEGER",
                existence: false
            }
            },
            type: "node",
            relationships: {
            HAS_REPORT: {
                count: 0,
                direction: "out",
                labels: ["Report"],
                properties: {}
            },
            HAS_METRIC: {
                count: 0,
                direction: "out",
                labels: ["Metric"],
                properties: {}
            }
            }
        },
        OF_METRIC: {
            count: 390,
            properties: {},
            type: "relationship"
        }
        }
    """
    def extract_parameters_from_query(self, query: str) -> List[str]:
        return self.llm_generator.extract_parameters_from_query(query)