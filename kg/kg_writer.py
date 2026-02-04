from kg.state_adapter import State, PriceState, SentimentState
from kg.kg_schema import (
    NODE_STOCK,
    NODE_SNAPSHOT,
    NODE_SIGNAL,
    NODE_ASPECT,
    NODE_COMPANY,
    REL_HAS_SNAPSHOT,
    REL_HAS_SIGNAL,
    REL_OF_ASPECT,
    REL_ISSUED,
)


class KGWriter:
    def __init__(self, driver):
        self._driver = driver

    def write_snapshot(self, state: State):
        with self._driver.session() as session:
            with session.begin_transaction() as tx:
                tx.run(
                    f"MERGE (s:{NODE_STOCK} {{symbol: $symbol}})",
                    symbol=state.symbol,
                )

                tx.run(
                    f"""
                    MERGE (c:{NODE_COMPANY} {{symbol: $symbol}})
                    WITH c
                    MATCH (s:{NODE_STOCK} {{symbol: $symbol}})
                    MERGE (c)-[:{REL_ISSUED}]->(s)
                    """,
                    symbol=state.symbol,
                )

                tx.run(
                    f"""
                    MERGE (sn:{NODE_SNAPSHOT} {{symbol: $symbol, as_of: $as_of}})
                    WITH sn
                    MATCH (s:{NODE_STOCK} {{symbol: $symbol}})
                    MERGE (s)-[:{REL_HAS_SNAPSHOT}]->(sn)
                    """,
                    symbol=state.symbol,
                    as_of=state.as_of,
                )

                if state.price is not None:
                    tx.run(
                        f"""
                        MATCH (sn:{NODE_SNAPSHOT} {{symbol: $symbol, as_of: $as_of}})
                        MERGE (sig:{NODE_SIGNAL} {{symbol: $symbol, as_of: $as_of, name: $name}})
                        SET sig.value = $value, sig.direction = $direction
                        MERGE (sn)-[:{REL_HAS_SIGNAL}]->(sig)
                        """,
                        symbol=state.symbol,
                        as_of=state.as_of,
                        name="trend",
                        value=state.price.trend,
                        direction=None,
                    )

                    tx.run(
                        f"""
                        MATCH (sn:{NODE_SNAPSHOT} {{symbol: $symbol, as_of: $as_of}})
                        MERGE (sig:{NODE_SIGNAL} {{symbol: $symbol, as_of: $as_of, name: $name}})
                        SET sig.value = $value, sig.direction = $direction
                        MERGE (sn)-[:{REL_HAS_SIGNAL}]->(sig)
                        """,
                        symbol=state.symbol,
                        as_of=state.as_of,
                        name="forecast_slope",
                        value=state.price.forecast_slope,
                        direction=None,
                    )

                    tx.run(
                        f"""
                        MATCH (sn:{NODE_SNAPSHOT} {{symbol: $symbol, as_of: $as_of}})
                        MERGE (sig:{NODE_SIGNAL} {{symbol: $symbol, as_of: $as_of, name: $name}})
                        SET sig.value = $value, sig.direction = $direction
                        MERGE (sn)-[:{REL_HAS_SIGNAL}]->(sig)
                        """,
                        symbol=state.symbol,
                        as_of=state.as_of,
                        name="downside_risk",
                        value=state.price.downside_risk,
                        direction=None,
                    )

                if state.sentiment is not None:
                    tx.run(
                        f"""
                        MATCH (sn:{NODE_SNAPSHOT} {{symbol: $symbol, as_of: $as_of}})
                        MERGE (sig:{NODE_SIGNAL} {{symbol: $symbol, as_of: $as_of, name: $name}})
                        SET sig.value = $value, sig.direction = $direction
                        MERGE (sn)-[:{REL_HAS_SIGNAL}]->(sig)
                        """,
                        symbol=state.symbol,
                        as_of=state.as_of,
                        name="global_score",
                        value=state.sentiment.global_score,
                        direction=None,
                    )

                    tx.run(
                        f"""
                        MATCH (sn:{NODE_SNAPSHOT} {{symbol: $symbol, as_of: $as_of}})
                        MERGE (sig:{NODE_SIGNAL} {{symbol: $symbol, as_of: $as_of, name: $name}})
                        SET sig.value = $value, sig.direction = $direction
                        MERGE (sn)-[:{REL_HAS_SIGNAL}]->(sig)
                        """,
                        symbol=state.symbol,
                        as_of=state.as_of,
                        name="label",
                        value=state.sentiment.label,
                        direction=None,
                    )

                    if state.sentiment.aspects is not None:
                        for aspect_name, aspect_value in state.sentiment.aspects.items():
                            tx.run(
                                f"MERGE (a:{NODE_ASPECT} {{name: $aspect_name}})",
                                aspect_name=aspect_name,
                            )

                            # Fix for CypherTypeError: aspect_value is a dict {'label': '...', 'score': ...}
                            # We must extract the numeric score for the 'value' property.
                            real_val = aspect_value
                            if isinstance(aspect_value, dict) and 'score' in aspect_value:
                                real_val = aspect_value['score']
                            
                            tx.run(
                                f"""
                                MATCH (sn:{NODE_SNAPSHOT} {{symbol: $symbol, as_of: $as_of}})
                                MATCH (a:{NODE_ASPECT} {{name: $aspect_name}})
                                MERGE (sig:{NODE_SIGNAL} {{symbol: $symbol, as_of: $as_of, name: $name}})
                                SET sig.value = $value, sig.direction = $direction
                                MERGE (sn)-[:{REL_HAS_SIGNAL}]->(sig)
                                MERGE (sig)-[:{REL_OF_ASPECT}]->(a)
                                """,
                                symbol=state.symbol,
                                as_of=state.as_of,
                                aspect_name=aspect_name,
                                name=aspect_name,
                                value=real_val,
                                direction=None,
                            )

                tx.commit()
