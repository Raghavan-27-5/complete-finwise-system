# Node Labels
NODE_STOCK = "Stock"
NODE_SNAPSHOT = "Snapshot"
NODE_SIGNAL = "Signal"
NODE_ASPECT = "Aspect"
NODE_COMPANY = "Company"

# Relationship Types
REL_HAS_SNAPSHOT = "HAS_SNAPSHOT"
REL_HAS_SIGNAL = "HAS_SIGNAL"
REL_OF_ASPECT = "OF_ASPECT"
REL_ISSUED = "ISSUED"

# Node Properties
STOCK_PROPERTIES = {
    "symbol": "string",
}

SNAPSHOT_PROPERTIES = {
    "symbol": "string",
    "as_of": "string",
}

SIGNAL_PROPERTIES = {
    "name": "string",
    "value": "float or string",
    "direction": "string or null",
}

ASPECT_PROPERTIES = {
    "name": "string",
}

# Constraint Definitions
CONSTRAINTS = [
    "CREATE CONSTRAINT stock_symbol_unique IF NOT EXISTS FOR (s:Stock) REQUIRE s.symbol IS UNIQUE",
    "CREATE CONSTRAINT company_symbol_unique IF NOT EXISTS FOR (c:Company) REQUIRE c.symbol IS UNIQUE",
    "CREATE CONSTRAINT aspect_name_unique IF NOT EXISTS FOR (a:Aspect) REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT snapshot_symbol_as_of_unique IF NOT EXISTS FOR (sn:Snapshot) REQUIRE (sn.symbol, sn.as_of) IS UNIQUE",
]
