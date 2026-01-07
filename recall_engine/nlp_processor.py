from typing import Dict, Any, Tuple, List
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_md")

def extract_entities_and_intent(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    doc = nlp(text)
    
    entities = {
        "companies": [],
        "metrics": [],
        "limit": [],
        "startDate": [],
        "endDate": [],
        "industry": None
    }
    
    intent = {
        "action": "unknown",
        "comparison": False,
        "trend": False,
        "timeframe": "current"
    }
    
    # Extract company entities
    entities["companies"] = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]]
    
    # Matcher for financial metrics
    matcher = Matcher(nlp.vocab)
    financial_patterns = [
        [{"LOWER": {"IN": ["revenue", "profit", "income", "earnings", "ebitda", "sales", "assets", "liabilities", "equity"]}},
         {"POS": "ADP", "OP": "?"},
         {"POS": "NUM", "OP": "?"}],
        [{"LOWER": "net"}, {"LOWER": {"IN": ["income", "profit", "loss"]}}],
        [{"LOWER": "gross"}, {"LOWER": "margin"}],
        [{"LOWER": "operating"}, {"LOWER": {"IN": ["income", "profit", "margin"]}}],
        [{"LOWER": "return"}, {"LOWER": "on"}, {"LOWER": {"IN": ["assets", "equity", "investment"]}}],
        [{"LOWER": "earnings"}, {"LOWER": "per"}, {"LOWER": "share"}],
        [{"LOWER": "price"}, {"LOWER": "to"}, {"LOWER": "earnings"}, {"LOWER": "ratio"}]
    ]
    matcher.add("FINANCIAL_METRIC", financial_patterns)
    matches = matcher(doc)
    entities["metrics"] = [doc[start:end].text for _, start, end in matches]

    # Extract date entities
    for ent in doc.ents:
        if ent.label_ == "DATE":
            if "start" in ent.text.lower() or "from" in ent.text.lower():
                entities["startDate"].append(ent.text)
            elif "end" in ent.text.lower() or "to" in ent.text.lower():
                entities["endDate"].append(ent.text)
            elif not entities["startDate"]:
                entities["startDate"].append(ent.text)
            else:
                entities["endDate"].append(ent.text)
    
    # Extract limit (number)
    for token in doc:
        if token.like_num:
            prev_token = doc[token.i - 1] if token.i > 0 else None
            if prev_token and prev_token.lower_ in ["top", "limit", "fetch", "get"]:
                entities["limit"].append(token.text)
    
    # Extract industry
    industries = ["IT Services", "Conglomerate", "Banking", "Telecommunications", "Technology", "Healthcare", "Energy", "Retail"]
    for token in doc:
        if token.text in industries:
            entities["industry"] = token.text
            break
    
    # Analyze intent
    intent.update(analyze_query_intent(doc))
    
    logger.debug(f"Extracted entities: {entities}")
    logger.debug(f"Analyzed intent: {intent}")
    
    return entities, intent

def analyze_query_intent(doc: spacy.tokens.Doc) -> Dict[str, Any]:
    intent = {
        "action": "unknown",
        "comparison": False,
        "trend": False,
        "timeframe": "current"
    }
    
    action_verbs = {
        "compare": "compare",
        "versus": "compare",
        "vs": "compare",
        "trend": "trend",
        "change": "trend",
        "grow": "trend",
        "decline": "trend",
        "show": "display",
        "display": "display",
        "give": "display",
        "provide": "display",
        "fetch": "display",
        "get": "display",
        "list": "display",
        "rank": "rank",
        "top": "rank",
        "analyze": "analyze",
        "predict": "predict",
        "forecast": "predict"
    }
    
    for token in doc:
        if token.lemma_ in action_verbs:
            intent["action"] = action_verbs[token.lemma_]
            if intent["action"] == "compare":
                intent["comparison"] = True
            elif intent["action"] == "trend":
                intent["trend"] = True
            break
    
    # Determine timeframe
    time_indicators = {"past": ["previous", "last", "past"],
                       "future": ["next", "upcoming", "future", "forecast"],
                       "current": ["current", "present", "now"]}
    
    for token in doc:
        for timeframe, indicators in time_indicators.items():
            if token.lower_ in indicators:
                intent["timeframe"] = timeframe
                break
        if intent["timeframe"] != "current":
            break
    
    return intent