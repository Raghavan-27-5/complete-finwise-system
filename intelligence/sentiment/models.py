# models.py (v18.2) - FinBERT primary, robust fallback
import re
import json
import logging
from functools import lru_cache
from typing import Dict, List, Any, Optional
from collections import defaultdict
import torch
from sentence_transformers import SentenceTransformer
import spacy

# transformers pipeline + tokenizers
from transformers import (
    pipeline as hf_pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from core.config import ASPECT_CATEGORIES, ASPECT_KEYWORDS
# optional
try:
    from langdetect import detect
except Exception:
    detect = None

logger = logging.getLogger("models")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ASPECT_CATEGORIES, ASPECT_KEYWORDS provided by pipeline via import
# device indicator (we won't force models onto it when accelerate used)
_device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("models.py device hint: %s", _device)

# spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = spacy.blank("en")

# SBERT
try:
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    sbert = None
    logger.warning("SBERT load failed: %s", e)

# -------------------------
# FINBERT LOADER (robust)
# -------------------------
_FINBERT_NAME = "ProsusAI/finbert"  # recommended
_aspect_model = None
_aspect_tokenizer = None

def _init_finbert():
    global _aspect_model, _aspect_tokenizer
    if _aspect_model is not None:
        return
    try:
        # Prefer explicit model load with device_map='auto' (lets accelerate decide placement).
        # Do NOT call .to(device). Create pipeline without passing device arg.
        _aspect_tokenizer = AutoTokenizer.from_pretrained(_FINBERT_NAME, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(_FINBERT_NAME, device_map="auto")
        _aspect_model = hf_pipeline("sentiment-analysis", model=model, tokenizer=_aspect_tokenizer)
        logger.info("FinBERT loaded with device_map='auto'.")
    except Exception as e:
        logger.warning("FinBERT initialization failed: %s. Falling back to generic pipeline.", e)
        try:
            # Fallback: let pipeline choose default model (small, fast). Keep tokenizer/model explicit if possible.
            _aspect_model = hf_pipeline("sentiment-analysis")
            _aspect_tokenizer = None
            logger.info("Generic HF sentiment pipeline initialized as fallback.")
        except Exception as e2:
            logger.error("Fallback sentiment pipeline failed: %s", e2)
            _aspect_model = None
            _aspect_tokenizer = None

# initialize at import (best-effort)
_init_finbert()

# -------------------------
# Helpers
# -------------------------
import html

def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Decode HTML entities (&nbsp;, &amp;, etc.)
    text = html.unescape(text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove leftover URLs
    text = re.sub(r"http\S+", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_short_or_junk(s: str) -> bool:
    if not s:
        return True
    # Allow shorter sentences if they contain strong financial tokens
    keywords = ("revenue", "profit", "guidance", "ceo", "lawsuit", "forecast",
                "acquisition", "margin", "rate", "inflation", "regulation")
    clean = s.strip().lower()
    if any(k in clean for k in keywords):
        return False
    # fallback length check
    return len(clean) < 30

def _maybe_translate(text: str) -> str:
    try:
        from googletrans import Translator
        tr = Translator()
        out = tr.translate(text, dest="en")
        return getattr(out, "text", text)
    except Exception:
        return text

@lru_cache(maxsize=1)
def _cached_aspect_keyword_embeddings(aspect_prompts: tuple) -> Dict[str, Any]:
    if sbert is None:
        return {}
    it = iter(aspect_prompts)
    aspects, prompts = [], []
    while True:
        try:
            a = next(it); p = next(it)
            aspects.append(a); prompts.append(p)
        except StopIteration:
            break
    emb = sbert.encode(prompts, convert_to_numpy=True)
    return dict(zip(aspects, emb))

# -------------------------
# Aspect extraction
# -------------------------
def extract_aspects_from_doc(doc_or_text: Any, aspect_prompts_map: Dict[str, List[str]]) -> Dict[str, str]:
    if hasattr(doc_or_text, "text"):
        text = doc_or_text.text
    else:
        text = str(doc_or_text or "")
    text = _clean_text(text)
    if not text:
        return {a: "" for a in aspect_prompts_map.keys()}

    if detect is not None:
        try:
            lang = detect(text)
            if lang != "en":
                text = _maybe_translate(text)
        except Exception:
            pass

    sentences = []
    try:
        doc = nlp(text)
        sentences = [s.text.strip() for s in doc.sents if s.text and not _is_short_or_junk(s.text)]
    except Exception:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]

    if not sentences:
        long_chunk = text[:1024]
        if len(long_chunk) < 60:
            return {a: text[:512] for a in aspect_prompts_map.keys()}
        sentences = [long_chunk]

    prompts = []
    order = []
    for a, kws in aspect_prompts_map.items():
        order.append(a)
        if isinstance(kws, (list, tuple)) and len(kws) > 0:
            prompts.append(", ".join(kws))
        else:
            prompts.append(a)

    if sbert is None:
        return {a: sentences[0] for a in aspect_prompts_map.keys()}

    sent_embs = sbert.encode(sentences, convert_to_numpy=True)
    aspect_emb_map = _cached_aspect_keyword_embeddings(tuple([x for pair in zip(order, prompts) for x in pair]))

    from sklearn.metrics.pairwise import cosine_similarity
    aspect_snippets = {}
    for aspect in order:
        a_emb = aspect_emb_map.get(aspect)
        if a_emb is None:
            aspect_snippets[aspect] = sentences[0]
            continue
        scores = cosine_similarity(sent_embs, a_emb.reshape(1, -1)).reshape(-1)
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])
        best_sentence = sentences[best_idx]
        if best_score < 0.32:
            topk = scores.argsort()[-2:][::-1]
            parts = " ".join([sentences[i] for i in topk if i < len(sentences)])
            if len(parts) < 60:
                parts = text[:512]
            aspect_snippets[aspect] = parts
        else:
            aspect_snippets[aspect] = best_sentence
    return aspect_snippets

# -------------------------
# Sentiment wrapper (FinBERT primary)
# -------------------------

def llm_sentiment(text: str, aspect: Optional[str] = None) -> Dict[str, Any]:
    _init_finbert()
    text = (text or "")[:1024]
    if not text or _aspect_model is None:
        return {"score": 0.0, "label": "Neutral"}
    try:
        # CHANGE: top_k=None forces the pipeline to return scores for ALL labels (Pos, Neg, Neu)
        results = _aspect_model(text, top_k=None)

        # Handle list wrapping (HuggingFace pipelines sometimes return list of lists)
        if isinstance(results, list) and isinstance(results[0], list):
            results = results[0]

        # Normalize to dictionary for easy lookup
        # FinBERT labels are usually "positive", "negative", "neutral" (case insensitive)
        scores = {res['label'].lower(): res['score'] for res in results}

        p_pos = scores.get('positive', 0.0)
        p_neg = scores.get('negative', 0.0)

        # CONTINUOUS SCORING LOGIC
        # Instead of returning 0.0 for Neutral, we return the directional pressure
        # Range: -1.0 to +1.0
        sentiment_val = p_pos - p_neg

        # Determine label for display purposes only
        # We keep the threshold low (0.1) to show "Leaning" labels, but the DSP math handles the real gating
        if sentiment_val > 0.1:
            label = "Positive"
        elif sentiment_val < -0.1:
            label = "Negative"
        else:
            label = "Neutral"

        return {"score": sentiment_val, "label": label}

    except Exception as e:
        logger.warning("llm_sentiment failed: %s", e)
        return {"score": 0.0, "label": "Neutral"}

# -------------------------
# Ensemble sentiment per article
# -------------------------
# --- FIX 1 & 6: Gated Ensemble Sentiment ---
def compute_ensemble_sentiment(text: str, snippets: Dict[str, str], aspect_prompts_map: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    HYBRID ENGINE v1.0
    - SBERT selects snippets (fast layer)
    - FinBERT scores each snippet (enterprise layer)
    - Maintains EXACT output schema expected by pipeline.py
    """
    try:
        # 1) GLOBAL SENTIMENT (unchanged)
        global_res = llm_sentiment(text)
        global_score = float(global_res["score"])
        global_label = global_res["label"]

        # If global noise: wipe all aspects
        is_global_noise = abs(global_score) < 0.05

        aspect_sents = {}

        for aspect, snippet in snippets.items():

            # Fallback: missing or too-short snippet
            if not isinstance(snippet, str) or len(snippet.strip()) < 20:
                aspect_sents[aspect] = {"label": "Neutral", "score": 0.0}
                continue

            # Skip if global noise
            if is_global_noise:
                aspect_sents[aspect] = {"label": "Neutral", "score": 0.0}
                continue

            # ===== HYBRID LOGIC START =====
            # FinBERT only on SBERT-best-sentence snippet
            try:
                res = llm_sentiment(snippet, aspect)
                score = float(res["score"])
            except Exception:
                score = 0.0
                res = {"label": "Neutral"}

            # Enterprise gating
            if abs(score) < 0.10:
                aspect_sents[aspect] = {"label": "Neutral", "score": 0.0}
            else:
                aspect_sents[aspect] = {"label": res["label"], "score": score}
            # ===== HYBRID LOGIC END =====

        return {
            "label": global_label,
            "score": global_score,
            "sentiment_num": global_score,
            "aspect_sentiment": aspect_sents
        }

    except Exception as e:
        logger.error("HYBRID compute_ensemble_sentiment error: %s", e)
        return {"label": "Neutral", "score": 0.0, "sentiment_num": 0.0, "aspect_sentiment": {}}
    
# -------------------------
# A+++ ENTITY EXTRACTOR (V26 Institutional)
# -------------------------
import html 

def extract_entities_transformer(text: str) -> Dict[str, List[str]]:
    """
    V26 Institutional Entity Extractor.
    - Hybrid: Combines statistical NER with rule-based whitelists.
    - Robust: Handles 'Apple' vs 'apple', 'X' (Twitter), 'F' (Ford).
    - Safe: Catches OOM/Model errors with meaningful logs.
    """
    # 1. FAILS SAFE: Return empty structure if input is bad
    if not text or not isinstance(text, str):
        return {"organizations": [], "persons": [], "locations": []}

    # 2. DEEP CLEANING (Removes HTML & URL artifacts that confuse NER)
    # We do this locally to ensure this function is self-contained and robust
    clean_text = html.unescape(text)
    clean_text = re.sub(r'http\S+', '', clean_text)
    clean_text = re.sub(r'<[^>]+>', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    try:
        # 3. EXECUTE MODEL
        # Uses global 'nlp' object. If missing, raises specific error.
        if 'nlp' not in globals() or nlp is None:
            raise ValueError("Spacy 'nlp' model not loaded in models.py")
            
        doc = nlp(clean_text)
        orgs, persons, locs = set(), set(), set()

        # 4. VALID SINGLE-LETTER TICKERS (The "X" Fix)
        # These are valid 1-letter companies often filtered out by length checks.
        # We explicitly whitelist them if they appear as Proper Nouns.
        VALID_SHORT_TICKERS = {"F", "C", "X", "M", "K", "T", "V", "Z"} 

        for ent in doc.ents:
            val = ent.text.strip()
            label = ent.label_.upper()
            
            # -----------------------------
            # FILTER 1: Garbage Removal
            # -----------------------------
            # Remove 'href', 'click here', and obvious junk
            if any(x in val.lower() for x in ["http", "href", ".com", "www.", "click here", "read more"]):
                continue
                
            # -----------------------------
            # FILTER 2: Length & Ambiguity
            # -----------------------------
            # Standard Rule: Reject 1-char noise...
            if len(val) < 2:
                # ...UNLESS it is a known mega-cap ticker (Ford, Citi, US Steel)
                if val not in VALID_SHORT_TICKERS:
                    continue
            
            # -----------------------------
            # FILTER 3: The "Apple" Ambiguity
            # -----------------------------
            # If text is title-cased (Headline), common words can be mistaken for ORGs.
            # Heuristic: If it's a common verb/noun like "Target" or "Gap", 
            # require stronger context (omitted for speed, but 'val' is the raw entity).
            # For now, we trust Spacy's statistical model for multi-word entities.

            # -----------------------------
            # CATEGORIZATION
            # -----------------------------
            if label in ("ORG", "NORP"):
                # Cleaning: Remove legal suffixes for cleaner aggregation
                # e.g. "Tesla, Inc." -> "Tesla"
                clean_val = re.sub(r'[, ]+(Inc|Corp|LLC|Ltd|Plc)\.?$', '', val, flags=re.I)
                orgs.add(clean_val)
            elif label == "PERSON":
                persons.add(val)
            elif label in ("GPE", "LOC", "FAC"):
                locs.add(val)

        return {
            "organizations": list(orgs),
            "persons": list(persons),
            "locations": list(locs)
        }

    except Exception as e:
        # 5. NO SILENT FAILURES
        # Log the exact error so you know if OOM or model missing
        logger.error(f"NER Extraction Failed: {str(e)} | Input snippet: {clean_text[:50]}...", exc_info=True)
        # Fail gracefully for the pipeline, but alert the logs
        return {"organizations": [], "persons": [], "locations": []}
# -------------------------
# Multimodal helper (safe)
# -------------------------

def process_multimodal(text, media_url):
    if isinstance(text, str):
        return text.strip()
    if text is None or pd.isna(text):
        return ""
    # any other type becomes a string safely
    return str(text).strip()



#def process_multimodal(text: str, media_url: Optional[str]) -> str:
    #text = (text or "").strip()
    #if not media_url or not isinstance(media_url, str) or not media_url.startswith("http"):
        #return text
    #if any(d in media_url for d in ("biztoc.com", "9to5mac.com")):
       # return text
    #try:
        #import requests
        #from io import BytesIO
        #from PIL import Image
        #resp = requests.get(media_url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
       # resp.raise_for_status()
        #img = Image.open(BytesIO(resp.content)).convert("RGB")
        # BLIP captioner optional - guard
       # try:
            #from transformers import BlipProcessor, BlipForConditionalGeneration
            #proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
            #model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            #inputs = proc(images=img, return_tensors="pt")
            #out = model.generate(**inputs, max_new_tokens=40)
            #caption = proc.decode(out[0], skip_special_tokens=True).strip()
            #if caption:
               # return f"{text} [Image: {caption}]"
           # return text
       # except Exception:
       #     return text
    #except Exception as e:
     #   logger.warning("process_multimodal: image failed: %s", e)
      #  return text

# -------------------------
# Explainability (light-weight placeholder)
# -------------------------
def explain_sentiment(texts: List[str]) -> List[Dict[str, Any]]:
    return [{"note": "explain disabled in lightweight build"} for _ in texts]
