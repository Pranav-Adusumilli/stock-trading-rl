# src/sentiment_fetcher.py
"""
Sentiment fetcher supporting:
 - FinBERT (transformers) if installed
 - VADER (nltk) fallback
 - NewsAPI headlines fetching (optional, requires API key)
Provides: get_sentiment_series(ticker, dates_index) -> pd.Series
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests

# Optional: NewsAPI key (set to "" to disable)
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")  # or set here

# Try FinBERT model (HuggingFace). If not available, fallback to VADER.
FINBERT_AVAILABLE = False
VADER_AVAILABLE = False
_finbert_tokenizer = None
_finbert_model = None
_vader = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    # Model: 'yiyanghkust/finbert-tone' or 'ProsusAI/finbert'
    _finbert_name = os.environ.get("FINBERT_MODEL", "yiyanghkust/finbert-tone")
    _finbert_tokenizer = AutoTokenizer.from_pretrained(_finbert_name)
    _finbert_model = AutoModelForSequenceClassification.from_pretrained(_finbert_name)
    _finbert_model.eval()
    FINBERT_AVAILABLE = True
except Exception:
    FINBERT_AVAILABLE = False

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    # ensure lexicon downloaded externally or via download_vader.py
    _vader = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False


def _fetch_headlines_newsapi(ticker, start_date, end_date):
    """Fetch headlines using NewsAPI (returns list of (YYYY-MM-DD, title))."""
    if not NEWSAPI_KEY:
        return []
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={ticker}&from={start_date}&to={end_date}&language=en&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
    )
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if data.get("status") != "ok":
            return []
        out = []
        for a in data.get("articles", []):
            dt = a.get("publishedAt", None)
            if dt:
                dt = dt.split("T")[0]
            title = a.get("title", "")
            if dt and title:
                out.append((dt, title))
        return out
    except Exception:
        return []


def _score_finbert_batch(texts):
    """
    Score a list of texts using FinBERT model.
    Returns list of floats (neg/neu/pos -> map to compound in [-1,1], use (pos-neg)).
    """
    if not FINBERT_AVAILABLE:
        return [0.0] * len(texts)
    # batching
    enc = _finbert_tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        out = _finbert_model(**enc)
        logits = out.logits.numpy()
        # Many FinBERT tone models have 3 labels: negative, neutral, positive
        # We'll compute (pos - neg) as compound-like score in [-1,1]
        from scipy.special import softmax
        probs = softmax(logits, axis=1)
        neg = probs[:, 0]
        pos = probs[:, -1]
        scores = (pos - neg).tolist()
    return scores


def _score_vader(headline):
    if not VADER_AVAILABLE:
        return 0.0
    return float(_vader.polarity_scores(headline)["compound"])


def _aggregate_daily_sentiment(headlines, scorer="finbert"):
    """
    headlines: list of (YYYY-MM-DD, headline)
    scorer: 'finbert' or 'vader' or 'hybrid'
    returns dict {YYYY-MM-DD: avg_score}
    """
    day_map = {}
    for dt, hl in headlines:
        day_map.setdefault(dt, []).append(hl)

    out = {}
    for dt, hlist in day_map.items():
        if not hlist:
            out[dt] = 0.0
            continue
        if scorer == "finbert" and FINBERT_AVAILABLE:
            scores = _score_finbert_batch(hlist)
        elif scorer == "vader" and VADER_AVAILABLE:
            scores = [_score_vader(h) for h in hlist]
        else:
            # try finbert, else vader, else zeros
            if FINBERT_AVAILABLE:
                scores = _score_finbert_batch(hlist)
            elif VADER_AVAILABLE:
                scores = [_score_vader(h) for h in hlist]
            else:
                scores = [0.0] * len(hlist)
        out[dt] = float(np.mean(scores)) if scores else 0.0
    return out


def get_sentiment_series(ticker, dates_index, start=None, end=None, scorer="finbert"):
    """
    Returns pd.Series indexed by dates_index (pd.DatetimeIndex) with sentiment scores.
    - Tries NewsAPI headlines first (if NEWSAPI_KEY provided). If none, returns zeros.
    - Uses FINBERT if available, otherwise VADER.
    """
    if start is None:
        start = dates_index.min().strftime("%Y-%m-%d")
    if end is None:
        end = dates_index.max().strftime("%Y-%m-%d")

    headlines = _fetch_headlines_newsapi(ticker, start, end)

    if len(headlines) == 0:
        # no headlines -> fallback zero series
        return pd.Series(0.0, index=dates_index, dtype=np.float32)

    daily_scores = _aggregate_daily_sentiment(headlines, scorer=scorer)

    # construct series aligned to dates_index
    values = [daily_scores.get(d.strftime("%Y-%m-%d"), np.nan) for d in dates_index]
    ser = pd.Series(values, index=dates_index, dtype=np.float32)
    ser = ser.ffill().bfill().fillna(0.0)
    return ser
