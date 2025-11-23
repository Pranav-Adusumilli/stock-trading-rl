# src/data_loader.py
"""
Robust data loader for single- and multi-asset pipelines.

Features:
- Read cached CSVs and coerce index to datetimes (drop NaT rows)
- Download via yfinance when cache missing or empty
- Add indicators via src.indicators.add_indicators
- fetch_multi_asset(..., use_sentiment=True) attaches per-timestep sentiment using
  src.sentiment_fetcher.get_sentiment_series if available (falls back to zeros)
- Safe type conversions and alignment for multi-asset training
"""

import os
from typing import List, Dict

import pandas as pd
import numpy as np
import yfinance as yf

from src.indicators import add_indicators


# -------------------------
# Utilities
# -------------------------
def _ensure_parent_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _read_csv_indexed(path: str) -> pd.DataFrame:
    """
    Read CSV at path, coerce index to datetime, drop rows that couldn't be parsed.
    Returns a DataFrame with a clean DatetimeIndex sorted ascending.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=False)
    # coerce index
    try:
        idx = pd.to_datetime(df.index, errors="coerce")
    except Exception:
        idx = pd.to_datetime(df.index.astype(str), errors="coerce")
    df.index = idx
    # drop rows where index could not be parsed
    df = df[~df.index.isna()].copy()
    df = df.sort_index()
    return df


def _download_and_cache(ticker: str, start: str, end: str, cache_path: str) -> pd.DataFrame:
    """
    Download OHLCV from yfinance and save to cache_path, then read via _read_csv_indexed.
    """
    print(f"Downloading {ticker} {start} -> {end} ...")
    raw = yf.download(ticker, start=start, end=end, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"Failed to download data for {ticker}")
    _ensure_parent_dir(cache_path)
    raw.to_csv(cache_path)
    return _read_csv_indexed(cache_path)


def _load_or_download(ticker: str, start: str, end: str, cache_dir: str) -> pd.DataFrame:
    """
    Load ticker CSV if present and valid; otherwise download and cache it.
    Returns DataFrame with DatetimeIndex.
    """
    cache_path = os.path.join(cache_dir, f"{ticker}.csv")
    _ensure_parent_dir(cache_path)

    # Try reading cache
    if os.path.exists(cache_path):
        try:
            df = _read_csv_indexed(cache_path)
            if df is not None and not df.empty:
                return df
        except Exception:
            # fall through to re-download
            pass

    # Download and cache
    df = _download_and_cache(ticker, start, end, cache_path)
    return df


# -------------------------
# Multi-asset fetcher
# -------------------------
def fetch_multi_asset(
    tickers: List[str],
    start: str,
    end: str,
    cache_dir: str = "data/",
    use_sentiment: bool = False
) -> Dict[str, pd.DataFrame]:

    """
    Load and align multiple tickers.

    Args:
        tickers: list of ticker strings
        start, end: date strings for download (YYYY-MM-DD)
        cache_dir: directory to store per-ticker CSVs
        use_sentiment: if True, try to import src.sentiment_fetcher.get_sentiment_series and attach
                       a 'sentiment' column aligned to the final index. If unavailable, a zero column
                       is attached instead.

    Returns:
        dict {ticker: cleaned_dataframe}
    """
    # 1) Load raw cleaned dataframes (datetime-indexed)
    raw_map = {}
    for t in tickers:
        df = _load_or_download(t, start, end, cache_dir)
        if df is None or df.empty:
            raise RuntimeError(f"No data for ticker {t}")
        raw_map[t] = df

    # 2) Build index union across tickers (use Close channel presence)
    # Ensure each df has a Close column (or close-like column)
    close_frames = []
    for t, df in raw_map.items():
        # find 'Close' column (case-insensitive)
        if "Close" not in df.columns:
            candidates = [c for c in df.columns if "close" in str(c).lower()]
            if candidates:
                df = df.rename(columns={candidates[0]: "Close"})
            else:
                # try first numeric column as fallback
                numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if not numeric_cols:
                    raise RuntimeError(f"Ticker {t} lacks numeric columns and no Close column found.")
                df["Close"] = df[numeric_cols[0]]
        # keep only Close for index union building
        temp = df[["Close"]].copy()
        temp.columns = [t]
        close_frames.append(temp)

    # concat outer to get union index
    all_close = pd.concat(close_frames, axis=1, join="outer").sort_index()
    # forward/back fill to cover gaps (so later reindexing not empty)
    all_close = all_close.ffill().bfill()
    final_index = all_close.index

    # 3) Optionally import sentiment getter
    sentiment_getter = None
    if use_sentiment:
        try:
            from src.sentiment_fetcher import get_sentiment_series  # type: ignore
            sentiment_getter = get_sentiment_series
        except Exception:
            sentiment_getter = None

    # 4) Reindex each df to final_index, fill, add indicators, attach sentiment
    out_map = {}
    for t in tickers:
        df = raw_map[t].reindex(final_index)
        df = df.ffill().bfill()

        # ensure Close numeric
        if "Close" not in df.columns:
            candidates = [c for c in df.columns if "close" in str(c).lower()]
            if candidates:
                df = df.rename(columns={candidates[0]: "Close"})
            else:
                numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                df["Close"] = df[numeric_cols[0]]

        df["Close"] = pd.to_numeric(df["Close"], errors="coerce").ffill().bfill()

        # compute indicators (expects add_indicators to handle DataFrame)
        df_ind = add_indicators(df)

        # sentiment: try to fetch per-day series aligned to final_index
        if use_sentiment and sentiment_getter is not None:
            try:
                s = sentiment_getter(t, final_index)
                # ensure it's a Series aligned to final_index
                if isinstance(s, pd.Series):
                    s = s.reindex(final_index).ffill().bfill().astype(np.float32)
                    df_ind["sentiment"] = s
                else:
                    s_arr = np.asarray(s, dtype=np.float32)
                    if s_arr.shape[0] == len(final_index):
                        df_ind["sentiment"] = pd.Series(s_arr, index=final_index)
                    else:
                        df_ind["sentiment"] = 0.0
            except Exception:
                df_ind["sentiment"] = 0.0
        else:
            df_ind["sentiment"] = 0.0

        # final cleanup
        df_ind = df_ind.dropna().astype(np.float32)
        out_map[t] = df_ind

    return out_map


# -------------------------
# Single-ticker fetcher (used by DQN scripts)
# -------------------------
def fetch_and_cache(ticker: str, start: str, end: str, cache_template: str = "data/{ticker}.csv", use_sentiment: bool = False) -> pd.DataFrame:
    """
    Fetch single ticker, cache to disk, add indicators (and optional sentiment).
    Returns a cleaned dataframe (no NaNs, datetime index).
    """
    cache_path = cache_template.format(ticker=ticker)
    _ensure_parent_dir(cache_path)

    df = None
    if os.path.exists(cache_path):
        try:
            df = _read_csv_indexed(cache_path)
        except Exception:
            df = None

    if df is None or df.empty:
        df = _download_and_cache(ticker, start, end, cache_path)

    # ensure Close present and numeric
    if "Close" not in df.columns:
        candidates = [c for c in df.columns if "close" in str(c).lower()]
        if candidates:
            df = df.rename(columns={candidates[0]: "Close"})
        else:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            df["Close"] = df[numeric_cols[0]]

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce").ffill().bfill()

    df = add_indicators(df)

    # attach sentiment column if requested
    if use_sentiment:
        try:
            from src.sentiment_fetcher import get_sentiment_series  # type: ignore
            idx = df.index
            s = get_sentiment_series(ticker, idx)
            if isinstance(s, pd.Series):
                df["sentiment"] = s.reindex(idx).ffill().bfill().astype(np.float32)
            else:
                arr = np.asarray(s, dtype=np.float32)
                if arr.shape[0] == len(idx):
                    df["sentiment"] = pd.Series(arr, index=idx)
                else:
                    df["sentiment"] = 0.0
        except Exception:
            df["sentiment"] = 0.0
    else:
        df["sentiment"] = 0.0

    df = df.dropna().astype(np.float32)
    return df


# -------------------------
# CLI helper
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default="2023-12-31")
    parser.add_argument("--cache_dir", type=str, default="data/")
    parser.add_argument("--multi", action="store_true", help="Fetch multiple tickers (reads ticker list from comma-separated --ticker)")
    parser.add_argument("--use_sentiment", action="store_true", help="Attach sentiment column if available")
    args = parser.parse_args()

    if args.multi:
        tickers = [s.strip() for s in args.ticker.split(",") if s.strip()]
        out = fetch_multi_asset(tickers, args.start, args.end, args.cache_dir, use_sentiment=args.use_sentiment)
        for k, v in out.items():
            print(k, v.shape)
    else:
        df = fetch_and_cache(args.ticker, args.start, args.end, cache_template=os.path.join(args.cache_dir, "{ticker}.csv"), use_sentiment=args.use_sentiment)
        print(df.head())
