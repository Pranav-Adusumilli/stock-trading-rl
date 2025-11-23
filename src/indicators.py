# src/indicators.py (replace existing add_indicators function with this)
import pandas as pd
import numpy as np

def _flatten_multiindex_cols(df: pd.DataFrame) -> pd.DataFrame:
    """If df.columns is a MultiIndex, flatten to single-level strings."""
    if hasattr(df.columns, "levels") and df.columns.nlevels > 1:
        df = df.copy()
        df.columns = ["_".join([str(c) for c in col]).strip() for col in df.columns.values]
    return df

def _find_close_column(df: pd.DataFrame):
    """
    Heuristically find the Close column name.
    Returns column name or None.
    """
    cols = list(df.columns)
    # exact match first
    for name in ["Close", "close", "Adj Close", "adj close", "adj_close", "AdjustedClose"]:
        for c in cols:
            if str(c).lower() == str(name).lower().replace(" ", "_"):
                return c
    # substring match
    for c in cols:
        if "close" in str(c).lower():
            return c
    # if nothing, try numeric columns except Volume if present
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) > 0:
        # prefer columns that look like prices (not Volume)
        for c in numeric_cols:
            if "vol" not in str(c).lower():
                return c
        return numeric_cols[0]
    return None

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common indicators to df (requires a Close series).
    This version is robust to MultiIndex columns and non-standard column names.
    """
    if df is None or df.empty:
        return df

    # flatten multiindex columns if present
    df = _flatten_multiindex_cols(df)

    # find Close-like column
    close_col = None
    if "Close" in df.columns:
        close_col = "Close"
    else:
        close_col = _find_close_column(df)

    if close_col is None:
        raise KeyError("Could not find a Close column in dataframe for indicators.")

    # Ensure the Close column is a 1-D numeric Series
    close_series = df[close_col]
    # If it's a DataFrame slice (unexpected), take first column
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]

    close_series = pd.to_numeric(close_series, errors="coerce")
    df = df.copy()
    df["Close"] = close_series

    # now add simple indicators (example: pct return, sma, rsi) — keep your existing logic here
    # Example minimal indicators:
    df["return"] = df["Close"].pct_change().fillna(0)
    df["sma_10"] = df["Close"].rolling(10).mean().fillna(method="bfill").fillna(method="ffill")
    df["sma_40"] = df["Close"].rolling(40).mean().fillna(method="bfill").fillna(method="ffill")
    # RSI simple implementation (14)
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(14).mean().fillna(0)
    ma_down = down.rolling(14).mean().fillna(0)
    rs = ma_up / (ma_down + 1e-12)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # fill remaining NaNs robustly
    df = df.fillna(method="bfill").fillna(method="ffill").fillna(0)

    return df
