# test_fetch_multi.py
# Simple standalone script to test multi-asset fetching + indicator pipeline.

from src.data_loader import fetch_multi_asset

tickers = ["AAPL", "MSFT", "GOOG", "NVDA"]

print("\nFetching multi-asset data...")
dfs = fetch_multi_asset(
    tickers=tickers,
    start="2018-01-01",
    end="2023-12-31",
    cache_dir="data/"
)

print("\n--- DATA SHAPES ---")
for k, v in dfs.items():
    print(f"{k}: {v.shape}")

print("\nSUCCESS: Multi-asset data loaded and aligned.")
