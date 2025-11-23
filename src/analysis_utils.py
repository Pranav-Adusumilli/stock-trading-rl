# src/analysis_utils.py
"""
Utility functions for analyzing trading performance.
Includes computation of:
- returns
- annualized return
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Annualized volatility
"""

import numpy as np


def compute_returns(equity_curve):
    """Compute simple period returns from an equity curve."""
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) < 2:
        return np.array([])
    returns = np.diff(eq) / (eq[:-1] + 1e-9)
    return returns


def annualized_return(equity_curve, periods_per_year=252):
    """Annualized return from equity curve."""
    returns = compute_returns(equity_curve)
    if len(returns) == 0:
        return 0.0
    avg = np.mean(returns)
    return (1.0 + avg) ** periods_per_year - 1.0


def sharpe_ratio(equity_curve, periods_per_year=252):
    """Sharpe ratio assuming risk-free rate of 0."""
    returns = compute_returns(equity_curve)
    if len(returns) == 0:
        return 0.0
    mean = np.mean(returns)
    std = np.std(returns, ddof=1) + 1e-9
    return (mean * np.sqrt(periods_per_year)) / std


def sortino_ratio(equity_curve, periods_per_year=252, mar=0.0):
    """Sortino ratio (downside deviation only)."""
    returns = compute_returns(equity_curve)
    if len(returns) == 0:
        return 0.0
    downside = returns[returns < mar]
    downside_std = np.std(downside, ddof=1) if len(downside) > 0 else 1e-9
    mean = np.mean(returns)
    return (mean * np.sqrt(periods_per_year)) / (downside_std + 1e-9)


def max_drawdown(equity_curve):
    """Maximum drawdown from equity curve."""
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    drawdown = (peak - eq) / (peak + 1e-9)
    return float(np.max(drawdown))


def annualized_volatility(equity_curve, periods_per_year=252):
    """Annualized volatility of returns."""
    returns = compute_returns(equity_curve)
    if len(returns) == 0:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))
