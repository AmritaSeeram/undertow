"""Price loading (Yahoo Finance, cached) and a synthetic data generator for tests."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# Pairs are only searched *within* a sector. Restricting the search space to economically
# related stocks cuts the number of hypothesis tests and reduces spurious cointegration.
# NOTE: these are today's large caps, so results carry survivorship bias (see README).
SECTORS: dict[str, list[str]] = {
    "energy": ["XOM", "CVX", "COP", "EOG", "OXY", "SLB"],
    "banks": ["JPM", "BAC", "WFC", "C", "USB", "PNC"],
    "staples": ["KO", "PEP", "PG", "CL", "KMB", "MDLZ"],
    "utilities": ["DUK", "SO", "D", "AEP", "EXC", "XEL"],
    "reits": ["PLD", "AMT", "SPG", "O", "PSA", "EQR"],
    "retail": ["HD", "LOW", "TGT", "WMT", "COST"],
    "pharma": ["JNJ", "PFE", "MRK", "ABT", "LLY", "BMY"],
    "industrials": ["CAT", "DE", "HON", "MMM", "UNP", "UPS"],
}
BENCHMARK = "SPY"


def all_tickers() -> list[str]:
    return sorted({t for names in SECTORS.values() for t in names})


def clean_prices(raw: pd.DataFrame, max_missing: float = 0.05) -> pd.DataFrame:
    """Drop tickers with too much missing history, forward-fill tiny gaps, drop leftover NaN rows."""
    keep = raw.columns[raw.isna().mean() <= max_missing]
    return raw[keep].ffill(limit=3).dropna()


def load_prices(tickers: list[str], start: str, end: str | None = None,
                cache_dir: str = "data") -> pd.DataFrame:
    """Adjusted close prices (splits and dividends included), cached to CSV."""
    import yfinance as yf  # imported lazily so tests run without network or yfinance

    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)
    digest = hashlib.md5(",".join(sorted(tickers)).encode()).hexdigest()[:8]
    path = cache / f"prices_{start}_{end or 'latest'}_{digest}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    raw = yf.download(sorted(tickers), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    prices = clean_prices(raw)
    prices.to_csv(path)
    return prices


def simulate_prices(n_days: int = 1500, n_pairs: int = 3, n_noise: int = 3, seed: int = 0,
                    phi: float = 0.9, spread_vol: float = 0.01):
    """Synthetic universe with *planted* cointegrated pairs plus pure random walks.

    Returns (prices, sectors). Everything sits in one sector so the selection step has to
    tell the real pairs apart from unrelated random walks. Used by the tests and `--synthetic`.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    market = rng.normal(0.0002, 0.01, n_days)

    def random_walk(vol: float) -> np.ndarray:
        return np.cumsum(market + rng.normal(0.0, vol, n_days))

    cols: dict[str, np.ndarray] = {}
    for k in range(n_pairs):
        lx = np.log(100) + random_walk(0.012)
        eps = rng.normal(0.0, spread_vol, n_days)
        s = np.zeros(n_days)
        for t in range(1, n_days):  # AR(1) / Ornstein-Uhlenbeck spread
            s[t] = phi * s[t - 1] + eps[t]
        beta = 0.8 + 0.2 * rng.random()
        cols[f"X{k}"] = np.exp(lx)
        cols[f"Y{k}"] = np.exp(np.log(80) + beta * (lx - np.log(100)) + s)
    for k in range(n_noise):
        cols[f"N{k}"] = np.exp(np.log(50) + random_walk(0.012))
    prices = pd.DataFrame(cols, index=idx)
    return prices, {"synthetic": list(prices.columns)}
