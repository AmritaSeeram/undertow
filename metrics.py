"""Performance statistics, including uncertainty estimates for the Sharpe ratio."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

DAYS = 252


def sharpe(r, periods: int = DAYS) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    return 0.0 if sd < 1e-12 else float(r.mean() / sd * np.sqrt(periods))


def sortino(r, periods: int = DAYS) -> float:
    r = np.asarray(r, dtype=float)
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    return 0.0 if downside < 1e-12 else float(r.mean() / downside * np.sqrt(periods))


def annual_vol(r, periods: int = DAYS) -> float:
    return float(np.asarray(r, dtype=float).std(ddof=1) * np.sqrt(periods))


def equity_curve(r: pd.Series) -> pd.Series:
    return (1.0 + r).cumprod()


def max_drawdown(r: pd.Series) -> float:
    eq = equity_curve(r)
    return float((eq / eq.cummax() - 1.0).min())


def cagr(r: pd.Series, periods: int = DAYS) -> float:
    years = len(r) / periods
    return float(equity_curve(r).iloc[-1] ** (1.0 / years) - 1.0)


def market_regression(r: pd.Series, bench: pd.Series) -> dict:
    """Regress strategy returns on the market. A market-neutral strategy has beta near 0."""
    df = pd.concat([r, bench], axis=1, join="inner").dropna()
    res = stats.linregress(df.iloc[:, 1], df.iloc[:, 0])
    return {"beta": float(res.slope), "alpha_annual": float(res.intercept * DAYS),
            "alpha_tstat": float(res.intercept / res.intercept_stderr), "r_squared": float(res.rvalue ** 2)}


def block_bootstrap_sharpe_ci(r, block: int = 10, n_boot: int = 5000, ci: float = 0.95, seed: int = 0):
    """Moving-block bootstrap CI for the annualised Sharpe (blocks keep short-run autocorrelation)."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    sample = r[(starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]]
    sr = sample.mean(axis=1) / sample.std(axis=1, ddof=1) * np.sqrt(DAYS)
    lo, hi = np.percentile(sr, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def probabilistic_sharpe(r, benchmark_sr_annual: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012): the probability that the true
    Sharpe exceeds a benchmark, adjusting for sample length, skewness and fat tails."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    sr = r.mean() / r.std(ddof=1)                       # per-period Sharpe
    sr0 = benchmark_sr_annual / np.sqrt(DAYS)
    skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2)
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(n - 1) / denom))


def summarize(r: pd.Series, bench: pd.Series | None = None) -> dict:
    lo, hi = block_bootstrap_sharpe_ci(r)
    out = {"days": len(r), "cagr": cagr(r), "vol": annual_vol(r), "sharpe": sharpe(r),
           "sharpe_ci95": (lo, hi), "prob_sharpe_gt_0": probabilistic_sharpe(r),
           "sortino": sortino(r), "max_drawdown": max_drawdown(r)}
    if bench is not None:
        out["vs_market"] = market_regression(r, bench)
    return out
