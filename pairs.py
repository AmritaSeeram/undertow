"""Pair selection: cointegration tests with multiple-testing control."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd

PValueFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class Pair:
    y: str            # the leg that is bought when the spread is low
    x: str            # the hedge leg
    alpha: float
    beta: float       # hedge ratio from log(y) = alpha + beta * log(x)
    mu: float         # mean of the formation-window spread  (log y - beta * log x)
    sigma: float      # std of the formation-window spread
    half_life: float  # days
    pvalue: float     # direction-adjusted Engle-Granger p-value


def statsmodels_pvalue(ly: np.ndarray, lx: np.ndarray) -> float:
    """Engle-Granger cointegration p-value (regress ly on lx, ADF test on the residuals)."""
    from statsmodels.tsa.stattools import coint

    return float(coint(ly, lx, trend="c", autolag="aic")[1])


def ols_hedge(ly: np.ndarray, lx: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([np.ones_like(lx), lx])
    (alpha, beta), *_ = np.linalg.lstsq(X, ly, rcond=None)
    return float(alpha), float(beta)


def half_life(spread: np.ndarray) -> float:
    """Half-life of mean reversion from the AR(1) form  dS_t = a + b * S_{t-1} + e."""
    s = np.asarray(spread, dtype=float)
    X = np.column_stack([np.ones(len(s) - 1), s[:-1]])
    b = np.linalg.lstsq(X, np.diff(s), rcond=None)[0][1]
    if b >= 0:
        return float("inf")   # no mean reversion
    if b <= -1:
        return 0.0            # reverts faster than daily data can resolve
    return float(-np.log(2) / np.log(1 + b))


def benjamini_hochberg(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Boolean mask of hypotheses rejected while controlling the false discovery rate."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    keep = np.zeros(m, dtype=bool)
    if m == 0:
        return keep
    order = np.argsort(p)
    passed = p[order] <= alpha * np.arange(1, m + 1) / m
    if passed.any():
        keep[order[: np.max(np.where(passed)[0]) + 1]] = True
    return keep


def select_pairs(formation: pd.DataFrame, sectors: dict[str, list[str]],
                 pvalue_fn: PValueFn = statsmodels_pvalue, fdr_alpha: float = 0.10,
                 hl_range: tuple[float, float] = (2.0, 40.0), max_pairs: int = 8) -> list[Pair]:
    """Pick tradable pairs using ONLY the formation window.

    1. Test every within-sector pair in both regression directions. Engle-Granger is not
       symmetric, so keep the better direction and double its p-value (Bonferroni over 2).
    2. Benjamini-Hochberg across all pairs tested: with hundreds of tests per window, some
       pairs look cointegrated purely by chance, and an uncorrected test would trade them.
    3. Require a positive hedge ratio and a half-life inside [hl_min, hl_max].
    4. Keep the `max_pairs` smallest p-values.
    """
    logp = np.log(formation)
    cands = []  # (p, y, x)
    for names in sectors.values():
        names = [n for n in names if n in logp.columns]
        for a, b in combinations(names, 2):
            la, lb = logp[a].to_numpy(), logp[b].to_numpy()
            p_ab, p_ba = pvalue_fn(la, lb), pvalue_fn(lb, la)
            p = min(1.0, 2.0 * min(p_ab, p_ba))
            cands.append((p, a, b) if p_ab <= p_ba else (p, b, a))
    if not cands:
        return []

    keep = benjamini_hochberg(np.array([c[0] for c in cands]), fdr_alpha)
    pairs = []
    for (p, y, x), ok in zip(cands, keep):
        if not ok:
            continue
        ly, lx = logp[y].to_numpy(), logp[x].to_numpy()
        alpha, beta = ols_hedge(ly, lx)
        if beta <= 0:
            continue
        spread = ly - beta * lx
        hl = half_life(spread)
        if not (hl_range[0] <= hl <= hl_range[1]):
            continue
        pairs.append(Pair(y, x, alpha, beta, float(spread.mean()), float(spread.std(ddof=1)), hl, p))
    pairs.sort(key=lambda q: q.pvalue)
    return pairs[:max_pairs]
