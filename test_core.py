import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

from undertow import metrics
from undertow.backtest import select_windows, simulate
from undertow.data import simulate_prices
from undertow.pairs import benjamini_hochberg, half_life, select_pairs
from undertow.strategy import Params, positions_from_z
from helpers import fast_eg_pvalue

P = Params()


def test_half_life_recovers_known_ar1():
    rng = np.random.default_rng(0)
    phi, s = 0.9, np.zeros(20000)
    for t in range(1, len(s)):
        s[t] = phi * s[t - 1] + rng.normal()
    assert half_life(s) == pytest.approx(np.log(0.5) / np.log(phi), rel=0.1)


def test_half_life_of_random_walk_is_not_short():
    rng = np.random.default_rng(1)
    assert half_life(np.cumsum(rng.normal(size=2000))) > 40


def test_benjamini_hochberg_known_example():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert benjamini_hochberg(p, 0.05).sum() == 2


def test_positions_enter_exit_and_stop():
    z = np.array([0, -2.5, -1.0, -0.2, 0.0, 2.5, 4.5, 3.0, 0.3, 2.2])
    pos = positions_from_z(z, P)
    assert list(pos) == [0, 1, 1, 0, 0, -1, 0, 0, 0, -1]  # stop-out at 4.5 blocks re-entry until |z|<=0.5


def test_planted_pairs_are_found_and_noise_is_rejected():
    prices, sectors = simulate_prices(n_days=600, n_pairs=3, n_noise=3, seed=3, phi=0.8)
    pairs = select_pairs(prices.iloc[:252], sectors, fast_eg_pvalue, fdr_alpha=0.10, max_pairs=10)
    found = {frozenset((p.x, p.y)) for p in pairs}
    planted = {frozenset((f"X{k}", f"Y{k}")) for k in range(3)}
    assert len(found & planted) >= 2
    assert not any("N" in n for pr in found for n in pr)


def test_no_lookahead_changing_the_future_cannot_change_the_past():
    prices, sectors = simulate_prices(n_days=1200, seed=5)
    p = replace(P)
    base = simulate(prices, select_windows(prices, sectors, p, fast_eg_pvalue), p).net

    cut = 900
    future_shock = prices.copy()
    future_shock.iloc[cut:] *= np.random.default_rng(9).uniform(0.5, 1.5, size=future_shock.iloc[cut:].shape)
    shocked = simulate(future_shock, select_windows(future_shock, sectors, p, fast_eg_pvalue), p).net

    before = base.index < prices.index[cut]
    pd.testing.assert_series_equal(base[before], shocked.reindex(base.index)[before])


def test_higher_costs_never_help():
    prices, sectors = simulate_prices(n_days=1200, seed=2)
    w = select_windows(prices, sectors, P, fast_eg_pvalue)
    total = [simulate(prices, w, replace(P, cost_bps=c)).net.sum() for c in (0, 5, 20)]
    assert total[0] >= total[1] >= total[2]


def test_strategy_profits_on_planted_cointegration_before_costs():
    prices, sectors = simulate_prices(n_days=2000, seed=1)
    p = replace(P, cost_bps=0.0, borrow_bps=0.0)
    res = simulate(prices, select_windows(prices, sectors, p, fast_eg_pvalue), p)
    assert metrics.sharpe(res.gross) > 0.5


def test_metrics_basics():
    r = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01])
    assert metrics.max_drawdown(pd.Series([0.1, -0.5, 0.1])) == pytest.approx(-0.5)
    assert metrics.sharpe(pd.Series([0.01] * 10)) == 0.0  # zero variance guard
    lo, hi = metrics.block_bootstrap_sharpe_ci(np.random.default_rng(0).normal(0.001, 0.01, 1000))
    assert lo < hi
    assert 0.0 <= metrics.probabilistic_sharpe(r) <= 1.0
