"""Walk-forward backtest: select pairs on past data, trade them on the next, repeat."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pairs import Pair, PValueFn, select_pairs, statsmodels_pvalue
from .strategy import Params, positions_from_z


@dataclass
class Window:
    start: int            # first trading row (iloc) of this window
    end: int              # one past the last trading row
    pairs: list[Pair]


@dataclass
class Result:
    net: pd.Series        # daily portfolio return after costs
    turnover: pd.Series   # traded notional per day (drives costs)
    short: pd.Series      # short-leg exposure per day (drives borrow fees)
    gross: pd.Series      # daily portfolio return before costs
    pair_log: pd.DataFrame


def select_windows(prices: pd.DataFrame, sectors: dict[str, list[str]], p: Params,
                   pvalue_fn: PValueFn = statsmodels_pvalue, verbose: bool = False) -> list[Window]:
    """Pair selection for every walk-forward window. It does not depend on entry/exit/cost
    settings, so it is computed once and reused across the whole parameter grid."""
    windows, i, n = [], p.formation, len(prices)
    while i < n - 1:
        end = min(i + p.trading, n)
        formation = prices.iloc[i - p.formation:i]  # strictly before the trading window
        pairs = select_pairs(formation, sectors, pvalue_fn, p.fdr_alpha, (p.hl_min, p.hl_max), p.max_pairs)
        windows.append(Window(i, end, pairs))
        if verbose:
            print(f"window {len(windows):3d}  {prices.index[i].date()}  pairs selected: {len(pairs)}")
        i = end
    return windows


def pair_pnl(py: np.ndarray, px: np.ndarray, pair: Pair, p: Params):
    """Daily P&L (per 1 unit of pair capital) for one pair over one trading window.

    py, px are prices from the last formation close to the last trading close (length T+1).
    Timing: a signal formed at close k is traded at close k + exec_lag (default: the next close)
    and earns the return from there onward. No signal ever sees a price from its own future.
    """
    z = ((np.log(py) - pair.beta * np.log(px)) - pair.mu) / pair.sigma  # fixed formation-window stats
    signal = positions_from_z(z, p)
    T = len(py) - 1
    lag = 1 + p.exec_lag
    held = np.zeros(T + 1)                 # position held over day k (earning the k-1 -> k return)
    if T + 1 > lag:
        held[lag:] = signal[: T + 1 - lag]
    h = held[1:]

    ry, rx = py[1:] / py[:-1] - 1, px[1:] / px[:-1] - 1
    wy, wx = 1.0 / (1.0 + pair.beta), pair.beta / (1.0 + pair.beta)  # weights sum to 1 (gross = 1)
    gross = h * (wy * ry - wx * rx)        # dollar-neutral: long one leg, short the other

    traded = np.abs(np.diff(held))
    cost = traded * p.cost_bps / 1e4
    cost[-1] += abs(held[-1]) * p.cost_bps / 1e4          # force-flat at the end of the window
    short_w = np.where(h > 0, wx, np.where(h < 0, wy, 0.0))
    borrow = short_w * p.borrow_bps / 1e4 / 252
    entries = int(np.sum((held[1:] != 0) & (held[:-1] == 0)))
    turn = traded.copy()
    turn[-1] += abs(held[-1])
    return gross, gross - cost - borrow, entries, turn, short_w


def simulate(prices: pd.DataFrame, windows: list[Window], p: Params) -> Result:
    nets, grosses, turns, shorts, log = [], [], [], [], []
    for w in windows:
        seg = prices.iloc[w.start - 1:w.end]
        idx = seg.index[1:]
        net_w, gross_w, turn_w, short_w = (np.zeros(len(idx)) for _ in range(4))
        for pair in w.pairs:
            g, n, entries, t, sh = pair_pnl(seg[pair.y].to_numpy(), seg[pair.x].to_numpy(), pair, p)
            gross_w += g
            net_w += n
            turn_w += t
            short_w += sh
            log.append(dict(window_start=idx[0], y=pair.y, x=pair.x, beta=pair.beta,
                            half_life=pair.half_life, pvalue=pair.pvalue, entries=entries,
                            net_return=n.sum() / p.max_pairs))
        # each pair owns 1/max_pairs of capital; unfilled slots simply sit in cash
        nets.append(pd.Series(net_w / p.max_pairs, index=idx))
        grosses.append(pd.Series(gross_w / p.max_pairs, index=idx))
        turns.append(pd.Series(turn_w / p.max_pairs, index=idx))
        shorts.append(pd.Series(short_w / p.max_pairs, index=idx))
    return Result(net=pd.concat(nets), gross=pd.concat(grosses), turnover=pd.concat(turns),
                  short=pd.concat(shorts), pair_log=pd.DataFrame(log))
