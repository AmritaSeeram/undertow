"""Strategy parameters and the trading-signal state machine."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Params:
    # --- signal ---
    entry_z: float = 2.0      # open when |z| >= entry_z
    exit_z: float = 0.5       # close when the spread has reverted to |z| <= exit_z
    stop_z: float = 4.0       # stop out if |z| >= stop_z (the relationship may have broken)
    # --- execution / frictions ---
    exec_lag: int = 1         # extra days between signal and trade (1 = trade at the NEXT close)
    cost_bps: float = 5.0     # commission + slippage, per side, in bps of traded notional
    borrow_bps: float = 50.0  # annualised borrow fee on the short leg, in bps
    # --- walk-forward layout ---
    formation: int = 252      # days used to pick pairs and estimate hedge ratios
    trading: int = 63         # days the selected pairs are traded before re-selecting
    # --- pair selection ---
    max_pairs: int = 8        # capital is split into this many equal slots; empty slots stay in cash
    fdr_alpha: float = 0.10   # Benjamini-Hochberg false-discovery-rate level
    hl_min: float = 2.0       # spread half-life bounds, in days
    hl_max: float = 40.0


def positions_from_z(z: np.ndarray, p: Params) -> np.ndarray:
    """Turn a z-score path into target positions: +1 long the spread, -1 short, 0 flat.

    The position at index t uses information up to and including t only.
    After a stop-out the strategy stays flat until the spread comes back inside the exit band,
    so it cannot immediately re-enter a relationship that just broke.
    """
    pos = np.zeros(len(z))
    cur, blocked = 0, False
    for t, zt in enumerate(z):
        if np.isnan(zt):
            pos[t] = cur
            continue
        if cur == 0:
            if blocked:
                if abs(zt) <= p.exit_z:
                    blocked = False
            elif p.entry_z <= abs(zt) < p.stop_z:
                cur = 1 if zt < 0 else -1  # spread too low -> long it; too high -> short it
        else:
            if abs(zt) >= p.stop_z:
                cur, blocked = 0, True
            elif (cur == 1 and zt >= -p.exit_z) or (cur == -1 and zt <= p.exit_z):
                cur = 0
        pos[t] = cur
    return pos
