"""Dependency-free stand-in for the Engle-Granger test, used only to keep unit tests fast
and runnable without statsmodels. The production code uses statsmodels (see pairs.py)."""
import numpy as np

# MacKinnon (1991) asymptotic critical values for a 2-variable cointegration test with a constant:
# t = -3.90 / -3.34 / -3.04 correspond to p = 0.01 / 0.05 / 0.10. Interpolate between them and
# decay exponentially beyond -3.90 so very strong cointegration gets a very small p-value.
_T = [-3.04, -3.34, -3.90]
_P = [0.10, 0.05, 0.01]


def fast_eg_pvalue(ly: np.ndarray, lx: np.ndarray) -> float:
    X = np.column_stack([np.ones_like(lx), lx])
    resid = ly - X @ np.linalg.lstsq(X, ly, rcond=None)[0]
    d, lag = np.diff(resid), resid[:-1]
    Z = np.column_stack([np.ones_like(lag), lag])
    coef = np.linalg.lstsq(Z, d, rcond=None)[0]
    e = d - Z @ coef
    se = np.sqrt(e @ e / (len(d) - 2) * np.linalg.inv(Z.T @ Z)[1, 1])
    t = coef[1] / se
    if t < -3.90:
        return max(1e-12, 0.01 * 10 ** (2.0 * (t + 3.90)))
    if t > -1.0:
        return 0.9
    return float(np.interp(t, [-3.90, -3.34, -3.04, -1.0], [0.01, 0.05, 0.10, 0.9]))
