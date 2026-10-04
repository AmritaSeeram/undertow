"""Does formation-window pair quality predict out-of-sample profit? Honest, time-ordered test."""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = ["beta", "half_life", "pvalue"]


def pair_quality_test(pair_log: pd.DataFrame) -> dict:
    """Train a random forest on the EARLIER 70% of pair-windows, test on the LATER 30%.
    AUC 0.5 means no better than a coin flip; the split is by time, never shuffled."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    df = pair_log.sort_values("window_start").reset_index(drop=True)
    df = df[df["entries"] > 0]  # only pairs that actually traded
    if len(df) < 40:
        return {"auc": None, "n_train": 0, "n_test": 0}
    cut = int(len(df) * 0.7)
    tr, te = df.iloc[:cut], df.iloc[cut:]
    y_tr, y_te = (tr["net_return"] > 0).astype(int), (te["net_return"] > 0).astype(int)
    if y_tr.nunique() < 2 or y_te.nunique() < 2:
        return {"auc": None, "n_train": len(tr), "n_test": len(te)}
    rf = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=0)
    rf.fit(tr[FEATURES], y_tr)
    auc = roc_auc_score(y_te, rf.predict_proba(te[FEATURES])[:, 1])
    return {"auc": float(auc), "n_train": len(tr), "n_test": len(te),
            "importance": dict(zip(FEATURES, map(float, rf.feature_importances_)))}
