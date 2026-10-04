"""Run the backtest and write dashboard/data.js.
   python run_export.py --synthetic        # demo data, no download
   python run_export.py --start 2012-01-01 # real data (needs internet)"""
import argparse, json
from pathlib import Path
from undertow.backtest import select_windows, simulate
from undertow.data import BENCHMARK, SECTORS, all_tickers, load_prices, simulate_prices
from undertow.ml import pair_quality_test
from undertow.pairs import statsmodels_pvalue
from undertow.strategy import Params


def main(argv=None, pvalue_fn=statsmodels_pvalue):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2012-01-01"); ap.add_argument("--end")
    ap.add_argument("--synthetic", action="store_true"); a = ap.parse_args(argv)
    p = Params()
    if a.synthetic:
        prices, sectors = simulate_prices(n_days=2000, seed=1)
    else:
        sectors, prices = SECTORS, load_prices(all_tickers(), a.start, a.end)
    windows = select_windows(prices, sectors, p, pvalue_fn, verbose=True)
    r = simulate(prices, windows, p)
    data = {"synthetic": a.synthetic, "dates": [d.strftime("%Y-%m-%d") for d in r.gross.index],
            "gross": [round(x, 7) for x in r.gross], "turnover": [round(x, 5) for x in r.turnover],
            "short": [round(x, 5) for x in r.short], "base": {"cost_bps": p.cost_bps, "borrow_bps": p.borrow_bps},
            "windows": len(windows), "avg_pairs": sum(len(w.pairs) for w in windows) / max(len(windows), 1),
            "ml": pair_quality_test(r.pair_log)}
    Path("dashboard").mkdir(exist_ok=True)
    Path("dashboard/data.js").write_text("window.UNDERTOW_DATA = " + json.dumps(data) + ";\n")
    print("wrote dashboard/data.js")

if __name__ == "__main__":
    main()
