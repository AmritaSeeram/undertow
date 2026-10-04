# Undertow

**A pairs-trading research project with a dashboard that shows how fast trading costs erase the edge.**

Most backtests hide their weaknesses. Undertow is built to expose them: every return is out-of-sample, costs are
explicit, and a slider lets anyone drag the trading cost up and watch the strategy's edge disappear.

It combines three things: a quantitative research engine (Python), a small machine-learning test, and an
accessible, responsive dashboard designed so a non-quant can read the result in seconds.

## What is inside

| Part | What it does |
|---|---|
| `undertow/` | Walk-forward pairs-trading engine: cointegration tests with Benjamini-Hochberg false-discovery control, half-life filter, z-score signals, next-close execution, per-side costs and borrow fees |
| `undertow/ml.py` | Random forest that asks whether pair statistics predict profit, trained on early windows and tested on later ones (never shuffled) |
| `dashboard/` | Static dashboard (HTML, CSS, JS, no build step). Live cost and borrow sliders, plain-English verdict, break-even cost, light and dark themes, keyboard accessible |
| `tests/` | 9 tests, including one that shocks all future prices and checks that no past return changes (no look-ahead) |

## Run it

```bash
pip install -r requirements.txt
pytest
python run_export.py --synthetic          # demo data, no download
python run_export.py --start 2012-01-01   # real data from Yahoo Finance
open dashboard/index.html                 # or double-click it
```

`dashboard/data.js` ships with **simulated demo data** so the page works immediately; the page labels it "Demo data".
Run the real export to replace it.

## Design choices

- The first thing on the page is a one-sentence verdict, not a table of statistics.
- Sliders are 44px tall, text meets WCAG AA contrast in both themes, and charts have text alternatives.
- Plain-language labels ("worst drop") sit beside the finance terms ("max drawdown").

## Limitations

- Universe is today's large caps, so results carry survivorship bias.
- Costs are an assumption, which is why the dashboard lets you change them.
- Statistical arbitrage on liquid US stocks is heavily competed. A weak or negative result is plausible and is reported as is.
