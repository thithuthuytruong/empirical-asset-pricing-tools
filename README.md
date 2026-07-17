# Empirical Asset Pricing Toolkit

A small Python toolkit for the portfolio-sort and Fama-MacBeth workflows
common in empirical asset pricing research (methodology following Bali,
Engle & Murray, *Empirical Asset Pricing*), plus supporting panel-data
utilities. Built as part of migrating a SAS-based research pipeline to
Python.

## Contents

| File | What it does |
|---|---|
| `src/single_sort.py` | Univariate portfolio sort: EW/VW returns, long-short portfolio, CAPM/FF/FFC alphas, Newey-West t-stats. Inclusive-both-sides breakpoint handling for ties. |
| `src/double_sort.py` | Bivariate portfolio sort — independent or dependent, controlled by one parameter. Outputs the standard grid + Diff/Avg rows and columns (Table 5.12/5.14-style layout). |
| `src/fama_macbeth.py` | Two-step Fama-MacBeth regression across multiple specifications side by side, with Newey-West-adjusted t-stats and subsample (subperiod) support. |
| `src/make_fm_panel.py` | Builds a Fama-MacBeth-ready panel from daily returns + characteristics at any frequency (monthly, annual, irregular per-entity). |
| `src/panel_helpers.py` | `make_sure_continuous_dates`, `lookahead_expand` — small utilities for gap-safe panel construction. |
| `src/winsorize_or_truncate.py` | Outlier handling (winsorize or truncate, one function, cross-sectional by default). |
| `src/symbolrankdates.py` | Rolling estimation window construction — see below for why this one's worth reading closely. |

## `symbolrankdates`: a debugging + performance case study

An earlier cross-join-based implementation of this function had two
issues, found and fixed by systematic differential testing against a
synthetic panel with realistic entry/exit turnover (IPOs/delistings):

1. **A correctness bug**: cross-joining every symbol against every
   window-date pair, then filtering only on whether a *window's*
   end-date fell within a symbol's overall active range — never
   checking the *specific date* — attached dates to a symbol before
   its actual first observation. On a realistic panel, roughly
   **15–25% of the output rows were phantom** (dates a symbol never
   had data for).
2. **A performance/memory problem**: the cross join's intermediate
   size scales with `symbols x window-date pairs`, independent of how
   much of that ever survives filtering — on a 1,000-symbol / 10-year
   panel this reached tens of millions of rows before any filter
   could shrink it.

The fix replaces both cross joins with `searchsorted`-based range
matching and an ordinary merge on real `(symbol, date)` pairs — same
technique as `single_sort`'s `range_join` — while deliberately
*preserving* the one legitimate exclusion rule the original enforced
(a symbol shouldn't be included in a window if its own data doesn't
reach that window's end date, e.g. it delisted partway through).

Reproduce the comparison:
```bash
python benchmarks/benchmark_symbolrankdates.py
```

Typical output on a 1,000-symbol synthetic panel with staggered
entry/exit:
```
CROSS-JOIN version:  16.33s, 10,496,686 rows, peak RSS: 2936 MB
FIXED version:        3.95s,  8,983,346 rows
Rows only in the cross-join version (phantom rows): 1,513,340 (14.4%)
```

## Tests

```bash
pip install -r requirements.txt
pytest tests/
```

## Notes

- Winsorization/truncation is intentionally NOT baked into `single_sort`,
  `double_sort`, or `fama_macbeth` — apply `winsorize_or_truncate` to
  your inputs first, matching how empirical asset pricing papers
  typically treat it as a preprocessing step, not part of the
  estimator itself.
- All functions assume input DataFrames already have any exchange
  filtering, delisting exclusions, etc. applied — sample construction
  choices are left to the caller rather than hardcoded into the
  toolkit.
- No proprietary data, signal definitions, or research findings are
  included in this repo — it's the infrastructure layer only.

## License

This repository is shared for portfolio/review purposes only. All rights
reserved — no license is granted to use, copy, modify, or distribute
this code.
