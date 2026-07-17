"""
Reproduces the correctness + performance comparison between a cross-join
based implementation of symbolrankdates and the searchsorted/merge-based
fix in src/symbolrankdates.py.

Run: python benchmarks/benchmark_symbolrankdates.py
"""
import sys
import os
import time
import resource

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from symbolrankdates import symbolrankdates


def cross_join_version(inputds, startdate, enddate, period='month', rolling_freq=1,
                        window_length=12, groupby='symbol', date_col='date'):
    """Original cross-join implementation, kept here only for comparison."""
    startdate = pd.Timestamp(startdate)
    enddate = pd.Timestamp(enddate)

    def add_period(date, n, snap=None):
        if period == 'month':
            result = date + pd.DateOffset(months=n)
            if snap == 'end':
                result = result + pd.offsets.MonthEnd(0)
            elif snap == 'begin':
                result = result.replace(day=1)
            return result

    rankdates = []
    rankdate = add_period(startdate, -1, snap='end')
    i = 0
    while rankdate <= enddate:
        rankstdate = add_period(rankdate, -window_length + 1, snap='begin')
        rankdates.append({'rankdate': rankdate, 'rankstdate': rankstdate, 'i': i})
        i += 1
        rankdate = add_period(rankdate, rolling_freq, snap='end')

    estdates = pd.DataFrame(rankdates)
    estdates = estdates[estdates['rankdate'] > startdate].reset_index(drop=True)

    trading_dates = inputds[[date_col]].drop_duplicates()
    trading_dates = trading_dates[
        (trading_dates[date_col] >= startdate) & (trading_dates[date_col] <= enddate)
    ].sort_values(date_col)

    estdates = estdates.merge(trading_dates, how='cross')
    estdates = estdates[
        (estdates[date_col] >= estdates['rankstdate']) & (estdates[date_col] <= estdates['rankdate'])
    ]

    firstandlastdates = inputds.sort_values([groupby, date_col]).groupby(groupby).agg(
        firstdate=(date_col, 'min'), lastdate=(date_col, 'max')
    ).reset_index()

    out = firstandlastdates.merge(estdates, how='cross')
    out = out[
        (out['firstdate'] <= out['rankdate']) &
        ((out['lastdate'] + pd.offsets.MonthEnd(0)) >= out['rankdate'])
    ]
    out = out[[groupby, date_col, 'rankdate']].sort_values([groupby, 'rankdate']).reset_index(drop=True)
    return out


def make_turnover_panel(n_symbols, start, end, seed=0):
    """Synthetic panel with staggered entry/exit (IPOs/delistings)."""
    rng = np.random.default_rng(seed)
    all_dates = pd.bdate_range(start, end)
    rows = []
    for i in range(n_symbols):
        start_idx = rng.integers(0, len(all_dates) - 250)
        span = rng.integers(250, len(all_dates) - start_idx)
        sym_dates = all_dates[start_idx:start_idx + span]
        for d in sym_dates:
            rows.append({'symbol': f'S{i}', 'date': d})
    return pd.DataFrame(rows)


def main():
    inputds = make_turnover_panel(n_symbols=1000, start='2010-01-01', end='2019-12-31')
    print(f'input rows: {len(inputds):,}')
    print()

    t0 = time.time()
    out_cross = cross_join_version(inputds, startdate='2010-01-01', enddate='2019-12-31',
                                    window_length=12, rolling_freq=1)
    t1 = time.time()
    peak_cross = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f'CROSS-JOIN version:  {t1 - t0:.2f}s, {len(out_cross):,} rows, peak RSS so far: {peak_cross:.0f} MB')

    t0 = time.time()
    out_fixed = symbolrankdates(inputds, startdate='2010-01-01', enddate='2019-12-31',
                                 window_length=12, rolling_freq=1)
    t1 = time.time()
    peak_fixed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f'FIXED version:        {t1 - t0:.2f}s, {len(out_fixed):,} rows, peak RSS so far: {peak_fixed:.0f} MB')

    print()
    phantom_rows = len(out_cross) - len(out_fixed)
    print(f'Rows only in the cross-join version (phantom dates a symbol never had): {phantom_rows:,}')
    print(f'  = {100 * phantom_rows / len(out_cross):.1f}% of the cross-join version\'s output')


if __name__ == '__main__':
    main()
