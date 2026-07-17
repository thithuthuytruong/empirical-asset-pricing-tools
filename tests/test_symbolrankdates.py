"""
Correctness tests for symbolrankdates, including the partial-history edge
case (symbols with staggered entry/exit) that the cross-join version got
wrong.

Run: pytest tests/test_symbolrankdates.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from symbolrankdates import symbolrankdates


def test_full_history_symbols_have_no_gaps_in_windows():
    symbols = [f'S{i}' for i in range(5)]
    dates = pd.bdate_range('2018-01-01', '2019-12-31')
    rows = [{'symbol': s, 'date': d} for s in symbols for d in dates]
    inputds = pd.DataFrame(rows)

    out = symbolrankdates(inputds, startdate='2018-01-01', enddate='2019-12-31',
                           window_length=6, rolling_freq=1)

    assert set(out['symbol'].unique()) == set(symbols)
    assert out.notna().all().all()


def test_partial_history_symbol_excluded_from_windows_before_its_start():
    symbols = [f'S{i}' for i in range(5)]
    dates = pd.bdate_range('2018-01-01', '2019-12-31')
    rows = [{'symbol': s, 'date': d} for s in symbols for d in dates]
    inputds = pd.DataFrame(rows)
    # S0 doesn't start trading until June 2018
    inputds = inputds[~((inputds.symbol == 'S0') & (inputds.date < pd.Timestamp('2018-06-01')))]

    out = symbolrankdates(inputds, startdate='2018-01-01', enddate='2019-12-31',
                           window_length=6, rolling_freq=1)

    real_pairs = set(zip(inputds['symbol'], inputds['date']))
    out_pairs = set(zip(out['symbol'], out['date']))
    # every (symbol, date) in the output must be a REAL observation --
    # this is the phantom-row bug the cross-join version had
    assert out_pairs.issubset(real_pairs)


def test_symbol_excluded_from_window_it_did_not_reach():
    # S1 stops trading in May 2019; a window ending June 2019 should
    # exclude S1 entirely, even though S1 has real data earlier in that
    # window -- this is the deliberate "still active as of rankdate"
    # rule (not a bug) that must be preserved.
    symbols = [f'S{i}' for i in range(5)]
    dates = pd.bdate_range('2018-01-01', '2019-12-31')
    rows = [{'symbol': s, 'date': d} for s in symbols for d in dates]
    inputds = pd.DataFrame(rows)
    inputds = inputds[~((inputds.symbol == 'S1') & (inputds.date > pd.Timestamp('2019-05-31')))]

    out = symbolrankdates(inputds, startdate='2018-01-01', enddate='2019-12-31',
                           window_length=6, rolling_freq=1)

    s1_rankdates = out[out.symbol == 'S1']['rankdate'].unique()
    assert pd.Timestamp('2019-06-30') not in s1_rankdates


def test_empty_input_does_not_crash():
    empty = pd.DataFrame({'symbol': [], 'date': pd.to_datetime([])})
    out = symbolrankdates(empty, startdate='2018-01-01', enddate='2018-12-31')
    assert len(out) == 0
