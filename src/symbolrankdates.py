# ============================================================
# ROLLING ESTIMATION
# ============================================================
"""
Symbolrankdates: Generate rolling estimation dates for each symbol.
Maps each symbol to its valid estimation windows based on the symbol's
first and last date in the dataset.

Parameters:
    inputds        : DataFrame — input dataset (must contain symbol and date columns)
    startdate      : str — start date of the sample period (e.g. '1990-01-01')
    enddate        : str — end date of the sample period (e.g. '2020-12-31')
    period         : str — frequency of rolling window (default: 'month', can use 'day', 'year')
    rolling_freq   : int — how often the window moves (default: 1, e.g. 1 = every month)
    window_length  : int — length of the estimation window (default: 12, e.g. 12 = 12 months)
    groupby        : str — column name for the identifier (default: 'symbol', can use 'permno')
    date_col       : str — column name for the date (default: 'date')

Output:
    DataFrame with columns:
        - symbol   : identifier
        - date     : trading date
        - rankdate : end date of the estimation window

How to use the output:
    Step 1: merge your input data to symbolrankdates output
            e.g.) inputds left join symbolrankdates on symbol and date
    Step 2: run your estimation (mean, var, regression etc) by rankdate

IMPLEMENTATION NOTE (fixed from an earlier cross-join version): rather
than a cross join of every window against every unique trading date,
followed by a second cross join of every symbol against that result,
this builds the (symbol, date, rankdate) mapping via ordinary
date-keyed and symbol-keyed merges. Two things this fixes, not just
speeds up:
  1. A cross-join-then-filter that only checks a symbol's overall
     [firstdate, lastdate] range against a WINDOW's rankdate (never
     checking date_col itself) can attach dates to a symbol BEFORE its
     own first real observation or AFTER its last — e.g. a symbol whose
     data starts mid-window still gets attached to every date in that
     window, including ones before it existed. On a realistic panel
     with entry/exit turnover (IPOs, delistings), this can silently
     produce a large fraction of phantom (symbol, date) rows.
  2. The cross join's INTERMEDIATE size (symbols x window-date pairs,
     before any filtering) grows far faster than the correct final
     output, and is what risks exhausting memory on a large panel — not
     just being slow.
  The "is this symbol still active as of this window's rankdate" check
  the original intended (excluding a symbol from a window if its last
  real date falls short of the window's end) IS preserved here, via a
  small symbol-keyed merge — this is a deliberate design choice, not a
  bug, and dropping it would silently include stale/post-delisting data
  in a window that symbol never really reached.

Example:
    symbolrankdates(dsf_raw, startdate='1990-01-01', enddate='2020-12-31')
    symbolrankdates(dsf_raw, startdate='1990-01-01', enddate='2020-12-31', window_length=24)
    symbolrankdates(dsf_raw, startdate='1990-01-01', enddate='2020-12-31', period='day', rolling_freq=5)
"""
def symbolrankdates(inputds, startdate, enddate, period='month', rolling_freq=1,
                    window_length=12, groupby='symbol', date_col='date'):
    import pandas as pd
    import numpy as np

    startdate = pd.Timestamp(startdate)
    enddate   = pd.Timestamp(enddate)

    def add_period(date, n, snap=None):
        if period == 'month':
            result = date + pd.DateOffset(months=n)
            if snap == 'end':
                result = result + pd.offsets.MonthEnd(0)
            elif snap == 'begin':
                result = result.replace(day=1)
            return result
        elif period == 'year':
            result = date + pd.DateOffset(years=n)
            if snap == 'end':
                result = result + pd.offsets.YearEnd(0)
            elif snap == 'begin':
                result = result.replace(month=1, day=1)
            return result
        elif period == 'day':
            return date + pd.DateOffset(days=n)

    # Step 1: Generate rolling estimation end dates (rankdates) — unchanged.
    rankdates = []
    rankdate  = add_period(startdate, -1, snap='end')
    i         = 0
    while rankdate <= enddate:
        rankstdate = add_period(rankdate, -window_length + 1, snap='begin')
        rankdates.append({'rankdate': rankdate, 'rankstdate': rankstdate, 'i': i})
        i        += 1
        rankdate  = add_period(rankdate, rolling_freq, snap='end')

    estdates = pd.DataFrame(rankdates)
    estdates = estdates[estdates['rankdate'] > startdate].reset_index(drop=True)

    # Step 2: Get unique trading dates within sample period — unchanged.
    trading_dates = inputds[[date_col]].drop_duplicates()
    trading_dates = trading_dates[
        (trading_dates[date_col] >= startdate) &
        (trading_dates[date_col] <= enddate)
    ].sort_values(date_col)
    dates_sorted = trading_dates[date_col].values

    # -------------------------------------------------------
    # Step 3 (FIXED): map each window to the dates it contains via
    # searchsorted, not a cross join — see module notes; a cross join
    # here materializes num_windows x num_unique_dates rows before
    # filtering, most of which get discarded.
    # -------------------------------------------------------
    starts = np.searchsorted(dates_sorted, estdates['rankstdate'].values, side='left')
    ends   = np.searchsorted(dates_sorted, estdates['rankdate'].values,   side='right')
    counts = ends - starts
    window_idx = np.repeat(np.arange(len(estdates)), counts)
    date_idx   = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)]) if counts.sum() else np.array([], dtype=int)
    estdates_expanded = estdates.iloc[window_idx].reset_index(drop=True)
    estdates_expanded[date_col] = dates_sorted[date_idx]

    # -------------------------------------------------------
    # Step 4/5 (FIXED, two parts):
    #
    # (a) Attach each REAL (symbol, date) pair to the windows containing
    # it, via an ordinary merge on date_col — NOT a cross join of every
    # symbol against every window-date pair. This alone fixes a genuine
    # bug in the original: cross-joining firstandlastdates against the
    # date-expanded window table, then filtering only on whether the
    # WINDOW's rankdate falls in [firstdate, lastdate], attaches every
    # date in that window to a symbol regardless of whether the symbol
    # actually has data on each individual date — e.g. a symbol whose
    # data starts mid-window still got attached to dates BEFORE its own
    # first real observation, since the check never looked at date_col,
    # only at rankdate.
    #
    # (b) Still apply the "is this symbol still active as of this
    # window's rankdate" check the original intended: even a symbol with
    # real data inside a window should be excluded from that window if
    # its own last real date falls short of the window's end (e.g. it
    # stopped trading partway through the window) — this is a
    # deliberate exclusion rule in the original, not the bug, and is
    # preserved here via a small, symbol-keyed (not cross) merge of
    # each symbol's lastdate.
    # -------------------------------------------------------
    symbol_dates = inputds[[groupby, date_col]].drop_duplicates()
    symbol_dates = symbol_dates[
        (symbol_dates[date_col] >= startdate) & (symbol_dates[date_col] <= enddate)
    ]

    symbolrankdates = symbol_dates.merge(
        estdates_expanded[[date_col, 'rankdate']], on=date_col, how='inner'
    )

    lastdates = inputds.groupby(groupby)[date_col].max().rename('lastdate').reset_index()
    symbolrankdates = symbolrankdates.merge(lastdates, on=groupby, how='left')
    symbolrankdates = symbolrankdates[
        (symbolrankdates['lastdate'] + pd.offsets.MonthEnd(0)) >= symbolrankdates['rankdate']
    ]

    # Step 6: sort and keep only relevant columns — unchanged.
    symbolrankdates = symbolrankdates[[groupby, date_col, 'rankdate']] \
        .sort_values([groupby, 'rankdate']) \
        .reset_index(drop=True)

    return symbolrankdates
