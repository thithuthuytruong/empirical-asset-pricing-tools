"""
winsorize_or_truncate: Handle outliers in one or more variables via
winsorization or truncation, controlled by the `method` parameter.

WINSORIZATION (method='winsorize'): values in the top h% of X are set
to the (100-h)th percentile of X; values in the bottom l% are set to
the lth percentile of X. Extreme observations are pulled in to a more
moderate level rather than removed.

TRUNCATION (method='truncate'): the same cutoffs (lth and (100-h)th
percentiles) are used, but instead of capping values at the cutoff,
values beyond it are set to MISSING (NaN) — effectively removing that
observation from any analysis using X, rather than moderating it.

In most empirical asset pricing work this is applied CROSS-SECTIONALLY,
period by period (e.g. winsorizing at the 1st/99th percentile WITHIN
each month) rather than on the pooled panel, since pooling ignores that
a variable's scale/distribution can shift over time. The `by` parameter
controls this — default is cross-sectional (grouped by date_col); pass
by=None for a single pooled cutoff across the whole dataset instead.

Parameters:
    inputds : DataFrame — input dataset.
    var     : str or list of str — column(s) to winsorize/truncate.
              Each is treated independently (percentiles computed
              separately per variable, and per group if `by` is set).
    method  : str (default: 'winsorize') — 'winsorize' or 'truncate',
              per above. Any other value raises ValueError.
    limits  : tuple (l, h) (default: (0.01, 0.01)) — FRACTIONS, not
              percentages: l = lower cutoff (0.01 = bottom 1%), h =
              upper cutoff (0.01 = top 1%). Common research values are
              0.005 and 0.01. l and h need not be equal; either can be
              0 to skip winsorizing/truncating that side.
    by      : str, list of str, or None (default: 'date') — column(s)
              to group by when computing percentiles, matching the
              standard cross-sectional (per-period) convention. Pass
              None to compute one pooled cutoff across the whole
              dataset instead.

Output:
    A COPY of inputds — the original is not modified — with each `var`
    winsorized/truncated in place (values overwritten), plus two new
    boolean flag columns per var: f'{var}_winsorized_lo' /
    f'{var}_winsorized_hi' (or f'{var}_truncated_lo'/'_hi' for
    method='truncate'), marking which observations were adjusted on
    which side. Existing NaNs in the original column are left as NaN
    and never get flagged or counted toward the percentile calculation
    (NaN comparisons are always False).

Example:
    # cross-sectional (per-month) winsorization at 1%/99%, standard convention
    df = winsorize_or_truncate(df, var=['size', 'bm'], method='winsorize',
                                limits=(0.01, 0.01), by='date')

    # pooled truncation at 0.5%/99.5% on a single variable
    df = winsorize_or_truncate(df, var='exret', method='truncate',
                                limits=(0.005, 0.005), by=None)
"""
def winsorize_or_truncate(inputds, var, method='winsorize', limits=(0.01, 0.01), by='date'):

    import pandas as pd
    import numpy as np

    if method not in ('winsorize', 'truncate'):
        raise ValueError(f"method must be 'winsorize' or 'truncate', got {method!r}")

    l, h = limits
    if not (0 <= l < 0.5) or not (0 <= h < 0.5):
        raise ValueError(f"limits must each be a fraction in [0, 0.5), got {limits}")

    varlist = [var] if isinstance(var, str) else list(var)
    missing_var = set(varlist) - set(inputds.columns)
    if missing_var:
        raise ValueError(f"var column(s) not found in inputds: {missing_var}")

    if by is None:
        group_keys = None
    else:
        group_keys = [by] if isinstance(by, str) else list(by)
        missing_by = set(group_keys) - set(inputds.columns)
        if missing_by:
            raise ValueError(f"'by' column(s) not found in inputds: {missing_by}")

    df = inputds.copy()
    tag = 'winsorized' if method == 'winsorize' else 'truncated'

    for v in varlist:
        if group_keys is None:
            # Pooled cutoffs — same scalar cutoff applied to every row.
            lo_cut = df[v].quantile(l)
            hi_cut = df[v].quantile(1 - h)
        else:
            # Cross-sectional cutoffs — one pair of cutoffs PER GROUP
            # (e.g. per date), broadcast back to each row via transform.
            grp = df.groupby(group_keys)[v]
            lo_cut = grp.transform(lambda x: x.quantile(l))
            hi_cut = grp.transform(lambda x: x.quantile(1 - h))

        is_lo = df[v] < lo_cut
        is_hi = df[v] > hi_cut

        df[f'{v}_{tag}_lo'] = is_lo
        df[f'{v}_{tag}_hi'] = is_hi

        if method == 'winsorize':
            df.loc[is_lo, v] = lo_cut if group_keys is None else lo_cut[is_lo]
            df.loc[is_hi, v] = hi_cut if group_keys is None else hi_cut[is_hi]
        else:  # truncate
            df.loc[is_lo | is_hi, v] = np.nan

    return df
