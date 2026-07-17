"""
make_fm_panel: Build a Fama-MacBeth-ready panel from a DAILY return/
outcome series and a separate, ARBITRARY-FREQUENCY characteristics
DataFrame (monthly, quarterly, annual, or even irregular per-entity
dates like staggered fiscal year-ends).

Rather than assuming a fixed calendar frequency, each daily row in
`dsf` is assigned to a PERIOD by matching it FORWARD to the nearest
`indepvar` date for that same entity (via merge_asof) — i.e. a daily
observation on date d belongs to whichever period ends at the smallest
indepvar date >= d. This makes the collapse frequency-agnostic: it
falls out of indepvar's own dates rather than being hardcoded, so the
exact same code path handles monthly, annual, or irregular indepvar.

`var` is then collapsed to ONE value per (groupby, period) — compounded
via log-compounding if var_is_return, else averaged, same convention as
single_sort's var_is_return — and led by `lookn` periods per entity.
The lead uses make_sure_continuous_dates() before lookahead_expand():
a genuine gap in an entity's period history first becomes an explicit
NaN row, so lookahead_expand's groupby().shift(-lookn) always advances
exactly lookn periods (or lands on NaN at a real gap) instead of
silently grabbing whatever row happens to be next in the data, however
far away it actually is — this guard is itself frequency-agnostic,
since both helper functions only look at the SET of distinct period
dates actually present, not at calendar arithmetic.

This reproduces the "Step 1" input-prep pattern for regressions of 
future returns (r_{t+1}) on current characteristics —
J=0, K=1 in single_sort/double_sort's terms, generalized to any
frequency and any lookn.

Parameters:
    dsf      : DataFrame — daily data, must contain date_col, groupby,
               and var. Any filtering (exchange, delisting exclusion,
               price floors, etc.) should already be applied before
               calling this function — it does not filter dsf itself.
    indepvar : DataFrame — characteristics at WHATEVER frequency you
               have them (monthly, annual, irregular...), must contain
               date_col, groupby, and 'subperiod' (required by
               fama_macbeth downstream — see that docstring; set
               indepvar['subperiod'] = 1 if you don't want to split
               your sample). One row per (groupby, period), dated as of
               the CHARACTERISTIC period's end date t.
    var      : str — the daily column in dsf to collapse and lead
               (e.g. 'exret', 'ret'). MUST already be on a percent
               scale if var_is_return=True (1.5 = 1.5%) — this function
               does not rescale it, matching single_sort's convention.
    begdate, enddate : str or Timestamp — sample bounds applied to the
               final panel's date_col.
    lookn    : int (default: 1) — number of PERIODS ahead to lead var
               by, where a period is whatever gap indepvar's own dates
               define for a given entity (one month if indepvar is
               monthly, one year if indepvar is annual, etc.). lookn=1
               reproduces r_{t+1}.
    groupby  : str — entity identifier column, shared by dsf and
               indepvar (default: 'symbol').
    date_col : str — date column, shared by dsf and indepvar
               (default: 'date').
    var_is_return : bool (default: True) — True compounds var via
               log-compounding within each period (returns); False
               averages it (characteristics/levels). Same rationale as
               single_sort's var_is_return.

Output:
    indepvar's columns, plus f'{var}_next{lookn}' — the entity's `var`
    value lookn periods after date_col, or NaN if that period is
    missing or absent from the entity's history. Ready to pass directly
    as `data` to fama_macbeth, with y=f'{var}_next{lookn}'.

    NOTE: make_sure_continuous_dates and lookahead_expand must already
    be defined in the same module (panel_helpers.py) — this function calls them directly, unprefixed.

Example (monthly indepvar):
    fm_input = make_fm_panel(
        dsf      = tmp1,               # daily, exret already computed & exchange-filtered
        indepvar = indepvar_monthly,   # one row per (symbol, month), incl. 'subperiod'
        var      = 'exret',
        begdate  = start_period,
        enddate  = end_period,
        lookn    = 1,
        )
    result = fama_macbeth(data=fm_input, y='exret_next1', specs=[['sortvar']],
                           spec_labels=['(1)'], date_col='date', lag=6)

Example (annual indepvar — same call, no other changes needed):
    fm_input = make_fm_panel(
        dsf      = tmp1,               # still daily
        indepvar = indepvar_annual,    # one row per (symbol, fiscal year-end), incl. 'subperiod'
        var      = 'exret',
        begdate  = start_period,
        enddate  = end_period,
        lookn    = 1,                  # r_{year+1} on characteristics at year t
        )
"""
def make_fm_panel(dsf, indepvar, var, begdate, enddate, lookn=1,
                   groupby='symbol', date_col='date', var_is_return=True):

    import pandas as pd
    import numpy as np
    from panel_helpers import make_sure_continuous_dates, lookahead_expand

    begdate = pd.Timestamp(begdate)
    enddate = pd.Timestamp(enddate)

    required_dsf = {date_col, groupby, var}
    missing_dsf = required_dsf - set(dsf.columns)
    if missing_dsf:
        raise ValueError(f"dsf is missing required column(s): {missing_dsf}")

    required_indep = {date_col, groupby, 'subperiod'}
    missing_indep = required_indep - set(indepvar.columns)
    if missing_indep:
        raise ValueError(
            f"indepvar is missing required column(s): {missing_indep}. "
            "'subperiod' must be an exact name — set indepvar['subperiod'] = 1 "
            "if you don't want to split your sample downstream in fama_macbeth."
        )

    # -------------------------------------------------------
    # Assign each daily observation to a PERIOD by matching it FORWARD
    # to the nearest indepvar date for that same entity — NOT a fixed
    # calendar frequency. merge_asof requires both frames sorted by
    # the 'on' column globally (date_col); groupby is added only as a
    # stable tie-breaker and doesn't disturb that global date ordering.
    # A daily row with no indepvar date on/after it for its entity
    # (e.g. trailing days past that entity's last indepvar date) gets
    # no match and is dropped below — there's no period to attach it to.
    # -------------------------------------------------------
    period_ends = indepvar[[groupby, date_col]].drop_duplicates()
    period_ends = period_ends.sort_values([date_col, groupby]).rename(columns={date_col: '_period_end'})

    daily = dsf[[groupby, date_col, var]].sort_values([date_col, groupby])
    daily = pd.merge_asof(
        daily, period_ends, left_on=date_col, right_on='_period_end',
        by=groupby, direction='forward'
    )
    daily = daily.dropna(subset=['_period_end'])

    # -------------------------------------------------------
    # Collapse daily var to ONE value per (groupby, period) — compound
    # if var_is_return, else average. Vectorized transform, same
    # mechanics as single_sort's Step 5 (see there for the full
    # rationale on why compounding vs. averaging matters).
    # -------------------------------------------------------
    grp = daily.groupby([groupby, '_period_end'])
    if var_is_return:
        daily['_log1p_var'] = np.log1p(daily[var] / 100)
        daily[var] = 100 * (np.exp(grp['_log1p_var'].transform('sum')) - 1)
    else:
        daily[var] = grp[var].transform('mean')

    daily['_last_date'] = grp[date_col].transform('max')
    period_panel = daily[daily[date_col] == daily['_last_date']].copy()
    period_panel = period_panel.drop_duplicates(subset=[groupby, '_period_end'])
    period_panel = period_panel[[groupby, '_period_end', var]].rename(columns={'_period_end': date_col})
    period_panel = period_panel.reset_index(drop=True)

    # -------------------------------------------------------
    # Fill gaps BEFORE leading (see module docstring for why), then
    # attach the lookn-period-ahead value.
    # -------------------------------------------------------
    period_panel = make_sure_continuous_dates(period_panel, identifier=groupby, date_col=date_col)
    period_panel = lookahead_expand(period_panel, groupby=groupby, lookn=lookn, var=var, date_col=date_col)

    lead_col = f'{var}_next{lookn}'
    fm_input = indepvar.merge(
        period_panel[[groupby, date_col, lead_col]], on=[groupby, date_col], how='left'
    )
    fm_input = fm_input[(fm_input[date_col] >= begdate) & (fm_input[date_col] <= enddate)].copy()

    return fm_input
