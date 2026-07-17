"""
Single_sort: Single-sort portfolio analysis with equal-weighted and value-weighted returns,
long-short portfolio, and risk-adjusted alphas (CAPM, FF3, FFC).

Parameters:
    inputsortvar : DataFrame — REQUIRED columns: (date, symbol, sortvar,
                   subperiod, weight). Both 'subperiod' and 'weight' MUST
                   be named exactly that — rename your columns to these
                   exact names before calling this function. This
                   function always reads columns literally named
                   'subperiod' and 'weight'; it does not accept different
                   column names for either.

                   'subperiod' divides the sample into subperiods for
                   SEPARATE analysis (e.g. pre/post a structural break).
                   Results are computed and reported INDEPENDENTLY within
                   each distinct value — never pooled across subperiods.
                   If you do NOT want to split your sample, set every row
                   to the same constant value before calling this function:
                       inputsortvar['subperiod'] = 1
                   Rows with a MISSING (NaN) 'subperiod' value are DROPPED
                   entirely before any further processing.

                   'weight' is the weighting variable (e.g. lagged market
                   cap), already attached to inputsortvar with whatever
                   convention you need (e.g. measured as of the end of
                   the formation month) — this function does not compute
                   or look it up itself.

                   `histexch` (stock exchange code) — or any other column referenced by
                   breakpoint_mask — is OPTIONAL — only needed if you pass
                   breakpoint_mask to restrict which entities are used to
                   COMPUTE breakpoints (e.g. KOSPI-only breakpoints).
                   If you don't use breakpoint_mask, no such column is required.
    inputvar     : DataFrame — must contain (date, symbol, var). Does NOT
                   need to contain 'weight' — weight is attached via
                   inputsortvar and carried through the holding-period
                   join automatically.
    ff3factor    : DataFrame — must contain (date, rmrf, SMB, HML, UMD)
    begdate      : str — start date (e.g. '1990-01-01')
    enddate      : str — end date (e.g. '2020-12-31')
    numPort      : int — number of portfolios (e.g. 5 or 10)
    sortvar      : str — variable used to sort symbols into portfolios
    var          : str — variable to compute portfolio-level statistics for.
                   Daily observations are first collapsed to one value per
                   (symbol, form_date, port_{sortvar}, calendar month) — so
                   a K>1 holding period still yields one observation per
                   month per cohort, not one per whole holding period.
                   How that collapse happens depends on var_is_return:
                   compounded (log-compounding) if True, averaged
                   (equal-weighted mean) if False. If True, var MUST
                   already be on a percent scale (1.5 = 1.5%) — verify
                   inputvar[var].describe() yourself; this function does
                   NOT rescale var.
    J            : int — months to wait after portfolio formation
    K            : int — holding period length in months
    lag          : int — number of lags for Newey-West t-statistics (default: 6)
    date_col     : str — column name for the date (default: 'date')
    groupby      : str — column name for the identifier (default: 'symbol')
    breakpoint_mask : optional boolean Series (aligned to inputsortvar's index)
                       marking which rows are used to COMPUTE breakpoints
                       (e.g. KOSPI-only). Breakpoints are still APPLIED to every
                       row regardless. If None, all rows are used to compute breakpoints.
                       If breakpoints should be computed on a subset, follow this code:
                       kospi_mask = inputsortvar['histexch'] == kospi_code
    var_is_return : bool (default: True) — set True for return series like
                    'exret'/'ret' (compounded), False for characteristics
                    or levels (averaged) — compounding a non-return
                    variable wouldn't be economically meaningful. See var
                    above for the mechanics.

Output:
    DataFrame with columns:
        - subperiod    : subperiod indicator
        - weight_type  : EW (Equal-weight) or VW (Value-weight)
        - Model        : Excess return (Average excess return), CAPM, FF, or FFC
        - Coefficient  : Excess return, alphas and factor sensitivities for each portfolios
        - Portfolio 1,...,numPort, numPort-1 : values and t-stat, adjusted following NW

Example for input:
    kospi_mask = inputsortvar['histexch'] == kospi_code   # True for KOSPI rows, False for KOSDAQ

    result = single_sort(
        inputsortvar = independentvar,
        inputvar     = dsf,
        ff3factor    = mff3factor,
        begdate      = '1998-01-01',
        enddate      = '2020-05-31',
        numPort      = 5,
        sortvar      = 'pc1',
        var          = 'exret',
        J            = 0,
        K            = 1,
        lag          = 6,
        breakpoint_mask = kospi_mask
        )
"""
def single_sort(inputsortvar, inputvar, ff3factor, begdate, enddate,
                numPort, sortvar, var, J, K, lag=6,
                date_col='date', groupby='symbol',
                breakpoint_mask=None, var_is_return=True):
    
    import pandas as pd
    import numpy as np
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    from statsmodels.stats.sandwich_covariance import cov_hac
    from scipy import stats as _stats

    begdate = pd.Timestamp(begdate)
    enddate = pd.Timestamp(enddate)

    required_cols = {'subperiod', 'weight', sortvar, date_col, groupby}
    missing = required_cols - set(inputsortvar.columns)
    if missing:
        raise ValueError(
            f"inputsortvar is missing required column(s): {missing}. "
            "Rename your columns to match before calling single_sort "
            "('subperiod' and 'weight' must be exact names)."
        )

    # -------------------------------------------------------
    # Step 1: Breakpoint-based portfolio assignment, INCLUSIVE
    # on both sides of each portfolio's range.
    # ------------------------------------------------------------
    # For numPort portfolios, use numPort-1 breakpoints (percentiles).
    # Breakpoints may be computed on a SUBSET (breakpoint_mask), then
    # applied to the FULL universe — matching the Fama-French convention.
    #
    # Both-sides-inclusive avoids empty portfolios when breakpoints tie
    # (e.g. 30th and 40th pct both = 0, which would leave no entity
    # satisfying "> 0 and <= 0"). Tradeoff: an entity landing exactly on
    # a breakpoint qualifies for TWO adjacent portfolios and appears as
    # TWO ROWS below — preferred over a zero-entity portfolio.
    #
    # CRITICAL: because of this, every later groupby/merge/dedup keyed on
    # (symbol, form_date) MUST also include port_{sortvar}, or a tied
    # entity's two legitimate rows silently collapse into one, dropping
    # it from one of its two portfolios with no error raised.
    # -------------------------------------------------------
    df = inputsortvar[inputsortvar['subperiod'].notna()].copy()
    df = df.sort_values(['subperiod', date_col, groupby])

    if breakpoint_mask is None:
        bp_mask = pd.Series(True, index=df.index)
    else:
        bp_mask = breakpoint_mask.reindex(df.index).fillna(False)

    quantile_levels = [i / numPort for i in range(1, numPort)]  # numPort-1 breakpoints

    bp_source = df[bp_mask]
    breaks = (
        bp_source.groupby(['subperiod', date_col])[sortvar]
        .quantile(quantile_levels)
        .unstack(level=-1)
    )
    breaks.columns = [f'_bp{i}' for i in range(1, numPort)]
    breaks = breaks.reset_index()

    df = df.merge(breaks, on=['subperiod', date_col], how='left')

    bp_cols = [f'_bp{i}' for i in range(1, numPort)]
    lower_bounds = [-np.inf] + [df[c] for c in bp_cols]
    upper_bounds = [df[c] for c in bp_cols] + [np.inf]

    # ------------------------------------------------------------
    # Diagnostic: NaN comparisons are always False, so missing sortvar
    # or breakpoint values silently exclude entities/dates from every
    # portfolio. This surfaces whether that's happening.
    # ------------------------------------------------------------
    dates_with_missing_breakpoints = breaks[breaks[bp_cols].isna().any(axis=1)]
    print(len(dates_with_missing_breakpoints), 'dates with at least one missing breakpoint')

    n_missing_sortvar = df[sortvar].isna().sum()
    print(n_missing_sortvar, 'rows with missing sortvar (excluded from all portfolios)')

    port_frames = []
    for k in range(1, numPort + 1):
        lo, hi = lower_bounds[k - 1], upper_bounds[k - 1]
        mask = (df[sortvar] >= lo) & (df[sortvar] <= hi)
        sub = df[mask].copy()
        sub[f'port_{sortvar}'] = k
        port_frames.append(sub)

    df = pd.concat(port_frames, ignore_index=True).drop(columns=bp_cols)
    # To verify the double-counting is behaving as expected after any
    # change to this step, run:
    #   dupes = df.groupby([subperiod, date_col, groupby]).size()
    #   print('entities appearing in >1 portfolio:', (dupes > 1).sum())
    #   print('total extra rows from duplication:', (dupes - 1)[dupes > 1].sum()) # should equal (len(df) - len(inputsortvar_pre_step))

    # -------------------------------------------------------
    # Step 2: Assign holding period dates
    # -------------------------------------------------------
    # HDATE1: J+1 months forward, snapped to the 1st of that month
    df['HDATE1'] = (df[date_col] + pd.DateOffset(months=J + 1)).values.astype('datetime64[M]').astype('datetime64[ns]')
    # HDATE2: J+K months forward, snapped to the LAST day of that month
    df['HDATE2'] = (df[date_col] + pd.DateOffset(months=J + K)) + pd.offsets.MonthEnd(0)
    df = df.rename(columns={date_col: 'form_date'})
    
    # -------------------------------------------------------
    # Diagnostic: number of stocks assigned to each portfolio AT FORMATION
    # ------------------------------------------------------------
    port_counts = df.groupby(['subperiod', 'form_date', f'port_{sortvar}'])[groupby] \
        .nunique().reset_index().rename(columns={groupby: 'n_stocks'})

    n_firms = port_counts.pivot_table(
        index=['form_date', 'subperiod'], columns=f'port_{sortvar}', values='n_stocks', aggfunc='sum'
    )
    n_firms.columns = [f'port_{sortvar}_{k}' for k in n_firms.columns]
    n_firms = n_firms.reset_index()
    # -------------------------------------------------------

    # -------------------------------------------------------
    # Step 3: Merge with return data within holding period
    # (searchsorted-based range join — avoids a cartesian merge;
    # a given entity may now legitimately appear under more than
    # one port_{sortvar} for the same form_date, due to inclusive
    # breakpoint boundaries from Step 1 — this is expected.)
    # -------------------------------------------------------
    def range_join(bounds_df, data_df, id_col, dcol):
        """
        For each row in df (with HDATE1/HDATE2 bounds), extract all rows from
        inputvar (any frequency — daily or monthly) whose date falls within
        [HDATE1, HDATE2], for the same identifier — without a cartesian merge.
        """
        data_sorted = data_df.sort_values([id_col, dcol]).reset_index(drop=True)
        bounds_grouped = {k: v for k, v in bounds_df.groupby(id_col)}

        results = []
        for entity, sub in data_sorted.groupby(id_col):
            b_sym = bounds_grouped.get(entity)
            if b_sym is None or b_sym.empty:
                continue

            dates = sub[dcol].values
            starts = np.searchsorted(dates, b_sym['HDATE1'].values, side='left')
            ends = np.searchsorted(dates, b_sym['HDATE2'].values, side='right')
            counts = ends - starts
            mask = counts > 0
            if not mask.any():
                continue

            starts_v, ends_v, counts_v = starts[mask], ends[mask], counts[mask]
            b_sym_valid = b_sym.loc[mask].reset_index(drop=True)

            idx = np.concatenate([np.arange(s, e) for s, e in zip(starts_v, ends_v)])
            block = sub.iloc[idx].copy()

            for col in b_sym_valid.columns:
                if col not in block.columns:
                    block[col] = np.repeat(b_sym_valid[col].values, counts_v)

            results.append(block)

        return pd.concat(results, ignore_index=True) if results else data_df.iloc[0:0]

    merged = range_join(df, inputvar, groupby, date_col)

    merged = merged.drop(columns=['HDATE1', 'HDATE2'])

    merged = merged.drop_duplicates(
        subset=['subperiod', date_col, f'port_{sortvar}', 'form_date', groupby]
    )

    # -------------------------------------------------------
    # Collapse multiple daily observations down to ONE value per
    # (symbol, form_date, port_{sortvar}, calendar month), before
    # EW/VW averaging. Compound if var is a return, average otherwise —
    # compounding a non-return variable would be economically meaningless.
    # -------------------------------------------------------
    merged['_ym'] = merged[date_col].dt.year * 100 + merged[date_col].dt.month  # fast integer grouping key

    merged = merged.sort_values([groupby, 'form_date', f'port_{sortvar}', '_ym', date_col])
    grp = merged.groupby([groupby, 'form_date', f'port_{sortvar}', '_ym'])

    if var_is_return:
        # var is assumed already on a PERCENT scale (1.5 = 1.5%),
        # Verify inputvar[var].describe() yourself before calling
        # this function — this function does NOT rescale var.
        merged['_log1p_var'] = np.log1p(merged[var] / 100)
        merged[var] = 100 * (np.exp(grp['_log1p_var'].transform('sum')) - 1)
        merged = merged.drop(columns=['_log1p_var'])
    else:
        merged[var] = grp[var].transform('mean')

    # Represent each calendar month by its last trading day's row, then
    # snap date_col to month-end below so it aligns with ff3factor's
    # month-end convention for the later merge.
    merged['_last_date'] = grp[date_col].transform('max')
    merged = merged[merged[date_col] == merged['_last_date']].copy()
    merged = merged.drop(columns=['_last_date'])
    merged[date_col] = merged[date_col] + pd.offsets.MonthEnd(0) # snap to true month-end

    merged = merged.drop_duplicates(subset=[groupby, date_col,'form_date', f'port_{sortvar}']).copy()

    # -------------------------------------------------------
    # Step 4: Calculate EW and VW portfolio returns
    # -------------------------------------------------------
    merged = merged[
        (merged[date_col] >= begdate) &
        (merged[date_col] <= enddate)]

    ew = merged.groupby(['subperiod', date_col, f'port_{sortvar}', 'form_date'])[var] \
        .mean().reset_index()
    ew['weight_type'] = 'EW'

    # ------------------------------------------------------------
    # VW average must sum weight only over entities where BOTH weight
    # and var are non-missing for that period — not just skip NaN in
    # the numerator while still including that entity's weight in
    # the denominator.
    # ------------------------------------------------------------
    valid_vw = merged[merged[var].notna() & merged['weight'].notna()].copy()
    valid_vw['_w_ret'] = valid_vw[var] * valid_vw['weight']
    vw = valid_vw.groupby(['subperiod', date_col, f'port_{sortvar}', 'form_date']) \
        .agg(_w_ret_sum=('_w_ret', 'sum'), _w_sum=('weight', 'sum')).reset_index()
    vw[var] = vw['_w_ret_sum'] / vw['_w_sum']
    vw = vw.drop(columns=['_w_ret_sum', '_w_sum'])
    vw['weight_type'] = 'VW'

    ds1 = pd.concat([ew, vw], ignore_index=True)

    # collapse across OVERLAPPING formation
    # cohorts, down to one return per (subperiod, weight_type, date, port).
    ewdat = ds1.groupby(['subperiod', 'weight_type', date_col, f'port_{sortvar}'])[var] \
        .mean().reset_index()

    # -------------------------------------------------------
    # Step 5: Calculate long-short portfolio
    # -------------------------------------------------------
    ls_list = []
    for (sp, wt, dt), group in ewdat.groupby(['subperiod', 'weight_type', date_col]):
        low  = group[group[f'port_{sortvar}'] == 1][var].values
        high = group[group[f'port_{sortvar}'] == numPort][var].values
        if len(low) > 0 and len(high) > 0:
            ls_ret = high[0] - low[0]
            ls_list.append({
                'subperiod'       : sp,
                'weight_type'     : wt,
                date_col          : dt,
                f'port_{sortvar}' : 99,
                var               : ls_ret
            })

    ewdat = pd.concat([ewdat, pd.DataFrame(ls_list)], ignore_index=True)

    ff3factor = ff3factor.copy()
    ff3factor[date_col] = ff3factor[date_col] + pd.offsets.MonthEnd(0) # Ensure ff3factor's date is snapped to calendar month-end

    ewdat = ewdat.merge(ff3factor, on=date_col, how='left')

    # -------------------------------------------------------
    # Step 6: Newey-West estimates
    # -------------------------------------------------------
    def nw_mean(series, lag):
        series = series.dropna()
        T = len(series)
        if T < 3:
            return np.nan, np.nan, np.nan
        mean = series.mean()
        model = OLS(series.values, np.ones(T)).fit()
        nw_cov = cov_hac(model, nlags=lag)
        se = np.sqrt(nw_cov[0, 0])
        tval = mean / se if se > 0 else np.nan
        pval = 2 * (1 - _stats.t.cdf(abs(tval), df=T - 1)) if not np.isnan(tval) else np.nan
        return mean, tval, pval

    def nw_regression(y, X, lag):
        """
        Returns {'alpha': (est, t, p), factor1: (est, t, p), ...}
        for ANY number of factor columns in X. Returns None if
        there isn't enough non-missing data to estimate.
        """
        data = pd.concat([y, X], axis=1).dropna()
        if len(data) < X.shape[1] + 2:
            return None
        cols = X.columns.tolist()
        y_ = data.iloc[:, 0].values
        X_ = add_constant(data[cols].values)
        T = len(y_)
        fit = OLS(y_, X_).fit()
        nw_cov = cov_hac(fit, nlags=lag)
        se = np.sqrt(np.diag(nw_cov))
        params = fit.params
        tvals = params / np.where(se > 0, se, np.nan)
        pvals = 2 * (1 - _stats.t.cdf(np.abs(tvals), df=T - 1))

        out = {'alpha': (params[0], tvals[0], pvals[0])}
        for i, col in enumerate(cols, start=1):
            out[col] = (params[i], tvals[i], pvals[i])
        return out

    def format_param(est, tval, pval):
        if est is None or (isinstance(est, float) and np.isnan(est)):
            return '', ''
        stars = ''
        if pval < 0.01:   stars = '***'
        elif pval < 0.05: stars = '** '
        elif pval < 0.1:  stars = '*  '
        return f'{est:.2f}{stars}', f'({tval:.2f})'

    # -------------------------------------------------------
    # Step 6b: Define factor models — extend/edit this dict freely.
    # A model is AUTOMATICALLY SKIPPED if any of its listed factor
    # columns aren't present in ewdat (e.g. drop 'FFC' entirely if
    # 'UMD' was never merged into ff3factor — no code change needed).
    # -------------------------------------------------------
    factor_models = {
        'CAPM': ['rmrf'],
        'FF':   ['rmrf', 'SMB', 'HML'],
        'FFC':  ['rmrf', 'SMB', 'HML', 'UMD'],
    }
    available_cols = set(ewdat.columns)
    factor_models = {
        name: factors for name, factors in factor_models.items()
        if all(f in available_cols for f in factors)
    }

    # -------------------------------------------------------
    # Step 7: Build TABLE output — rows = (Model,
    # Coefficient), columns = portfolio 1..numPort + long-short
    # (labeled 'numPort-1'). Built separately per (subperiod,
    # weight_type). Estimate and t-stat are stacked as two rows 
    # (estimate, then t-stat in parentheses directly below).
    # -------------------------------------------------------
    ls_label = f'{numPort}-1'
    port_values = list(range(1, numPort + 1)) + [99]
    port_labels = {k: str(k) for k in range(1, numPort + 1)}
    port_labels[99] = ls_label
    port_col_order = [port_labels[k] for k in port_values]

    final_rows = []

    for (sp, wt), grp_all in ewdat.groupby(['subperiod', 'weight_type']):

        # ---- Excess return row (simple NW mean, no factor model) ----
        est_row = {'subperiod': sp, 'weight_type': wt, 'Model': 'Excess return', 'Coefficient': 'Excess return'}
        t_row   = {'subperiod': sp, 'weight_type': wt, 'Model': 'Excess return', 'Coefficient': ''}
        for port in port_values:
            g = grp_all[grp_all[f'port_{sortvar}'] == port].sort_values(date_col)
            est, tval, pval = nw_mean(g[var], lag)
            est_str, t_str = format_param(est, tval, pval)
            est_row[port_labels[port]] = est_str
            t_row[port_labels[port]] = t_str
        final_rows.append(est_row)
        final_rows.append(t_row)

        # ---- One (Model, Coefficient) row-pair per factor / alpha ----
        for model_name, factors in factor_models.items():
            coef_names = ['alpha'] + factors
            coef_est = {c: {'subperiod': sp, 'weight_type': wt, 'Model': model_name, 'Coefficient': c} for c in coef_names}
            coef_t   = {c: {'subperiod': sp, 'weight_type': wt, 'Model': model_name, 'Coefficient': ''} for c in coef_names}

            for port in port_values:
                g = grp_all[grp_all[f'port_{sortvar}'] == port].sort_values(date_col)
                res = nw_regression(g[var], g[factors], lag)
                for c in coef_names:
                    if res is None or c not in res:
                        est_str, t_str = '', ''
                    else:
                        est, tval, pval = res[c]
                        est_str, t_str = format_param(est, tval, pval)
                    coef_est[c][port_labels[port]] = est_str
                    coef_t[c][port_labels[port]] = t_str

            for c in coef_names:
                final_rows.append(coef_est[c])
                final_rows.append(coef_t[c])

    result = pd.DataFrame(final_rows)
    result = result[['subperiod', 'weight_type', 'Model', 'Coefficient'] + port_col_order]
    return result
