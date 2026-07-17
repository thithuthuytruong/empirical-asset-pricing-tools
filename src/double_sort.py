"""
double_sort: Bivariate (double) sort portfolio analysis — independent or
dependent, controlled by the `dependent_sort` boolean. Builds on the same
conventions as single_sort (inclusive-both-sides breakpoints, J/K holding
periods, log-compounding vs. averaging, EW/VW, Newey-West).

INDEPENDENT (dependent_sort=False): breakpoints for sortvar1 (X1) and
sortvar2 (X2) are each computed MARGINALLY — independently of one another
— then portfolios are formed as the nP1 x nP2 intersection of the two
groupings. Output includes the main grid, the X1 diff/avg (per X2 group),
the X2 diff/avg (per X1 group), and the four corner combinations
(Diff x Diff, Diff x Avg, Avg x Diff, Avg x Avg).

DEPENDENT (dependent_sort=True): sortvar1 (X1) is the CONTROL variable —
argument order matters. X1 breakpoints are computed first (marginally),
then X2 breakpoints are computed SEPARATELY WITHIN EACH X1 group
(conditional breakpoints). Because dependent-sort is only designed to
assess the X2-Y relation after controlling for X1, only the main grid,
the X2 diff, and the X2 avg (each per X1 group) are computed — no X1
diff/avg, no corner cells.

Parameters:
    inputsortvar, inputvar, ff3factor, begdate, enddate, var, J, K, lag,
    date_col, groupby, breakpoint_mask, var_is_return : same meaning as
        in single_sort — see that docstring for the full rationale on
        each. inputsortvar must additionally contain sortvar1 and
        sortvar2 (in place of single_sort's single sortvar).
    sortvar1, numPort1 : first sort variable and its number of groups.
        In dependent-sort mode this is the CONTROL variable.
    sortvar2, numPort2 : second sort variable and its number of groups.
        In dependent-sort mode this is the variable of interest, sorted
        WITHIN each sortvar1 group.
    dependent_sort : bool (default: False) — False = independent sort,
        True = dependent sort (sortvar1 as control, per above).
    alpha_models : tuple of str (default: ('FFC',)) — which factor
        model(s)' ALPHA to include as additional row-blocks, alongside
        Excess return. Factor loadings (rmrf/SMB/HML/UMD) themselves are
        never shown. Models not available (e.g. 'FFC' with no UMD column
        in ff3factor) are silently skipped.
    port1_name, port2_name : str (default: None) — display prefix for
        column/row group labels (e.g. 'β', 'MktCap'). Defaults to
        sortvar1 / sortvar2 if not given.

    Breakpoint universe: only entities with BOTH sortvar1 and sortvar2
    non-missing (and satisfying breakpoint_mask, if given) are used to
    COMPUTE breakpoints for either variable. Breakpoints are still APPLIED to
    every row regardless.

    As in single_sort, breakpoint ranges are INCLUSIVE on both sides —
    see single_sort's docstring for the full tie-handling rationale.
    This can compound across the two dimensions in dependent-sort mode.

Output:
    A WIDE DataFrame, one block of rows per (subperiod, weight_type),
    concatenated vertically:
        subperiod, weight_type, {port2_name}, Coefficient,
        {port1_name} 1, ..., {port1_name} numPort1,
        [{port1_name} Diff, {port1_name} Avg]   (independent-sort only)
    Each (port2 group, Coefficient) pair occupies TWO rows: the
    estimate, then the Newey-West t-stat (in parentheses) directly
    below with a blank Coefficient label. Rows run through port2 groups
    1..numPort2, then 'Diff' (high-low), then 'Avg' (average of
    averages).

Example:
    kospi_mask = indepvar['histexch'] == kospi_code   # True for KOSPI rows, False for KOSDAQ
    result = double_sort(
        inputsortvar = indepvar,
        inputvar     = tmp1,
        ff3factor    = mff3factor,
        begdate      = start_period,
        enddate      = end_period,
        numPort1     = 3,
        numPort2     = 4,
        sortvar1     = 'beta',
        sortvar2     = 'mktcap',
        var          = 'exret',
        J            = 0,
        K            = 1,
        dependent_sort  = False,
        alpha_models    = ('FFC',),
        breakpoint_mask = kospi_mask,
        )
"""
def double_sort(inputsortvar, inputvar, ff3factor, begdate, enddate,
                numPort1, numPort2, sortvar1, sortvar2, var, J, K,
                dependent_sort=False, alpha_models=('FFC',),
                port1_name=None, port2_name=None, lag=6,
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
    port1_name = port1_name or sortvar1
    port2_name = port2_name or sortvar2

    required_cols = {'subperiod', 'weight', sortvar1, sortvar2, date_col, groupby}
    missing = required_cols - set(inputsortvar.columns)
    if missing:
        raise ValueError(
            f"inputsortvar is missing required column(s): {missing}. "
            "'subperiod' and 'weight' must be exact names."
        )

    df = inputsortvar[inputsortvar['subperiod'].notna()].copy()
    df = df.sort_values(['subperiod', date_col, groupby])

    # ------------------------------------------------------------
    # Bivariate breakpoint universe: BOTH sort variables (and
    # breakpoint_mask, if given) must be valid to use an entity for
    # computing breakpoints on EITHER variable. Stored as a COLUMN (not a free-standing
    # Series) so it survives the row expansion from tie-inclusive
    # breakpoint assignment below without needing to be re-aligned to
    # a fresh post-concat index later (a Series.reindex against an
    # expanded frame's new RangeIndex would silently return all-False).
    # ------------------------------------------------------------
    if breakpoint_mask is None:
        df['_bp_eligible'] = True
    else:
        df['_bp_eligible'] = breakpoint_mask.reindex(df.index).fillna(False)
    df['_bp_eligible'] = df['_bp_eligible'] & df[sortvar1].notna() & df[sortvar2].notna()

    def assign_groups_inclusive(data, sortvar, numPort, group_keys, out_col):
        """
        Shared breakpoint + inclusive-both-sides assignment (see
        single_sort Step 1 for the full tie-handling rationale).
        group_keys are the columns breakpoints are computed PER — e.g.
        ['subperiod', date_col] for a marginal sort, or
        ['subperiod', date_col, 'port1'] for a sort computed within
        existing port1 groups (dependent-sort's X2 step). Returns
        `data` with out_col assigned, EXPANDED with extra rows for
        entities tied on a breakpoint.
        """
        quantile_levels = [i / numPort for i in range(1, numPort)]
        bp_source = data[data['_bp_eligible']]
        breaks = (
            bp_source.groupby(group_keys)[sortvar]
            .quantile(quantile_levels)
            .unstack(level=-1)
        )
        bp_cols = [f'_bp{i}' for i in range(1, numPort)]
        breaks.columns = bp_cols
        breaks = breaks.reset_index()

        data = data.merge(breaks, on=group_keys, how='left')
        lower_bounds = [-np.inf] + [data[c] for c in bp_cols]
        upper_bounds = [data[c] for c in bp_cols] + [np.inf]

        frames = []
        for k in range(1, numPort + 1):
            lo, hi = lower_bounds[k - 1], upper_bounds[k - 1]
            mask = (data[sortvar] >= lo) & (data[sortvar] <= hi)
            sub = data[mask].copy()
            sub[out_col] = k
            frames.append(sub)
        return pd.concat(frames, ignore_index=True).drop(columns=bp_cols)

    # -------------------------------------------------------
    # Step 1: X1 groups — ALWAYS marginal. X1 is the control variable in
    # dependent-sort mode and one of two independent variables in
    # independent-sort mode; either way its own breakpoints never depend
    # on X2.
    # -------------------------------------------------------
    df = assign_groups_inclusive(df, sortvar1, numPort1, ['subperiod', date_col], 'port1')

    # -------------------------------------------------------
    # Step 2: X2 groups.
    #   INDEPENDENT: breakpoints computed MARGINALLY — blind to port1,
    #     same as X1. This is what produces unequal portfolio sizes when
    #     X1 and X2 are correlated.
    #   DEPENDENT: breakpoints computed WITHIN each port1 group, so an
    #     entity's X2 group is only ever relative to peers sharing its
    #     X1 group.
    # -------------------------------------------------------
    group_keys2 = ['subperiod', date_col, 'port1'] if dependent_sort else ['subperiod', date_col]
    df = assign_groups_inclusive(df, sortvar2, numPort2, group_keys2, 'port2')
    df = df.drop(columns=['_bp_eligible'])

    n_obs = df.groupby(['port1', 'port2']).size()
    print(n_obs.min(), 'min /', n_obs.max(), 'max obs per (port1,port2) cell at formation')

    # -------------------------------------------------------
    # Step 3: Holding period dates (identical to single_sort)
    # -------------------------------------------------------
    df['HDATE1'] = (df[date_col] + pd.DateOffset(months=J + 1)).values.astype('datetime64[M]').astype('datetime64[ns]')
    df['HDATE2'] = (df[date_col] + pd.DateOffset(months=J + K)) + pd.offsets.MonthEnd(0)
    df = df.rename(columns={date_col: 'form_date'})

    # -------------------------------------------------------
    # Step 4: range-join with return data (identical mechanics to
    # single_sort's range_join — see there for the full rationale)
    # -------------------------------------------------------
    def range_join(bounds_df, data_df, id_col, dcol):
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
        subset=['subperiod', date_col, 'port1', 'port2', 'form_date', groupby]
    )

    # -------------------------------------------------------
    # Step 5: collapse daily obs to ONE value per (symbol, form_date,
    # port1, port2, calendar month) — compound if var_is_return, else
    # average (see single_sort for the full rationale).
    # -------------------------------------------------------
    merged['_ym'] = merged[date_col].dt.year * 100 + merged[date_col].dt.month
    merged = merged.sort_values([groupby, 'form_date', 'port1', 'port2', '_ym', date_col])
    grp = merged.groupby([groupby, 'form_date', 'port1', 'port2', '_ym'])

    if var_is_return:
        merged['_log1p_var'] = np.log1p(merged[var] / 100)
        merged[var] = 100 * (np.exp(grp['_log1p_var'].transform('sum')) - 1)
        merged = merged.drop(columns=['_log1p_var'])
    else:
        merged[var] = grp[var].transform('mean')

    merged['_last_date'] = grp[date_col].transform('max')
    merged = merged[merged[date_col] == merged['_last_date']].copy()
    merged = merged.drop(columns=['_last_date'])
    merged[date_col] = merged[date_col] + pd.offsets.MonthEnd(0)
    merged = merged.drop_duplicates(subset=[groupby, date_col, 'form_date', 'port1', 'port2']).copy()

    # -------------------------------------------------------
    # Step 6: EW / VW portfolio returns (identical mechanics to
    # single_sort Step 4)
    # -------------------------------------------------------
    merged = merged[(merged[date_col] >= begdate) & (merged[date_col] <= enddate)]

    ew = merged.groupby(['subperiod', date_col, 'port1', 'port2', 'form_date'])[var].mean().reset_index()
    ew['weight_type'] = 'EW'

    valid_vw = merged[merged[var].notna() & merged['weight'].notna()].copy()
    valid_vw['_w_ret'] = valid_vw[var] * valid_vw['weight']
    vw = valid_vw.groupby(['subperiod', date_col, 'port1', 'port2', 'form_date']) \
        .agg(_w_ret_sum=('_w_ret', 'sum'), _w_sum=('weight', 'sum')).reset_index()
    vw[var] = vw['_w_ret_sum'] / vw['_w_sum']
    vw = vw.drop(columns=['_w_ret_sum', '_w_sum'])
    vw['weight_type'] = 'VW'

    ds1 = pd.concat([ew, vw], ignore_index=True)
    # collapse across OVERLAPPING formation cohorts, down to one return
    # per (subperiod, weight_type, date, port1, port2)
    ewdat = ds1.groupby(['subperiod', 'weight_type', date_col, 'port1', 'port2'])[var].mean().reset_index()
    ewdat['port1'] = ewdat['port1'].astype(str)
    ewdat['port2'] = ewdat['port2'].astype(str)

    p1_labels = [str(i) for i in range(1, numPort1 + 1)]
    p2_labels = [str(i) for i in range(1, numPort2 + 1)]

    # -------------------------------------------------------
    # Step 7: Diff and Avg cells
    #   X2 diff/avg (row-level: high-low / mean across port2, per
    #     port1) — computed for BOTH independent and dependent sort,
    #     since this is the "assess X2 vs Y controlling for X1" result
    #     dependent-sort exists for.
    #   X1 diff/avg (column-level) and the four corner cells — computed
    #     for INDEPENDENT sort only. Dependent-sort's X1 groups aren't a
    #     meaningful comparison set for these, since X2's own breakpoints
    #     already differ across X1 groups by construction.
    # -------------------------------------------------------
    extra = []
    for (sp, wt, dt), g in ewdat.groupby(['subperiod', 'weight_type', date_col]):
        pivot = g.pivot_table(index='port2', columns='port1', values=var, aggfunc='mean')
        pivot = pivot.reindex(index=p2_labels, columns=p1_labels)

        x2diff = pivot.loc[p2_labels[-1]] - pivot.loc[p2_labels[0]]   # indexed by port1
        x2avg = pivot.mean(axis=0, skipna=True)                        # indexed by port1
        for p1 in p1_labels:
            extra.append({'subperiod': sp, 'weight_type': wt, date_col: dt,
                           'port1': p1, 'port2': 'Diff', var: x2diff[p1]})
            extra.append({'subperiod': sp, 'weight_type': wt, date_col: dt,
                           'port1': p1, 'port2': 'Avg', var: x2avg[p1]})

        if not dependent_sort:
            x1diff = pivot[p1_labels[-1]] - pivot[p1_labels[0]]        # indexed by port2
            x1avg = pivot.mean(axis=1, skipna=True)                     # indexed by port2
            for p2 in p2_labels:
                extra.append({'subperiod': sp, 'weight_type': wt, date_col: dt,
                               'port1': 'Diff', 'port2': p2, var: x1diff[p2]})
                extra.append({'subperiod': sp, 'weight_type': wt, date_col: dt,
                               'port1': 'Avg', 'port2': p2, var: x1avg[p2]})

            # Diff-in-diff corner: (D - C) - (B - A) = D - C - B + A
            try:
                A = pivot.loc[p2_labels[0], p1_labels[0]]
                B = pivot.loc[p2_labels[0], p1_labels[-1]]
                C = pivot.loc[p2_labels[-1], p1_labels[0]]
                D = pivot.loc[p2_labels[-1], p1_labels[-1]]
                extra.append({'subperiod': sp, 'weight_type': wt, date_col: dt,
                               'port1': 'Diff', 'port2': 'Diff', var: D - C - B + A})
            except KeyError:
                pass  # a corner cell is missing this period — skip

            # Mixed Diff/Avg corners
            extra.append({'subperiod': sp, 'weight_type': wt, date_col: dt,
                           'port1': 'Avg', 'port2': 'Diff', var: x2diff.mean()})
            extra.append({'subperiod': sp, 'weight_type': wt, date_col: dt,
                           'port1': 'Diff', 'port2': 'Avg', var: x1diff.mean()})
            extra.append({'subperiod': sp, 'weight_type': wt, date_col: dt,
                           'port1': 'Avg', 'port2': 'Avg', var: np.nanmean(pivot.values)})

    ewdat = pd.concat([ewdat, pd.DataFrame(extra)], ignore_index=True)

    ff3factor = ff3factor.copy()
    ff3factor[date_col] = ff3factor[date_col] + pd.offsets.MonthEnd(0)
    ewdat = ewdat.merge(ff3factor, on=date_col, how='left')

    # -------------------------------------------------------
    # Step 8: Newey-West estimates (identical to single_sort Step 6)
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

    factor_models = {
        'CAPM': ['rmrf'],
        'FF':   ['rmrf', 'SMB', 'HML'],
        'FFC':  ['rmrf', 'SMB', 'HML', 'UMD'],
    }
    available_cols = set(ewdat.columns)
    factor_models = {name: f for name, f in factor_models.items() if all(c in available_cols for c in f)}
    alpha_models = tuple(m for m in alpha_models if m in factor_models)  # drop unavailable models silently

    # -------------------------------------------------------
    # Step 9: Newey-West estimate/t-stat for every (subperiod,
    # weight_type, cell, row-block) combination we'll display —
    # 'Excess return' plus each requested model's alpha only (no
    # factor loadings).
    # -------------------------------------------------------
    row_blocks = [('Excess return', 'Excess return', 'Excess return')]
    for m in alpha_models:
        row_blocks.append((m, 'alpha', f'{m} \u03b1'))

    cell_keys = ewdat[['port1', 'port2']].drop_duplicates()
    est_lookup = {}   # (sp, wt, model, coef, p1, p2) -> (est_str, t_str)

    for (sp, wt), grp_all in ewdat.groupby(['subperiod', 'weight_type']):
        for p1, p2 in cell_keys.itertuples(index=False):
            g = grp_all[(grp_all['port1'] == p1) & (grp_all['port2'] == p2)].sort_values(date_col)

            est, tval, pval = nw_mean(g[var], lag)
            est_lookup[(sp, wt, 'Excess return', 'Excess return', p1, p2)] = format_param(est, tval, pval)

            for model_name in alpha_models:
                factors = factor_models[model_name]
                res = nw_regression(g[var], g[factors], lag)
                if res is None or 'alpha' not in res:
                    est_lookup[(sp, wt, model_name, 'alpha', p1, p2)] = ('', '')
                else:
                    est_lookup[(sp, wt, model_name, 'alpha', p1, p2)] = format_param(*res['alpha'])

    # -------------------------------------------------------
    # Step 10: assemble the WIDE table. port1
    # columns include Diff/Avg only for independent-sort; port2 rows
    # always include Diff/Avg (X2 diff/avg apply in both modes).
    #
    # Display label for the 'Diff' sentinel is 'numPort-1' (e.g. '3-1'),
    # not the bare word 'Diff' — spelling out which two groups were
    # differenced (and that it's high minus low) lets a reader catch
    # the direction at a glance, matching single_sort's own long-short
    # column convention (f'{numPort}-1').
    # -------------------------------------------------------
    def disp_group(val, numPort):
        if val == 'Diff':
            return f'{numPort}-1'
        return val   # numeric group label or 'Avg'

    port1_order = list(p1_labels) + (['Diff', 'Avg'] if not dependent_sort else [])
    port2_order = list(p2_labels) + ['Diff', 'Avg']

    wide_rows = []
    for sp in sorted(ewdat['subperiod'].unique()):
        for wt in ['EW', 'VW']:
            for p2 in port2_order:
                p2_disp = disp_group(p2, numPort2)
                for model_name, coef, disp_label in row_blocks:
                    est_row = {'subperiod': sp, 'weight_type': wt,
                               port2_name: f'{port2_name} {p2_disp}', 'Coefficient': disp_label}
                    t_row = {'subperiod': sp, 'weight_type': wt,
                             port2_name: f'{port2_name} {p2_disp}', 'Coefficient': ''}
                    for p1 in port1_order:
                        p1_disp = disp_group(p1, numPort1)
                        est_str, t_str = est_lookup.get((sp, wt, model_name, coef, p1, p2), ('', ''))
                        est_row[f'{port1_name} {p1_disp}'] = est_str
                        t_row[f'{port1_name} {p1_disp}'] = t_str
                    wide_rows.append(est_row)
                    wide_rows.append(t_row)

    result = pd.DataFrame(wide_rows)
    col_order = ['subperiod', 'weight_type', port2_name, 'Coefficient'] + \
                [f'{port1_name} {disp_group(p1, numPort1)}' for p1 in port1_order]
    result = result[col_order]
    return result
