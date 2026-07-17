"""
fama_macbeth: Fama and MacBeth (1973) two-step cross-sectional regression.

STEP 1 (cross-sectional): for each date_col period t, run an OLS
regression of y on the independent variables of that specification
(with an intercept), using only that period's cross-section of
entities. This produces a time series of coefficients — an intercept
delta_0,t and a slope delta_k,t for each independent variable — along
with that period's R-squared, adjusted R-squared, and n.

STEP 2 (time-series): average each coefficient (and R2/Adj.R2/n) across
all periods. To test whether the average coefficient differs from zero,
its standard error is estimated with a Newey-West (1987) HAC correction
(same nw_mean approach used in single_sort/double_sort) rather than a
plain time-series standard error, since periodic FM coefficients are
often autocorrelated.

Multiple specifications (e.g. Table 6.3's univariate columns (1)-(3)
and multivariate column (4)) are run independently — a period's
regression for one specification does not see the others' variables —
and assembled side by side into one wide table.

SUBPERIODS: 'subperiod' MUST be an exact-named column in `data`,
matching single_sort's/double_sort's convention — used for robustness
checks across subsamples (e.g. pre/post a structural break). Each
distinct 'subperiod' value is run INDEPENDENTLY (steps 1 and 2 both
restart within each subperiod) and the results are stacked vertically
in the output, one block per subperiod. If you don't want to split
your sample, set every row to the same constant value before calling:
    data['subperiod'] = 1
Rows with a MISSING (NaN) 'subperiod' are DROPPED before any further
processing.

NOTE ON WINSORIZATION: per the source text, independent variables (and
sometimes the dependent variable, EXCEPT when it's a security return)
are usually winsorized before this procedure. That is NOT done here —
winsorize your columns in `data` before calling this function.

NOTE ON REGRESSION TYPE: step 1 here is always OLS. The text notes WLS,
logistic, probit, or multinomial models are equally valid substitutes
for step 1 in principle — swapping those in would require editing the
per-period regression call below; this implementation only covers OLS.

Parameters:
    data       : DataFrame — must contain date_col, y, 'subperiod', and
                 every variable named in any spec in `specs`.
    y          : str — dependent variable column name.
    specs      : list of lists of str — one inner list per
                 specification/column, each naming the independent
                 variable(s) for that column's per-period regression.
                 E.g. [['beta'], ['size'], ['bm'], ['beta','size','bm']]
                 reproduces Table 6.3's columns (1)-(4).
    spec_labels : list of str (default: None) — column labels, e.g.
                 ['(1)','(2)','(3)','(4)']. Defaults to '1','2',...
    date_col   : str — column used to group periods (default: 'date').
    lag        : int — Newey-West lag count for step 2 (default: 6,
                 matching Table 6.3's six lags).
    min_obs    : int (default: None) — minimum non-missing observations
                 required to run a given period's regression for a
                 given spec. Defaults to len(xvars) + 2 (minimum
                 degrees of freedom to estimate an intercept, every
                 slope, and have >0 residual df).
    coef_decimals : int (default: 2) — decimal places for coefficient/
                 intercept estimates (t-stats always use 2).
    r2_decimals   : int (default: 3) — decimal places for R2/Adj. R2
                 estimates, kept separate from coef_decimals since R2
                 values are typically small (0.011 reads better than
                 0.01).

Output:
    A DataFrame with columns ['subperiod', 'Coefficient'] + spec_labels,
    one block of rows per subperiod stacked vertically. Within each
    subperiod block:
        - one (estimate, t-stat) row-pair per variable that appears in
          ANY spec, in first-seen order — 'Intercept' always first.
          A spec that doesn't include a given variable leaves that
          cell blank, matching Table 6.3's blank cells. Estimates are
          starred by p-value (***/**/* at 1%/5%/10%), matching
          single_sort's/double_sort's display convention; t-stats sit
          directly below in parentheses with a blank Coefficient label.
        - 'R2' and 'Adj. R2' row-pairs: same treatment as a coefficient
          — the time-series average (Adj.) R-squared, with its own
          Newey-West t-stat/p-value/stars testing whether the average
          is reliably different from zero, and the t-stat directly
          below. Both are reported (SAS's macro carries R2 through
          alongside Adj. R2 as a side effect of not dropping _RSQ_
          before its GMM step; this does the same thing deliberately).
        - 'n' row: time-series average number of observations per
          period's cross-sectional regression (rounded), NOT a total
          across periods — matches Table 6.3's n row. No t-stat: "is
          average sample size different from zero" isn't a meaningful
          test, so n stays a single descriptive value.
        - 'First period' / 'Last period' / 'T' rows: the first date,
          last date, and COUNT of periods actually used in step 1 for
          that spec (i.e. periods with >= min_obs valid observations).
          T is the effective time-series length feeding every Newey-
          West estimate in that spec's column — it can differ across
          specs when min_obs excludes different periods for each, so
          it's reported per spec rather than once overall.

Example:
    result = fama_macbeth(
        data   = stockmonth,
        y      = 'ret_lead1',
        specs  = [['beta'], ['size'], ['bm'], ['beta', 'size', 'bm']],
        spec_labels = ['(1)', '(2)', '(3)', '(4)'],
        date_col = 'date',
        lag      = 6,
        )
"""
def fama_macbeth(data, y, specs, spec_labels=None, date_col='date', lag=6, min_obs=None,
                  coef_decimals=2, r2_decimals=3):

    import pandas as pd
    import numpy as np
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    from statsmodels.stats.sandwich_covariance import cov_hac
    from scipy import stats as _stats

    if spec_labels is None:
        spec_labels = [str(i + 1) for i in range(len(specs))]
    if len(spec_labels) != len(specs):
        raise ValueError("spec_labels must be the same length as specs.")

    all_needed = {date_col, y, 'subperiod'}
    for spec in specs:
        all_needed |= set(spec)
    missing = all_needed - set(data.columns)
    if missing:
        raise ValueError(
            f"data is missing required column(s): {missing}. "
            "'subperiod' must be an exact name — set data['subperiod'] = 1 "
            "if you don't want to split your sample."
        )

    data = data[data['subperiod'].notna()].copy()

    # -------------------------------------------------------
    # Step 1: per-period cross-sectional OLS, run SEPARATELY for each
    # spec (so one spec's missing values don't reduce another spec's
    # sample — matches Table 6.3's differing n across columns) and
    # SEPARATELY for each subperiod (a period's cross-section is drawn
    # only from rows sharing its subperiod value).
    # -------------------------------------------------------
    def run_spec(sub_data, xvars):
        need = min_obs if min_obs is not None else len(xvars) + 2
        records = []
        for dt, g in sub_data.groupby(date_col):
            sub = g[[y] + list(xvars)].dropna()
            if len(sub) < need:
                continue
            Y_ = sub[y].values
            X_ = add_constant(sub[list(xvars)].values)
            fit = OLS(Y_, X_).fit()
            rec = {date_col: dt, 'n': len(sub), 'r2': fit.rsquared, 'adj_r2': fit.rsquared_adj,
                   'Intercept': fit.params[0]}
            for i, xv in enumerate(xvars, start=1):
                rec[xv] = fit.params[i]
            records.append(rec)
        return pd.DataFrame(records)

    # -------------------------------------------------------
    # Step 2: Newey-West time-series mean of each coefficient — and,
    # the same way, of R2 and Adj. R2 (identical mechanics to
    # single_sort's/double_sort's nw_mean).
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

    def format_param(est, tval, pval, decimals=2):
        if est is None or (isinstance(est, float) and np.isnan(est)):
            return '', ''
        stars = ''
        if pval < 0.01:   stars = '***'
        elif pval < 0.05: stars = '** '
        elif pval < 0.1:  stars = '*  '
        return f'{est:.{decimals}f}{stars}', f'({tval:.2f})'

    all_vars = ['Intercept']
    for spec in specs:
        for v in spec:
            if v not in all_vars:
                all_vars.append(v)

    # -------------------------------------------------------
    # Run steps 1-2 independently within each subperiod, then assemble
    # one wide block per subperiod and stack them vertically.
    # -------------------------------------------------------
    rows = []
    for sp in sorted(data['subperiod'].unique()):
        sub_data = data[data['subperiod'] == sp]

        spec_results = []
        for xvars in specs:
            ts = run_spec(sub_data, xvars)
            coefs = {}
            for c in ['Intercept'] + list(xvars):
                coefs[c] = nw_mean(ts[c], lag) if c in ts.columns else (np.nan, np.nan, np.nan)
            spec_results.append({
                'coefs': coefs,
                'r2_stats': nw_mean(ts['r2'], lag) if len(ts) else (np.nan, np.nan, np.nan),
                'adj_r2_stats': nw_mean(ts['adj_r2'], lag) if len(ts) else (np.nan, np.nan, np.nan),
                'avg_n': ts['n'].mean() if len(ts) else np.nan,
                'first_date': ts[date_col].min() if len(ts) else None,
                'last_date': ts[date_col].max() if len(ts) else None,
                'T': len(ts),
            })

        for v in all_vars:
            est_row = {'subperiod': sp, 'Coefficient': v}
            t_row = {'subperiod': sp, 'Coefficient': ''}
            for label, res in zip(spec_labels, spec_results):
                if v in res['coefs']:
                    est, tval, pval = res['coefs'][v]
                    est_str, t_str = format_param(est, tval, pval, decimals=coef_decimals)
                else:
                    est_str, t_str = '', ''
                est_row[label] = est_str
                t_row[label] = t_str
            rows.append(est_row)
            rows.append(t_row)

        r2_est_row = {'subperiod': sp, 'Coefficient': 'R2'}
        r2_t_row = {'subperiod': sp, 'Coefficient': ''}
        adj_r2_est_row = {'subperiod': sp, 'Coefficient': 'Adj. R2'}
        adj_r2_t_row = {'subperiod': sp, 'Coefficient': ''}
        n_row = {'subperiod': sp, 'Coefficient': 'n'}
        first_row = {'subperiod': sp, 'Coefficient': 'First period'}
        last_row = {'subperiod': sp, 'Coefficient': 'Last period'}
        T_row = {'subperiod': sp, 'Coefficient': 'T'}
        for label, res in zip(spec_labels, spec_results):
            est, tval, pval = res['r2_stats']
            r2_est_row[label], r2_t_row[label] = format_param(est, tval, pval, decimals=r2_decimals)
            est, tval, pval = res['adj_r2_stats']
            adj_r2_est_row[label], adj_r2_t_row[label] = format_param(est, tval, pval, decimals=r2_decimals)
            n_row[label] = '' if np.isnan(res['avg_n']) else f"{res['avg_n']:.0f}"
            first_row[label] = '' if res['first_date'] is None else pd.Timestamp(res['first_date']).strftime('%Y-%m-%d')
            last_row[label] = '' if res['last_date'] is None else pd.Timestamp(res['last_date']).strftime('%Y-%m-%d')
            T_row[label] = str(res['T'])
        rows.append(r2_est_row)
        rows.append(r2_t_row)
        rows.append(adj_r2_est_row)
        rows.append(adj_r2_t_row)
        rows.append(n_row)
        rows.append(first_row)
        rows.append(last_row)
        rows.append(T_row)

    result = pd.DataFrame(rows)
    result = result[['subperiod', 'Coefficient'] + spec_labels]
    return result
