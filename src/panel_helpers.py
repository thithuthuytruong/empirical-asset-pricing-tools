"""
Shared panel-construction helpers used by make_fm_panel.py and
symbolrankdates.py.
"""

"""
make_sure_continuous_dates: Ensure all trading dates are present for each identifier.
Adds missing dates with NaN values for omitted dates.

Parameters:
    inputds    : DataFrame — input dataset
    identifier : str — column name for the identifier (default: 'symbol', can use 'permno')
    date_col   : str — column name for the date (default: 'date', can use 'datetime')

Output:
    DataFrame with continuous dates — missing dates are filled with NaN

Example:
    make_sure_continuous_dates(dsf_raw)                        # default: symbol, date
    make_sure_continuous_dates(dsf_raw, identifier='permno')   # using permno
    make_sure_continuous_dates(dsf_raw, date_col='datetime')   # using different date column
"""
def make_sure_continuous_dates(inputds, identifier='symbol', date_col='date'):
    import pandas as pd

    df = inputds.sort_values([identifier, date_col]).copy()

    firstandlastdates = df.groupby(identifier)[date_col].agg(['min', 'max']).reset_index()
    firstandlastdates.columns = [identifier, 'firstdate', 'lastdate']

    trading_dates = df[[date_col]].drop_duplicates()
    trading_dates = trading_dates.sort_values(date_col)

    filled = []
    for _, row in firstandlastdates.iterrows():
        dates_in_range = trading_dates[
            (trading_dates[date_col] >= row['firstdate']) &
            (trading_dates[date_col] <= row['lastdate'])
        ].copy()
        dates_in_range[identifier] = row[identifier]
        filled.append(dates_in_range)

    date_ranges = pd.concat(filled, ignore_index=True)

    result = date_ranges.merge(df, on=[identifier, date_col], how='left')
    result = result.sort_values([identifier, date_col]).reset_index(drop=True)

    return result
# -----------------------------------------------------------

"""
Lookahead_expand: Add future values of the next n periods of a specific variable.

Parameters:
    dataset  : DataFrame — input dataset
    groupby  : str — column name to group by (default: 'symbol', can use 'permno')
    lookn    : int — number of future periods to look ahead
    var      : str — variable to find future values for
    date_col : str — column name for the date (default: 'date', can use 'datetime')

Output:
    DataFrame with additional columns: var_next1, var_next2, ... var_nextN

Example:
    lookahead_expand(dsf_raw, lookn=3, var='ret')                       # default: symbol, date
    lookahead_expand(dsf_raw, groupby='permno', lookn=3, var='ret')     # using permno
    lookahead_expand(dsf_raw, lookn=5, var='closingprice')              # different variable
    lookahead_expand(dsf_raw, lookn=3, var='ret', date_col='datetime')  # different date column
"""
def lookahead_expand(dataset, groupby='symbol', lookn=1, var='ret', date_col='date'):

    df = dataset.sort_values([groupby, date_col]).copy()

    for j in range(1, lookn + 1):
        df[f'{var}_next{j}'] = df.groupby(groupby)[var].shift(-j)

    return df
# -----------------------------------------------------------
