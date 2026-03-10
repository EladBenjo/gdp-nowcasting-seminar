import numpy as np
import pandas as pd

def build_high_freq_lags(daily_series, quarterly_index, num_lags):
    """
    For each quarterly date, extract the last `num_lags` daily values before that quarter end.
    """
    x_list = []
    for q_date in quarterly_index:
        relevant_days = daily_series[:q_date].dropna().tail(num_lags)
        if len(relevant_days) < num_lags:
            x_list.append([np.nan] * num_lags)
        else:
            x_list.append(relevant_days.values)
    return np.array(x_list, dtype=np.float64)