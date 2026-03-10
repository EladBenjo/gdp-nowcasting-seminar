import pandas as pd
import numpy as np
import altair as alt
alt.data_transformers.enable("vegafusion")
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Optional, List, Union


def log_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes log-difference for each column in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with numeric values and datetime index.

    Returns:
        pd.DataFrame: Log-differenced DataFrame (approximate growth rates).
    """
    return np.log(df).diff().dropna()


def decompose_and_plot_altair(series: pd.Series, model: str = "additive", period: int = 4):
    """
    Decomposes a time series and plots trend, seasonal, and residual components using Altair.

    Args:
        series (pd.Series): A pandas Series with DatetimeIndex.
        model (str): "additive" or "multiplicative".
        period (int): Seasonal period (e.g., 4 for quarterly, 12 for monthly).
    """
    # Decompose
    decomposition = seasonal_decompose(series.dropna(), model=model, period=period)

    # Prepare dataframe
    df = pd.DataFrame({
        "Date": decomposition.observed.index,
        "Trend": decomposition.trend.values,
        "Seasonal": decomposition.seasonal.values,
        "Residual": decomposition.resid.values
    }).melt(id_vars="Date", var_name="Component", value_name="Value")

    # Plot
    chart = alt.Chart(df).mark_line().encode(
        x="Date:T",
        y="Value:Q",
        color="Component:N",
        tooltip=["Date:T", "Component:N", "Value:Q"]
    ).properties(
        title=f"Seasonal Decomposition: {series.name}",
        width=800,
        height=400
    ).interactive()

    return chart


def to_long(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
    """
    Convert a wide time-series DataFrame to long format for Altair.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex or a date column.
        date_col (Optional[str]): Name of the date column if index is not datetime-like.

    Returns:
        pd.DataFrame: Long-format DataFrame with columns ['Date', 'Variable', 'Value'].
    """
    # Resolve date column
    if date_col is None:
        # Try to use index
        if isinstance(df.index, pd.DatetimeIndex):
            df_reset = df.reset_index().rename(columns={df.index.name or 'index': 'Date'})
        else:
            # Try common date column names
            for candidate in ['Date', 'date', 'DATE', 'timestamp', 'Timestamp']:
                if candidate in df.columns:
                    date_col = candidate
                    break
            if date_col is None:
                raise ValueError("No datetime index and no date column found. Provide 'date_col'.")
            df_reset = df.copy()
    else:
        df_reset = df.copy()

    # Ensure a 'Date' column exists and is datetime
    if 'Date' not in df_reset.columns:
        df_reset = df_reset.rename(columns={date_col: 'Date'})
    df_reset['Date'] = pd.to_datetime(df_reset['Date'], errors='coerce')

    # Keep only numeric columns for plotting
    numeric_cols = df_reset.select_dtypes(include='number').columns.tolist()

    # Melt to long format
    df_long = df_reset.melt(id_vars='Date', value_vars=numeric_cols,
                            var_name='Variable', value_name='Value')
    return df_long


def plot_time_series_altair(
    df: pd.DataFrame,
    title: str = "Time Series Plot",
    date_col: Optional[str] = None,
    variables: Optional[List[str]] = None,
    interactive_mode: str = "dim",  # 'dim' (fade unselected) or 'filter'
    width: int = 900,
    height: int = 420,
    tooltip_extra: Optional[List[str]] = None
) -> alt.Chart:
    """
    Plot an interactive multi-series time chart with legend-based multi-select.

    Args:
        df (pd.DataFrame): Wide DataFrame with datetime index or a date column.
        title (str): Chart title.
        date_col (Optional[str]): Name of date column if index is not datetime-like.
        variables (Optional[List[str]]): Subset of columns to include; numeric only will be used.
        interactive_mode (str): 'dim' to fade unselected lines, or 'filter' to hide them.
        width (int): Chart width.
        height (int): Chart height.
        tooltip_extra (Optional[List[str]]): Extra columns to include in tooltip (if present).

    Returns:
        alt.Chart: Interactive Altair chart with legend toggling.
    """
    # Prepare data in long format
    df_long = to_long(df, date_col=date_col)

    # Optional subsetting of variables
    if variables is not None:
        df_long = df_long[df_long['Variable'].isin(variables)]

    # Build a point selection bound to legend for multi-select toggle
    sel = alt.selection_point(fields=['Variable'], bind='legend', toggle='true', empty='all')

    # Tooltip
    tooltip_fields = ['Date:T', 'Variable:N', 'Value:Q']
    if tooltip_extra:
        # Add extra fields if they exist in the data
        for col in tooltip_extra:
            if col in df_long.columns:
                # Infer type (N or Q); keep it simple: numeric -> Q else N
                _type = 'Q' if pd.api.types.is_numeric_dtype(df_long[col]) else 'N'
                tooltip_fields.append(f'{col}:{_type}')

    base = alt.Chart(df_long).encode(
        x=alt.X('Date:T', title=None),
        y=alt.Y('Value:Q', title=None),
        color=alt.Color('Variable:N', legend=alt.Legend(title="Series")),
        tooltip=tooltip_fields
    ).add_params(sel)

    if interactive_mode == "filter":
        chart = base.transform_filter(sel).mark_line()
    elif interactive_mode == "dim":
        # Put the conditional in the encoding layer, not in mark_line()
        chart = base.encode(
            opacity=alt.condition(sel, alt.value(1.0), alt.value(0.12))
        ).mark_line()
    else:
        raise ValueError("interactive_mode must be 'dim' or 'filter'.")

    chart = chart.add_params(sel).properties(
        title=title,
        width=width,
        height=height
    ).interactive()

    return chart