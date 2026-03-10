import pandas as pd
import altair as alt

def plot_missing_heatmap(
    df: pd.DataFrame,
    freq: str = "QE",  # 'M' = Month, 'Q' = Quarter
    threshold: float = 0.3,
    max_vars: int = 40
):
    """
    Create an interactive Altair heatmap of missing values by time and variable.

    Args:
        df (pd.DataFrame): DataFrame with datetime index.
        freq (str): Resampling frequency ('M', 'Q', etc.).
        threshold (float): Include only variables with > threshold missing overall.
        max_vars (int): Maximum number of variables to plot.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex")

    # Select relevant variables
    overall_missing = df.isna().mean()
    vars_to_plot = overall_missing[overall_missing > threshold].sort_values(ascending=False).index.tolist()
    vars_to_plot = vars_to_plot[:max_vars]  # Limit to avoid overcrowded plot

    if not vars_to_plot:
        print(f"No variables with more than {threshold*100:.0f}% missing.")
        return

    # Resample and calculate missingness
    missing_by_time = df[vars_to_plot].resample(freq).apply(lambda x: x.isna().mean())
    missing_df = (
        missing_by_time
        .reset_index()
        .melt(id_vars=missing_by_time.index.name, var_name='Variable', value_name='MissingRate')
        .dropna()
    )

    # Rename time column for Altair
    time_col = missing_df.columns[0]
    missing_df.rename(columns={time_col: "TimeBin"}, inplace=True)

    # Format datetime for nicer display
    missing_df["TimeBin_str"] = missing_df["TimeBin"].dt.strftime("%Y-%m-%d")

    # Altair heatmap
    heatmap = alt.Chart(missing_df).mark_rect().encode(
        x=alt.X("TimeBin_str:O", title="Time Bin", sort=missing_df["TimeBin_str"].unique().tolist()),
        y=alt.Y("Variable:N", title="Variable"),
        color=alt.Color("MissingRate:Q", scale=alt.Scale(scheme="reds"), title="% Missing"),
        tooltip=[
            alt.Tooltip("Variable:N"),
            alt.Tooltip("TimeBin_str:O", title="Time"),
            alt.Tooltip("MissingRate:Q", format=".1%", title="Missing Rate")
        ]
    ).properties(
        width=600,
        height=25 * len(vars_to_plot),
        title=f"Missing Data Heatmap (>{int(threshold*100)}% missing)"
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )

    return heatmap
