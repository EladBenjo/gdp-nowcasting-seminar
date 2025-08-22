import os
import pandas as pd
from typing import Dict

def load_csv_folder_as_dataframes(folder_path: str, suffix: str = '_df') -> Dict[str, pd.DataFrame]:
    """
    Loads all CSV files from a given folder into a dictionary of pandas DataFrames.

    Parameters:
        folder_path (str): The path to the folder containing CSV files.
        suffix (str): Optional suffix to append to dataframe names (default is '_df').

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are file-based names and values are DataFrames.
    """
    data_frames = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df_name = filename.replace('.csv', suffix)
            try:
                df = pd.read_csv(file_path)
                data_frames[df_name] = df
                print(f"[INFO] Loaded {filename} as '{df_name}'")
            except Exception as e:
                print(f"[ERROR] Failed to load {filename}: {e}")

    return data_frames

def info_and_describe(data_frames):
    """
    Prints information and descriptive statistics for each DataFrame in a dictionary.

    Args:
        data_frames (dict): A dictionary where keys are DataFrame names (str) and values are pandas DataFrame objects.

    Returns:
        None

    Side Effects:
        Prints the output of DataFrame.info() and DataFrame.describe() for each DataFrame in the dictionary.
    """
    for df_name, df in data_frames.items():
        print(f"\n--- Info for {df_name} ---")
        df.info()
        print(f"\n--- Description for {df_name} ---")
        print(df.describe())

def clean_month_year(series):
    """
    Parses a pandas Series of date strings into datetime objects.
    Supports formats like 'Jan-96', '25-Apr', and '1-Nov' (assumed 2001).

    Args:
        series (pd.Series): Series of date strings.

    Returns:
        pd.Series: Series of datetime64[ns] objects.
    """
    def parse_date(val):
        if not isinstance(val, str) or '-' not in val:
            return pd.NaT

        parts = val.split('-')

        # Case: "Jan-96"
        if len(parts[1]) == 2 and parts[0].isalpha():
            return pd.to_datetime(val, format='%b-%y', errors='coerce')

        # Case: "25-Apr" → assume fixed year (e.g., 2025)
        elif len(parts[0]) == 2 and parts[0].isdigit():
            return pd.to_datetime(f"{parts[1]}-2025", format='%b-%Y', errors='coerce')

        # Case: "1-Nov", "2-Jan", etc. → assume "Nov 2001", "Jan 2002"
        elif parts[0].isdigit() and len(parts[0]) == 1:
            year = 2000 + int(parts[0])
            return pd.to_datetime(f"{parts[1]}-{year}", format='%b-%Y', errors='coerce')

        return pd.NaT

    return series.apply(parse_date)

def set_monthly_dates_from_start(df: pd.DataFrame, start_date: str = '1996-01-01') -> pd.DataFrame:
    """
    Assigns a monthly DatetimeIndex starting from a given start_date.

    Args:
        df (pd.DataFrame): The DataFrame to update.
        start_date (str): The date to start from (default is '1996-01-01').

    Returns:
        pd.DataFrame: DataFrame with a new 'Date' column and datetime index.
    """
    n = len(df)
    dates = pd.date_range(start=start_date, periods=n, freq='MS')  # 'MS' = Month Start
    df = df.copy()
    df['Date'] = dates
    df.set_index('Date', inplace=True)
    return df


def convert_object_columns_to_float(df):
    """
    Converts all columns of type 'object' to 'float', excluding the 'Date' column.

    Args:
        df (pd.DataFrame): Input DataFrame to modify in-place or return modified copy.

    Returns:
        pd.DataFrame: A DataFrame with object columns converted to float where applicable.
    """
    for col in df.columns:
        if col != "Date" and df[col].dtype == 'object':
            try:
                df[col] = df[col].str.replace(",", "")  # Optional: remove thousands separator
                df[col] = df[col].astype(float)
            except Exception as e:
                print(f"Could not convert column '{col}' to float: {e}")
    return df

def clean_percentage_columns(df):
    """
    Removes '%' character from all object columns and converts them to float if possible.

    Args:
        df (pd.DataFrame): DataFrame with potential percentage strings (e.g., '42%').

    Returns:
        pd.DataFrame: Modified DataFrame with cleaned and converted columns.
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].str.contains('%').any():
                try:
                    df[col] = df[col].str.replace('%', '', regex=False).astype(float)
                except Exception as e:
                    print(f"Could not convert column '{col}' to float: {e}")
    return df

def print_date_range(df, df_name="DataFrame"):
    """
    Prints the earliest and latest date in a DataFrame, prioritizing:
    1. A column named 'Date'
    2. A DatetimeIndex
    3. Any other datetime-like column

    Args:
        df (pd.DataFrame): The input DataFrame.
        df_name (str): Optional name of the DataFrame for display.
    """
    print(f"\n[INFO] Processing '{df_name}'")

    # Case 1: 'Date' column
    if 'Date' in df.columns:
        date_series = pd.to_datetime(df['Date'], errors='coerce')
        valid_dates = date_series.dropna()
        if not valid_dates.empty:
            print("[INFO] Using 'Date' column")
            print(f"Earliest date: {valid_dates.min()}")
            print(f"Latest date: {valid_dates.max()}")
            return

    # Case 2: DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        print("[INFO] Using DatetimeIndex")
        print(f"Earliest date: {df.index.min()}")
        print(f"Latest date: {df.index.max()}")
        return

    # Case 3: Find any datetime-like column
    for col in df.columns:
        converted = pd.to_datetime(df[col], errors='coerce')
        if converted.notna().sum() > 0:
            print(f"[INFO] Using datetime column: '{col}'")
            print(f"Earliest date: {converted.min()}")
            print(f"Latest date: {converted.max()}")
            return

    print("[WARNING] No datetime information found.")


def merge_series_freq(dfs: dict, date_column: str = 'Date') -> pd.DataFrame:
    """
    Merges multiple daily DataFrames using an outer join on the date index.

    Args:
        dfs (dict): Dictionary of {name: DataFrame}, where each DataFrame is daily
                    and has either a DatetimeIndex or a 'Date' column.
        date_column (str): Optional, name of the date column if not using DatetimeIndex.

    Returns:
        pd.DataFrame: Merged DataFrame with a complete date range and all series aligned.
    """
    standardized_dfs = []

    for name, df in dfs.items():
        # Copy to avoid modifying original
        df = df.copy()

        # Case 1: DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            temp = df
            temp.index.name = 'Date'

        # Case 2: Has date column
        elif date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            temp = df.set_index(date_column)
            temp.index.name = 'Date'

        else:
            raise ValueError(f"[ERROR] No datetime info found in '{name}'.")

        # Rename columns to include source name
        temp = temp.add_prefix(f"{name}_")
        standardized_dfs.append(temp)

    # Outer join on full date range
    merged_df = pd.concat(standardized_dfs, axis=1, join='outer')

    # Sort index for consistency
    merged_df = merged_df.sort_index()

    return merged_df

