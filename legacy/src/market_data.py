import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_series(tickers: dict, start='1995-01-01', end=None, interval='1d') -> dict:
    """
    Fetches daily time series data from Yahoo Finance for given tickers.

    Parameters:
        tickers (dict): Mapping from name to Yahoo Finance ticker.
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format (default: today).
        interval (str): Data interval (e.g., '1d', '1wk', '1mo').

    Returns:
        dict[str, pd.DataFrame]: Dictionary of dataframes for each series.
    """
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    data_dict = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=start, end=end, interval=interval)
            df = df[['Close']].rename(columns={'Close': name})
            df.index.name = 'Date'
            data_dict[name] = df
            print(f"[INFO] Fetched {name} from {ticker} ({len(df)} rows)")
        except Exception as e:
            print(f"[ERROR] Failed to fetch {name} ({ticker}): {e}")
    
    return data_dict
