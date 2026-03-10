import os
import pandas as pd

SOURCE_FOLDER = "/content/drive/MyDrive/gdp_nowcasting_seminar/Data/pickles"


def save_all_dfs_to_drive(data_frames, target_folder=SOURCE_FOLDER):
    """
    Saves each DataFrame in the data_frames dictionary as a pickle file to the specified Google Drive folder.

    Args:
        data_frames (dict): Dictionary of DataFrames where keys are file names (without extension).
        target_folder (str): Folder path in Google Drive to save the pickle files.
    """
    # Create the folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    for name, df in data_frames.items():
        filename = os.path.join(target_folder, f"{name}.pkl")
        df.to_pickle(filename)
        print(f"Saved {name} to {filename}")


def load_all_pickles_from_drive(source_folder=SOURCE_FOLDER):
    """
    Loads all .pkl files from a specified Google Drive folder into a dictionary of DataFrames.

    Args:
        source_folder (str): Path to the folder containing .pkl files.

    Returns:
        dict: Dictionary of DataFrames with keys as file names (without .pkl extension).
    """
    data_frames = {}

    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Folder not found: {source_folder}")

    for filename in os.listdir(source_folder):
        if filename.endswith(".pkl"):
            filepath = os.path.join(source_folder, filename)
            df_name = filename.replace(".pkl", "")
            data_frames[df_name] = pd.read_pickle(filepath)
            print(f"Loaded {df_name} from {filepath}")

    return data_frames
def check_duplicate_indices(dfs: dict) -> None:
    """
    Checks for duplicated indices in a dictionary of DataFrames.
    Prints the names of DataFrames with non-unique indices and shows the duplicated entries.

    Args:
        dfs (dict): Dictionary of {name: DataFrame}.
    """
    for name, df in dfs.items():
        idx = df.index if isinstance(df.index, pd.DatetimeIndex) else df.get('Date')

        if idx is None:
            print(f"[WARNING] '{name}' has no recognizable date index or 'Date' column.")
            continue

        # Convert to datetime if not already
        idx = pd.to_datetime(idx, errors='coerce')
        duplicated = idx[idx.duplicated(keep=False)]

        if not duplicated.empty:
            print(f"\n[❗] Duplicated dates found in '{name}':")
            print(duplicated.value_counts().sort_index())
        else:
            print(f"[OK] '{name}' has unique dates.")
