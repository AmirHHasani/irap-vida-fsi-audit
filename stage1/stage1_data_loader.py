# stage1_data_loader.py
"""
Function for loading data for the Stage 1 risk modeling pipeline.
"""
import pandas as pd
import sys
# ADD THIS IMPORT
import stage1_config as cfg


def load_data(file_path):
    """
    Loads data from a specified CSV file path.

    Args:
        file_path (str or Path): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    
    Raises:
        FileNotFoundError: If the file cannot be found at the specified path.
        Exception: For other pandas-related reading errors.
    """
    print(f"   Loading data from: {file_path}")
    try:
        # Read a small chunk of the file to inspect columns first to avoid KeyErrors when cfg.ID_COL absent
        sample = pd.read_csv(file_path, nrows=5)
        use_dtype = {}
        if cfg.ID_COL in sample.columns:
            use_dtype[cfg.ID_COL] = str
        # Force Dataset ID to string to treat as categorical (not numeric)
        if cfg.DATASET_ID_COL in sample.columns:
            use_dtype[cfg.DATASET_ID_COL] = str
        # prefer canonical index if present in input
        if getattr(cfg, 'CANONICAL_INDEX_COL', None) in sample.columns:
            use_dtype[getattr(cfg, 'CANONICAL_INDEX_COL')] = str

        if use_dtype:
            df = pd.read_csv(file_path, dtype=use_dtype)
        else:
            df = pd.read_csv(file_path)
        print("   Data loaded successfully.")
        return df
    except FileNotFoundError:
        # Re-raise the specific error to be caught by main.py
        raise FileNotFoundError(f"The data file was not found at '{file_path}'.")
    except Exception as e:
        print(f"   [ERROR] Could not read the data file. Error: {e}", file=sys.stderr)
        # Re-raise a more general exception
        raise Exception(f"Failed to load or parse data from '{file_path}'.")