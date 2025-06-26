import pandas as pd
import os

def load_data(filename: str) -> pd.DataFrame:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"file not found: {filename}")

    df = pd.read_csv(filename, parse_dates=['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    return df

def basic_info(df: pd.DataFrame):
    print("\nBasic Info:")
    print(df.info())
    print("\nDescriptive Stats:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())

