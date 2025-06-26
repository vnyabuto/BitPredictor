# src/feature_engineering.py

import pandas as pd
import numpy as np

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df['returns'] = df['close'].pct_change()
    return df

def add_moving_averages(df: pd.DataFrame, windows=[5, 10, 20]) -> pd.DataFrame:
    for w in windows:
        df[f'sma_{w}'] = df['close'].rolling(window=w).mean()
        df[f'ema_{w}'] = df['close'].ewm(span=w, adjust=False).mean()
    return df

def add_volatility(df: pd.DataFrame, window=10) -> pd.DataFrame:
    df['volatility'] = df['close'].rolling(window=window).std()
    return df

def add_rsi(df: pd.DataFrame, window=14) -> pd.DataFrame:
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_returns(df)
    df = add_moving_averages(df)
    df = add_volatility(df)
    df = add_rsi(df)
    df = add_macd(df)

    df = df.dropna()
    return df

def add_features(df):
    df = df.copy()

    # % change features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Moving Averages
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_15'] = df['close'].rolling(window=15).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()

    # RSI (14-period)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Lag features
    for lag in [1, 2, 3]:
        df[f'lag_{lag}'] = df['close'].shift(lag)

    # Drop rows with NaNs (from rolling calculations)
    df = df.dropna()

    return df