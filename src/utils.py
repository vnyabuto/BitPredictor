import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_price(df: pd.DataFrame):
    plt.figure(figsize=(14, 6))
    plt.plot(df['close'], label='Close Price')
    plt.title('Bitcoin Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_volume(df: pd.DataFrame):
    plt.figure(figsize=(14, 4))
    plt.plot(df['volume'], label='Volume', color='orange')
    plt.title('Bitcoin Trading Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_returns_hist(df: pd.DataFrame):
    df['returns'] = df['close'].pct_change()
    plt.figure(figsize=(10, 5))
    sns.histplot(df['returns'].dropna(), bins=100, kde=True, color='purple')
    plt.title('Distribution of Close Price Returns')
    plt.xlabel('Returns')
    plt.grid(True)
    plt.show()
