import pandas as pd
import numpy as np

def backtest_with_sltp(df, signals,
                       fee_rate=0.001,
                       slippage_rate=0.001,
                       sl_pct=0.005,
                       tp_pct=0.01,
                       initial_capital=10000):
    """
    Backtest classification signals with stop-loss & take-profit.
    """
    df = df.copy()
    df['signal'] = signals
    df['return'] = df['close'].pct_change().fillna(0)
    df['position'] = df['signal'].shift(1).fillna(0)

    # Cap each bar's return by SL/TP
    df['capped_ret'] = df['return'].clip(lower=-sl_pct, upper=tp_pct)
    df['gross_ret'] = df['position'] * df['capped_ret']

    # Trading costs when position changes
    df['trade'] = df['position'].diff().abs()
    df['net_ret'] = df['gross_ret'] - df['trade'] * (fee_rate + slippage_rate)

    # Cumulatives
    df['cumulative_market_return'] = (1 + df['return']).cumprod()
    df['cumulative_strategy_return'] = (1 + df['net_ret']).cumprod()
    df['capital'] = initial_capital * df['cumulative_strategy_return']

    return df
