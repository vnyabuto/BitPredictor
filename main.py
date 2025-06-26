from src.data_loader import load_data, basic_info
from src.utils import plot_price, plot_volume, plot_returns_hist
from src.feature_engineering import engineer_features, add_features
from src.model_training import (
    prepare_dataset,
    train_xgb_tuned,
    save_model,
    predict_proba_direction
)
from src.backtest import backtest_with_sltp
import matplotlib.pyplot as plt

def run():
    df = load_data("data/bitpredict.csv")
    basic_info(df)
    plot_price(df); plot_volume(df); plot_returns_hist(df)

    # Build features once
    df_feats = engineer_features(df)
    df_feats = add_features(df_feats)
    X, y = prepare_dataset(df_feats)

    # Train/tune
    model = train_xgb_tuned(X, y)
    save_model(model)

    # Threshold grid
    thresholds = [round(x,2) for x in list(__import__('numpy').arange(0.3, 0.85, 0.05))]
    results = {}

    for T in thresholds:
        sigs = predict_proba_direction(model, X, threshold=T)
        count = sigs.sum()
        bt = backtest_with_sltp(df_feats, sigs)
        final = bt['cumulative_strategy_return'].iloc[-1]
        results[T] = {'trades': int(count), 'return': float(final)}
        print(f"T={T:.2f} → Signals={count}, Return={final:.2f}")

    print("\nThreshold sweep summary:")
    for T, info in results.items():
        print(f"  {T:.2f}: Trades={info['trades']}, Return={info['return']:.2f}")

    # Choose the best threshold by highest return (with at least N trades)
    best = max((T for T in thresholds if results[T]['trades']>50),
               key=lambda T: results[T]['return'],
               default=None)
    if best is None:
        print("\n⚠️ No threshold produced enough trades—consider adding features or removing SL/TP.")
        return

    print(f"\n🏆 Best threshold with ≥50 trades: T={best:.2f}")

    # Final backtest at best T
    final_sigs = predict_proba_direction(model, X, threshold=best)
    bt_final = backtest_with_sltp(df_feats, final_sigs)
    ax = bt_final[['cumulative_market_return','cumulative_strategy_return']] \
        .plot(title=f"Final Equity Curve (T={best:.2f})", figsize=(12,6))
    ax.set_ylabel("Cumulative Return")
    plt.grid(True)
    plt.show()

    print(f"\n📈 Buy&Hold: {bt_final['cumulative_market_return'].iloc[-1]:.2f}×")
    print(f"🤖 Strategy: {bt_final['cumulative_strategy_return'].iloc[-1]:.2f}×")

if __name__ == "__main__":
    run()
