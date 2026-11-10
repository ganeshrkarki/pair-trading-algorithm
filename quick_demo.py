"""Quick demo focusing on one good pair with visualization."""
import sys
import pandas as pd
sys.path.append('.')

# Test on synthetic cointegrated data to show the strategy works
import numpy as np
from arbitrage import (
    cointegration_check, build_pairs_signals, backtest_pairs, plot_pair_backtest
)

# Create synthetic cointegrated pair for fast demo
np.random.seed(42)
n = 600  # 2.4 years of daily data
common_trend = np.cumsum(np.random.normal(0, 1, n))

# X and Y move together with relationship: Y = 0.8 * X + noise
price_x = 100 + common_trend + np.random.normal(0, 2, n)
price_y = 100 + 0.8 * common_trend + np.random.normal(0, 1.5, n)

df = pd.DataFrame({
    'Price_X': price_x,
    'Price_Y': price_y,
})

print("="*60)
print("PAIRS TRADING EDUCATION DEMO")
print("="*60)
print(f"\nData: {len(df)} trading days of synthetic cointegrated pair")
print(f"Price_X range: ${df['Price_X'].min():.0f} - ${df['Price_X'].max():.0f}")
print(f"Price_Y range: ${df['Price_Y'].min():.0f} - ${df['Price_Y'].max():.0f}")

# Step 1: Check cointegration
print("\n" + "="*60)
print("STEP 1: CHECK IF PAIR MOVES TOGETHER (COINTEGRATION TEST)")
print("="*60)
coint_result = cointegration_check(df['Price_X'], df['Price_Y'])
print(f"p-value: {coint_result['p_value']:.4f}")
if coint_result['is_cointegrated_5pct']:
    print("✓ COINTEGRATED! These two stocks move together long-term.")
    print("  → Good for pairs trading strategy!")
else:
    print("✗ NOT cointegrated at 5% level")
    print("  → Still testing to show how strategy works...")

# Step 2: Build signals
print("\n" + "="*60)
print("STEP 2: CREATE MEAN-REVERSION SIGNALS (Z-SCORE RULES)")
print("="*60)
signals = build_pairs_signals(df, lookback=30, entry_z=2.0, exit_z=0.5)
entries = signals[signals['Signal'].diff() != 0]
print(f"Total signal changes: {len(entries)}")
print(f"Long (z <= -2.0):  {(signals['Signal'] == 1).sum()} days")
print(f"Short (z >= +2.0): {(signals['Signal'] == -1).sum()} days")
print(f"Flat (|z| < 0.5):  {(signals['Signal'] == 0).sum()} days")

# Step 3: Backtest
print("\n" + "="*60)
print("STEP 3: BACKTEST STRATEGY PERFORMANCE")
print("="*60)
signals, metrics = backtest_pairs(signals, initial_capital=100000.0)
print(f"Initial Capital: ${metrics['initial_capital']:,.0f}")
print(f"Final Equity:    ${metrics['final_equity']:,.0f}")
print(f"Return:          {metrics['total_return']*100:+.2f}%")
print(f"Sharpe Ratio:    {metrics['annualized_sharpe']:.3f} (higher is better)")
print(f"Max Drawdown:    {metrics['max_drawdown']*100:.2f}% (risk)")

# Step 4: Visualize
print("\n" + "="*60)
print("STEP 4: VISUALIZE - Generating chart...")
print("="*60)
plot_pair_backtest(signals, "Stock_X", "Stock_Y", metrics)
print("✓ Chart saved as: pair_backtest_Stock_X_Stock_Y.png")
