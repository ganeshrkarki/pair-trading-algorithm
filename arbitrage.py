import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import coint


def fetch_pair_data(symbol_x: str, symbol_y: str, start: str, end: str) -> pd.DataFrame:
	"""Fetch adjusted close prices for two symbols and align by date."""
	x_raw = yf.download(symbol_x, start=start, end=end, auto_adjust=True, progress=False)
	y_raw = yf.download(symbol_y, start=start, end=end, auto_adjust=True, progress=False)

	x_close = _extract_close_series(x_raw)
	y_close = _extract_close_series(y_raw)

	df = pd.DataFrame({
		"Price_X": x_close,
		"Price_Y": y_close,
	}).dropna()
	return df


def _extract_close_series(data: pd.DataFrame) -> pd.Series:
	"""Handle both regular and MultiIndex columns from yfinance."""
	if data.empty:
		return pd.Series(dtype=float)

	if isinstance(data.columns, pd.MultiIndex):
		if ("Close",) == tuple(data.columns[0][:1]):
			close = data["Close"]
			if isinstance(close, pd.DataFrame):
				return close.iloc[:, 0]
			return close
		# Fallback for uncommon MultiIndex layouts
		for col in data.columns:
			if "Close" in col:
				return data[col]

	return data["Close"]


def fit_hedge_ratio(price_x: pd.Series, price_y: pd.Series) -> float:
	"""Estimate beta in spread = X - beta * Y using OLS."""
	x = price_x.astype(float).values
	y = price_y.astype(float).values
	model = OLS(x, add_constant(y)).fit()
	beta = float(model.params[1])
	return beta


def cointegration_check(price_x: pd.Series, price_y: pd.Series) -> dict:
	"""Return Engle-Granger cointegration test results."""
	stat, p_value, critical_values = coint(price_x, price_y)
	return {
		"test_stat": float(stat),
		"p_value": float(p_value),
		"critical_values": {
			"1%": float(critical_values[0]),
			"5%": float(critical_values[1]),
			"10%": float(critical_values[2]),
		},
		"is_cointegrated_5pct": bool(p_value < 0.05),
	}


def build_pairs_signals(
	data: pd.DataFrame,
	lookback: int = 30,
	entry_z: float = 2.0,
	exit_z: float = 0.5,
) -> pd.DataFrame:
	"""Create mean-reversion signals based on spread z-score."""
	if data.empty:
		raise ValueError("Input data is empty.")
	if lookback < 2:
		raise ValueError("lookback must be at least 2.")
	if exit_z >= entry_z:
		raise ValueError("exit_z must be smaller than entry_z.")

	df = data.copy()
	beta = fit_hedge_ratio(df["Price_X"], df["Price_Y"])
	df["HedgeRatio"] = beta

	df["Spread"] = df["Price_X"] - beta * df["Price_Y"]
	df["SpreadMean"] = df["Spread"].rolling(lookback).mean()
	df["SpreadStd"] = df["Spread"].rolling(lookback).std()
	df["ZScore"] = (df["Spread"] - df["SpreadMean"]) / df["SpreadStd"]

	# State machine: 1 = long spread, -1 = short spread, 0 = flat
	state = 0
	signals = []
	for z in df["ZScore"]:
		if np.isnan(z):
			signals.append(0)
			continue

		if state == 0:
			if z <= -entry_z:
				state = 1
			elif z >= entry_z:
				state = -1
		elif state == 1:
			if z >= -exit_z:
				state = 0
		elif state == -1:
			if z <= exit_z:
				state = 0

		signals.append(state)

	df["Signal"] = signals
	df["Position_X"] = df["Signal"]
	df["Position_Y"] = -df["Signal"] * beta
	return df


def backtest_pairs(signals_df: pd.DataFrame, initial_capital: float = 100000.0) -> tuple[pd.DataFrame, dict]:
	"""Run a simple daily-return backtest for the pair positions."""
	if signals_df.empty:
		raise ValueError("signals_df is empty.")

	df = signals_df.copy()
	df["Ret_X"] = df["Price_X"].pct_change().fillna(0.0)
	df["Ret_Y"] = df["Price_Y"].pct_change().fillna(0.0)

	# Use yesterday's positions to avoid look-ahead bias.
	px = df["Position_X"].shift(1).fillna(0.0)
	py = df["Position_Y"].shift(1).fillna(0.0)
	df["StrategyRet"] = px * df["Ret_X"] + py * df["Ret_Y"]

	df["Equity"] = initial_capital * (1.0 + df["StrategyRet"]).cumprod()
	df["Drawdown"] = df["Equity"] / df["Equity"].cummax() - 1.0

	total_return = float(df["Equity"].iloc[-1] / initial_capital - 1.0)
	sharpe = _annualized_sharpe(df["StrategyRet"])
	max_drawdown = float(df["Drawdown"].min())
	trade_count = int((df["Signal"].diff().abs() > 0).sum())

	metrics = {
		"initial_capital": float(initial_capital),
		"final_equity": float(df["Equity"].iloc[-1]),
		"total_return": total_return,
		"annualized_sharpe": sharpe,
		"max_drawdown": max_drawdown,
		"signal_changes": trade_count,
	}
	return df, metrics


def _annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
	std = float(returns.std())
	if std == 0.0:
		return 0.0
	mean = float(returns.mean())
	return (mean / std) * np.sqrt(periods_per_year)


def plot_pair_backtest(signals_df: pd.DataFrame, symbol_x: str, symbol_y: str, metrics: dict) -> None:
	"""Plot spread, z-score, buy/sell signals, and equity curve. Save to file."""
	fig, axes = plt.subplots(4, 1, figsize=(14, 10))

	# Plot 1: Spread and Mean
	ax = axes[0]
	ax.plot(signals_df.index, signals_df["Spread"], label="Spread", color="blue", linewidth=1)
	ax.plot(signals_df.index, signals_df["SpreadMean"], label="Spread Mean (30d)", color="orange", linewidth=2)
	ax.fill_between(signals_df.index, signals_df["SpreadMean"] - 2*signals_df["SpreadStd"],
					 signals_df["SpreadMean"] + 2*signals_df["SpreadStd"], alpha=0.2, color="gray", label="±2 Std Dev")
	ax.set_ylabel("Spread")
	ax.legend(loc="best")
	ax.set_title(f"Pair: {symbol_x} vs {symbol_y} - Spread & Mean")
	ax.grid(True, alpha=0.3)

	# Plot 2: Z-Score with Entry/Exit Bands
	ax = axes[1]
	ax.plot(signals_df.index, signals_df["ZScore"], label="Z-Score", color="green", linewidth=1)
	ax.axhline(y=2.0, color="red", linestyle="--", label="Entry Band (+2.0)")
	ax.axhline(y=-2.0, color="red", linestyle="--")
	ax.axhline(y=0.5, color="orange", linestyle=":", label="Exit Band (±0.5)")
	ax.axhline(y=-0.5, color="orange", linestyle=":")
	ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
	ax.set_ylabel("Z-Score")
	ax.legend(loc="best")
	ax.set_title("Z-Score Mean Reversion Signal")
	ax.grid(True, alpha=0.3)

	# Plot 3: Trade Signals
	ax = axes[2]
	signal_changes = signals_df[signals_df["Signal"].diff() != 0]
	long_entries = signal_changes[signal_changes["Signal"] == 1]
	short_entries = signal_changes[signal_changes["Signal"] == -1]
	exits = signal_changes[signal_changes["Signal"] == 0]

	ax.scatter(long_entries.index, long_entries["ZScore"], marker="^", color="green", s=100, label="Long Entry")
	ax.scatter(short_entries.index, short_entries["ZScore"], marker="v", color="red", s=100, label="Short Entry")
	ax.scatter(exits.index, exits["ZScore"], marker="x", color="gray", s=100, label="Exit")
	ax.plot(signals_df.index, signals_df["ZScore"], color="lightgray", linewidth=0.5, alpha=0.5)
	ax.set_ylabel("Z-Score @ Signal")
	ax.legend(loc="best")
	ax.set_title("Buy/Sell/Exit Signals")
	ax.grid(True, alpha=0.3)

	# Plot 4: Equity Curve
	ax = axes[3]
	ax.plot(signals_df.index, signals_df["Equity"], label="Strategy Equity", color="darkgreen", linewidth=2)
	ax.axhline(y=100000, color="gray", linestyle="--", alpha=0.7, label="Initial Capital")
	ax.set_ylabel("Equity ($)")
	ax.set_xlabel("Date")
	ax.legend(loc="best")
	ax.set_title(f"Equity Curve | Return: {metrics['total_return']*100:.2f}% | Sharpe: {metrics['annualized_sharpe']:.3f}")
	ax.grid(True, alpha=0.3)

	plt.tight_layout()
	filename = f"pair_backtest_{symbol_x}_{symbol_y}.png"
	plt.savefig(filename, dpi=100, bbox_inches="tight")
	print(f"  → Chart saved: {filename}")
	plt.close()


def run_example() -> None:
	"""Test multiple cointegrated pairs and visualize results."""
	pairs = [
		("EWZ", "EWA", "2019-01-01", "2025-01-01"),  # Emerging markets: Brazil vs Australia
		("TLT", "IEF", "2019-01-01", "2025-01-01"),  # Bonds: Long-term vs Intermediate
		("EEM", "VWO", "2019-01-01", "2025-01-01"),  # Emerging market ETFs
		("KO", "PEP", "2019-01-01", "2025-01-01"),   # Consumer staples (for comparison)
	]

	for symbol_x, symbol_y, start, end in pairs:
		try:
			print(f"\n{'='*60}")
			print(f"Testing Pair: {symbol_x} vs {symbol_y}")
			print(f"{'='*60}")

			data = fetch_pair_data(symbol_x, symbol_y, start, end)
			coint_result = cointegration_check(data["Price_X"], data["Price_Y"])
			signals = build_pairs_signals(data, lookback=30, entry_z=2.0, exit_z=0.5)
			signals, metrics = backtest_pairs(signals, initial_capital=100000.0)

			print(f"Cointegration p-value: {coint_result['p_value']:.4f} {'✓ COINTEGRATED' if coint_result['is_cointegrated_5pct'] else '✗ NOT COINTEGRATED'}")
			print(f"Final Equity: ${metrics['final_equity']:.2f}")
			print(f"Total Return: {metrics['total_return']*100:.2f}%")
			print(f"Sharpe Ratio: {metrics['annualized_sharpe']:.4f}")
			print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
			print(f"Number of Trades: {metrics['signal_changes']}")

			# Ask to plot
			plot_pair_backtest(signals, symbol_x, symbol_y, metrics)

		except Exception as e:
			print(f"Error processing {symbol_x}/{symbol_y}: {str(e)}")


if __name__ == "__main__":
	run_example()
