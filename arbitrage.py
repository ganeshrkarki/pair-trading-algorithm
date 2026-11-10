import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import coint

TRADING_DAYS = 252


def fetch_pair_data(symbol_x: str, symbol_y: str, start: str, end: str) -> pd.DataFrame:
    """Adjusted closes for two symbols, aligned on common dates."""
    raw = yf.download([symbol_x, symbol_y], start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"no data for {symbol_x}/{symbol_y}")
    close = raw["Close"]
    df = pd.DataFrame({"Price_X": close[symbol_x], "Price_Y": close[symbol_y]}).dropna()
    if (df <= 0).any().any():
        raise ValueError("non-positive prices in data")
    return df


def fit_hedge_ratio(price_x: pd.Series, price_y: pd.Series) -> float:
    """OLS slope of X on Y (with intercept), so spread = X - beta * Y."""
    model = OLS(price_x.astype(float).values, add_constant(price_y.astype(float).values)).fit()
    return float(model.params[1])


def cointegration_check(price_x: pd.Series, price_y: pd.Series) -> dict:
    """Engle-Granger test. Null hypothesis: no cointegration."""
    stat, p_value, _ = coint(price_x, price_y)
    return {"test_stat": float(stat), "p_value": float(p_value)}


def build_pairs_signals(
    data: pd.DataFrame,
    formation_end: str,
    lookback: int = 30,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.DataFrame:
    """Fit the hedge ratio on the formation window, then generate z-score
    signals for the trading window only.

    Signal 1 = long the spread (long X, short beta*Y), -1 = short the spread.
    The rolling mean/std only look backwards, so the first trading-day
    z-scores use the tail of the formation window, which is already known.
    """
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if exit_z >= entry_z:
        raise ValueError("exit_z must be smaller than entry_z")

    formation = data.loc[:formation_end]
    if len(formation) < lookback * 2:
        raise ValueError("formation window is too short")
    beta = fit_hedge_ratio(formation["Price_X"], formation["Price_Y"])

    df = data.copy()
    df["HedgeRatio"] = beta
    df["Spread"] = df["Price_X"] - beta * df["Price_Y"]
    df["SpreadMean"] = df["Spread"].rolling(lookback).mean()
    df["SpreadStd"] = df["Spread"].rolling(lookback).std()
    df["ZScore"] = (df["Spread"] - df["SpreadMean"]) / df["SpreadStd"]

    df = df.loc[df.index > pd.Timestamp(formation_end)].copy()

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
        elif state == 1 and z >= -exit_z:
            state = 0
        elif state == -1 and z <= exit_z:
            state = 0
        signals.append(state)

    df["Signal"] = signals
    return df


def backtest_pairs(
    signals_df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    cost_bps: float = 5.0,
):
    """Daily backtest of the spread position.

    The signal is computed at day t's close and the position is put on at
    day t+1's close, so it first earns the day t+1 -> t+2 move. Returns are
    measured on gross exposure: one unit is long 1 share of X and short beta
    shares of Y, so capital at risk is X + |beta| * Y.

    cost_bps is charged on gross exposure traded. Opening or closing a
    position costs cost_bps; flipping straight from long to short costs twice.
    """
    if signals_df.empty:
        raise ValueError("signals_df is empty")

    df = signals_df.copy()
    beta = df["HedgeRatio"].iloc[0]
    df["Position"] = df["Signal"].shift(1).fillna(0)

    spread_change = df["Price_X"].diff() - beta * df["Price_Y"].diff()
    gross = (df["Price_X"] + abs(beta) * df["Price_Y"]).shift(1)
    df["SpreadRet"] = (spread_change / gross).fillna(0.0)
    df["Cost"] = df["Position"].diff().abs().fillna(df["Position"].abs()) * cost_bps / 10_000
    df["StrategyRet"] = df["Position"].shift(1).fillna(0) * df["SpreadRet"] - df["Cost"]

    df["Equity"] = initial_capital * (1 + df["StrategyRet"]).cumprod()
    df["Drawdown"] = df["Equity"] / df["Equity"].cummax() - 1

    trades = extract_trades(df, cost_bps)
    closed = trades[~trades["open"]] if not trades.empty else trades
    metrics = {
        "total_return": float(df["Equity"].iloc[-1] / initial_capital - 1),
        "annualized_sharpe": _annualized_sharpe(df["StrategyRet"]),
        "max_drawdown": float(df["Drawdown"].min()),
        "trades": int(len(closed)),
        "win_rate": float((closed["return"] > 0).mean()) if len(closed) else float("nan"),
        "avg_holding_days": float(closed["days"].mean()) if len(closed) else float("nan"),
        "exposure": float((df["Position"] != 0).mean()),
    }
    return df, metrics, trades


def extract_trades(bt: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """One row per round trip with its net return.

    A position opened at close i earns the spread moves of days i+1..j and is
    closed at close j. Net return = (1 - c) * prod(1 + side * move) * (1 - c) - 1,
    with c = cost_bps. Open trades are marked to the last close and only pay
    the entry cost.
    """
    pos = bt["Position"].to_numpy()
    moves = bt["SpreadRet"].to_numpy()
    c = cost_bps / 10_000
    rows = []
    entry = None
    for i in range(len(pos)):
        prev = pos[i - 1] if i else 0
        if prev != 0 and pos[i] != prev:
            growth = np.prod(1 + prev * moves[entry + 1:i + 1])
            rows.append(_trade_row(bt, entry, i, prev, (1 - c) * growth * (1 - c) - 1, False))
            entry = None
        if pos[i] != 0 and pos[i] != prev:
            entry = i
    if entry is not None:
        side = pos[entry]
        growth = np.prod(1 + side * moves[entry + 1:])
        rows.append(_trade_row(bt, entry, len(pos) - 1, side, (1 - c) * growth - 1, True))
    return pd.DataFrame(rows, columns=["entry", "exit", "side", "days", "return", "open"])


def _trade_row(bt, start, end, side, ret, is_open):
    return {
        "entry": bt.index[start],
        "exit": bt.index[end],
        "side": "long spread" if side > 0 else "short spread",
        "days": end - start,
        "return": float(ret),
        "open": is_open,
    }


def _annualized_sharpe(returns: pd.Series) -> float:
    std = float(returns.std())
    if std == 0.0 or np.isnan(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(TRADING_DAYS))
