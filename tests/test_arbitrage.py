import unittest

import numpy as np
import pandas as pd

from arbitrage import backtest_pairs, build_pairs_signals, cointegration_check, fit_hedge_ratio

FORMATION_END = "2021-12-31"


def synthetic_pair(n=750, beta=0.8, seed=42):
    """Y is a random walk; X = beta * Y + stationary AR(1) noise, so X and Y are cointegrated."""
    rng = np.random.default_rng(seed)
    y = 100 + rng.normal(0, 1, n).cumsum()
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.9 * noise[t - 1] + rng.normal(0, 1)
    x = 20 + beta * y + noise
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"Price_X": x, "Price_Y": y}, index=idx)


class ArbitrageTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_pair()

    def test_detects_cointegration_and_recovers_beta(self):
        self.assertLess(cointegration_check(self.data["Price_X"], self.data["Price_Y"])["p_value"], 0.05)
        self.assertAlmostEqual(fit_hedge_ratio(self.data["Price_X"], self.data["Price_Y"]), 0.8, delta=0.05)

    def test_independent_walks_are_not_cointegrated(self):
        rng = np.random.default_rng(1)
        a = pd.Series(100 + rng.normal(0, 1, 750).cumsum())
        b = pd.Series(100 + rng.normal(0, 1, 750).cumsum())
        self.assertGreater(cointegration_check(a, b)["p_value"], 0.05)

    def test_hedge_ratio_ignores_trading_window(self):
        shocked = self.data.copy()
        shocked.loc[shocked.index > FORMATION_END, "Price_X"] *= 2
        a = build_pairs_signals(self.data, FORMATION_END)
        b = build_pairs_signals(shocked, FORMATION_END)
        self.assertEqual(a["HedgeRatio"].iloc[0], b["HedgeRatio"].iloc[0])
        self.assertTrue((a.index > pd.Timestamp(FORMATION_END)).all())

    def test_signals_follow_entry_and_exit_bands(self):
        sig = build_pairs_signals(self.data, FORMATION_END, entry_z=2.0, exit_z=0.5)
        entries = sig[(sig["Signal"] != 0) & (sig["Signal"].shift(1).fillna(0) == 0)]
        self.assertTrue((entries["ZScore"].abs() >= 2.0).all())
        prev = sig["Signal"].shift(1).fillna(0)
        long_exits = sig[(sig["Signal"] == 0) & (prev == 1)]
        short_exits = sig[(sig["Signal"] == 0) & (prev == -1)]
        self.assertGreater(len(long_exits) + len(short_exits), 0)
        self.assertTrue((long_exits["ZScore"] >= -0.5).all())
        self.assertTrue((short_exits["ZScore"] <= 0.5).all())

    def test_position_lags_signal_by_one_day(self):
        sig = build_pairs_signals(self.data, FORMATION_END)
        bt, _, _ = backtest_pairs(sig)
        pd.testing.assert_series_equal(bt["Position"], sig["Signal"].shift(1).fillna(0), check_names=False)

    def test_costs_lower_returns(self):
        sig = build_pairs_signals(self.data, FORMATION_END)
        _, free, _ = backtest_pairs(sig, cost_bps=0)
        _, costly, _ = backtest_pairs(sig, cost_bps=20)
        self.assertLess(costly["total_return"], free["total_return"])
        self.assertGreater(free["trades"], 0)

    def test_trade_returns_match_equity_without_costs(self):
        sig = build_pairs_signals(self.data, FORMATION_END)
        bt, m, trades = backtest_pairs(sig, cost_bps=0)
        compounded = (1 + trades["return"]).prod() - 1
        self.assertAlmostEqual(compounded, m["total_return"], places=10)


if __name__ == "__main__":
    unittest.main()
