import argparse
import os

import pandas as pd

from arbitrage import backtest_pairs, build_pairs_signals, cointegration_check, fetch_pair_data
from plotting import plot_pair_backtest

PAIRS = [
    ("EEM", "VWO"),  # two emerging-market ETFs tracking similar indexes
    ("EWZ", "EWA"),  # Brazil vs Australia, both commodity-heavy
    ("KO", "PEP"),   # consumer staples peers
    ("TLT", "IEF"),  # long vs intermediate Treasuries
]


def parse_args():
    p = argparse.ArgumentParser(description="Out-of-sample pairs trading backtest.")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--formation-end", default="2021-12-31")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--lookback", type=int, default=30)
    p.add_argument("--entry-z", type=float, default=2.0)
    p.add_argument("--exit-z", type=float, default=0.5)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--out", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []

    for x, y in PAIRS:
        data = fetch_pair_data(x, y, args.start, args.end)
        formation = data.loc[:args.formation_end]
        trading = data.loc[data.index > pd.Timestamp(args.formation_end)]

        signals = build_pairs_signals(data, args.formation_end, args.lookback, args.entry_z, args.exit_z)
        bt, m, trades = backtest_pairs(signals, cost_bps=args.cost_bps)
        trades.to_csv(os.path.join(args.out, f"trades_{x}_{y}.csv"), index=False)
        plot_pair_backtest(bt, x, y, m, args.entry_z, args.exit_z,
                           os.path.join(args.out, f"backtest_{x}_{y}.png"))

        rows.append({
            "pair": f"{x}/{y}",
            "hedge_ratio": signals["HedgeRatio"].iloc[0],
            "coint_p_formation": cointegration_check(formation["Price_X"], formation["Price_Y"])["p_value"],
            "coint_p_trading": cointegration_check(trading["Price_X"], trading["Price_Y"])["p_value"],
            **m,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    print(f"Formation {args.start} to {args.formation_end}, trading to {args.end}, "
          f"lookback {args.lookback}, entry |z|>{args.entry_z}, exit |z|<{args.exit_z}, "
          f"{args.cost_bps:g} bps per side\n")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
