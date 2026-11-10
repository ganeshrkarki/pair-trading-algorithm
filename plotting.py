import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_pair_backtest(bt: pd.DataFrame, symbol_x: str, symbol_y: str, metrics: dict,
                       entry_z: float, exit_z: float, path: str) -> None:
    fig, (ax_spread, ax_z, ax_eq) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    beta = bt["HedgeRatio"].iloc[0]

    ax_spread.plot(bt.index, bt["Spread"], lw=1, label=f"{symbol_x} - {beta:.2f} x {symbol_y}")
    ax_spread.plot(bt.index, bt["SpreadMean"], lw=1.5, label="Rolling mean")
    ax_spread.set_title(f"{symbol_x} / {symbol_y} spread (hedge ratio fit on formation window)")
    ax_spread.legend(loc="upper left")

    ax_z.plot(bt.index, bt["ZScore"], lw=1, color="tab:green")
    for level, style in ((entry_z, "--"), (-entry_z, "--"), (exit_z, ":"), (-exit_z, ":")):
        ax_z.axhline(level, color="gray", ls=style, lw=1)
    changes = bt[bt["Position"].diff().fillna(bt["Position"]) != 0]
    for side, marker, color in ((1, "^", "green"), (-1, "v", "red"), (0, "x", "black")):
        pts = changes[changes["Position"] == side]
        label = {1: "Enter long spread", -1: "Enter short spread", 0: "Exit"}[side]
        ax_z.scatter(pts.index, pts["ZScore"], marker=marker, color=color, s=40, zorder=3, label=label)
    ax_z.set_ylabel("z-score")
    ax_z.legend(loc="upper left", ncol=3)

    ax_eq.plot(bt.index, bt["Equity"], color="darkgreen")
    ax_eq.set_ylabel("Equity")
    ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k"))
    ax_eq.set_title(f"Return {metrics['total_return']:.1%} | Sharpe {metrics['annualized_sharpe']:.2f} | "
                    f"Max DD {metrics['max_drawdown']:.1%} | {metrics['trades']} trades")

    for ax in (ax_spread, ax_z, ax_eq):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
