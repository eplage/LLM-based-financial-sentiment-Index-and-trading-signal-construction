import pandas as pd
import numpy as np

def backtest(daily_df, signal_col="signal_mom"):
    """
    Backtest the Sentiment Momentum strategy using equal-weighted positions
    across all tickers that have an active signal.

    The strategy return for each ticker is:
        next_day_return * signal

    And the portfolio return each day is the average return of all tickers
    with non-neutral signals.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Output dataframe from generate_signals().
        Must contain columns: ['return_1d', signal_col]
    signal_col : str
        Name of the signal column to use (default: 'signal_mom').

    Returns
    -------
    results : dict
        Dictionary of performance metrics (total return, Sharpe, etc.)
    equity_curve : pd.Series
        Time series of cumulative equity for plotting.
    """

    df = daily_df.copy()
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Strategy return per ticker: signal * next day's return
    df["strategy_ret"] = df[signal_col] * df["return_1d"]

    # Aggregate returns across tickers each day
    def daily_portfolio(group):
        active = group[group[signal_col] != 0]  # only days with a trade
        if len(active) == 0:
            return pd.Series({"port_ret": 0.0})
        return pd.Series({"port_ret": active["strategy_ret"].mean()})

    # Daily portfolio returns
    port = (
        df.groupby("date")
          .apply(daily_portfolio)
          .reset_index()
          .sort_values("date")
    )

    # Equity curve
    port["equity"] = (1 + port["port_ret"]).cumprod()

    # Performance statistics
    total_return = port["equity"].iloc[-1] - 1
    ann_factor = 252

    avg_daily = port["port_ret"].mean()
    vol_daily = port["port_ret"].std()

    ann_return = (1 + avg_daily) ** ann_factor - 1
    ann_vol = vol_daily * np.sqrt(ann_factor)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    # Max Drawdown
    rolling_max = port["equity"].cummax()
    drawdown = port["equity"] / rolling_max - 1
    max_dd = drawdown.min()

    results = {
        "Total Return": float(total_return),
        "Annualized Return": float(ann_return),
        "Annualized Volatility": float(ann_vol),
        "Sharpe Ratio": float(sharpe),
        "Max Drawdown": float(max_dd)
    }

    return results, port.set_index("date")["equity"]
