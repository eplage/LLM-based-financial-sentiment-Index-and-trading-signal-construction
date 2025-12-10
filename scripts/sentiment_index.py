import pandas as pd
import numpy as np

def build_sentiment_index(df, lookback=10):
    """
    Build a daily sentiment index with rolling z-scores and sentiment momentum.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataset containing columns:
        ['ticker', 'date_time', 'close', 'sentiment_score']
    lookback : int
        Number of days for rolling mean/std calculation.

    Returns
    -------
    pd.DataFrame
        Daily-level dataframe with sentiment index and sentiment momentum signals.
    """

    # Ensure proper datetime format
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["date"] = df["date_time"].dt.date

    # --- Daily average sentiment for each ticker ---
    daily_sent = (
        df.groupby(["ticker", "date"])["sentiment_score"]
          .mean()
          .rename("sentiment_mean")
          .reset_index()
    )

    # --- Last closing price per day ---
    daily_close = (
        df.dropna(subset=["close"])
          .sort_values(["ticker", "date_time"])
          .groupby(["ticker", "date"])["close"]
          .last()
          .rename("close")
          .reset_index()
    )

    # Combine sentiment + price
    daily_df = pd.merge(daily_sent, daily_close, on=["ticker", "date"], how="left")
    daily_df = daily_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # --- Rolling z-score of sentiment ---
    def calc_z(group):
        group["roll_mean"] = group["sentiment_mean"].rolling(
            lookback, min_periods=5
        ).mean()
        group["roll_std"] = group["sentiment_mean"].rolling(
            lookback, min_periods=5
        ).std()
        group["sentiment_z"] = (
            (group["sentiment_mean"] - group["roll_mean"]) / group["roll_std"]
        )
        return group

    daily_df = daily_df.groupby("ticker", group_keys=False).apply(calc_z)

    # --- 1-day ahead return for backtesting ---
    daily_df["return_1d"] = (
        daily_df.groupby("ticker")["close"].pct_change().shift(-1)
    )

    # --- Sentiment Momentum: change in sentiment vs. yesterday ---
    daily_df["sentiment_change"] = (
        daily_df.groupby("ticker")["sentiment_mean"].diff()
    )

    return daily_df
