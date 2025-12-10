import pandas as pd
import numpy as np

def generate_signals(daily_df, z_threshold=1.0):
    """
    Generate trading signals for Strategy 1: Sentiment Momentum.

    Trading Rules:
    --------------
    Long (buy):
        sentiment_z > +1
        AND sentiment_change > 0

    Short (sell):
        sentiment_z < -1
        AND sentiment_change < 0

    Neutral:
        otherwise

    Parameters
    ----------
    daily_df : pd.DataFrame
        Output dataframe from build_sentiment_index().
        Must contain: ['sentiment_z', 'sentiment_change']
    z_threshold : float
        Threshold for defining unusually high or low sentiment.

    Returns
    -------
    pd.DataFrame
        With an additional column named 'signal_mom'
        containing -1, 0, or 1 for each date/ticker.
    """

    df = daily_df.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Initialize the signal column
    df["signal_mom"] = 0

    # ---- Long Entry ----
    long_mask = (
        (df["sentiment_z"] > z_threshold) &
        (df["sentiment_change"] > 0)
    )
    df.loc[long_mask, "signal_mom"] = 1

    # ---- Short Entry ----
    short_mask = (
        (df["sentiment_z"] < -z_threshold) &
        (df["sentiment_change"] < 0)
    )
    df.loc[short_mask, "signal_mom"] = -1

    return df
