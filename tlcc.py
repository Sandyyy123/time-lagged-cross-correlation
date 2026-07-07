"""
Time-Lagged Cross-Correlation (TLCC) core.

Given two continuous, equally-sampled time series (observer vs target),
compute the Pearson correlation at each integer lag over a bounded window,
then extract the peak coefficient and the lag at which it occurs.

Design mirror (HCI dyadic tracking study):
    - sampling interval : 1000 ms  (1 Hz)
    - lag window        : -5 s  ...  +10 s
    - => lags -5, -4, ... , +9, +10  == 16 discrete lags == 16 coefficients
A positive peak lag means the observer's signal best matches the target's
signal when the observer is shifted forward in time i.e. the observer LAGS
the target by that many seconds (temporal cognitive lag).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def lagged_pearson(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """Pearson r between x and y with y shifted by `lag` samples.

    lag > 0 : y is shifted forward (x compared to y's future)  -> x leads
    lag < 0 : y is shifted backward                            -> x lags
    Overlapping (non-NaN) region only; returns np.nan if < 3 overlap points
    or if either overlapping segment is constant (undefined correlation).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if lag > 0:
        xs, ys = x[:-lag], y[lag:]
    elif lag < 0:
        xs, ys = x[-lag:], y[:lag]
    else:
        xs, ys = x, y
    if len(xs) < 3:
        return np.nan
    if np.std(xs) == 0 or np.std(ys) == 0:
        return np.nan
    return float(np.corrcoef(xs, ys)[0, 1])


def tlcc(
    observer: np.ndarray,
    target: np.ndarray,
    fs_hz: float = 1.0,
    lo_s: float = -5.0,
    hi_s: float = 10.0,
) -> pd.DataFrame:
    """Time-lagged cross-correlation over [lo_s, hi_s] at 1/fs_hz resolution.

    Returns a tidy DataFrame with one row per lag:
        lag_samples | lag_seconds | r
    For fs=1 Hz, lo=-5, hi=+10 -> exactly 16 rows (16 coefficients).
    """
    step = 1.0 / fs_hz
    lags_s = np.arange(lo_s, hi_s + 1e-9, step)
    rows = []
    for ls in lags_s:
        lag_samp = int(round(ls * fs_hz))
        rows.append(
            {
                "lag_samples": lag_samp,
                "lag_seconds": round(ls, 3),
                "r": lagged_pearson(observer, target, lag_samp),
            }
        )
    return pd.DataFrame(rows)


def peak_of(tlcc_df: pd.DataFrame) -> dict:
    """Extract the peak (max |r|) row from a TLCC frame.

    Returns dict: peak_r, peak_lag_s, zero_lag_r.
    Peak is on signed r (strongest positive alignment); switch to
    tlcc_df['r'].abs() if anti-correlation is meaningful for the study.
    """
    valid = tlcc_df.dropna(subset=["r"])
    if valid.empty:
        return {"peak_r": np.nan, "peak_lag_s": np.nan, "zero_lag_r": np.nan}
    top = valid.loc[valid["r"].idxmax()]
    zero = tlcc_df.loc[tlcc_df["lag_seconds"] == 0.0, "r"]
    return {
        "peak_r": float(top["r"]),
        "peak_lag_s": float(top["lag_seconds"]),
        "zero_lag_r": float(zero.iloc[0]) if len(zero) else np.nan,
    }
