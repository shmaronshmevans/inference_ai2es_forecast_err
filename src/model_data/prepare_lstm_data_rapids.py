"""GPU-accelerated (cuDF) data prep for LSTM inference.

This is the inference-time counterpart to
`model_data.prepare_lstm_data`: same pivots, same merges, same
target column, but executed on the GPU via cuDF/CuPy for speed.

The training-mode path (``train=True``) mirrors the pandas version
exactly.  The inference-mode path (``train=False``) is what the
real-time scoring pipeline calls; see the `prepare_lstm_data`
docstring below for the exact filter / merge semantics.
"""

import sys

sys.path.append("..")

import warnings

import cudf
import pandas as pd

from model_data import (
    encode,
    get_closest_nysm_stations,
    get_error,
)
from model_data.skip_landtype_geo import LEGACY_LSTM_CLUSTERS_CSV, skip_landtype_geo


def add_suffix(df, station):
    """Append `_{station}` to every column except `valid_time`/`time`."""
    cols = ["valid_time", "time"]
    return df.rename(
        columns={c: f"{c}_{station}" if c not in cols else c for c in df.columns}
    )


def dataframe_wrapper(stations, df):
    """Long-to-wide pivot on cuDF dataframes, one column block per
    station."""
    master_df = add_suffix(df[df["station"] == stations[0]], stations[0])
    for station in stations[1:]:
        df1 = add_suffix(df[df["station"] == station], station)
        master_df = master_df.merge(
            df1, on="valid_time", suffixes=(None, f"_{station}")
        )
    return master_df


def _map_geo_column(geo_df_pd: pd.DataFrame, target_df_pd: pd.DataFrame, col: str):
    """Map a single geographic feature column from `geo_df_pd` onto
    `target_df_pd` by station name (avoids positional misalignment)."""
    geo_dict = dict(zip(geo_df_pd["station"], geo_df_pd[col]))
    target_df_pd[col] = target_df_pd["station"].map(geo_dict)
    return target_df_pd


def prepare_lstm_data(nysm_df, hrrr_df, station, metvar, fh,
                      now=None, sequence_length=30, train=False):
    """Build the per-station inference dataframe (cuDF).

    Parameters
    ----------
    nysm_df : pandas.DataFrame or cudf.DataFrame
        NYSM observations for the relevant year.
    hrrr_df : pandas.DataFrame or cudf.DataFrame
        HRRR forecasts for the relevant `fh`.
    station : str
        Central NYSM station id.
    metvar : str
        Target variable for `target_error`.
    fh : int
        Forecast hour (1..18) being scored.
    now : datetime-like, optional
        The inference anchor time.  When `None` (training-style use),
        falls back to `max(hrrr.valid_time) - fh`.
    sequence_length : int
        Length of the past window the sequencer will consume
        (default 30).  Used here to size the past filter.
    train : bool
        If True, behave like the training data prep (no time clipping,
        inner join).  If False (default), apply the inference-time
        windowing described below.

    Inference behaviour (`train=False`)
    -----------------------------------
    The encoder-decoder consumes a window of
    ``sequence_length + forecast_steps`` rows where the future portion
    contains real future HRRR forecasts and the last-known NYSM
    observations (persisted by the sequencer).  To match that during
    inference we filter:

    * HRRR rows over ``[now - (sequence_length - 1)h, now + fh]``
    * NYSM rows over ``[now - (sequence_length - 1)h, now]``

    and merge with a LEFT join (HRRR <- NYSM).  The future HRRR rows
    survive with NaN in the NYSM columns; the sequencer overwrites
    those NaNs with the last past NYSM observation (architectural
    persistence) before the model sees them.

    The returned dataframe is in RAW units.  Per-window z-score
    normalization is performed by the sequencer using past-only
    statistics, so training and inference share identical
    normalization without persisting any stats files.
    """
    if not isinstance(nysm_df, cudf.DataFrame):
        nysm_df = cudf.from_pandas(nysm_df)
    if not isinstance(hrrr_df, cudf.DataFrame):
        hrrr_df = cudf.from_pandas(hrrr_df)

    hrrr_df = hrrr_df.sort_values("valid_time")
    hrrr_df["valid_time"] = cudf.to_datetime(hrrr_df["valid_time"])

    if not train:
        # Resolve `now` (the actual inference time).  HRRR's parquet
        # extends out to `now + fh`, so falling back to the max minus fh
        # gives us the right anchor.
        if now is None:
            now_ts = pd.Timestamp(hrrr_df["valid_time"].max()) - pd.Timedelta(hours=fh)
        else:
            now_ts = pd.Timestamp(now)

        past_start = now_ts - pd.Timedelta(hours=sequence_length - 1)
        future_end = now_ts + pd.Timedelta(hours=fh)

        # Keep HRRR rows over [past_start, future_end] so the encoder
        # sees real future HRRR forecasts in the future portion.
        hrrr_df = hrrr_df[
            (hrrr_df["valid_time"] >= past_start)
            & (hrrr_df["valid_time"] <= future_end)
        ]

        # NYSM observations only exist up through `now`.
        nysm_df = nysm_df[
            (nysm_df["valid_time"] >= past_start)
            & (nysm_df["valid_time"] <= now_ts)
        ]

    stations = get_closest_nysm_stations.get_closest_stations_csv(station)
    hrrr_df1 = hrrr_df[hrrr_df["station"].isin(stations)]
    nysm_df1 = nysm_df[nysm_df["station"].isin(stations)]

    geo_df_pd = None
    if not skip_landtype_geo():
        try:
            geo_df = cudf.read_csv(LEGACY_LSTM_CLUSTERS_CSV)
            geo_df_pd = geo_df.to_pandas()
        except FileNotFoundError:
            warnings.warn(
                f"{LEGACY_LSTM_CLUSTERS_CSV} not found; continuing without "
                "lulc_cat / elev_cat / slope_cat.",
                stacklevel=2,
            )
    if geo_df_pd is not None:
        # Map geo features by station name (round-trip via pandas to avoid the
        # positional-indexing bug the previous version had).
        hrrr_df1_pd = hrrr_df1.to_pandas()
        for col in ["lulc_cat", "elev_cat", "slope_cat"]:
            hrrr_df1_pd = _map_geo_column(geo_df_pd, hrrr_df1_pd, col)
        hrrr_df1 = cudf.from_pandas(hrrr_df1_pd)

    _hrrr_drop = [
        "index",
        "lead time",
        "lsm",
        "latitude",
        "longitude",
        "time",
        "orog",
    ]
    _present = [c for c in _hrrr_drop if c in hrrr_df1.columns]
    if _present:
        hrrr_df1 = hrrr_df1.drop(columns=_present)

    master_df = dataframe_wrapper(stations, hrrr_df1)
    master_df2 = dataframe_wrapper(stations, nysm_df1)

    # LEFT join (HRRR <- NYSM) at inference so the future HRRR rows
    # survive with NaN in the NYSM columns.  The sequencer overwrites the
    # trailing 64 NYSM columns of those future rows with the last past
    # observation (architectural persistence).
    merge_how = "inner" if train else "left"
    master_df = master_df.merge(
        master_df2, on="valid_time", how=merge_how, suffixes=(None, "_xab")
    )
    master_df = master_df.sort_values("valid_time").reset_index(drop=True)

    the_df = get_error.nwp_error(metvar, station, master_df.to_pandas())
    valid_times = the_df["valid_time"].tolist()
    the_df = encode.encode(the_df, "valid_time", 366)
    the_df = the_df[the_df.columns.drop(list(the_df.filter(regex="station")))]
    new_df = the_df.drop(columns="valid_time")

    # Leave `new_df` in raw units - the sequencer normalizes each
    # (sequence_length + forecast_steps) window using its own past
    # statistics.  The future-row NYSM cells remain NaN here; they get
    # overwritten by the sequencer's persistence step downstream.

    features = [c for c in new_df.columns if c != "target_error" and "images" not in c]
    target = "target_error"
    lstm_df = new_df.copy()

    return lstm_df, features, stations, target, valid_times
