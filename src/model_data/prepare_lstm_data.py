"""Pandas-based data prep for LSTM training.

For a given station and target variable, this module:

1. Selects the four nearest NYSM stations (the central station + the
   three closest neighbours via `get_closest_nysm_stations`).
2. Pivots HRRR forecasts and NYSM observations from long to wide,
   producing one column set per neighbour.
3. Merges HRRR + NYSM on `valid_time`, optionally attaches geographic features
   (`lulc_cat`, `elev_cat`, `slope_cat`) from lstm_clusters.csv unless
   :envvar:`FORECAST_APP_SKIP_GEO` is set, encodes `valid_time` with
   sin/cos features, and computes the target column `target_error =
   NWP - NYSM` (`get_error.nwp_error`).
4. Returns a 5-tuple `(lstm_df, features, stations, target, valid_times)`
   ready to be wrapped by `model_architecture.sequencer.SequenceDatasetMultiTask`.

Important: the returned dataframe is in RAW units.  Z-score
normalization is performed inside the sequencer, on a per-window
basis, using stats from the past portion of each window only.  See
`model_architecture/sequencer.py` for the rationale.
"""

import sys

sys.path.append("..")

import warnings

import pandas as pd

from model_data import (
    encode,
    get_closest_nysm_stations,
    get_error,
)
from model_data.skip_landtype_geo import LEGACY_LSTM_CLUSTERS_CSV, skip_landtype_geo


def create_geo_dict(geo_df, c, df1):
    """Map column `c` from `geo_df` onto `df1` keyed by `station`.

    Mapping by station name (rather than by row position) prevents
    silent misalignment when `geo_df` and `df1` have different lengths.
    """
    geo_dict = dict(zip(geo_df["station"], geo_df[c]))
    df1[c] = df1["station"].map(geo_dict)
    return df1


def columns_drop_hrrr(df):
    """Drop HRRR housekeeping columns that aren't model features."""
    return df.drop(
        columns=[
            "index",
            "lead time",
            "lsm",
            "latitude",
            "longitude",
            "time",
            "orog",
        ],
        errors="ignore",
    )


def columns_drop_nysm(df):
    """Drop NYSM columns that aren't used as model features."""
    return df.drop(
        columns=[
            "ta9m",
            "td",
            "relh",
            "srad",
            "pres",
            "mslp",
            "wspd_sonic_mean",
            "wspd_sonic",
            "wmax_sonic",
            "wdir_sonic",
            "snow_depth",
            "precip_total",
        ],
        errors="ignore",
    )


def add_suffix(master_df, station):
    """Append `_{station}` to every column except `valid_time`/`time`."""
    cols = ["valid_time", "time"]
    return master_df.rename(
        columns={c: c + f"_{station}" for c in master_df.columns if c not in cols}
    )


def dataframe_wrapper(stations, df):
    """Long-to-wide pivot: one row per `valid_time`, one set of columns
    per station."""
    master_df = df[df["station"] == stations[0]]
    master_df = add_suffix(master_df, stations[0])
    for station in stations[1:]:
        df1 = df[df["station"] == station]
        df1 = add_suffix(df1, station)
        master_df = master_df.merge(
            df1, on="valid_time", suffixes=(None, f"_{station}")
        )
    return master_df


def prepare_lstm_data(nysm_df, hrrr_df, station, metvar, fh, train=False):
    """Build the per-station training dataframe for the LSTM.

    Parameters
    ----------
    nysm_df : pandas.DataFrame
        NYSM observations for the relevant year (long format).
    hrrr_df : pandas.DataFrame
        HRRR forecasts for the relevant `fh` (long format).
    station : str
        Central NYSM station id.
    metvar : str
        Target variable for `target_error` (e.g. `t2m`, `u_total`,
        `tp`).
    fh : int
        Forecast hour the data corresponds to (used for naming /
        bookkeeping by callers).
    train : bool
        If True, use the full date range in `hrrr_df`; if False,
        restrict to the past 30 hours up to `hrrr_df['valid_time'].max()`.

    Returns
    -------
    lstm_df : pandas.DataFrame
        Wide dataframe with one row per `valid_time`, suitable for
        sequencing.
    features : list[str]
        Feature column names (excludes `target_error`, `valid_time`,
        and any image-path columns).
    stations : list[str]
        The neighbouring stations used to build the wide frame.
    target : str
        The name of the target column (always `"target_error"`).
    valid_times : list[pd.Timestamp]
        The `valid_time` values present in `lstm_df`, in order.
    """
    hrrr_df = hrrr_df.sort_values("valid_time")
    hrrr_df["valid_time"] = pd.to_datetime(hrrr_df["valid_time"])

    current_time = hrrr_df["valid_time"].max()

    if train is False:
        # Filter the previous 30 hours including the current hour.
        mask = (hrrr_df["valid_time"] >= current_time - pd.Timedelta(hours=29)) & (
            hrrr_df["valid_time"] <= current_time
        )
        filtered_times = hrrr_df.loc[mask, "valid_time"]
        mytimes = filtered_times.tolist()
        nysm_df = nysm_df[nysm_df["valid_time"].isin(mytimes)]
    else:
        mytimes = hrrr_df["valid_time"].tolist()

    stations = get_closest_nysm_stations.get_closest_stations_csv(station)

    hrrr_df1 = hrrr_df[hrrr_df["station"].isin(stations)].copy()
    nysm_df1 = nysm_df[nysm_df["station"].isin(stations)].copy()

    geo_df = None
    if not skip_landtype_geo():
        try:
            geo_df = pd.read_csv(LEGACY_LSTM_CLUSTERS_CSV)
        except FileNotFoundError:
            warnings.warn(
                f"{LEGACY_LSTM_CLUSTERS_CSV} not found (FORECAST_APP_SKIP_GEO=0 or "
                "use_geo_encoding); continuing without lulc_cat / elev_cat / slope_cat.",
                stacklevel=2,
            )
    if geo_df is not None:
        # Map geo columns by station name (the previous code aligned a sliced
        # geo_df1 by row position, which silently produced wrong rows whenever
        # geo_df1 had a different length than hrrr_df1).
        hrrr_df1 = create_geo_dict(geo_df, "lulc_cat", hrrr_df1)
        hrrr_df1 = create_geo_dict(geo_df, "elev_cat", hrrr_df1)
        hrrr_df1 = create_geo_dict(geo_df, "slope_cat", hrrr_df1)

    hrrr_df1 = columns_drop_hrrr(hrrr_df1)

    # Long -> wide pivot per data source.
    master_df = dataframe_wrapper(stations, hrrr_df1)
    master_df2 = dataframe_wrapper(stations, nysm_df1)

    # Inner join on valid_time gives us one row per timestamp with
    # both HRRR and NYSM columns for every neighbour.
    master_df = master_df.merge(master_df2, on="valid_time", suffixes=(None, "_xab"))

    # Compute the per-row forecast error (target_error = NWP - NYSM).
    the_df = get_error.nwp_error(metvar, station, master_df)
    valid_times = the_df["valid_time"].tolist()

    # Replace `valid_time` with sin/cos encodings of day-of-year
    # (period = 366 to handle leap years gracefully).
    the_df = encode.encode(the_df, "valid_time", 366)
    the_df = the_df[the_df.columns.drop(list(the_df.filter(regex="station")))]

    new_df = the_df.drop(columns="valid_time")

    # The dataframe stays in RAW units; the sequencer z-scores each
    # window using past-only statistics.
    features = [
        c
        for c in new_df.columns
        if c != "target_error" and c != "valid_time" and "images" not in c
    ]

    lstm_df = new_df.copy()
    target = "target_error"

    return lstm_df, features, stations, target, valid_times
