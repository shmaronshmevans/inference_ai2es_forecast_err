"""Evaluate inference output against the realised forecast error.

For every `(forecast_hour, metvar, station)` combination this script:

1. Reads the prediction parquet written by `lstm_s2s_engine.py`
   (`fh{fh}_{metvar}_inference_out.parquet`).
2. Re-derives the *actual* forecast error at `valid_time == now` using
   `get_error.nwp_error` against the freshest HRRR + NYSM data.
3. Appends one row per station to
   `fh{fh}_{metvar}_error_metrics.parquet` with the realised error
   (`actual`), the prediction (`model_output`), and their difference.

Run this AFTER `pipeline.py` (or `lstm_s2s_engine.py`) has produced
predictions for the current hour.

CLI
---
    python eval_model_metrics.py
"""

import sys

sys.path.append("..")

from datetime import datetime

import warnings

import numpy as np
import pandas as pd

from model_data import (
    encode,
    get_closest_nysm_stations,
    get_error,
    hrrr_data,
    nysm_data,
)
from model_data.skip_landtype_geo import LEGACY_LSTM_CLUSTERS_CSV, skip_landtype_geo


# Output root used by `lstm_s2s_engine.py`.  Override via env var if
# your filesystem layout differs.
import os

OUT_DIR = os.environ.get("LSTM_OUTPUT_DIR", "/home/aevans/inference/FINAL_OUTPUT")


def create_geo_dict(geo_df, c, df1):
    """Map column `c` from `geo_df` onto `df1` by station name."""
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


def prepare_lstm_data(nysm_df, hrrr_df, station, metvar, fh=None, train=False):
    """Build the joined HRRR+NYSM dataframe and add the realised error.

    This is a stripped-down sibling of
    `prepare_lstm_data.prepare_lstm_data`: it does the same join and
    error-derivation but skips feature engineering (encoding,
    normalization) since this helper only needs `target_error` for
    evaluation.

    `fh` is accepted for signature parity with the training/inference
    helpers and is unused here.
    """
    hrrr_df = hrrr_df.sort_values("valid_time")
    hrrr_df["valid_time"] = pd.to_datetime(hrrr_df["valid_time"])

    current_time = hrrr_df["valid_time"].max()

    if not train:
        # Past 30 hours including the current hour.
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
                f"{LEGACY_LSTM_CLUSTERS_CSV} not found; continuing without geo columns.",
                stacklevel=2,
            )
    if geo_df is not None:
        # Map geo features by station name.
        hrrr_df1 = create_geo_dict(geo_df, "lulc_cat", hrrr_df1)
        hrrr_df1 = create_geo_dict(geo_df, "elev_cat", hrrr_df1)
        hrrr_df1 = create_geo_dict(geo_df, "slope_cat", hrrr_df1)

    hrrr_df1 = columns_drop_hrrr(hrrr_df1)

    master_df = dataframe_wrapper(stations, hrrr_df1)
    master_df2 = dataframe_wrapper(stations, nysm_df1)
    master_df = master_df.merge(master_df2, on="valid_time", suffixes=(None, "_xab"))

    # Add the realised forecast error column (`target_error`).
    return get_error.nwp_error(metvar, station, master_df)


def main(now):
    """Score predictions made for `now` against the realised error.

    Parameters
    ----------
    now : datetime.datetime
        The valid_time of the predictions to evaluate (typically the
        rounded-to-hour wall-clock time the predictions were written).
    """
    year = now.year
    nysm_meta = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
    stations = nysm_meta["stid"].unique()

    nysm_df = nysm_data.load_nysm_data(year).reset_index(drop=True)

    for fh in np.arange(1, 19):
        for metvar in ["t2m", "u_total", "tp"]:
            guess_df = pd.read_parquet(
                f"{OUT_DIR}/fh{fh}_{metvar}_inference_out.parquet"
            ).reset_index()
            error_df = pd.read_parquet(
                f"{OUT_DIR}/fh{fh}_{metvar}_error_metrics.parquet"
            ).reset_index()

            # Load HRRR forecasts for this fh once per (fh, metvar).
            hrrr_df = hrrr_data.read_hrrr_data(str(fh).zfill(2), year)

            outs = []
            for station in stations:
                known_df = prepare_lstm_data(
                    nysm_df, hrrr_df, station, metvar, fh=fh
                )
                known_df = known_df[known_df["valid_time"] == now]
                if known_df.empty:
                    continue
                known = known_df["target_error"].iloc[0]

                filtered_lstm = guess_df[guess_df["valid_time"] == now]
                if filtered_lstm.empty:
                    continue
                lstm_output = filtered_lstm["model_output"].iloc[0]

                outs.append(
                    {
                        "valid_time": now,
                        "stid": station,
                        "actual": known,
                        "model_output": lstm_output,
                        "difference": lstm_output - known,
                    }
                )

            if not outs:
                continue
            child = pd.DataFrame(outs)
            error_df = pd.concat([error_df, child], ignore_index=True)
            error_df.set_index(["valid_time", "stid"], inplace=True)
            error_df.to_parquet(
                f"{OUT_DIR}/fh{fh}_{metvar}_error_metrics.parquet"
            )


if __name__ == "__main__":
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    try:
        main(now)
    except Exception as e:
        print("Runtime Error:", str(e))
        raise
