"""
app/data/fetch_asos.py
======================
Download ASOS station observations from the Iowa Environmental Mesonet (IEM)
REST API, resample them to 1-hourly intervals, and write per-year parquets
that mimic the schema produced by
``src/data_cleaning/get_resampled_nysm_data.py``.

The output parquets are written to::

    {parquets_dir}/mesonet/mesonet_1H_obs_{YYYY}.parquet

Each parquet stores ``station`` and ``time_1H`` as columns (or a legacy
multi-index on read) with the following data columns (matching the NYSM
schema consumed by ``src/model_data/nysm_data.py``):

    lat, lon, elev, tair, ta9m, td, relh, srad, pres, mslp,
    wspd_sonic, wspd_sonic_mean, wmax_sonic, wdir_sonic,
    precip_total, snow_depth

.. note::
   ASOS does not report solar radiation (``srad``), 9-m temperature
   (``ta9m``), sonic-anemometer wind, or snow depth with the same
   precision as NYSM.  Missing ASOS columns are filled with ``-999``
   so the downstream pipeline does not break.

IEM API reference: https://mesonet.agron.iastate.edu/request/asos/

Usage
-----
From a notebook::

    from app.data.fetch_asos import fetch_asos_for_bbox
    from app.utils.config_loader import load_config
    cfg = load_config()
    fetch_asos_for_bbox(cfg, cfg.training.start_date, cfg.training.end_date)
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# IEM ASOS network inventory endpoint
_ASOS_NETWORK_URL = "https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson"
# IEM ASOS 1-minute data endpoint
_ASOS_DATA_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Maximum date range per IEM request (to avoid timeouts)
_CHUNK_DAYS = 30

# Retry settings
_MAX_RETRIES = 3
_RETRY_DELAY = 5  # seconds


def _get_asos_networks_for_states(states: list[str]) -> list[str]:
    """Return a list of IEM network codes for the given US state abbreviations."""
    return [f"{s}_ASOS" for s in states]


def _fetch_station_inventory(network: str) -> pd.DataFrame:
    """
    Download station metadata (id, lat, lon, elevation, name) for one IEM
    network, returning a DataFrame with columns:
    ``station``, ``lat``, ``lon``, ``elev``, ``name``.
    """
    url = _ASOS_NETWORK_URL.format(network=network)
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY)
            else:
                logger.warning("Could not fetch inventory for %s: %s", network, exc)
                return pd.DataFrame()

    try:
        import json
        data = json.loads(resp.text)
        records = []
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [None, None])
            records.append(
                {
                    "station": props.get("sid", ""),
                    "name": props.get("sname", ""),
                    "lat": float(coords[1]) if coords[1] is not None else np.nan,
                    "lon": float(coords[0]) if coords[0] is not None else np.nan,
                    "elev": float(props.get("elevation", np.nan)),
                    "network": network,
                }
            )
        return pd.DataFrame(records)
    except Exception as exc:
        logger.warning("Failed to parse inventory for %s: %s", network, exc)
        return pd.DataFrame()


def _states_in_bbox(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> list[str]:
    """
    Return a hard-coded list of US state abbreviations whose bounding box
    overlaps the requested box.  Extend this list as needed for other regions.

    The function uses a simple look-up table of approximate state bounding
    boxes — it intentionally errs on the side of including more states so
    that border stations are not silently dropped.
    """
    # Approximate (lat_min, lat_max, lon_min, lon_max) per state
    _STATE_BOXES: dict[str, tuple[float, float, float, float]] = {
        "NY": (40.5, 45.0, -79.8, -71.8),
        "PA": (39.7, 42.3, -80.5, -74.7),
        "NJ": (38.9, 41.4, -75.6, -73.9),
        "CT": (41.0, 42.1, -73.7, -71.8),
        "MA": (41.2, 42.9, -73.5, -69.9),
        "VT": (42.7, 45.0, -73.4, -71.5),
        "NH": (42.7, 45.3, -72.6, -70.6),
        "ME": (43.1, 47.5, -71.1, -67.0),
        "RI": (41.1, 42.0, -71.9, -71.2),
        "OH": (38.4, 42.3, -84.8, -80.5),
        "MI": (41.7, 48.3, -90.4, -82.4),
        "WV": (37.2, 40.6, -82.6, -77.7),
        "MD": (37.9, 39.7, -79.5, -75.0),
        "DE": (38.4, 39.8, -75.8, -75.0),
        "VA": (36.5, 39.5, -83.7, -75.2),
    }
    matched = []
    for state, (slat_min, slat_max, slon_min, slon_max) in _STATE_BOXES.items():
        if slat_max < lat_min or slat_min > lat_max:
            continue
        if slon_max < lon_min or slon_min > lon_max:
            continue
        matched.append(state)
    return matched if matched else ["NY"]


def get_asos_stations_in_bbox(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> pd.DataFrame:
    """
    Return metadata for all ASOS stations whose location falls inside the
    given bounding box.

    Parameters
    ----------
    lat_min, lat_max, lon_min, lon_max:
        Bounding box in decimal degrees.

    Returns
    -------
    pd.DataFrame
        Columns: ``station``, ``name``, ``lat``, ``lon``, ``elev``,
        ``network``.  Rows are sorted by station id.
    """
    states = _states_in_bbox(lat_min, lat_max, lon_min, lon_max)
    frames = []
    for state in states:
        network = f"{state}_ASOS"
        df_inv = _fetch_station_inventory(network)
        if not df_inv.empty:
            frames.append(df_inv)

    if not frames:
        return pd.DataFrame(columns=["station", "name", "lat", "lon", "elev", "network"])

    all_stations = pd.concat(frames, ignore_index=True)
    mask = (
        (all_stations["lat"] >= lat_min)
        & (all_stations["lat"] <= lat_max)
        & (all_stations["lon"] >= lon_min)
        & (all_stations["lon"] <= lon_max)
    )
    return all_stations[mask].drop_duplicates("station").sort_values("station").reset_index(drop=True)


def _fetch_asos_chunk(
    stations: list[str],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Download ASOS 1-minute observations from the IEM API for *stations*
    between *start* and *end* and return a raw DataFrame.

    Parameters
    ----------
    stations:
        List of ASOS station identifiers (e.g. ``["KBUF", "KALB"]``).
    start, end:
        UTC datetime window (inclusive).

    Returns
    -------
    pd.DataFrame
        Raw comma-separated output parsed from the IEM response.
    """
    params = {
        "station": stations,
        "data": [
            "tmpf", "dwpf", "relh", "drct", "sknt", "gust",
            "p01i", "alti", "mslp", "vsby", "skyc1", "skyl1",
        ],
        "year1": start.year, "month1": start.month, "day1": start.day,
        "hour1": start.hour, "minute1": start.minute,
        "year2": end.year, "month2": end.month, "day2": end.day,
        "hour2": end.hour, "minute2": end.minute,
        "tz": "UTC",
        "format": "comma",
        "latlon": "yes",
        "elev": "yes",
        "missing": "M",
        "trace": "0.0001",
        "direct": "yes",
        "report_type": [1, 3],  # 1=routine, 3=special
    }

    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(_ASOS_DATA_URL, params=params, timeout=120)
            resp.raise_for_status()
            # IEM returns a text/csv file with a comment header
            text = resp.text
            # Skip lines starting with '#'
            lines = [l for l in text.splitlines() if not l.startswith("#")]
            if len(lines) <= 1:
                return pd.DataFrame()
            return pd.read_csv(StringIO("\n".join(lines)), low_memory=False)
        except requests.RequestException as exc:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY)
            else:
                logger.warning("IEM request failed: %s", exc)
                return pd.DataFrame()
    return pd.DataFrame()


def _fahrenheit_to_celsius(f: pd.Series) -> pd.Series:
    return (f - 32.0) * 5.0 / 9.0


def _knots_to_ms(k: pd.Series) -> pd.Series:
    return k * 0.514444


def _inhg_to_hpa(p: pd.Series) -> pd.Series:
    return p * 33.8639


def _resample_station(df_station: pd.DataFrame) -> pd.DataFrame:
    """
    Resample one station's raw ASOS observations to top-of-the-hour (1H),
    producing a DataFrame with an index named ``time_1H``.

    Aggregation rules
    -----------------
    * ``precip_total`` — hourly sum of per-observation increments
    * ``wspd_sonic`` / ``wmax_sonic`` — max over the hour (conservative for gusts)
    * ``wspd_sonic_mean`` — mean wind speed over the hour
    * all other variables — instantaneous value at the top of each hour
    """
    df_station = df_station.copy()
    df_station["valid_time"] = pd.to_datetime(df_station["valid_time"], utc=True, errors="coerce")
    df_station = df_station.dropna(subset=["valid_time"])
    df_station = df_station.set_index("valid_time").sort_index()

    # ---- Unit conversions ----
    for col_f, col_c in [("tmpf", "tair"), ("dwpf", "td")]:
        if col_f in df_station.columns:
            df_station[col_c] = _fahrenheit_to_celsius(
                pd.to_numeric(df_station[col_f], errors="coerce")
            )

    for col_k, col_m in [("sknt", "wspd_sonic"), ("gust", "wmax_sonic")]:
        if col_k in df_station.columns:
            df_station[col_m] = _knots_to_ms(
                pd.to_numeric(df_station[col_k], errors="coerce")
            )

    if "alti" in df_station.columns:
        df_station["pres"] = _inhg_to_hpa(
            pd.to_numeric(df_station["alti"], errors="coerce")
        )

    if "mslp" in df_station.columns:
        df_station["mslp"] = pd.to_numeric(df_station["mslp"], errors="coerce")

    if "relh" in df_station.columns:
        df_station["relh"] = pd.to_numeric(df_station["relh"], errors="coerce")

    if "p01i" in df_station.columns:
        # p01i = precipitation over last 1 minute (inches)
        precip_inch = pd.to_numeric(df_station["p01i"], errors="coerce").fillna(0.0)
        # Convert inches to mm and sum over the hour
        df_station["precip_total"] = precip_inch * 25.4

    if "drct" in df_station.columns:
        df_station["wdir_sonic"] = pd.to_numeric(df_station["drct"], errors="coerce")

    # ---- Per-variable resampling to 1H top-of-hour ----
    top_of_hour = df_station.resample("1h", label="right", closed="right").last()
    top_of_hour.index.name = "time_1H"

    precip_1h = df_station["precip_total"].resample("1h", label="right", closed="right").sum()
    top_of_hour["precip_total"] = precip_1h

    if "wspd_sonic" in df_station.columns:
        top_of_hour["wspd_sonic_mean"] = (
            df_station["wspd_sonic"].resample("1h", label="right", closed="right").mean()
        )
        top_of_hour["wmax_sonic"] = (
            df_station["wmax_sonic"].resample("1h", label="right", closed="right").max()
            if "wmax_sonic" in df_station.columns
            else np.nan
        )

    return top_of_hour


def _build_station_df(
    raw: pd.DataFrame,
    station_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert raw IEM output to the NYSM-compatible 1H schema.

    Parameters
    ----------
    raw:
        Raw DataFrame returned by :func:`_fetch_asos_chunk`.
    station_meta:
        Metadata DataFrame with ``station``, ``lat``, ``lon``, ``elev``.

    Returns
    -------
    pd.DataFrame
        Multi-indexed ``(station, time_1H)`` with NYSM-compatible columns.
    """
    if raw.empty:
        return pd.DataFrame()

    # IEM returns columns: station, valid, lat, lon, elevation, tmpf, …
    raw = raw.rename(
        columns={
            "valid": "valid_time",
            "elevation": "elev_raw",
        }
    )

    frames = []
    for stid, grp in raw.groupby("station"):
        if stid not in station_meta["station"].values:
            continue
        meta_row = station_meta[station_meta["station"] == stid].iloc[0]
        resampled = _resample_station(grp)
        resampled["station"] = stid
        resampled["lat"] = meta_row["lat"]
        resampled["lon"] = meta_row["lon"]
        resampled["elev"] = meta_row["elev"]
        frames.append(resampled)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames)
    out = out.reset_index().set_index(["station", "time_1H"])

    # ---- Ensure all expected NYSM columns are present ----
    nysm_cols = {
        "lat": np.nan,
        "lon": np.nan,
        "elev": np.nan,
        "tair": -999.0,
        "ta9m": -999.0,   # not measured by ASOS; filled as missing
        "td": -999.0,
        "relh": -999.0,
        "srad": -999.0,   # not measured by ASOS; filled as missing
        "pres": -999.0,
        "mslp": -999.0,
        "wspd_sonic": -999.0,
        "wspd_sonic_mean": -999.0,
        "wmax_sonic": -999.0,
        "wdir_sonic": -999.0,
        "precip_total": 0.0,
        "snow_depth": -999.0,  # not reliably measured by all ASOS stations
    }
    for col, fill in nysm_cols.items():
        if col not in out.columns:
            out[col] = fill
        else:
            out[col] = out[col].fillna(fill)

    return out[list(nysm_cols.keys())]


def _flatten_mesonet_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Expand ``(station, time_1H)`` index to columns; normalize timestamps."""
    out = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df.copy()
    if "time_1H" in out.columns:
        out["time_1H"] = pd.to_datetime(out["time_1H"], utc=True)
    return out


def _mesonet_from_parquet(path: Path) -> pd.DataFrame:
    """Load a mesonet parquet written by this module (flat or legacy MultiIndex)."""
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.MultiIndex):
        return df
    if {"station", "time_1H"}.issubset(df.columns):
        return df.set_index(["station", "time_1H"])
    raise ValueError(
        f"Unexpected mesonet parquet schema at {path}: "
        "expected columns station and time_1H or a MultiIndex."
    )


def _dataframe_to_arrow_table(out: pd.DataFrame):
    """
    Build a PyArrow table without calling ``DataFrame.to_parquet`` or
    ``pa.Table.from_pandas``, avoiding pandas extension-type registration
    bugs (e.g. ``ArrowKeyError: pandas.period already defined`` in Jupyter).
    """
    import pyarrow as pa

    arrays: list = []
    names: list[str] = []
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            s_dt = pd.to_datetime(s, utc=True)
            naive = s_dt.dt.tz_convert("UTC").dt.tz_localize(None)
            arrays.append(pa.array(naive.astype("datetime64[ns]")))
        elif pd.api.types.is_bool_dtype(s):
            arrays.append(pa.array(s.tolist(), type=pa.bool_()))
        elif pd.api.types.is_integer_dtype(s) and not s.isna().any():
            arrays.append(pa.array(s.astype("int64").to_numpy(), type=pa.int64()))
        elif pd.api.types.is_numeric_dtype(s):
            arrays.append(pa.array(s.astype("float64").to_numpy(), type=pa.float64()))
        else:
            arrays.append(
                pa.array(
                    [None if pd.isna(x) else str(x) for x in s.tolist()],
                    type=pa.large_string(),
                )
            )
        names.append(col)
    return pa.Table.from_arrays(arrays, names=names)


def _write_mesonet_parquet(df: pd.DataFrame, out_path: Path) -> None:
    """
    Write mesonet data to parquet.

    Uses flat columns (no MultiIndex in the file) so downstream
    ``reset_index()`` paths stay correct.  Writes via PyArrow built from
    plain arrays — never ``DataFrame.to_parquet`` — so notebook kernels
    do not hit pandas/pyarrow extension registration errors.
    """
    import pyarrow.parquet as pq

    out = _flatten_mesonet_for_parquet(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = _dataframe_to_arrow_table(out)
    pq.write_table(table, out_path)


def fetch_asos_for_bbox(
    cfg,
    start_date: date,
    end_date: date,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Download ASOS observations for every station inside the bounding box
    defined in *cfg*, resample to 1H, and write per-year parquets to::

        {cfg.data.parquets_dir}/mesonet/mesonet_1H_obs_{YYYY}.parquet

    The function is *resume-friendly*: existing parquets are read back and
    only new months are appended.

    A station metadata CSV is also written to::

        {cfg.output.models_dir}/lookups/station_meta.csv

    for use by :func:`app.data.fetch_hrrr.fetch_and_stage` and
    :func:`app.data.build_station_clusters.build_clusters`.

    Parameters
    ----------
    cfg:
        :class:`~app.utils.config_loader.AppConfig` loaded from ``config.yaml``.
    start_date, end_date:
        Inclusive date range.
    show_progress:
        Print status messages.

    Returns
    -------
    pd.DataFrame
        Station metadata (``station``, ``lat``, ``lon``, ``elev``, ``name``)
        for all stations found inside the bbox.
    """
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()

    bbox = cfg.bbox
    if show_progress:
        print(
            f"Searching ASOS stations in "
            f"lat=[{bbox.lat_min}, {bbox.lat_max}] "
            f"lon=[{bbox.lon_min}, {bbox.lon_max}] …"
        )

    station_meta = get_asos_stations_in_bbox(
        bbox.lat_min, bbox.lat_max, bbox.lon_min, bbox.lon_max
    )
    if station_meta.empty:
        raise RuntimeError(
            "No ASOS stations found inside the bounding box. "
            "Check your bbox coordinates in config.yaml."
        )

    if show_progress:
        print(f"  Found {len(station_meta)} ASOS stations.")

    # Save metadata for downstream use
    lookups_dir = Path(cfg.output.models_dir) / "lookups"
    lookups_dir.mkdir(parents=True, exist_ok=True)
    meta_path = lookups_dir / "station_meta.csv"
    station_meta.to_csv(meta_path, index=False)
    if show_progress:
        print(f"  Station metadata saved to {meta_path}")

    out_dir = Path(cfg.data.parquets_dir) / "mesonet"
    out_dir.mkdir(parents=True, exist_ok=True)

    station_ids = station_meta["station"].tolist()

    # Fetch in _CHUNK_DAYS chunks, accumulate per year
    year_frames: dict[int, list[pd.DataFrame]] = {}

    start_dt = datetime(start_date.year, start_date.month, start_date.day)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, 23, 59)

    chunk_start = start_dt
    while chunk_start <= end_dt:
        chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS - 1), end_dt)
        if show_progress:
            print(
                f"  [fetch] {chunk_start.date()} → {chunk_end.date()} "
                f"({len(station_ids)} stations) …",
                end="",
                flush=True,
            )
        raw = _fetch_asos_chunk(station_ids, chunk_start, chunk_end)
        if not raw.empty:
            df_chunk = _build_station_df(raw, station_meta)
            if not df_chunk.empty:
                for year, grp in df_chunk.groupby(level="time_1H", group_keys=False):
                    yr = pd.Timestamp(year).year
                    year_frames.setdefault(yr, []).append(grp)
                if show_progress:
                    print(f" {len(df_chunk)} rows.")
            else:
                if show_progress:
                    print(" (empty after resampling)")
        else:
            if show_progress:
                print(" (no data)")
        chunk_start = chunk_end + timedelta(days=1)

    # Write per-year parquets
    for year, frames in year_frames.items():
        out_path = out_dir / f"mesonet_1H_obs_{year}.parquet"
        new_df = pd.concat(frames)
        new_df = new_df[~new_df.index.duplicated(keep="last")]
        if out_path.exists():
            existing = _mesonet_from_parquet(out_path)
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            _write_mesonet_parquet(combined, out_path)
        else:
            _write_mesonet_parquet(new_df.sort_index(), out_path)
        if show_progress:
            print(f"  Written: {out_path} ({len(new_df)} rows for {year})")

    return station_meta


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from app.utils.config_loader import load_config

    cfg = load_config()
    fetch_asos_for_bbox(cfg, cfg.training.start_date, cfg.training.end_date)
