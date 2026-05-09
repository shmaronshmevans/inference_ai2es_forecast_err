"""
app/data/fetch_hrrr.py
======================
Download HRRR grib2 files with ``herbie-data``, extract the variables the
forecast-error pipeline needs, and write per-forecast-hour parquets in the
same layout that ``src/model_data/hrrr_data.py`` already knows how to read.

Output layout
-------------
For each forecast hour *fh* and each calendar month:

    {parquets_dir}/hrrr_data/fh{fh:02d}/
        HRRR_{YYYY}_{MM}_direct_compare_to_nysm_sites_mask_water.parquet

Each parquet has the following columns (matching the schema produced by
``src/data_cleaning/all_models_comparison_to_mesos_lstm.py``):

    valid_time, time (init time), station, latitude, longitude,
    t2m, d2m, sh2, r2, u10, v10, u_total, u_dir,
    tp, mslma, orog, tcc, asnow, cape, dswrf, dlwrf, gh, lsm,
    lead time

Herbie / what lands in ``hrrr_raw_dir``
---------------------------------------
This module calls ``Herbie.xarray(...)`` with per-variable search strings. That path
downloads the **``.idx``** inventory first, then **HTTP byte-range** slices of the
remote GRIB (often written as temporary ``subset_*`` files), not necessarily the full
canonical ``hrrr.*.grib2`` filename. After cfgrib reads a subset, Herbie typically
**removes** it again (``remove_grib=True``), so the folder often looks like “mostly
``.idx``”. Full-archive downloads require ``Herbie.download()`` **without** subsetting
and are much larger—only needed if you want intact GRIBs for other tools.

Usage
-----
Run as a module from the ``app/`` directory::

    python -m app.data.fetch_hrrr

Or call from a notebook::

    from app.data.fetch_hrrr import fetch_and_stage
    from app.utils.config_loader import load_config
    cfg = load_config()
    fetch_and_stage(cfg, cfg.training.start_date, cfg.training.end_date)
"""

from __future__ import annotations

import logging
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HRRR variables we need from each grib2 file (herbie search strings).
# These map to the column names the rest of the pipeline expects.
# ---------------------------------------------------------------------------
_SEARCH_SPECS: list[dict] = [
    # (herbie search regex, variable nickname inside the extracted xarray)
    {"search": ":TMP:2 m above", "var": "t2m"},
    {"search": ":DPT:2 m above", "var": "d2m"},
    {"search": ":SPFH:2 m above", "var": "sh2"},
    {"search": ":RH:2 m above", "var": "r2"},
    {"search": ":UGRD:10 m above", "var": "u10"},
    {"search": ":VGRD:10 m above", "var": "v10"},
    {"search": ":ACPCP:|:APCP:|:PRATE:", "var": "tp"},
    {"search": ":PRMSL:", "var": "mslma"},
    {"search": ":HGT:surface", "var": "orog"},
    {"search": ":TCDC:entire", "var": "tcc"},
    {"search": ":DSWRF:surface", "var": "dswrf"},
    {"search": ":DLWRF:surface", "var": "dlwrf"},
    {"search": ":HGT:500 mb", "var": "gh"},
    {"search": ":CAPE:surface", "var": "cape"},
    {"search": ":LAND:surface", "var": "lsm"},
]


def _longitude_deg_to_180(lon: np.ndarray | pd.Series | float) -> np.ndarray:
    """
    Map longitude degrees to ``(-180, 180]``.

    HRRR GRIB uses 0–360° east (e.g. ~281° for Pennsylvania) while mesonet
    metadata uses negative west longitudes (e.g. −79°).  Without aligning
    conventions, merges on ``(lat, lon)`` fail and BallTree nearest-neighbour
    distances are meaningless.
    """
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def _kelvin_to_celsius(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert columns from Kelvin to Celsius in-place."""
    for c in cols:
        if c in df.columns:
            df[c] = df[c] - 273.15
    return df


def _compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add wind speed (``u_total``) and wind direction (``u_dir``) derived from
    the U and V components, using MetPy when available or a plain numpy
    fallback.
    """
    try:
        import metpy.calc as mpcalc
        from metpy.units import units as mpunits

        u = mpunits.Quantity(df["u10"].values, "m/s")
        v = mpunits.Quantity(df["v10"].values, "m/s")
        df["u_total"] = mpcalc.wind_speed(u, v).magnitude
        df["u_dir"] = mpcalc.wind_direction(u, v, convention="from").magnitude
    except Exception:
        df["u_total"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
        df["u_dir"] = (np.degrees(np.arctan2(-df["u10"], -df["v10"])) + 360) % 360
    return df


def _extract_grib2(
    herbie_obj,
    station_lookup: pd.DataFrame,
    init_time: datetime,
    fh: int,
    *,
    prefetch_full_grib: bool = False,
) -> pd.DataFrame | None:
    """
    Download one HRRR grib2 file and extract all required variables into a
    flat DataFrame with one row per station.

    Parameters
    ----------
    herbie_obj:
        A configured :class:`herbie.Herbie` instance for the requested
        init-time and forecast hour.
    station_lookup:
        DataFrame with columns ``station``, ``lat``, ``lon``.  The grib2 grid
        is subsetted to the nearest grid point for every station.
    init_time:
        Model initialisation time (UTC).
    fh:
        Forecast hour (integer).

    Returns
    -------
    pd.DataFrame or None
        Flat, per-station DataFrame, or ``None`` on any download/parse error.
    """
    try:
        from sklearn.neighbors import BallTree
    except ImportError:
        raise ImportError("scikit-learn is required — pip install scikit-learn")

    valid_time = init_time + timedelta(hours=fh)

    # Optional: materialize full GRIB once per init/fxx. Herbie uses overwrite=false;
    # skips network when the canonical file already exists under save_dir.
    if prefetch_full_grib:
        try:
            herbie_obj.download(errors="warn")
        except Exception as exc:
            logger.debug("Herbie full GRIB prefetch skipped or failed: %s", exc)

    frames: list[pd.DataFrame] = []
    for spec in _SEARCH_SPECS:
        try:
            ds = herbie_obj.xarray(spec["search"], overwrite=False)
            if ds is None:
                continue
            # herbie can return a Dataset or a DataArray depending on the field
            if hasattr(ds, "to_dataset"):
                ds = ds.to_dataset(name=spec["var"])
            df_grid = ds.to_dataframe().reset_index()

            # Normalise coordinate column names
            df_grid = df_grid.rename(
                columns={"latitude": "lat", "longitude": "lon"}
            )
            lat_col = "lat" if "lat" in df_grid.columns else "y"
            lon_col = "lon" if "lon" in df_grid.columns else "x"

            # Pick up the actual data column (first non-coord column)
            data_col = next(
                (
                    c
                    for c in df_grid.columns
                    if c not in {lat_col, lon_col, "time", "step", "valid_time"}
                ),
                None,
            )
            if data_col is None:
                continue

            df_grid = df_grid[[lat_col, lon_col, data_col]].rename(
                columns={data_col: spec["var"]}
            )
            df_grid[lon_col] = _longitude_deg_to_180(df_grid[lon_col])
            frames.append(df_grid)
        except Exception as exc:
            logger.debug("Could not extract %s: %s", spec["var"], exc)
            continue

    if not frames:
        return None

    # Merge all variable frames on (lat, lon)
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(
            f, on=["lat", "lon"], how="outer", suffixes=("", "_dup")
        )
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        merged = merged.drop(columns=dup_cols)

    # Nearest-neighbour assignment: find the closest grid point for each station
    tree_coords = np.deg2rad(merged[["lat", "lon"]].values)
    tree = BallTree(tree_coords, metric="haversine")
    q_lon = _longitude_deg_to_180(station_lookup["lon"].values)
    query_coords = np.deg2rad(
        np.column_stack([station_lookup["lat"].values, q_lon])
    )
    _, indices = tree.query(query_coords, k=1)
    indices = indices.flatten()

    result = merged.iloc[indices].reset_index(drop=True).copy()
    result["station"] = station_lookup["station"].values
    result["latitude"] = result["lat"]
    result["longitude"] = result["lon"]
    result["valid_time"] = valid_time
    result["time"] = init_time
    result["lead time"] = fh

    # Unit conversions and derived fields
    result = _kelvin_to_celsius(result, ["t2m", "d2m"])
    result = _compute_derived(result)

    # Ensure all expected columns are present (fill missing with NaN)
    expected = [
        "valid_time", "time", "station", "latitude", "longitude",
        "t2m", "d2m", "sh2", "r2", "u10", "v10", "u_total", "u_dir",
        "tp", "mslma", "orog", "tcc", "cape", "dswrf", "dlwrf", "gh",
        "lsm", "lead time",
    ]
    for col in expected:
        if col not in result.columns:
            result[col] = np.nan

    return result[expected]


def fetch_and_stage(
    cfg,
    start_date: date,
    end_date: date,
    station_lookup: pd.DataFrame | None = None,
    show_progress: bool = True,
) -> None:
    """
    Download HRRR for every forecast hour in ``cfg.data.forecast_hours`` over
    ``[start_date, end_date]``, and write cleaned per-station parquets into
    ``cfg.data.parquets_dir/hrrr_data/fh{fh:02d}/``.

    The function is *resume-friendly*: it skips month-year combinations for
    which the output parquet already exists.

    Parameters
    ----------
    cfg:
        :class:`~app.utils.config_loader.AppConfig` loaded from ``config.yaml``.
    start_date, end_date:
        Inclusive date range to fetch.  Specify as :class:`datetime.date`
        or :class:`datetime.datetime`.
    station_lookup:
        DataFrame with columns ``station``, ``lat``, ``lon`` defining the
        station grid to extract HRRR data for.  When ``None``, the function
        tries to read ``{cfg.output.models_dir}/lookups/station_meta.csv``
        (written by :func:`app.data.fetch_asos.fetch_asos_for_bbox`).
    show_progress:
        Print a progress line for each (fh, year-month) pair processed.
    """
    try:
        from herbie import Herbie
    except ImportError:
        raise ImportError(
            "herbie-data is required — pip install herbie-data"
        )

    if station_lookup is None:
        meta_path = Path(cfg.output.models_dir) / "lookups" / "station_meta.csv"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"station_meta.csv not found at {meta_path}. "
                "Run fetch_asos_for_bbox (or build_station_clusters) first, "
                "or pass station_lookup explicitly."
            )
        station_lookup = pd.read_csv(meta_path)

    for required_col in ("station", "lat", "lon"):
        if required_col not in station_lookup.columns:
            raise ValueError(
                f"station_lookup must have a '{required_col}' column"
            )

    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()

    # Collect all (year, month) pairs in the requested window
    months: list[tuple[int, int]] = []
    current = date(start_date.year, start_date.month, 1)
    end_month_start = date(end_date.year, end_date.month, 1)
    while current <= end_month_start:
        months.append((current.year, current.month))
        # advance one month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    for fh in cfg.data.forecast_hours:
        fh_str = str(fh).zfill(2)
        out_dir = Path(cfg.data.parquets_dir) / "hrrr_data" / f"fh{fh_str}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for year, month in months:
            month_str = str(month).zfill(2)
            out_path = (
                out_dir
                / f"HRRR_{year}_{month_str}_direct_compare_to_nysm_sites_mask_water.parquet"
            )
            if out_path.exists():
                if show_progress:
                    print(f"  [skip] fh={fh_str} {year}-{month_str} already staged")
                continue

            if show_progress:
                print(f"  [fetch] fh={fh_str} {year}-{month_str} …", end="", flush=True)

            month_start = date(year, month, 1)
            # last day of month
            if month == 12:
                month_end = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                month_end = date(year, month + 1, 1) - timedelta(days=1)

            # Clip to user-requested date range
            day_start = max(month_start, start_date)
            day_end = min(month_end, end_date)

            rows: list[pd.DataFrame] = []
            first_failure: str | None = None
            n_extract_empty = 0
            n_ok = 0
            current_day = day_start
            while current_day <= day_end:
                for init_hour in range(24):
                    # UTC wall-clock instant for metadata / valid_time
                    init_dt = datetime(
                        current_day.year,
                        current_day.month,
                        current_day.day,
                        init_hour,
                        tzinfo=timezone.utc,
                    )
                    # Herbie's HRRR templates match archives using naive UTC clock times
                    # (timezone-aware datetimes can prevent GRIB discovery on some setups).
                    init_for_herbie = init_dt.replace(tzinfo=None)
                    try:
                        H = Herbie(
                            init_for_herbie,
                            model="hrrr",
                            product="sfc",
                            fxx=fh,
                            save_dir=str(cfg.data.hrrr_raw_dir),
                            verbose=False,
                            overwrite=False,
                            priority=["aws", "nomads", "google", "azure"],
                        )
                        df_row = _extract_grib2(
                            H,
                            station_lookup,
                            init_dt,
                            fh,
                            prefetch_full_grib=cfg.data.hrrr_prefetch_full_grib,
                        )
                        if df_row is not None and len(df_row) > 0:
                            rows.append(df_row)
                            n_ok += 1
                        else:
                            n_extract_empty += 1
                    except Exception as exc:
                        if first_failure is None:
                            first_failure = f"{type(exc).__name__}: {exc}"
                        logger.debug(
                            "Skipping HRRR init=%s fh=%s: %s", init_dt, fh, exc
                        )
                current_day += timedelta(days=1)

            if rows:
                month_df = pd.concat(rows, ignore_index=True).fillna(-999)
                # Drop the redundant 'new_tp' column if present (legacy artefact)
                if "new_tp" in month_df.columns:
                    month_df = month_df.drop(columns=["new_tp"])
                month_df.to_parquet(out_path, index=False)
                if show_progress:
                    print(f" {len(month_df)} rows written.")
            else:
                if show_progress:
                    parts = [" no data found, skipping."]
                    if first_failure:
                        parts.append(f" Example failure: {first_failure[:200]}")
                    elif n_extract_empty > 0 and n_ok == 0:
                        parts.append(
                            " Herbie ran without raising, but every hour returned no"
                            " extracted rows — usually **eccodes** + **cfgrib** missing or"
                            " GRIB variables not matched (see ``logging`` DEBUG on"
                            " ``app.data.fetch_hrrr``)."
                        )
                    parts.append(
                        " Init times are UTC; confirm network access to AWS/NOMADS archives."
                    )
                    print("".join(parts))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from app.utils.config_loader import load_config

    cfg = load_config()
    fetch_and_stage(cfg, cfg.training.start_date, cfg.training.end_date)
