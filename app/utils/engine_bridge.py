"""
app/utils/engine_bridge.py
==========================
Bridge utilities that let the ``app/`` layer call the existing
``src/engine_*_training.py`` and ``src/*_s2s_engine.py`` functions with
custom data paths and station lists instead of the hard-coded NYSM/server
paths embedded in those files.

How it works
------------
1. **Station CSV** — the training engines read a NYSM-style CSV at a
   hard-coded path to find which stations belong to a "climate division".
   :func:`make_station_csv` writes an equivalent CSV (with a synthetic
   ``climate_division_name`` = ``"bbox_stations"``) so the engines can
   find your ASOS stations without any source modifications.

2. **Data-loader patching** — ``nysm_data.load_nysm_data`` and
   ``hrrr_data.read_hrrr_data`` resolve paths at call-time from module-level
   strings.  :func:`patch_data_loaders` temporarily replaces those functions
   with thin wrappers that redirect to the app's output directories.

3. **Model-dir env var** — ``MODEL_DIR`` in every engine already reads from
   ``os.environ["LSTM_MODEL_DIR"]``; setting that before the import is enough.

Usage example (in a notebook)
------------------------------
>>> from app.utils.engine_bridge import prepare_env, patch_data_loaders
>>> prepare_env(cfg)
>>> with patch_data_loaders(cfg):
...     from src.engine_lstm_training import main as train_main
...     train_main(...)
"""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Station CSV
# ---------------------------------------------------------------------------

_SYNTHETIC_CLIM_DIV = "bbox_stations"


def make_station_csv(cfg, station_meta: Optional[pd.DataFrame] = None) -> Path:
    """
    Write a NYSM-style station CSV so the training engines can enumerate
    your ASOS stations without touching the original NYSM data file.

    The file is written to::

        {cfg.output.models_dir}/lookups/app_stations.csv

    Parameters
    ----------
    cfg:
        :class:`~app.utils.config_loader.AppConfig` loaded from ``config.yaml``.
    station_meta:
        DataFrame with at least ``station``, ``lat``, ``lon`` columns.
        When ``None``, reads the CSV at
        ``{cfg.output.models_dir}/lookups/station_meta.csv``.

    Returns
    -------
    Path
        Path to the written CSV.
    """
    lookups_dir = Path(cfg.output.models_dir) / "lookups"
    lookups_dir.mkdir(parents=True, exist_ok=True)

    if station_meta is None:
        meta_path = lookups_dir / "station_meta.csv"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"station_meta.csv not found at {meta_path}. "
                "Run fetch_asos_for_bbox first."
            )
        station_meta = pd.read_csv(meta_path)

    # Build a NYSM-compatible CSV: stid, lat, lon, elev, climate_division_name
    out_df = station_meta[["station"]].rename(columns={"station": "stid"}).copy()
    for col in ("lat", "lon", "elev"):
        if col in station_meta.columns:
            out_df[col] = station_meta[col].values
        else:
            out_df[col] = np.nan
    out_df["climate_division_name"] = _SYNTHETIC_CLIM_DIV

    out_path = lookups_dir / "app_stations.csv"
    out_df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

def prepare_env(cfg) -> dict[str, str]:
    """
    Set all environment variables read by ``src/engine_*_training.py``,
    ``src/*_s2s_engine.py``, and ``model_data.prepare_lstm_data`` so they use the
    app's output directories and optional toggles (not hard-coded server paths).

    The env vars are set on ``os.environ`` immediately (no context manager).
    Re-run this function if you change ``cfg``.

    Returns
    -------
    dict[str, str]
        The full set of env vars that were set.
    """
    env: dict[str, str] = {}

    def _set(key: str, value: str) -> None:
        os.environ[key] = value
        env[key] = value

    _set("LSTM_MODEL_DIR", str(cfg.output.models_dir))
    _set("LSTM_OUTPUT_DIR", str(cfg.output.results_dir))
    # Geo categoricals from lstm_clusters.csv — off by default for the app (see config.yaml).
    _set(
        "FORECAST_APP_SKIP_GEO",
        "0" if getattr(cfg.data, "use_geo_encoding", False) else "1",
    )

    patch_lookup_parquet(cfg)

    return env


# ---------------------------------------------------------------------------
# Data-loader patching
# ---------------------------------------------------------------------------

def _valid_time_to_naive_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone from ``valid_time`` so it matches naive ``datetime`` bounds.

    Parquets from ASOS/HRRR pipelines often store UTC-aware timestamps; the
    training engines compare ``valid_time`` to naive ``datetime`` from the train
    window, which raises ``TypeError`` if mixed.

    Uses :meth:`~pandas.Series.dt.tz` (not ``is_datetime64tz_dtype`` alone) so
    PyArrow-backed columns like ``timestamp[us, tz=UTC][pyarrow]`` are handled
    after ``pd.to_datetime`` materializes them.
    """
    if df is None or "valid_time" not in df.columns:
        return df
    out = df.copy()
    vt = pd.to_datetime(out["valid_time"])
    try:
        if hasattr(vt.dt, "tz") and vt.dt.tz is not None:
            vt = vt.dt.tz_convert("UTC").dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    out["valid_time"] = vt
    return out


def _load_data_loader_modules():
    """
    Import observation / HRRR helper modules under every name the training
    engines may use.

    ``engine_*_training.py`` does ``from model_data import nysm_data`` (with
    ``src/`` on ``sys.path``), while notebooks may use
    ``import src.model_data.nysm_data``.  Those are **distinct** entries in
    ``sys.modules``, so patches must be applied to both or loaders keep the
    hard-coded server paths.
    """
    repo_root = str(Path(__file__).parent.parent.parent)
    src_root = str(Path(repo_root) / "src")
    for p in (repo_root, src_root):
        if p not in sys.path:
            sys.path.insert(0, p)

    nysm_modules = []
    hrrr_modules = []
    for name in ("src.model_data.nysm_data", "model_data.nysm_data"):
        try:
            nysm_modules.append(importlib.import_module(name))
        except ImportError:
            pass
    for name in ("src.model_data.hrrr_data", "model_data.hrrr_data"):
        try:
            hrrr_modules.append(importlib.import_module(name))
        except ImportError:
            pass
    if not nysm_modules:
        raise ImportError("Could not import nysm_data (tried src.model_data and model_data).")
    if not hrrr_modules:
        raise ImportError("Could not import hrrr_data (tried src.model_data and model_data).")
    return nysm_modules, hrrr_modules


@contextmanager
def patch_data_loaders(cfg):
    """
    Context manager that temporarily replaces the ``nysm_data.load_nysm_data``
    and ``hrrr_data.read_hrrr_data`` functions with versions that read from
    the app's output directories.

    Patches **both** ``src.model_data.*`` and ``model_data.*`` module objects,
    because the training engines import the latter when ``src/`` is on
    ``sys.path``.

    The original functions are restored on exit, even on error.

    Usage
    -----
    >>> with patch_data_loaders(cfg):
    ...     train_main(...)
    """
    nysm_modules, hrrr_modules = _load_data_loader_modules()

    mesonet_dir = Path(cfg.data.parquets_dir) / "mesonet"
    hrrr_base_dir = Path(cfg.data.parquets_dir) / "hrrr_data"

    # ---- Build patched versions ----

    original_nysm = [(m, m.load_nysm_data) for m in nysm_modules]
    original_hrrr = [(m, m.read_hrrr_data) for m in hrrr_modules]

    def patched_nysm(start_year):
        """Read app mesonet parquets instead of the hardcoded NYSM path."""
        frames = []
        for pq in sorted(mesonet_dir.glob("mesonet_1H_obs_*.parquet")):
            df = pd.read_parquet(pq)
            df = df.reset_index()
            if "time_1H" in df.columns:
                df = df.rename(columns={"time_1H": "valid_time"})
            frames.append(df)
        if not frames:
            raise FileNotFoundError(
                f"No mesonet parquets found in {mesonet_dir}. "
                "Run 01_setup_data.ipynb first."
            )
        out = pd.concat(frames, ignore_index=True)
        out.fillna({"snow_depth": -999, "ta9m": -999}, inplace=True)
        out.dropna(inplace=True)
        return _valid_time_to_naive_utc(out)

    def patched_hrrr(fh, year):
        """Read app HRRR parquets instead of the hardcoded server path."""
        fh_str = str(fh).zfill(2)
        fh_dir = hrrr_base_dir / f"fh{fh_str}"
        frames = []
        for pq in sorted(fh_dir.glob("HRRR_*.parquet")):
            frames.append(pd.read_parquet(pq))
        if not frames:
            raise FileNotFoundError(
                f"No HRRR parquets found in {fh_dir}. "
                "Run 01_setup_data.ipynb first."
            )
        df = pd.concat(frames, ignore_index=True).fillna(-999)
        if "new_tp" in df.columns:
            df = df.drop(columns=["new_tp"])
        return _valid_time_to_naive_utc(df)

    # ---- Apply patches ----
    for m in nysm_modules:
        m.load_nysm_data = patched_nysm
    for m in hrrr_modules:
        m.read_hrrr_data = patched_hrrr

    try:
        yield
    finally:
        # ---- Restore originals ----
        for m, orig in original_nysm:
            m.load_nysm_data = orig
        for m, orig in original_hrrr:
            m.read_hrrr_data = orig


# ---------------------------------------------------------------------------
# Station CSV patching
# ---------------------------------------------------------------------------

@contextmanager
def patch_station_csv(cfg, station_csv_path: Optional[Path] = None):
    """
    Context manager that monkey-patches the hard-coded NYSM station CSV path
    inside ``src/engine_lstm_training.py`` (and BNN/Hybrid equivalents) so
    they read from the app's custom station CSV instead.

    The patch is applied by overwriting the ``pd.read_csv`` call in each
    engine module with a wrapper that intercepts reads for the known NYSM path.

    Parameters
    ----------
    cfg:
        :class:`~app.utils.config_loader.AppConfig`.
    station_csv_path:
        Explicit path to the app station CSV.  Defaults to the file written
        by :func:`make_station_csv`.
    """
    if station_csv_path is None:
        station_csv_path = Path(cfg.output.models_dir) / "lookups" / "app_stations.csv"
    if not station_csv_path.exists():
        raise FileNotFoundError(
            f"app_stations.csv not found at {station_csv_path}. "
            "Call make_station_csv(cfg) first."
        )

    repo_root = str(Path(__file__).parent.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    _NYSM_CSV_PATH = "/home/aevans/nwp_bias/src/landtype/data/nysm.csv"

    import pandas as _pd_real
    _original_read_csv = _pd_real.read_csv

    def _patched_read_csv(path_or_buf, *args, **kwargs):
        if isinstance(path_or_buf, str) and path_or_buf == _NYSM_CSV_PATH:
            return _original_read_csv(station_csv_path, *args, **kwargs)
        return _original_read_csv(path_or_buf, *args, **kwargs)

    _pd_real.read_csv = _patched_read_csv

    # Also patch inside engine modules that have already imported pandas
    engine_modules = [
        "src.engine_lstm_training",
        "src.engine_bnn_training",
        "src.engine_hybrid_training",
    ]
    original_module_read_csvs = {}
    for mod_name in engine_modules:
        try:
            mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
            if hasattr(mod, "pd") and hasattr(mod.pd, "read_csv"):
                original_module_read_csvs[mod_name] = mod.pd.read_csv
                mod.pd.read_csv = _patched_read_csv
        except (ImportError, AttributeError):
            pass

    try:
        yield _SYNTHETIC_CLIM_DIV
    finally:
        _pd_real.read_csv = _original_read_csv
        for mod_name, original in original_module_read_csvs.items():
            mod = sys.modules.get(mod_name)
            if mod is not None and hasattr(mod, "pd"):
                mod.pd.read_csv = original


def make_station_climdiv_pickle(
    cfg, station_meta: Optional[pd.DataFrame] = None
) -> Path:
    """
    Write a ``station_to_climdiv.pkl`` mapping every bbox station to the
    synthetic climate division ``"bbox_stations"``.

    The inference engines (``lstm_s2s_engine.py``, ``bnn_s2s_engine.py``)
    load this pickle to resolve the model-weight file names.

    Parameters
    ----------
    cfg:
        :class:`~app.utils.config_loader.AppConfig`.
    station_meta:
        Optional pre-loaded station metadata.  When ``None``, reads
        ``{cfg.output.models_dir}/lookups/station_meta.csv``.

    Returns
    -------
    Path
        Path of the written pickle.
    """
    import pickle

    lookups_dir = Path(cfg.output.models_dir) / "lookups"
    lookups_dir.mkdir(parents=True, exist_ok=True)

    if station_meta is None:
        meta_path = lookups_dir / "station_meta.csv"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"station_meta.csv not found at {meta_path}. "
                "Run fetch_asos_for_bbox first."
            )
        station_meta = pd.read_csv(meta_path)

    mapping = {row["station"]: _SYNTHETIC_CLIM_DIV for _, row in station_meta.iterrows()}
    out_path = lookups_dir / "station_to_climdiv.pkl"
    with open(out_path, "wb") as fh:
        pickle.dump(mapping, fh)
    return out_path


@contextmanager
def patch_rapids_data_loaders(cfg):
    """
    Context manager that patches ``nysm_data_rapids.load_nysm_data`` and
    ``hrrr_data_rapids.read_hrrr_data`` to read from the app's output
    directories (same semantics as :func:`patch_data_loaders` but for the
    inference engines that use the RAPIDS/cuDF code path).

    Falls back gracefully when ``src.model_data.nysm_data_rapids`` or
    ``src.model_data.hrrr_data_rapids`` are not importable (e.g. when
    running on a CPU-only machine).
    """
    repo_root = str(Path(__file__).parent.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    mesonet_dir = Path(cfg.data.parquets_dir) / "mesonet"
    hrrr_base_dir = Path(cfg.data.parquets_dir) / "hrrr_data"

    def patched_nysm_rapids(start_year):
        frames = []
        for pq in sorted(mesonet_dir.glob("mesonet_1H_obs_*.parquet")):
            df = pd.read_parquet(pq)
            df = df.reset_index()
            if "time_1H" in df.columns:
                df = df.rename(columns={"time_1H": "valid_time"})
            frames.append(df)
        if not frames:
            raise FileNotFoundError(
                f"No mesonet parquets found in {mesonet_dir}. "
                "Run 01_setup_data.ipynb first."
            )
        out = pd.concat(frames, ignore_index=True)
        out.fillna({"snow_depth": -999, "ta9m": -999}, inplace=True)
        out.dropna(inplace=True)
        return _valid_time_to_naive_utc(out)

    def patched_hrrr_rapids(fh, year):
        fh_str = str(fh).zfill(2)
        fh_dir = hrrr_base_dir / f"fh{fh_str}"
        frames = []
        for pq in sorted(fh_dir.glob("HRRR_*.parquet")):
            frames.append(pd.read_parquet(pq))
        if not frames:
            raise FileNotFoundError(
                f"No HRRR parquets found in {fh_dir}. "
                "Run 01_setup_data.ipynb first."
            )
        df = pd.concat(frames, ignore_index=True).fillna(-999)
        if "new_tp" in df.columns:
            df = df.drop(columns=["new_tp"])
        return _valid_time_to_naive_utc(df)

    originals = {}
    for mod_name, attr, replacement in [
        ("src.model_data.nysm_data_rapids", "load_nysm_data",  patched_nysm_rapids),
        ("src.model_data.hrrr_data_rapids", "read_hrrr_data",  patched_hrrr_rapids),
    ]:
        try:
            mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
            originals[mod_name] = (mod, attr, getattr(mod, attr))
            setattr(mod, attr, replacement)
        except ImportError:
            pass

    try:
        yield
    finally:
        for mod_name, (mod, attr, original) in originals.items():
            setattr(mod, attr, original)


@contextmanager
def patch_inference_pickle(cfg):
    """
    Context manager that monkey-patches the hard-coded pickle path inside
    ``src/lstm_s2s_engine.py`` and ``src/bnn_s2s_engine.py`` to point to
    the app's ``station_to_climdiv.pkl``.
    """
    import builtins

    pickle_path = str(Path(cfg.output.models_dir) / "lookups" / "station_to_climdiv.pkl")
    _ORIGINAL_PATH = (
        "/home/aevans/inference_ai2es_forecast_err/MODELS/lookups/station_to_climdiv.pkl"
    )

    _orig_open = builtins.open

    def _patched_open(path, *args, **kwargs):
        if isinstance(path, str) and path == _ORIGINAL_PATH:
            return _orig_open(pickle_path, *args, **kwargs)
        return _orig_open(path, *args, **kwargs)

    builtins.open = _patched_open
    try:
        yield
    finally:
        builtins.open = _orig_open


def patch_lookup_parquet(cfg) -> None:
    """
    Override ``LOOKUP_PARQUET`` in ``get_closest_nysm_stations`` so reads use the
    app's ``triangulate_stations.parquet``.

    Patches **both** ``src.model_data.get_closest_nysm_stations`` and
    ``model_data.get_closest_nysm_stations`` (distinct ``sys.modules`` entries when
    ``src/`` is on ``sys.path`` — same issue as :func:`patch_data_loaders`).

    Call this once before any training or inference run (the ``02_train`` notebook
    does this in Step 2).
    """
    repo_root = str(Path(__file__).parent.parent.parent)
    src_root = str(Path(repo_root) / "src")
    for p in (repo_root, src_root):
        if p not in sys.path:
            sys.path.insert(0, p)

    lookup_path = str(
        Path(cfg.output.models_dir) / "lookups" / "triangulate_stations.parquet"
    )
    for mod_name in (
        "src.model_data.get_closest_nysm_stations",
        "model_data.get_closest_nysm_stations",
    ):
        try:
            mod = importlib.import_module(mod_name)
            mod.LOOKUP_PARQUET = lookup_path
        except ImportError:
            pass
