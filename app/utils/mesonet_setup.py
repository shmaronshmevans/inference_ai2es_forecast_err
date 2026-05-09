"""
Resolve whether to use pre-existing mesonet parquets (e.g. on a mounted volume)
vs downloading ASOS from IEM, and stage parquets into ``parquets_dir/mesonet``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from app.utils.config_loader import AppConfig


def _has_mesonet_parquets(directory: Path) -> bool:
    return bool(list(directory.glob("mesonet_1H_obs_*.parquet")))


def use_local_mesonet(cfg: AppConfig) -> bool:
    """
    Return True if Step 2 should **not** call the IEM ASOS downloader.

    * ``mesonet_source: local`` → always True (requires ``mesonet_local_dir``).
    * ``mesonet_source: asos`` → True when ``mesonet_local_dir`` points to a
      directory that already contains ``mesonet_1H_obs_*.parquet`` (mounted
      or copied data).
    """
    if cfg.data.mesonet_source == "local":
        return True
    root = cfg.data.mesonet_local_dir
    if root is None:
        return False
    root = Path(root)
    return root.is_dir() and _has_mesonet_parquets(root)


def ensure_mesonet_parquets_staged(cfg: AppConfig) -> Path:
    """
    Ensure ``{parquets_dir}/mesonet`` contains the hourly mesonet parquets.

    When ``mesonet_local_dir`` is set and differs from that folder, symlink each
    ``mesonet_1H_obs_*.parquet`` into the app layout (fallback: copy if symlinks
    are not allowed).
    """
    src = cfg.data.mesonet_local_dir
    if src is None:
        raise ValueError("mesonet_local_dir must be set when using local mesonet data.")
    src = Path(src).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"mesonet_local_dir is not a directory: {src}")
    dst = Path(cfg.data.parquets_dir) / "mesonet"
    dst.mkdir(parents=True, exist_ok=True)

    if src == dst.resolve():
        return dst

    linked = 0
    copied = 0
    for pq in sorted(src.glob("mesonet_1H_obs_*.parquet")):
        link = dst / pq.name
        if link.is_symlink() and link.resolve() == pq.resolve():
            continue
        if link.exists() and not link.is_symlink():
            continue
        try:
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(pq)
            linked += 1
        except OSError:
            shutil.copy2(pq, link)
            copied += 1

    print(
        f"Mesonet staging: {linked} symlink(s), {copied} copy/copies → {dst}"
    )
    return dst


def infer_station_meta_from_mesonet_dir(local_dir: Path) -> pd.DataFrame:
    """
    Build ``station``, ``lat``, ``lon``, ``elev`` metadata from the first
    ``mesonet_1H_obs_*.parquet`` found under *local_dir*.
    """
    paths = sorted(Path(local_dir).glob("mesonet_1H_obs_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No mesonet_1H_obs_*.parquet files under {local_dir}"
        )
    df = pd.read_parquet(paths[0])
    if getattr(df.index, "nlevels", 1) > 1:
        df = df.reset_index()
    need = ["station", "lat", "lon"]
    for c in need:
        if c not in df.columns:
            raise ValueError(
                f"Mesonet parquet {paths[0].name} has no {c!r} column; "
                "set data.station_meta_csv in config.yaml."
            )
    cols = ["station", "lat", "lon"]
    if "elev" in df.columns:
        cols.append("elev")
    meta = df[cols].drop_duplicates(subset=["station"]).reset_index(drop=True)
    if "elev" not in meta.columns:
        meta["elev"] = np.nan
    return meta


def load_station_meta_for_local_mesonet(cfg: AppConfig) -> pd.DataFrame:
    """Station table from ``station_meta_csv`` or inferred from mesonet parquets."""
    if cfg.data.station_meta_csv is not None:
        return pd.read_csv(cfg.data.station_meta_csv)
    assert cfg.data.mesonet_local_dir is not None
    return infer_station_meta_from_mesonet_dir(Path(cfg.data.mesonet_local_dir))
