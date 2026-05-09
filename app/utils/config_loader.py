"""
app/utils/config_loader.py
==========================
Loads and validates ``app/config.yaml``, returning a nested dataclass that
every notebook imports as its single source of truth.

Usage
-----
>>> from app.utils.config_loader import load_config
>>> cfg = load_config()          # reads config.yaml relative to app/
>>> print(cfg.bbox.lat_min)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# Dataclasses — one per YAML section
# ---------------------------------------------------------------------------

@dataclass
class BboxConfig:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def __post_init__(self) -> None:
        if self.lat_min >= self.lat_max:
            raise ValueError("bbox.lat_min must be less than bbox.lat_max")
        if self.lon_min >= self.lon_max:
            raise ValueError("bbox.lon_min must be less than bbox.lon_max")


@dataclass
class DataConfig:
    hrrr_raw_dir: Path
    parquets_dir: Path
    mesonet_source: str                     # "asos" | "local"
    forecast_hours: List[int]
    metvars: List[str]
    mesonet_local_dir: Optional[Path] = None
    station_meta_csv: Optional[Path] = None
    # When True, call Herbie.download() once per init/fxx before extracting variables.
    # Together with overwrite=False (always), existing files in hrrr_raw_dir are reused.
    hrrr_prefetch_full_grib: bool = False
    # Landcover / elevation / slope categoricals from lstm_clusters.csv (legacy NY path).
    # Keep false for this repo unless you provide that CSV on disk.
    use_geo_encoding: bool = False

    def __post_init__(self) -> None:
        valid_sources = {"asos", "local"}
        if self.mesonet_source not in valid_sources:
            raise ValueError(
                f"data.mesonet_source must be one of {valid_sources}, "
                f"got '{self.mesonet_source}'"
            )
        if self.mesonet_source == "local" and self.mesonet_local_dir is None:
            raise ValueError(
                "data.mesonet_local_dir must be set when mesonet_source == 'local'"
            )
        if not self.forecast_hours:
            raise ValueError("data.forecast_hours must contain at least one value")
        if not self.metvars:
            raise ValueError("data.metvars must contain at least one variable")


@dataclass
class TrainingConfig:
    start_date: date
    end_date: date
    train_frac: float
    val_frac: float
    test_frac: float
    epochs: int
    batch_size: int
    num_layers: int
    learning_rate: float
    weight_decay: float
    sequence_length: int
    device_id: int
    mc_samples: int

    def __post_init__(self) -> None:
        frac_sum = round(self.train_frac + self.val_frac + self.test_frac, 6)
        if abs(frac_sum - 1.0) > 1e-5:
            raise ValueError(
                f"train_frac + val_frac + test_frac must equal 1.0, got {frac_sum}"
            )
        if self.start_date >= self.end_date:
            raise ValueError("training.start_date must be before training.end_date")

    @property
    def train_end(self) -> date:
        """Last date of the training window (exclusive start of val)."""
        full_days = (self.end_date - self.start_date).days
        return self.start_date + timedelta(days=int(full_days * self.train_frac))

    @property
    def val_end(self) -> date:
        """Last date of the validation window (exclusive start of test)."""
        full_days = (self.end_date - self.start_date).days
        return self.train_end + timedelta(days=int(full_days * self.val_frac))

    @property
    def test_end(self) -> date:
        """Last date of the test window — always equals end_date."""
        return self.end_date


@dataclass
class OutputConfig:
    models_dir: Path
    results_dir: Path


@dataclass
class AppConfig:
    """Top-level configuration object, mirroring the structure of config.yaml."""

    bbox: BboxConfig
    data: DataConfig
    model: str
    training: TrainingConfig
    output: OutputConfig

    def __post_init__(self) -> None:
        valid_models = {"lstm", "bnn", "hybrid"}
        if self.model not in valid_models:
            raise ValueError(
                f"model must be one of {valid_models}, got '{self.model}'"
            )

    def make_output_dirs(self) -> None:
        """Create all output directories if they do not already exist."""
        for d in (
            self.data.hrrr_raw_dir,
            self.data.parquets_dir,
            self.output.models_dir,
            self.output.results_dir,
            self.output.models_dir / "lookups",
        ):
            Path(d).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path | None = None) -> AppConfig:
    """
    Read and validate ``config.yaml``, returning a fully-populated
    :class:`AppConfig` instance.

    Parameters
    ----------
    config_path:
        Explicit path to the YAML file.  When ``None`` (default) the function
        searches:
        1. ``./config.yaml``  (current working directory — useful when the
           notebooks run with ``app/`` as cwd)
        2. ``../config.yaml`` (one level up — useful when notebooks live in
           ``app/notebooks/``)

    Returns
    -------
    AppConfig
        Validated configuration ready for use in notebooks and helper modules.

    Raises
    ------
    FileNotFoundError
        If the YAML file cannot be found at any expected location.
    ValueError
        If any required field is missing or contains an invalid value.
    """
    if config_path is None:
        candidates = [
            Path("config.yaml"),
            Path("../config.yaml"),
            Path(__file__).parent.parent / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break
        else:
            raise FileNotFoundError(
                "config.yaml not found. Pass an explicit path or run from the "
                "app/ directory."
            )

    config_path = Path(config_path).resolve()
    with open(config_path) as fh:
        raw = yaml.safe_load(fh)

    # Resolve all paths relative to the directory containing config.yaml.
    base = config_path.parent

    def _path(val: Optional[str]) -> Optional[Path]:
        if val is None:
            return None
        p = Path(val)
        return p if p.is_absolute() else (base / p).resolve()

    def _date(val: str) -> date:
        return date.fromisoformat(str(val))

    b = raw["bbox"]
    bbox = BboxConfig(
        lat_min=float(b["lat_min"]),
        lat_max=float(b["lat_max"]),
        lon_min=float(b["lon_min"]),
        lon_max=float(b["lon_max"]),
    )

    d = raw["data"]
    data = DataConfig(
        hrrr_raw_dir=_path(d["hrrr_raw_dir"]),
        parquets_dir=_path(d["parquets_dir"]),
        mesonet_source=str(d.get("mesonet_source", "asos")),
        forecast_hours=[int(h) for h in d.get("forecast_hours", [1, 6, 12, 18])],
        metvars=[str(v) for v in d.get("metvars", ["t2m"])],
        mesonet_local_dir=_path(d.get("mesonet_local_dir")),
        station_meta_csv=_path(d.get("station_meta_csv")),
        hrrr_prefetch_full_grib=bool(d.get("hrrr_prefetch_full_grib", False)),
        use_geo_encoding=bool(d.get("use_geo_encoding", False)),
    )

    t = raw["training"]
    training = TrainingConfig(
        start_date=_date(t["start_date"]),
        end_date=_date(t["end_date"]),
        train_frac=float(t.get("train_frac", 0.70)),
        val_frac=float(t.get("val_frac", 0.15)),
        test_frac=float(t.get("test_frac", 0.15)),
        epochs=int(t.get("epochs", 50)),
        batch_size=int(t.get("batch_size", 1000)),
        num_layers=int(t.get("num_layers", 3)),
        learning_rate=float(t.get("learning_rate", 5e-5)),
        weight_decay=float(t.get("weight_decay", 0.0)),
        sequence_length=int(t.get("sequence_length", 30)),
        device_id=int(t.get("device_id", 0)),
        mc_samples=int(t.get("mc_samples", 30)),
    )

    o = raw["output"]
    output = OutputConfig(
        models_dir=_path(o["models_dir"]),
        results_dir=_path(o["results_dir"]),
    )

    return AppConfig(
        bbox=bbox,
        data=data,
        model=str(raw.get("model", "lstm")),
        training=training,
        output=output,
    )
