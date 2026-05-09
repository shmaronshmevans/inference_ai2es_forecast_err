"""Optional lstm_clusters.csv (land cover / elevation / slope categoricals).

The legacy training stack expects ``.../landtype/data/lstm_clusters.csv``.
The app workflow disables that path via :envvar:`FORECAST_APP_SKIP_GEO`, set by
``app.utils.engine_bridge.prepare_env`` from ``config.yaml``.

If the env var is **unset**, geo features are skipped when that legacy path does
not exist on disk (so notebooks still work if ``prepare_env`` was not re-run).
"""
from __future__ import annotations

import os
from pathlib import Path

LEGACY_LSTM_CLUSTERS_CSV = Path("/home/aevans/nwp_bias/src/landtype/data/lstm_clusters.csv")


def skip_landtype_geo() -> bool:
    """Return True to skip reading lstm_clusters and attaching geo categoricals."""
    v = os.environ.get("FORECAST_APP_SKIP_GEO", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    # Env unset: behave like "skip" when the CSV is absent (typical app checkout).
    return not LEGACY_LSTM_CLUSTERS_CSV.is_file()
