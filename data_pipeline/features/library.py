"""The feature library: the 27 non-regime features of PRD 9.4, and the feature table of PRD 9.6. Owner: M1.

One function, `build_features`, is used by training and by serving (PRD 9.5). Every feature is computed from
the forecast run and static fields only, except `clim_*`, which come from training seasons (see
`climatology.py`). Definitions and units: `docs/features.md`.

Rows of the Golden Dataset hold **valid cells only**, so on the IMD grid a "missing" cell is a non-valid cell.
Spatial features (`nbr_*`, `rain_grad`, `vort850`) ignore such cells: neighbourhood statistics use only the
valid cells inside the window, and finite differences fall back to one-sided differences (grid.py).
Everything is computed per `(run_id, lead_day)` field, so nothing crosses runs or leads except the two lead
features, which look at the neighbouring leads of the **same run**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import ndimage

from data_pipeline.features.grid import cell_size_m, earth_radius_m, masked_derivative
from data_pipeline.ingestion.config import IngestionConfig, load_config
from data_pipeline.ingestion.errors import ValidationError
from data_pipeline.ingestion.imd import imd_axes

FEATURE_VERSION = "v1"

# The 27 features, PRD 9.4 order and exact names.
FEATURE_COLUMNS = [
    "rain_mm", "nbr_mean_3", "nbr_max_3", "nbr_mean_5", "nbr_max_5", "rain_grad", "rain_prev_lead", "rain_next_lead",
    "u850", "v850", "wspd850", "vort850", "q850", "msl", "shear_200_850",
    "elevation_m", "slope", "aspect_sin", "aspect_cos", "dist_coast_km",
    "clim_mean", "clim_p95",
    "doy_sin", "doy_cos", "lead_day", "latitude", "longitude",
]  # fmt: skip
KEY_COLUMNS = ["run_id", "lead_day", "cell_id"]
TARGET_COLUMNS = ["obs_mm", "obs_ge_15_6", "obs_ge_64_5", "obs_ge_115_6"]  # PRD 9.6 targets, NOT features
STATIC_FEATURES = ["elevation_m", "slope", "aspect_sin", "aspect_cos", "dist_coast_km"]
CLIM_FEATURES = ["clim_mean", "clim_p95"]

# Feature table = keys, season, version, the other 26 features (lead_day is already a key), targets.
TABLE_COLUMNS = [
    "run_id", "lead_day", "cell_id", "season", "feature_set_version",
    *[c for c in FEATURE_COLUMNS if c != "lead_day"],
    *TARGET_COLUMNS,
]  # fmt: skip
FEATURE_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("lead_day", pa.int8()),
        ("cell_id", pa.int32()),
        ("season", pa.int16()),
        ("feature_set_version", pa.string()),
        *[(c, pa.float32()) for c in FEATURE_COLUMNS if c != "lead_day"],
        ("obs_mm", pa.float32()),
        ("obs_ge_15_6", pa.bool_()),
        ("obs_ge_64_5", pa.bool_()),
        ("obs_ge_115_6", pa.bool_()),
    ]
)
assert FEATURE_SCHEMA.names == TABLE_COLUMNS and len(FEATURE_COLUMNS) == 27

_GOLDEN_NEEDED = [
    "run_id", "lead_day", "cell_id", "season", "initialization_time", "latitude", "longitude",
    "rain_mm", "obs_mm", "u850", "v850", "u200", "v200", "q850", "msl",
]  # fmt: skip


def feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """The 27 model inputs, in order (what B2 trains on). Keys, targets and the version are left out."""
    return features[FEATURE_COLUMNS]


# ---- spatial helpers (arrays shaped (fields, lat, lon)) ---------------------------------------


def neighbourhood_stats(rain: np.ndarray, valid: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and max of rain over the `size` x `size` cells around every cell (the cell itself included),
    counting only valid cells inside the window. Windows are counted in grid cells, never in degrees or km."""
    win = (1, size, size)
    x = np.where(valid, rain, 0.0).astype("float64")
    n_valid = ndimage.uniform_filter(valid.astype("float64"), size=win, mode="constant", cval=0.0)
    total = ndimage.uniform_filter(x, size=win, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n_valid > 0, total / n_valid, np.nan)
    biggest = ndimage.maximum_filter(np.where(valid, rain, -np.inf), size=win, mode="constant", cval=-np.inf)
    return mean, np.where(np.isfinite(biggest), biggest, np.nan)


def gradient_size(field: np.ndarray, valid: np.ndarray, dy: float, dx: np.ndarray) -> np.ndarray:
    """Size of the horizontal gradient (field units per unit of dy/dx). NaN if either component is undefined."""
    return np.hypot(masked_derivative(field, valid, dy, -2), masked_derivative(field, valid, dx, -1))


def relative_vorticity(
    u: np.ndarray, v: np.ndarray, valid: np.ndarray, lat: np.ndarray, config: IngestionConfig
) -> np.ndarray:
    """zeta = dv/dx - du/dy + u tan(lat) / R, in s-1, by finite differences on the metric grid.

    The last term is the curvature term of relative vorticity on a sphere (R implied by 27.75 km per 0.25 deg).
    """
    dy, dx = cell_size_m(lat, config)
    dv_dx = masked_derivative(v, valid, dx, -1)
    du_dy = masked_derivative(u, valid, dy, -2)
    return dv_dx - du_dy + u * np.tan(np.radians(lat))[:, None] / earth_radius_m(config)


# ---- the feature table --------------------------------------------------------------------------


def build_features(
    golden: pd.DataFrame,
    static: pd.DataFrame,
    climatology: pd.DataFrame,
    config: IngestionConfig | None = None,
) -> pd.DataFrame:
    """Feature table (PRD 9.6) for a batch of Golden Dataset rows.

    `golden`: rows of `read_golden` (whole `(run_id, lead_day)` fields; e.g. one month). `static`: the table of
    `get_static_geography` (all cells). `climatology`: table of `get_climatology`, made from **training seasons
    only**. There is no default for it on purpose.
    """
    config = config or load_config()
    n_lat, n_lon = config.imd.n_lat, config.imd.n_lon
    missing = [c for c in _GOLDEN_NEEDED if c not in golden.columns]
    if missing:
        raise ValidationError(f"Golden rows are missing columns {missing}.")
    if golden.empty:
        raise ValidationError("No Golden Dataset rows to build features from.")
    for name, table, cols in (
        ("static geography", static, STATIC_FEATURES),
        ("climatology", climatology, CLIM_FEATURES),
    ):
        if not set(cols) <= set(table.columns):
            raise ValidationError(
                f"The {name} table is missing columns {sorted(set(cols) - set(table.columns))}."
            )
        if not set(golden["cell_id"]) <= set(table["cell_id"]):
            raise ValidationError(f"The {name} table does not cover every cell of the golden rows.")

    lead = golden["lead_day"].to_numpy().astype("int64")
    run = golden["run_id"].to_numpy()
    cell = golden["cell_id"].to_numpy().astype("int64")
    ilat, ilon = cell // n_lon, cell % n_lon
    fields = pd.MultiIndex.from_arrays([run, lead])
    gidx, uniq = pd.factorize(fields)
    n_fields = len(uniq)

    def cube(column: str) -> np.ndarray:  # (fields, lat, lon); NaN where the cell is not in the golden rows
        a = np.full((n_fields, n_lat, n_lon), np.nan)
        a[gidx, ilat, ilon] = golden[column].to_numpy().astype("float64")
        return a

    def at_rows(a: np.ndarray) -> np.ndarray:
        return a[gidx, ilat, ilon].astype("float32")

    lat_axis, _ = imd_axes(config.imd)
    dy, dx = cell_size_m(lat_axis, config)
    rain, u850, v850 = cube("rain_mm"), cube("u850"), cube("v850")
    valid = np.isfinite(rain)
    out: dict[str, np.ndarray] = {"rain_mm": at_rows(rain)}

    for size in (3, 5):
        mean, biggest = neighbourhood_stats(rain, valid, size)
        out[f"nbr_mean_{size}"], out[f"nbr_max_{size}"] = at_rows(mean), at_rows(biggest)
    out["rain_grad"] = at_rows(gradient_size(rain, valid, dy / 1000.0, dx / 1000.0))  # mm per km

    # rain of the previous / next lead of the SAME run (the key includes run_id); NaN if that lead is absent
    for name, shift in (("rain_prev_lead", -1), ("rain_next_lead", 1)):
        other = uniq.get_indexer(pd.MultiIndex.from_arrays([run, lead + shift]))
        vals = np.full(len(golden), np.nan, dtype="float32")
        ok = other >= 0
        vals[ok] = rain[other[ok], ilat[ok], ilon[ok]]
        out[name] = vals

    out["u850"], out["v850"] = at_rows(u850), at_rows(v850)
    out["wspd850"] = at_rows(np.hypot(u850, v850))
    out["vort850"] = at_rows(
        relative_vorticity(u850, v850, valid & np.isfinite(u850) & np.isfinite(v850), lat_axis, config)
    )
    out["q850"], out["msl"] = at_rows(cube("q850")), at_rows(cube("msl"))
    out["shear_200_850"] = at_rows(np.hypot(cube("u200") - u850, cube("v200") - v850))

    df = pd.DataFrame(out)
    for cols, table in ((STATIC_FEATURES, static), (CLIM_FEATURES, climatology)):
        joined = table.set_index("cell_id")[cols].reindex(cell)
        for c in cols:
            df[c] = joined[c].to_numpy().astype("float32")

    doy = golden["initialization_time"].dt.dayofyear.to_numpy()  # day of year of the forecast start date
    angle = 2.0 * np.pi * doy / config.features.doy_period_days
    df["doy_sin"], df["doy_cos"] = np.sin(angle).astype("float32"), np.cos(angle).astype("float32")
    df["latitude"], df["longitude"] = golden["latitude"].to_numpy(), golden["longitude"].to_numpy()

    obs = golden["obs_mm"].to_numpy().astype("float32")
    names = [f"obs_ge_{t:g}".replace(".", "_") for t in config.features.thresholds_mm]
    if names != TARGET_COLUMNS[1:]:
        raise ValidationError(
            f"thresholds.yaml rain_thresholds_mm gives targets {names}, expected {TARGET_COLUMNS[1:]}."
        )
    table = pd.DataFrame(
        {
            "run_id": run,
            "lead_day": golden["lead_day"].to_numpy(),
            "cell_id": golden["cell_id"].to_numpy(),
            "season": golden["season"].to_numpy(),
            "feature_set_version": FEATURE_VERSION,
        }
    )
    table = pd.concat([table, df[[c for c in FEATURE_COLUMNS if c != "lead_day"]]], axis=1)
    table["obs_mm"] = obs
    for name, t in zip(names, config.features.thresholds_mm, strict=True):
        table[name] = pd.array(
            np.where(np.isnan(obs), pd.NA, obs >= t), dtype="boolean"
        )  # null where obs is missing
    table = table[TABLE_COLUMNS].astype(
        {
            f.name: f.type.to_pandas_dtype()
            for f in FEATURE_SCHEMA
            if pa.types.is_floating(f.type) or pa.types.is_integer(f.type)
        }
    )
    check_features(table)
    return table


def check_features(table: pd.DataFrame) -> None:
    """Schema and key checks on a feature table."""
    if list(table.columns) != TABLE_COLUMNS:
        raise ValidationError(
            f"Feature table columns differ from the schema: missing {sorted(set(TABLE_COLUMNS) - set(table.columns))}, "
            f"unexpected {sorted(set(table.columns) - set(TABLE_COLUMNS))}."
        )
    if table.duplicated(KEY_COLUMNS).any():
        raise ValidationError(
            f"{int(table.duplicated(KEY_COLUMNS).sum())} duplicate (run_id, lead_day, cell_id) rows."
        )
    always = ["rain_mm", "u850", "v850", "wspd850", "q850", "msl", "shear_200_850",
              "elevation_m", "slope", "aspect_sin", "aspect_cos",
              "doy_sin", "doy_cos", "latitude", "longitude", "nbr_mean_3", "nbr_max_3", "nbr_mean_5", "nbr_max_5"]  # fmt: skip
    bad = [c for c in always if table[c].isna().any()]
    if bad:
        raise ValidationError(f"Features {bad} have missing values but must always be defined.")
