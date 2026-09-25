"""Data path for the regime engine: real IMD + TIGGE files -> labels -> A1-A6 -> leave-one-season-out
regime probabilities and the 14 regime features of B3 (PRD 10.3, 10.4, 11.1 to 11.7).

Nothing here is a new model. It only feeds the existing parts:

* labels        `labels/`: core-zone daily rain -> 1981-2010 day-of-year climatology -> z -> `detect_spells`
* A1-A6         `phase/features.py: extract_single_lead_a_features`, `build_phase_feature_vector`
* phase model   `phase/model.py: PhaseModel`, `select_best_c_loso`; OOD flag `phase/confidence.py`
* per-run step  `RegimeEngine.process_run_lead` (Layer A probabilities, Layer B low-pressure systems, Layer C
                coast/mountain) - this is the function that makes the 14 features, and it is called here for
                every (run, lead) of every season

Leakage rules (PRD 10.3/10.4), all enforced by construction:

* the phase model, the OOD limits, the low-pressure strength percentiles and the Layer C percentile tables are
  fitted on the *training* seasons of each fold only. A development season is processed only by the engine that
  did not see it (`regime_source = "oof"`). Holdout seasons are processed once, by the engine fitted on all
  development seasons (`"final"`). Holdout seasons are never used for any fit.
* labels use IMD only (no forecasts); the label climatology is 1981-2010 (`regime.yaml`), fixed.

A4 (core-zone pressure anomaly) is the core-zone mean msl minus the mean over the development seasons. The
phase model standardises every input (`StandardScaler`) and the OOD limits are percentiles, so a constant shift
of A4 cannot change any output: using the development mean instead of a per-fold mean is exact, not an
approximation.

Assumption (not in the PRD, flagged in the report): `coast_normal` is the unit gradient of the land-sea mask
after a Gaussian smoothing of `COAST_SMOOTH_CELLS` cells (about 80 km), so that it is defined out to roughly the
100 km decay scale of `onshore_flux`. On the raw 0.25 degree mask it would be non-zero only next to the coast.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.ndimage import gaussian_filter

from data_pipeline.alignment.cells import load_valid_cells
from data_pipeline.alignment.golden import align_static, atmosphere_fullgrid_windows
from data_pipeline.alignment.temporal import imd_date
from data_pipeline.features.grid import cell_size_m
from data_pipeline.features.library import relative_vorticity
from data_pipeline.features.static import get_static_geography
from data_pipeline.ingestion.config import CONFIG_DIR, IngestionConfig, load_config
from data_pipeline.ingestion.errors import MissingInputError, ValidationError
from data_pipeline.ingestion.imd import imd_axes, read_imd_year
from data_pipeline.ingestion.tigge import read_atmosphere_month, run_id
from regime_engine.contract import REGIME_14_FEATURES
from regime_engine.engine import RegimeEngine
from regime_engine.labels.climatology import compute_doy_climatology, compute_standardized_anomalies
from regime_engine.labels.core_zone import compute_core_zone_daily_mean
from regime_engine.labels.label_checker import check_july_august_statistics
from regime_engine.labels.spell_detector import detect_spells
from regime_engine.lps.detector import LPSDetector
from regime_engine.phase.confidence import PhaseOODDetector
from regime_engine.phase.features import build_phase_feature_vector, extract_single_lead_a_features
from regime_engine.phase.model import PhaseModel, select_best_c_loso
from regime_engine.local_context.indices import compute_raw_fluxes

log = logging.getLogger(__name__)

A_IDS = ["A1", "A2", "A3", "A4", "A5", "A6"]
COAST_SMOOTH_CELLS = 3.0  # assumption, see module docstring
FIELD_VARS = ["msl", "u850", "v850", "q850", "u200", "v200"]
KEY = ["run_id", "lead_day", "cell_id"]


# ---- paths and config ---------------------------------------------------------------------------


def regime_dir(config: IngestionConfig) -> Path:
    return config.data_dir / "regime"


def load_regime_config(path: Path | str | None = None) -> dict[str, Any]:
    return yaml.safe_load(Path(path or CONFIG_DIR / "regime.yaml").read_text())


def _label_window(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """1 June to 3 October: JJAS plus the three days that a lead-3 forecast of 30 September points to."""
    return pd.Timestamp(year, 6, 1), pd.Timestamp(year, 10, 3)


# ---- 1. IMD-derived labels (PRD 11.2) -----------------------------------------------------------


def _mask_hash(valid_ids: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(valid_ids.astype("int32")).tobytes()).hexdigest()[:16]


def core_zone_daily_year(
    year: int, config: IngestionConfig, valid_ids: np.ndarray, box: dict[str, float]
) -> pd.Series:
    """Daily mean IMD rain over the valid cells of the core zone for one year (`compute_core_zone_daily_mean`).

    IMD years are read through `read_imd_year`, so `.grd` and NetCDF files both work. One year at a time.
    """
    rain = read_imd_year(config.imd.year_path(year), year, config)["rain"]
    lat, lon = rain["lat"].values, rain["lon"].values
    il = np.flatnonzero((lat >= box["lat_min"]) & (lat <= box["lat_max"]))
    jl = np.flatnonzero((lon >= box["lon_min"]) & (lon <= box["lon_max"]))
    sub = rain.values[:, il[0] : il[-1] + 1, jl[0] : jl[-1] + 1]
    n_days = sub.shape[0]
    ii, jj = np.meshgrid(il, jl, indexing="ij")
    cell_id = (ii * config.imd.n_lon + jj).ravel()
    df = pd.DataFrame(
        {
            "imd_date": pd.DatetimeIndex(rain["time"].values).repeat(cell_id.size),
            "cell_id": np.tile(cell_id, n_days),
            "latitude": np.tile(lat[ii].ravel(), n_days),
            "longitude": np.tile(lon[jj].ravel(), n_days),
            "rain_mm": sub.reshape(n_days, -1).ravel().astype("float64"),
        }
    )
    out = compute_core_zone_daily_mean(df, valid_cells=valid_ids, **box)
    out.index = pd.DatetimeIndex(out.index, name="date")
    return out


def core_zone_series(
    years: Iterable[int], config: IngestionConfig, box: dict[str, float], cache: bool = True
) -> pd.Series:
    """Core-zone daily rain of the given years, cached one file per year (keyed by the valid-cell mask)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    grid = load_valid_cells(config)
    valid_ids = grid.loc[grid["is_valid"], "cell_id"].to_numpy()
    mask_key = _mask_hash(valid_ids) + json.dumps(box, sort_keys=True)
    parts = []
    for year in sorted(set(int(y) for y in years)):
        path = regime_dir(config) / "core_zone" / f"core_{year}.parquet"
        series = None
        if cache and path.is_file():
            stored = (pq.read_schema(path).metadata or {}).get(b"mask_key", b"").decode()
            if stored == mask_key:
                series = pd.read_parquet(path)["core_rain_mm"]
        if series is None:
            series = core_zone_daily_year(year, config, valid_ids, box).rename("core_rain_mm")
            if cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                table = pa.Table.from_pandas(series.to_frame(), preserve_index=True)
                meta = {**(table.schema.metadata or {}), b"mask_key": mask_key.encode()}  # keep the pandas index info
                pq.write_table(table.replace_schema_metadata(meta), path)
        parts.append(series)
    if not parts:
        raise ValidationError("core_zone_series needs at least one year.")
    out = pd.concat(parts).sort_index()
    out.index = pd.DatetimeIndex(out.index, name="date")
    return out


def build_phase_labels(
    config: IngestionConfig,
    base_years: Sequence[int],
    season_years: Sequence[int],
    regime_cfg: dict[str, Any] | None = None,
    cache: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Daily phase labels for `base_years` and `season_years` from IMD, plus the July-August check (PRD 11.2).

    Returns `(labels, report)`. `labels` is indexed by date with `core_rain_mm`, `z_score`, `phase_label`
    (`active` / `break` / `normal`), `year`, `is_base`. The day-of-year mean and spread come from `base_years`
    only (1981-2010 in `regime.yaml`). Spells are detected within one season window at a time (1 June - 3 Oct),
    never across two years. The check of PRD 11.2 runs on the base years; `report["passed"]` says whether it passed.
    """
    rcfg = (regime_cfg or load_regime_config())["layer_a"]
    box = dict(rcfg["label_box"])
    base_years, season_years = sorted(set(base_years)), sorted(set(season_years))
    series = core_zone_series(base_years + season_years, config, box, cache=cache)
    clim = compute_doy_climatology(
        series[series.index.year.isin(base_years)], window_days=int(rcfg["climatology_window_days"])
    )
    z_all = compute_standardized_anomalies(series, clim)
    frames = []
    for year in sorted(set(base_years) | set(season_years)):
        lo, hi = _label_window(year)
        z = z_all.loc[lo:hi]
        if z.empty:
            continue
        labels = detect_spells(
            z,
            active_z_min=rcfg["active_z_min"],
            break_z_max=rcfg["break_z_max"],
            min_run_days=rcfg["min_run_days"],
        )
        frames.append(
            pd.DataFrame(
                {
                    "core_rain_mm": series.loc[z.index],
                    "z_score": z,
                    "phase_label": labels,
                    "year": year,
                    "is_base": year in base_years,
                }
            )
        )
    out = pd.concat(frames)
    out.index.name = "date"
    base_labels = out.loc[out["is_base"], "phase_label"]
    chk = rcfg["label_check"]
    report = check_july_august_statistics(
        base_labels,
        active_target=chk["active_days_per_season"]["target"],
        active_tol=chk["active_days_per_season"]["tolerance"],
        break_target=chk["break_days_per_season"]["target"],
        break_tol=chk["break_days_per_season"]["tolerance"],
        no_break_share_target=chk["seasons_without_break_share"]["target"],
        no_break_share_tol=chk["seasons_without_break_share"]["tolerance"],
    )
    report.update(base_years=[base_years[0], base_years[-1]], n_base_years=len(base_years))
    return out, report


def write_phase_labels(labels: pd.DataFrame, report: dict[str, Any], config: IngestionConfig) -> Path:
    d = regime_dir(config)
    d.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(d / "labels.parquet")
    (d / "label_check.json").write_text(json.dumps(report, indent=2, default=float))
    return d / "labels.parquet"


def read_phase_labels(config: IngestionConfig) -> pd.Series:
    path = regime_dir(config) / "labels.parquet"
    if not path.is_file():
        raise MissingInputError(f"No phase labels at {path}. Run `python scripts/build_regime.py labels` first.")
    return pd.read_parquet(path)["phase_label"]


# ---- 2. atmosphere fields and A1-A6 -------------------------------------------------------------


def fields_path(year: int, config: IngestionConfig) -> Path:
    return regime_dir(config) / "fields" / f"fields_{year}.nc"


def build_season_fields(
    year: int,
    config: IngestionConfig,
    tigge_dir: Path | str | None = None,
    months: Iterable[int] | None = None,
) -> Path:
    """Full-grid C1 atmosphere windows of one season -> `<DATA_DIR>/regime/fields/fields_<year>.nc`.

    Reads only the atmosphere GRIB files of the season (`read_atmosphere_month`), one month at a time.
    """
    parts = []
    for month in months if months is not None else config.alignment.season_months:
        parts.append(atmosphere_fullgrid_windows(read_atmosphere_month(year, month, config, tigge_dir), config))
    if not parts:
        raise ValidationError(f"No months given for season {year}.")
    ds = xr.concat(parts, dim="init_time").sortby("init_time")
    path = fields_path(year, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    # no compression option: it needs the netCDF4 backend, which not every environment has (about 150 MB per season)
    ds.astype("float32").to_netcdf(path)
    return path


def read_season_fields(year: int, config: IngestionConfig) -> xr.Dataset:
    path = fields_path(year, config)
    if not path.is_file():
        raise MissingInputError(f"No regime fields at {path}. Run `python scripts/build_regime.py fields {year}`.")
    return xr.load_dataset(path)


def lead_a_table(fields: xr.Dataset, config: IngestionConfig) -> pd.DataFrame:
    """A1-A6 for every (run, lead) of a season (PRD 11.3), one row each, from full-grid window fields.

    `A4` is returned as the core-zone mean msl in hPa (the anomaly offset is applied by `run_regime_loso`).
    """
    lat, lon = imd_axes(config.imd)
    inits = pd.DatetimeIndex(fields["init_time"].values)
    leads = [int(k) for k in fields["lead_day"].values]
    u, v = fields["u850"].values.astype("float64"), fields["v850"].values.astype("float64")
    n_i, n_k = u.shape[:2]
    everywhere = np.ones((n_i * n_k, lat.size, lon.size), dtype=bool)
    vort = relative_vorticity(u.reshape(-1, lat.size, lon.size), v.reshape(-1, lat.size, lon.size), everywhere, lat, config)
    vort = vort.reshape(n_i, n_k, lat.size, lon.size)
    rows = []
    for i, t in enumerate(inits):
        for j, k in enumerate(leads):
            f = {name: fields[name].values[i, j].astype("float64") for name in FIELD_VARS}
            f["vort850"] = vort[i, j]
            a = extract_single_lead_a_features(f, lat, lon, training_core_msl_mean=0.0)
            rows.append(
                {
                    "run_id": run_id(t),
                    "init_time": t,
                    "season": int(t.year),
                    "lead_day": k,
                    "imd_date": imd_date(t, k, config.alignment.imd_stamp),
                    **a,
                }
            )
    return pd.DataFrame(rows)


# ---- 3. static inputs of Layer C ----------------------------------------------------------------


def coast_normal(lsm: np.ndarray, lat: np.ndarray, config: IngestionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Unit vector from sea to land (PRD 8.6): normalised gradient of the smoothed land-sea mask, (x, y)."""
    dy, dx = cell_size_m(lat, config)
    sm = gaussian_filter(np.asarray(lsm, float), sigma=COAST_SMOOTH_CELLS, mode="nearest")
    gy, gx = np.gradient(sm, axis=0) / dy, np.gradient(sm, axis=1) / dx
    norm = np.hypot(gx, gy)
    ok = norm > 1e-12
    return np.where(ok, gx / np.where(ok, norm, 1.0), 0.0), np.where(ok, gy / np.where(ok, norm, 1.0), 0.0)


def build_cell_inputs(config: IngestionConfig, static: xr.Dataset) -> pd.DataFrame:
    """Per valid cell: the columns `RegimeEngine.process_run_lead` wants as `grid_cells_df`.

    `static`: raw TIGGE `orog`/`lsm` (`read_static_from_grib`). Terrain gradient is recovered from the cached
    static geography: `dz/dx = -aspect_sin * slope`, `dz/dy = -aspect_cos * slope` (its definition in `static.py`).
    """
    lat, lon = imd_axes(config.imd)
    geo = get_static_geography(config, static=static).set_index("cell_id")
    lsm = align_static(static, config)["lsm"].transpose("lat", "lon").values
    nx, ny = coast_normal(lsm, lat, config)
    grid = load_valid_cells(config)
    cells = grid.loc[grid["is_valid"], ["cell_id", "latitude", "longitude"]].sort_values("cell_id").reset_index(drop=True)
    cid = cells["cell_id"].to_numpy()
    g = geo.loc[cid]
    cells["grad_h_x"] = -(g["aspect_sin"] * g["slope"]).to_numpy()
    cells["grad_h_y"] = -(g["aspect_cos"] * g["slope"]).to_numpy()
    cells["coast_normal_x"] = nx.ravel()[cid]
    cells["coast_normal_y"] = ny.ravel()[cid]
    cells["dist_coast_km"] = g["dist_coast_km"].to_numpy()
    return cells


# ---- 4. leave-one-season-out regime features ----------------------------------------------------


@dataclass
class SeasonInputs:
    """Everything the engine needs for one season, arranged by (init, lead)."""

    season: int
    inits: pd.DatetimeIndex
    leads: list[int]
    msl: np.ndarray  # (init, lead, lat, lon)
    vort850: np.ndarray  # (init, lead, lat, lon), s-1
    u_cell: np.ndarray  # (init, lead, cell)
    v_cell: np.ndarray
    q_cell: np.ndarray


def season_inputs(fields: xr.Dataset, cell_ids: np.ndarray, config: IngestionConfig) -> SeasonInputs:
    lat, lon = imd_axes(config.imd)
    n_i, n_k = fields.sizes["init_time"], fields.sizes["lead_day"]
    u, v = fields["u850"].values.astype("float64"), fields["v850"].values.astype("float64")
    everywhere = np.ones((n_i * n_k, lat.size, lon.size), dtype=bool)
    vort = relative_vorticity(u.reshape(-1, lat.size, lon.size), v.reshape(-1, lat.size, lon.size), everywhere, lat, config)
    flat = lambda name: fields[name].values.reshape(n_i, n_k, -1)[:, :, cell_ids].astype("float32")  # noqa: E731
    return SeasonInputs(
        season=int(pd.Timestamp(fields["init_time"].values[0]).year),
        inits=pd.DatetimeIndex(fields["init_time"].values),
        leads=[int(k) for k in fields["lead_day"].values],
        msl=fields["msl"].values.astype("float32"),  # float32 keeps five seasons well under 2 GB
        vort850=vort.reshape(n_i, n_k, lat.size, lon.size).astype("float32"),
        u_cell=flat("u850"),
        v_cell=flat("v850"),
        q_cell=flat("q850"),
    )


@dataclass
class RegimeResult:
    domain: pd.DataFrame  # one row per (run, lead): probabilities, confidence, OOD, low-pressure centres, source
    cell_features: pd.DataFrame | None  # keys + 14 features, if not streamed to `on_season_done`
    best_c: float
    dev_core_msl_mean_hpa: float
    folds: list[dict[str, Any]]
    final_engine: RegimeEngine
    dev_seasons: list[int]
    holdout_seasons: list[int]
    rows_per_season: dict[int, int] = field(default_factory=dict)


def _phase_frame(a_table: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, lead) with the 20 phase-model inputs (`build_phase_feature_vector`)."""
    rows = []
    for rid, run in a_table.groupby("run_id", sort=False):
        lead_feats = {int(r.lead_day): {a: float(getattr(r, a)) for a in A_IDS} for r in run.itertuples()}
        init = pd.Timestamp(run["init_time"].iloc[0])
        for r in run.itertuples():
            vec, names = build_phase_feature_vector(lead_feats, int(r.lead_day), int(init.dayofyear))
            rows.append({"run_id": rid, "season": int(r.season), "imd_date": r.imd_date,
                         "lead_day": int(r.lead_day), **dict(zip(names, vec))})
    return pd.DataFrame(rows)


def run_regime_loso(
    a_table: pd.DataFrame,
    inputs: dict[int, SeasonInputs],
    labels: pd.Series,
    cell_inputs: pd.DataFrame,
    dev_seasons: Sequence[int],
    holdout_seasons: Sequence[int],
    config: IngestionConfig,
    regime_cfg: dict[str, Any] | None = None,
    on_season_done: Callable[[int, pd.DataFrame], None] | None = None,
) -> RegimeResult:
    """Leave-one-season-out regime probabilities and the 14 B3 features (PRD 10.3).

    * every development season `h` is processed by an engine fitted on the other development seasons
      (`regime_source = "oof"`);
    * every holdout season is processed by an engine fitted on all development seasons (`"final"`);
    * each (run, lead) goes through `RegimeEngine.process_run_lead`.

    `a_table`: `lead_a_table` of every season concerned, concatenated. `inputs`: `season_inputs` per season.
    `labels`: phase labels by date. `on_season_done(season, cell_features)` receives each season's 14-feature
    table as soon as it is made (to write it to disk); if it is None the tables are returned in the result.
    """
    dev, hold = sorted(int(s) for s in dev_seasons), sorted(int(s) for s in holdout_seasons)
    if len(dev) < 2:
        raise ValidationError(f"Leave-one-season-out needs at least 2 development seasons, got {dev}.")
    if set(dev) & set(hold):
        raise ValidationError(f"Seasons {sorted(set(dev) & set(hold))} are both development and holdout.")
    missing = [s for s in dev + hold if s not in inputs or s not in set(a_table["season"])]
    if missing:
        raise MissingInputError(f"No regime fields / A-features for seasons {missing}.")
    rcfg = regime_cfg or load_regime_config()

    # A4 anomaly (exact, see module docstring) and the 20 phase inputs
    dev_mean = float(a_table.loc[a_table["season"].isin(dev), "A4"].mean())
    a_table = a_table.copy()
    a_table["A4"] = a_table["A4"] - dev_mean
    phase = _phase_frame(a_table)
    feat_cols = [c for c in phase.columns if c not in ("run_id", "season", "imd_date", "lead_day")]
    phase["label"] = pd.DatetimeIndex(phase["imd_date"]).map(labels.to_dict().get)
    train_rows = phase.dropna(subset=["label"])
    train_rows = train_rows[train_rows["season"].isin(dev)]
    for s in dev:
        if s not in set(train_rows["season"]):
            raise ValidationError(f"Development season {s} has no labelled days; build labels for it first.")

    c_grid = rcfg["layer_a"]["phase_model"]["penalty_C_grid"]
    best_c = float(
        select_best_c_loso(
            train_rows[feat_cols].to_numpy(float), train_rows["label"].to_numpy(), train_rows["season"].to_numpy(),
            c_grid=[float(c) for c in c_grid],
        )
    )

    # pass 1 (no model needed): raw low-pressure zeta maxima and positive Layer C fluxes per development season
    raw_detector = _new_engine(rcfg).lps_detector
    zetas: dict[int, list[float]] = {}
    up_pos: dict[int, np.ndarray] = {}
    on_pos: dict[int, np.ndarray] = {}
    lats, lons = imd_axes(config.imd)
    grad_x, grad_y = cell_inputs["grad_h_x"].to_numpy(float), cell_inputs["grad_h_y"].to_numpy(float)
    cn_x, cn_y = cell_inputs["coast_normal_x"].to_numpy(float), cell_inputs["coast_normal_y"].to_numpy(float)
    dist_c = cell_inputs["dist_coast_km"].to_numpy(float)
    decay = rcfg["layer_c"]["coast_decay_scale_km"]
    for s in dev:
        si, z, ups, ons = inputs[s], [], [], []
        for i in range(len(si.inits)):
            for j in range(len(si.leads)):
                z += [
                    c["zeta_max"]
                    for c in raw_detector.detect(
                        si.vort850[i, j].astype("float64"), si.msl[i, j].astype("float64"), lats, lons
                    )
                ]
                up, on = compute_raw_fluxes(
                    si.u_cell[i, j].astype("float64"), si.v_cell[i, j].astype("float64"),
                    si.q_cell[i, j].astype("float64"), grad_x, grad_y, cn_x, cn_y, dist_c, decay,
                )
                ups.append(up[up > 1e-9])
                ons.append(on[on > 1e-9])
        zetas[s] = z
        up_pos[s], on_pos[s] = np.concatenate(ups), np.concatenate(ons)

    def fit_engine(train_seasons: Sequence[int]) -> RegimeEngine:
        eng = _new_engine(rcfg)
        tr = train_rows[train_rows["season"].isin(train_seasons)]
        eng.phase_model = PhaseModel(C=best_c).fit(tr[feat_cols].to_numpy(float), tr["label"].to_numpy())
        curr = {a: tr[f"{a}_curr"].to_numpy(float) for a in A_IDS}
        ood_cfg = rcfg["layer_a"]["out_of_range_flag"]
        eng.ood_detector = PhaseOODDetector(
            min_features_outside=ood_cfg["min_features_outside"], percentile_range=tuple(ood_cfg["percentile_range"])
        ).fit(curr)
        eng.lps_detector.fit_strength_percentiles([x for s in train_seasons for x in zetas[s]])
        eng.influence_table.fit(
            np.concatenate([up_pos[s] for s in train_seasons]), np.concatenate([on_pos[s] for s in train_seasons])
        )
        return eng

    domain_rows: list[dict[str, Any]] = []
    kept: list[pd.DataFrame] = []
    rows_per_season: dict[int, int] = {}
    folds: list[dict[str, Any]] = []

    def process(engine: RegimeEngine, season: int, source: str) -> None:
        si = inputs[season]
        a_season = a_table[a_table["season"] == season]
        frames = []
        for i, t in enumerate(si.inits):
            rid = run_id(t)
            run = a_season[a_season["run_id"] == rid]
            lead_feats = {int(r.lead_day): {a: float(getattr(r, a)) for a in A_IDS} for r in run.itertuples()}
            for j, k in enumerate(si.leads):
                domain, _cell, b3 = engine.process_run_lead(
                    run_id=rid, lead_day=k, doy=int(t.dayofyear), lead_a_features=lead_feats,
                    grid_cells_df=cell_inputs, vort850=si.vort850[i, j].astype("float64"),
                    msl=si.msl[i, j].astype("float64"), u850_cell=si.u_cell[i, j].astype("float64"),
                    v850_cell=si.v_cell[i, j].astype("float64"), q850_cell=si.q_cell[i, j].astype("float64"),
                    lats_grid=lats, lons_grid=lons, regime_source=source,
                )
                ph = domain["phase"]
                domain_rows.append({
                    "run_id": rid, "lead_day": k, "season": season,
                    "imd_date": imd_date(t, k, config.alignment.imd_stamp),
                    "p_active": ph["active_probability"], "p_normal": ph["normal_probability"],
                    "p_break": ph["break_probability"], "regime_confidence": ph["regime_confidence"],
                    "confidence_band": ph["confidence_band"], "regime_source": source,
                    "ood_flag": domain["quality"]["ood_flag"], "regime_available": domain["quality"]["regime_available"],
                    "lps_detected": domain["lps"]["detected"], "n_lps": len(domain["lps"]["centres"]),
                    "lps_settings": domain["lps"]["settings"],
                })
                f = b3.copy()
                f.insert(0, "lead_day", np.int8(k))
                f.insert(0, "run_id", rid)
                f["regime_source"] = source
                frames.append(f)
        table = pd.concat(frames, ignore_index=True)
        table["cell_id"] = table["cell_id"].astype("int32")
        table[REGIME_14_FEATURES] = table[REGIME_14_FEATURES].astype("float32")
        table.insert(3, "season", np.int16(season))
        rows_per_season[season] = len(table)
        if on_season_done is not None:
            on_season_done(season, table)
        else:
            kept.append(table)

    for h in dev:
        train_seasons = [s for s in dev if s != h]
        log.info("regime fold: hold out %d, fit on %s", h, train_seasons)
        folds.append({"held_out": h, "fit_on": train_seasons, "regime_source": "oof"})
        process(fit_engine(train_seasons), h, "oof")
    final = fit_engine(dev)
    for s in hold:
        folds.append({"held_out": s, "fit_on": dev, "regime_source": "final"})
        process(final, s, "final")

    return RegimeResult(
        domain=pd.DataFrame(domain_rows),
        cell_features=pd.concat(kept, ignore_index=True) if kept else None,
        best_c=best_c, dev_core_msl_mean_hpa=dev_mean, folds=folds, final_engine=final,
        dev_seasons=dev, holdout_seasons=hold, rows_per_season=rows_per_season,
    )


def _new_engine(rcfg: dict[str, Any]) -> RegimeEngine:
    return RegimeEngine(config=rcfg)


# ---- 5. saving and joining ----------------------------------------------------------------------


def save_regime_artifacts(result: RegimeResult, out_dir: Path | str) -> Path:
    """The final (all development seasons) regime engine: phase model + scaler, OOD limits, low-pressure strength
    percentiles and Layer C tables, plus a JSON summary. What serving needs to make `regime_source = "final"`."""
    import joblib

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    e = result.final_engine
    joblib.dump(
        {
            "phase_model": e.phase_model,
            "ood_thresholds": e.ood_detector.thresholds,
            "lps_training_zeta_maxima": e.lps_detector.training_zeta_maxima,
            "influence_table": e.influence_table.to_dict(),
            "best_c": result.best_c,
            "dev_core_msl_mean_hpa": result.dev_core_msl_mean_hpa,
            "dev_seasons": result.dev_seasons,
            "lps_settings": e.lps_detector.settings_label,
        },
        out / "regime_final.joblib",
    )
    (out / "regime_summary.json").write_text(json.dumps(
        {"best_c": result.best_c, "dev_seasons": result.dev_seasons, "holdout_seasons": result.holdout_seasons,
         "dev_core_msl_mean_hpa": result.dev_core_msl_mean_hpa, "folds": result.folds,
         "rows_per_season": result.rows_per_season, "lps_settings": e.lps_detector.settings_label}, indent=2))
    return out


def merge_regime_features(features: pd.DataFrame, regime: pd.DataFrame, domain: pd.DataFrame) -> pd.DataFrame:
    """The M3 input frame: the 27-feature table plus the 14 regime features, `regime_source`, `regime_available`
    and `ood_flag`, joined on `(run_id, lead_day, cell_id)`. Raises if any feature row has no regime row."""
    r = regime[[*KEY, "regime_source", *REGIME_14_FEATURES]]
    out = features.merge(r, on=KEY, how="left", validate="one_to_one")
    if out["regime_source"].isna().any():
        raise ValidationError(f"{int(out['regime_source'].isna().sum())} feature rows have no regime row.")
    d = domain[["run_id", "lead_day", "ood_flag", "regime_available"]]
    return out.merge(d, on=["run_id", "lead_day"], how="left", validate="many_to_one")


def load_m3_frame(config: IngestionConfig, seasons: Sequence[int]) -> pd.DataFrame:
    """The input frame of `ml.orchestration` for `seasons`: the 27 features plus `region_code`, the 14 regime
    features, `regime_source`, `ood_flag` and `regime_available`, read from the tables the pipeline wrote.

    Refuses to continue if the regime tables and the feature table do not describe exactly the same rows (the same
    valid-cell mask), so a stale table from another mask cannot slip in.
    """
    from data_pipeline.features.pipeline import read_features

    seasons = sorted(int(s) for s in seasons)
    files = [regime_dir(config) / "regime_features" / f"season_{s}.parquet" for s in seasons]
    absent = [f for f in files if not f.is_file()]
    if absent:
        raise MissingInputError(f"No regime features at {absent}. Run `python scripts/build_regime.py oof` first.")
    regime = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    domain_path = regime_dir(config) / "regime_domain.parquet"
    if not domain_path.is_file():
        raise MissingInputError(f"No regime domain table at {domain_path}.")
    domain = pd.read_parquet(domain_path)
    features = read_features(config, seasons=seasons, with_region_code=True)
    if len(regime) != len(features):
        raise ValidationError(
            f"{len(regime):,} regime rows but {len(features):,} feature rows for seasons {seasons}: they were built "
            "on different valid-cell masks or for different runs. Rebuild both from the same mask."
        )
    return merge_regime_features(features, regime, domain)
