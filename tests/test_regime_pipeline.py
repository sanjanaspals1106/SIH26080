"""Regime engine data path (`regime_engine/pipeline.py`) on small synthetic inputs. Nothing is downloaded and no
real data is read; these tests check that the wiring is right: labels from IMD, A1-A6 from fields, and that
`RegimeEngine.process_run_lead` runs for every (run, lead) with strictly out-of-fold models."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.ndimage import gaussian_filter
from tests.stage3_world import make_grid, make_tigge_static

import regime_engine.pipeline as rp
from data_pipeline.alignment import imd_grid, save_valid_cells
from data_pipeline.ingestion import MissingInputError, ValidationError, load_config
from data_pipeline.ingestion.imd import imd_axes
from data_pipeline.ingestion.synthetic import write_synthetic_imd_year
from regime_engine.contract import REGIME_14_FEATURES
from regime_engine.engine import RegimeEngine
from regime_engine.phase.model import PhaseModel


@pytest.fixture()
def cfg(tmp_path):
    return load_config(data_dir=tmp_path / "data")


# ---- labels -------------------------------------------------------------------------------------


@pytest.fixture()
def imd_world(cfg):
    grid = imd_grid(cfg)
    grid["is_valid"], grid["valid_fraction"] = True, np.float32(1.0)
    save_valid_cells(grid, cfg)
    for year in (2019, 2020, 2021):
        write_synthetic_imd_year(cfg, year, missing_fraction=0.05, seed=year)
    return cfg


def test_labels_come_from_imd_with_a_base_period_climatology(imd_world, monkeypatch):
    labels, report = rp.build_phase_labels(imd_world, [2019, 2020], [2021])
    assert list(labels.columns) == ["core_rain_mm", "z_score", "phase_label", "year", "is_base"]
    assert set(labels["phase_label"]) <= {"active", "normal", "break"}
    assert set(labels["year"]) == {2019, 2020, 2021}
    assert labels.loc[labels["year"] == 2021, "is_base"].eq(False).all()
    assert labels.index.min() == pd.Timestamp(2019, 6, 1) and labels.loc[labels["year"] == 2021].index.max() == pd.Timestamp(2021, 10, 3)
    assert labels["z_score"].notna().all()
    assert report["base_years"] == [2019, 2020] and "passed" in report and report["n_seasons"] == 2  # base years only
    # a spell is >= 3 days in a row, and never crosses two seasons
    lab = labels["phase_label"].to_numpy()
    yr = labels["year"].to_numpy()
    runs = pd.Series(lab).ne(pd.Series(lab).shift()).cumsum()
    for _, g in pd.Series(lab).groupby(runs):
        if g.iloc[0] != "normal":
            assert len(g) >= 3
    # second call uses the per-year cache: the IMD files are not opened again
    monkeypatch.setattr(rp, "core_zone_daily_year", lambda *a, **k: pytest.fail("recomputed"))
    again, _ = rp.build_phase_labels(imd_world, [2019, 2020], [2021])
    pd.testing.assert_frame_equal(labels, again)
    assert (yr[1:] >= yr[:-1]).all()


def test_labels_read_netcdf_years_too(imd_world):
    """The same code path with a year that exists only as an IMD NetCDF file."""
    grd = imd_world.imd.year_path(2021)
    from data_pipeline.ingestion.imd import read_imd_year

    ds = read_imd_year(grd, 2021, imd_world)
    out = xr.Dataset({"RAINFALL": (("TIME", "LATITUDE", "LONGITUDE"), ds["rain"].values)},
                     coords={"TIME": ds["time"].values, "LATITUDE": ds["lat"].values, "LONGITUDE": ds["lon"].values})
    (imd_world.data_dir / "imd").mkdir(parents=True, exist_ok=True)
    out.to_netcdf(imd_world.data_dir / "imd" / "RF25_ind2021_rfp25.nc")
    from_grd, _ = rp.build_phase_labels(imd_world, [2019, 2020], [2021], cache=False)
    grd.unlink()
    from_nc, _ = rp.build_phase_labels(imd_world, [2019, 2020], [2021], cache=False)
    pd.testing.assert_frame_equal(from_grd, from_nc)


# ---- layer C static inputs ----------------------------------------------------------------------


def test_coast_normal_points_from_sea_to_land(cfg):
    lat, lon = imd_axes(cfg.imd)
    lsm = np.where(np.arange(lon.size)[None, :] < 40, 0.0, 1.0) * np.ones((lat.size, 1))  # sea in the west
    nx, ny = rp.coast_normal(lsm, lat, cfg)
    assert nx[60, 40] > 0.99 and abs(ny[60, 40]) < 0.01  # at the coast: toward the east (land)
    assert nx[60, 43] > 0.9  # still defined a few cells inland
    assert nx[60, 5] == 0.0  # open sea far from the coast


def test_cell_inputs_have_the_columns_the_engine_needs(cfg):
    save_valid_cells(make_grid(cfg), cfg)
    cells = rp.build_cell_inputs(cfg, make_tigge_static())
    assert list(cells.columns) == ["cell_id", "latitude", "longitude", "grad_h_x", "grad_h_y",
                                   "coast_normal_x", "coast_normal_y", "dist_coast_km"]
    assert len(cells) == int(make_grid(cfg)["is_valid"].sum()) and cells["cell_id"].is_monotonic_increasing
    assert (cells["grad_h_y"] > 0).all() and (cells["grad_h_x"] > 0).all()  # terrain rises to the north and east
    assert cells["dist_coast_km"].notna().all()
    assert np.isfinite(cells[["coast_normal_x", "coast_normal_y"]].to_numpy()).all()


# ---- A1-A6 and leave-one-season-out --------------------------------------------------------------

N_INIT, LEADS, N_CELLS = 8, (1, 2, 3), 60


def make_fields(cfg, season, seed):
    """Smooth random full-grid fields for 8 start days in June (dims init_time, lead_day, lat, lon)."""
    rng = np.random.default_rng(seed)
    lat, lon = imd_axes(cfg.imd)
    shape = (N_INIT, len(LEADS), lat.size, lon.size)

    def smooth(scale, sigma=4):
        f = rng.normal(size=shape) * scale
        return np.stack([[gaussian_filter(f[i, j], sigma) for j in range(shape[1])] for i in range(shape[0])])

    data = {
        "msl": 100500.0 + smooth(4000.0, 6),
        "u850": 5.0 + smooth(60.0),
        "v850": smooth(60.0),
        "q850": 0.01 + smooth(0.02),
        "u200": -5.0 + smooth(60.0),
        "v200": smooth(60.0),
    }
    coords = {"init_time": pd.date_range(f"{season}-06-01", periods=N_INIT), "lead_day": list(LEADS), "lat": lat, "lon": lon}
    return xr.Dataset({k: (("init_time", "lead_day", "lat", "lon"), v.astype("float32")) for k, v in data.items()}, coords=coords)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    cfg = load_config(data_dir=tmp_path_factory.mktemp("regime"))
    rng = np.random.default_rng(0)
    ids = np.sort(rng.choice(129 * 135, N_CELLS, replace=False))
    lat, lon = imd_axes(cfg.imd)
    cells = pd.DataFrame({
        "cell_id": ids, "latitude": lat[ids // 135], "longitude": lon[ids % 135],
        "grad_h_x": rng.normal(0, 0.05, N_CELLS), "grad_h_y": rng.normal(0, 0.05, N_CELLS),
        "coast_normal_x": np.cos(np.arange(N_CELLS)), "coast_normal_y": np.sin(np.arange(N_CELLS)),
        "dist_coast_km": rng.uniform(0, 500, N_CELLS),
    })
    dev, hold = [2021, 2022, 2023], [2024]
    fields = {s: make_fields(cfg, s, seed=s) for s in dev + hold}
    a_table = pd.concat([rp.lead_a_table(f, cfg) for f in fields.values()], ignore_index=True)
    inputs = {s: rp.season_inputs(f, ids, cfg) for s, f in fields.items()}
    # labels for the development years only: the holdout must not need any
    dates = pd.date_range("2021-06-01", periods=60)
    rows = [(d + pd.DateOffset(years=y - 2021), ["active", "normal", "break"][(d.dayofyear + y) % 3]) for y in dev for d in dates]
    labels = pd.Series(dict(rows), name="phase_label")
    return cfg, cells, a_table, inputs, labels, dev, hold


def test_a_features_are_one_row_per_run_and_lead(world):
    cfg, _, a_table, _, _, dev, hold = world
    assert len(a_table) == 4 * N_INIT * 3
    assert list(a_table[["A1", "A2", "A3", "A4", "A5", "A6"]].columns) == rp.A_IDS
    assert a_table[rp.A_IDS].notna().all().all()
    assert a_table["run_id"].str.startswith("tigge_ecmwf_cf_").all()
    one = a_table[(a_table["run_id"] == "tigge_ecmwf_cf_2021060100") & (a_table["lead_day"] == 2)].iloc[0]
    assert one["imd_date"] == pd.Timestamp("2021-06-03")  # end_date stamp: init + lead
    assert 950 < one["A4"] < 1050  # core-zone msl in hPa (the anomaly offset is applied later)
    assert 0 < a_table["A3"].mean() and 15 <= a_table["A3"].min() and a_table["A3"].max() <= 32  # trough latitude


class RecordingPhaseModel(PhaseModel):
    """A PhaseModel that remembers how many rows it was fitted on (module level, so that it can be pickled)."""

    def fit(self, X, y):
        self.n_train = len(X)
        return super().fit(X, y)


def run_loso(world, monkeypatch, **kw):
    cfg, cells, a_table, inputs, labels, dev, hold = world
    calls, fits = [], []
    orig = RegimeEngine.process_run_lead

    def spy(self, *a, **k):
        calls.append((k["run_id"], k["lead_day"], k["regime_source"], id(self), self.phase_model.n_train))
        return orig(self, *a, **k)

    monkeypatch.setattr(RegimeEngine, "process_run_lead", spy)
    monkeypatch.setattr(rp, "PhaseModel", RecordingPhaseModel)
    result = rp.run_regime_loso(a_table, inputs, labels, cells, kw.get("dev", dev), kw.get("hold", hold), cfg)
    return result, calls, fits


def test_process_run_lead_is_called_for_every_run_and_lead(world, monkeypatch):
    result, calls, _ = run_loso(world, monkeypatch)
    assert len(calls) == 4 * N_INIT * 3
    assert len({(c[0], c[1]) for c in calls}) == len(calls)  # each (run, lead) exactly once
    assert len(result.domain) == len(calls)


def test_development_seasons_are_out_of_fold_and_holdout_is_final(world, monkeypatch):
    result, calls, fits = run_loso(world, monkeypatch)
    by_season = {}
    for run, lead, source, eng, n_train in calls:
        by_season.setdefault(int(run[15:19]), set()).add((source, eng, n_train))
    for s in (2021, 2022, 2023):
        assert {c[0] for c in by_season[s]} == {"oof"} and len(by_season[s]) == 1
    assert {c[0] for c in by_season[2024]} == {"final"}
    # an out-of-fold engine never shares its model with another season, and never saw its own season
    engines = {next(iter(v))[1] for s, v in by_season.items() if s != 2024}
    assert len(engines) == 3
    n_oof = {next(iter(v))[2] for s, v in by_season.items() if s != 2024}
    n_final = next(iter(by_season[2024]))[2]
    assert len(n_oof) == 1 and n_final > next(iter(n_oof))  # 2 of 3 dev seasons vs all 3
    assert n_final == round(next(iter(n_oof)) * 3 / 2)
    assert [f["held_out"] for f in result.folds] == [2021, 2022, 2023, 2024]
    assert result.folds[0]["fit_on"] == [2022, 2023] and result.folds[3]["fit_on"] == [2021, 2022, 2023]
    assert 2024 not in {s for f in result.folds for s in f["fit_on"]}  # the holdout is never fitted on
    assert set(result.domain.loc[result.domain["season"] == 2024, "regime_source"]) == {"final"}
    assert set(result.domain.loc[result.domain["season"] != 2024, "regime_source"]) == {"oof"}


def test_the_14_features_are_produced_for_every_cell(world, monkeypatch):
    result, *_ = run_loso(world, monkeypatch)
    f = result.cell_features
    assert list(f.columns) == ["run_id", "lead_day", "cell_id", "season", *REGIME_14_FEATURES, "regime_source"]
    assert len(f) == 4 * N_INIT * 3 * N_CELLS
    assert not f.duplicated(["run_id", "lead_day", "cell_id"]).any()
    assert np.isfinite(f[REGIME_14_FEATURES].to_numpy()).all()
    p = f[["p_active", "p_normal", "p_break"]].sum(axis=1)
    assert np.allclose(p, 1.0, atol=3e-4)  # rounded to 4 decimals by the contract
    assert f[["orographic_influence", "coastal_influence", "lps_influence"]].ge(0).all().all()
    assert f[["orographic_influence", "coastal_influence", "lps_influence"]].le(1).all().all()
    assert f[["upslope_flux", "onshore_flux"]].ge(0).all().all()
    assert set(f["lps_present"]) <= {0.0, 1.0}
    assert result.domain["regime_available"].all()
    assert result.domain["lps_settings"].eq("untuned").all()  # the detector is untuned until the catalogue is used


def test_holdout_labels_are_not_needed_and_seasons_may_be_streamed(world, monkeypatch):
    cfg, cells, a_table, inputs, labels, dev, hold = world
    assert not any(d.year == 2024 for d in labels.index)  # the fixture has no holdout labels at all
    seen = {}
    monkeypatch.setattr(RegimeEngine, "process_run_lead", RegimeEngine.process_run_lead)
    result = rp.run_regime_loso(a_table, inputs, labels, cells, dev, hold, cfg,
                                on_season_done=lambda s, t: seen.__setitem__(s, len(t)))
    assert result.cell_features is None and seen == {s: N_INIT * 3 * N_CELLS for s in dev + hold}


def test_guards(world):
    cfg, cells, a_table, inputs, labels, dev, hold = world
    with pytest.raises(ValidationError, match="at least 2 development seasons"):
        rp.run_regime_loso(a_table, inputs, labels, cells, [2021], hold, cfg)
    with pytest.raises(ValidationError, match="both development and holdout"):
        rp.run_regime_loso(a_table, inputs, labels, cells, dev, [2023], cfg)
    with pytest.raises(MissingInputError, match="No regime fields"):
        rp.run_regime_loso(a_table, inputs, labels, cells, dev, [2030], cfg)


def test_merge_with_the_feature_table(world, monkeypatch):
    result, *_ = run_loso(world, monkeypatch)
    keys = result.cell_features[["run_id", "lead_day", "cell_id"]].copy()
    features = keys.assign(rain_mm=1.0)
    merged = rp.merge_regime_features(features, result.cell_features, result.domain)
    assert len(merged) == len(features)
    assert {*REGIME_14_FEATURES, "regime_source", "ood_flag", "regime_available"} <= set(merged.columns)
    with pytest.raises(ValidationError, match="no regime row"):
        rp.merge_regime_features(features.iloc[:-1].assign(cell_id=lambda d: d["cell_id"]).pipe(
            lambda d: pd.concat([d, d.iloc[[0]].assign(run_id="tigge_ecmwf_cf_1999010100")])), result.cell_features, result.domain)


def test_artifacts_can_be_saved(world, monkeypatch, tmp_path):
    import joblib

    result, *_ = run_loso(world, monkeypatch)
    out = rp.save_regime_artifacts(result, tmp_path / "regime")
    blob = joblib.load(out / "regime_final.joblib")
    assert blob["dev_seasons"] == [2021, 2022, 2023] and blob["best_c"] == result.best_c
    assert blob["phase_model"].is_fitted and blob["influence_table"]["upslope_quantiles"]
