"""T4 lag test (PRD 8.2): the decision rule and a synthetic end-to-end run."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data_pipeline.alignment import decide_imd_stamp, lag_test
from data_pipeline.ingestion import ValidationError

M = 0.02


@pytest.mark.parametrize(
    "avgs, expected",
    [
        ({-1: 0.30, 0: 0.50, 1: 0.20}, (0, "end_date")),  # shift 0 highest
        ({-1: 0.60, 0: 0.40, 1: 0.10}, (-1, "start_date")),  # shift -1 highest by more than 0.02
        ({-1: 0.41, 0: 0.40, 1: 0.10}, (-1, "end_date")),  # best beats shift 0 by 0.01 < 0.02 -> keep
        ({-1: 0.10, 0: 0.40, 1: 0.41}, (1, "end_date")),  # +1 is best but within 0.02 -> keep
        ({-1: 0.5, 0: 0.5, 1: 0.1}, (0, "end_date")),  # tie goes to shift 0
    ],
)
def test_decision_table(avgs, expected):
    assert decide_imd_stamp(avgs, M) == expected


def test_difference_of_exactly_the_margin_does_not_keep_end_date():
    # "differ by less than the margin -> keep": equal to the margin is not less (0.75 - 0.5 is exact in binary)
    assert decide_imd_stamp({-1: 0.75, 0: 0.5, 1: 0.0}, 0.25) == (-1, "start_date")
    assert decide_imd_stamp({-1: 0.74, 0: 0.5, 1: 0.0}, 0.25) == (-1, "end_date")


def test_shift_plus_one_highest_stops_the_pipeline():
    with pytest.raises(ValidationError, match="Stop"):
        decide_imd_stamp({-1: 0.1, 0: 0.3, 1: 0.6}, M)


def test_all_three_shifts_are_required():
    with pytest.raises(ValidationError, match="needs shifts"):
        decide_imd_stamp({0: 0.5, 1: 0.2}, M)


# ---- end to end on synthetic fields --------------------------------------------------------

N_DAYS, NLAT, NLON = 30, 12, 14
DAYS = pd.date_range("2024-07-01", periods=N_DAYS)


@pytest.fixture(scope="module")
def truth():
    """Independent random 'weather' for each day. imd[D] is the IMD field dated D."""
    rng = np.random.default_rng(42)
    fields = rng.gamma(0.6, 8.0, size=(N_DAYS, NLAT, NLON))
    return xr.DataArray(
        fields,
        dims=("time", "lat", "lon"),
        coords={"time": DAYS, "lat": np.arange(NLAT) * 0.25, "lon": np.arange(NLON) * 0.25},
    )


def forecast_from(truth, offset_days, noise=0.05, seed=1):
    """Lead-1 window rain for start dates I in 2024-07-03..2024-07-20 that equals truth dated I + offset."""
    rng = np.random.default_rng(seed)
    inits = DAYS[2:20]
    data = np.stack([truth.sel(time=i + pd.Timedelta(days=offset_days)).values for i in inits])
    data = data + rng.normal(0, noise, data.shape)
    return xr.DataArray(
        data,
        dims=("init_time", "lat", "lon"),
        coords={"init_time": inits, "lat": truth["lat"], "lon": truth["lon"]},
    )


MASK = np.ones((NLAT, NLON), dtype=bool)


def test_forecast_matching_imd_dated_I_plus_1_keeps_end_date(cfg, truth):
    res = lag_test(forecast_from(truth, 1), truth, MASK, cfg.alignment)
    assert res.best_shift == 0 and res.imd_stamp == "end_date"
    assert res.averages[0] > 0.95 and abs(res.averages[-1]) < 0.4 and abs(res.averages[1]) < 0.4
    assert res.n_dates == {-1: 18, 0: 18, 1: 18}


def test_forecast_matching_imd_dated_I_switches_to_start_date(cfg, truth):
    res = lag_test(forecast_from(truth, 0), truth, MASK, cfg.alignment)
    assert res.best_shift == -1 and res.imd_stamp == "start_date"


def test_forecast_matching_imd_dated_I_plus_2_stops(cfg, truth):
    with pytest.raises(ValidationError, match="shift \\+1"):
        lag_test(forecast_from(truth, 2), truth, MASK, cfg.alignment)


def test_lag_test_is_deterministic(cfg, truth):
    a = lag_test(forecast_from(truth, 1), truth, MASK, cfg.alignment)
    b = lag_test(forecast_from(truth, 1), truth, MASK, cfg.alignment)
    assert a == b


def test_invalid_cells_are_ignored(cfg, truth):
    fc = forecast_from(truth, 1)
    mask = MASK.copy()
    mask[:, :4] = False
    fc.values[:, :, :4] = 1e6  # garbage in the masked cells must not matter
    res = lag_test(fc, truth, mask, cfg.alignment)
    assert res.best_shift == 0 and res.averages[0] > 0.95


def test_dates_without_imd_are_skipped_and_no_data_is_an_error(cfg, truth):
    short = truth.isel(time=slice(0, 10))  # IMD stops early: the later start dates have no observation
    res = lag_test(forecast_from(truth, 1), short, MASK, cfg.alignment)
    assert res.n_dates[0] < 18
    with pytest.raises(ValidationError, match="no usable start date"):
        lag_test(forecast_from(truth, 1), truth.isel(time=slice(0, 1)), MASK, cfg.alignment)
