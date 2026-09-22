"""Temporal alignment, method C1: T1, T2, T3, T6 and the window arithmetic (PRD 8.1 to 8.3)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data_pipeline.alignment import (
    c1_atmosphere_windows,
    c1_rain_windows,
    imd_date,
    window_hours,
)
from data_pipeline.alignment.temporal import (
    atmosphere_steps,
    check_tp_cumulative,
    cumulative_at,
    imd_day_end_utc_hours,
    window_bounds_utc,
)
from data_pipeline.ingestion import IngestionError, ValidationError

STEPS = list(range(0, 79, 6))
# Rain rate (mm per hour) inside each 6 h block: block i covers hours 6i to 6i+6.
RATES = np.arange(1.0, 14.0)  # 13 blocks: 1, 2, ..., 13


def tp_from_rates(rates=RATES, factors=(1.0, 2.0), inits=("2024-07-01",)):
    """Cumulative tp on a tiny grid; cell factors scale the rain so cells differ."""
    cum = np.concatenate([[0.0], np.cumsum(6 * rates)])  # cum at steps 0, 6, ..., 78
    data = cum[None, :, None, None] * np.asarray(factors)[None, None, None, :]
    return xr.DataArray(
        np.broadcast_to(data, (len(inits), len(STEPS), 1, len(factors))).astype("float64").copy(),
        dims=("init_time", "lead_hours", "lat", "lon"),
        coords={
            "init_time": pd.to_datetime(list(inits)),
            "lead_hours": STEPS,
            "lat": [10.0],
            "lon": 70.0 + 0.25 * np.arange(len(factors)),
        },
        name="tp",
    )


def hourly_sum(lo, hi, rates=RATES):
    """Independent check: add up rain hour by hour, each hour with the rate of its 6 h block."""
    return sum(rates[h // 6] for h in range(lo, hi))


# ---- T6 and the window definition ----------------------------------------------------------


def test_t6_0830_ist_is_0300_utc(cfg):
    assert imd_day_end_utc_hours(cfg.alignment) == 3


def test_t6_inconsistent_config_is_rejected(cfg):
    bad = dataclasses.replace(cfg.alignment, imd_day_end_utc="04:00")
    with pytest.raises(ValidationError, match="08:30 IST"):
        imd_day_end_utc_hours(bad)


def test_windows_are_03_27_27_51_51_75(cfg):
    assert [window_hours(k, cfg.alignment) for k in (1, 2, 3)] == [(3, 27), (27, 51), (51, 75)]


def test_atmosphere_steps_inside_each_window(cfg):
    steps = list(range(6, 73, 6))
    assert atmosphere_steps(1, steps, cfg.alignment) == [6, 12, 18, 24]
    assert atmosphere_steps(2, steps, cfg.alignment) == [30, 36, 42, 48]
    assert atmosphere_steps(3, steps, cfg.alignment) == [54, 60, 66, 72]


def test_imd_date_and_window_bounds(cfg):
    init = pd.Timestamp("2024-07-15")
    assert imd_date(init, 1, "end_date") == pd.Timestamp("2024-07-16")
    assert imd_date(init, 3, "end_date") == pd.Timestamp("2024-07-18")
    assert imd_date(init, 1, "start_date") == pd.Timestamp("2024-07-15")
    assert window_bounds_utc(init, 1, cfg.alignment) == (
        pd.Timestamp("2024-07-15 03:00"),
        pd.Timestamp("2024-07-16 03:00"),
    )
    with pytest.raises(IngestionError, match="imd_stamp"):
        imd_date(init, 1, "middle")


def test_only_c1_is_implemented(cfg):
    with pytest.raises(IngestionError, match="only C1"):
        c1_rain_windows(tp_from_rates(), dataclasses.replace(cfg.alignment, method="C0"))


# ---- cumulative interpolation --------------------------------------------------------------


def test_cumulative_is_linear_between_steps_and_zero_at_step_0(cfg):
    tp = tp_from_rates(factors=(1.0,))
    tp[:, 0] = 999.0  # whatever the file holds at step 0, PRD 8.3 says the value there is 0
    assert cumulative_at(tp, 0).item() == 0
    assert cumulative_at(tp, 3).item() == pytest.approx(3 * RATES[0])  # half of the first 6 h block
    assert cumulative_at(tp, 6).item() == pytest.approx(6 * RATES[0])
    assert cumulative_at(tp, 27).item() == pytest.approx(hourly_sum(0, 27))
    with pytest.raises(ValidationError, match="do not cover"):
        cumulative_at(tp, 100)


# ---- T3: synthetic alignment test ----------------------------------------------------------


def test_t3_lead_windows_have_the_expected_rain(cfg):
    tp = tp_from_rates()
    out = c1_rain_windows(tp, cfg.alignment)
    assert out["rain_mm"].dims == ("init_time", "lead_day", "lat", "lon")
    assert list(out["lead_day"].values) == [1, 2, 3]
    got = out["rain_mm"].isel(init_time=0, lat=0, lon=0).values  # cell factor 1
    # Worked by hand: lead 1 = 3 h at rate 1 + 18 h at rates 2,3,4 (6 h each) + 3 h at rate 5 = 3+54+15 = 72
    assert got[0] == pytest.approx(72.0)
    assert got == pytest.approx([hourly_sum(3, 27), hourly_sum(27, 51), hourly_sum(51, 75)])
    assert got == pytest.approx([72.0, 168.0, 264.0])
    # a second cell with factor 2 has exactly twice the rain
    assert out["rain_mm"].isel(init_time=0, lat=0, lon=1).values == pytest.approx(2 * got)


def test_t3_a_wrong_window_would_be_caught(cfg):
    """00-24 UTC (method C0 style) or a window shifted by one lead gives different numbers."""
    tp = tp_from_rates(factors=(1.0,))
    got = c1_rain_windows(tp, cfg.alignment)["rain_mm"].isel(init_time=0, lat=0, lon=0).values
    c0_like = [hourly_sum(0, 24), hourly_sum(24, 48), hourly_sum(48, 72)]
    assert not np.allclose(got, c0_like)
    assert not np.isclose(got[0], got[1]) and not np.isclose(got[1], got[2])  # leads are not mixed up
    assert got[0] != pytest.approx(hourly_sum(27, 51)) and got[2] != pytest.approx(hourly_sum(27, 51))


def test_t3_uniform_rain_gives_24_hours_of_rain_in_every_lead(cfg):
    tp = tp_from_rates(rates=np.full(13, 0.5), factors=(1.0,))
    got = c1_rain_windows(tp, cfg.alignment)["rain_mm"].values.ravel()
    assert got == pytest.approx([12.0, 12.0, 12.0])  # 24 h x 0.5 mm/h


def test_units_and_start_times_are_preserved(cfg):
    tp = tp_from_rates(inits=("2024-07-01", "2024-07-02"))
    out = c1_rain_windows(tp, cfg.alignment)
    assert out["rain_mm"].dtype == np.float32
    assert list(out["init_time"].values) == list(tp["init_time"].values)
    assert list(out["lat"].values) == [10.0] and list(out["lon"].values) == [70.0, 70.25]


# ---- T1 and T2 -----------------------------------------------------------------------------


def test_t1_passes_for_cumulative_rain(cfg):
    assert check_tp_cumulative(tp_from_rates(), cfg.alignment) == 1.0


def test_t1_fails_when_tp_is_not_cumulative(cfg):
    not_cum = tp_from_rates()
    not_cum.values[:] = np.random.default_rng(0).random(not_cum.shape)  # goes up and down
    with pytest.raises(ValidationError, match="T1 failed"):
        c1_rain_windows(not_cum, cfg.alignment)


def test_small_negative_windows_are_clipped_and_counted(cfg):
    tp = tp_from_rates(factors=(1.0,) * 2000)
    tp.values[0, 1:, 0, 0] = 0.0  # one series: 6 mm at step 6, then it "drops" back to 0
    tp.values[0, 1, 0, 0] = 10.0
    out = c1_rain_windows(tp, cfg.alignment)  # 1 series of 2000 is under the 0.1% limits of T1 and T2
    assert out.attrs["n_clipped"] == 1 and out.attrs["n_values"] == 3 * 2000
    assert out["clipped"].isel(init_time=0, lat=0, lon=0, lead_day=0).item()
    assert out["rain_mm"].isel(init_time=0, lat=0, lon=0).values.min() == 0.0  # never negative


def test_t2_fails_when_too_many_windows_are_negative(cfg):
    tp = tp_from_rates(factors=(1.0,))
    tp.values[:] = (
        5.0 - 1e-4 * tp["lead_hours"].values[None, :, None, None]
    )  # falls by less than the T1 tolerance
    with pytest.raises(ValidationError, match="T2 failed"):
        c1_rain_windows(tp, cfg.alignment)
    out = c1_rain_windows(tp, cfg.alignment, check=False)  # the counts are still there
    assert out.attrs["n_clipped"] > 0


# ---- atmosphere ----------------------------------------------------------------------------


def test_atmosphere_is_the_mean_of_the_four_steps_in_the_window(cfg):
    steps = list(range(6, 73, 6))
    data = np.array(steps, dtype="float32")[None, :, None, None] * np.ones((1, 1, 1, 1), "float32")
    atm = xr.Dataset(
        {"u850": (("init_time", "lead_hours", "lat", "lon"), data)},
        coords={
            "init_time": pd.to_datetime(["2024-07-01"]),
            "lead_hours": steps,
            "lat": [10.0],
            "lon": [70.0],
        },
    )
    atm["u850"].attrs["units"] = "m s**-1"
    out = c1_atmosphere_windows(atm, cfg.alignment)
    assert out["u850"].dims == ("init_time", "lead_day", "lat", "lon")
    assert out["u850"].values.ravel() == pytest.approx(
        [15.0, 39.0, 63.0]
    )  # mean(6..24), mean(30..48), mean(54..72)
    assert out["u850"].attrs["units"] == "m s**-1"


def test_atmosphere_without_steps_in_a_window_is_an_error(cfg):
    atm = xr.Dataset(
        {"u850": (("init_time", "lead_hours", "lat", "lon"), np.ones((1, 2, 1, 1)))},
        coords={
            "init_time": pd.to_datetime(["2024-07-01"]),
            "lead_hours": [6, 12],
            "lat": [10.0],
            "lon": [70.0],
        },
    )
    with pytest.raises(ValidationError, match="lead day 2"):
        c1_atmosphere_windows(atm, cfg.alignment)
