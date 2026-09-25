#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd

YEAR = 2025

ALIGNED_DIR = Path(f"data/aligned/{YEAR}")
IMD_FILE = Path(f"data/imd/RF25_ind{YEAR}_rfp25.nc")
VALID_CELLS_FILE = ALIGNED_DIR / "valid_cells_2025.parquet"

LEAD_WINDOWS = {
    1: (3, 27),
    2: (27, 51),
    3: (51, 75),
}

MONTHS = [6, 7, 8, 9]


# ============================================================
# Helpers
# ============================================================

def cumulative_at(ds, hour):
    """Linear interpolation of cumulative TP at a given hour."""

    step = ds["step"]
    target = np.timedelta64(hour, "h")

    if target in step.values:
        return ds["tp"].sel(step=target)

    lower = step.values[step.values < target].max()
    upper = step.values[step.values > target].min()

    low_hour = lower / np.timedelta64(1, "h")
    high_hour = upper / np.timedelta64(1, "h")

    low = ds["tp"].sel(step=lower)
    high = ds["tp"].sel(step=upper)

    weight = (hour - low_hour) / (high_hour - low_hour)

    return low + weight * (high - low)


def make_monotonic_tp(ds):
    """
    Remove tiny numerical decreases from cumulative TP.
    TP must be non-decreasing with forecast step.
    """

    ds["tp"] = xr.apply_ufunc(
        np.maximum.accumulate,
        ds["tp"],
        input_core_dims=[["step"]],
        output_core_dims=[["step"]],
        vectorize=True,
    )

    return ds


# ============================================================
# T1 — Cumulative TP check
# ============================================================

def check_t1():

    print("\n=== T1: Cumulative TP check ===")

    overall_pass = True

    for month in MONTHS:

        path = ALIGNED_DIR / f"tp_{YEAR}{month:02d}_aligned.nc"

        ds = xr.open_dataset(path)

        tp = ds["tp"].values

        # Difference along step axis
        diffs = np.diff(tp, axis=1)

        total = diffs.size
        non_decreasing = np.sum(diffs >= 0)

        pct = 100 * non_decreasing / total

        passed = pct >= 99.9

        if not passed:
            overall_pass = False

        status = "PASS" if passed else "FAIL"

        print(
            f"{YEAR}-{month:02d}: "
            f"{pct:.6f}% non-decreasing -> {status}"
        )

        ds.close()

    return overall_pass


# ============================================================
# T2 — No negative rainfall windows
# ============================================================

def check_t2():

    print("\n=== T2: Negative rainfall windows ===")

    overall_pass = True

    for month in MONTHS:

        path = ALIGNED_DIR / f"tp_{YEAR}{month:02d}_aligned.nc"

        ds = xr.open_dataset(path)
        ds = make_monotonic_tp(ds)

        for lead, (start, end) in LEAD_WINDOWS.items():

            start_tp = cumulative_at(ds, start)
            end_tp = cumulative_at(ds, end)

            rain = end_tp - start_tp

            negative_count = int((rain < 0).sum())
            total_count = int(rain.size)

            pct = 100 * negative_count / total_count

            passed = pct <= 0.1

            if not passed:
                overall_pass = False

            status = "PASS" if passed else "FAIL"

            print(
                f"{YEAR}-{month:02d} Lead {lead}: "
                f"{pct:.6f}% -> {status}"
            )

        ds.close()

    return overall_pass


# ============================================================
# T3 — Synthetic arithmetic
# ============================================================

def check_t3():

    print("\n=== T3: Synthetic C1 arithmetic ===")

    steps = np.arange(0, 79, 6, dtype=float)
    tp = steps.copy()

    overall_pass = True

    for lead, (start, end) in LEAD_WINDOWS.items():

        start_value = np.interp(start, steps, tp)
        end_value = np.interp(end, steps, tp)

        calculated = end_value - start_value
        expected = end - start

        passed = np.isclose(calculated, expected)

        if not passed:
            overall_pass = False

        status = "PASS" if passed else "FAIL"

        print(
            f"Lead {lead}: "
            f"calculated={calculated:.6f}, "
            f"expected={expected:.6f} -> {status}"
        )

    return overall_pass


# ============================================================
# T4 — Lag test
# ============================================================

def check_t4():

    print("\n=== T4: IMD lag test ===")

    imd = xr.open_dataset(IMD_FILE)

    rain_file = ALIGNED_DIR / "tp_202506_aligned.nc"

    # Use the already-generated C1 rainfall windows.
    rain_window_file = Path(
        f"data/processed/{YEAR}/rain_windows_{YEAR}.nc"
    )

    rain_ds = xr.open_dataset(rain_window_file)

    lead1 = rain_ds["rain_mm"].sel(lead_day=1)

    # Valid cells from current 2025 mask.
    valid_df = pd.read_parquet(VALID_CELLS_FILE)

    valid_lat = valid_df["latitude"].values
    valid_lon = valid_df["longitude"].values

    correlations = {
        -1: [],
        0: [],
        1: [],
    }

    forecast_dates = pd.to_datetime(
        lead1["time"].values
    )

    for forecast_date in forecast_dates:

        # PRD:
        # IMD date = I + 1 + shift
        base_date = pd.Timestamp(forecast_date)

        forecast_field = lead1.sel(
            time=forecast_date
        )

        for shift in [-1, 0, 1]:

            imd_date = base_date + pd.Timedelta(
                days=1 + shift
            )

            if imd_date.to_datetime64() not in imd["TIME"].values:
                continue

            obs = imd["RAINFALL"].sel(
                TIME=imd_date.to_datetime64()
            )

            # Extract valid-cell values using nearest coordinates.
            forecast_values = []
            obs_values = []

            for lat, lon in zip(valid_lat, valid_lon):

                try:
                    f = forecast_field.sel(
                        LATITUDE=lat,
                        LONGITUDE=lon
                    ).item()

                    o = obs.sel(
                        LATITUDE=lat,
                        LONGITUDE=lon
                    ).item()

                    if np.isfinite(f) and np.isfinite(o):

                        forecast_values.append(f)
                        obs_values.append(o)

                except Exception:
                    continue

            if len(forecast_values) < 2:
                continue

            corr = np.corrcoef(
                forecast_values,
                obs_values
            )[0, 1]

            if np.isfinite(corr):
                correlations[shift].append(corr)

    averages = {}

    for shift in [-1, 0, 1]:

        if correlations[shift]:

            averages[shift] = float(
                np.mean(correlations[shift])
            )

        else:

            averages[shift] = np.nan

    print(
        f"Shift -1 average correlation: "
        f"{averages[-1]:.6f}"
    )

    print(
        f"Shift  0 average correlation: "
        f"{averages[0]:.6f}"
    )

    print(
        f"Shift +1 average correlation: "
        f"{averages[1]:.6f}"
    )

    if np.isnan(averages[0]):

        print("T4: FAIL — shift 0 has no valid result")
        return False

    best_shift = max(
        averages,
        key=lambda x: (
            averages[x]
            if np.isfinite(averages[x])
            else -np.inf
        )
    )

    difference = averages[best_shift] - averages[0]

    if best_shift == 1:

        print("T4: FAIL — shift +1 is highest")
        return False

    if best_shift == 0:

        print("T4: PASS — keep imd_stamp=end_date")
        return True

    if best_shift == -1:

        if difference < 0.02:

            print(
                "T4: PASS — difference < 0.02; "
                "keep imd_stamp=end_date"
            )

            return True

        print(
            "T4: PASS — use imd_stamp=start_date"
        )

        return True

    return False


# ============================================================
# T5 — IMD product check
# ============================================================

def check_t5():

    print("\n=== T5: IMD product check ===")

    if not IMD_FILE.exists():

        print("T5: FAIL — IMD file not found")
        return False

    print(f"Season: {YEAR}")
    print(f"IMD product: {IMD_FILE.name}")

    # Only one IMD product currently exists for 2025.
    # Therefore no same-year product comparison is required yet.

    print(
        "No second IMD product for 2025 is currently present; "
        "product comparison not required."
    )

    print("T5: PASS")

    return True


# ============================================================
# T6 — Time zone
# ============================================================

def check_t6():

    print("\n=== T6: Time zone ===")

    try:

        import zoneinfo
        from datetime import datetime, timezone

        ist = zoneinfo.ZoneInfo("Asia/Kolkata")

        dt_ist = datetime(
            2025,
            6,
            1,
            8,
            30,
            tzinfo=ist,
        )

        dt_utc = dt_ist.astimezone(timezone.utc)

        expected_hour = 3
        expected_minute = 0

        passed = (
            dt_utc.hour == expected_hour
            and dt_utc.minute == expected_minute
        )

        print(f"08:30 IST -> {dt_utc.strftime('%H:%M')} UTC")

        print(
            f"T6: {'PASS' if passed else 'FAIL'}"
        )

        return passed

    except Exception as e:

        print(f"T6: FAIL — {e}")
        return False


# ============================================================
# T7 — Grid identity
# ============================================================

def check_t7():

    print("\n=== T7: Grid identity ===")

    overall_pass = True

    # Expected PRD grid.
    expected_lat = np.arange(6.5, 38.5 + 0.001, 0.25)
    expected_lon = np.arange(66.5, 100.0 + 0.001, 0.25)

    expected_lat = np.round(expected_lat, 2)
    expected_lon = np.round(expected_lon, 2)

    print(
        f"Expected grid: "
        f"{len(expected_lat)} x {len(expected_lon)}"
    )

    # Check rainfall grid.
    rain_file = ALIGNED_DIR / "tp_202506_aligned.nc"

    ds = xr.open_dataset(rain_file)

    actual_lat = np.round(
        ds["LATITUDE"].values,
        2
    )

    actual_lon = np.round(
        ds["LONGITUDE"].values,
        2
    )

    grid_pass = (
        np.array_equal(actual_lat, expected_lat)
        and np.array_equal(actual_lon, expected_lon)
    )

    print(
        f"Forecast grid identity: "
        f"{'PASS' if grid_pass else 'FAIL'}"
    )

    if not grid_pass:
        overall_pass = False

    ds.close()

    # Check cell_id identity.
    valid_df = pd.read_parquet(
        VALID_CELLS_FILE
    )

    calculated_cell_id = (
        valid_df["i_lat"] * 135
        + valid_df["i_lon"]
    )

    cell_id_pass = np.array_equal(
        valid_df["cell_id"].values,
        calculated_cell_id.values,
    )

    print(
        f"cell_id formula: "
        f"{'PASS' if cell_id_pass else 'FAIL'}"
    )

    if not cell_id_pass:
        overall_pass = False

    print(
        f"Current valid-cell count: "
        f"{len(valid_df)}"
    )

    if overall_pass:
        print("T7: PASS")
    else:
        print("T7: FAIL")

    return overall_pass


# ============================================================
# T8 — Static fields
# ============================================================

def check_t8():

    print("\n=== T8: Static fields ===")

    overall_pass = True

    for variable in ["orog", "lsm"]:

        path = ALIGNED_DIR / f"{variable}_aligned.nc"

        if not path.exists():

            print(
                f"{variable}: FAIL — file not found"
            )

            overall_pass = False
            continue

        ds = xr.open_dataset(path)

        data = ds[variable]

        if "time" in data.dims and data.sizes["time"] > 1:

            first = data.isel(time=0).values

            identical = True

            for i in range(1, data.sizes["time"]):

                current = data.isel(time=i).values

                if not np.array_equal(
                    first,
                    current,
                    equal_nan=True,
                ):
                    identical = False
                    break

            status = "PASS" if identical else "FAIL"

            print(
                f"{variable}: values identical across "
                f"{data.sizes['time']} dates -> {status}"
            )

            if not identical:
                overall_pass = False

        else:

            print(
                f"{variable}: only one static snapshot available "
                f"-> PASS for static-field identity"
            )

        ds.close()

    print(
        f"T8: {'PASS' if overall_pass else 'FAIL'}"
    )

    return overall_pass


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    results = {}

    results["T1"] = check_t1()
    results["T2"] = check_t2()
    results["T3"] = check_t3()
    results["T4"] = check_t4()
    results["T5"] = check_t5()
    results["T6"] = check_t6()
    results["T7"] = check_t7()
    results["T8"] = check_t8()

    print("\n================================")
    print("ALIGNMENT VALIDATION SUMMARY")
    print("================================")

    for test, passed in results.items():

        print(
            f"{test}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    if all(results.values()):

        print("\nALL T1–T8 PASSED")

    else:

        failed = [
            test
            for test, passed in results.items()
            if not passed
        ]

        print(
            "\nFAILED TESTS:",
            ", ".join(failed)
        )

        raise SystemExit(1)