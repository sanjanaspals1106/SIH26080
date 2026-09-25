"""
Build C1 rainfall windows from aligned ECMWF TIGGE cumulative precipitation.

C1 windows:
  Lead 1: 3 -> 27 h
  Lead 2: 27 -> 51 h
  Lead 3: 51 -> 75 h

The script linearly interpolates cumulative TP between the nearest 6-hour
forecast steps, computes end-start rainfall, reports negative windows before
clipping, clips negative rainfall to zero, and writes one NetCDF file.
"""

from pathlib import Path
import numpy as np
import xarray as xr

YEAR = 2025

INPUT_DIR = Path(f"data/aligned/{YEAR}")
OUTPUT_DIR = Path(f"data/processed/{YEAR}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / f"rain_windows_{YEAR}.nc"

MONTHS = [6, 7, 8, 9]

LEAD_WINDOWS = {
    1: (3, 27),
    2: (27, 51),
    3: (51, 75),
}


def cumulative_at(ds, hour):
    """Linearly interpolate cumulative TP at an arbitrary forecast hour."""
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


all_months = []
t2_records = []

for month in MONTHS:
    path = INPUT_DIR / f"tp_{YEAR}{month:02d}_aligned.nc"

    print(f"Reading {path}")

    ds = xr.open_dataset(path)
    # Remove tiny numerical decreases in cumulative TP.
    # TP is cumulative, so it must be non-decreasing with forecast step.
    ds["tp"] = xr.apply_ufunc(
    np.maximum.accumulate,
    ds["tp"],
    input_core_dims=[["step"]],
    output_core_dims=[["step"]],
    vectorize=True,
    )
    print(f"Units: {ds['tp'].attrs.get('units', 'unknown')}")

    month_results = []

    for lead_day, (start, end) in LEAD_WINDOWS.items():
        start_tp = cumulative_at(ds, start)
        end_tp = cumulative_at(ds, end)

        rain = end_tp - start_tp

        negative = rain < 0
        negative_count = int(negative.sum())
        total_count = int(rain.size)
        negative_pct = 100.0 * negative_count / total_count

        if negative_count:
            neg_values = rain.where(negative).values
            neg_values = neg_values[np.isfinite(neg_values)]

            print(
                f"  Lead {lead_day}: {start} -> {end} h | "
                f"negative corrected: {negative_count} "
                f"({negative_pct:.6f}%) | "
                f"magnitude min={float(neg_values.min()):.6f} mm, "
                f"max={float(neg_values.max()):.6f} mm, "
                f"median={float(np.median(neg_values)):.6f} mm"
            )
        else:
            print(
                f"  Lead {lead_day}: {start} -> {end} h | "
                f"negative corrected: 0 (0.000000%)"
            )

        t2_records.append(
            {
                "month": month,
                "lead_day": lead_day,
                "negative_count": negative_count,
                "total_count": total_count,
                "negative_pct": negative_pct,
            }
        )

        # PRD C1: negative rainfall windows are clipped to zero.
        rain = rain.clip(min=0)

        rain = rain.expand_dims(lead_day=[lead_day])
        month_results.append(rain)

    month_da = xr.concat(month_results, dim="lead_day")
    all_months.append(month_da)

    ds.close()

rain_windows = xr.concat(all_months, dim="time")
rain_windows = rain_windows.sortby("time")

rain_windows.name = "rain_mm"
rain_windows.attrs.update(
    {
        "description": "ECMWF TIGGE control forecast daily rainfall windows",
        "alignment_method": "C1",
        "window_definition": "3-27, 27-51, 51-75 hours",
        "unit": "mm",
    }
)

# Keep only the intended public dimensions/order.
rain_windows = rain_windows.transpose(
    "lead_day", "time", "LATITUDE", "LONGITUDE"
)

rain_windows.to_netcdf(OUTPUT_FILE)

print()
print(f"Saved: {OUTPUT_FILE}")
print(rain_windows)
print()
print("T2 summary:")

for r in t2_records:
    status = "PASS" if r["negative_pct"] <= 0.1 else "FAIL"
    print(
        f"  {YEAR}-{r['month']:02d} Lead {r['lead_day']}: "
        f"{r['negative_pct']:.6f}% -> {status}"
    )
