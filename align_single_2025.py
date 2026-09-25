import xarray as xr
import glob
import os

IMD = "data/imd/RF25_ind2025_rfp25.nc"
imd = xr.open_dataset(IMD)

os.makedirs("data/aligned/2025", exist_ok=True)

for f in sorted(glob.glob("data/tigge/2025/single_*.grib")):

    month = f.split("single_")[1].split(".")[0]

    print(f"Processing {month}...")

    ds = xr.open_dataset(
        f,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""}
    )

    # Spatial alignment
    ds = ds.sel(
        latitude=imd.LATITUDE,
        longitude=imd.LONGITUDE
    )

    # MSL: keep forecast steps 6–72
    msl = ds.msl.sel(step=ds.step[1:])

    msl.to_netcdf(
        f"data/aligned/2025/msl_{month}_aligned.nc"
    )

    # Static fields: one copy only
    if month == "06":
        orog = ds.orog.isel(time=0, step=0)
        lsm = ds.lsm.isel(time=0, step=0)

        orog.to_netcdf(
            "data/aligned/2025/orog_aligned.nc"
        )

        lsm.to_netcdf(
            "data/aligned/2025/lsm_aligned.nc"
        )

    print(f"{month} complete")

print("SINGLE-LEVEL ALIGNMENT COMPLETE")
