import xarray as xr
import glob
import os

IMD = "data/imd/RF25_ind2025_rfp25.nc"
imd = xr.open_dataset(IMD)

os.makedirs("data/aligned/2025", exist_ok=True)

for f in sorted(glob.glob("data/tigge/2025/pressure_*.grib")):

    month = f.split("pressure_")[1].split(".")[0]

    print(f"Aligning pressure {month}...")

    ds = xr.open_dataset(
        f,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""}
    )

    aligned = ds.sel(
        latitude=imd.LATITUDE,
        longitude=imd.LONGITUDE
    )

    output = f"data/aligned/2025/pressure_{month}_aligned.nc"

    aligned.to_netcdf(output)

    print("Saved:", output)
    print("Shape:", aligned.u.shape)

print("ALL PRESSURE DATA ALIGNED")
