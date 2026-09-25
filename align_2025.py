import xarray as xr
import glob
import os

IMD = "data/imd/RF25_ind2025_rfp25.nc"
imd = xr.open_dataset(IMD)

os.makedirs("data/aligned/2025", exist_ok=True)

for f in sorted(glob.glob("data/tigge/2025/tp_*.grib")):

    month = f.split("tp_2025")[1].split(".")[0]

    print(f"Aligning {month}...")

    tigge = xr.open_dataset(
        f,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""}
    )

    aligned = tigge.sel(
        latitude=imd.LATITUDE,
        longitude=imd.LONGITUDE
    )

    output = f"data/aligned/2025/tp_2025{month}_aligned.nc"

    aligned.to_netcdf(output)

    print(f"Saved: {output}")
    print(f"Shape: {aligned.tp.shape}")

print("ALL 2025 PRECIPITATION ALIGNED")