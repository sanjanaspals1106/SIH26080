import cdsapi
import calendar

client = cdsapi.Client()
dataset = "tigge-forecasts"

for month in ["06", "07", "08", "09"]:
    days = [f"{d:02d}" for d in range(1, calendar.monthrange(2025, int(month))[1] + 1)]

    # Single-level
    client.retrieve(dataset, {
        "origin": "ecmf",
        "forecast_type": "control_forecast",
        "level_type": "single_level",
        "variable": [
            "mean_sea_level_pressure",
            "orography",
            "land_sea_mask"
        ],
        "year": ["2025"],
        "month": [month],
        "day": days,
        "time": "00:00",
        "leadtime_hour": ["0","6","12","18","24","30","36","42","48","54","60","66","72"],
        "grid": "0.25/0.25",
        "area": [40,55,0,100],
        "data_format": "grib",
    }, f"data/tigge/2025/single_{month}.grib")

    # Pressure levels
    client.retrieve(dataset, {
        "origin": "ecmf",
        "forecast_type": "control_forecast",
        "level_type": "pressure",
        "level": ["850","200"],
        "variable": [
            "u_component_of_wind",
            "v_component_of_wind",
            "specific_humidity"
        ],
        "year": ["2025"],
        "month": [month],
        "day": days,
        "time": "00:00",
        "leadtime_hour": ["6","12","18","24","30","36","42","48","54","60","66","72"],
        "grid": "0.25/0.25",
        "area": [40,55,0,100],
        "data_format": "grib",
    }, f"data/tigge/2025/pressure_{month}.grib")

    print(f"{month} complete")
