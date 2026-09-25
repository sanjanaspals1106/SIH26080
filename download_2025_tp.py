import cdsapi
import calendar
import os

client = cdsapi.Client()

dataset = "tigge-forecasts"

for month in ["06", "07", "08", "09"]:

    days = [
        f"{d:02d}"
        for d in range(1, calendar.monthrange(2025, int(month))[1] + 1)
    ]

    request = {
        "origin": "ecmf",
        "forecast_type": "control_forecast",
        "level_type": "single_level",
        "variable": ["total_precipitation"],
        "year": ["2025"],
        "month": [month],
        "day": days,
        "time": "00:00",
        "leadtime_hour": [
            "0", "6", "12", "18",
            "24", "30", "36", "42",
            "48", "54", "60", "66",
            "72", "78"
        ],
        "grid": "0.25/0.25",
        "area": [40, 55, 0, 100],
        "data_format": "grib",
    }

    output = f"data/tigge/2025/tp_2025{month}.grib"

    print(f"\nDownloading {output}...")

    client.retrieve(dataset, request, output)

    print(f"Completed {month}")
