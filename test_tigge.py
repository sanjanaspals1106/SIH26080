import cdsapi

client = cdsapi.Client()

client.retrieve(
    "tigge-forecasts",
    {
        "origin": "ecmwf",
        "forecast_type": "control_forecast",
        "level_type": "single_level",
        "variable": ["total_precipitation"],
        "year": ["2025"],
        "month": ["09"],
        "day": ["01"],
        "time": "00:00",
        "leadtime_hour": ["6"],
        "area": [40, -55, 0, 100],
        "data_format": "grib",
    },
    "data/tigge_test/tp_20250901.grib",
)

print("DOWNLOAD OK")
