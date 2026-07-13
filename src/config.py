"""Station and parameter configuration, verified against USGS site metadata."""

STATIONS = {
    "buford_dam": {
        "usgs_id": "02334430",
        "usgs_site_name": "CHATTAHOOCHEE RIVER AT BUFORD DAM, NEAR BUFORD, GA",
        # Verified via dataretrieval.nwis.get_info / waterdata.get_monitoring_locations
        "lat": 34.156667,
        "lon": -84.078417,
        "parameters": {
            "00060": "discharge_cfs",
            "00010": "water_temp_c",
            "00300": "dissolved_oxygen_mgl",
            "00095": "conductance_uscm",
        },
    },
    "medlock_bridge": {
        "usgs_id": "02335000",
        # Real mainstem station near Medlock Bridge Rd (not the same as
        # 02334578, a small tributary elsewhere on the watershed).
        "usgs_site_name": "CHATTAHOOCHEE RIVER NEAR NORCROSS, GA",
        "lat": 33.997222,
        "lon": -84.201944,
        "parameters": {
            "00060": "discharge_cfs",
            "00010": "water_temp_c",
            "63680": "turbidity_fnu",
            "00095": "conductance_uscm",
        },
    },
}

OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HOURLY_VARS = [
    "temperature_2m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
]
