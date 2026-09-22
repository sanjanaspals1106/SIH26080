"""Deterministic Synthetic Fixture for M3 Rainfall Model Tests (PRD Rule H7).

WARNING:
    TEST/MOCK DATA ONLY.
    Generated strictly for unit testing and contract verification of pipeline code.
    NEVER use for skill claims or reporting (PRD Rule H7).
"""

from typing import List
import numpy as np
import pandas as pd

# All 6 regions from PRD §14.3
ALL_REGIONS = [
    "HIMALAYA",
    "NORTHEAST",
    "WEST_COAST",
    "NORTHWEST_WEST",
    "SOUTH_EAST",
    "CENTRAL_EAST",
]

DEFAULT_SEASONS = [2018, 2019, 2020]
DEFAULT_LEADS = [1, 2, 3]


def generate_synthetic_m3_data(
    seasons: List[int] = DEFAULT_SEASONS,
    leads: List[int] = DEFAULT_LEADS,
    regions: List[str] = ALL_REGIONS,
    n_cells_per_region: int = 5,
    dates_per_season: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a small deterministic synthetic dataset for M3 testing.

    Includes:
    - season
    - run_id
    - lead_day (1, 2, 3)
    - cell_id
    - region_code (all PRD §14.3 region codes)
    - rain_mm (forecast rainfall >= 0 with realistic zeros and tails)
    - obs_mm (observed rainfall >= 0 with realistic zeros and tails)
    """
    rng = np.random.RandomState(seed)
    records = []

    cell_counter = 1000
    region_cells = {}
    for region in regions:
        region_cells[region] = [cell_counter + i for i in range(n_cells_per_region)]
        cell_counter += 100

    for season in seasons:
        for day in range(1, dates_per_season + 1):
            date_str = f"{season}07{day:02d}"
            run_id = f"tigge_ecmwf_cf_{date_str}00"

            for lead in leads:
                for region in regions:
                    for cell_id in region_cells[region]:
                        # Monsoon rainfall has zero-inflation (~60% dry) + log-normal tail
                        is_rain_forecast = rng.rand() > 0.60
                        rain_mm = (
                            float(np.round(rng.exponential(scale=15.0), 2))
                            if is_rain_forecast
                            else 0.0
                        )

                        is_rain_obs = rng.rand() > 0.60
                        obs_mm = (
                            float(np.round(rng.exponential(scale=18.0), 2))
                            if is_rain_obs
                            else 0.0
                        )

                        rec = {
                            "season": season,
                            "run_id": run_id,
                            "lead_day": lead,
                            "cell_id": cell_id,
                            "region_code": region,
                            # Rain group
                            "rain_mm": rain_mm,
                            "nbr_mean_3": float(np.round(rain_mm * 0.95, 2)),
                            "nbr_max_3": float(np.round(rain_mm * 1.10, 2)),
                            "nbr_mean_5": float(np.round(rain_mm * 0.90, 2)),
                            "nbr_max_5": float(np.round(rain_mm * 1.20, 2)),
                            "rain_grad": float(np.round(rng.uniform(0, 5), 2)),
                            # NaNs allowed per PRD §9.4
                            "rain_prev_lead": np.nan if lead == 1 else float(np.round(rain_mm * 0.9, 2)),
                            "rain_next_lead": np.nan if lead == 3 else float(np.round(rain_mm * 1.1, 2)),
                            # Circulation and moisture group
                            "u850": float(np.round(rng.normal(5, 3), 2)),
                            "v850": float(np.round(rng.normal(2, 3), 2)),
                            "wspd850": float(np.round(rng.uniform(1, 15), 2)),
                            "vort850": float(np.round(rng.normal(1e-5, 5e-6), 6)),
                            "q850": float(np.round(rng.uniform(10, 20), 2)),
                            "msl": float(np.round(rng.uniform(995, 1010), 2)),
                            "shear_200_850": float(np.round(rng.uniform(10, 35), 2)),
                            # Geography group
                            "elevation_m": float(cell_id % 500),
                            "slope": float(np.round(rng.uniform(0, 10), 2)),
                            "aspect_sin": float(np.round(rng.uniform(-1, 1), 3)),
                            "aspect_cos": float(np.round(rng.uniform(-1, 1), 3)),
                            "dist_coast_km": float(np.round(rng.uniform(10, 800), 1)),
                            # Climatology group
                            "clim_mean": float(np.round(rng.uniform(2, 25), 2)),
                            "clim_p95": float(np.round(rng.uniform(20, 120), 2)),
                            # Time and place group
                            "doy_sin": float(np.round(np.sin(2 * np.pi * (180 + day) / 365.25), 4)),
                            "doy_cos": float(np.round(np.cos(2 * np.pi * (180 + day) / 365.25), 4)),
                            "latitude": float(np.round(8.0 + (cell_id % 30) * 0.8, 3)),
                            "longitude": float(np.round(68.0 + (cell_id % 25) * 0.9, 3)),
                            # 14 Regime Features (PRD §11.7)
                            "p_active": 0.35,
                            "p_normal": 0.45,
                            "p_break": 0.20,
                            "regime_confidence": 0.40,
                            "lps_present": 1.0 if (day % 3 == 0) else 0.0,
                            "distance_to_lps_km": float(np.round(rng.uniform(150, 1200), 1)),
                            "bearing_sin": float(np.round(rng.uniform(-1, 1), 3)),
                            "bearing_cos": float(np.round(rng.uniform(-1, 1), 3)),
                            "lps_strength": float(np.round(rng.uniform(0.1, 0.9), 2)),
                            "lps_influence": float(np.round(rng.uniform(0.05, 0.8), 2)),
                            "upslope_flux": float(np.round(rng.uniform(0.0, 50.0), 2)),
                            "onshore_flux": float(np.round(rng.uniform(0.0, 40.0), 2)),
                            "orographic_influence": float(np.round(rng.uniform(0.1, 0.9), 2)),
                            "coastal_influence": float(np.round(rng.uniform(0.1, 0.9), 2)),
                            # Regime source metadata (PRD §11.6: 'oof' for development/training)
                            "regime_source": "oof",
                            # Target
                            "obs_mm": obs_mm,
                        }
                        records.append(rec)

    df = pd.DataFrame(records)
    # Add mock identifier attribute
    df.attrs["_mock"] = True
    df.attrs["description"] = "SYNTHETIC FIXTURE FOR M3 UNIT TESTS ONLY (PRD H7)"
    return df
