"""Low-Pressure System (LPS) rule-based detector.

PRD Section 11.4 & Appendix B:
Domain: 5-30°N, 60-100°E
1. Relative vorticity zeta at 850 hPa smoothed with Gaussian (sigma = 1.5°).
2. Candidate centres are local maxima of smoothed zeta > zeta_min (1.5e-5 s^-1).
3. MSLP drop at centre >= dp_min (2 hPa) compared to average MSLP in 500-800 km ring.
4. Merge candidates within 500 km, keeping the stronger one (higher zeta_max).
5. Output centres with strength_percentile computed against training distribution.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from regime_engine.lps.geo_utils import haversine_distance_km


class LPSDetector:
    """Rule-based detector for monsoon low-pressure systems (depressions & lows)."""

    def __init__(
        self,
        zeta_min: float = 1.5e-5,
        dp_min_hpa: float = 2.0,
        sigma_deg: float = 1.5,
        ring_km: Tuple[float, float] = (500.0, 800.0),
        merge_km: float = 500.0,
        domain: Tuple[float, float, float, float] = (5.0, 30.0, 60.0, 100.0),
        settings_label: str = "untuned",
    ):
        self.zeta_min = zeta_min
        self.dp_min_hpa = dp_min_hpa
        self.sigma_deg = sigma_deg
        self.ring_km = ring_km
        self.merge_km = merge_km
        self.lat_min, self.lat_max, self.lon_min, self.lon_max = domain
        self.settings_label = settings_label
        self.training_zeta_maxima: np.ndarray = np.array([])

    def fit_strength_percentiles(self, training_zeta_values: List[float]) -> "LPSDetector":
        """Store sorted list of training zeta maxima to compute strength percentiles."""
        if len(training_zeta_values) > 0:
            self.training_zeta_maxima = np.sort(np.array(training_zeta_values, dtype=float))
        return self

    def compute_strength_percentile(self, zeta_val: float) -> float:
        """Compute the empirical percentile rank of zeta_val against training maxima [0.0, 1.0]."""
        if len(self.training_zeta_maxima) == 0:
            # Fallback scaling if training maxima not yet populated: 1.5e-5 is 0.1, 5e-5 is 1.0
            return float(np.clip((zeta_val - 1.0e-5) / 4.0e-5, 0.05, 1.0))

        # Searchsorted to find rank
        rank = np.searchsorted(self.training_zeta_maxima, zeta_val, side="right")
        pct = float(rank / len(self.training_zeta_maxima))
        return float(np.clip(pct, 0.0, 1.0))

    def detect(
        self,
        vort850: np.ndarray,
        msl: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Run detection on window-average fields.

        Args:
            vort850: 2D array of 850 hPa relative vorticity (s^-1) on (lats, lons).
            msl: 2D array of MSLP (hPa or Pa). If values > 2000, converted to hPa.
            lats: 1D array of latitudes (sorted).
            lons: 1D array of longitudes (sorted).

        Returns:
            List of detected centers: [{'lat', 'lon', 'zeta_max', 'mslp_min_hpa', 'strength_percentile'}]
        """
        # Crop to detection domain (5-30°N, 60-100°E)
        lat_mask = (lats >= self.lat_min) & (lats <= self.lat_max)
        lon_mask = (lons >= self.lon_min) & (lons <= self.lon_max)

        dom_lats = lats[lat_mask]
        dom_lons = lons[lon_mask]

        if len(dom_lats) < 3 or len(dom_lons) < 3:
            return []

        sub_vort = vort850[np.ix_(lat_mask, lon_mask)].copy()
        sub_msl = msl[np.ix_(lat_mask, lon_mask)].copy()
        if np.nanmean(sub_msl) > 2000.0:
            sub_msl = sub_msl / 100.0

        # Estimate grid resolution (in degrees)
        dlat = abs(float(np.mean(np.diff(dom_lats))))
        dlon = abs(float(np.mean(np.diff(dom_lons))))
        grid_res = max(dlat, dlon, 0.01)
        sigma_pixels = self.sigma_deg / grid_res

        # 1. Smooth vorticity with Gaussian filter
        smoothed_vort = gaussian_filter(sub_vort, sigma=sigma_pixels, mode="nearest")

        # 2. Find local maxima of smoothed vorticity above zeta_min
        local_max = maximum_filter(smoothed_vort, size=3, mode="nearest")
        candidates_mask = (smoothed_vort == local_max) & (smoothed_vort >= self.zeta_min)

        cand_indices = np.argwhere(candidates_mask)
        if len(cand_indices) == 0:
            return []

        # Meshgrid of domain coordinates for distance calculation
        lon_grid, lat_grid = np.meshgrid(dom_lons, dom_lats)

        candidates = []
        r_inner, r_outer = self.ring_km

        for (i_lat, i_lon) in cand_indices:
            c_lat = float(dom_lats[i_lat])
            c_lon = float(dom_lons[i_lon])
            c_zeta = float(smoothed_vort[i_lat, i_lon])
            c_msl = float(sub_msl[i_lat, i_lon])

            # 3. Check MSLP depression in 500-800 km ring
            dist_grid = haversine_distance_km(c_lat, c_lon, lat_grid, lon_grid)
            ring_mask = (dist_grid >= r_inner) & (dist_grid <= r_outer)

            ring_pressures = sub_msl[ring_mask]
            valid_ring = ring_pressures[~np.isnan(ring_pressures)]

            if len(valid_ring) > 0:
                mean_ring_msl = float(np.mean(valid_ring))
                dp = mean_ring_msl - c_msl
                # Must be at least dp_min below ring average
                if dp >= self.dp_min_hpa:
                    candidates.append({
                        "lat": c_lat,
                        "lon": c_lon,
                        "zeta_max": c_zeta,
                        "mslp_min_hpa": c_msl,
                        "dp": dp,
                    })

        if not candidates:
            return []

        # 4. Merge candidates within merge_km (500 km), keeping the one with higher zeta_max
        # Sort candidates descending by zeta_max
        candidates.sort(key=lambda x: x["zeta_max"], reverse=True)
        merged_centers = []

        for cand in candidates:
            too_close = False
            for kept in merged_centers:
                dist = haversine_distance_km(cand["lat"], cand["lon"], kept["lat"], kept["lon"])
                if dist < self.merge_km:
                    too_close = True
                    break
            if not too_close:
                cand["strength_percentile"] = self.compute_strength_percentile(cand["zeta_max"])
                merged_centers.append(cand)

        return merged_centers
