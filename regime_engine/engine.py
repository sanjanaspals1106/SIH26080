"""RegimeEngine: Top-level orchestrator coordinating Layers A, B, and C.

PRD Section 11 & Appendix B:
Coordinates:
- Layer A: Monsoon phase classification & confidence
- Layer B: Low-Pressure System detection & influence
- Layer C: Coast & Mountain local context indices
Produces:
- Domain-level contract (§11.6)
- Cell-level contract (§11.6)
- 14 regime features for ML model B3 (§11.7)
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from regime_engine.phase import (
    PhaseModel,
    PhaseOODDetector,
    build_phase_feature_vector,
    compute_regime_confidence,
)
from regime_engine.lps import (
    LPSDetector,
    compute_lps_cell_features,
)
from regime_engine.local_context import (
    compute_raw_fluxes,
    InfluencePercentileTable,
)
from regime_engine.contract import (
    REGIME_14_FEATURES,
    build_domain_regime_contract,
    build_cell_regime_contract,
    assemble_14_regime_features,
    validate_training_regime_source,
)


class RegimeEngine:
    """Integrated engine coordinating monsoon weather regime inference."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Path] = None,
    ):
        if config is not None:
            self.cfg = config
        elif config_path is not None and Path(config_path).exists():
            self.cfg = yaml.safe_load(Path(config_path).read_text())
        else:
            self.cfg = {}

        # Layer A components
        c_val = self.cfg.get("layer_a", {}).get("phase_model", {}).get("penalty_C", 1.0) or 1.0
        self.phase_model = PhaseModel(C=c_val)
        self.ood_detector = PhaseOODDetector()

        # Layer B components
        det_cfg = self.cfg.get("layer_b", {}).get("detector", {})
        self.lps_detector = LPSDetector(
            zeta_min=det_cfg.get("zeta_min", 1.5e-5),
            dp_min_hpa=det_cfg.get("dp_min_hpa", 2.0),
            sigma_deg=det_cfg.get("sigma_deg", 1.5),
            settings_label=self.cfg.get("layer_b", {}).get("settings_label", "untuned"),
        )
        self.no_lps_dist = self.cfg.get("layer_b", {}).get("no_lps_distance_km", 3000.0)
        self.lps_inf_scale = self.cfg.get("layer_b", {}).get("lps_influence_scale_km", 600.0)

        # Layer C components
        self.influence_table = InfluencePercentileTable()
        self.coast_decay_scale = self.cfg.get("layer_c", {}).get("coast_decay_scale_km", 100.0)

    def process_run_lead(
        self,
        run_id: str,
        lead_day: int,
        doy: int,
        lead_a_features: Dict[int, Dict[str, float]],
        grid_cells_df: pd.DataFrame,
        vort850: np.ndarray,
        msl: np.ndarray,
        u850_cell: np.ndarray,
        v850_cell: np.ndarray,
        q850_cell: np.ndarray,
        lats_grid: np.ndarray,
        lons_grid: np.ndarray,
        regime_source: str = "final",
    ) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
        """Process one forecast run at a given lead day.

        Args:
            run_id: Forecast run identifier (e.g. tigge_ecmwf_cf_2024071500).
            lead_day: Lead day (1, 2, or 3).
            doy: Day of year (1..366).
            lead_a_features: Dict of {lead_day: {'A1'..'A6': float}} for leads 1, 2, 3.
            grid_cells_df: Cell coordinates and static geography:
                ['cell_id', 'latitude', 'longitude', 'grad_h_x', 'grad_h_y',
                 'coast_normal_x', 'coast_normal_y', 'dist_coast_km']
            vort850: 2D vorticity grid (lat, lon) for LPS detection.
            msl: 2D MSLP grid (lat, lon) for LPS detection.
            u850_cell: 1D array of u850 values at cell locations.
            v850_cell: 1D array of v850 values at cell locations.
            q850_cell: 1D array of q850 values at cell locations.
            lats_grid: 1D latitudes of grid.
            lons_grid: 1D longitudes of grid.
            regime_source: 'oof' for training data, 'final' for holdout/inference.

        Returns:
            (domain_regime_dict, cell_regime_df, b3_features_df)
        """
        # 1. Layer A: Phase prediction
        feat_vec, feat_names = build_phase_feature_vector(lead_a_features, target_lead=lead_day, doy=doy)
        if self.phase_model.is_fitted:
            probs = self.phase_model.predict_proba(feat_vec.reshape(1, -1))[0]
            p_act, p_norm, p_brk = float(probs[0]), float(probs[1]), float(probs[2])
        else:
            # Climatological uniform baseline if model not fitted yet
            p_act, p_norm, p_brk = 0.20, 0.60, 0.20

        conf_val, conf_band = compute_regime_confidence(np.array([p_act, p_norm, p_brk]))

        curr_a = lead_a_features.get(lead_day, lead_a_features.get(1, {}))
        is_ood = self.ood_detector.predict(curr_a)

        # 2. Layer B: LPS detection
        centers = self.lps_detector.detect(vort850, msl, lats_grid, lons_grid)
        lps_cell_df = compute_lps_cell_features(
            grid_cells_df,
            centers,
            no_lps_distance_km=self.no_lps_dist,
            lps_influence_scale_km=self.lps_inf_scale,
        )

        # 3. Layer C: Local context (coast & mountains)
        grad_hx = grid_cells_df.get("grad_h_x", np.zeros(len(grid_cells_df))).to_numpy(dtype=float)
        grad_hy = grid_cells_df.get("grad_h_y", np.zeros(len(grid_cells_df))).to_numpy(dtype=float)
        norm_x = grid_cells_df.get("coast_normal_x", np.zeros(len(grid_cells_df))).to_numpy(dtype=float)
        norm_y = grid_cells_df.get("coast_normal_y", np.zeros(len(grid_cells_df))).to_numpy(dtype=float)
        dist_c = grid_cells_df.get("dist_coast_km", np.full(len(grid_cells_df), 500.0)).to_numpy(dtype=float)

        upslope_flux, onshore_flux = compute_raw_fluxes(
            u850=u850_cell,
            v850=v850_cell,
            q850=q850_cell,
            grad_h_x=grad_hx,
            grad_h_y=grad_hy,
            coast_normal_x=norm_x,
            coast_normal_y=norm_y,
            dist_coast_km=dist_c,
            coast_decay_scale_km=self.coast_decay_scale,
        )

        orog_inf, coast_inf, orog_fav, coast_fav = self.influence_table.compute_influence(
            upslope_flux, onshore_flux
        )

        # Assemble cell results dataframe
        cell_res = lps_cell_df.copy()
        cell_res["upslope_flux"] = upslope_flux
        cell_res["onshore_flux"] = onshore_flux
        cell_res["orographic_influence"] = orog_inf
        cell_res["coastal_influence"] = coast_inf
        cell_res["orographic_favorable"] = orog_fav
        cell_res["coastal_favorable"] = coast_fav

        # Domain level contract
        domain_contract = build_domain_regime_contract(
            run_id=run_id,
            lead_day=lead_day,
            p_active=p_act,
            p_normal=p_norm,
            p_break=p_brk,
            regime_confidence=conf_val,
            confidence_band=conf_band,
            regime_source=regime_source,
            lps_detected=len(centers) > 0,
            lps_centres=centers,
            lps_settings=self.lps_detector.settings_label,
            ood_flag=is_ood,
            regime_available=True,
        )

        # 14 features for B3
        b3_features_df = assemble_14_regime_features(domain_contract, cell_res)

        return domain_contract, cell_res, b3_features_df
