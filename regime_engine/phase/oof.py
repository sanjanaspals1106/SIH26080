"""Out-of-fold (OOF) regime probability generation.

PRD Section 10.3 & 11.6:
The correction model must be trained on regime probabilities that look like the ones
it will see in use: made by a model that never saw that season.
For training rows, regime_source must be 'oof'.
For holdout / production inference, regime_source is 'final'.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from regime_engine.phase.model import PhaseModel, select_best_c_loso, CLASSES
from regime_engine.phase.confidence import compute_regime_confidence, PhaseOODDetector


def generate_oof_phase_predictions(
    features_df: pd.DataFrame,
    labels_series: pd.Series,
    development_seasons: List[int],
    feature_cols: List[str],
    c_grid: List[float] = [0.01, 0.1, 1.0, 10.0],
) -> pd.DataFrame:
    """Generate strictly out-of-fold phase probabilities for all development seasons.

    Follows PRD 10.3:
    For each held-out season h in development_seasons:
        Train on (development_seasons except h)
        Predict season h.
    The resulting probabilities have regime_source = 'oof'.

    Args:
        features_df: DataFrame containing ['season', 'imd_date', 'lead_day'] + feature_cols.
        labels_series: Target labels indexed by imd_date ('active', 'normal', 'break').
        development_seasons: List of development season years.
        feature_cols: 20 column names for the phase model.
        c_grid: Hyperparameter grid for C.

    Returns:
        DataFrame with ['run_id', 'lead_day', 'imd_date', 'p_active', 'p_normal', 'p_break',
                        'regime_confidence', 'confidence_band', 'regime_source']
    """
    dev_mask = features_df["season"].isin(development_seasons)
    df_dev = features_df.loc[dev_mask].copy()

    # Map labels to dates
    df_dev["label"] = df_dev["imd_date"].map(labels_series)
    df_dev = df_dev.dropna(subset=["label"])

    X_all = df_dev[feature_cols].to_numpy(dtype=float)
    y_all = df_dev["label"].to_numpy()
    seasons_all = df_dev["season"].to_numpy()

    # Select best C globally on development seasons using LOSO
    best_c = select_best_c_loso(X_all, y_all, seasons_all, c_grid=c_grid)

    oof_records = []

    for h in development_seasons:
        train_mask = seasons_all != h
        val_mask = seasons_all == h

        if not np.any(val_mask):
            continue

        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_val = X_all[val_mask]
        val_df = df_dev.loc[val_mask]

        model = PhaseModel(C=best_c)
        model.fit(X_tr, y_tr)

        # Fit OOD detector on training fold
        # A1-A6 curr features are features in X_tr
        ood_detector = PhaseOODDetector()
        a_feat_dict = {}
        for aid in ["A1", "A2", "A3", "A4", "A5", "A6"]:
            col = f"{aid}_curr"
            if col in feature_cols:
                col_idx = feature_cols.index(col)
                a_feat_dict[aid] = X_tr[:, col_idx]
        ood_detector.fit(a_feat_dict)

        probs_val = model.predict_proba(X_val)

        for i, (_, row) in enumerate(val_df.iterrows()):
            p_act, p_norm, p_brk = probs_val[i]
            conf, band = compute_regime_confidence(probs_val[i])

            curr_a = {
                aid: float(row[f"{aid}_curr"]) if f"{aid}_curr" in row else 0.0
                for aid in ["A1", "A2", "A3", "A4", "A5", "A6"]
            }
            is_ood = ood_detector.predict(curr_a)

            oof_records.append({
                "run_id": row.get("run_id", f"run_{row['season']}_{row['imd_date']}"),
                "lead_day": int(row["lead_day"]),
                "imd_date": row["imd_date"],
                "season": int(row["season"]),
                "p_active": float(p_act),
                "p_normal": float(p_norm),
                "p_break": float(p_brk),
                "regime_confidence": float(conf),
                "confidence_band": band,
                "ood_flag": bool(is_ood),
                "regime_source": "oof",  # PRD 11.6: oof for training rows
            })

    return pd.DataFrame(oof_records)


def fit_final_phase_model(
    features_df: pd.DataFrame,
    labels_series: pd.Series,
    development_seasons: List[int],
    feature_cols: List[str],
    c_grid: List[float] = [0.01, 0.1, 1.0, 10.0],
) -> Tuple[PhaseModel, PhaseOODDetector, float]:
    """Fit final phase model on all development seasons (PRD 10.3).

    Returns:
        (fitted_model, fitted_ood_detector, best_c)
    """
    dev_mask = features_df["season"].isin(development_seasons)
    df_dev = features_df.loc[dev_mask].copy()
    df_dev["label"] = df_dev["imd_date"].map(labels_series)
    df_dev = df_dev.dropna(subset=["label"])

    X_all = df_dev[feature_cols].to_numpy(dtype=float)
    y_all = df_dev["label"].to_numpy()
    seasons_all = df_dev["season"].to_numpy()

    best_c = select_best_c_loso(X_all, y_all, seasons_all, c_grid=c_grid)

    final_model = PhaseModel(C=best_c)
    final_model.fit(X_all, y_all)

    ood_detector = PhaseOODDetector()
    a_feat_dict = {}
    for aid in ["A1", "A2", "A3", "A4", "A5", "A6"]:
        col = f"{aid}_curr"
        if col in feature_cols:
            col_idx = feature_cols.index(col)
            a_feat_dict[aid] = X_all[:, col_idx]
    ood_detector.fit(a_feat_dict)

    return final_model, ood_detector, best_c
