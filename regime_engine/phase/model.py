"""Multinomial logistic regression model for Layer A monsoon phase.

PRD Section 11.3 & Appendix B:
- Model: Multinomial logistic regression with L2 penalty.
- Inputs standardised with training-season mean and spread.
- Regularization C chosen from {0.01, 0.1, 1, 10} by lowest mean LOSO log-loss.
- Target classes: 'active', 'normal', 'break'.
- Outputs: p_active, p_normal, p_break (summing to 1.0).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

CLASSES = ["active", "normal", "break"]


class PhaseModel:
    """Multinomial logistic regression phase model with L2 regularization and scaling."""

    def __init__(self, C: float = 1.0):
        self.C = C
        self.scaler = StandardScaler()
        # multi_class='multinomial' with l2 penalty
        self.clf = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        self.classes_ = CLASSES
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PhaseModel":
        """Fit scaler and logistic regression classifier.

        Args:
            X: Feature matrix of shape (n_samples, 20).
            y: Target array of string labels ('active', 'normal', 'break') or integers.
        """
        # Ensure y has categorical labels mapped to standard classes
        y_arr = np.asarray(y)
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y_arr)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities [p_active, p_normal, p_break].

        Guarantees returned order is always ['active', 'normal', 'break'] and sums to 1.

        Args:
            X: Feature matrix of shape (n_samples, 20).

        Returns:
            np.ndarray of shape (n_samples, 3) with probabilities summing to 1.
        """
        if not self.is_fitted:
            raise RuntimeError("PhaseModel must be fitted before calling predict_proba.")

        X_scaled = self.scaler.transform(X)
        raw_probs = self.clf.predict_proba(X_scaled)

        # Map clf.classes_ to standard ['active', 'normal', 'break'] order
        clf_classes = list(self.clf.classes_)
        n_samples = len(X)
        standard_probs = np.zeros((n_samples, 3), dtype=float)

        for i, cls_name in enumerate(CLASSES):
            if cls_name in clf_classes:
                idx = clf_classes.index(cls_name)
                standard_probs[:, i] = raw_probs[:, idx]
            else:
                standard_probs[:, i] = 0.0

        # Ensure probabilities strictly sum to 1.0
        row_sums = standard_probs.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        standard_probs = standard_probs / row_sums

        return standard_probs

    def predict(self, X: np.ndarray) -> List[str]:
        """Predict most likely phase label for each sample."""
        probs = self.predict_proba(X)
        best_indices = np.argmax(probs, axis=1)
        return [CLASSES[i] for i in best_indices]


def select_best_c_loso(
    X: np.ndarray,
    y: np.ndarray,
    seasons: np.ndarray,
    c_grid: List[float] = [0.01, 0.1, 1.0, 10.0],
) -> float:
    """Select the best regularization parameter C using Leave-One-Season-Out (LOSO) log-loss.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Labels ('active', 'normal', 'break').
        seasons: Array of season years corresponding to each row.
        c_grid: Candidates for C (default [0.01, 0.1, 1, 10]).

    Returns:
        Best C value minimizing mean LOSO log-loss.
    """
    unique_seasons = np.unique(seasons)
    if len(unique_seasons) < 2:
        return 1.0  # Cannot perform LOSO with < 2 seasons

    best_c = 1.0
    best_loss = float("inf")

    for c_val in c_grid:
        fold_losses = []
        for held_out in unique_seasons:
            train_mask = seasons != held_out
            val_mask = seasons == held_out

            X_tr, y_tr = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            # Check if all classes are present in training fold
            tr_classes = np.unique(y_tr)
            if len(tr_classes) < 2:
                continue

            model = PhaseModel(C=c_val)
            try:
                model.fit(X_tr, y_tr)
                probs = model.predict_proba(X_val)
                # Compute log loss over classes
                # sklearn sorts `labels` alphabetically, so put the probability columns in that order too
                order = np.argsort(CLASSES)
                loss = log_loss(y_val, probs[:, order], labels=[CLASSES[i] for i in order])
                fold_losses.append(loss)
            except Exception:
                continue

        if len(fold_losses) > 0:
            mean_loss = float(np.mean(fold_losses))
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_c = c_val

    return best_c
