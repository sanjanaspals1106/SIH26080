"""Matplotlib visualization wrappers for PRD §16.9 standard verification displays.

Thin wrappers around prepared verification data:
- No seaborn dependencies.
- No GUI / frontend code.
- Does not mutate input data.
- Returns matplotlib.figure.Figure instances.
"""

from typing import Optional, Sequence, Union
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from verification.plots.data import (
    B0ToB3Row,
    CellImprovementPoint,
    FSSNeighbourhoodPoint,
    PerformanceDiagramPoint,
    RawRainBiasPoint,
    ReliabilityBinPoint,
    SeasonImprovementDot,
    prepare_average_improvement_map_data,
    prepare_b0_to_b3_table_data,
    prepare_fss_neighbourhood_data,
    prepare_per_season_improvement_data,
    prepare_performance_diagram_data,
    prepare_raw_rain_bias_data,
    prepare_reliability_diagram_data,
)


def plot_performance_diagram(
    data: Union[PerformanceDiagramPoint, Sequence[PerformanceDiagramPoint]],
    ax: Optional[plt.Axes] = None,
    show_isolines: bool = True,
) -> plt.Figure:
    """Plots a performance diagram (POD vs Success Ratio) per PRD §16.9."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    # Optional CSI and frequency-bias background isolines
    if show_isolines:
        sr_grid = np.linspace(0.01, 1.0, 100)
        pod_grid = np.linspace(0.01, 1.0, 100)
        SR, POD = np.meshgrid(sr_grid, pod_grid)
        CSI = 1.0 / (1.0 / SR + 1.0 / POD - 1.0)
        csi_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        cs = ax.contour(SR, POD, CSI, levels=csi_levels, colors="lightgray", linestyles="dashed", linewidths=0.8)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

        # Frequency bias lines: bias = POD / SR
        for bias_val in [0.5, 0.8, 1.0, 1.25, 2.0]:
            x_vals = np.linspace(0.01, 1.0, 50)
            y_vals = bias_val * x_vals
            valid = y_vals <= 1.0
            if np.any(valid):
                ax.plot(x_vals[valid], y_vals[valid], color="gainsboro", linestyle="dotted", linewidth=0.8)

    points = [data] if isinstance(data, PerformanceDiagramPoint) else list(data)

    for pt in points:
        if pt.success_ratio is not None and pt.pod is not None:
            lbl = pt.label or "Model"
            ax.scatter(pt.success_ratio, pt.pod, s=60, label=lbl)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Success Ratio (1 - FAR)")
    ax.set_ylabel("Probability of Detection (POD)")
    ax.set_title("Performance Diagram")
    if points and any(pt.label for pt in points):
        ax.legend(loc="lower right")

    return fig


def plot_fss_vs_neighbourhood(
    data: Sequence[FSSNeighbourhoodPoint],
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plots Fractions Skill Score against neighbourhood size (PRD §16.8, §16.9)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    pts = sorted(data, key=lambda p: p.neighbourhood_cells)
    kms = [p.approximate_km for p in pts if p.fss is not None]
    fss_vals = [p.fss for p in pts if p.fss is not None]

    if kms and fss_vals:
        ax.plot(kms, fss_vals, marker="o", linewidth=2, label="FSS")

    # Useful skill reference line (0.5 + f0 / 2)
    useful_vals = [p.useful_skill_line for p in pts if p.useful_skill_line is not None]
    if useful_vals:
        u_val = useful_vals[0]
        ax.axhline(u_val, linestyle="--", color="gray", label=f"Useful Skill ({u_val:.2f})")

    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Neighbourhood Scale (km)")
    ax.set_ylabel("Fractions Skill Score (FSS)")
    ax.set_title("FSS vs. Neighbourhood Size")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")

    return fig


def plot_reliability_diagram(
    data: Sequence[ReliabilityBinPoint],
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plots reliability diagram with 1:1 diagonal reference line (PRD §16.9)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    # 1:1 perfect reliability diagonal
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", label="Perfect Reliability")

    valid_pts = [
        p for p in data
        if p.mean_predicted_probability is not None and p.observed_frequency is not None
    ]

    if valid_pts:
        x_vals = [p.mean_predicted_probability for p in valid_pts]
        y_vals = [p.observed_frequency for p in valid_pts]
        ax.plot(x_vals, y_vals, marker="s", linewidth=1.5, label="Model Reliability")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Relative Frequency")
    ax.set_title("Reliability Diagram")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")

    return fig


def plot_raw_rain_bias(
    data: Sequence[RawRainBiasPoint],
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plots forecast bias by raw-rain size categories (PRD §16.9)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    labels = [p.bin_name for p in data]
    biases = [p.bias if p.bias is not None else 0.0 for p in data]
    indices = np.arange(len(labels))

    ax.bar(indices, biases, width=0.5, align="center")
    ax.axhline(0.0, color="black", linestyle="-", linewidth=0.8)

    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_xlabel("Raw Rain Category")
    ax.set_ylabel("Bias (mm)")
    ax.set_title("Forecast Bias by Raw Rain Size")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    return fig


def plot_average_improvement_map(
    data: Sequence[CellImprovementPoint],
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plots spatial average improvement map (PRD §15 F1, §16.9)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    has_coords = all(p.latitude is not None and p.longitude is not None for p in data)

    if has_coords and len(data) > 0:
        lons = [p.longitude for p in data]
        lats = [p.latitude for p in data]
        diffs = [p.mean_improvement_mm if p.mean_improvement_mm is not None else 0.0 for p in data]
        sc = ax.scatter(lons, lats, c=diffs, cmap="coolwarm", s=30)
        fig.colorbar(sc, ax=ax, label="Mean Improvement (mm)")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
    else:
        # Fallback to bar/line representation across cell IDs
        c_ids = [str(p.cell_id) for p in data]
        diffs = [p.mean_improvement_mm if p.mean_improvement_mm is not None else 0.0 for p in data]
        ax.plot(range(len(c_ids)), diffs, marker=".", linestyle="none")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Grid Cell Index")
        ax.set_ylabel("Mean Improvement (mm)")

    ax.set_title("Average Improvement per Cell (|raw-obs| - |corrected-obs|)")
    return fig


def plot_per_season_improvement_dots(
    data: Sequence[SeasonImprovementDot],
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plots per-season metric difference dots across seasons (PRD §16.6, §16.9)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.get_figure()

    seasons = [str(d.season) for d in data]
    diffs = [d.metric_difference if d.metric_difference is not None else 0.0 for d in data]
    indices = np.arange(len(seasons))

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.plot(indices, diffs, marker="o", linestyle="none", markersize=8)

    ax.set_xticks(indices)
    ax.set_xticklabels(seasons, rotation=30)
    cmp_label = data[0].system_comparison if data else "Comparison"
    met_name = data[0].metric_name if data else "Metric"
    ax.set_xlabel("Season")
    ax.set_ylabel(f"{met_name} Difference ({cmp_label})")
    ax.set_title(f"Per-Season Improvement Dots ({cmp_label})")
    ax.grid(True, linestyle=":", alpha=0.5)

    return fig


def plot_b0_to_b3_comparison(
    data: Sequence[B0ToB3Row],
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plots progression of metric across B0 -> B1 -> B2 -> B3 systems (PRD §10.9, §16.9)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    labels = [f"{r.system_id}\n({r.forecast_type})" for r in data]
    vals = [r.metric_value for r in data]
    indices = np.arange(len(labels))

    valid_indices = [i for i, v in enumerate(vals) if v is not None]
    valid_vals = [vals[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]

    # Calculate symmetric or asymmetric error bars if CIs present
    yerr_low = []
    yerr_high = []
    has_ci = False
    for i in valid_indices:
        r = data[i]
        if r.ci_low is not None and r.ci_high is not None and r.metric_value is not None:
            yerr_low.append(r.metric_value - r.ci_low)
            yerr_high.append(r.ci_high - r.metric_value)
            has_ci = True
        else:
            yerr_low.append(0.0)
            yerr_high.append(0.0)

    if has_ci:
        ax.errorbar(
            valid_indices,
            valid_vals,
            yerr=[yerr_low, yerr_high],
            fmt="o",
            capsize=5,
            linewidth=1.5,
            markersize=7,
        )
    else:
        ax.plot(valid_indices, valid_vals, marker="o", linewidth=1.5, markersize=7)

    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=10)
    m_name = data[0].metric_name.upper() if data else "METRIC"
    ax.set_ylabel(m_name)
    ax.set_title(f"B0 → B3 Model Progression ({m_name})")
    ax.grid(True, linestyle=":", alpha=0.6)

    return fig
