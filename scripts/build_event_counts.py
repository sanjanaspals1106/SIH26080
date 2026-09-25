#!/usr/bin/env python
"""M4: builds docs/event-counts.md from the Golden Dataset (PRD §10.7, §13.2).

Counts observed-rainfall events per season/lead/threshold, decides which heavy-rain
probability models are available (PRD §13.2), and writes the markdown report. If no
Golden Dataset season is on disk yet, writes an honest "pending data" placeholder
instead of inventing numbers (PRD H7).

Run from the repository root: python scripts/build_event_counts.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from data_pipeline.alignment.cells import load_valid_cells  # noqa: E402
from data_pipeline.alignment.golden import read_golden  # noqa: E402
from data_pipeline.ingestion import load_config  # noqa: E402
from data_pipeline.ingestion.errors import MissingInputError  # noqa: E402
from data_pipeline.ingestion.imd import imd_axes  # noqa: E402
from protocol.splits import assign_season_splits  # noqa: E402
from verification.events import (  # noqa: E402
    DailyObservation,
    count_events_in_grid_series,
    determine_model_availability,
    generate_event_counts_markdown,
    load_rain_thresholds,
)

DOCS_PATH = Path(__file__).resolve().parent.parent / "docs" / "event-counts.md"

DECISION_RULES_SECTION = """
## PRD §13.2 Decision Rules for Model Availability

| Threshold | Rule | Built Model |
|---|---|---|
| >= 64.5 mm | >= 30 events in development seasons | Standalone binary classifier |
| >= 115.6 mm | < 30 events in dev, but 64.5 mm has >= 30 | Chained form: P(>=115.6) = P(>=64.5) x P(>=115.6 given >=64.5) |
| >= 64.5 mm | < 30 events in development seasons | No heavy-rain model. Only P(>=15.6) shown. Hotspots use 15.6 mm |
"""


def _pending_placeholder() -> str:
    return (
        "# Observed Rainfall Event Counts and Model Availability\n\n"
        "**Status:** Pending Gate G1 data download (ECDS & IMD).\n\n"
        "> [!NOTE]\n"
        "> Real gridded observation data is not yet present in `data/golden/`.\n"
        "> In accordance with PRD honesty rules, synthetic or invented counts are forbidden.\n"
        "> This document will be populated during Gate G2 when real seasons are ingested.\n"
        + DECISION_RULES_SECTION
        + "\n*(Updated automatically once observation grids are available.)*\n"
    )


def build_event_counts(output_path: Path | str | None = None) -> Path:
    """Builds `docs/event-counts.md` from the Golden Dataset. Never invents numbers (PRD H7)."""
    output_path = Path(output_path) if output_path else DOCS_PATH
    config = load_config()

    try:
        golden = read_golden(config)
    except MissingInputError:
        golden = None

    if golden is None or golden.empty:
        content = _pending_placeholder()
    else:
        seasons = sorted(int(s) for s in golden["season"].unique().tolist())
        split = assign_season_splits(seasons)
        dev_seasons = split.development_seasons

        valid = load_valid_cells(config)
        lat, lon = imd_axes(config.imd)
        n_lat, n_lon = lat.size, lon.size
        valid_mask = np.zeros((n_lat, n_lon), dtype=bool)
        ivalid = valid.loc[valid["is_valid"], "cell_id"].to_numpy()
        valid_mask[ivalid // n_lon, ivalid % n_lon] = True

        thresholds = load_rain_thresholds()

        observations: list[DailyObservation] = []
        for (season, lead_day, imd_date_val), rows in golden.groupby(
            ["season", "lead_day", "imd_date"], sort=False
        ):
            grid = np.full((n_lat, n_lon), np.nan)
            cell_ids = rows["cell_id"].to_numpy()
            grid[cell_ids // n_lon, cell_ids % n_lon] = rows["obs_mm"].to_numpy()
            observations.append(
                DailyObservation(
                    season=int(season),
                    lead_day=int(lead_day),
                    date=imd_date_val,
                    observed_rain=grid,
                    valid_mask=valid_mask,
                )
            )

        records = count_events_in_grid_series(observations, thresholds=thresholds)

        dev_totals = {
            th: sum(r.n_events for r in records if r.season in dev_seasons and r.threshold_mm == th)
            for th in thresholds
        }
        decision = determine_model_availability(dev_totals, min_events=30)
        content = generate_event_counts_markdown(records, dev_seasons, decision=decision)
        content += DECISION_RULES_SECTION

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=None, help="output path (default docs/event-counts.md)")
    args = parser.parse_args()
    path = build_event_counts(output_path=args.output)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
