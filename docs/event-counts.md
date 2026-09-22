# Observed Rainfall Event Counts and Model Availability

**Status:** Pending Gate G1 data download (ECDS & IMD).

> [!NOTE]
> Real gridded observation data is not yet present in `data/golden/`.
> In accordance with PRD honesty rules, synthetic or invented counts are forbidden.
> This document will be populated during Gate G2 when real seasons are ingested.

## PRD §13.2 Decision Rules for Model Availability

| Threshold | Rule | Built Model |
|---|---|---|
| ≥ 64.5 mm | ≥ 30 events in development seasons | Standalone binary classifier |
| ≥ 115.6 mm | < 30 events in dev, but 64.5 mm has ≥ 30 | Chained form: P(≥115.6) = P(≥64.5) × P(≥115.6 given ≥64.5) |
| ≥ 64.5 mm | < 30 events in development seasons | No heavy-rain model. Only P(≥15.6) shown. Hotspots use 15.6 mm |

*(Updated automatically once observation grids are available).* 