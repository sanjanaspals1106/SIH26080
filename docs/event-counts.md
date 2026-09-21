# Event counts

Owner: **M4**. PRD section 13.2. Written **before choosing the probability models**, from
**development seasons only**.

An **event** is an 8-connected group of cells at or above the threshold on one date. Groups on
consecutive dates that share at least one cell are merged into one event (PRD 10.7).
`n_cell_days` is the number of (cell, date) pairs at or above the threshold.

## Counts per season and lead

| Season | Lead day | Threshold (mm) | n_events | n_cell_days |
|---|---|---|---|---|
| | 1 | 15.6 | | |
| | 1 | 64.5 | | |
| | 1 | 115.6 | | |
| | 2 | 15.6 | | |
| | 2 | 64.5 | | |
| | 2 | 115.6 | | |
| | 3 | 15.6 | | |
| | 3 | 64.5 | | |
| | 3 | 115.6 | | |

(Repeat the nine rows for every development season.)

## Totals over development seasons, and what is built

| Threshold (mm) | Total n_events | Total n_cell_days | What is built (PRD 13.2) |
|---|---|---|---|
| 15.6 | | | |
| 64.5 | | | |
| 115.6 | | | |

Rule from PRD 13.2:

| Events at the threshold in development seasons | What is built |
|---|---|
| At least 30 | A classifier of its own |
| 115.6 mm has fewer than 30, but 64.5 mm has at least 30 | Chained form: P(≥115.6) = P(≥64.5) × P(≥115.6 given ≥64.5) |
| 64.5 mm has fewer than 30 | No heavy-rain model. The screen says "Not enough events". Only P(≥15.6) is shown. Hotspots use 15.6 mm (F6). |
