# Product Requirements Document (PRD)
## SIH 2026 — Problem Statement 26080

### Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts

**Version:** 1.0
**Date:** 21 September 2026
**Build team:** 5 builders (1 of them builds the whole frontend)
**How it runs:** Locally on laptops. No Docker.

---

# 0. About This Document

## 0.1 Purpose

This document tells the team exactly what to build, with which data, with which methods, and how to check that it works. Every choice is written down as a **Decision**. Every fact that must still be checked is written down as a **Check**, with a rule for what to do with the result. No item is left as "maybe".

## 0.2 Tags

| Tag | Meaning |
|---|---|
| **[Verified]** | Confirmed from public sources on 21 Sep 2026. Sources are in Appendix C. |
| **[Decision]** | Fixed by this document. To change it, the whole team must agree. |
| **[Assumption]** | A starting value. It is stored in a config file. It may be tuned **only on development seasons** (see §10). |
| **[Check]** | Someone must check it. Section 3 says who, how, and what to do with the result. |

## 0.3 Words used in this document

- **Must** means required. **Must not** means forbidden. **May** means allowed but not required.
- **Season** means one June 1 – September 30 period of one year.
- **Cell** means one 0.25° square of the project grid (§8.5).
- A **glossary** of weather and statistics words is in §27.

---

# 1. Summary

## 1.1 The problem

Weather models (NWP) forecast rainfall on a grid. Their mistakes are not random. They depend on the weather situation: an active monsoon spell, a break spell, a low-pressure system, the coast, or the mountains. One correction for all situations works less well than a correction that knows the situation.

## 1.2 The product

The product is a **post-processing layer** on top of a raw NWP rainfall forecast. It does not replace the NWP model. It learns from past forecasts and past observed rainfall, then produces:

1. A **weather situation description** ("regime"): monsoon phase, low-pressure influence, coastal and mountain influence.
2. A **corrected rainfall forecast** for every cell.
3. **Heavy-rain probabilities** for every cell.
4. A **model-estimated range** for every cell.
5. **District products** built from the cell products.
6. **Six required features** (§15): raw/corrected toggle, audit trail, historical analog finder, regime transition detection, district priority table, and spatial hotspots.
7. A **verification report** that compares raw and corrected forecasts on seasons the team has never used for any decision.

## 1.3 The five outputs the problem statement asks for

| PS output | Where it is in this document |
|---|---|
| Weather regime classifier | §11 |
| Bias-corrected rainfall forecast | §12 |
| Heavy-rainfall probability | §13 |
| District-level rainfall product | §14 and §15 |
| Verification report (RMSE, ETS, CSI, POD, FAR, FSS) | §16 |

---

# 2. Fixed Decisions

Everyone builds to this table. If another section seems to disagree with it, this table wins and the other section is a mistake to report.

| ID | Decision |
|---|---|
| **D1** | **Area:** all of India (all valid IMD cells). The default map view is all of India. The **demo highlight region** is `WEST_COAST` (defined in §14.3). |
| **D2** | **Seasons:** complete JJAS seasons up to and including 2025. Target: 2016–2025 (10 seasons). Wanted minimum: 5 seasons. The real number **N is fixed at gate G1** (§24). If N is smaller than 5, the rules for small N in §10.2 apply. No season is added after G1. Download newest season first. A season is used only if **both** TIGGE and IMD data exist for it. |
| **D3** | **Forecast:** ECMWF **control forecast** from TIGGE (`type = cf`), start time **00 UTC** only, lead days **1, 2, 3**. |
| **D4** | **Truth (observed rain):** IMD gridded daily rainfall, 0.25°. |
| **D5** | **Project grid:** the IMD 0.25° grid (129 × 135 points). Only **valid cells** are used (§8.5). |
| **D6** | **Time alignment:** method **C1** (§8.3) if 6-hourly precipitation is downloaded for every chosen season. Otherwise method **C0** for every season. C0 and C1 are never mixed. |
| **D7** | **Districts:** DataMeet district boundaries based on the 2011 census (about 640 districts, the 2011 census count). Districts created after 2011 are not included. The UI and the documentation must say this. |
| **D8** | **Rain thresholds (per 24 h):** 15.6 mm, 64.5 mm, 115.6 mm. |
| **D9** | **Mode:** **Replay only.** The system shows stored past forecasts. There is no live mode. |
| **D10** | **Models:** XGBoost for rainfall correction, quantiles and heavy-rain probability. Logistic regression for monsoon phase. A rule-based detector for low-pressure systems. Simple rules for coastal and mountain influence. |
| **D11** | **Splits:** by season. Leave-one-season-out on development seasons. A locked holdout run, once (§10). |
| **D12** | **Six required features:** F1–F6 (§15). **Per-feature attribution methods (SHAP) are not part of this project.** Explanation is given only by the audit trail (F2). |
| **D13** | **Software:** Python 3.11; XGBoost 2.0 or newer; FastAPI; PostgreSQL 14 or newer (local install); React 18 + TypeScript + Vite; Leaflet through react-leaflet 4; pandas, NumPy, xarray; cfgrib and eccodes; scikit-learn. |
| **D14** | **Computers:** macOS or Linux. Windows users must use WSL2 (Ubuntu). CPU only. The whole project must run on a laptop with **8 GB RAM**. |
| **D15** | **No Docker. No PostGIS.** PostgreSQL is installed locally. District shapes are kept as GeoJSON and processed in Python. |
| **D16** | **Storage:** cell-level data is stored in **Parquet files**. PostgreSQL stores district-level data, metadata, results, events and lookups (§20). |
| **D17** | **Static geography** (height and land–sea mask) comes from the TIGGE fields `orog` and `lsm`. |
| **D18** | **Wording rules:** §5. |
| **D19** | **Approved fallback if something fails:** §17. |
| **D20** | **The finale is expected to be a 36-hour build.** All modelling and evaluation must be finished **before** it. The finale is for joining the parts, fixing, and demonstrating (§24). |

---

# 3. Open Checks

These are facts nobody could confirm from outside. Each has an owner, a way to check, a pass rule, and a fixed action if it fails.

| ID | Check | Owner | How | Pass rule | If it fails |
|---|---|---|---|---|---|
| **CK1** | Shape of the downloaded precipitation | M1 | Open one real file with xarray | `tp` has dimensions (step, latitude, longitude) | The sample file has one flat dimension called `values`. If the real file does too: (a) if the number of distinct latitudes × distinct longitudes equals the number of values, reshape it into a grid; (b) otherwise treat the values as scattered points and regrid them with the rule in §8.4. |
| **CK2** | What the ECDS download form allows | M1 | Open the TIGGE form on ECDS; use "Show API request code" | The form allows: type `cf`, an area, 6-hourly steps, and a grid spacing | Use what the form offers. If only daily steps (24, 48, 72 h) are possible, use method **C0** (D6). If no grid option exists, keep the native grid and regrid (§8.4). Write what was used in `docs/data-status.md`. |
| **CK3** | The ECMWF control forecast supplies the fields we need | M1 | Try a one-day request for each field in §7.2 | `u`, `v` at 850 and 200 hPa, `q` at 850 hPa, `msl`, `orog`, `lsm` all download | If any of the wind, humidity or pressure fields is missing, the regime engine cannot be built. Then follow §7.5. |
| **CK4** | Precipitation is cumulative and in mm | M1 | Check that `tp` never decreases from one step to the next, and read its units | Non-decreasing in at least 99.9% of cells; unit is kg m⁻² (same as mm) | Stop and find the reason before any further work. |
| **CK5** | IMD rainfall exists for each chosen season | M1 | Download each year with `imdlib`; check for missing values | At least 95% of valid cells have data on at least 95% of JJAS days | Drop that season (D2). |
| **CK6** | IMD date-stamp convention | M1 | The lag test (T4 in §8.7) | See §8.2 | See §8.2. |
| **CK7** | A list of published active/break days exists | M2 | Look for the Pai et al. (2016) list | List found | Use the statistics check in §11.2 instead. |
| **CK8** | The low-pressure catalogue can be read | M2 | Download the Vishnu et al. tracks from Zenodo; read them | Dates, latitudes, longitudes and a time step are readable | Use default detector settings and label them "untuned" (§11.4). |
| **CK9** | DataMeet district file can be used | M1 | Download the districts file; read its licence page and its description | File downloads, has polygons, licence allows use, and the file states its year and number of districts (write both in `docs/data-sources.md`) | Use, in this order: a file the organisers give you; any other openly licensed India district file. Record the source in `docs/data-sources.md`. Stop district work until one is chosen. |
| **CK10** | ECMWF model version per season | M4 | Read ECMWF's public list of model upgrades | Version found for each season | Write "version unknown" for that season in the report. Do not guess. |
| **CK11** | SIH finale date and duration | Team | Read the official SIH portal | Confirmed | Keep D20 as written. |
| **CK12** | ECDS request size limit | M1 | Try one full-month request | Request accepted | Split requests into smaller pieces (for example half a month). |

---

# 4. Users and Scope

## 4.1 Primary user

A **district disaster-management or emergency-response official**. They need to know:

- Which districts may get significant rain?
- Where are the heavy-rain hotspots?
- How likely is heavy or very heavy rain? How sure is the forecast?
- Why did the corrected forecast change?
- Has a situation like this happened before?

## 4.2 Other users

- **Meteorological analyst:** raw vs corrected, regime, verification.
- **Flood or hydrology user:** hotspots, probabilities, uncertainty.
- **Research user:** model comparison, verification with intervals, model versions.

## 4.3 In scope

Ingestion and time alignment. Features. Regime description. Rainfall correction. Heavy-rain probabilities. Model-estimated range. District products. The six required features (F1–F6). Verification. API. Dashboard.

## 4.4 Out of scope

Making the NWP forecast. Official warnings. Radar nowcasting. Flood simulation. Automatic dispatch. **Western disturbances** (they occur mostly outside June–September). **Live mode.** **Per-feature attribution methods such as SHAP.** Ensemble members. Second rainfall datasets.

---

# 5. Wording and Honesty Rules

These rules apply to the dashboard, slides, README and the demo.

| ID | Rule |
|---|---|
| **H1** | We say "regime information improves the forecast" **only if** the pre-set test in §10.8 says so. Otherwise we say "no demonstrated benefit" and show the result. |
| **H2** | Every skill number shown anywhere must come from the verification engine, from real data. It must show its evaluation set and how many events it is based on. |
| **H3** | The q10–q90 range is called "model-estimated range". It is never called a guaranteed interval. |
| **H4** | The system supports decisions. It is **not** an official warning. The priority table and every alert-like display must say this. |
| **H5** | The audit trail (F2) states what the model did. It must not claim to prove why the weather happened. |
| **H6** | Every screen shows a badge: `REPLAY`. Every replayed date shows its evaluation set: `DEVELOPMENT` (predictions made without the model having seen that season) or `HOLDOUT`. |
| **H7** | Mock or made-up data must show a **MOCK DATA** banner and must never be used for any skill claim. Verification mocks contain empty values (null), never invented numbers. |
| **H8** | If a required feature is not finished, its screen shows "Not available yet". It must not show mock data as if it were real. |
| **H9** | Single-day pictures (for example the improvement map in F1) are examples. The screen must say "one day only, not evidence" and link to the Verification page. |

---

# 6. System Overview

The system has two parts that share the same code for features.

```text
┌──────────────── OFFLINE PART (run by the team, on their laptops) ─────────────────┐
│                                                                                   │
│  TIGGE (ECMWF forecasts)      IMD rainfall       District boundaries              │
│          │                         │                     │                        │
│          └────────────┬────────────┘                     │                        │
│                       ▼                                  │                        │
│        Download → check → align in time and space (§8)   │                        │
│                       ▼                                  │                        │
│               Feature library (§9)  ◄────────────────────┘                        │
│                       ▼                                                           │
│    Regime engine: A phase · B low-pressure · C coast/mountain (§11)               │
│                       ▼                                                           │
│    Models: raw · quantile mapping · global ML · regime-aware ML (§12, §13)        │
│                       ▼                                                           │
│    Grid products → district products → hotspots, priority, analogs, transitions   │
│                       ▼                                                           │
│    Verification on held-out seasons (§16)                                         │
│                       ▼                                                           │
│    Parquet files (cells)  +  PostgreSQL (districts, results, events)              │
└───────────────────────────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼───────── ONLINE PART (also on the laptop) ───────────────┐
│    FastAPI reads stored results only → React dashboard                             │
│    The API never trains a model and never runs a model.                            │
└───────────────────────────────────────────────────────────────────────────────────┘
```

**Rules [Decision]**

- The frontend never reads GRIB files, model files or passwords.
- The API only reads stored results.
- Every stored result records: run ID, model version, feature-set version, alignment method, and fallback flags.
- The same feature code is used for training and for producing results. There is exactly one copy of it.

---

# 7. Data

## 7.1 Sources

| Data | Use | Facts |
|---|---|---|
| **TIGGE, ECMWF control forecast** | The forecast we correct | **[Verified]** The archive starts in October 2006. Access is through the ECMWF Data Store (ECDS). The old download service was replaced by ECDS on 21 April 2026, so old tutorials may not work. **[Verified]** Public access is free after registration, with a 48-hour delay after the forecast start time (this does not matter for past data). **[Verified]** ECMWF's data in TIGGE is licensed CC BY 4.0. **[Verified]** Rain is stored as an accumulation that **starts at step 0** and is given in kg m⁻² (the same as mm). **[Verified]** TIGGE rain comes in 6-hourly steps. |
| **IMD gridded rainfall** | Observed rain (truth) | **[Verified]** Daily, 0.25°, made from rain gauges. The grid has 129 × 135 points, from 6.5°N, 66.5°E to 38.5°N, 100.0°E. Missing values are −999. The `imdlib` Python package downloads and reads it. **[Verified]** The IMD data page describes a 1901–2018 archive. The package also supports a separate real-time product, and other sources describe the data as running to the present. Which product covers which year is checked in CK5. **[Verified]** The number of gauges changed over time (about 4000 at the peak in the early 1990s, about 2000 in the 2010s), so extremes are measured less well than average rain. |
| **District boundaries** | Districts | **[Verified]** DataMeet publishes community-made India district boundaries. It states that the data is not perfect. Decision D7 and check CK9 apply. |
| **Low-pressure catalogue** | Tuning the low-pressure detector | **[Verified]** Vishnu et al. (2020) published tracks and strengths of tropical low-pressure systems found in five reanalyses, for 1979–2019, on Zenodo (doi:10.5281/zenodo.3890646). |
| **Published active/break rule** | Labels for monsoon phase | **[Verified]** Rajeevan et al. (2010). Details in §11.2. |

**Not used:** ERA5, satellite rainfall, NCMRWF products, ensemble members. This keeps the project small.

## 7.2 Exact download list (owner: M1) [Decision]

Every request uses: dataset **TIGGE** on ECDS, `type = cf`, `origin = ecmwf`, start time **00:00**, every day from **June 1 to September 30**, area **North 40, West 55, South 0, East 100**, and grid spacing **0.25°** if the form allows it (CK2).

Build the request in the ECDS form, click **"Show API request code"**, and turn that code into a loop over dates.

| Field | TIGGE name | Level | Steps (hours after start) | Used for |
|---|---|---|---|---|
| Total precipitation | `tp` | surface | 0, 6, 12, …, 78 | Rain windows |
| Mean sea-level pressure | `msl` | surface | 6, 12, …, 72 | Low-pressure detection, trough position |
| Wind (u and v) | `u`, `v` | 850 hPa | 6, 12, …, 72 | Circulation, vorticity, mountain and coast influence |
| Wind (u and v) | `u`, `v` | 200 hPa | 6, 12, …, 72 | Wind shear |
| Specific humidity | `q` | 850 hPa | 6, 12, …, 72 | Moisture |
| Orography (model height) | `orog` | surface | 0 (one date only) | Height, slope, mountain influence |
| Land–sea mask | `lsm` | surface | 0 (one date only) | Coast distance and direction |

**[Verified]** All of these parameters are in TIGGE's parameter list, and the pressure-level fields exist at 850 and 200 hPa. Whether the ECMWF control forecast supplies each one is check **CK3**.

**Size estimate [Assumption]:** the sample file has 29,250 values per step and 4 bytes per value, so one field for one step is about 117 KB. One day needs about 86 field-steps, about 10 MB. One season is about 1.2 GB. Ten seasons are about 12 GB.

## 7.3 Download plan [Decision]

1. **Start the downloads as soon as CK2 and CK3 pass.** ECDS requests wait in a queue. Do not wait for other work.
2. Download **newest season first**: 2025, then 2024, then 2023, and continue backwards one season at a time.
3. Download one month at a time. Make separate requests for the rain group (`tp`) and for the atmosphere group (`msl`, `u`, `v`, `q`). If a request is refused for size, split it (CK12).
4. Download IMD rainfall with `imdlib` for the same years, **plus 1981–2010** (needed for labels and valid cells).
5. Keep a running list in `docs/data-status.md`.

## 7.4 The data status file (owner: M1)

`docs/data-status.md` must contain, before gate G1 is passed:

| Item | Value |
|---|---|
| Seasons downloaded and checked (list of years) | |
| **N** (number of seasons used) | |
| Time alignment method (C1 or C0) | |
| Grid spacing and area actually received | |
| Steps actually received | |
| IMD product used for each season | |
| Number of valid cells | |
| Dates missing per season | |
| ECMWF model version per season (CK10) | |
| District file: source, year, licence, number of districts (CK9) | |

## 7.5 If the wind, humidity or pressure fields cannot be downloaded (CK3 fails)

The regime engine cannot be built. Then:

- The project builds B0, B1 and B2 only.
- Every screen that needs regime information shows "Not available yet" (rule H8).
- The report says clearly that regime-aware results were not produced.
- Work on F3 (analogs) and F4 (transitions) stops, because they need regime information.

---

# 8. Time and Space Alignment

**Owner: M1.** Purpose: make sure the forecast and the observation describe the same 24 hours in the same place. Wrong alignment silently ruins every result.

## 8.1 Two facts

1. **TIGGE rain is cumulative.** The value at step 48 h is the total from hour 0 to hour 48. Daily rain is a **difference** of two cumulative values.
2. **IMD's rain day ends at 08:30 IST, which is 03:00 UTC.** TIGGE steps fall at 00, 06, 12 and 18 UTC. So the IMD day boundary lies **between** two TIGGE steps.

## 8.2 IMD date-stamp rule [Decision]

An IMD value for date **D** is assumed to cover the 24 hours **ending at 08:30 IST on D**, that is from D−1 03:00 UTC to D 03:00 UTC. This is the setting `imd_stamp = end_date` (the default). The alternative is `imd_stamp = start_date`, meaning the value dated D covers D 03:00 UTC to D+1 03:00 UTC.

**The lag test (T4) decides the setting.** Use the newest chosen season, all valid cells. For each shift s ∈ {−1, 0, +1} days, pair the lead-1 forecast window (§8.3) with the IMD field dated (I + 1 + s), where I is the start date. For each pair compute the cell-by-cell correlation between the two fields. Average over all dates.

| Result | Action |
|---|---|
| Shift 0 has the highest average correlation | Keep `end_date`. |
| Shift −1 has the highest | Set `imd_stamp = start_date`. |
| Shift +1 has the highest | Stop. Something else is wrong. The team must find the cause before going on. |
| The best and shift-0 averages differ by less than 0.02 | Keep `end_date`. |

The result and the three averages are written in `docs/data-status.md`.

## 8.3 Turning cumulative forecasts into daily windows

Let I be the forecast start date (00 UTC) and k the lead day (1, 2 or 3).

**IMD date for lead k:** D = I + k if `end_date`; D = I + k − 1 if `start_date`.

**Method C1 (decision D6, used when 6-hourly `tp` is available)**

- The window runs from hour 24(k−1)+3 to hour 24k+3 after the start. For k = 1, 2, 3 that is hours 3→27, 27→51, 51→75.
- Cumulative rain at any hour is found by **straight-line interpolation** between the two nearest 6-hourly steps. The value at step 0 is 0.
- Window rain = cumulative at the end − cumulative at the start.
- This assumes rain falls evenly inside each 6-hour step. That is a small, known approximation.
- If a window value is below 0 (rounding), set it to 0 and count it (test T2).
- **Atmosphere for lead k** = the average of the four 6-hourly snapshots inside the window. For k = 1 these are steps 6, 12, 18, 24; for k = 2 steps 30, 36, 42, 48; for k = 3 steps 54, 60, 66, 72.

**Method C0 (used only when 6-hourly `tp` is not available, D6)**

- Window rain = `tp(24k) − tp(24(k−1))`, with `tp(0) = 0`. So the windows are 00 UTC to 00 UTC.
- This is **3 hours different** from the IMD window. Every stored row carries `alignment_offset_hours = 3`, and every report says so.
- Atmosphere for lead k = the average of the atmosphere steps that are available inside the window. If only the step at 24k exists, use it alone.

**C0 and C1 are never mixed** in one experiment (D6).

## 8.4 Space

- **Project grid:** the IMD 0.25° grid (D5). Each cell has a fixed `cell_id`, used by every table and file.
- **Reading the sample's flat layout (CK1):** if the file has a single `values` dimension, follow the rule in CK1.
- **Rain regridding [Decision]:** if the forecast grid is finer than 0.25°, average all forecast points that fall inside each cell. If the forecast grid is coarser, give each cell the value of the forecast cell that contains its centre. Both keep area-average rain about right. **Do not use bilinear interpolation for rain**, because it changes totals.
- **Atmosphere regridding:** bilinear interpolation.
- **Missing values:** IMD −999 becomes NaN.

## 8.5 Valid cells [Decision]

A cell is **valid** if IMD has a non-missing value on at least 95% of JJAS days in 1981–2010. Only valid cells are used for training, verification and maps. M1 counts them and writes the number in `docs/data-status.md`. **[Assumption]** The expected count is about 4,000–5,000.

## 8.6 Static geography [Decision]

From the one-date `orog` and `lsm` files (D17), on the project grid:

- `elevation_m` = `orog`.
- `slope` = size of the height gradient (metres per metre), by central differences.
- `aspect_sin`, `aspect_cos` = direction of the downhill slope.
- `dist_coast_km` = distance from a land cell to the nearest sea cell (`lsm < 0.5`). Use a distance transform and convert cells to kilometres with 27.75 km per 0.25° of latitude and 27.75 × cos(latitude) km per 0.25° of longitude. **[Assumption]** This approximation is good enough.
- `coast_normal` = unit vector that points from sea to land, taken from the gradient of the land–sea mask.
- `region_code`: see §14.3.

## 8.7 Required alignment tests (owner: M1)

| ID | Test | Pass rule |
|---|---|---|
| **T1** | Cumulative check (CK4) | `tp` non-decreasing in ≥ 99.9% of cells |
| **T2** | No negative windows | Values that had to be set from below 0 to 0 are at most 0.1% of all window values |
| **T3** | Window arithmetic | A synthetic cumulative series gives exactly the expected C1 and C0 windows |
| **T4** | Lag test | Rule in §8.2 |
| **T5** | IMD product check | Write down which IMD product each season uses. If two products cover the same year, compare their mean JJAS rain per cell and report the difference. |
| **T6** | Time zone | 08:30 IST converts to 03:00 UTC in code |
| **T7** | Grid identity | The same `cell_id` gives the same latitude and longitude in every table; the valid-cell count is recorded |
| **T8** | Static fields | `orog` and `lsm` are identical on different dates |

---

# 9. Data Pipeline and Features

## 9.1 Ingestion (owner: M1)

- One **adapter** reads TIGGE files and one reads IMD files. Both produce the same internal format. Nobody else in the team reads GRIB files.
- **Run ID** = `tigge_ecmwf_cf_YYYYMMDDHH`, for example `tigge_ecmwf_cf_2024071500`.
- Running ingestion twice for the same date must not create duplicates.

## 9.2 Checks on every run

Correct file type. Expected variables. Expected steps. Coordinates. Units. No empty fields. Rain ≥ 0 and below 1000 mm in 24 h **[Assumption]**. Tests T1–T2 for the run. A failed check writes a row to `pipeline_events` and triggers the fallback rules (§17). It must never pass silently.

## 9.3 Golden dataset

A Parquet dataset with one row per **(run_id, lead_day, cell_id)**. Columns: `season`, `imd_date`, window rain, observed rain, the window-average atmosphere values, and the fields of the data contract (§9.7). M1 delivers **one season first** (newest), so the other members can start, then all seasons.

## 9.4 Features

Every feature must be computable **at forecast time**, from the forecast run and static fields only.

| Group | Features |
|---|---|
| **Rain** | `rain_mm`; `nbr_mean_3`, `nbr_max_3`, `nbr_mean_5`, `nbr_max_5` (mean and maximum in 3×3 and 5×5 cells); `rain_grad` (size of the rain gradient); `rain_prev_lead`, `rain_next_lead` (rain of the neighbouring lead days from the same run) |
| **Circulation and moisture** | `u850`, `v850`, `wspd850` (wind speed), `vort850` (relative vorticity from finite differences), `q850`, `msl`, `shear_200_850` (size of the wind difference between 200 and 850 hPa) |
| **Geography** | `elevation_m`, `slope`, `aspect_sin`, `aspect_cos`, `dist_coast_km` |
| **Climatology** | `clim_mean`, `clim_p95` (JJAS mean and 95th percentile of observed rain for the cell, from **training seasons only**) |
| **Time and place** | `doy_sin`, `doy_cos`, `lead_day`, `latitude`, `longitude` |

That is **27 features**. They are the inputs of B2 (§12).

The regime engine adds **14 more** for B3 (§11.7): `p_active`, `p_normal`, `p_break`, `regime_confidence`, `lps_present`, `distance_to_lps_km`, `bearing_sin`, `bearing_cos`, `lps_strength`, `lps_influence`, `upslope_flux`, `onshore_flux`, `orographic_influence`, `coastal_influence`. That gives **41 features** for B3.

Missing values: `rain_prev_lead` at lead 1 and `rain_next_lead` at lead 3 are left empty (NaN). XGBoost handles NaN. Logistic regression does not (§11.3 has the rule for that model).

## 9.5 Rules for features [Decision]

- One feature library. Training and result-making call the same functions.
- Feature sets have a version name, `feature_set_version`, stored with every result.
- Statistics such as climatology are computed **only from training seasons** (§10.4).
- The initial hour is always 00 UTC, so it is not a feature.

## 9.6 Feature table

A Parquet table keyed by `(run_id, lead_day, cell_id)`, with `season`, `feature_set_version`, and (for training) the targets: `obs_mm`, and the flags `obs_ge_15_6`, `obs_ge_64_5`, `obs_ge_115_6`.

## 9.7 Internal data contract

```json
{
  "run_id": "tigge_ecmwf_cf_2024071500",
  "source": "ECMWF-TIGGE-control",
  "initialization_time": "2024-07-15T00:00:00Z",
  "lead_day": 1,
  "imd_date": "2024-07-16",
  "window_start_utc": "2024-07-15T03:00:00Z",
  "window_end_utc": "2024-07-16T03:00:00Z",
  "alignment_method": "C1",
  "alignment_offset_hours": 0,
  "unit": "mm",
  "grid": "IMD_0.25",
  "season": 2024
}
```

---

# 10. Experiment Protocol

**Owner: M4** (with M3 for the models). Purpose: make sure the results are honest. Without this section, results would look better than they are.

## 10.1 Basic rules [Decision]

1. **A season is the unit of splitting.** Weather in neighbouring cells and consecutive days is strongly linked, so random row splits leak information.
2. **Development uses leave-one-season-out (LOSO):** each development season is held out once while the others train the models.
3. **One locked holdout run.** The final numbers come from seasons the team has never used for any decision.
4. **The holdout is the newest season(s)**, to look like real use.
5. **Nothing from a held-out season is used to train, tune, calibrate, or choose anything.**

## 10.2 How many seasons go where

N is the number of seasons fixed at gate G1.

| N | Development seasons | Holdout | Consequence |
|---|---|---|---|
| **N ≥ 8** | the oldest N−2 | the newest 2 | Full protocol |
| **5 ≤ N ≤ 7** | the oldest N−1 | the newest 1 | One holdout season. Intervals will be wide. Say so. |
| **3 ≤ N ≤ 4** | all N (LOSO only) | none | Every result is labelled `DEVELOPMENT ONLY`. No result is called final. |
| **N ≤ 2** | — | — | **Stop.** The project cannot be evaluated. The team must download more seasons before going on. |

Example for the target of 10 seasons (2016–2025): development 2016–2023, holdout 2024–2025.

## 10.3 Out-of-fold regime probabilities

The correction model must be trained on regime probabilities that look like the ones it will see in use: made by a model that **never saw that season**.

```text
for each held-out development season h:
    train_seasons = development seasons except h

    # (a) phase model that predicts the held-out season
    phase_model_full = fit_phase_model(train_seasons)
    phase_probs_h    = phase_model_full.predict(features of h)

    # (b) phase probabilities for the training rows: inner loop
    for each s in train_seasons:
        phase_probs_train[s] = fit_phase_model(train_seasons except s).predict(features of s)

    # (c) fit everything else on train_seasons only
    fit climatology, quantile-mapping tables, percentile tables
    fit B2 and B3 (B3 uses phase_probs_train)
    predict season h  ->  out-of-fold predictions for h

pool all out-of-fold predictions -> choose settings, fit calibrators
final models: fit on all development seasons (same inner loop)
holdout: predict once with the final models
```

The phase model is small, so this is cheap. Layers B and C have no trained parameters. Only their percentile tables are fitted on training seasons.

## 10.4 What is fitted on which data

| Part | Fitted on | Applied to |
|---|---|---|
| Label climatology (mean and spread per day of year) | IMD **1981–2010** (fixed, ends before the first forecast season) | All seasons |
| Phase model | Training seasons of each fold (inner loop for training rows) | Held-out season, holdout, new runs |
| Percentile tables for influence scores (§11.6) | Training seasons | Anything |
| Cell climatology features | Training seasons | Anything |
| Quantile mapping (B1) | Training seasons | Anything |
| Low-pressure detector settings | Development seasons that lie inside the catalogue years (2016–2019 if available), then **frozen** | All seasons |
| B2, B3, probability and quantile models | Training seasons | Held-out season, holdout, new runs |
| Calibrators | All pooled out-of-fold predictions of development seasons | Holdout, new runs |
| Analog library and bias table (§15) | Development seasons, **without the query's own season** | Query |

## 10.5 Holdout lock [Decision]

1. Before the first holdout run, commit `config/protocol.yaml` (splits, seeds, primary metrics, decision rule) and tag the code in git.
2. `run_holdout.py` writes a file `holdout.lock` with the git commit, the config hash and the time. **It refuses to run a second time** unless `--force` is used. A forced run is logged in `pipeline_events`.
3. Final numbers come only from the locked run.
4. If anyone looks at holdout results and then changes anything, the holdout is **burned**. From then on only LOSO results are reported, labelled `DEVELOPMENT ONLY`.

## 10.6 Forbidden inputs

A code review rejects any change that breaks one of these rules.

| ID | Forbidden |
|---|---|
| **L1** | Observed rain from any time at or after the forecast start, used as a feature |
| **L2** | Analysis or reanalysis fields as inputs. Only forecast fields are allowed. |
| **L3** | Label-based or in-sample phase probabilities in training rows |
| **L4** | Any statistic that uses holdout seasons (climatology, percentile tables, scalers, quantile-mapping tables) |
| **L5** | Random splits or shuffled cross-validation |
| **L6** | Choosing settings, thresholds or calibration on the holdout |
| **L7** | `cell_id` as a feature |
| **L8** | Rain, forecast or observed, as an input of the phase model |
| **L9** | Columns that never change (for example start hour) |

## 10.7 Counting events

Rows are strongly linked, so counting cell-days overstates the evidence. Every score is stored with `n_samples` (cell-days) **and** `n_events`.

An **event** is an 8-connected group of cells that are at or above the threshold on one date. Groups on consecutive dates that share at least one cell are merged into one event.

## 10.8 Primary metrics and the decision rule

These are fixed in `config/protocol.yaml` **before** the holdout run. Everything else is exploratory and must be labelled so.

| ID | Primary metric |
|---|---|
| **P1** | RMSE of the corrected mean against observed rain, all leads together |
| **P2** | ETS at 64.5 mm |
| **P3** | FSS at 64.5 mm with a 5×5-cell neighbourhood (about 140 km) |
| **P4** | Brier skill score for P(rain ≥ 64.5 mm), compared with the training-season climatology |
| **P5** | Share of observations that fall inside the q10–q90 range (check, not a skill claim; target 80%) |

**Minimum events:** if the holdout has fewer than **30 events** at 64.5 mm, replace 64.5 mm by **15.6 mm** in P2, P3 and P4. Report 64.5 mm as exploratory.

**Decision rule for rule H1 (regime helps):** B3 is better than B2 on a metric only if the 95% interval of the paired difference (§16.6) is entirely on B3's side. The claim "regime information helps" needs this for **P1 and for at least one of P2, P3, P4**, and **no** primary metric may be significantly worse. Otherwise the report says "no demonstrated benefit".

## 10.9 The four systems compared

| ID | System | Question it answers |
|---|---|---|
| B0 | Raw NWP | Reference |
| B1 | Quantile mapping | Does machine learning beat a simple classic fix? |
| B2 | Global machine learning **without** regime features | Does machine learning help at all? |
| B3 | Global machine learning **with** regime features | **Does regime information help?** |

## 10.10 Robustness checks

Report these. Do not tune with them.

1. **Spatial check:** train without some 5°×5° blocks of cells and without latitude/longitude, then test on those blocks. It shows the model is not just remembering places.
2. **Alignment check:** if both C1 and C0 can be built, compare their development scores.

## 10.11 Data size and memory [Decision]

Rows are about `seasons × 122 days × 3 leads × valid cells`. With about 4,500 valid cells [Assumption] and 10 seasons that is about 1.6 × 10⁷ rows. Rules:

- Store as Parquet.
- Train XGBoost with the `hist` method and `QuantileDMatrix` to save memory.
- The config value `train_cell_stride` is **1** (all cells). If training runs out of memory on an 8 GB machine, set it to **2** (every second cell in each direction) and write that in the report. Verification always uses all cells.
- Never remove dry rows from training. That changes the base rates and breaks probabilities.

---

# 11. Regime Engine

**Owner: M2.** Purpose: describe the weather situation for every forecast, so the correction model can use it.

## 11.1 Three layers [Decision]

| Layer | What it describes | Form of the answer |
|---|---|---|
| **A: Monsoon phase** | Active, normal or break. A large-scale state that lasts days to weeks. | Three probabilities. The same for every cell in a run and lead. |
| **B: Low-pressure system** | A moving low or depression. | A rule-based detector. Per-cell distance, direction, strength and an influence score. |
| **C: Local context** | Coast and mountains. | Per-cell influence scores. Not classes, because coast and mountains overlap with the other layers. |

**All regime outputs use forecast fields only.** Observed rain is used only to make phase labels (§11.2).

## 11.2 Layer A labels

**The published rule [Verified].** A break (active) spell is a period when the standardised rainfall of the core monsoon zone is below −1 (above +1) for at least 3 days in a row. The zone is about 18–28°N and 65–68°E to 88°E (sources differ on the western edge). The rule was made for July and August. On average there are about 7 active days and about 7 break days in those two months, breaks last longer, and about 26% of years have no break. The authors found these rain-based spells similar to those from IMD's older wind-and-pressure rules.

**Steps [Decision]**

1. Use the box **18–28°N, 68–88°E**. Compute the daily mean IMD rain over the valid cells in this box.
2. Compute the day-of-year mean and standard deviation from IMD **1981–2010**, using a 31-day window centred on each day of the year to smooth them.
3. Standardised anomaly `z = (rain − mean) / standard deviation`.
4. **Active** = `z > +1` for at least 3 days in a row (all days in the run are labelled active). **Break** = `z < −1` for at least 3 days in a row. **Normal** = everything else.
5. The label belongs to the **IMD date**. There is no smoothing of the daily series.
6. The rule was made for July and August. Using it in June and September is an extension. Report skill separately for July–August and for June and September.

**Checking the labelling code (CK7) [Decision]**

- **If the published list exists:** run the code on IMD rain for the years the list covers and compare the days. Report how many days match.
- **If it does not exist:** run the code on 1981–2010 and check the July–August statistics. Pass if all three hold: the average number of active days per season is within 7 ± 3, the average number of break days per season is within 7 ± 3, and the share of seasons with no break day is within 26% ± 10 percentage points **[Assumption: tolerances]**. If not, check the box, the climatology and the code. Do not change the rule.

## 11.3 Layer A features and model

**Features (from forecast fields, window averages of §8.3)**

| ID | Feature | Definition |
|---|---|---|
| A1 | Peninsular westerly wind | Mean `u850` over 10–20°N, 65–85°E |
| A2 | Core-zone vorticity | Mean `vort850` over 18–28°N, 68–88°E |
| A3 | Trough position | Latitude of the lowest value of the zonal-mean `msl` over 75–90°E, searched between 15°N and 32°N |
| A4 | Core-zone pressure anomaly | Mean `msl` over the core zone minus its training-season mean |
| A5 | Moisture | Mean `q850` over 10–25°N, 70–90°E |
| A6 | Wind shear | Mean of (`u850` − `u200`) over 5–20°N, 60–100°E |
| — | Season position | `doy_sin`, `doy_cos` |

Each of A1–A6 is used for the lead itself and for the two neighbouring leads of the same run. If a neighbouring lead does not exist (lead 0 or lead 4), **repeat the nearest available lead**. That gives 18 + 2 = 20 inputs. **[Assumption]** All boxes are starting values, kept in `config/regime.yaml`.

**Sanity check [Decision]:** using training seasons, make maps of the average forecast 850 hPa wind on active days minus break days. Look at them. If A1–A6 clearly do not separate the two classes, tell the team and revise the boxes. Do not change the labels.

**Model [Decision]**

- Multinomial logistic regression with L2 penalty. Inputs are standardised with training-season mean and spread. `C` is chosen from {0.01, 0.1, 1, 10} by the lowest mean LOSO log-loss.
- Output: `p_active`, `p_normal`, `p_break`. They add up to 1.
- No extra calibration. Report a reliability plot.

**Evaluation:** LOSO log-loss and Brier score; macro-F1 and per-class recall (normal dominates, so accuracy alone misleads); each compared with the climatological probabilities. **If the model does not beat the climatological probabilities, say so in the report and still use its outputs. Do not change the labels to make it work.**

**Confidence and out-of-range flag [Decision]**

- `regime_confidence = 1 − H / ln 3`, where H is the entropy of the three probabilities. It runs from 0 (no information) to 1 (certain).
- `confidence_band`: **High** if ≥ 0.50, **Medium** if 0.20 to below 0.50, **Low** if below 0.20 **[Assumption]**.
- `ood_flag = true` if at least **2 of the 6** features A1–A6 lie outside the 1st–99th percentile range of the training seasons.
- Low confidence is not a failure. It only means the phase does not push the correction much.

## 11.4 Layer B: low-pressure systems

**A rule-based detector. Nothing is trained**, because lows and depressions are too rare for a trustworthy trained model. IMD calls a system with 17–27 knots a depression and 28–33 knots a deep depression **[Verified]**. This detector finds low-pressure systems in general and does not assign IMD classes.

**Algorithm [Decision; numbers are [Assumption], see the tuning below]**

For each (run, lead), on the window-average fields over 5–30°N, 60–100°E:

1. Compute 850 hPa relative vorticity ζ and smooth it with a Gaussian (σ = 1.5°).
2. Candidate centres are local maxima of smoothed ζ above `zeta_min` (start value 1.5 × 10⁻⁵ s⁻¹).
3. Keep a candidate only if the mean sea-level pressure at its centre is at least `dp_min` hPa below the average pressure in a ring 500–800 km around it (start value 2 hPa).
4. If two candidates are within 500 km, keep the stronger one.
5. For each kept centre store latitude, longitude, ζ maximum, pressure at the centre and `strength_percentile` (the rank of its ζ maximum among centres found in training seasons).

**Per-cell features [Decision]**

| Feature | Meaning |
|---|---|
| `lps_present` | 1 if at least one centre is found anywhere in the area, else 0 |
| `distance_to_lps_km` | Distance to the nearest centre. 3000 if none. |
| `bearing_sin`, `bearing_cos` | Direction to the nearest centre. 0 and 0 if none. |
| `lps_strength` | `strength_percentile` of the nearest centre. 0 if none. |
| **`lps_influence`** | `lps_strength × exp(−(distance / 600 km)²)`. 0 if none. |

**`lps_influence` is a simple index from 0 to 1. It is not a probability.** It means "how strong the nearest low is, reduced with distance".

**Tuning (owner: M2) [Decision]**

- Use development seasons that lie inside the catalogue years (2016–2019 if they are development seasons). Try every combination of `zeta_min` ∈ {1.0, 1.5, 2.0} × 10⁻⁵ s⁻¹, `dp_min` ∈ {1, 2, 3} hPa and smoothing σ ∈ {1.0°, 1.5°}.
- A detection **matches** a catalogue system if they are within `300 km + 100 km × (lead − 1)`. The catalogue reference for a window is every system present in at least 2 of the catalogue times inside the window, at its average position.
- Choose the combination with the highest CSI. On a tie choose the larger `zeta_min`. Report hit rate (POD), false-alarm ratio and CSI. **Freeze** the settings before the LOSO runs.
- If the catalogue cannot be read (CK8), use the start values and label the detector **"untuned"** everywhere it appears.

## 11.5 Layer C: coast and mountains

**Raw indices [Decision]** (per cell, from window-average winds and humidity)

```text
grad_h          = gradient of `elevation_m`
upslope_flux    = q850 × max(0, wind · grad_h)          # moist wind blowing up a slope
onshore_flux    = q850 × max(0, wind · coast_normal) × exp(−dist_coast_km / 100)
```

**Influence scores [Decision]**

- `orographic_influence` = 0 if `upslope_flux` = 0. Otherwise it is the **percentile rank of `upslope_flux` among the positive training values**. It runs above 0 up to 1.
- `coastal_influence` = the same for `onshore_flux`.
- The ranks come from tables made from training seasons only, stored as an artefact and applied unchanged to new runs.
- Flat cells and cells with downhill or offshore wind get 0. This is why zero is handled separately.

**Flags for the screen:** `orographic_favorable` = `orographic_influence ≥ 0.90`; `coastal_favorable` = `coastal_influence ≥ 0.90` **[Assumption]**. They are display flags, not observed truth. On development seasons, compare the error (observed − raw) of flagged and unflagged cells, and report it in the verification report.

## 11.6 Regime output contract

Domain level (one per run and lead):

```json
{
  "run_id": "tigge_ecmwf_cf_2024071500",
  "lead_day": 1,
  "phase": {
    "active_probability": 0.42,
    "normal_probability": 0.46,
    "break_probability": 0.12,
    "regime_confidence": 0.18,
    "confidence_band": "low",
    "regime_source": "final"
  },
  "lps": {
    "detected": true,
    "centres": [{ "lat": 20.4, "lon": 86.1, "zeta_max": 2.1e-5, "mslp_min_hpa": 996.0, "strength_percentile": 0.74 }],
    "settings": "tuned"
  },
  "quality": { "ood_flag": false, "regime_available": true }
}
```

Cell level (one per run, lead and cell):

```json
{
  "cell_id": 8123,
  "lps_present": true, "distance_to_lps_km": 260, "bearing_to_lps_deg": 235, "lps_influence": 0.52,
  "orographic_influence": 0.88, "coastal_influence": 0.35,
  "orographic_favorable": false, "coastal_favorable": false
}
```

The numbers are examples. `regime_source` is `oof` for training rows and `final` for everything else. **Training code must stop with an error if any training row has `regime_source = final`.**

## 11.7 The 14 regime features given to B3

`p_active`, `p_normal`, `p_break`, `regime_confidence` (Layer A) · `lps_present`, `distance_to_lps_km`, `bearing_sin`, `bearing_cos`, `lps_strength`, `lps_influence` (Layer B) · `upslope_flux`, `onshore_flux`, `orographic_influence`, `coastal_influence` (Layer C).

---

# 12. Rainfall Correction

**Owner: M3.** All systems are trained and scored under the protocol in §10 and on exactly the same rows.

## 12.1 The four systems

| ID | System | Definition |
|---|---|---|
| **B0** | Raw NWP | The aligned window rain (§8.3). No change. |
| **B1** | Quantile mapping | For each lead and each `region_code` (§14.3), use all training cell-days to make the forecast-rain distribution and the observed-rain distribution. For a raw value x: p = share of training forecast values ≤ x (at ties, use the middle of the tie); corrected = the observed value at probability p. Use 100 probability steps with straight-line interpolation. Above the largest step, multiply by the ratio of the 99.5th percentiles (observed ÷ forecast). |
| **B2** | Global machine learning, no regime | XGBoost with the 27 features of §9.4. |
| **B3** | Global machine learning, with regime | XGBoost with the 41 features (27 + 14 of §11.7). |

## 12.2 The XGBoost model [Decision]

- **Objective:** `reg:tweedie`. Rain has many zeros and a long tail, and this objective handles both. It cannot give negative rain.
- **Target:** observed rain in mm. Raw rain is one of the inputs.
- **Monotone constraint:** +1 on `rain_mm`. More forecast rain never gives less predicted rain when everything else is the same.
- **Method:** `tree_method = hist`.
- **Settings search:** 20 random combinations from the lists below, using a seed from `protocol.yaml`. Each combination is scored on **3 validation seasons** (the newest 3 development seasons, one at a time, trained on the other development seasons). Choose the combination with the lowest mean RMSE (metric P1). **B2 and B3 use the same 20 combinations**, so the comparison is fair. No early stopping.

| Setting | Values |
|---|---|
| `tweedie_variance_power` | 1.2, 1.5, 1.8 |
| `max_depth` | 4, 5, 6, 7, 8 |
| `min_child_weight` | 50, 100, 200, 400 (large, because rows are strongly linked) |
| `learning_rate` | 0.03, 0.05 |
| `n_estimators` | 200, 400, 800 |
| `subsample` | 0.7, 0.8 |
| `colsample_bytree` | 0.7, 0.8 |
| `reg_lambda` | 5, 10, 20 |

## 12.3 Output [Decision]

- **`corrected_mean_mm`** is the model's expected rain. It is the headline "AI corrected" value and is the value scored in RMSE and the threshold metrics.
- The median (`q50`, §13.7) is a different number. It is normally lower than the mean for rain. **Never present the two as the same.**
- **Extrapolation flag:** if a cell's `rain_mm` is above the 99.9th percentile of the training seasons **[Assumption]**, set `extrapolation_flag = true`. Trees cannot go beyond their training range. That cell uses the fallback in §17.

## 12.4 Model records

Each fitted model is saved with a `model_versions` record: name, role, algorithm, settings, feature-set version, training seasons, git commit, and development scores. XGBoost version must be **2.0 or newer** (D13).

---

# 13. Probabilities, Range and Extremes

**Owner: M3** (models) **and M4** (calibration and checks).

## 13.1 Outputs [Decision]

For every cell, calibrated probabilities that the 24-hour rain is at least:

```text
15.6 mm    moderate or more
64.5 mm    heavy
115.6 mm   very heavy
```

IMD's 24-hour categories are heavy 64.5–115.5 mm, very heavy 115.6–204.4 mm and extremely heavy 204.5 mm and above **[Established]**. A cell value is an average over about 700 km², so exceeding a threshold is **rarer for a cell than at a single rain gauge**. The report must say this. **Extremely heavy (204.5 mm) is not modelled**, because it is too rare on this grid.

## 13.2 Count events first [Decision]

Before choosing models, M4 writes `docs/event-counts.md`: the number of events (§10.7) and cell-days at each threshold, per season and lead, for development seasons.

| Events at the threshold in development seasons | What is built |
|---|---|
| At least 30 | A classifier of its own |
| 115.6 mm has fewer than 30, but 64.5 mm has at least 30 | **Chained form:** P(≥115.6) = P(≥64.5) × P(≥115.6 given ≥64.5) |
| 64.5 mm has fewer than 30 | No heavy-rain model. The screen says "Not enough events". Only P(≥15.6) is shown. Hotspots use 15.6 mm (F6). |

## 13.3 Models [Decision]

- XGBoost `binary:logistic` with the 41 features of B3.
- Natural base rate. **No oversampling and no removing of dry rows**, because that would make the outputs stop being probabilities.
- Settings: the same 20-combination search as in §12.2, but scored by the Brier score.
- After prediction, enforce `P(≥115.6) ≤ P(≥64.5) ≤ P(≥15.6)` for every cell.

## 13.4 Calibration [Decision]

Fitted on the **pooled out-of-fold predictions** of development seasons (never on training predictions, never on the holdout).

- If the pooled predictions contain **at least 200 events** at that threshold: isotonic regression.
- Otherwise: Platt scaling (a logistic curve on the raw score).

Report reliability diagrams per lead.

## 13.5 Baselines for probabilities

- **Climatology** from the training seasons.
- **Raw neighbourhood fraction:** the share of the 5×5 cells around the cell whose raw rain is at or above the threshold.

## 13.6 Rules for very high rain [Decision]

| ID | Rule |
|---|---|
| **X1** | Alerts use probabilities, not the point value. A mean-type forecast reduces peaks, so the screen shows probabilities and the upper range beside the corrected mean. |
| **X2** | If a cell has `extrapolation_flag = true`, the fallback of §17 is used and a visible flag is shown. |
| **X3** | If the correction lowers a raw forecast that is at or above 64.5 mm, while P(≥64.5) is 0.5 or more, the screen shows both values and the message "correction lowers a heavy raw forecast". |
| **X4** | The attention levels of F5 are fixed display levels. They are not tuned to look good. |
| **X5** | Low regime confidence never hides a heavy-rain probability. |
| **X6** | Districts with few cells show the small-district flag (§14.4). |

## 13.7 Model-estimated range [Decision]

- Three XGBoost models with `reg:quantileerror` and `quantile_alpha` = 0.1, 0.5, 0.9, trained on `log1p(observed rain)`. Quantiles stay correct after the reverse step `expm1`. **(Means do not, which is why the mean model works in mm.)**
- Sort the three values in each cell so that q10 ≤ q50 ≤ q90 (quantiles can cross). Clip at 0.
- Use `hist`. Do not use `exact`.
- **Coverage check:** on out-of-fold predictions, measure per lead how many observations fall inside q10–q90. The target is 80%. If any lead differs from 80% by more than 10 percentage points, the screen must show the measured coverage next to the range ("in testing, this range contained X% of observations").
- The label is always **"model-estimated range (10th–90th percentile)"** (rule H3).
- There is no district-level range (§14.4). A district shows the range of its wettest cell.

---

# 14. District Products

**Owner: M1** (weights and aggregation) **and M4** (checks).

## 14.1 Grid first

All models work on cells. A district average hides local rain (for example, district mean 30 mm while its wettest cell has 90 mm). Cell results are always kept. Verification works on cells first.

## 14.2 Weights [Decision]

For district j and cell i:

```text
w_ij = area(cell i ∩ district j) / area(district j)
```

- Compute the overlaps once, offline, with `geopandas`, in the equal-area projection **EPSG:6933**.
- **Only valid cells count.** Drop the weights of non-valid cells (sea, outside India) and **divide the remaining weights so they add up to 1** for each district.
- A cell counts as **"main"** for a district if its weight is at least **0.05** (`w_min`).

## 14.3 Region boxes [Decision]

Regions are used for grouping results and for the demo highlight. They are simple boxes, **not official regions**. A cell gets the first region that fits, in this order:

| Order | `region_code` | Rule |
|---|---|---|
| 1 | `HIMALAYA` | latitude ≥ 28.0 |
| 2 | `NORTHEAST` | longitude ≥ 88.5 |
| 3 | `WEST_COAST` | longitude < 77.0 and latitude < 21.0 |
| 4 | `NORTHWEST_WEST` | longitude < 77.0 and latitude ≥ 21.0 |
| 5 | `SOUTH_EAST` | latitude < 18.0 |
| 6 | `CENTRAL_EAST` | everything else |

(The cell's centre is used. `WEST_COAST` is the demo highlight region, D1.)

## 14.4 District fields [Decision]

Let `m_i` be the corrected mean of cell i, `p_i(t)` its calibrated probability at threshold t, and S the main cells of the district.

| Field | Definition | Note |
|---|---|---|
| `raw_mean_mm` | Σ w_i · raw_i | Area-weighted mean |
| `corrected_mean_mm` | Σ w_i · m_i | Expected district-mean rain. Means add up exactly. |
| `wettest_cell_id` | the cell in S with the largest `m_i` | |
| `wettest_cell_mean_mm` | `m` of that cell | |
| `wettest_cell_q10_mm`, `_q50_mm`, `_q90_mm` | that cell's range | Always labelled "range for the wettest cell" |
| `heavy_prob_max_cell` | max over S of `p_i(64.5)` | The highest cell probability. It is a **lower limit** for "some cell gets heavy rain". |
| `very_heavy_prob_max_cell` | max over S of `p_i(115.6)` | |
| `heavy_area_fraction_expected` | Σ w_i · `p_i(64.5)` | Expected share of the district area with heavy rain. This is exact. |
| `very_heavy_area_fraction_expected` | Σ w_i · `p_i(115.6)` | |
| `n_effective_cells` | 1 / Σ w_i² | How many cells effectively make up the district |
| `is_small` | `n_effective_cells < 4` | Small districts have noisy numbers |
| `centroid_lat`, `centroid_lon` | Centre of the district | Used by F4 |

If the 64.5 mm or 115.6 mm model is not available (§13.2), the matching fields are null.

**Why there are no district quantiles and no "district probability":** the quantile of an average is not the average of quantiles. And "some cell gets heavy rain" cannot be found by multiplying cell probabilities, because neighbouring cells are strongly linked. The fields above are the ones that can be calculated correctly.

**Observed counterparts** (for verification), with the same weights and S: `observed_mean_mm = Σ w_i · obs_i`, `observed_max_cell_mm = max over S of obs_i`, `observed_heavy_area_fraction = Σ w_i · 1[obs_i ≥ 64.5]`.

**Screen wording**

| Field | Label on screen |
|---|---|
| `corrected_mean_mm` | "AI-corrected district mean (expected)" |
| `wettest_cell_q10..q90` | "Range for the wettest cell (model-estimated)" |
| `heavy_prob_max_cell` | "Chance of heavy rain in the most exposed cell" |
| `heavy_area_fraction_expected` | "Expected share of the district with heavy rain" |

**Districts after 2011** are not in the file (D7). The dashboard's Model Information page says so.

---

# 15. Required Features

All six features are required. Each one has a **minimum version** below. The minimum version is what "done" means.

If a feature cannot show real data (for example, because a data check failed), it shows "Not available yet" (rule H8).

## F1. Raw vs AI-Corrected Toggle

**Owner: M5** (screen) and **M4** (improvement data). **Purpose:** switch between the original ECMWF forecast and the corrected forecast, and make the improvement obvious.

**Map layers**

| Layer | Meaning |
|---|---|
| `raw` | Raw ECMWF forecast rain |
| `corrected` | AI-corrected mean |
| `difference` | corrected − raw (blue for lower, red for higher) |
| `observed` | IMD observed rain |
| `improvement` | `|raw − observed| − |corrected − observed|` per cell. Positive means the corrected forecast is closer to the observation. |

**Screen must have**

1. Toggle buttons for the layers above.
2. A **split view**: two maps side by side (raw on the left, corrected on the right), with linked zoom and pan.
3. The `improvement` layer in two colours (one for "corrected closer", one for "raw closer") plus grey for no change.
4. A line for the selected date and lead: **"Corrected is closer to the observation in X of Y cells and A of B districts. One day only, not evidence."** The numbers are real counts. The words link to the Verification page (rule H9).
5. A district panel with raw, corrected, observed, and difference.

**Done when:** all five items work on real replayed data.

## F2. AI Correction Audit Trail

**Owner: M1** (assembly in the API) and **M3** (model data). **Purpose:** show how and why the forecast was corrected. This is the only explanation in the system (D12).

**The audit trail for one district and lead shows these steps, in this order:**

1. **Raw forecast:** raw mean and raw wettest-cell value.
2. **Detected regime:** phase probabilities, confidence band, nearest low-pressure system (distance, direction, influence), coastal and mountain influence of the district.
3. **What happened in the past under this regime:** from the bias table (below).
4. **Correction applied:** corrected − raw (district mean and wettest cell).
5. **Corrected forecast:** corrected mean.
6. **Confidence and uncertainty:** heavy-rain probabilities, the model-estimated range for the wettest cell, and its measured coverage (§13.7).
7. **Record:** run ID, model versions, feature-set version, alignment method, fallback flags, evaluation set.

**Bias table (data-driven).** For each district, lead, phase (most probable phase) and `lps_near` (nearest low-pressure system within 500 km of the district centre: yes or no), M4 computes from **development seasons, without the query's own season**: for each matching date, the difference `observed district mean − raw district mean`. The table stores the **median**, the **25th and 75th percentiles**, and the **number of dates**. If the number of dates is below 20, the screen shows "few past cases".

**Sentence rules.** The summary sentences are built from templates that only state numbers, for example: "The model raised the district mean from {raw} to {corrected} mm ({diff:+}). Most likely phase: {phase} (confidence {band})." **Sentences must not claim causes** (rule H5).

**Done when:** all seven steps show real values for any replayed district.

## F3. Historical Analog Finder

**Owner: M2.** **Purpose:** find past situations that look like the current one, and show what rain and forecast errors followed.

**Vector [Decision].** For one (run, lead): `p_active`, `p_break`, A1–A6 (from the window average at that lead), `lps_present`, `lps_strength`. That is **10 numbers, the same for all districts.**

**Library.** All (run, lead) pairs of the **same lead** from **development seasons other than the query's own season**. For a holdout query, all development seasons.

**Method**

1. Standardise each of the 10 numbers with the library's mean and spread.
2. Distance = ordinary (Euclidean) distance.
3. Take the nearest analog. Then repeatedly take the next nearest whose date is **at least 5 days** away from every analog already taken. Stop at **K = 5**.
4. `distance_percentile` = the rank of the analog's distance among all library distances (small = very similar).

**Outputs for a chosen district** (from `district_history`): for each analog, its date and season; the distance and percentile; **observed** district mean and observed wettest-cell rain; **raw forecast** district mean at that lead; **corrected forecast** district mean (out-of-fold); and the error (observed − raw). The screen also shows the median error of the 5 analogs, marked "5 cases only".

**Not shown:** a "similarity percent". Only distance and percentile are shown, because a percent would have no clear meaning.

**Done when:** for any replayed (run, lead, district) the five analogs and their outcomes appear, and no analog comes from the query's own season.

## F4. Regime Transition Detection

**Owner: M2** (detection) **and M5** (screen). **Purpose:** show changes of monsoon phase and low-pressure events that can change the forecast.

**Series [Decision].** For a run with lead-1 IMD date d₁, build a series over IMD dates from d₁ − 6 to d₁ + 2: dates before d₁ come from the **lead-1 forecast of the earlier runs**, and d₁, d₁+1, d₁+2 come from **leads 1, 2, 3 of this run**. If a run is missing, the series has a gap and no event is found across the gap.

**Phase transitions**

- Smooth `p_active`, `p_normal`, `p_break` with a **3-day mean** (using the dates that exist).
- `state(d)` = the phase with the highest smoothed probability.
- A **transition** happens at date d if `state(d−1) = X`, `state(d) = Y ≠ X`, and the smoothed probability of Y is at least 0.5 on d **and** on d+1. If d+1 is beyond the series, the event is stored as `confirmed = false` ("pending").
- Event type: `PHASE:X->Y`. Its `confidence` is the mean smoothed probability of Y on d and d+1.

**Low-pressure events**

| Event | Rule |
|---|---|
| `LPS_FORMS` | No system detected on d−1, one detected on d and on d+1 |
| `LPS_ENDS` | A system detected on d−1, none on d and d+1 |
| `LPS_NEAR_DISTRICT` | For a district: the nearest system is 500 km or closer to the district centre on d, and was farther than 500 km on d−1 |

Low-pressure events are rule-based and have **no probability** (`confidence = null`).

**Screen.** A timeline chart of the three phase probabilities with event markers, and a list of events. For each event, show the **change in the domain-mean corrected rain**: mean of up to 3 dates after the event minus mean of up to 3 dates before it, with the number of dates used. This shows how the forecast changes around the event.

**Done when:** for any replayed run, the timeline, the event list and the change numbers show real values, and a district can be chosen to see `LPS_NEAR_DISTRICT`.

## F5. District Priority Table

**Owner: M4** (rules) **and M5** (screen). **Purpose:** show which districts need more attention. It uses the 2011 DataMeet boundaries (D7).

**Attention level [Decision]** for a district, using §14.4 fields and the display levels below **[Assumption]** (stored in `config/thresholds.yaml`):

| Level | Rule |
|---|---|
| `HIGH` | `heavy_prob_max_cell ≥ 0.50` **or** `very_heavy_prob_max_cell ≥ 0.20` |
| `WATCH` | not `HIGH`, and (`heavy_prob_max_cell ≥ 0.20` **or** `corrected_mean_mm ≥ 15.6`) |
| `NORMAL` | everything else |
| `UNAVAILABLE` | the district's forecast is a fallback product (§17) |

**If the 64.5 mm model is not available (§13.2):** the heavy-rain fields are null, and the levels use the highest cell probability of P(≥15.6 mm) instead: `HIGH` if it is at least 0.80, `WATCH` if it is at least 0.50, otherwise `NORMAL` **[Assumption]**. The table says which rule is in use.

**Order.** Sort by level (`HIGH`, `WATCH`, `NORMAL`, `UNAVAILABLE`), then by `heavy_prob_max_cell` (high first), then `heavy_area_fraction_expected`, then `wettest_cell_mean_mm`, then district name.

**Screen.** A sortable table with these columns: rank, district, state, level, corrected mean, raw mean, wettest-cell mean, `heavy_prob_max_cell`, `very_heavy_prob_max_cell`, `heavy_area_fraction_expected`, `is_small`. A filter by state. A note: **"Model-based attention level. Not an official warning."** (rule H4). Clicking a row opens the district page.

**Done when:** the table shows all districts for any replayed run and lead, sorts correctly, and shows the note.

## F6. Spatial Hotspots

**Owner: M4** (computation) **and M5** (map). **Purpose:** show where heavy rain is most likely.

**Rule [Decision].** A cell is a **hotspot cell** if `P(≥ hotspot threshold) ≥ 0.50` and `corrected_mean_mm ≥ 15.6`. The hotspot threshold is **64.5 mm**. If the 64.5 mm model is not available (§13.2), it is **15.6 mm**, and the screen says which threshold is used.

**Grouping.** Group touching hotspot cells (8-connected). For each group store: number of cells, highest probability, highest corrected mean, centre, outline (a GeoJSON polygon made by joining the cell squares), and the list of districts that contain at least one main cell of the group.

**Screen.** A map layer with the outlines, shaded by the highest probability. Clicking an outline shows its numbers and its districts.

**Done when:** hotspots for any replayed run and lead are drawn on the map with their numbers, and the threshold in use is visible.

---

# 16. Verification

**Owner: M4.** Purpose: measure honestly whether the corrected forecast is better than the raw forecast. The verification engine is separate from the frontend, is a Python package, and stores its results.

## 16.1 Rules [Decision]

- Verification is done **on cells first**. (FSS needs a grid.) Districts are not verified separately in this project.
- All forecast types are scored on **exactly the same** dates, cells and leads, and only on cells that have an observation.
- Forecast types: `raw_nwp` (B0), `quantile_mapping` (B1), `global_ml` (B2), `regime_aware_ml` (B3).
- Two evaluation sets are kept apart: `development` (out-of-fold predictions on development seasons) and `holdout` (final models on holdout seasons). **Only `holdout` numbers from the locked run are final** (§10.5).
- Every score is stored with `n_samples`, `n_events` and a confidence interval (§16.6).

## 16.2 Metrics

For a threshold t, count over the chosen samples: **a** = hits (forecast ≥ t and observed ≥ t), **b** = false alarms (forecast ≥ t, observed < t), **c** = misses (forecast < t, observed ≥ t), **d** = correct negatives, and n = a + b + c + d. Formulas are in Appendix A.

| Metric | Type | Threshold | What it tells you |
|---|---|---|---|
| **RMSE** | Continuous | none | Typical size of the error. Large errors count most. It favours smooth forecasts, so read it together with FSS and frequency bias. |
| **Bias, MAE** | Continuous | none | Average over- or under-forecast; typical absolute error |
| **POD** | Categorical | 15.6, 64.5, 115.6 mm | Share of observed events that were forecast |
| **FAR** | Categorical | same | Share of forecast events that did not happen. This is the false alarm **ratio** b/(a+b), not the false alarm rate. |
| **CSI** | Categorical | same | Hits divided by hits + misses + false alarms |
| **ETS** | Categorical | same | CSI after removing hits expected by chance |
| **Frequency bias** | Categorical | same | (a+b)/(a+c). If it is above 1, events are forecast too often. It explains why POD can rise without real skill. |
| **FSS** | Spatial | same, with neighbourhoods of 1, 3, 5, 9 cells | Do forecast events fall near observed events? Small position errors are forgiven. |
| **Brier score, BSS** | Probability | same | Accuracy of probabilities; skill against a reference (§13.5) |
| **Reliability** | Probability | same | Does "30%" happen about 30% of the time? |
| **Pinball loss, coverage** | Range | q10, q50, q90 | Accuracy of the range; share of observations inside q10–q90 |

RMSE is for continuous rain. The others use thresholds. To compare raw and corrected, both are scored on the same samples and the **paired difference** is reported with its interval.

## 16.3 Fairness [Decision]

- A corrected forecast changes how often events are forecast. Always show frequency bias, and compare with B1, which is built to match how often rain occurs.
- Show FSS with absolute thresholds **and** with percentile thresholds (each system's own top percent). Percentile thresholds separate position skill from frequency effects.

## 16.4 What is stored

For each **(imd_date, lead_day, forecast_type, evaluation_set, threshold, group)**:

- counts a, b, c, d;
- for each neighbourhood size: three FSS sums (sum of squared differences, sum of squared forecast fractions, sum of squared observed fractions);
- for continuous scores: n, sum of errors, sum of absolute errors, sum of squared errors;
- for probabilities: reliability-bin counts and sums, sum of Brier terms;
- for the range: pinball sums and coverage counts.

These **components** are stored in Parquet files (`data/verification/components/`). They can be added up over any period or group, and the bootstrap can be re-run, without recomputing from the raw grids. **FSS is always added up as (sum of numerators) ÷ (sum of denominators), never as an average of daily FSS.** The final results (with intervals) are stored in PostgreSQL (§20).

## 16.5 Groups [Decision]

`all`; lead day; month (Jun, Jul, Aug, Sep); `region_code`; phase (most probable phase); `lps_near` (nearest low-pressure system within 500 km of the cell, yes or no); `orographic_favorable`; `coastal_favorable`; raw-rain size (below 1 mm, 1–15.6 mm, 15.6–64.5 mm, 64.5 mm and above).

**A group is shown only if it has at least 10 events.** Otherwise the screen says "Not enough events".

## 16.6 How to show that the correction really helps [Decision]

1. **Fixed plan.** The primary metrics P1–P5 and the decision rule (§10.8) are fixed before the holdout run. Everything else is exploratory.
2. **Paired differences.** For each date, compute the score of two systems on the same cells (for example B3 and B0, or B3 and B2) and take the difference.
3. **Block bootstrap.** Resample blocks of 7 consecutive dates with replacement, until the original number of dates is reached. Use the **same blocks for both systems**. Repeat 2,000 times. Recompute the metric from the stored components each time. The 2.5th and 97.5th percentiles are the 95% interval. Repeat with blocks of 3 and 14 days to see whether the answer changes. **Never use ordinary standard errors or t-tests on cells or days**, because they are not independent.
4. **Season consistency.** Report in how many seasons the corrected forecast was better. With at least 6 seasons, add a Wilcoxon signed-rank test on the per-season differences.
5. **Equal-accuracy test.** For RMSE, add a Diebold–Mariano test with an autocorrelation-robust variance, on the daily loss values.
6. **Many metrics.** Only P1–P5 can support headline claims. Report all others without picking.
7. **The four systems in a row:** B0 → B1 → B2 → B3. "B3 vs B2" is the test of regime information (§10.8).
8. **Show where it helps and where it hurts:** maps of average improvement per cell, and results by group.
9. No improvement numbers appear in this document or in any mock. They exist only after this procedure has run on real data.

## 16.7 Edge cases [Decision]

- If there are no observed events (a + c = 0) or no forecast events (a + b = 0), the affected metric is **undefined**. Store NULL and the reason. Never store 0.
- A missing observation is excluded for all systems equally.

## 16.8 How to compute FSS

1. Turn the forecast and observed fields into 0/1 fields at threshold t.
2. For a neighbourhood of n × n cells, compute in each window the fraction of cells that are 1. Use only valid cells inside the window (`scipy.ndimage.uniform_filter` on the 0/1 field and on the valid-cell mask, then divide). **Sea and missing cells must not count as dry.**
3. `FSS = 1 − Σ(Pf − Po)² / (ΣPf² + ΣPo²)`, summed over all valid points and all dates.
4. Also report the "useful skill" line `0.5 + f₀/2`, where f₀ is the observed event fraction (Roberts & Lean 2008) **[Established]**.
5. One cell is about 28 km. Neighbourhoods of 1, 3, 5, 9 cells are about 28, 84, 140 and 250 km.

## 16.9 Standard plots

Performance diagram (POD against success ratio, with CSI lines and frequency-bias lines). FSS against neighbourhood size. Reliability diagrams with counts. Bias by raw-rain size. Map of average improvement per cell. Per-season improvement dots. The B0→B3 table with intervals.

## 16.10 Python interface

`verification.compute(...)`, `verification.aggregate(...)`, `verification.bootstrap(...)`. Components go to Parquet; results go to PostgreSQL.

## 16.11 Output contract

Values are `null` here **on purpose**. No result exists until the engine has run on real data.

```json
{
  "_example": true,
  "forecast_type": "regime_aware_ml",
  "comparison_to": "raw_nwp",
  "evaluation_set": "holdout",
  "lead_day": 1,
  "threshold_mm": 64.5,
  "neighbourhood_cells": 5,
  "group": { "type": "all", "value": "all" },
  "metrics": {
    "rmse":           { "value": null, "ci_low": null, "ci_high": null },
    "pod":            { "value": null, "ci_low": null, "ci_high": null },
    "far":            { "value": null, "ci_low": null, "ci_high": null },
    "csi":            { "value": null, "ci_low": null, "ci_high": null },
    "ets":            { "value": null, "ci_low": null, "ci_high": null },
    "fss":            { "value": null, "ci_low": null, "ci_high": null },
    "frequency_bias": { "value": null, "ci_low": null, "ci_high": null }
  },
  "paired_difference": { "ets": { "value": null, "ci_low": null, "ci_high": null } },
  "n_samples": null,
  "n_events": null,
  "bootstrap": { "block_days": 7, "resamples": 2000 },
  "model_version_id": null
}
```

---

# 17. Fallbacks and Quality Flags

## 17.1 The ladder [Decision]

| Situation | What is served | `fallback_reason` |
|---|---|---|
| Normal | B3 | none |
| Regime features not available for this run | B2, with `regime_available = false` | `REGIME_UNAVAILABLE` |
| Model files not available | B1 | `ML_UNAVAILABLE` |
| Run-level `ood_flag = true` | B1 | `OOD_INPUT` |
| A cell has `extrapolation_flag = true` | Raw forecast **for that cell** | `EXTRAPOLATION` |
| Checks failed, or no correction exists | Raw forecast | `VALIDATION_FAILED` or `NO_CORRECTION` |

## 17.2 Rules [Decision]

- Every stored result has: `product_type` (`raw_nwp`, `quantile_mapping`, `global_ml`, `regime_aware_ml`), `fallback_used`, `fallback_reason`, `regime_available`, `ood_flag`, `extrapolation_flag`.
- **A raw forecast is never labelled as AI-corrected.** The screen shows "Showing raw NWP ({reason})".
- **In a fallback product, the range and the probabilities are empty** (null). The district's attention level is `UNAVAILABLE`. Cells in fallback are not hotspot candidates.
- If `regime_available = false`, F3 and F4 show "Not available yet" for that run.
- Low regime confidence is not a failure (§11.3).
- The share of fallback results is written to `pipeline_events` and shown on the Model Information page.

---

# 18. Backend and API

## 18.1 Owner and order of work

**Owner: M1**, starting when gate G2 is passed (§24). Until then **M5 builds a mock API** from the contracts below, so the frontend is never blocked.

## 18.2 Rules [Decision]

- **Stack:** Python, FastAPI, Uvicorn, Pydantic 2, SQLAlchemy 2 with `psycopg` 3, PostgreSQL 14 or newer (local).
- **Phase 1:** the API reads district-level data from files (Parquet or JSON) through one service layer.
- **Phase 2:** the same service layer reads district-level data from PostgreSQL. The API contract does not change.
- **Cell-level data always stays in Parquet** (D16). The API reads it with `pyarrow`.
- The API **never trains or runs a model**, and there are **no endpoints that change data**. Data is loaded by scripts.
- Secrets are only in environment variables (§22).

## 18.3 Conventions

- Base URL `/api/v1`. All answers are JSON. Times are ISO 8601 UTC. Rain is in mm. Probabilities are 0 to 1.
- Every answer carries `run_id`, `mode` (always `"replay"`), `evaluation_set` (`"development"` or `"holdout"`), and where relevant `model_version` and `flags` (§17).
- **Mock answers carry `"_mock": true`.** The frontend then shows the MOCK DATA banner (rule H7).
- `forecast_id` = `{run_id}_L{lead_day}_{district_id}`.
- Error format:

```json
{ "error": { "code": "FORECAST_NOT_FOUND", "message": "Forecast run was not found." } }
```

- Status codes: 400 wrong request, 404 not found, 422 wrong parameter format, 500 server error, 503 needed data not available.

## 18.4 Endpoints (all required)

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status, version |
| `GET /runs`, `GET /runs/{run_id}` | Replay dates (start date, end date, evaluation set), run details, alignment method, quality flags |
| `GET /forecasts/districts` | District table for a run and lead (filter by state, attention level) |
| `GET /forecasts/districts/{district_id}` | One district for a run and lead |
| `GET /forecasts/grid?variable=` | Cell layer. Variables: `raw`, `corrected`, `difference`, `observed`, `improvement`, `q10`, `q50`, `q90`, `p_ge_15_6`, `p_ge_64_5`, `p_ge_115_6`, `lps_influence`, `orographic_influence`, `coastal_influence` |
| `GET /forecasts/improvement-summary` | The counts for the F1 line (cells and districts where corrected is closer) |
| `GET /regime` | Domain-level regime object (§11.6), indicators A1–A6 with training percentiles |
| `GET /regime/transitions` | F4 series, events, and change numbers (optionally for one district) |
| `GET /analogs` | F3 analogs and outcomes for one district |
| `GET /forecasts/{forecast_id}/audit` | F2 audit trail |
| `GET /districts/priority` | F5 table |
| `GET /hotspots` | F6 hotspots |
| `GET /verification`, `GET /verification/reliability` | Scores with intervals; reliability data |
| `GET /model-info` | Versions, seasons, protocol status, limitations, fallback share |
| `GET /map/metadata` | Grid bounds, layers, dates, leads, thresholds, district file year |
| `GET /map/districts` | The simplified district outlines (GeoJSON), served once |

## 18.5 Example answers (all values are mock values)

**District forecast** (`GET /forecasts/districts`):

```json
{
  "_mock": true, "run_id": "tigge_ecmwf_cf_2024071500", "mode": "replay", "evaluation_set": "holdout",
  "lead_day": 1, "imd_date": "2024-07-16",
  "model_version": { "correction": "regime_xgb_v1", "heavy_rain": "hr_xgb_v1", "phase": "phase_lr_v1" },
  "districts": [{
    "district_id": "D001", "district_name": "Example District", "state": "Example State",
    "is_small": false, "product_type": "regime_aware_ml",
    "raw_mean_mm": 18.0, "corrected_mean_mm": 24.5, "observed_mean_mm": 27.0,
    "wettest_cell_mean_mm": 41.0, "wettest_cell_q10_mm": 12.0, "wettest_cell_q50_mm": 33.0, "wettest_cell_q90_mm": 78.0,
    "heavy_prob_max_cell": 0.22, "very_heavy_prob_max_cell": 0.05,
    "heavy_area_fraction_expected": 0.06, "very_heavy_area_fraction_expected": 0.01,
    "attention_level": "WATCH", "priority_rank": 37,
    "flags": { "fallback_used": false, "fallback_reason": null, "ood_flag": false, "extrapolation_flag": false }
  }]
}
```

**Audit trail** (`GET /forecasts/{forecast_id}/audit`):

```json
{
  "_mock": true, "forecast_id": "tigge_ecmwf_cf_2024071500_L1_D001", "evaluation_set": "holdout",
  "steps": {
    "raw": { "district_mean_mm": 18.0, "wettest_cell_mm": 33.0 },
    "regime": {
      "phase": { "active": 0.42, "normal": 0.46, "break": 0.12, "confidence_band": "low" },
      "nearest_lps": { "distance_km": 260, "bearing_deg": 235, "influence": 0.52, "settings": "tuned" },
      "orographic_influence": 0.88, "coastal_influence": 0.35
    },
    "history": { "phase": "normal", "lps_near": true, "n_dates": 14, "median_diff_mm": 3.1, "q25_diff_mm": -2.0, "q75_diff_mm": 9.4, "note": "few past cases" },
    "correction": { "district_mean_mm": 6.5, "wettest_cell_mm": 8.0 },
    "corrected": { "district_mean_mm": 24.5, "wettest_cell_mm": 41.0 },
    "confidence": {
      "heavy_prob_max_cell": 0.22, "very_heavy_prob_max_cell": 0.05,
      "range_wettest_cell_mm": { "q10": 12.0, "q50": 33.0, "q90": 78.0 },
      "measured_coverage_q10_q90": 0.78
    },
    "record": { "model_version": "regime_xgb_v1", "feature_set_version": "fs_v1", "alignment_method": "C1", "fallback_used": false }
  },
  "summary": "The model raised the district mean from 18.0 to 24.5 mm (+6.5). Most likely phase: normal (confidence low)."
}
```

**Analogs** (`GET /analogs`):

```json
{
  "_mock": true, "run_id": "tigge_ecmwf_cf_2024071500", "lead_day": 1, "district_id": "D001",
  "analogs": [{
    "rank": 1, "imd_date": "2019-07-28", "season": 2019,
    "distance": 1.31, "distance_percentile": 0.6,
    "observed_mean_mm": 22.0, "observed_wettest_cell_mm": 47.0,
    "raw_mean_mm": 15.0, "corrected_mean_mm": 21.0, "error_observed_minus_raw_mm": 7.0
  }],
  "median_error_observed_minus_raw_mm": 5.5, "n_analogs": 5
}
```

**Transitions** (`GET /regime/transitions`):

```json
{
  "_mock": true, "run_id": "tigge_ecmwf_cf_2024071500",
  "series": [{ "imd_date": "2024-07-14", "p_active": 0.20, "p_normal": 0.60, "p_break": 0.20, "lps_present": false }],
  "events": [{
    "event_type": "PHASE:normal->active", "event_date": "2024-07-17", "confirmed": false,
    "confidence": 0.58, "district_id": null,
    "domain_mean_corrected_change_mm": 3.2, "n_dates_before": 3, "n_dates_after": 1
  }]
}
```

**Hotspots** (`GET /hotspots`):

```json
{
  "_mock": true, "run_id": "tigge_ecmwf_cf_2024071500", "lead_day": 1, "threshold_mm": 64.5,
  "hotspots": [{
    "hotspot_id": 1, "n_cells": 6, "max_probability": 0.63, "max_corrected_mean_mm": 58.0,
    "centroid": { "lat": 17.4, "lon": 73.6 }, "district_ids": ["D001", "D002"],
    "outline": { "type": "Polygon", "coordinates": [] }
  }]
}
```

**Priority table** (`GET /districts/priority`): the same fields as the district forecast example, sorted by the rule in F5, with `attention_level`, `priority_rank` and the note `"Model-based attention level. Not an official warning."`.

**Verification** (`GET /verification`): the structure in §16.11. Query parameters: `forecast_type`, `comparison_to`, `evaluation_set`, `lead_day`, `threshold`, `neighbourhood`, `group_type`, `group_value`.

**Model information** (`GET /model-info`):

```json
{
  "_mock": true,
  "models": [{ "role": "correction_mean", "version": "regime_xgb_v1", "algorithm": "XGBoost Tweedie", "feature_set_version": "fs_v1", "n_features": 41 }],
  "data": {
    "nwp": "ECMWF-TIGGE-control", "truth": "IMD 0.25 degree", "alignment_method": "C1",
    "lead_days": [1, 2, 3], "season_scope": "JJAS", "district_file": "DataMeet, 2011 districts"
  },
  "protocol": { "n_seasons": null, "development_seasons": [], "holdout_seasons": [], "holdout_locked": false, "status": "DEVELOPMENT ONLY" },
  "limitations": [
    "Districts created after 2011 are not included.",
    "Western disturbances are not modelled (June-September only).",
    "Influence values are percentile or simple indices, not probabilities.",
    "The range is model-estimated, not guaranteed."
  ],
  "fallback_share_recent": null
}
```

---

# 19. Frontend

**Owner: M5 (one person builds the whole frontend).** The frontend only uses the API contracts. It never uses model files, GRIB files or passwords.

## 19.1 Stack [Decision]

React 18, TypeScript, Vite, React Router 6, Axios, Leaflet 1.9 with react-leaflet 4, Recharts. (Do not use React 19 or react-leaflet 5. They need each other and mixing versions breaks the build.)

## 19.2 Pages [Decision]

| Page | Contents |
|---|---|
| **1. Dashboard** | Top bar: replay date picker, lead selector, `REPLAY` badge, evaluation-set badge. Main map (Leaflet, cells drawn as rectangles on a canvas layer, district outlines on top). Layer buttons (F1). Split view (F1). Hotspot outlines (F6). Improvement line (F1). Side panel for the selected district. **District priority table** (F5) below the map. |
| **2. District Detail** | Tabs: **Forecast** (raw, corrected, observed, difference, range for the wettest cell, probabilities), **Audit trail** (F2), **Analogs** (F3), **Regime** (the district's regime numbers). |
| **3. Regime and Transitions** | Phase probabilities by lead (bar chart), low-pressure centres on a map, indicators A1–A6 against training percentiles, transitions timeline and event list (F4). |
| **4. Verification** | Raw, quantile mapping, global ML and regime-aware ML side by side, with intervals and `n_events`. Filters: lead, threshold, neighbourhood size, region, phase, month. Charts of §16.9. |
| **5. Model Information** | Model versions, seasons used, protocol status, limits (including the 2011 district note), fallback share. |

## 19.3 Map drawing rules [Decision]

- Cells: about 4,500 rectangles on a Leaflet canvas renderer (`preferCanvas: true`).
- Districts: one GeoJSON file, simplified so that it is under 10 MB **[Assumption]**, served once.
- **Base map [Decision]:** the district outlines are the base map and always work without internet. An OpenStreetMap tile layer may be added underneath as an extra. If it is used, the screen must show "© OpenStreetMap contributors". If tiles fail to load, the map continues to work with the district outlines only, and no error is shown to the user. Nothing else on the page may depend on the tile layer.
- Colours: a colour-blind-safe rain scale with the same bins on every layer. `difference` and `improvement` use two-colour diverging scales.

## 19.4 Honesty rules on screen [Decision]

- Always show the `REPLAY` badge and the evaluation-set badge (rule H6).
- Show the **MOCK DATA** banner if any answer has `_mock: true` (H7).
- Show "Showing raw NWP ({reason})" for fallback cells and hide the corrected value there (§17).
- Show the note "Model-based attention level. Not an official warning." on the priority table and the district panel (H4).
- Show "Not enough events" instead of a score for small groups (§16.5).
- Show "Not available yet" for any feature without real data (H8).
- Show "One day only, not evidence" beside the improvement line (H9).
- Use the wording of §14.4 and §13.7 for districts and ranges.
- Show the `is_small` flag on district rows.

## 19.5 Order of work for M5 [Decision]

1. Mock API server from the contracts (§18), so everything below can be built at once.
2. Dashboard: map, layer buttons, date and lead pickers.
3. Split view, improvement layer and the improvement line (F1).
4. District panel and priority table (F5).
5. District Detail: Forecast tab, Audit trail (F2), Analogs (F3).
6. Hotspots layer (F6).
7. Regime and Transitions page (F4).
8. Verification page.
9. Model Information page.
10. Switch from mock API to the real API. Remove every mock.

## 19.6 API client

A folder `src/api/` with `client.ts`, `forecasts.ts`, `regime.ts`, `analogs.ts`, `verification.ts`, `model.ts`. Components never build URLs by hand.

## 19.7 States

`Loading…`, `No forecast available`, `Data unavailable`, `Showing raw NWP ({reason})`, `Not enough events`, `Not available yet`.

---

# 20. Storage

## 20.1 Parquet files (cell level and large tables) [Decision]

```text
data/golden/season=YYYY/                     aligned rows (§9.3)
data/features/                               feature tables (§9.6)
data/serving/grid/run_id=<id>/part.parquet   one file per run: all leads, all valid cells
data/verification/components/                stored counts and sums (§16.4)
```

Columns of the serving grid file: `cell_id`, `lead_day`, `raw_mm`, `corrected_mean_mm`, `q10_mm`, `q50_mm`, `q90_mm`, `p_ge_15_6`, `p_ge_64_5`, `p_ge_115_6`, `obs_mm` (empty if unknown), `lps_influence`, `orographic_influence`, `coastal_influence`, `product_type`, `fallback_reason`, `extrapolation_flag`, `model_version_id`.

## 20.2 PostgreSQL (district level, metadata, results) [Decision]

No PostGIS. District outlines are stored as GeoJSON text (JSONB).

```sql
CREATE TABLE districts (
  district_id TEXT PRIMARY KEY, name TEXT NOT NULL, state TEXT NOT NULL,
  geojson JSONB, centroid_lat REAL, centroid_lon REAL,
  n_effective_cells REAL, is_small BOOLEAN, source_year INT
);
CREATE TABLE grid_cells (
  cell_id INT PRIMARY KEY, latitude NUMERIC(6,3) NOT NULL, longitude NUMERIC(6,3) NOT NULL,
  is_valid BOOLEAN, elevation_m REAL, slope REAL, dist_coast_km REAL, region_code TEXT,
  UNIQUE (latitude, longitude)
);
CREATE TABLE cell_district_weights (
  cell_id INT REFERENCES grid_cells, district_id TEXT REFERENCES districts,
  area_weight REAL NOT NULL, is_main BOOLEAN,
  PRIMARY KEY (cell_id, district_id)
);
CREATE TABLE model_versions (
  model_version_id SERIAL PRIMARY KEY, name TEXT NOT NULL, model_role TEXT, algorithm TEXT,
  feature_set_version TEXT, training_seasons INT[], git_commit TEXT,
  settings JSONB, dev_scores JSONB, created_at TIMESTAMPTZ DEFAULT now(), is_active BOOLEAN DEFAULT FALSE
);
CREATE TABLE experiment_runs (
  experiment_id SERIAL PRIMARY KEY, evaluation_set TEXT NOT NULL,   -- development | holdout
  git_commit TEXT, config_hash TEXT, locked BOOLEAN DEFAULT FALSE, forced_rerun BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE nwp_runs (
  run_id TEXT PRIMARY KEY, source TEXT, initialization_time TIMESTAMPTZ NOT NULL, season INT,
  evaluation_set TEXT,                     -- development | holdout
  mode TEXT DEFAULT 'replay', alignment_method TEXT, alignment_offset_hours SMALLINT DEFAULT 0,
  imd_stamp TEXT, status TEXT, created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE regime_predictions (
  run_id TEXT REFERENCES nwp_runs, lead_day SMALLINT, imd_date DATE,
  active_probability REAL, normal_probability REAL, break_probability REAL,
  regime_confidence REAL, confidence_band TEXT, regime_source TEXT,   -- oof | final
  regime_available BOOLEAN, ood_flag BOOLEAN, indicators JSONB,
  domain_mean_raw_mm REAL, domain_mean_corrected_mm REAL, lps_settings TEXT,
  model_version_id INT REFERENCES model_versions,
  PRIMARY KEY (run_id, lead_day)
);
CREATE TABLE lps_detections (
  run_id TEXT, lead_day SMALLINT, lps_id SMALLINT, latitude REAL, longitude REAL,
  zeta_max REAL, mslp_min_hpa REAL, strength_percentile REAL,
  PRIMARY KEY (run_id, lead_day, lps_id)
);
CREATE TABLE district_forecasts (
  run_id TEXT, lead_day SMALLINT, district_id TEXT REFERENCES districts, imd_date DATE,
  raw_mean_mm REAL, corrected_mean_mm REAL, observed_mean_mm REAL,
  wettest_cell_id INT, wettest_cell_mean_mm REAL,
  wettest_cell_q10_mm REAL, wettest_cell_q50_mm REAL, wettest_cell_q90_mm REAL,
  heavy_prob_max_cell REAL, very_heavy_prob_max_cell REAL,
  heavy_area_fraction_expected REAL, very_heavy_area_fraction_expected REAL,
  attention_level TEXT, priority_rank INT, is_small BOOLEAN,
  product_type TEXT, fallback_used BOOLEAN, fallback_reason TEXT,
  model_version_id INT REFERENCES model_versions,
  PRIMARY KEY (run_id, lead_day, district_id)
);
CREATE TABLE hotspots (
  run_id TEXT, lead_day SMALLINT, hotspot_id INT, threshold_mm REAL,
  n_cells INT, max_probability REAL, max_corrected_mean_mm REAL,
  centroid_lat REAL, centroid_lon REAL, outline JSONB, district_ids TEXT[],
  PRIMARY KEY (run_id, lead_day, hotspot_id)
);
CREATE TABLE district_history (              -- past outcomes, used by analogs (F3)
  run_id TEXT, lead_day SMALLINT, district_id TEXT, imd_date DATE, season INT,
  observed_mean_mm REAL, observed_max_cell_mm REAL, raw_mean_mm REAL, corrected_mean_mm REAL,
  prediction_source TEXT,                    -- oof | final
  phase TEXT, lps_near BOOLEAN,
  PRIMARY KEY (run_id, lead_day, district_id)
);
CREATE TABLE bias_table (                    -- history shown in the audit trail (F2)
  district_id TEXT, lead_day SMALLINT, phase TEXT, lps_near BOOLEAN,
  season_excluded INT,                       -- NULL = all development seasons
  n_dates INT, median_diff_mm REAL, q25_diff_mm REAL, q75_diff_mm REAL
);
CREATE TABLE analog_results (
  run_id TEXT, lead_day SMALLINT, rank SMALLINT,
  analog_run_id TEXT, analog_imd_date DATE, analog_season INT,
  distance REAL, distance_percentile REAL,
  PRIMARY KEY (run_id, lead_day, rank)
);
CREATE TABLE transitions (
  run_id TEXT, event_id INT, event_type TEXT, from_state TEXT, to_state TEXT,
  event_date DATE, confirmed BOOLEAN, confidence REAL, district_id TEXT,
  domain_mean_corrected_change_mm REAL, n_dates_before INT, n_dates_after INT,
  PRIMARY KEY (run_id, event_id)
);
CREATE TABLE verification_results (
  result_id BIGSERIAL PRIMARY KEY, experiment_id INT REFERENCES experiment_runs,
  forecast_type TEXT, comparison_to TEXT, evaluation_set TEXT,
  lead_day SMALLINT, threshold_mm REAL, neighbourhood_cells SMALLINT,
  group_type TEXT, group_value TEXT, metric TEXT,
  value REAL, ci_low REAL, ci_high REAL, n_samples BIGINT, n_events BIGINT, undefined_reason TEXT,
  model_version_id INT REFERENCES model_versions
);
CREATE TABLE pipeline_events (
  event_id BIGSERIAL PRIMARY KEY, run_id TEXT, stage TEXT, level TEXT, message TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

**Indexes:** `district_forecasts (run_id, district_id)`, `district_history (district_id, lead_day, season)`, `verification_results (forecast_type, lead_day, metric)`, `bias_table (district_id, lead_day, season_excluded)`.

**Notes**

- `district_history.corrected_mean_mm` is an **out-of-fold** prediction for development seasons (`prediction_source = oof`) and a final-model prediction for holdout seasons (`final`).
- Analogs and the bias table use development seasons only (§10.4).

---

# 21. Team

Five people build. One of them builds the whole frontend. The sixth team member does not build code in this plan. Submission material is handled outside this document.

## 21.1 Who owns what

### M1 — Data pipeline, district products, backend

**Owns:** ECDS and IMD downloads (§7), `docs/data-status.md`, time and space alignment and tests T1–T8 (§8), static geography, ingestion, golden dataset, feature library (§9), district weights and district fields (§14), `district_history`, and — starting at gate G2 — the API, the database and the audit-trail assembly (§18, §20, F2).

**Must know:** GRIB and xarray, cumulative rain, regridding, IST and UTC, FastAPI, SQL.

**Done when:** gate G1 is passed; the feature library is used by everyone; all endpoints return real data.

### M2 — Regime engine, analogs, transitions

**Owns:** everything in §11 (labels, phase model, low-pressure detector and its tuning, coast and mountain indices, regime contract), F3 (analog finder) and the detection part of F4 (transitions).

**Must know:** active and break monsoon, low-pressure systems, vorticity, moisture flux, out-of-fold prediction.

**Done when:** labels pass the check in §11.2; phase probabilities exist out-of-fold for every development season; the regime contract validates; F3 and F4 work on real data.

### M3 — Rainfall models

**Owns:** B0–B3 and the settings search (§12), the probability and range models (§13.3, §13.7), extrapolation flag, model records, and writing the serving grid files (§20.1).

**Must know:** XGBoost (Tweedie, logistic and quantile objectives, monotone constraints), model saving and loading.

**Done when:** B0–B3 and the probability and range models produce out-of-fold and final predictions under the protocol; models are saved with `model_versions` records; training stops with an error if any training row has `regime_source = final`.

### M4 — Protocol, calibration, verification, priority and hotspots

**Owns:** the experiment protocol and holdout lock (§10), event counts and calibration (§13.2, §13.4), the coverage check (§13.7), the verification engine and statistics (§16), the bias table (F2), the improvement summary (F1), the priority rules (F5), and hotspot computation (F6).

**Must know:** contingency tables, calibration, spatial verification, the block bootstrap, train–validation–test separation.

**Done when:** `run_holdout.py` locks correctly; the verification tests in §23 pass; primary metrics and the decision rule are committed before the holdout run.

### M5 — Frontend

**Owns:** the whole frontend (§19) and the mock API used until the real API exists.

**Must know:** React, TypeScript, Leaflet, Recharts and the honesty rules (§19.4).

**Done when:** all pages of §19.2 show real data; F1, F5 and F6 screens and the screens for F2, F3, F4 work; no mock remains.

## 21.2 Independent review

| Work | Reviewer | What the reviewer checks |
|---|---|---|
| Alignment and golden dataset (M1) | M4 | Tests T1–T8, base rates, the lag-test result |
| Labels and regime features (M2) | M3 | No forbidden inputs (L2, L8), out-of-fold generation, label check |
| Training code (M3) | M4 | Forbidden-input list (§10.6), split logic, `regime_source` check |
| Verification code (M4) | M3 | Hand-computed example, FSS masks, edge cases |
| District fields (M1) | M4 | Formulas of §14.4, weights add up to 1 |
| Analogs and transitions (M2) | M1 | Rules of F3 and F4, no analog from the query's own season |
| API and mock API (M1, M5) | each other | Contract match, `_mock` behaviour |

## 21.3 Hand-offs

- The **golden dataset** is on the critical path. M1 delivers **one season first**, then all seasons. Others use the one-season dataset until G1.
- **Contracts are frozen at G0** (§9.7, §11.6, §16.11, §18). A change needs a note to everyone who uses it.
- M5 builds against the mock API from the first day.

---

# 22. Setup and Repository

## 22.1 Machines [Decision]

macOS or Linux. Windows users must use **WSL2 with Ubuntu**. CPU only. 8 GB RAM is enough (§10.11). One machine is named the **reference machine**. The final demo must be tested on it (gate G6).

## 22.2 Setup steps (every builder) [Decision]

1. Install **Miniforge** (conda-forge). Create the environment from `environment.yml` (below): `conda env create -f environment.yml`.
2. Install **PostgreSQL 14 or newer** locally (macOS: Homebrew or Postgres.app; Linux: package manager). Create a database `sih_rain` and a user `sih`.
3. Install **Node.js 20 LTS**.
4. Register on the **ECMWF Data Store (ECDS)**. From your profile page, copy the exact lines shown for the API into `~/.cdsapirc` (do not type them from memory; they change).
5. Clone the repository. Copy `.env.example` to `.env`.
6. Run the checks: `python -c "import xarray, cfgrib, xgboost, sklearn, fastapi; print(xgboost.__version__)"` (must print 2.0 or higher), connect to PostgreSQL, run `npm run dev`.

## 22.3 environment.yml [Decision]

```yaml
name: sih-rain
channels: [conda-forge]
dependencies:
  - python=3.11
  - numpy
  - pandas
  - xarray
  - scipy
  - scikit-learn
  - xgboost>=2.0
  - cfgrib
  - eccodes
  - pyarrow
  - geopandas
  - shapely
  - pyproj
  - matplotlib
  - fastapi
  - uvicorn
  - pydantic>=2
  - sqlalchemy>=2
  - pytest
  - pip
  - pip:
      - cdsapi
      - imdlib
      - "psycopg[binary]>=3"
```

## 22.4 Frontend packages [Decision]

`react@18`, `react-dom@18`, `react-router-dom@6`, `axios`, `leaflet@1.9`, `react-leaflet@4`, `recharts`, `typescript`, `vite`, `eslint`, and the `@types` packages for them.

## 22.5 Repository [Decision]

```text
SIH26080/
├── data_pipeline/      ingestion/  alignment/  features/  districts/
├── regime_engine/      labels/  phase/  lps/  local_context/  analogs/  transitions/
├── ml/                 baselines/  training/  inference/  models/
├── probability/        classifiers, calibrators, range models
├── protocol/           split logic, cross-fit runner, run_holdout.py, holdout.lock
├── verification/       metrics/  stats/  reports/  plots/
├── backend/            app/ (main.py, api/, schemas/, services/, db/)
├── frontend/           src/ (api/, components/, pages/, hooks/, types/, maps/, charts/)
├── config/             alignment.yaml  regime.yaml  protocol.yaml  thresholds.yaml
│                       districts.yaml  verification.yaml  regions.yaml
├── data/               (not in git) golden/  features/  serving/  verification/
├── database/           schema.sql
├── tests/
├── docs/               data-status.md  data-sources.md  event-counts.md  data-contracts.md
├── environment.yml
├── .env.example
└── README.md           how to run everything from a clean machine
```

`config/` holds every number marked [Assumption]. `data/` is not stored in git.

## 22.6 Environment variables

`DATABASE_URL`, `DATA_DIR`, `MODEL_DIR`, `MODEL_VERSION` (backend and pipeline) and `VITE_API_BASE_URL` (frontend). **No secret is stored in the frontend or in git.** The ECDS key lives in `~/.cdsapirc`.

## 22.7 Order in which the pipeline runs

1. Download (ECDS, IMD, districts) → 2. Check and align → 3. Golden dataset → 4. Features → 5. Labels and regime engine → 6. Protocol runner (LOSO) → 7. Calibration and range checks → 8. Final models → 9. Locked holdout run → 10. Results into Parquet and PostgreSQL → 11. Analogs, transitions, hotspots, priority, bias table → 12. API → 13. Frontend.

---

# 23. Testing

| Area | Required tests |
|---|---|
| **Alignment (M1)** | T1–T8 (§8.7). Missing-value handling. |
| **Features (M1)** | Training and result-making call the same functions. The feature table matches `feature_set_version`. **No feature column comes from observed rain** (a test compares column names and origin against the forbidden list). |
| **Labels (M2)** | The check in §11.2. A synthetic series with a known 3-day spell gives the expected labels. |
| **Regime (M2)** | Phase probabilities add up to 1. The detector finds a synthetic vortex and ignores noise. Percentile tables are applied unchanged to new data. Influence is 0 when the raw index is 0. |
| **Protocol (M4)** | `run_holdout.py` refuses a second run without `--force`. Training stops if any training row has `regime_source = final`. No statistic uses holdout seasons. |
| **Models (M3)** | Saving and loading. Output shape. Missing features. The monotone constraint holds on a sweep of `rain_mm` with everything else fixed. No negative corrected values. |
| **Probability (M4)** | `P(≥115.6) ≤ P(≥64.5) ≤ P(≥15.6)` in every cell. The calibrator is fitted on out-of-fold predictions only. Range values are sorted. |
| **Verification (M4, reviewed by M3)** | A hand-computed contingency example. FSS = 1 for identical fields. FSS of a shifted field matches a hand calculation. Sea and missing cells are excluded. **Zero-event cases give NULL, not 0.** The bootstrap gives the same answer with the same seed. |
| **Districts (M1)** | Weights add up to 1 per district after removing non-valid cells. The mean is exact. `heavy_area_fraction_expected` equals Σ w·p on a small example. |
| **F1 (M4, M5)** | The improvement sign is correct on a small example. The counts on screen equal the API counts. |
| **F2 (M1)** | All seven steps are present. Summary sentences contain only numbers from the data. |
| **F3 (M2)** | No analog is from the query's own season. Analogs are at least 5 days apart. K = 5 unless the library is too small. |
| **F4 (M2)** | Events follow the rules. A gap in the series gives no event across the gap. |
| **F5 (M4)** | Levels and sort order follow the rules. Fallback districts are `UNAVAILABLE`. |
| **F6 (M4)** | Touching cells form one hotspot. The threshold in use is stored. |
| **Backend (M1)** | Answers match the contracts. Errors use the standard format. No model is called during a request. |
| **Frontend (M5)** | Loading, empty and error states. MOCK banner for `_mock: true`. Fallback banner. "Not enough events" state. "Not available yet" state. |

---

# 24. Work Plan

The plan is in **gates**, not dates. A gate is passed only when **all** its exit conditions are true. Work on the next gate may start early, but the next gate cannot be declared passed before the previous one.

## 24.1 Right now (before the portal submission deadline)

Everyone can start at the same time:

- **All:** install the tools (§22.2). Read §2, §3 and §5.
- **M1:** ECDS account; checks CK2, CK3, CK9, CK12; **start the downloads immediately** (they queue on the server); download IMD 1981–2010 and the seasons.
- **M2:** write the labelling code and run the statistics check of §11.2. It needs only IMD rain, not TIGGE.
- **M3:** write the B1 and XGBoost training code and test it on **clearly labelled synthetic** arrays (allowed only for testing code, rule H7).
- **M4:** write `config/protocol.yaml` and the verification engine with the hand-computed tests.
- **M5:** build the mock API and the dashboard skeleton.

## 24.2 Gates

| Gate | Name | Exit conditions |
|---|---|---|
| **G0** | Setup | Every machine passes the checks in §22.2. Repository and `config/` files exist. Contracts are frozen (§9.7, §11.6, §16.11, §18). CK2, CK3, CK9, CK12 are done. Downloads have started. The mock API runs. |
| **G1** | Data ready (**data freeze**) | Downloads for the seasons are finished and checked (CK4, CK5). **N is fixed** and written in `docs/data-status.md`. Tests T1–T8 pass. The golden dataset covers all N seasons. The feature library version 1 exists. The valid-cell count is written down. **No season is added after G1.** |
| **G2** | Baselines and labels | B0 and B1 are scored by the verification engine with LOSO. Layer A labels are computed and checked (CK7). `docs/event-counts.md` is written. The protocol runner works and the holdout lock is tested with a dummy run. The API (phase 1) serves one real run. The dashboard shows real raw rain on the map. |
| **G3** | Models | The regime engine is complete (A, B, C), with out-of-fold phase probabilities. B2 and B3 have out-of-fold predictions. Probabilities and ranges are calibrated and checked on out-of-fold data. District products, `district_history` and the serving grid files exist for all runs. The API returns real data for forecasts, regime and verification (development results). |
| **G4** | Required features | F1–F6 work on real data in the API and the frontend. PostgreSQL phase 2 is in use. |
| **G5** | Evaluation | Configs are frozen and tagged. Final models are fitted. **The holdout run is done once** and locked. Verification results with intervals are stored. The claims are decided by §10.8. (If N ≤ 4, the LOSO results are the final report, labelled `DEVELOPMENT ONLY`.) |
| **G6** | Demo ready | The full path works **without internet from a clean start on the reference machine**, following the README. The tile base map is optional (§19.3); with the network switched off, the map must still show data layers and district outlines. No mock remains. Every honesty rule of §5 is visible on screen. The Model Information page lists all limits. A replay walk-through has been rehearsed. A backup copy exists (below). |
| **G7** | Finale | No new modelling. Only fixes, joining and rehearsal. |

## 24.3 Order of the required features inside G3–G4

Build in this order (features that need only district and cell results come first):

1. F5 (priority table) → 2. F1 (toggle, split view, improvement) → 3. F2 (audit trail) → 4. F6 (hotspots) → 5. F3 (analogs) → 6. F4 (transitions).

Each feature has a minimum version (§15). When a feature is not finished, its screen says "Not available yet" (rule H8). It never shows mock data.

## 24.4 Backup [Decision]

Before G6 and before the finale, make a backup of: `data/serving/`, `models/`, a `pg_dump` of the database, the built frontend, and the README. Keep it on two devices.

## 24.5 The finale (expected 36 hours, D20)

1. First hours: set up the finale machines from the README and the backup. Run the demo path.
2. Then: fix bugs and rehearse. **Do not start new modelling.**
3. If a check fails on the finale machines, use the reference machine's backup.

---

# 25. Risks

| ID | Risk | Effect | What we do | Owner |
|---|---|---|---|---|
| RK1 | TIGGE requests are slow or limited | Fewer seasons | Start at once (§24.1). CK2, CK12. If N ≤ 4, the rule in §10.2 applies. | M1 |
| RK2 | Wind, humidity or pressure fields are missing | No regime engine | §7.5 | M1 |
| RK3 | Time-alignment error | Every result is wrong | Tests T1–T8, the lag test | M1 |
| RK4 | Too few seasons | Weak claims | §10.2, wide intervals, honest labels | M4 |
| RK5 | Leakage | False good results | L1–L9, holdout lock, cross-review | M4 |
| RK6 | Too few heavy-rain events | No heavy-rain model | §13.2 | M4 |
| RK7 | B3 is not better than B2 | No regime claim | Decision rule §10.8. Report it as it is. | M3, M4 |
| RK8 | Phase labels are noisy | Weak phase model | §11.2 and §11.3 checks. Report it. | M2 |
| RK9 | IMD product changes between years | Inconsistent truth | Test T5 | M1 |
| RK10 | ECMWF model upgrades inside the archive | Changing errors | CK10, newest-season holdout | M4 |
| RK11 | Six required features with a small team | Unfinished features | Minimum versions, build order (§24.3), rule H8 | Team |
| RK12 | Late integration | Broken demo | Mock API, frozen contracts | M1, M5 |
| RK13 | Local installs fail on some machines (GRIB libraries, PostgreSQL) | Lost time | conda environment, WSL2, reference machine | Team |
| RK14 | Out of memory on 8 GB | Training fails | §10.11 (`train_cell_stride`) | M3 |
| RK15 | Outputs mistaken for warnings | Misuse | Rule H4, screen notes | M5 |
| RK16 | 2011 district boundaries are out of date | Some new districts missing | D7, note on Model Information page | M1 |
| RK17 | IMD under-measures extremes | Truth is uncertain for heavy rain | State it in the report (§7.1) | M4 |

---

# 26. Done Criteria

The project is done when gate G6 is passed **and** all of these are true.

**Data.** ECMWF forecasts and IMD rain are loaded for N seasons; T1–T8 pass; the golden dataset and the feature library exist; `docs/data-status.md` is complete.

**Protocol.** Splits follow §10.2. Phase probabilities in training rows are out-of-fold. The holdout ran once and is locked. Primary metrics and the decision rule were committed before it.

**Regime and models.** Labels pass §11.2. Layers A, B and C produce the contract of §11.6. B0, B1, B2 and B3 are produced, saved, loaded and versioned.

**Probabilities and range.** Calibrated probabilities exist for every threshold the event counts allow. The range coverage is measured and shown.

**Products.** Cell and district products follow §14.4. Fallbacks follow §17.

**Verification.** RMSE, POD, FAR, CSI, ETS, FSS (plus frequency bias, Brier score, reliability) are computed on held-out data with intervals, `n_events` and paired differences. The regime claim follows §10.8.

**Required features.** F1–F6 each meet their "done when" line in §15 on real data.

**Backend and frontend.** All endpoints of §18.4 return valid JSON. All pages of §19.2 show real data. All honesty rules of §5 are visible.

---

# 27. Glossary

| Word | Simple meaning |
|---|---|
| **NWP** | Numerical weather prediction: a computer weather forecast. |
| **ECMWF** | The European weather centre whose model we use. |
| **TIGGE** | An archive of past forecasts from many weather centres. |
| **IMD** | India Meteorological Department. Its gridded rain is our observed rain. |
| **Cell** | One 0.25° square (about 28 km × 28 km) of the map. |
| **Lead day** | How many days ahead: 1, 2 or 3. |
| **Forecast start** | The time the forecast run began (always 00 UTC here). |
| **IMD day** | A 24-hour period that ends at 08:30 IST (03:00 UTC). |
| **Cumulative** | A running total from the start of the forecast. |
| **Regime** | The weather situation: monsoon phase, low-pressure system, coast and mountain influence. |
| **Active / break monsoon** | Spells of much more / much less rain than usual over central India. |
| **Low-pressure system (LPS)** | A moving area of low pressure that can bring heavy rain (includes depressions). |
| **Vorticity** | How strongly the wind circulates. High values at 850 hPa mark low-pressure systems. |
| **hPa** | Pressure unit. 850 hPa is about 1.5 km up. |
| **MSLP (`msl`)** | Mean sea-level pressure. |
| **Wind shear** | Difference between the wind high up and low down. |
| **Orography** | Terrain height. |
| **Out-of-fold** | A prediction made by a model that did not see that season. |
| **LOSO** | Leave-one-season-out: each season is tested once by models trained on the other seasons. |
| **Holdout** | Newest seasons that are kept aside and used once at the end. |
| **Calibration** | Making probabilities honest: "30%" should happen about 30% of the time. |
| **Reliability diagram** | A chart that checks calibration. |
| **Quantile** | q10 means 10% of values are below it. q90 means 90% are below it. |
| **Coverage** | The share of observations that fall inside a range. |
| **POD** | Probability of detection: the share of real events that were forecast. |
| **FAR** | False alarm ratio: the share of forecast events that did not happen. |
| **CSI, ETS** | Scores that combine hits, misses and false alarms (ETS removes luck). |
| **FSS** | Fractions skill score: checks position skill over neighbourhoods. |
| **RMSE** | Root mean square error of rain amounts. |
| **Frequency bias** | How often events are forecast compared with how often they happen. |
| **Bootstrap** | Re-sampling blocks of days many times to see how uncertain a score is. |
| **XGBoost** | A machine-learning method that builds many small decision trees. |
| **Tweedie** | A model form suited to rain (many zeros and a long tail). |
| **Quantile mapping** | A simple correction that matches the forecast's rain distribution to the observed one. |
| **Parquet** | A compact file format for large tables. |
| **GeoJSON** | A text format for map shapes. |
| **Replay** | Showing stored past forecasts. |
| **Evaluation set** | `DEVELOPMENT` (out-of-fold) or `HOLDOUT` (final test). |
| **Mock data** | Made-up data used only to build screens. |

---

# Appendix A — Formulas

**Contingency counts for threshold t:** a = hits, b = false alarms, c = misses, d = correct negatives, n = a + b + c + d.

```text
POD            = a / (a + c)
FAR (ratio)    = b / (a + b)
CSI            = a / (a + b + c)
Frequency bias = (a + b) / (a + c)
a_ref          = (a + b)(a + c) / n
ETS            = (a − a_ref) / (a + b + c − a_ref)
Bias = mean(F − O)     MAE = mean(|F − O|)     RMSE = sqrt(mean((F − O)²))
```

**FSS** (0/1 fields; Pf and Po are the fractions of cells equal to 1 in each n × n window):

```text
FSS         = 1 − Σ(Pf − Po)² / ( ΣPf² + ΣPo² )      sums over all valid points and all dates
FSS_useful  = 0.5 + f0 / 2                              f0 = observed event fraction
```

**Probability and range:**

```text
Brier score = mean((p − y)²)              BSS = 1 − BS / BS_reference
Pinball loss (level α) = mean( max(α·(y − q), (α − 1)·(y − q)) )
Coverage = share of observations between q10 and q90       (target 0.80)
```

**Regime numbers:**

```text
regime_confidence = 1 − H / ln 3,     H = −Σ p_k ln p_k
lps_influence     = lps_strength × exp(−(distance_km / 600)²)
upslope_flux      = q850 × max(0, wind · grad_h)
onshore_flux      = q850 × max(0, wind · coast_normal) × exp(−dist_coast_km / 100)
influence         = 0 if raw index = 0, else percentile rank among positive training values
```

**Alignment, method C1** (start date I, lead k, hours after start):

```text
window_start = 24(k − 1) + 3          window_end = 24k + 3
cumulative(h) = straight-line interpolation between the two nearest 6-hourly steps (value at step 0 is 0)
window_rain   = max(0, cumulative(window_end) − cumulative(window_start))
IMD date      = I + k          (imd_stamp = end_date)
              = I + k − 1      (imd_stamp = start_date)
```

**Lag test:** for shift s in {−1, 0, +1}, average over dates the cell-by-cell correlation between the lead-1 window rain and the IMD field dated (I + 1 + s).

**District fields:** with weights w_i (add up to 1), main cells S = {i : w_i ≥ 0.05}:

```text
corrected_mean_mm            = Σ w_i · m_i
heavy_prob_max_cell          = max over S of p_i(64.5)
heavy_area_fraction_expected = Σ w_i · p_i(64.5)
n_effective_cells            = 1 / Σ w_i²
```

**Block bootstrap:** take blocks of L consecutive dates (L = 7) with replacement until the original number of dates is reached. Recompute the metric for each system from the stored components with the **same** blocks. Repeat 2,000 times. The 2.5th and 97.5th percentiles of the metric, and of the paired difference, are the 95% interval.

---

# Appendix B — Parameter Registry

Every value below is stored in `config/`. **It may be changed only using development seasons, never using the holdout.**

| Parameter | Value | Status | File |
|---|---|---|---|
| Label box | 18–28°N, 68–88°E | Decision | `regime.yaml` |
| Label base period | IMD 1981–2010 | Decision | `regime.yaml` |
| Climatology window | 31 days | Assumption | `regime.yaml` |
| Active / break rule | z > +1 / z < −1 for at least 3 days | Verified (published rule) | `regime.yaml` |
| Label check tolerances | 7 ± 3 days; 26% ± 10 points | Assumption | `regime.yaml` |
| Phase feature boxes A1–A6 | §11.3 | Assumption | `regime.yaml` |
| Phase model penalty `C` | one of 0.01, 0.1, 1, 10 (by LOSO) | Decision | `regime.yaml` |
| Confidence bands | High ≥ 0.50, Medium ≥ 0.20, else Low | Assumption | `regime.yaml` |
| Out-of-range flag | ≥ 2 of 6 phase features outside 1st–99th percentile | Assumption | `regime.yaml` |
| Low-pressure detector | `zeta_min` 1.5e-5, `dp_min` 2 hPa, σ 1.5° (start values); ring 500–800 km; merge 500 km | Assumption (tuned in §11.4) | `regime.yaml` |
| Tuning grid | `zeta_min` {1.0, 1.5, 2.0}e-5; `dp_min` {1, 2, 3}; σ {1.0°, 1.5°} | Decision | `regime.yaml` |
| Match distance | 300 km + 100 km × (lead − 1) | Assumption | `regime.yaml` |
| `lps_influence` scale | 600 km | Assumption | `regime.yaml` |
| Coast decay scale | 100 km | Assumption | `regime.yaml` |
| Influence flags | ≥ 0.90 | Assumption | `thresholds.yaml` |
| Rain thresholds | 15.6, 64.5, 115.6 mm | Decision | `thresholds.yaml` |
| Extrapolation limit | training 99.9th percentile of `rain_mm` | Assumption | `thresholds.yaml` |
| Settings search | 20 random combinations, 3 validation seasons | Decision | `protocol.yaml` |
| Minimum events (model, primary threshold) | 30 | Assumption | `protocol.yaml` |
| Isotonic minimum | 200 events | Assumption | `thresholds.yaml` |
| Coverage tolerance | 80% ± 10 points | Assumption | `verification.yaml` |
| Group minimum | 10 events | Assumption | `verification.yaml` |
| FSS neighbourhoods | 1, 3, 5, 9 cells; primary 5 | Assumption | `verification.yaml` |
| Bootstrap | 7-day blocks (also 3, 14), 2,000 resamples | Assumption | `verification.yaml` |
| Valid-cell rule | non-missing on ≥ 95% of JJAS days in 1981–2010 | Decision | `alignment.yaml` |
| `w_min` | 0.05 | Assumption | `districts.yaml` |
| Small district | `n_effective_cells < 4` | Assumption | `districts.yaml` |
| Attention levels | HIGH: p64.5 ≥ 0.50 or p115.6 ≥ 0.20; WATCH: p64.5 ≥ 0.20 or mean ≥ 15.6 | Assumption | `thresholds.yaml` |
| Hotspot rule | p ≥ 0.50 and corrected mean ≥ 15.6 | Assumption | `thresholds.yaml` |
| Analogs | K = 5, at least 5 days apart, same lead, other seasons only | Decision | `regime.yaml` |
| Transition smoothing and confirmation | 3-day mean; probability ≥ 0.5 on 2 dates | Assumption | `regime.yaml` |
| LPS event distance | 500 km | Assumption | `regime.yaml` |
| `train_cell_stride` | 1 (2 if out of memory) | Decision | `protocol.yaml` |

---

# Appendix C — Sources Checked on 21 September 2026

Items tagged **[Established]** in the text are standard knowledge and are not listed.

**Forecast and observation data**
- TIGGE parameter list (pressure-level fields, single-level fields, accumulations start at step 0), ECMWF: https://confluence.ecmwf.int/spaces/TIGGE/pages/40109884/Parameters
- TIGGE on the ECMWF Data Store: https://ecds.ecmwf.int/datasets/tigge-forecasts?tab=download
- Move of TIGGE to ECDS (21 Apr 2026): https://confluence.ecmwf.int/display/DAC/Decommissioning+of+ECMWF+Public+Datasets+Service
- TIGGE licence and 48-hour delay: https://ecds.ecmwf.int/licences/tigge-licence
- TIGGE rain in 6-hourly accumulations (hydrology study): https://www.tandfonline.com/doi/full/10.1080/02626667.2021.1982138
- IMD gridded rainfall page: https://www.imdpune.gov.in/Clim_Pred_LRF_New/Grided_Data_Download.html
- `imdlib` documentation (grid size, −999, real-time product): https://imdlib.readthedocs.io/en/latest/Usage.html
- IMD gauge counts over time (research paper): https://arxiv.org/pdf/2404.12419
- DataMeet community maps of India (districts): https://projects.datameet.org/maps/

**Regime rules and catalogues**
- Rajeevan, Gadgil & Bhate (2010), active and break spells: https://www.clivar.org/sites/default/files/documents/aamp/12_Rajeevan.pdf
- Core monsoon zone box: https://www.tropmet.res.in/erpas/files/active_break_selection.php
- Pai et al. (2016), long-term active/break days (as cited in the literature)
- Vishnu et al. (2020), low-pressure system tracks: doi:10.5281/zenodo.3890646
- IMD depression wind thresholds: https://rsmcnewdelhi.imd.gov.in/images/pdf/cyclone_science_plan.pdf

**Software**
- XGBoost quantile regression (`reg:quantileerror`, new in 2.0.0, quantiles can cross, use `hist`): https://xgboost.readthedocs.io/en/release_2.0.0/python/examples/quantile_regression.html

**SIH schedule**
- SIH 2026 timeline (third-party page; finale expected in December, 36 hours): https://blogs.reskilll.com/?p=236
- Past software finales of 36 hours: https://www.iitbbs.ac.in/wp-content/uploads/2025/01/Press-Release-Grand-Finale-of-Smart-India-Hackathon-begins.pdf
- Confirm dates and rules on the official SIH portal (check CK11).

**Methods (Established)**
Roberts & Lean (2008) for FSS and its useful-skill line. Diebold & Mariano (1995) for the equal-accuracy test.
