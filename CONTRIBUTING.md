# Contributing

Five people build this project (M1 to M5). The full rules are in `docs/PRD.md`; this page is the
short version.

## Branches and pull requests

- Branch names: `m<N>/<short-topic>`, for example `m1/alignment-tests` or `m5/dashboard-map`.
- **No direct commits to `main`.** Everything goes through a pull request.
- **Every PR needs one reviewer** from the table below (PRD section 21.2). The PR template has a
  checklist; fill it in.
- Never commit data (`data/`, `*.grib`, `*.nc`), model files, `.env`, or your `~/.cdsapirc` key.
- Do not look at holdout results and then change anything. That burns the holdout (PRD 10.5).

## Independent review (PRD section 21.2)

| Work | Reviewer | What the reviewer checks |
|---|---|---|
| Alignment and golden dataset (M1) | M4 | Tests T1–T8, base rates, the lag-test result |
| Labels and regime features (M2) | M3 | No forbidden inputs (L2, L8), out-of-fold generation, label check |
| Training code (M3) | M4 | Forbidden-input list (§10.6), split logic, `regime_source` check |
| Verification code (M4) | M3 | Hand-computed example, FSS masks, edge cases |
| District fields (M1) | M4 | Formulas of §14.4, weights add up to 1 |
| Analogs and transitions (M2) | M1 | Rules of F3 and F4, no analog from the query's own season |
| API and mock API (M1, M5) | each other | Contract match, `_mock` behaviour |

## Contracts

Contracts are frozen at gate G0 (PRD 21.3): the internal data contract (§9.7), the regime output
(§11.6), the verification output (§16.11) and the API (§18). A change to any of them needs
**a message to everyone who uses it** before the PR is merged. Update `docs/data-contracts.md` and
`docs/PRD.md` in the same PR. The database schema (`database/schema.sql`) is treated the same way.

## Forbidden inputs (PRD section 10.6)

A code review rejects any change that breaks one of these rules: L1 to L9 (listed in the PR
template).
