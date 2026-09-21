# Data sources

Owner: **M1**. One row per source in PRD section 7.1. Fill in the empty cells as each source is
downloaded. The URLs and the one licence given here are the ones the PRD states (7.1 and
Appendix C); everything else must be checked by whoever downloads the data.

| Source | URL | Licence | Year | Downloaded by | Date |
|---|---|---|---|---|---|
| TIGGE, ECMWF control forecast (via ECMWF Data Store) | https://ecds.ecmwf.int/datasets/tigge-forecasts?tab=download | CC BY 4.0 (PRD 7.1; licence page: https://ecds.ecmwf.int/licences/tigge-licence) | | | |
| IMD gridded daily rainfall, 0.25° (via `imdlib`) | https://www.imdpune.gov.in/Clim_Pred_LRF_New/Grided_Data_Download.html | | | | |
| District boundaries (DataMeet, 2011 census districts) | https://projects.datameet.org/maps/ | | | | |
| Low-pressure system catalogue (Vishnu et al. 2020, Zenodo) | https://doi.org/10.5281/zenodo.3890646 | | 1979–2019 (PRD 7.1) | | |
| Published active/break rule (Rajeevan et al. 2010) | https://www.clivar.org/sites/default/files/documents/aamp/12_Rajeevan.pdf | | 2010 | | |

## Notes to fill in

- **District file (CK9):** the file states its year and its number of districts. Write both here,
  and the licence in the table above.
  - Year stated by the file:
  - Number of districts stated by the file:
- **Two sources for one variable?** If two IMD products cover the same year, say which one each
  season uses in `docs/data-status.md` (test T5).
- Downloaded data is **never committed** (`data/` is git-ignored). The ECDS key lives in
  `~/.cdsapirc` and must never be committed.
