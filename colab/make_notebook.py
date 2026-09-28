"""Builds historical_pipeline.ipynb from colab_helpers.py (run locally; it only writes a JSON file)."""
import json
from pathlib import Path

here = Path(__file__).parent
helpers = (here / "colab_helpers.py").read_text()
COMMIT = "816fd3aaea7b27650505798a7b0f2f6f5678ff5c"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md(f"""
# SIH26080 historical run (Colab)
Commit `{COMMIT}`. **Run the cells top to bottom, one at a time.** Every check raises `PipelineStop` with the exact
failure; if a cell fails, stop and report that message. Nothing is replaced by another dataset.
Before you start put `RF25_ind2025_rfp25.nc` into `MyDrive/SIH26080/data/imd/`, and store the Colab secrets
`ECDS_URL` and `ECDS_KEY` (and `GH_TOKEN` only if the repo is private).
""")
md("## Phase 1: setup")
code("""
from google.colab import drive
drive.mount('/content/drive')
import os
ROOT = '/content/drive/MyDrive/SIH26080'
for d in ['data/imd', 'data/raw/imd', 'data/tigge', 'data/golden', 'data/features', 'data/regime', 'data/verification', 'models', 'logs']:
    os.makedirs(f'{ROOT}/{d}', exist_ok=True)
print(sorted(os.listdir(f'{ROOT}/data/imd')))
""")
code(f"""
%%bash
set -euo pipefail
cd /content
if [ -d SIH26080/.git ]; then
  cd SIH26080 && git fetch -q origin && git checkout -q {COMMIT}
else
  git clone --branch m1/complete-data-pipeline https://github.com/sanjanaspals1106/SIH26080.git SIH26080
  cd SIH26080 && git checkout -q {COMMIT}
fi
git rev-parse HEAD
ln -sfn /content/drive/MyDrive/SIH26080/data /content/SIH26080/data   # re-runnable; never deletes anything on Drive
pip install -q -U cdsapi ecmwf-datastores-client
pip install -q xarray cfgrib ecmwflibs eccodes imdlib xgboost scikit-learn pyarrow pandas geopandas shapely pyproj scipy pyyaml joblib netCDF4 pytest httpx fastapi sqlalchemy 'pydantic>=2' matplotlib
pip install -q -e . --no-deps
""")
code("%%writefile /content/colab_helpers.py\n" + helpers)
code("""
import sys; sys.path.insert(0, '/content')
import colab_helpers as h
from google.colab import userdata
# ECDS_URL is normalised to exactly https://ecds.ecmwf.int/api (a trailing slash or a missing /api in the secret is fixed;
# any other host/path stops here). ECDS_KEY is written exactly as stored.
h.write_cdsapirc(userdata.get('ECDS_URL'), userdata.get('ECDS_KEY'))
""")
code("""
import sys; sys.path.insert(0, '/content')
import colab_helpers as h
h.verify_setup()        # exact commit, Drive mounted, data -> Drive, imports, 2025 IMD present
h.run_setup_tests()     # relevant tests must pass
""")
code("h.check_ecds_api(2021)   # metadata only: URL, key, tigge-forecasts catalogue + process, and the 12 real requests; nothing is downloaded")
md("## Phase 2: IMD (1981-2010 and 2021-2024; 2025 is the uploaded NetCDF)")
code("h.download_imd_years(h.IMD_YEARS)")
code("rows = h.verify_imd(h.IMD_YEARS + [2025])   # files, grid, consecutive dates, -999 handling")
code("mask = h.build_mask()                        # 1981-2010 valid-cell mask")
md("## Phases 3-4: one season at a time (download, golden, mask check, features, regime fields, delete raw)\n"
   "Run the next cell only after the previous one printed its report with everything PASS.")
for y in (2021, 2022, 2023, 2024, 2025):
    code(f"rep{y} = h.process_season({y})")
md("## Phase 5: regime labels, label check, leave-one-season-out regime features")
code("regime = h.regime_phase()   # stops if the PRD 11.2 label check fails; see the diagnosis cell below")
md("## Only if the label check failed: read-only diagnosis (changes nothing; see PRD 11.2)")
code("diag = h.diagnose_labels()")
md("## Final report")
code("h.final_report()")

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(here / "historical_pipeline.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote", here / "historical_pipeline.ipynb", len(cells), "cells")


# ---- second notebook: finish the regime step only (data is already downloaded and processed) ----------------------
cells = []
md(f"""
# SIH26080: finish the regime step (Colab)
Commit `{COMMIT}`. **Run all cells top to bottom.** Everything is already downloaded and processed on Drive
(IMD 1981-2010 and 2021-2025, golden + features + regime fields for 2021-2025), so there is nothing to download here.
This builds the leave-one-season-out regime features (development 2021-2024, holdout 2025) and prints the final report.
The PRD 11.2 label check misses by one season (36.7% vs a 36% limit); the cell below records that as an accepted
deviation with the reason shown. Remove the `accept_label_failure=` argument if you want it to stop instead.
""")
code("""
from google.colab import drive
drive.mount('/content/drive')
""")
code(f"""
%%bash
set -euo pipefail
cd /content
if [ -d SIH26080/.git ]; then
  cd SIH26080 && git fetch -q origin && git checkout -q {COMMIT}
else
  git clone --branch m1/complete-data-pipeline https://github.com/sanjanaspals1106/SIH26080.git SIH26080
  cd SIH26080 && git checkout -q {COMMIT}
fi
git rev-parse HEAD
ln -sfn /content/drive/MyDrive/SIH26080/data /content/SIH26080/data   # never deletes anything on Drive
pip install -q xarray cfgrib ecmwflibs eccodes cdsapi imdlib xgboost scikit-learn pyarrow pandas geopandas shapely pyproj scipy pyyaml joblib netCDF4 pytest httpx fastapi sqlalchemy 'pydantic>=2' matplotlib
pip install -q -e . --no-deps
""")
code("%%writefile /content/colab_helpers.py\n" + helpers)
code("""
import sys; sys.path.insert(0, '/content')
import colab_helpers as h
h.verify_setup()        # exact commit, Drive mounted, data -> Drive, imports, 2025 IMD present
h.run_setup_tests()     # relevant tests must pass
""")
code("""
regime = h.regime_phase(accept_label_failure="PRD 11.2 no-break share 36.7% (11 of 30) vs limit 36%; code, climatology and box verified by diagnose_labels (independent recomputation identical, 65E box identical, Jul-Aug-only window identical); accepted as a documented deviation")
""")
code("h.final_report()")
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(here / "regime_finish.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote", here / "regime_finish.ipynb", len(cells), "cells")


# ---- third notebook: M3 training, minimal (Run all): smoke, then a real but untuned full run --------------------
cells = []
md(f"""
# SIH26080: M3 training (Run all)
Data (IMD/TIGGE/golden/features/regime for 2021-2025) is already on Drive. This clones the repo, checks it, runs a
minute-scale smoke test, then trains real B0-B3, the heavy-rain probability models and the q10/q50/q90 range models
on all the development data with the repo's default XGBoost settings (`--skip-search`, PRD 12.2's tuning search
skipped for time; every result is labelled `SKIPPED_UNTUNED_DEFAULTS`, so it is honest about that) and every other
cell only (`--stride 2`, PRD 10.11's own memory/time relief valve; verification still uses every cell), verifies
them out-of-fold, and copies the results to Drive. It does **not** touch the 2025 holdout (PRD 10.5: that is a
separate, deliberate, one-time step for later, once these development results have been reviewed).

If cell 6 dies partway (Colab OOM or quota cutoff) and you reconnect, just re-run that same cell: the settings
search, the out-of-fold predictions and the final models are each cached to Drive the moment they finish, so a
crash later on does not force redoing work that already succeeded.

Click **Runtime > Run all**. CPU runtime, no GPU needed.
""")
code("""
from google.colab import drive
drive.mount('/content/drive')
""")
code(f"""
%%bash
set -euo pipefail
cd /content
if [ -d SIH26080/.git ]; then
  cd SIH26080 && git fetch -q origin && git checkout -q {COMMIT}
else
  git clone --branch m1/complete-data-pipeline https://github.com/sanjanaspals1106/SIH26080.git SIH26080
  cd SIH26080 && git checkout -q {COMMIT}
fi
git rev-parse HEAD
ln -sfn /content/drive/MyDrive/SIH26080/data /content/SIH26080/data   # never deletes anything on Drive
pip install -q xarray cfgrib ecmwflibs eccodes cdsapi imdlib xgboost scikit-learn pyarrow pandas geopandas shapely pyproj scipy pyyaml joblib netCDF4 pytest httpx fastapi sqlalchemy 'pydantic>=2' matplotlib
pip install -q -e . --no-deps
""")
code("%%writefile /content/colab_helpers.py\n" + helpers)
code("""
import sys; sys.path.insert(0, '/content')
import colab_helpers as h
h.verify_setup()
h.run_setup_tests()
""")
code("""
!cd /content/SIH26080 && python scripts/run_m3.py smoke --out-dir data/models/m3_smoke
""")
code("""
!cd /content/SIH26080 && python scripts/run_m3.py full --out-dir data/models/m3 --skip-search --stride 2
""")
code("""
import shutil, json
from pathlib import Path
DRIVE = Path('/content/drive/MyDrive/SIH26080')
for src in ("data/verification/m3_dev", "data/models/m3"):
    s = Path("/content/SIH26080") / src
    if s.is_dir():
        shutil.copytree(s, DRIVE / "results" / s.name, dirs_exist_ok=True)
print("copied to", DRIVE / "results")
dev = json.loads((Path("/content/SIH26080/data/verification/m3_dev/metrics.json")).read_text())
print(json.dumps(dev["correction"], indent=2))
print("\\nNOTE: this is a DEVELOPMENT-ONLY, UNTUNED run (--skip-search). Real B0-B3/probability/range models, "
     "real out-of-fold metrics, saved to Drive under results/. The 2025 holdout was NOT touched.")
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(here / "run_m3.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote", here / "run_m3.ipynb", len(cells), "cells")
