"""Helpers for the historical run in Google Colab (commit a57d824). Every check raises `PipelineStop` with the exact
failure; nothing continues silently and nothing is replaced by another dataset.

Repo checkout: /content/SIH26080 (override with env SIH_REPO). Drive: /content/drive/MyDrive/SIH26080 (env SIH_DRIVE).
`REPO/data` is a symlink to `DRIVE/data`, so every output under data/ persists on Drive.
"""

from __future__ import annotations

import calendar
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

COMMIT = "816fd3aaea7b27650505798a7b0f2f6f5678ff5c"
REPO = Path(os.environ.get("SIH_REPO", "/content/SIH26080"))
DRIVE = Path(os.environ.get("SIH_DRIVE", "/content/drive/MyDrive/SIH26080"))
DEV, HOLD = [2021, 2022, 2023, 2024], [2025]
SEASONS = DEV + HOLD
BASE = (1981, 2010)
IMD_YEARS = list(range(BASE[0], BASE[1] + 1)) + [2021, 2022, 2023, 2024]  # 2025 is the uploaded NetCDF
STATIC_GRIB = "data/tigge/2021/single_06.grib"  # kept after 2021 is processed: it is the orog/lsm source
TIGGE_TOL = 0.03  # GRIB messages have a fixed size, so a complete file is within 3% of the 2025 size of that month

# byte sizes of the complete 2025 files (fixed-size messages: same for any year with the same month length)
TIGGE_SIZES = {
    "tp_2025{m}.grib": {"06": 34183590, "07": 35323043, "08": 35323043, "09": 34183590},
    "single_{m}.grib": {"06": 68408730, "07": 70689021, "08": 70689021, "09": 68408730},
    "pressure_{m}.grib": {"06": 126293040, "07": 130502808, "08": 130502808, "09": 126293040},
}
N_RUNS = 30 + 31 + 31 + 30  # start dates 1 June - 30 September


if str(REPO) not in sys.path:  # a kernel started before `pip install -e .` cannot import the repo otherwise
    sys.path.insert(0, str(REPO))


class PipelineStop(Exception):
    """A check failed. The run must stop here and the message be reported as it is."""


def stop(msg: str) -> None:
    raise PipelineStop(msg)


# ---- shell -------------------------------------------------------------------------------------


def sh(cmd: str, log: str | None = None, cwd: Path | None = None) -> str:
    """Run a bash command, stream its output, append it to a log on Drive, and stop on a non-zero exit."""
    cwd = cwd or REPO
    logf = None
    if log:
        (DRIVE / "logs").mkdir(parents=True, exist_ok=True)
        logf = open(DRIVE / "logs" / log, "a")
        logf.write(f"\n$ {cmd}\n")
    print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(["bash", "-c", cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail: list[str] = []
    for line in p.stdout:
        print(line, end="", flush=True)
        tail = (tail + [line])[-40:]
        if logf:
            logf.write(line)
    code = p.wait()
    if logf:
        logf.close()
    if code != 0:
        stop(f"command failed (exit {code}): {cmd}\n--- last output ---\n" + "".join(tail))
    return "".join(tail)


def size(path: Path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return str(n)


# ---- phase 1: setup ---------------------------------------------------------------------------


def verify_setup() -> None:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        stop(f"repository is at {head}, expected {COMMIT}")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, capture_output=True, text=True).stdout
    if dirty.strip():
        stop(f"tracked files were modified after checkout:\n{dirty}")
    if not os.path.ismount("/content/drive") and "SIH_DRIVE" not in os.environ:
        stop("Google Drive is not mounted at /content/drive")
    link = REPO / "data"
    if not link.is_symlink() or Path(os.path.realpath(link)) != Path(os.path.realpath(DRIVE / "data")):
        stop(f"{link} is not a symlink to {DRIVE / 'data'}")
    import importlib

    for mod in ("numpy", "pandas", "xarray", "cfgrib", "eccodes", "xgboost", "sklearn", "geopandas", "pyarrow", "cdsapi", "imdlib", "yaml", "joblib"):
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001
            stop(f"cannot import {mod}: {type(exc).__name__}: {exc}")
    nc = REPO / "data/imd/RF25_ind2025_rfp25.nc"
    if not nc.is_file():
        stop(f"the 2025 IMD NetCDF is not on Drive: expected {DRIVE / 'data/imd/RF25_ind2025_rfp25.nc'}")
    print("setup OK:", head, "| data ->", os.path.realpath(link))


SETUP_TESTS = [
    "tests/test_ingestion_imd_netcdf.py", "tests/test_regions_and_mask.py", "tests/test_phase_model_logloss.py",
    "tests/test_fold_climatology.py", "tests/test_regime_pipeline.py", "tests/test_multiseason_wiring.py",
]


def run_setup_tests() -> None:
    sh("python -m pytest -q -p no:cacheprovider " + " ".join(SETUP_TESTS), log="setup_tests.log")


# ---- ECDS access (TIGGE moved to the ECMWF Data Store, API base https://ecds.ecmwf.int/api) ----------------

ECDS_HOST, ECDS_API_PATH = "ecds.ecmwf.int", "/api"
ECDS_URL = f"https://{ECDS_HOST}{ECDS_API_PATH}"


def normalize_ecds_url(url: str | None) -> str:
    """The one base URL the client must get: exactly `https://ecds.ecmwf.int/api`, no trailing slash.

    The client builds endpoints as `f"{url}/catalogue/..."`, so a base of `https://ecds.ecmwf.int/` gives the 404
    `https://ecds.ecmwf.int//catalogue/v1/messages` and `.../api/` gives `.../api//catalogue`. A secret with a
    trailing slash or without `/api` is corrected; any other host or path is refused (not guessed).
    """
    from urllib.parse import urlsplit

    u = (url or "").strip().strip("\"'").strip()
    if not u:
        stop("the ECDS_URL secret is empty")
    parts = urlsplit(u)
    path = parts.path.rstrip("/")
    if parts.scheme != "https" or parts.netloc.lower() != ECDS_HOST or parts.query or parts.fragment or path not in ("", ECDS_API_PATH):
        stop(f"ECDS_URL must be {ECDS_URL} (a trailing slash or a missing /api is corrected); got {u!r}")
    return ECDS_URL


def write_cdsapirc(url: str | None, key: str | None, path: str = "/root/.cdsapirc") -> str:
    """Write ~/.cdsapirc with the normalised URL. The key is written exactly as given (never printed, never changed)."""
    if not key:
        stop("the ECDS_KEY secret is empty")
    good = normalize_ecds_url(url)
    with open(path, "w") as f:
        f.write(f"url: {good}\nkey: {key}\n")
    os.chmod(path, 0o600)
    print(f"~/.cdsapirc written: url = {good} (key not shown)")
    return good


def _read_rc(path: str) -> tuple[str, str]:
    kv = dict(line.split(":", 1) for line in Path(path).expanduser().read_text().splitlines() if ":" in line)
    return kv.get("url", "").strip(), kv.get("key", "").strip()


def collect_tigge_requests(year: int) -> list[tuple[str, dict, str]]:
    """The exact `(dataset, request, target)` triples the two download scripts would submit for `year`, captured with a
    stub client (nothing is sent). The scientific request stays defined only in those scripts."""
    import types

    calls: list[tuple[str, dict, str]] = []

    class Stub:
        def retrieve(self, dataset, request, target):
            calls.append((dataset, request, target))

    fake = types.ModuleType("cdsapi")
    fake.Client = lambda *a, **k: Stub()
    import contextlib
    import io

    saved = sys.modules.get("cdsapi")
    sys.modules["cdsapi"] = fake
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # the scripts print "Downloading ..." as they go
            for script in ("download_2025_tp.py", "download_2025_atmos.py"):
                exec(compile((REPO / script).read_text().replace("2025", str(year)), script, "exec"), {"__name__": "stub"})
    finally:
        if saved is not None:
            sys.modules["cdsapi"] = saved
        else:
            del sys.modules["cdsapi"]
    return calls


def check_ecds_api(year: int = 2021, rc: str = "~/.cdsapirc") -> dict:
    """Metadata-only connectivity check (no job is submitted, nothing is downloaded): the URL in `rc` is the API base,
    the key authenticates, `tigge-forecasts` is in the catalogue and has a process, and the server accepts every real
    request of the download scripts for `year` (`estimate_costs` validates the full request like a submit would;
    `apply_constraints` cannot be used: it rejects non-constraint fields such as `grid` and `area`)."""
    from ecmwf.datastores import Client

    url, key = _read_rc(rc)
    if url != ECDS_URL:
        stop(f"{rc} has url {url!r}; it must be {ECDS_URL!r} (run write_cdsapirc)")
    client = Client(url=url, key=key)
    try:
        who = client.check_authentication()
        coll = client.get_collection("tigge-forecasts")
        client.get_process("tigge-forecasts")
    except Exception as exc:  # noqa: BLE001
        stop(f"ECDS API check failed: {type(exc).__name__}: {str(exc)[:300]}")
    reqs = collect_tigge_requests(year)
    if len(reqs) != 12:
        stop(f"the download scripts produce {len(reqs)} requests for {year}, expected 12 (4 months x rain, single-level, pressure-level)")
    bad, costs = [], []
    for dataset, request, target in reqs:
        try:
            out = client.estimate_costs(dataset, request)
            costs.append((Path(target).name, out.get("cost"), out.get("limit")))
            if out.get("cost") is not None and out.get("limit") is not None and out["cost"] > out["limit"]:
                bad.append(f"{Path(target).name}: cost {out['cost']} exceeds the limit {out['limit']}")
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{Path(target).name}: {type(exc).__name__}: {str(exc)[:200]}")
    if bad:
        stop("ECDS rejected the request of the download scripts: " + "; ".join(bad))
    info = {"url": url, "collection": coll.id, "requests_checked": len(costs), "max_cost_vs_limit": max((c[1], c[2]) for c in costs),
            "user_ok": bool(who)}
    print("ECDS API OK:", info)
    return info


# ---- phase 2: IMD -------------------------------------------------------------------------------


def _config():
    sys.path.insert(0, str(REPO))
    from data_pipeline.ingestion import load_config

    return load_config()


def check_imd_year(year: int, cfg=None) -> dict:
    """One IMD year: file present, grid and dates right, missing values handled (-999 -> NaN, nothing else changes)."""
    import numpy as np
    import pandas as pd

    from data_pipeline.ingestion.imd import imd_axes, read_imd_year, resolve_imd_path

    cfg = cfg or _config()
    path = resolve_imd_path(year, cfg.imd)
    if path is None:
        stop(f"IMD {year}: no file found (expected {cfg.imd.year_path(year)} or a NetCDF)")
    try:
        ds = read_imd_year(cfg.imd.year_path(year), year, cfg)
    except Exception as exc:  # noqa: BLE001
        stop(f"IMD {year}: {path.name} could not be read: {type(exc).__name__}: {exc}")
    rain = ds["rain"]
    days = 366 if calendar.isleap(year) else 365
    if rain.shape != (days, 129, 135):
        stop(f"IMD {year}: shape {rain.shape}, expected {(days, 129, 135)}")
    lat, lon = imd_axes(cfg.imd)
    if not (np.allclose(ds["lat"].values, lat) and np.allclose(ds["lon"].values, lon)):
        stop(f"IMD {year}: grid axes differ from 6.5-38.5N, 66.5-100E at 0.25 degrees")
    t = pd.DatetimeIndex(ds["time"].values)
    if t[0] != pd.Timestamp(year, 1, 1) or t[-1] != pd.Timestamp(year, 12, 31) or not (np.diff(t.values) == np.timedelta64(1, "D")).all():
        stop(f"IMD {year}: dates are not consecutive from 1 Jan to 31 Dec")
    values = rain.values
    n_nan = int(np.isnan(values).sum())
    if path.suffix.lower() == ".grd":
        raw = np.fromfile(path, dtype="float32")
        if int((raw == -999).sum()) != n_nan or not np.array_equal(np.isnan(values).ravel(), raw == -999):
            stop(f"IMD {year}: NaN cells are not exactly the -999 cells of the file")
        if not np.array_equal(values.ravel()[~np.isnan(values).ravel()], raw[raw != -999]):
            stop(f"IMD {year}: non-missing values changed while reading")
    if (values == -999).any() or (values[~np.isnan(values)] < 0).any():
        stop(f"IMD {year}: -999 or negative rain remain after missing-value handling")
    if n_nan == values.size or np.isnan(values).mean() < 0.3 or np.isnan(values).mean() > 0.9:
        stop(f"IMD {year}: missing fraction {np.isnan(values).mean():.3f} is outside 0.3-0.9 (India land is about 0.29 of the grid)")
    return {"year": year, "file": path.name, "days": days, "nan_frac": round(float(np.isnan(values).mean()), 4),
            "max_mm": round(float(np.nanmax(values)), 1)}


def download_imd_years(years: list[int]) -> None:
    sh(f"python -c \"from data_pipeline.ingestion import download_imd; download_imd({sorted(years)})\"", log="imd_download.log")


def verify_imd(years: list[int]) -> list[dict]:
    cfg = _config()
    rows = [check_imd_year(y, cfg) for y in years]
    for r in rows:
        print(r)
    fracs = [r["nan_frac"] for r in rows]
    print(f"IMD OK: {len(rows)} years, missing fraction {min(fracs)}-{max(fracs)}")
    return rows


def build_mask() -> dict:
    sh("python scripts/build_golden.py mask", log="mask.log")
    from data_pipeline.alignment import load_valid_cells

    cfg = _config()
    grid = load_valid_cells(cfg)
    n = int(grid["is_valid"].sum())
    base = str(grid.attrs.get("base_years"))
    if base != str(list(BASE)):
        stop(f"mask base years {base}, expected {list(BASE)}")
    if not 3000 <= n <= 7000:  # PRD 7 expects about 4,000-5,000; far outside means the IMD data is not what we think
        stop(f"{n} valid cells is far from the PRD's expectation of about 4,000-5,000")
    print(f"mask OK: {n} valid cells, base years {base}")
    return {"n_valid": n, "base_years": base}


# ---- phase 3: TIGGE -----------------------------------------------------------------------------


def expected_tigge_files(year: int) -> dict[str, int]:
    out = {}
    for pattern, by_month in TIGGE_SIZES.items():
        for m, nbytes in by_month.items():
            out[pattern.format(m=m).replace("2025", str(year))] = nbytes
    return out


def check_tigge_files(year: int) -> None:
    """All 12 files exist and are complete: GRIB messages are fixed size, so size tells a truncated download."""
    d = REPO / "data/tigge" / str(year)
    bad = []
    for name, ref in expected_tigge_files(year).items():
        f = d / name
        if not f.is_file():
            bad.append(f"{name}: missing")
        elif abs(f.stat().st_size - ref) > TIGGE_TOL * ref:
            bad.append(f"{name}: {f.stat().st_size:,} bytes, expected about {ref:,}")
    if bad:
        stop(f"TIGGE {year} is incomplete: " + "; ".join(bad))


def download_tigge(year: int) -> None:
    """The proven 2025 request scripts with 2025 replaced by `year`. A .complete marker means all 12 requests finished;
    without it any leftover files are removed first (an interrupted download leaves a truncated file)."""
    d = REPO / "data/tigge" / str(year)
    marker = d / ".complete"
    if marker.exists():
        check_tigge_files(year)
        print(f"TIGGE {year}: already complete")
        return
    check_ecds_api(year)  # metadata only: stops here, before any file is touched, if the URL, key or request is wrong
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for script in ("download_2025_tp.py", "download_2025_atmos.py"):
        tmp = Path(f"/content/{year}_{script}")
        tmp.write_text((REPO / script).read_text().replace("2025", str(year)))
        sh(f"python {tmp}", log=f"tigge_{year}.log")
    check_tigge_files(year)
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


# ---- phase 4: golden + features + regime fields --------------------------------------------------


def verify_golden(year: int, n_valid: int | None = None) -> dict:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    cfg = _config()
    files = sorted((cfg.alignment.golden_dir / f"season_{year}").glob("golden_*.parquet"))
    if len(files) != 4:
        stop(f"golden {year}: {len(files)} files, expected 4 (Jun-Sep)")
    if n_valid is None:
        from data_pipeline.alignment import load_valid_cells

        g = load_valid_cells(cfg)
        n_valid = int(g["is_valid"].sum())
    rows, runs, missing, dmin, dmax = 0, set(), 0, None, None
    for f in files:
        t = pq.read_table(f, columns=["run_id", "obs_missing", "imd_date", "rain_mm"])
        rows += t.num_rows
        runs |= set(pc.unique(t["run_id"]).to_pylist())
        missing += int(pc.sum(t["obs_missing"]).as_py() or 0)
        lo, hi = pc.min(t["imd_date"]).as_py(), pc.max(t["imd_date"]).as_py()
        dmin, dmax = min(dmin or lo, lo), max(dmax or hi, hi)
        if pc.min(t["rain_mm"]).as_py() < 0 or pc.max(t["rain_mm"]).as_py() >= 1000:
            stop(f"golden {year}: rain_mm outside [0, 1000) in {f.name}")
    if len(runs) != N_RUNS:
        stop(f"golden {year}: {len(runs)} forecast runs, expected {N_RUNS} (a TIGGE day is missing)")
    if rows != N_RUNS * 3 * n_valid:
        stop(f"golden {year}: {rows:,} rows, expected {N_RUNS * 3 * n_valid:,} ({N_RUNS} runs x 3 leads x {n_valid} cells)")
    share = missing / rows
    if share > 0.05:  # CK5: IMD must cover the valid cells
        stop(f"golden {year}: {share:.1%} of rows have no IMD observation (limit 5%)")
    return {"rows": rows, "runs": len(runs), "imd_dates": f"{dmin}..{dmax}", "obs_missing_share": round(share, 5)}


def verify_features(year: int, golden_rows: int) -> dict:
    import pyarrow.parquet as pq

    cfg = _config()
    files = sorted((cfg.features.dir / "cell_features" / f"season_{year}").glob("cell_features_*.parquet"))
    if len(files) != 4:
        stop(f"features {year}: {len(files)} files, expected 4")
    rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    if rows != golden_rows:
        stop(f"features {year}: {rows:,} rows but golden has {golden_rows:,}")
    from data_pipeline.features import FEATURE_SCHEMA

    for f in files:
        if pq.read_schema(f).names != FEATURE_SCHEMA.names:
            stop(f"features {year}: columns of {f.name} differ from FEATURE_SCHEMA")
    t = pq.read_table(files[0], columns=["clim_mean", "clim_p95", "rain_mm"])
    if t["clim_mean"].null_count or t["clim_p95"].null_count or t["rain_mm"].null_count:
        stop(f"features {year}: empty clim_mean / clim_p95 / rain_mm values")
    return {"rows": rows}


def verify_fields(year: int) -> dict:
    import numpy as np
    import xarray as xr

    path = REPO / "data/regime/fields" / f"fields_{year}.nc"
    if not path.is_file():
        stop(f"regime fields {year}: {path} was not written")
    ds = xr.load_dataset(path)
    want = {"init_time": N_RUNS, "lead_day": 3, "lat": 129, "lon": 135}
    if dict(ds.sizes) != want:
        stop(f"regime fields {year}: sizes {dict(ds.sizes)}, expected {want}")
    for v in ds.data_vars:
        if not np.isfinite(ds[v].values).all():
            stop(f"regime fields {year}: non-finite values in {v}")
    return {"vars": list(ds.data_vars), "file": human(size(path))}


def delete_raw_tigge(year: int) -> None:
    """Remove the season's raw GRIB files. 2021/single_06.grib stays: it is the orog/lsm source of every later step."""
    d = REPO / "data/tigge" / str(year)
    for f in d.iterdir():
        if year == 2021 and f.name == "single_06.grib":
            continue
        f.unlink()
    if year != 2021:
        d.rmdir()


def season_size(year: int) -> int:
    return (size(REPO / "data/golden" / f"season_{year}") + size(REPO / "data/features/cell_features" / f"season_{year}")
            + size(REPO / "data/regime/fields" / f"fields_{year}.nc"))


def print_report(rep: dict) -> None:
    yn = lambda v: "YES" if v is True else ("NO" if v is False else str(v))  # noqa: E731
    print(f"\nSEASON {rep['season']}:\n"
          f"  TIGGE downloaded: {yn(rep['tigge'])}\n  IMD available: {yn(rep['imd'])}\n"
          f"  Golden: {rep['golden']}\n  Features: {rep['features']}\n  Regime fields: {rep['fields']}\n"
          f"  Persistent size: {rep['size']}\n  Raw TIGGE deleted: {yn(rep['deleted'])}\n", flush=True)


def process_season(year: int) -> dict:
    """One season, start to finish. Stops (raises) at the first failure; raw TIGGE is deleted only after everything passed."""
    rep = {"season": year, "tigge": False, "imd": False, "golden": "not run", "features": "not run",
           "fields": "not run", "size": "-", "deleted": False}
    step = "start"
    try:
        step = "imd"
        check_imd_year(year)
        rep["imd"] = True
        step = "tigge"
        download_tigge(year)
        rep["tigge"] = True
        step = "golden"
        sh(f"python scripts/build_golden.py season {year} --tigge-dir data/tigge --static-grib {STATIC_GRIB} "
           f"--tp-tolerance-mm 0.02 --clip-max-share 0.02", log=f"golden_{year}.log")
        built = [s for s in SEASONS if (REPO / "data/golden" / f"season_{s}").exists()]
        sh("python scripts/build_golden.py check", log=f"golden_{year}.log")  # strict: PRD mask, same cells everywhere
        g = verify_golden(year)
        print("golden", year, g)
        rep["golden"] = f"PASS ({g['rows']:,} rows, mask consistent across seasons {built})"
        step = "features"
        sh(f"python scripts/build_features.py season {year} --train {' '.join(map(str, DEV))} --holdout {' '.join(map(str, HOLD))} "
           f"--static-grib {STATIC_GRIB} --skip-districts", log=f"features_{year}.log")
        f = verify_features(year, g["rows"])
        rep["features"] = f"PASS ({f['rows']:,} rows)"
        step = "fields"
        sh(f"python scripts/build_regime.py fields {year} --tigge-dir data/tigge", log=f"fields_{year}.log")
        fl = verify_fields(year)
        rep["fields"] = f"PASS ({fl['file']})"
        rep["size"] = human(season_size(year))
        step = "delete"
        delete_raw_tigge(year)
        rep["deleted"] = True
    except Exception:
        key = {"golden": "golden", "features": "features", "fields": "fields"}.get(step)
        if key:
            rep[key] = "FAIL"
        print_report(rep)
        raise
    print_report(rep)
    return rep


# ---- phase 5: regime ---------------------------------------------------------------------------


def _label_gate(chk: dict, accept: str | None, out_dir: Path) -> None:
    """The PRD 11.2 label check is a gate. It only opens for a failed check if a written reason is given, and then the
    reason, the numbers and the diagnostics are saved next to the labels. Thresholds and the rule are never changed."""
    if chk.get("passed"):
        return
    if not accept:
        stop("label check FAILED (PRD 11.2): " + json.dumps(chk) + " -- do not change the rule; check the box, climatology and code"
             " (h.diagnose_labels()). To proceed anyway, pass accept_label_failure='<reason>' to regime_phase().")
    if len(accept.strip()) < 30:
        stop("accept_label_failure needs a real reason (at least 30 characters) that says what was checked")
    diag_path = out_dir / "label_diagnostics.json"
    record = {"accepted_at": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": COMMIT, "reason": accept.strip(), "check": chk,
              "diagnostics": json.loads(diag_path.read_text()) if diag_path.is_file() else "label_diagnostics.json not found"}
    (out_dir / "label_check_override.json").write_text(json.dumps(record, indent=2))
    print("LABEL CHECK FAILED BUT ACCEPTED (recorded in label_check_override.json):", accept.strip())


def regime_phase(accept_label_failure: str | None = None) -> dict:
    sh(f"python scripts/build_regime.py labels --base {BASE[0]} {BASE[1]} --seasons {' '.join(map(str, SEASONS))}", log="regime_labels.log")
    chk = json.loads((REPO / "data/regime/label_check.json").read_text())
    print("label check:", json.dumps(chk, indent=2))
    _label_gate(chk, accept_label_failure, REPO / "data/regime")
    missing = [s for s in SEASONS if not (REPO / "data/regime/fields" / f"fields_{s}.nc").is_file()]
    if missing:
        stop(f"regime fields missing for seasons {missing}")
    sh(f"python scripts/build_regime.py oof --dev {' '.join(map(str, DEV))} --holdout {' '.join(map(str, HOLD))} --static-grib {STATIC_GRIB}",
       log="regime_oof.log")
    return verify_regime()


def verify_regime() -> dict:
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    cfg = _config()  # also puts the repo on sys.path, before the imports below
    from data_pipeline.alignment import check_mask_consistency
    from regime_engine.contract import REGIME_14_FEATURES

    check_mask_consistency(cfg, seasons=SEASONS)
    out = {}
    for s in SEASONS:
        f = REPO / "data/regime/regime_features" / f"season_{s}.parquet"
        feat_rows = sum(pq.ParquetFile(p).metadata.num_rows
                        for p in (cfg.features.dir / "cell_features" / f"season_{s}").glob("cell_features_*.parquet"))
        t = pq.read_table(f, columns=["regime_source", *REGIME_14_FEATURES]).to_pandas()
        want = "oof" if s in DEV else "final"
        if len(t) != feat_rows:
            stop(f"regime {s}: {len(t):,} rows but features have {feat_rows:,}")
        if set(t["regime_source"]) != {want}:
            stop(f"regime {s}: regime_source {set(t['regime_source'])}, expected {want}")
        if not np.isfinite(t[REGIME_14_FEATURES].to_numpy()).all():
            stop(f"regime {s}: non-finite regime features")
        out[s] = {"rows": len(t), "source": want}
    dom = pd.read_parquet(REPO / "data/regime/regime_domain.parquet")
    if len(dom) != len(SEASONS) * N_RUNS * 3:
        stop(f"regime domain table has {len(dom)} rows, expected {len(SEASONS) * N_RUNS * 3}")
    summ = json.loads((REPO / "ml/models/regime/regime_summary.json").read_text())
    if summ["dev_seasons"] != DEV or summ["holdout_seasons"] != HOLD:
        stop(f"regime summary seasons {summ['dev_seasons']}/{summ['holdout_seasons']} are not {DEV}/{HOLD}")
    print("regime OK:", out, "| phase model C =", summ["best_c"], "| folds:", summ["folds"])
    return {"seasons": out, "best_c": summ["best_c"], "folds": summ["folds"]}


def _runs(mask) -> list[int]:
    """Lengths of the runs of consecutive True values."""
    import itertools

    return [len(list(g)) for k, g in itertools.groupby(mask) if k]


def diagnose_labels(cfg=None, west_edges: tuple = (65.0,), box_variants: bool = True) -> dict:
    """READ-ONLY diagnosis of a failed July-August label check (PRD 11.2: check the box, the climatology and the code;
    do not change the rule). Nothing in the pipeline, the config or the saved labels is changed.

    1. code check: z-scores and spells are re-derived from `labels.parquet` with a separate implementation and compared;
    2. climatology check: z statistics of July-August, and the lowest z a dry day can reach (a break needs z < -1);
    3. the seasons without a break: how near each one is to having one;
    4. sensitivity, reported only and never adopted: spells detected inside July-August only, and other western box edges.
    """
    import numpy as np
    import pandas as pd

    from regime_engine import pipeline as rp

    cfg = cfg or _config()
    rc = rp.load_regime_config()
    la = rc["layer_a"]
    lab = pd.read_parquet(rp.regime_dir(cfg) / "labels.parquet")
    base = lab[lab["is_base"]]
    half = int(la["climatology_window_days"]) // 2
    zmin, zmax, run = la["break_z_max"], la["active_z_min"], int(la["min_run_days"])
    out: dict = {"base_years": [int(base["year"].min()), int(base["year"].max())], "n_base_years": int(base["year"].nunique())}

    # 1. independent re-derivation (window from the saved labels: 1 Jun - 3 Oct covers every July-August window)
    doy = base.index.dayofyear.to_numpy()
    rain = base["core_rain_mm"].to_numpy(float)
    ja = np.flatnonzero(base.index.month.isin([7, 8]))
    z_ind = np.empty(len(ja))
    clim = {}
    for j, i in enumerate(ja):
        d = doy[i]
        if d not in clim:
            pool = rain[(np.abs(doy - d) <= half) & ~np.isnan(rain)]
            clim[d] = (pool.mean(), max(pool.std(ddof=1), 1e-6))
        z_ind[j] = (rain[i] - clim[d][0]) / clim[d][1]
    dz = float(np.nanmax(np.abs(z_ind - base["z_score"].to_numpy()[ja])))
    out["z_recomputed_max_abs_diff"] = dz
    mism = 0
    for y, g in base.groupby("year"):
        z = g["z_score"].to_numpy()
        lab_ind = np.array(["normal"] * len(z), dtype=object)
        for name, cond in (("active", z > zmax), ("break", z < zmin)):
            i = 0
            while i < len(z):
                if cond[i]:
                    k = i
                    while k < len(z) and cond[k]:
                        k += 1
                    if k - i >= run:
                        lab_ind[i:k] = name
                    i = k
                else:
                    i += 1
        m = g.index.month.isin([7, 8])
        mism += int((lab_ind[m] != g["phase_label"].to_numpy()[m]).sum())
    out["spell_label_mismatches_vs_independent"] = mism
    print(f"[code] z recomputed independently: max |diff| = {dz:.2e}; spell labels differing from an independent run-length labelling: {mism}")

    # 2. climatology / distribution of z in July-August
    jab = base.iloc[ja]
    zz = jab["z_score"].to_numpy()
    lowest = np.array([-(m / sd) for m, sd in clim.values()])
    out["ja_z"] = {"mean": float(np.nanmean(zz)), "sd": float(np.nanstd(zz)), "share_z_below_-1": float((zz < zmin).mean()),
                   "share_z_above_+1": float((zz > zmax).mean()), "lowest_reachable_z_min": float(lowest.min()),
                   "lowest_reachable_z_median": float(np.median(lowest))}
    print(f"[z] Jul-Aug z: mean {np.nanmean(zz):+.2f}, sd {np.nanstd(zz):.2f}; days z<-1: {(zz < zmin).mean():.1%}, days z>+1: {(zz > zmax).mean():.1%}; "
          f"z of a completely dry day = {np.median(lowest):+.2f} (median over days; a break day needs z < {zmin:g}, so the closer this is to {zmin:g} the fewer days can qualify)")

    # 3. per-season table and the no-break seasons
    rows = []
    for y, g in jab.groupby("year"):
        below = (g["z_score"] < zmin).to_numpy()
        rows.append({"year": int(y), "active_days": int((g["phase_label"] == "active").sum()), "break_days": int((g["phase_label"] == "break").sum()),
                     "days_z<-1": int(below.sum()), "longest_z<-1_run": max(_runs(below), default=0), "lowest_z": round(float(g["z_score"].min()), 2)})
    tab = pd.DataFrame(rows).set_index("year")
    nob = tab[tab["break_days"] == 0].sort_values("longest_z<-1_run", ascending=False)
    out["no_break_years"] = nob.reset_index().to_dict("records")
    out["n_no_break"] = int(len(nob))
    out["n_no_break_with_a_2day_run"] = int((nob["longest_z<-1_run"] == 2).sum())
    print(f"[seasons] no break in {len(nob)} of {len(tab)} seasons (limit for the check: {int(np.floor(0.36 * len(tab)))}); closest to having one:")
    print(nob.to_string())

    # 4a. spells inside July-August only (no run may start before 1 July or continue after 31 August)
    nb_win = 0
    for y, g in jab.groupby("year"):
        below = (g["z_score"] < zmin).to_numpy()
        nb_win += int(max(_runs(below), default=0) < run)
    out["sensitivity_spells_within_JulAug"] = {"no_break_seasons": nb_win, "share": nb_win / len(tab)}
    print(f"[sens] spells detected inside 1 Jul-31 Aug only: {nb_win} of {len(tab)} seasons without a break ({nb_win / len(tab):.1%})")

    # 4b. other western box edges (reads the 30 IMD years again; the cached 68E series is not touched: cache=False)
    if box_variants:
        import copy

        out["sensitivity_west_edge"] = {}
        base_years = sorted(int(y) for y in base["year"].unique())
        for w in west_edges:
            rc2 = copy.deepcopy(rc)
            rc2["layer_a"]["label_box"]["lon_min"] = float(w)
            _, rep = rp.build_phase_labels(cfg, base_years, [], rc2, cache=False)
            out["sensitivity_west_edge"][str(w)] = rep
            print(f"[sens] west edge {w:g}E (report only): active {rep['mean_active_days']:.2f}, break {rep['mean_break_days']:.2f}, "
                  f"no-break share {rep['no_break_share']:.3f} -> check {'PASS' if rep['passed'] else 'FAIL'}")
    path = rp.regime_dir(cfg) / "label_diagnostics.json"
    path.write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("written:", path)
    return out


def final_report() -> None:
    (DRIVE / "models").mkdir(exist_ok=True)
    if (REPO / "ml/models/regime").exists():
        shutil.copytree(REPO / "ml/models/regime", DRIVE / "models/regime", dirs_exist_ok=True)
    print("PERSISTENT STORAGE (Drive):")
    total = 0
    for p in sorted((DRIVE).iterdir()):
        if p.is_dir():
            for q in sorted(p.iterdir()):
                n = size(q)
                total += n
                print(f"  {q.relative_to(DRIVE)}: {human(n)}")
    print(f"  TOTAL: {human(total)}")
    ov = REPO / "data/regime/label_check_override.json"
    print("LABEL CHECK OVERRIDE:", ("ACCEPTED, reason: " + json.loads(ov.read_text())["reason"]) if ov.is_file() else "none (the check passed)")
    print("ARTIFACTS:")
    for pat in ("data/golden/grid_cells.parquet", "data/golden/season_*/*.parquet", "data/features/cell_features/season_*/*.parquet",
                "data/features/static/*", "data/features/climatology/*", "data/regime/labels.parquet", "data/regime/label_check.json",
                "data/regime/fields/*.nc", "data/regime/regime_features/*.parquet", "data/regime/regime_domain.parquet"):
        files = sorted(DRIVE.glob(pat))
        print(f"  {pat}: {len(files)} files, {human(sum(size(f) for f in files))}")
    print("  models/regime/: ", [p.name for p in (DRIVE / 'models/regime').glob('*')])
