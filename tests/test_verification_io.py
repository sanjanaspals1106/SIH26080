"""Unit tests for verification component persistence (PRD §16.4 and §20.1).

Required tests:
1. write/read round-trip
2. None values preserved
3. multiple component types round-trip
4. filtering works
5. duplicate identity rejected
6. replace=True replaces matching record
7. append preserves existing different records
8. deterministic schema
9. incompatible/corrupt schema fails clearly
10. empty input handled safely
"""

from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from verification.components import VerificationComponent
from verification.io import (
    COMPONENT_PYARROW_SCHEMA,
    SCHEMA_COLUMNS,
    DuplicateComponentError,
    IncompatibleSchemaError,
    get_component_identity,
    load_components,
    save_components,
)


def _create_sample_component(
    imd_date="2024-07-15",
    lead_day=1,
    forecast_type="regime_aware_ml",
    evaluation_set="development",
    threshold_mm=15.6,
    neighbourhood_cells=None,
    group_type="all",
    group_value="all",
    n_samples=100,
    n_events=12,
    sum_error=5.5,
    sum_abs_error=12.0,
    sum_squared_error=45.0,
    a=10,
    b=2,
    c=3,
    d=85,
    fss_numerator_sum=0.0,
    fss_forecast_fraction_sq_sum=0.0,
    fss_observed_fraction_sq_sum=0.0,
    brier_n=100,
    sum_brier_terms=8.5,
    pinball_q10_sum=1.2,
    pinball_q50_sum=2.4,
    pinball_q90_sum=1.8,
    range_n=100,
    coverage_count=80,
) -> VerificationComponent:
    return VerificationComponent(
        imd_date=imd_date,
        lead_day=lead_day,
        forecast_type=forecast_type,
        evaluation_set=evaluation_set,
        threshold_mm=threshold_mm,
        neighbourhood_cells=neighbourhood_cells,
        group_type=group_type,
        group_value=group_value,
        n_samples=n_samples,
        n_events=n_events,
        n=n_samples,
        sum_error=sum_error,
        sum_abs_error=sum_abs_error,
        sum_squared_error=sum_squared_error,
        a=a,
        b=b,
        c=c,
        d=d,
        fss_numerator_sum=fss_numerator_sum,
        fss_forecast_fraction_sq_sum=fss_forecast_fraction_sq_sum,
        fss_observed_fraction_sq_sum=fss_observed_fraction_sq_sum,
        brier_n=brier_n,
        sum_brier_terms=sum_brier_terms,
        pinball_q10_sum=pinball_q10_sum,
        pinball_q50_sum=pinball_q50_sum,
        pinball_q90_sum=pinball_q90_sum,
        range_n=range_n,
        coverage_count=coverage_count,
    )


def test_1_write_read_roundtrip(tmp_path):
    """Test 1: Write VerificationComponent records and read them back, ensuring exact field equality."""
    c1 = _create_sample_component(lead_day=1, threshold_mm=15.6)
    c2 = _create_sample_component(lead_day=2, threshold_mm=64.5, sum_error=-3.2)

    saved_count = save_components([c1, c2], target=tmp_path)
    assert saved_count == 2

    loaded = load_components(target=tmp_path)
    assert len(loaded) == 2

    # Map by identity to verify fields
    loaded_map = {get_component_identity(c): c for c in loaded}

    id1 = get_component_identity(c1)
    id2 = get_component_identity(c2)
    assert id1 in loaded_map
    assert id2 in loaded_map

    lc1 = loaded_map[id1]
    assert lc1.imd_date == c1.imd_date
    assert lc1.lead_day == c1.lead_day
    assert lc1.forecast_type == c1.forecast_type
    assert lc1.evaluation_set == c1.evaluation_set
    assert lc1.threshold_mm == pytest.approx(c1.threshold_mm)
    assert lc1.neighbourhood_cells == c1.neighbourhood_cells
    assert lc1.group_type == c1.group_type
    assert lc1.group_value == c1.group_value
    assert lc1.n_samples == c1.n_samples
    assert lc1.n_events == c1.n_events
    assert lc1.sum_error == pytest.approx(c1.sum_error)
    assert lc1.sum_squared_error == pytest.approx(c1.sum_squared_error)
    assert lc1.a == c1.a
    assert lc1.b == c1.b
    assert lc1.c == c1.c
    assert lc1.d == c1.d
    assert lc1.coverage_count == c1.coverage_count


def test_2_none_values_preserved(tmp_path):
    """Test 2: Verify None values in imd_date, threshold_mm, neighbourhood_cells, n_events round-trip as None."""
    comp = VerificationComponent(
        imd_date=None,
        lead_day=1,
        forecast_type="raw_nwp",
        evaluation_set="development",
        threshold_mm=None,
        neighbourhood_cells=None,
        group_type="all",
        group_value="all",
        n_samples=50,
        n_events=None,
        n=50,
        sum_error=1.5,
    )

    save_components([comp], target=tmp_path)
    loaded = load_components(target=tmp_path)

    assert len(loaded) == 1
    lc = loaded[0]

    assert lc.imd_date is None
    assert lc.threshold_mm is None
    assert lc.neighbourhood_cells is None
    assert lc.n_events is None
    assert lc.n_samples == 50
    assert lc.sum_error == pytest.approx(1.5)


def test_3_multiple_component_types_roundtrip(tmp_path):
    """Test 3: Continuous, categorical, and spatial FSS components round-trip together."""
    # 1. Continuous
    cont = VerificationComponent(
        imd_date="2024-07-20",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=None,
        neighbourhood_cells=None,
        group_type="all",
        group_value="all",
        n_samples=100,
        n_events=None,
        n=100,
        sum_error=4.0,
        sum_squared_error=25.0,
    )
    # 2. Categorical
    cat = VerificationComponent(
        imd_date="2024-07-20",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=15.6,
        neighbourhood_cells=None,
        group_type="all",
        group_value="all",
        n_samples=100,
        n_events=15,
        a=12,
        b=3,
        c=2,
        d=83,
    )
    # 3. Spatial FSS
    fss = VerificationComponent(
        imd_date="2024-07-20",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=64.5,
        neighbourhood_cells=5,
        group_type="all",
        group_value="all",
        n_samples=100,
        n_events=15,
        fss_numerator_sum=1.2,
        fss_forecast_fraction_sq_sum=3.4,
        fss_observed_fraction_sq_sum=3.2,
    )

    save_components([cont, cat, fss], target=tmp_path)
    loaded = load_components(target=tmp_path)

    assert len(loaded) == 3

    # Check that each type preserves its signature fields
    cont_matches = [c for c in loaded if c.threshold_mm is None and c.neighbourhood_cells is None]
    cat_matches = [c for c in loaded if c.threshold_mm == 15.6 and c.neighbourhood_cells is None]
    fss_matches = [c for c in loaded if c.threshold_mm == 64.5 and c.neighbourhood_cells == 5]

    assert len(cont_matches) == 1
    assert len(cat_matches) == 1
    assert len(fss_matches) == 1

    assert cont_matches[0].sum_error == pytest.approx(4.0)
    assert cat_matches[0].a == 12
    assert cat_matches[0].n_events == 15
    assert fss_matches[0].fss_numerator_sum == pytest.approx(1.2)


def test_4_filtering_works(tmp_path):
    """Test 4: Verify load_components filters by each specified metadata dimension."""
    c_dev_lead1 = _create_sample_component(
        forecast_type="raw_nwp", evaluation_set="development", lead_day=1, threshold_mm=15.6
    )
    c_dev_lead2 = _create_sample_component(
        forecast_type="raw_nwp", evaluation_set="development", lead_day=2, threshold_mm=15.6
    )
    c_hold_lead1 = _create_sample_component(
        forecast_type="regime_aware_ml", evaluation_set="holdout", lead_day=1, threshold_mm=64.5
    )
    c_continuous = _create_sample_component(
        forecast_type="regime_aware_ml", evaluation_set="holdout", lead_day=1, threshold_mm=None
    )
    c_fss = _create_sample_component(
        forecast_type="regime_aware_ml", evaluation_set="holdout", lead_day=1, threshold_mm=64.5, neighbourhood_cells=5
    )
    c_region = _create_sample_component(
        forecast_type="regime_aware_ml", evaluation_set="holdout", lead_day=1, threshold_mm=64.5,
        group_type="region_code", group_value="WEST_COAST"
    )

    save_components(
        [c_dev_lead1, c_dev_lead2, c_hold_lead1, c_continuous, c_fss, c_region],
        target=tmp_path,
    )

    # Filter by forecast_type
    res_ft = load_components(target=tmp_path, forecast_type="raw_nwp")
    assert len(res_ft) == 2
    assert all(c.forecast_type == "raw_nwp" for c in res_ft)

    # Filter by evaluation_set
    res_ev = load_components(target=tmp_path, evaluation_set="holdout")
    assert len(res_ev) == 4
    assert all(c.evaluation_set == "holdout" for c in res_ev)

    # Filter by lead_day
    res_lead = load_components(target=tmp_path, lead_day=2)
    assert len(res_lead) == 1
    assert res_lead[0].lead_day == 2

    # Filter by threshold_mm (continuous where threshold_mm is None)
    res_cont = load_components(target=tmp_path, threshold_mm=None)
    assert len(res_cont) == 1
    assert res_cont[0].threshold_mm is None

    # Filter by threshold_mm (specific value)
    res_thresh = load_components(target=tmp_path, threshold_mm=15.6)
    assert len(res_thresh) == 2
    assert all(c.threshold_mm == pytest.approx(15.6) for c in res_thresh)

    # Filter by neighbourhood_cells
    res_fss = load_components(target=tmp_path, neighbourhood_cells=5)
    assert len(res_fss) == 1
    assert res_fss[0].neighbourhood_cells == 5

    # Filter by group_type & group_value
    res_grp = load_components(target=tmp_path, group_type="region_code", group_value="WEST_COAST")
    assert len(res_grp) == 1
    assert res_grp[0].group_value == "WEST_COAST"


def test_5_duplicate_identity_rejected(tmp_path):
    """Test 5: Duplicate component identity raises DuplicateComponentError unless replace=True."""
    c1 = _create_sample_component(lead_day=1, threshold_mm=15.6, sum_error=10.0)
    c2_dup = _create_sample_component(lead_day=1, threshold_mm=15.6, sum_error=20.0)

    # Duplicate within same batch
    with pytest.raises(DuplicateComponentError, match="Duplicate component identity"):
        save_components([c1, c2_dup], target=tmp_path, replace=False)

    # Save c1 first
    save_components([c1], target=tmp_path)

    # Try saving duplicate in subsequent call
    with pytest.raises(DuplicateComponentError, match="Component identity already exists"):
        save_components([c2_dup], target=tmp_path, replace=False)


def test_6_replace_true_replaces_matching_record(tmp_path):
    """Test 6: replace=True replaces only the matching record and updates its values."""
    c1 = _create_sample_component(lead_day=1, threshold_mm=15.6, sum_error=10.0)
    c2 = _create_sample_component(lead_day=2, threshold_mm=15.6, sum_error=50.0)

    save_components([c1, c2], target=tmp_path)

    # Replacement for c1 with new sum_error
    c1_updated = _create_sample_component(lead_day=1, threshold_mm=15.6, sum_error=99.0)
    save_components([c1_updated], target=tmp_path, replace=True)

    loaded = load_components(target=tmp_path)
    assert len(loaded) == 2

    loaded_map = {c.lead_day: c for c in loaded}
    assert loaded_map[1].sum_error == pytest.approx(99.0)  # Replaced!
    assert loaded_map[2].sum_error == pytest.approx(50.0)  # Preserved!


def test_7_append_preserves_existing_different_records(tmp_path):
    """Test 7: Appending records with different identities preserves existing records."""
    c1 = _create_sample_component(lead_day=1, threshold_mm=15.6)
    save_components([c1], target=tmp_path)

    c2 = _create_sample_component(lead_day=2, threshold_mm=15.6)
    save_components([c2], target=tmp_path, replace=False)

    loaded = load_components(target=tmp_path)
    assert len(loaded) == 2
    leads = sorted([c.lead_day for c in loaded])
    assert leads == [1, 2]


def test_8_deterministic_schema(tmp_path):
    """Test 8: Verify Parquet output conforms strictly to COMPONENT_PYARROW_SCHEMA."""
    comp = _create_sample_component()
    file_target = tmp_path / "test_schema.parquet"

    save_components([comp], target=file_target)

    schema = pq.read_schema(file_target)
    assert schema.names == list(SCHEMA_COLUMNS)
    assert schema.equals(COMPONENT_PYARROW_SCHEMA)


def test_9_incompatible_corrupt_schema_fails_clearly(tmp_path):
    """Test 9: Incompatible columns or corrupt files raise IncompatibleSchemaError."""
    # 1. Parquet with wrong schema (missing columns)
    bad_table = pa.Table.from_arrays([pa.array([1, 2, 3])], names=["unknown_col"])
    bad_file = tmp_path / "bad_schema.parquet"
    pq.write_table(bad_table, bad_file)

    with pytest.raises(IncompatibleSchemaError, match="missing required columns"):
        load_components(target=bad_file)

    comp = _create_sample_component()
    with pytest.raises(IncompatibleSchemaError, match="missing required columns"):
        save_components([comp], target=bad_file)

    # 2. Corrupt file with non-parquet garbage
    corrupt_file = tmp_path / "corrupt.parquet"
    corrupt_file.write_bytes(b"THIS_IS_NOT_A_PARQUET_FILE")

    with pytest.raises(IncompatibleSchemaError, match="Failed to read"):
        load_components(target=corrupt_file)


def test_10_empty_input_handled_safely(tmp_path):
    """Test 10: Empty input batches and empty read paths are handled gracefully."""
    # Saving empty list is a safe no-op returning 0
    assert save_components([], target=tmp_path) == 0

    # Loading from non-existent directory returns empty list
    non_existent = tmp_path / "does_not_exist"
    assert load_components(target=non_existent) == []

    # Loading from empty directory returns empty list
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    assert load_components(target=empty_dir) == []
