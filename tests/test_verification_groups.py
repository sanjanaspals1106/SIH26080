"""Unit tests for verification grouping and display eligibility (PRD §16.5).

Tests:
1. month derived correctly
2. region masks separate cells
3. phase grouping works
4. lps_near yes/no works
5. orographic flag works
6. coastal flag works
7. raw rain bins correct at boundaries 1, 15.6, 64.5
8. missing group metadata is skipped
9. all group still works unchanged
10. common mask preserved across groups
11. n_events=9 => not displayable
12. n_events=10 => displayable
13. n_events=None => unknown/not displayable
14. multiple requested group types produce separate records
"""

from datetime import date
import numpy as np
import pytest

from verification.compute import compute
from verification.groups import (
    RAW_RAIN_BINS,
    SUPPORTED_GROUP_TYPES,
    check_group_display_eligibility,
    generate_group_masks,
    get_display_status,
    get_min_events_for_grouping,
    get_month_name,
    get_rain_thresholds,
    get_raw_rain_size_masks,
    is_group_displayable,
)


def test_1_month_derived_correctly():
    """Test 1: Verify month abbreviation (Jun, Jul, Aug, Sep) derived from imd_date."""
    # Dates from strings
    assert get_month_name("2023-06-01") == "Jun"
    assert get_month_name("2023-07-15") == "Jul"
    assert get_month_name("2023-08-20") == "Aug"
    assert get_month_name("2023-09-30") == "Sep"

    # From date object
    assert get_month_name(date(2024, 6, 15)) == "Jun"
    assert get_month_name(date(2024, 7, 4)) == "Jul"
    assert get_month_name(date(2024, 8, 15)) == "Aug"
    assert get_month_name(date(2024, 9, 1)) == "Sep"

    # Verify mask generation for month
    masks = generate_group_masks(
        shape=(4,),
        group_types=["month"],
        imd_date="2023-07-15",
    )
    assert ("month", "Jul") in masks
    np.testing.assert_array_equal(masks[("month", "Jul")], np.ones(4, dtype=bool))


def test_2_region_masks_separate_cells():
    """Test 2: Verify region_code generates masks that separate cells properly."""
    region_codes = np.array(["WEST_COAST", "HIMALAYA", "WEST_COAST", "NORTHEAST"])
    masks = generate_group_masks(
        shape=(4,),
        group_types=["region_code"],
        region_code=region_codes,
    )

    assert ("region_code", "WEST_COAST") in masks
    assert ("region_code", "HIMALAYA") in masks
    assert ("region_code", "NORTHEAST") in masks

    np.testing.assert_array_equal(
        masks[("region_code", "WEST_COAST")],
        np.array([True, False, True, False]),
    )
    np.testing.assert_array_equal(
        masks[("region_code", "HIMALAYA")],
        np.array([False, True, False, False]),
    )
    np.testing.assert_array_equal(
        masks[("region_code", "NORTHEAST")],
        np.array([False, False, False, True]),
    )


def test_3_phase_grouping_works():
    """Test 3: Verify phase grouping works for scalar upstream phase and per-cell phase."""
    # Scalar phase
    masks_scalar = generate_group_masks(
        shape=(3,),
        group_types=["phase"],
        phase="active",
    )
    assert ("phase", "active") in masks_scalar
    np.testing.assert_array_equal(masks_scalar[("phase", "active")], np.ones(3, dtype=bool))

    # Array phase
    phase_arr = np.array(["active", "normal", "break"])
    masks_arr = generate_group_masks(
        shape=(3,),
        group_types=["phase"],
        phase=phase_arr,
    )
    assert ("phase", "active") in masks_arr
    assert ("phase", "normal") in masks_arr
    assert ("phase", "break") in masks_arr

    np.testing.assert_array_equal(masks_arr[("phase", "active")], np.array([True, False, False]))
    np.testing.assert_array_equal(masks_arr[("phase", "normal")], np.array([False, True, False]))
    np.testing.assert_array_equal(masks_arr[("phase", "break")], np.array([False, False, True]))


def test_4_lps_near_yes_no_works():
    """Test 4: Verify lps_near generates yes/no groups."""
    # Scalar boolean True
    m_yes = generate_group_masks(shape=(2,), group_types=["lps_near"], lps_near=True)
    assert ("lps_near", "yes") in m_yes
    np.testing.assert_array_equal(m_yes[("lps_near", "yes")], np.ones(2, dtype=bool))

    # Scalar boolean False
    m_no = generate_group_masks(shape=(2,), group_types=["lps_near"], lps_near=False)
    assert ("lps_near", "no") in m_no
    np.testing.assert_array_equal(m_no[("lps_near", "no")], np.ones(2, dtype=bool))

    # Per-cell array
    lps_arr = np.array([True, False, True, False])
    m_arr = generate_group_masks(shape=(4,), group_types=["lps_near"], lps_near=lps_arr)
    assert ("lps_near", "yes") in m_arr
    assert ("lps_near", "no") in m_arr
    np.testing.assert_array_equal(m_arr[("lps_near", "yes")], np.array([True, False, True, False]))
    np.testing.assert_array_equal(m_arr[("lps_near", "no")], np.array([False, True, False, True]))


def test_5_orographic_flag_works():
    """Test 5: Verify orographic flag produces true/false groups."""
    # Scalar True
    m_true = generate_group_masks(
        shape=(2,), group_types=["orographic_favorable"], orographic_favorable=True
    )
    assert ("orographic_favorable", "true") in m_true
    np.testing.assert_array_equal(m_true[("orographic_favorable", "true")], np.ones(2, dtype=bool))

    # Scalar False
    m_false = generate_group_masks(
        shape=(2,), group_types=["orographic_favorable"], orographic_favorable=False
    )
    assert ("orographic_favorable", "false") in m_false
    np.testing.assert_array_equal(m_false[("orographic_favorable", "false")], np.ones(2, dtype=bool))

    # Array
    arr = np.array([True, False])
    m_arr = generate_group_masks(
        shape=(2,), group_types=["orographic_favorable"], orographic_favorable=arr
    )
    assert ("orographic_favorable", "true") in m_arr
    assert ("orographic_favorable", "false") in m_arr
    np.testing.assert_array_equal(m_arr[("orographic_favorable", "true")], np.array([True, False]))
    np.testing.assert_array_equal(m_arr[("orographic_favorable", "false")], np.array([False, True]))


def test_6_coastal_flag_works():
    """Test 6: Verify coastal flag produces true/false groups."""
    # Scalar True
    m_true = generate_group_masks(
        shape=(2,), group_types=["coastal_favorable"], coastal_favorable=True
    )
    assert ("coastal_favorable", "true") in m_true

    # Scalar False
    m_false = generate_group_masks(
        shape=(2,), group_types=["coastal_favorable"], coastal_favorable=False
    )
    assert ("coastal_favorable", "false") in m_false

    # Array
    arr = np.array([False, True])
    m_arr = generate_group_masks(
        shape=(2,), group_types=["coastal_favorable"], coastal_favorable=arr
    )
    np.testing.assert_array_equal(m_arr[("coastal_favorable", "true")], np.array([False, True]))
    np.testing.assert_array_equal(m_arr[("coastal_favorable", "false")], np.array([True, False]))


def test_7_raw_rain_bins_correct_at_boundaries():
    """Test 7: Verify raw rain size bins strictly respect boundaries at 1.0, 15.6, and 64.5 mm."""
    # Rain values testing points just below and at the boundaries:
    # 0.0, 0.999 -> below_1
    # 1.0, 15.599 -> 1_to_15_6
    # 15.6, 64.499 -> 15_6_to_64_5
    # 64.5, 100.0 -> 64_5_and_above
    rain = np.array([0.0, 0.999, 1.0, 15.599, 15.6, 64.499, 64.5, 100.0])

    bin_masks = get_raw_rain_size_masks(rain)

    # Check that all 4 bins exist
    for b in RAW_RAIN_BINS:
        assert b in bin_masks

    # Index 0, 1: below_1
    np.testing.assert_array_equal(
        bin_masks["below_1"],
        np.array([True, True, False, False, False, False, False, False]),
    )
    # Index 2, 3: 1_to_15_6
    np.testing.assert_array_equal(
        bin_masks["1_to_15_6"],
        np.array([False, False, True, True, False, False, False, False]),
    )
    # Index 4, 5: 15_6_to_64_5
    np.testing.assert_array_equal(
        bin_masks["15_6_to_64_5"],
        np.array([False, False, False, False, True, True, False, False]),
    )
    # Index 6, 7: 64_5_and_above
    np.testing.assert_array_equal(
        bin_masks["64_5_and_above"],
        np.array([False, False, False, False, False, False, True, True]),
    )

    # Verify no cell is unassigned or assigned to multiple bins
    stacked = np.stack(list(bin_masks.values()), axis=0)
    assert np.all(np.sum(stacked, axis=0) == 1)


def test_8_missing_group_metadata_is_skipped():
    """Test 8: Verify that missing group metadata causes that group to be skipped, not inventing a value."""
    # Request multiple group types where metadata is omitted (None)
    masks = generate_group_masks(
        shape=(4,),
        group_types=["all", "phase", "lps_near", "region_code", "coastal_favorable"],
        phase=None,
        lps_near=None,
        region_code=None,
        coastal_favorable=None,
    )

    # Only "all" should be produced; the missing metadata groups are safely skipped
    assert list(masks.keys()) == [("all", "all")]


def test_9_all_group_still_works_unchanged():
    """Test 9: Verify default all group still works unchanged and identically with or without group_types."""
    f = [10.0, 20.0, 30.0]
    o = [12.0, 18.0, 25.0]

    res_default = compute(
        forecast=f,
        observed=o,
        imd_date="2023-07-15",
        lead_day=1,
        thresholds=None,
        neighbourhood_sizes=None,
    )

    res_explicit_all = compute(
        forecast=f,
        observed=o,
        imd_date="2023-07-15",
        lead_day=1,
        thresholds=None,
        neighbourhood_sizes=None,
        group_types=["all"],
    )

    c_def = res_default.get_continuous()
    c_exp = res_explicit_all.get_continuous()

    assert c_def is not None
    assert c_exp is not None
    assert c_def.group_type == "all"
    assert c_def.group_value == "all"
    assert c_def.n_samples == c_exp.n_samples == 3
    assert c_def.sum_error == pytest.approx(c_exp.sum_error)
    assert c_def.sum_squared_error == pytest.approx(c_exp.sum_squared_error)


def test_10_common_mask_preserved_across_groups():
    """Test 10: Verify that common_mask is strictly enforced across all subgroups."""
    forecast = [5.0, 20.0, 50.0, 80.0]
    observed = [5.0, 20.0, 50.0, 80.0]
    # Common mask excludes cell 0 (index 0 is False)
    common_mask = [False, True, True, True]
    region_code = ["WEST_COAST", "WEST_COAST", "HIMALAYA", "HIMALAYA"]

    result = compute(
        forecast=forecast,
        observed=observed,
        imd_date="2023-07-15",
        lead_day=1,
        common_mask=common_mask,
        group_types=["all", "region_code"],
        region_code=region_code,
        thresholds=None,
        neighbourhood_sizes=None,
    )

    # "all" group evaluates only indices 1, 2, 3 (n_samples = 3)
    c_all = result.get_continuous(group_type="all", group_value="all")
    assert c_all is not None
    assert c_all.n_samples == 3

    # WEST_COAST has cells at index 0 and 1, but index 0 is excluded by common_mask -> n_samples = 1
    c_west = result.get_continuous(group_type="region_code", group_value="WEST_COAST")
    assert c_west is not None
    assert c_west.n_samples == 1

    # HIMALAYA has cells at index 2 and 3, both inside common_mask -> n_samples = 2
    c_him = result.get_continuous(group_type="region_code", group_value="HIMALAYA")
    assert c_him is not None
    assert c_him.n_samples == 2


def test_11_n_events_9_not_displayable():
    """Test 11: Verify n_events=9 is classified as not displayable (< 10 threshold)."""
    assert is_group_displayable(9) is False

    eligibility = check_group_display_eligibility(9)
    assert eligibility.is_displayable is False
    assert eligibility.status == "not_displayable"
    assert eligibility.n_events == 9
    assert eligibility.min_events == 10
    assert "Not enough events" in eligibility.message
    assert bool(eligibility) is False
    assert get_display_status(9) == "not_displayable"


def test_12_n_events_10_displayable():
    """Test 12: Verify n_events=10 is classified as displayable (>= 10 threshold)."""
    assert is_group_displayable(10) is True

    eligibility = check_group_display_eligibility(10)
    assert eligibility.is_displayable is True
    assert eligibility.status == "displayable"
    assert eligibility.n_events == 10
    assert eligibility.min_events == 10
    assert bool(eligibility) is True
    assert get_display_status(10) == "displayable"


def test_13_n_events_none_unknown_not_displayable():
    """Test 13: Verify n_events=None has status unknown and is not displayable yet."""
    assert is_group_displayable(None) is False

    eligibility = check_group_display_eligibility(None)
    assert eligibility.is_displayable is False
    assert eligibility.status == "unknown"
    assert eligibility.n_events is None
    assert bool(eligibility) is False
    assert get_display_status(None) == "unknown"


def test_14_multiple_requested_group_types_produce_separate_records():
    """Test 14: Verify multiple requested group types produce separate records in compute(...)."""
    forecast = [5.0, 25.0, 70.0, 10.0]
    observed = [4.0, 20.0, 65.0, 12.0]
    region_code = ["WEST_COAST", "WEST_COAST", "HIMALAYA", "HIMALAYA"]

    result = compute(
        forecast=forecast,
        observed=observed,
        imd_date="2023-08-10",
        lead_day=2,
        group_types=["all", "month", "region_code", "lead_day"],
        region_code=region_code,
        thresholds=[15.6],
        neighbourhood_sizes=None,
    )

    groups = result.list_groups()
    expected_groups = [
        ("all", "all"),
        ("month", "Aug"),
        ("region_code", "HIMALAYA"),
        ("region_code", "WEST_COAST"),
        ("lead_day", "2"),
    ]
    for eg in expected_groups:
        assert eg in groups

    # Verify separate continuous records exist
    assert result.get_continuous("all", "all") is not None
    assert result.get_continuous("month", "Aug") is not None
    assert result.get_continuous("region_code", "WEST_COAST") is not None
    assert result.get_continuous("region_code", "HIMALAYA") is not None
    assert result.get_continuous("lead_day", "2") is not None

    # Verify separate contingency records exist
    assert result.get_contingency(15.6, "all", "all") is not None
    assert result.get_contingency(15.6, "month", "Aug") is not None
    assert result.get_contingency(15.6, "region_code", "WEST_COAST") is not None
