"""Tests for periodic descriptor computation."""

import numpy as np
import pytest

from tcml_workflow.descriptors import (
    compute_periodic_descriptors,
    compute_periodic_dataframe,
    descriptor_columns,
    parse_formula,
)


def test_parse_formula_basic():
    assert parse_formula("Li3PO4") == {"Li": 3.0, "P": 1.0, "O": 4.0}


def test_parse_formula_fractional():
    out = parse_formula("Li6.5La3Zr1.5Ta0.5O12")
    assert out["Li"] == 6.5
    assert out["Zr"] == 1.5
    assert out["O"] == 12


def test_parse_formula_expands_parentheses():
    # Regression: a flat regex read Cd(InS2)2 as Cd1In1S2
    assert parse_formula("Cd(InS2)2") == {"Cd": 1.0, "In": 2.0, "S": 4.0}
    assert parse_formula("Hg3(TeI)2") == {"Hg": 3.0, "Te": 2.0, "I": 2.0}


def test_coefficients_with_parentheses():
    # Cd(InS2)2 = CdIn2S4: group 12 = 1/7, group 13 = 2/7, group 16 = 4/7
    v = compute_periodic_descriptors("Cd(InS2)2", n_groups=17)
    assert abs(v[11] - 1 / 7) < 1e-9
    assert abs(v[12] - 2 / 7) < 1e-9
    assert abs(v[15] - 4 / 7) < 1e-9


def test_descriptor_columns_length():
    assert len(descriptor_columns(17)) == 34
    assert len(descriptor_columns(18)) == 36


def test_descriptor_shape_17_groups():
    v = compute_periodic_descriptors("LiTiS2", n_groups=17)
    assert v.shape == (34,)


def test_descriptor_shape_18_groups():
    v = compute_periodic_descriptors("LiTiS2", n_groups=18)
    assert v.shape == (36,)


def test_coefficients_sum_to_one_for_known_elements():
    # LiTiS2: Li=1/4 group 1, Ti=1/4 group 4, S=2/4 group 16
    # All in groups ≤ 17 → coef sum should be exactly 1.0
    v = compute_periodic_descriptors("LiTiS2", n_groups=17)
    coefs = v[:17]
    assert abs(coefs.sum() - 1.0) < 1e-9
    # Expected: group 1 = 0.25, group 4 = 0.25, group 16 = 0.50
    assert abs(coefs[0] - 0.25) < 1e-9     # group 1 (Li)
    assert abs(coefs[3] - 0.25) < 1e-9     # group 4 (Ti)
    assert abs(coefs[15] - 0.50) < 1e-9    # group 16 (S)


def test_property_is_weighted_z():
    # Mixed group 1: 2× Li (Z=3) + 1× Cs (Z=55) → prop = (2·3 + 1·55)/3 = 20.33
    v = compute_periodic_descriptors("Li2Cs", n_groups=17)
    coefs = v[:17]
    props = v[17:34]
    assert abs(coefs[0] - 1.0) < 1e-9
    assert abs(props[0] - (2 * 3 + 55) / 3) < 1e-9


def test_noble_gas_excluded_when_n_groups_17():
    # XeO2: Xe is group 18 → should be silently dropped from coef/prop
    # but total still goes into the denominator
    v = compute_periodic_descriptors("XeO2", n_groups=17)
    coefs = v[:17]
    # Only O contributes; coef sum < 1 because Xe is excluded
    assert abs(coefs.sum() - 2 / 3) < 1e-9
    assert abs(coefs[15] - 2 / 3) < 1e-9     # group 16 (O)


def test_noble_gas_included_when_n_groups_18():
    v = compute_periodic_descriptors("XeO2", n_groups=18)
    coefs = v[:18]
    assert abs(coefs.sum() - 1.0) < 1e-9
    assert abs(coefs[17] - 1 / 3) < 1e-9     # group 18 (Xe)


def test_compute_periodic_dataframe_shape():
    df = compute_periodic_dataframe(["LiTiS2", "Mg2Si", "Cu2O"], n_groups=17)
    assert df.shape == (3, 34)
    assert list(df.columns)[:3] == ["coef_01", "coef_02", "coef_03"]


def test_compute_periodic_dataframe_indexed_by_mp_id():
    df = compute_periodic_dataframe(
        ["LiTiS2", "Mg2Si"], n_groups=17, index=["mp-7", "mp-14"],
    )
    assert list(df.index) == ["mp-7", "mp-14"]
