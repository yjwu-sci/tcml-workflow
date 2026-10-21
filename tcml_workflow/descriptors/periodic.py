"""34/36-dimensional periodic-table-group compositional descriptors.

Implements the encoding from Wu et al., STAM Methods 5, 2513218 (2025):
for each of the 17 or 18 periodic-table groups, two features are computed:

  coef_g  = fraction of total atoms belonging to group g
  prop_g  = weighted-average atomic number of elements in group g

The default uses 17 groups (excluding noble gases),
producing a 34-dim vector. The sister `csp-workflow-mp` package uses
18 groups (36-dim). Both schemes are supported via `n_groups`.

Vendored from `csp_workflow_mp.descriptor` so tcml-workflow has no
install-time dependency on csp-workflow-mp.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


_ELEMENT_GROUP: dict[str, int] = {
    "H": 1, "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Fr": 1,
    "Be": 2, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Ra": 2,
    "Sc": 3, "Y": 3,
    "La": 3, "Ce": 3, "Pr": 3, "Nd": 3, "Pm": 3, "Sm": 3, "Eu": 3,
    "Gd": 3, "Tb": 3, "Dy": 3, "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3, "Lu": 3,
    "Ac": 3, "Th": 3, "Pa": 3, "U": 3, "Np": 3, "Pu": 3, "Am": 3, "Cm": 3,
    "Ti": 4, "Zr": 4, "Hf": 4,
    "V": 5, "Nb": 5, "Ta": 5,
    "Cr": 6, "Mo": 6, "W": 6,
    "Mn": 7, "Tc": 7, "Re": 7,
    "Fe": 8, "Ru": 8, "Os": 8,
    "Co": 9, "Rh": 9, "Ir": 9,
    "Ni": 10, "Pd": 10, "Pt": 10,
    "Cu": 11, "Ag": 11, "Au": 11,
    "Zn": 12, "Cd": 12, "Hg": 12,
    "B": 13, "Al": 13, "Ga": 13, "In": 13, "Tl": 13,
    "C": 14, "Si": 14, "Ge": 14, "Sn": 14, "Pb": 14,
    "N": 15, "P": 15, "As": 15, "Sb": 15, "Bi": 15,
    "O": 16, "S": 16, "Se": 16, "Te": 16, "Po": 16,
    "F": 17, "Cl": 17, "Br": 17, "I": 17, "At": 17,
    "He": 18, "Ne": 18, "Ar": 18, "Kr": 18, "Xe": 18, "Rn": 18,
}

_ATOMIC_NUMBER: dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
    "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
    "Np": 93, "Pu": 94, "Am": 95, "Cm": 96,
}

def parse_formula(formula: str) -> dict[str, float]:
    """Parse e.g. ``"Li3PO4"`` → ``{'Li': 3, 'P': 1, 'O': 4}``.

    Delegates to ``pymatgen.core.Composition``, so parenthesised groups are
    expanded (``"Cd(InS2)2"`` → ``{'Cd': 1, 'In': 2, 'S': 4}``) and
    non-integer stoichiometry is supported (``"Li6.5La3Zr1.5Ta0.5O12"``).
    The Phonix and Materials Project formula strings contain parentheses
    (e.g. 816 of the 6,716 training formulas), so a flat regex is not enough.
    """
    from pymatgen.core import Composition
    return {str(el): float(n) for el, n in Composition(formula).get_el_amt_dict().items()}


def compute_periodic_descriptors(
    formula: str,
    n_groups: int = 17,
) -> np.ndarray:
    """Compute the 2*n_groups periodic descriptor vector for one formula.

    Parameters
    ----------
    formula : str
        Chemical formula such as ``"LiTiS2"``.
    n_groups : int
        Number of periodic-table groups to include (17 = default,
        excludes noble gases; 18 = sister `csp-workflow-mp` standard).

    Returns
    -------
    np.ndarray of shape (2*n_groups,)
        ``[coef_01..coef_{n_groups}, prop_01..prop_{n_groups}]``.
    """
    elements = parse_formula(formula)
    total = sum(elements.values())
    if total == 0:
        return np.zeros(2 * n_groups)

    coefs = np.zeros(n_groups)
    props = np.zeros(n_groups)
    counts: dict[int, float] = {}
    weighted_z: dict[int, float] = {}

    for elem, c in elements.items():
        g = _ELEMENT_GROUP.get(elem)
        z = _ATOMIC_NUMBER.get(elem, 0)
        if g is None or g > n_groups:
            continue
        counts[g] = counts.get(g, 0.0) + c
        weighted_z[g] = weighted_z.get(g, 0.0) + c * z

    for g in range(1, n_groups + 1):
        gc = counts.get(g, 0.0)
        coefs[g - 1] = gc / total
        if gc > 0:
            props[g - 1] = weighted_z[g] / gc

    return np.concatenate([coefs, props])


def descriptor_columns(n_groups: int = 17) -> list[str]:
    """Stable column names: ``coef_01..coef_n, prop_01..prop_n``."""
    return (
        [f"coef_{i:02d}" for i in range(1, n_groups + 1)]
        + [f"prop_{i:02d}" for i in range(1, n_groups + 1)]
    )


def compute_periodic_dataframe(
    formulas: Iterable[str],
    n_groups: int = 17,
    index: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Batch compute periodic descriptors → DataFrame.

    Parameters
    ----------
    formulas : iterable of str
        Chemical formulas to encode.
    n_groups : int
        17 (default) or 18 (as in csp-workflow-mp).
    index : iterable of str, optional
        Index for the DataFrame; defaults to the formulas themselves.
    """
    formulas = list(formulas)
    arr = np.stack([compute_periodic_descriptors(f, n_groups) for f in formulas])
    return pd.DataFrame(
        arr,
        columns=descriptor_columns(n_groups),
        index=index if index is not None else formulas,
    )
