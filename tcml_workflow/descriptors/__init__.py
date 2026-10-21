"""Descriptor computation for tcml-workflow.

Two descriptor families are exposed:

- periodic : 34-dim (17 groups, default) or 36-dim (18 groups) element-group
             compositional features (formula → vector).
- tda      : persistent-homology features on cation/anion sublattices, via the
             ion-site TDA pipeline (log1p PI_total_energy,
             entropy density=False, rho overflow cap 1e6).

Both return DataFrames keyed by material_id so they can be joined and
fed to NestedCVPipeline as a single feature matrix.
"""

from tcml_workflow.descriptors.periodic import (
    compute_periodic_descriptors,
    compute_periodic_dataframe,
    descriptor_columns,
    parse_formula,
)
from tcml_workflow.descriptors.tda import compute_tda_descriptors

__all__ = [
    "compute_periodic_descriptors",
    "compute_periodic_dataframe",
    "descriptor_columns",
    "parse_formula",
    "compute_tda_descriptors",
]
