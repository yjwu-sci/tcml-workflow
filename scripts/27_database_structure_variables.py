"""Structural variables of the composition + database structure descriptor set.

For every material of the training table, from the Phonix structure used for training:
  n_atoms_prim   number of atoms in the primitive cell (Phonix `natoms_prim`; for the one entry without it,
                 the primitive cell found by spglib)
  vol_per_atom   cell volume / number of atoms (A^3)
  density        mass / volume (g cm^-3)
The fourth variable, the space-group number, is taken from the training table by scripts/35.

Writes results/database_structure/database_structure_variables.csv (one row per material).
Run from the tcml-workflow root with the `tcml` conda env, after scripts/02.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
from tcml_workflow import PhonixLoader

OUT = REPO / "results/database_structure"
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
tr = pd.read_parquet("data/processed/train.parquet")
raw = PhonixLoader("data/phonix/summary.parquet").clean().set_index("mp_id")
rows = []
for mp_id in tr["mp_id"]:
    s = AseAtomsAdaptor.get_structure(PhonixLoader.get_structure(raw.loc[mp_id]))
    n_prim = raw.loc[mp_id, "natoms_prim"]
    if pd.isna(n_prim):
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        n_prim = len(SpacegroupAnalyzer(s, symprec=0.01).find_primitive())
        print(f"  {mp_id}: natoms_prim missing in Phonix, spglib primitive cell has {n_prim} atoms")
    rows.append(dict(mp_id=mp_id, n_atoms_prim=int(n_prim), vol_per_atom=s.volume / len(s),
                     density=float(s.density)))
out = pd.DataFrame(rows)
assert len(out) == len(tr) and out["density"].notna().all()
out.to_csv(OUT / "database_structure_variables.csv", index=False)
print(f"wrote {OUT / 'database_structure_variables.csv'}: {len(out)} materials in {time.time() - t0:.0f} s")
