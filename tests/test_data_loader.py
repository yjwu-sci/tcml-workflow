"""Tests for PhonixLoader filtering and dedup logic."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tcml_workflow import PhonixLoader
from tcml_workflow.data_loader import PhonixFilterConfig


def _toy_structure_json(symbols=("Si", "Si")):
    return json.dumps({
        "cell": [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]],
        "positions": [[0, 0, 0], [1.36, 1.36, 1.36]],
        "symbols": list(symbols),
        "pbc": [True, True, True],
    })


@pytest.fixture
def toy_phonix(tmp_path):
    """Toy Phonix-summary parquet covering all filter branches."""
    rows = [
        # mp-1: valid, low fc3_error → should be kept
        dict(mp_id="mp-1", formula="Si", spg_number=227, structure=_toy_structure_json(),
             **{"klat[W/mK]": 150.0, "fc3_error[%]": 0.5, "nac": 2.0, "qmesh": "14x14x14"}),
        # mp-2: klat too high → drop
        dict(mp_id="mp-2", formula="X", spg_number=1, structure=_toy_structure_json(),
             **{"klat[W/mK]": 5e7, "fc3_error[%]": 0.5, "nac": 2.0, "qmesh": "14x14x14"}),
        # mp-3: fc3_error too high → drop
        dict(mp_id="mp-3", formula="Y", spg_number=1, structure=_toy_structure_json(),
             **{"klat[W/mK]": 3.0, "fc3_error[%]": 99.0, "nac": 2.0, "qmesh": "14x14x14"}),
        # mp-1 duplicate with WORSE fc3_error → should be dropped on dedup
        dict(mp_id="mp-1", formula="Si", spg_number=227, structure=_toy_structure_json(),
             **{"klat[W/mK]": 152.0, "fc3_error[%]": 2.0, "nac": 2.0, "qmesh": "10x10x10"}),
        # mp-4: low κ insulator, valid
        dict(mp_id="mp-4", formula="LiF", spg_number=225, structure=_toy_structure_json(("Li", "F")),
             **{"klat[W/mK]": 2.3, "fc3_error[%]": 0.7, "nac": 2.0, "qmesh": "16x16x16"}),
    ]
    df = pd.DataFrame(rows)
    pq = tmp_path / "toy.parquet"
    df.to_parquet(pq)
    return pq


def test_clean_filters_and_dedups(toy_phonix):
    loader = PhonixLoader(toy_phonix)
    df = loader.clean()

    # mp-2 (klat extreme) and mp-3 (fc3 high) are gone; mp-1 dedup keeps the better row
    assert set(df["mp_id"]) == {"mp-1", "mp-4"}
    mp1_row = df[df["mp_id"] == "mp-1"].iloc[0]
    assert mp1_row["fc3_error[%]"] == pytest.approx(0.5)


def test_log_k_target_present(toy_phonix):
    df = PhonixLoader(toy_phonix).clean()
    assert "log_k" in df.columns
    assert np.allclose(10 ** df["log_k"], df["klat[W/mK]"])


def test_get_structure_returns_atoms(toy_phonix):
    df = PhonixLoader(toy_phonix).clean()
    atoms = PhonixLoader.get_structure(df.iloc[0])
    assert len(atoms) == 2
    assert atoms.pbc.all()


def test_stratified_split_keeps_distribution(toy_phonix):
    loader = PhonixLoader(toy_phonix)
    train_df, test_df = loader.train_test_split(test_frac=0.5, stratify_log_k_bins=2)
    assert len(train_df) + len(test_df) == len(loader.clean())


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PhonixLoader(tmp_path / "nope.parquet").clean()
