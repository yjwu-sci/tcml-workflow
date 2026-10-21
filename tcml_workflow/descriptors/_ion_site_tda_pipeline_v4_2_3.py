# -*- coding: utf-8 -*-
# ion_site_tda_pipeline_v4_2_3.py
# ------------------------------------------------------------
# Ion-site TDA pipeline
# Fixes from v4.2.1:
#   [Fix 1] PI_total_energy: apply log1p to prevent extreme outliers
#            (unnormalized sum can reach ~25,000 for large/complex supercells)
#   [Fix 2] pd_stats entropy: use density=False to avoid double-normalization
#   [Fix 3] rho_all_dimless / rho_an_rel / rho_cat_rel: cap at 1e6 to prevent
#            ~1e27 overflow when near-duplicate atom positions exist in disordered CIFs
# Original v4.2.1 is preserved unchanged for comparison.
# ------------------------------------------------------------

import os, re, argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler

# ---------- Optional deps ----------
GUDHI_CORE = False
PI_AVAILABLE = False
try:
    import gudhi as gd
    _ = gd.AlphaComplex
    GUDHI_CORE = True
except Exception:
    GUDHI_CORE = False

try:
    from gudhi.representations import PersistenceImage
    PI_AVAILABLE = True
except Exception:
    PI_AVAILABLE = False

try:
    from pymatgen.io.cif import CifParser
    from pymatgen.core.structure import Structure
    from pymatgen.core.periodic_table import Element
    PYMATGEN_AVAILABLE = True
except Exception:
    PYMATGEN_AVAILABLE = False


# ---------- IO helpers ----------
def _sheet_arg(s):
    return int(s) if isinstance(s, str) and s.isdigit() else s

def str2bool(s):
    return str(s).lower() in ["true","1","yes","y","t"]

def load_table(path: str, sheet=0) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in [".xlsx", ".xlsm", ".xls", ".xlsb"]:
        try:
            import openpyxl  # noqa
        except Exception as e:
            raise ImportError("openpyxl is required for Excel. Install via conda-forge or pip.") from e
        return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input file: {path} (use .xlsx or .csv)")

def parse_structure(cif_path: str, occupancy_tol: float = 10.0):
    if not PYMATGEN_AVAILABLE:
        raise RuntimeError("pymatgen not installed")
    s = None
    try:
        parser = CifParser(cif_path, occupancy_tolerance=occupancy_tol)
        structs = parser.parse_structures(primitive=False)
        if structs:
            s = structs[0]
    except Exception:
        s = None
    if s is None:
        s = Structure.from_file(cif_path)
    return s

def make_supercell(struct, na, nb, nc):
    s = struct.copy()
    if (na, nb, nc) != (1, 1, 1):
        s.make_supercell([na, nb, nc])
    return s

def majority_species_symbol(site):
    try:
        items = list(site.species.items())
        if not items:
            return None
        sp, occ = max(items, key=lambda kv: kv[1])
        sym = getattr(sp, "symbol", str(sp))
        Element(sym)  # sanity
        return sym
    except Exception:
        return None


# ---------- Ion-site classification + Scheme A ----------
ANION_PRIMARY = {"F","Cl","Br","I","O","S","Se","Te","N"}  # high-confidence anions
ANION_SECONDARY = {"P","C"}  # context-dependent

def _pauling_X(sym):
    try:
        return Element(sym).X
    except Exception:
        return None

def determine_ion_sets(element_symbols,
                       mode="relative", gap=0.15,
                       treat_hydride=True,
                       fixed_set=None,
                       force_bipartition=True):
    """
    Return (anion_set, cation_set) for the given composition.
    - 'relative': ANION_PRIMARY U {X >= Xmax - gap} (+ hydride rule)
    - 'fixed'   : use fixed_set
    - Scheme A: if a multi-element composition has no cation, move the least
      electronegative element to the cation set; a single element is not split.
    """
    elems = [e for e in set(element_symbols) if isinstance(e, str)]
    unique_elems = set(elems)

    if mode == "fixed":
        anions = set(fixed_set) if fixed_set else set(ANION_PRIMARY | ANION_SECONDARY)
    else:
        Xs = {e: _pauling_X(e) for e in unique_elems}
        finite_X = [x for x in Xs.values() if x is not None]
        Xmax = max(finite_X) if finite_X else None

        anions = set()
        anions |= (ANION_PRIMARY & unique_elems)
        if Xmax is not None:
            for e, x in Xs.items():
                if x is not None and x >= (Xmax - gap):
                    anions.add(e)
        if treat_hydride and "H" in unique_elems:
            very_electropositive = any((_pauling_X(e) is not None and _pauling_X(e) <= 1.3)
                                       for e in unique_elems if e != "H")
            if very_electropositive:
                anions.add("H")

    cations = unique_elems - anions

    if force_bipartition and len(unique_elems) > 1 and len(cations) == 0:
        def x_val(e):
            x = _pauling_X(e)
            return x if x is not None else 9.9
        if len(anions) > 0:
            e_min = min(anions, key=lambda e: x_val(e))
            anions = set(anions)
            anions.discard(e_min)
            cations = set([e_min])

    if len(unique_elems) == 1:  # single-element => no hard split
        anions = set()
        cations = set()

    return anions, cations


# ---------- scaling / normalization ----------
def mean_nn_distance(points):
    if points is None or len(points) < 2:
        return 1.0
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    nn = dists[:, 1]
    nn = nn[nn > 0]
    if len(nn) == 0:
        return 1.0
    m = float(np.median(nn))
    return m if np.isfinite(m) and m > 0 else 1.0

def apply_scale(points, scale='none', ref_d=None):
    if scale == 'none' or points is None or len(points) == 0:
        return points, 1.0
    if scale == 'nn':
        d = mean_nn_distance(points) if ref_d is None else ref_d
        return points / d, d
    return points, 1.0

def normalize_features(feats: dict, mode: str, n_atoms: int, volume: float):
    out = feats.copy()
    if mode == 'per_atom' and n_atoms > 0:
        for k in list(out.keys()):
            if k.endswith(("_sum_pers", "_count")):
                out[k] = out[k] / n_atoms
    elif mode == 'per_volume' and volume > 0:
        for k in list(out.keys()):
            if k.endswith(("_sum_pers", "_count")):
                out[k] = out[k] / volume
    return out


# ---------- TDA helpers (hotfix included) ----------
def finite_diag(diag):
    """Return Nx2 array with only finite birth/death and death>=birth."""
    if diag is None or len(diag) == 0:
        return np.empty((0, 2), float)
    d = np.array(diag, float)
    if d.ndim != 2 or d.shape[1] != 2:
        return np.empty((0, 2), float)
    m = np.isfinite(d).all(axis=1)
    d = d[m]
    if d.size == 0:
        return np.empty((0, 2), float)
    d = d[d[:, 1] >= d[:, 0]]
    return d

def alpha_cutoff(points: np.ndarray) -> float:
    """Largest alpha-square value that the periodic image can actually resolve.

    An alpha-complex filtration value is the squared circumradius of a Delaunay
    simplex. In a perfect crystal many atom quadruples are exactly coplanar, so
    their circumsphere is degenerate and the computed radius diverges; the
    resulting classes are born and die at the same enormous value and carry no
    structural information (their persistence is pure floating-point
    cancellation). Truncating the filtration at half the shortest extent of the
    point cloud removes them, and is the physically meaningful bound anyway: no
    loop or void wider than the periodic image is resolvable from it.

    Set TCML_ALPHA_CUTOFF to a float to override with a fixed value, or to
    "inf" to disable truncation (reproduces the pre-2026-08 behaviour).
    """
    env = os.environ.get("TCML_ALPHA_CUTOFF")
    if env is not None:
        return float(env)
    if points is None or len(points) < 2:
        return float("inf")
    extent = np.ptp(np.asarray(points, float), axis=0)
    extent = extent[extent > 0]
    if extent.size == 0:
        return float("inf")
    return float((extent.min() / 2.0) ** 2)


def gudhi_pd(points: np.ndarray, maxdim=2):
    if not GUDHI_CORE:
        raise RuntimeError("GUDHI AlphaComplex not found. Install via conda-forge or pip.")
    if points is None or len(points) == 0:
        return {d: [] for d in range(maxdim+1)}
    ac = gd.AlphaComplex(points=points)
    st = ac.create_simplex_tree(max_alpha_square=alpha_cutoff(points))
    st.compute_persistence()
    return {dim: st.persistence_intervals_in_dimension(dim).tolist() for dim in range(maxdim+1)}

def pd_stats(diag):
    d = finite_diag(diag)
    if d.size == 0:
        return dict(count=0, sum_pers=0.0, mean_pers=0.0, max_pers=0.0, p95_pers=0.0,
                    birth_mean=0.0, death_mean=0.0, entropy=0.0)
    birth, death = d[:, 0], d[:, 1]
    pers = np.clip(death - birth, 0, None)

    cnt = len(pers)
    sm  = float(np.sum(pers))
    mn  = float(np.mean(pers))
    mx  = float(np.max(pers))
    p95 = float(np.percentile(pers, 95))
    bmn = float(np.mean(birth))
    dmn = float(np.mean(death))

    if cnt < 2 or (float(np.max(pers)) - float(np.min(pers))) <= 1e-15:
        ent = 0.0
    else:
        bins = min(20, max(5, cnt // 2))
        lo   = float(np.min(pers))
        hi   = float(np.max(pers)) + 1e-12
        hist, _ = np.histogram(pers, bins=bins, range=(lo, hi), density=False)  # [Fix 2]
        probs = hist / (np.sum(hist) + 1e-12)
        ent   = float(-np.sum(probs * np.log(probs + 1e-12)))

    return dict(count=cnt, sum_pers=sm, mean_pers=mn, max_pers=mx, p95_pers=p95,
                birth_mean=bmn, death_mean=dmn, entropy=ent)

def persistence_image_features(diag, resolution=20):
    d = finite_diag(diag)
    if not PI_AVAILABLE or d.shape[0] == 0:
        return dict(PI_total_energy=0.0, PI_cog_birth=0.0, PI_cog_pers=0.0)

    birth, death = d[:, 0], d[:, 1]
    pers = np.clip(death - birth, 0, None)

    if not np.isfinite(pers).any() or (float(np.max(pers)) - float(np.min(pers)) <= 1e-15):
        return dict(PI_total_energy=0.0, PI_cog_birth=0.0, PI_cog_pers=0.0)

    pts = np.vstack([birth, pers]).T
    scaler = MinMaxScaler()
    pts_norm = scaler.fit_transform(pts)

    pim = PersistenceImage(resolution=[resolution, resolution])
    vec = pim.fit_transform([pts_norm])[0].reshape(-1)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    grid = vec.reshape(resolution, resolution)
    xs, ys = np.meshgrid(np.arange(resolution), np.arange(resolution), indexing='xy')
    energy = float(np.log1p(np.sum(vec)))  # [Fix 1] log1p prevents extreme outliers (was ~25000, now ~10)
    denom  = float(np.sum(grid) + 1e-12)
    cx = float(np.sum(xs * grid) / denom)
    cy = float(np.sum(ys * grid) / denom)
    return dict(PI_total_energy=energy, PI_cog_birth=cx, PI_cog_pers=cy)

def collect_features(prefix, diag):
    out = {}
    out.update({f"{prefix}_{k}": v for k, v in pd_stats(diag).items()})
    out.update({f"{prefix}_{k}": v for k, v in persistence_image_features(diag).items()})
    return out


# ---------- main ----------
def main(args):
    if not GUDHI_CORE:
        print("[FATAL] GUDHI AlphaComplex not found. `conda install -c conda-forge gudhi` (or pip).")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_table(args.excel, sheet=args.sheet)
    kappa = pd.to_numeric(df[args.kappa_col], errors='coerce')
    df = df.loc[kappa.notna()].copy()
    df["kappa_rt"] = kappa.loc[kappa.notna()].astype(float)

    # collect cifs
    all_cifs = []
    for root, _, files in os.walk(args.cif_root):
        for f in files:
            if f.lower().endswith(".cif"):
                all_cifs.append(os.path.join(root, f))

    def find_cif_for_id(mid):
        # Match either an exact "<mid>.cif" or a "<mid>_*.cif" / "<mid>-*.cif"
        # variant, rather than a plain `startswith(mid)`. This prevents a
        # shorter digit-id (e.g. "1639") from accidentally matching a longer
        # one's CIF (e.g. "16390.cif") when both exist in the same cif_root.
        # The Phonix database has 1,073 such prefix collisions; without this
        # guard ~16% of materials would receive features from the wrong CIF.
        target_exact = f"{mid}.cif"
        for p in all_cifs:
            base = os.path.basename(p)
            if base == target_exact:
                return p
            if base.startswith(f"{mid}_") or base.startswith(f"{mid}-"):
                if base.lower().endswith(".cif"):
                    return p
        return None

    def digits(s):
        s = str(s)
        m = re.search(r"(\d+)", s)
        return m.group(1) if m else None

    rows_iter = list(df.iterrows())
    if args.limit is not None:
        rows_iter = rows_iter[:args.limit]

    features = []
    n_ok, n_no_cif, n_fail = 0, 0, 0

    for _, row in rows_iter:
        mid_full = str(row[args.id_col])
        mid_digits = digits(mid_full)
        if mid_digits is None:
            n_fail += 1
            continue
        cif_path = find_cif_for_id(mid_digits)
        if not cif_path:
            n_no_cif += 1
            continue

        try:
            s = parse_structure(cif_path)
            try:
                if args.make_ordered and hasattr(s, "get_ordered_structure") and not s.is_ordered:
                    s = s.get_ordered_structure()
            except Exception:
                pass
            s = make_supercell(s, *tuple(args.supercell))
            vol = s.lattice.volume

            # present element set + raw coordinates (UNSCALED)
            present_elems = []
            all_pts, cat_pts, an_pts = [], [], []
            for site in s.sites:
                sym = majority_species_symbol(site)
                if sym:
                    present_elems.append(sym)
                    x, y, z = site.coords
                    all_pts.append([x, y, z])

            an_set, ca_set = determine_ion_sets(
                set(present_elems),
                mode=args.anion_mode,
                gap=args.anion_gap,
                treat_hydride=(args.hydride_rule.lower()=="on"),
                fixed_set=set(args.anion_fixed) if args.anion_fixed else None,
                force_bipartition=(args.force_bipartition.lower()=="on")
            )

            # split raw coords by ion-site (UNSCALED)
            for site in s.sites:
                sym = majority_species_symbol(site)
                if sym is None:
                    continue
                x, y, z = site.coords
                if sym in an_set:
                    an_pts.append([x, y, z])
                elif sym in ca_set:
                    cat_pts.append([x, y, z])
                # all_pts were already added above

            all_pts = np.array(all_pts)
            cat_pts = np.array(cat_pts)
            an_pts  = np.array(an_pts)

            # === Density proxies on UN-SCALED coordinates ===
            eps = 1e-9
            def _nn(points):
                return mean_nn_distance(points) if points is not None and len(points) > 1 else np.nan
            d_all = _nn(all_pts); d_an = _nn(an_pts); d_cat = _nn(cat_pts)

            _RHO_CAP = 1e6  # [Fix 3] guard against ~1e27 overflow from near-duplicate atoms in disordered CIFs
            rho_all_dimless = float(min(1.0 / max(d_all, eps)**3, _RHO_CAP)) if np.isfinite(d_all) else 0.0
            rho_an_rel  = float(min((d_all / max(d_an,  eps))**3, _RHO_CAP)) if (np.isfinite(d_all) and np.isfinite(d_an))  else 0.0
            rho_cat_rel = float(min((d_all / max(d_cat, eps))**3, _RHO_CAP)) if (np.isfinite(d_all) and np.isfinite(d_cat)) else 0.0
            rho_imbalance = rho_cat_rel - rho_an_rel

            # Now scale points for TDA (if requested)
            scale_ref = None
            if args.scale == 'nn':
                scale_ref = mean_nn_distance(all_pts)
            pts_all_s, d_ref = apply_scale(all_pts, args.scale, ref_d=scale_ref)
            pts_cat_s, _    = apply_scale(cat_pts, args.scale, ref_d=d_ref)
            pts_an_s, _     = apply_scale(an_pts,  args.scale, ref_d=d_ref)

            maxdim = 2  # only up to H2 is needed; a flag decides whether H0 features are output
            diags_all = gudhi_pd(pts_all_s, maxdim=maxdim)
            diags_cat = gudhi_pd(pts_cat_s, maxdim=maxdim)
            diags_an  = gudhi_pd(pts_an_s,  maxdim=maxdim)

            rec = {"material_id": mid_full, "cif_file": os.path.basename(cif_path), "kappa_rt": float(row["kappa_rt"]),
                   "n_all": int(len(all_pts)), "n_cat": int(len(cat_pts)), "n_an": int(len(an_pts)),
                   "anion_elements": ",".join(sorted(an_set)), "cation_elements": ",".join(sorted(ca_set)),
                   "supercell_na": args.supercell[0], "supercell_nb": args.supercell[1], "supercell_nc": args.supercell[2],
                   "scale_mode": args.scale, "scale_ref": float(d_ref) if d_ref is not None else float('nan'),
                   "volume": float(vol),
                   # density proxies:
                   "rho_all_dimless": rho_all_dimless,
                   "rho_an_rel": rho_an_rel, "rho_cat_rel": rho_cat_rel, "rho_imbalance": rho_imbalance}

            # dims to export
            dims = []
            if args.include_h0: dims.append((0, "H0"))
            dims.extend([(1, "H1"), (2, "H2")])

            for dim, tag in dims:
                f_all = collect_features(f"ALL_{tag}", diags_all.get(dim, []))
                f_cat = collect_features(f"CAT_{tag}", diags_cat.get(dim, []))
                f_an  = collect_features(f"AN_{tag}",  diags_an.get(dim, []))
                f_all = normalize_features(f_all, args.norm, rec["n_all"], rec["volume"])
                f_cat = normalize_features(f_cat, args.norm, rec["n_cat"], rec["volume"])
                f_an  = normalize_features(f_an,  args.norm, rec["n_an"],  rec["volume"])
                rec.update(f_all); rec.update(f_cat); rec.update(f_an)

            # cross-site features
            if "ALL_H1_sum_pers" in rec:
                rec["X_H1_total_pers_sum"] = rec.get("CAT_H1_sum_pers", 0.0) + rec.get("AN_H1_sum_pers", 0.0)
                rec["X_H1_loop_imbalance_p95"] = rec.get("CAT_H1_p95_pers", 0.0) - rec.get("AN_H1_p95_pers", 0.0)
            if "ALL_H2_sum_pers" in rec:
                rec["X_H2_total_pers_sum"] = rec.get("CAT_H2_sum_pers", 0.0) + rec.get("AN_H2_sum_pers", 0.0)
                rec["X_H2_void_imbalance_p95"] = rec.get("CAT_H2_p95_pers", 0.0) - rec.get("AN_H2_p95_pers", 0.0)

            features.append(rec); n_ok += 1

        except Exception as e:
            print(f"[WARN] {mid_full} failed: {e}")
            n_fail += 1
            continue

    feat_df = pd.DataFrame(features)
    feat_path = os.path.join(args.out_dir, "ion_site_tda_features.csv")
    feat_df.to_csv(feat_path, index=False)
    print({"n_ok": n_ok, "n_no_cif": n_no_cif, "n_fail": n_fail, "features_csv": feat_path})

    if len(feat_df) == 0:
        print("[WARN] No rows processed successfully; skip correlation/scatter.")
        return

    # --- Spearman Top-10 with masks on log10(kappa) ---
    feat_df["_kappa"] = pd.to_numeric(feat_df["kappa_rt"], errors="coerce")
    feat_df = feat_df.loc[feat_df["_kappa"] > 0].copy()
    feat_df["_log10k"] = np.log10(feat_df["_kappa"].values)

    def valid_mask_for_feature(df, col):
        n_an = pd.to_numeric(df.get("n_an"), errors="coerce")
        n_cat = pd.to_numeric(df.get("n_cat"), errors="coerce")
        n_all = pd.to_numeric(df.get("n_all"), errors="coerce")
        if col.startswith("AN_"):
            base = n_an > 0
        elif col.startswith("CAT_"):
            base = n_cat > 0
        elif col.startswith("ALL_"):
            base = n_all > 0
        elif col.startswith("X_"):
            base = (n_an > 0) & (n_cat > 0)
        else:
            base = pd.Series(True, index=df.index)
        vals = pd.to_numeric(df[col], errors="coerce")
        mask = base & np.isfinite(vals) & (vals != 0) & np.isfinite(df["_log10k"])
        return mask, vals

    # include new rho_* in the ranking
    tda_cols = [c for c in feat_df.columns if re.match(r"^(ALL_|CAT_|AN_|X_|rho_)", c)]
    rows = []
    for col in tda_cols:
        if col.startswith("rho_"):
            msk = np.isfinite(feat_df[col]) & np.isfinite(feat_df["_log10k"])
            vals = feat_df[col]
        else:
            msk, vals = valid_mask_for_feature(feat_df, col)
        if msk.sum() < 8:
            continue
        rho, p = spearmanr(vals[msk].values, feat_df.loc[msk, "_log10k"].values)
        if np.isfinite(rho):
            rows.append((col, float(rho), float(p), int(msk.sum())))

    top_df = pd.DataFrame(rows, columns=["Feature", "SpearmanRho_log10k", "p_value", "N_used"])
    if len(top_df) > 0:
        top_df["AbsRho"] = top_df["SpearmanRho_log10k"].abs()
        top_df = top_df.sort_values("AbsRho", ascending=False).drop(columns=["AbsRho"]).head(10)

    top_path = os.path.join(args.out_dir, "ion_site_top10_logk.csv")
    top_df.to_csv(top_path, index=False)
    print({"top10_logk_csv": top_path, "n_top": int(len(top_df))})

    # scatter (log10 κ)
    for feat in top_df["Feature"].tolist():
        try:
            if feat.startswith("rho_"):
                x = feat_df[feat].values
                y = feat_df["_log10k"].values
                msk = np.isfinite(x) & np.isfinite(y)
            else:
                msk, vals = valid_mask_for_feature(feat_df, feat)
                x = vals[msk].values
                y = feat_df.loc[msk, "_log10k"].values
            plt.figure()
            plt.scatter(x, y, s=12, alpha=0.7)
            plt.xlabel(feat); plt.ylabel("log10 κ (W/m·K)"); plt.title(f"{feat} vs log10 κ (N={int(msk.sum())})")
            plt.tight_layout()
            safe = re.sub(r'[^A-Za-z0-9_]+','_', feat)
            plt.savefig(os.path.join(args.out_dir, f"scatter_logk_{safe}.png"), dpi=150)
            plt.close()
        except Exception as e:
            print(f"[WARN] scatter failed for {feat}: {e}")
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Path to .xlsx or .csv")
    ap.add_argument("--sheet", default=0, type=_sheet_arg, help="Sheet index or name (for Excel)")
    ap.add_argument("--cif_root", required=True, help="Directory of CIFs (unzipped)")
    ap.add_argument("--kappa_col", default="thermal conductivity (W/mK)")
    ap.add_argument("--id_col", default="material id for cif")
    ap.add_argument("--out_dir", default="tda_out")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N rows")

    ap.add_argument("--make_ordered", type=str2bool, default=True,
                    help="Convert disordered to ordered proxy when possible")
    ap.add_argument("--supercell", nargs=3, type=int, default=[1,1,1], help="Supercell multipliers na nb nc")
    ap.add_argument("--scale", choices=["none","nn"], default="none",
                    help="Coordinate scaling: none or divide by global mean NN distance")
    ap.add_argument("--norm", choices=["none","per_atom","per_volume"], default="per_atom",
                    help="Normalize *_sum_pers and *_count by atom count or volume")
    ap.add_argument("--include_h0", type=str2bool, default=True, help="Also output H0 features (default: True)")

    # ion-site broad rules
    ap.add_argument("--anion_mode", choices=["relative","fixed"], default="relative",
                    help="Ion-site rule: relative (composition-aware) or fixed (use --anion_fixed)")
    ap.add_argument("--anion_gap", type=float, default=0.15,
                    help="Δ in X >= Xmax - Δ (Pauling) to mark anions when --anion_mode=relative")
    ap.add_argument("--anion_fixed", nargs="*", default=["O","N","F","Cl","Br","I","S","Se","Te","P","C"],
                    help="Fixed anion set when --anion_mode=fixed")
    ap.add_argument("--hydride_rule", choices=["on","off"], default="on",
                    help="Treat H as anion when very electropositive metals present (on/off)")

    # Scheme A: force bipartition
    ap.add_argument("--force_bipartition", choices=["on","off"], default="on",
                    help="Enforce non-empty cation set for multi-element systems (move least EN element to cations)")

    args = ap.parse_args()
    main(args)
