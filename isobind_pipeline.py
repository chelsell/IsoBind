#!/usr/bin/env python3
# ⚠️ Review carefully
import argparse, csv, math, sys, json
from collections import defaultdict, Counter
from itertools import combinations
import pandas as pd
import numpy as np

# ---------- utilities ----------
def read_csv_loose(path):
    df = pd.read_csv(path)
    # normalize headers lower-case for robust access
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def coerce_float(x):
    try: return float(x)
    except: return np.nan

# ---------- 1) aggregate ----------
def cmd_aggregate(args):
    prof = read_csv_loose(args.profiles)
    memb = read_csv_loose(args.members)

    # expected cols
    c_chem = [c for c in prof.columns if c.startswith("chembl")]
    c_tgt  = [c for c in prof.columns if "target" in c]
    c_mtyp = [c for c in prof.columns if "metric_type" in c]
    c_val  = [c for c in prof.columns if c == "value" or c.endswith(":value") or "value" in c]
    if not c_chem or not c_tgt or not c_mtyp or not c_val:
        raise ValueError(f"profiles.csv missing required columns; got {prof.columns.tolist()}")

    prof = prof.rename(columns={
        c_chem[0]: "chembl_id",
        c_tgt[0]:  "target_id",
        c_mtyp[0]: "metric_type",
        c_val[0]:  "value"
    })

    if "assay_conf" not in prof.columns:
        prof["assay_conf"] = np.nan

    prof["chembl_id"] = prof["chembl_id"].astype(str)
    prof["target_id"] = prof["target_id"].astype(str)
    prof["metric_type"] = prof["metric_type"].astype(str)
    prof["value"] = prof["value"].apply(coerce_float)

    # members map: bucket_id, member (chembl_id)
    mcols = memb.columns
    if "bucket_id" not in mcols or "member" not in mcols:
        # try to guess
        b = [c for c in mcols if "bucket" in c or "root" in c][0]
        m = [c for c in mcols if "member" in c or "chembl" in c or "id" == c][0]
        memb = memb.rename(columns={b:"bucket_id", m:"member"})
    memb["member"] = memb["member"].astype(str)

    P = prof.merge(memb, left_on="chembl_id", right_on="member", how="inner")
    if len(P)==0:
        raise ValueError("Join produced 0 rows — check that CHEMBL IDs in profiles match 'member' values.")

    # normalize metric names (split unit suffixes like Ki_nM)
    # keep raw metric_type for audit
    P["metric_type_raw"] = P["metric_type"]
    m = P["metric_type"].str.extract(r"^(?P<metric>Ki|IC50|Kd|EC50|pKi|pIC50)(?:[_\-\s]?(?P<unit>nM|uM|µM|pM))?$",
                                     expand=True)
    P["metric"] = m["metric"].fillna(P["metric_type"])
    P["unit"] = m["unit"].fillna(P.get("unit", pd.Series(index=P.index, dtype='object')))

    # precedence: numeric binding first, then everything else
    order = {"pKi":0, "pIC50":0, "Ki":1, "IC50":2, "Kd":3, "EC50":4}
    P["metric_rank"] = P["metric"].map(order).fillna(9).astype(int)

    # to a common "potency" score:
    # - if pKi/pIC50 -> use negative to keep "lower is stronger" convention
    # - if Ki/IC50/Kd/EC50 with units -> log10(nM)
    def to_score(row):
        m = row["metric"]; val = row["value"]; unit = str(row["unit"] or "").lower()
        if pd.isna(val): return np.nan
        if m in ("pKi","pIC50"):
            return -float(val)  # higher pKi -> more potent -> smaller (negative) score
        if m in ("Ki","IC50","Kd","EC50"):
            # convert to nM then log10
            factor = 1.0
            if unit in ("nm","nanomolar","na"): factor = 1.0
            elif unit in ("um","µm","micromolar"): factor = 1000.0
            elif unit in ("pm","picomolar"): factor = 0.001
            # unknown unit: assume nM; you can tighten this if you have units
            try:
                return math.log10(val*factor)
            except:
                return np.nan
        # fallback: treat as inhibition % => larger is stronger; convert to negative fraction
        return -float(val)
    P["score"] = P.apply(to_score, axis=1)

    # aggregate within (bucket,target)
    def agg_group(g: pd.DataFrame):
        g = g.sort_values(["metric_rank", "assay_conf"], ascending=[True, False])
        g_num = g[g["score"].notna()]
        if len(g_num):
            v = g_num["score"].median()
            src = "median_score_bestmetric"
        else:
            v = np.nan
            src = "no_numeric"
        return pd.Series({
            "score": v,
            "n_rows": len(g),
            "best_metric": g.iloc[0]["metric"],
            "best_unit": g.iloc[0]["unit"],
            "src": src
        })

    agg = (P.groupby(["bucket_id","target_id"]).apply(agg_group).reset_index())
    # drop NaN scores only if you truly want pure numeric ranks
    agg = agg[agg["score"].notna()].reset_index(drop=True)

    agg.to_csv(args.out, index=False)
    print(f"[aggregate] wrote {args.out} with {len(agg)} (bucket,target) rows")

# ---------- 2) ranks ----------
def cmd_ranks(args):
    bp = read_csv_loose(args.bucket_profiles)
    need = {"bucket_id","target_id","score"}
    if not need.issubset(set(bp.columns)):
        raise ValueError(f"bucket_profiles missing {need}, has {bp.columns.tolist()}")
    # lower score = stronger -> sort ascending
    bp = bp.sort_values(["bucket_id","score","target_id"])
    rows = []
    for bkt, g in bp.groupby("bucket_id"):
        g = g.reset_index(drop=True)
        for i, r in g.iterrows():
            rows.append((bkt, r["target_id"], i+1, r["score"]))
    out = pd.DataFrame(rows, columns=["bucket_id","target_id","rank","score"])
    out.to_csv(args.out, index=False)
    print(f"[ranks] wrote {args.out} with {len(out)} rows")

# ---------- 3) RBO (SIGIR 2024) ----------
def _import_sigir_rbo(rbo_path: str):
    import sys, importlib
    if rbo_path and rbo_path not in sys.path:
        sys.path.append(rbo_path)
    try:
        mod = importlib.import_module("rbo.Python.rbo")
    except Exception as e:
        raise ImportError(
            f"Could not import SIGIR 2024 RBO from '{rbo_path}'. "
            f"Expected module 'rbo.Python.rbo'. Error: {e}"
        )
    # sanity: require a callable named 'rbo'
    if not hasattr(mod, "rbo") or not callable(getattr(mod, "rbo")):
        raise ImportError(
            f"'rbo.Python.rbo' does not expose a callable 'rbo'. "
            f"Available: {dir(mod)}"
        )
    return mod.rbo

def cmd_rbo(args):
    rt = read_csv_loose(args.ranked)
    if not {"bucket_id","target_id","rank"}.issubset(rt.columns):
        raise ValueError("ranked_targets.csv must have bucket_id,target_id,rank")

    # build lists (deterministic order by rank then target_id)
    lists = (rt.sort_values(["bucket_id","rank","target_id"])
               .groupby("bucket_id")["target_id"].apply(list).to_dict())
    buckets = list(lists.keys())
    if not buckets:
        raise ValueError("No ranked lists found.")

    # import tie-aware RBO from SIGIR 2024 impl
    rbo_fn = _import_sigir_rbo(getattr(args, "rbo_path", None))

    # identity smoke test
    test_b = buckets[0]
    ident = rbo_fn(lists[test_b], lists[test_b], p=args.p)["ext"]
    if not (abs(ident - 1.0) < 1e-12):
        raise AssertionError(f"RBO identity check failed (got {ident})")

    # pairwise distances
    from itertools import combinations
    rows = []
    for a, b in combinations(buckets, 2):
        s = rbo_fn(lists[a], lists[b], p=args.p)["ext"]
        # clamp just in case
        s = max(0.0, min(1.0, float(s)))
        d = 1.0 - s
        rows.append((a, b, d))

    out = pd.DataFrame(rows, columns=["bucket_i","bucket_j","distance"])
    out.to_csv(args.out, index=False)
    print(f"[rbo] (SIGIR) wrote {args.out} with {len(out)} distances across {len(buckets)} buckets")

# ---------- 4) embed ----------
def cmd_embed(args):
    dist = read_csv_loose(args.dist)
    ids = sorted(set(dist["bucket_i"]).union(dist["bucket_j"]))
    idx = {k:i for i,k in enumerate(ids)}
    n = len(ids)
    M = np.zeros((n,n), dtype=float)
    for _,r in dist.iterrows():
        i,j = idx[r["bucket_i"]], idx[r["bucket_j"]]
        d = float(r["distance"])
        M[i,j] = M[j,i] = d
    # fill diagonal
    np.fill_diagonal(M, 0.0)

    from sklearn.manifold import MDS
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0, normalized_stress="auto")
    X = mds.fit_transform(M)
    pd.DataFrame({"bucket_id": ids, "x": X[:,0], "y": X[:,1]}).to_csv(args.out, index=False)
    print(f"[embed] wrote {args.out}  | stress={mds.stress_:.6f}")

# ---------- 5) qc ----------
def cmd_qc(args):
    # quick structural sentinels
    dist = read_csv_loose(args.dist)
    # symmetry
    pairs = {(r["bucket_i"], r["bucket_j"]): r["distance"] for _,r in dist.iterrows()}
    asym = []
    for (a,b),d in list(pairs.items()):
        if (b,a) in pairs and abs(pairs[(b,a)] - d) > 1e-9:
            asym.append(((a,b), d, pairs[(b,a)]))
    print(f"[qc] asymmetric pairs: {len(asym)} (expected 0 in our format)")

    # nearest neighbors (collision hunt)
    nn = defaultdict(lambda: (None, 9e9))
    for _,r in dist.iterrows():
        a,b,d = r["bucket_i"], r["bucket_j"], float(r["distance"])
        if d < nn[a][1]: nn[a] = (b,d)
        if d < nn[b][1]: nn[b] = (a,d)
    nn_rows = [(k, v[0], v[1]) for k,v in nn.items()]
    nn_df = pd.DataFrame(nn_rows, columns=["bucket_id","nn_bucket","nn_distance"]).sort_values("nn_distance")
    print("[qc] 20 closest pairs by distance (should be mechanistically related):")
    print(nn_df.head(20).to_string(index=False))

    # optional: make sure some knowns exist and are distinct (if canonical_map provided)
    if args.canon:
        canon = read_csv_loose(args.canon)
        # pick a few if present
        sample_names = ["CHEMBL742","CHEMBL45","CHEMBL2106195","CHEMBL192"]
        if {"bucket_id","canonical_chembl_id"}.issubset(canon.columns):
            look = canon[canon["canonical_chembl_id"].isin(sample_names)]
            if len(look):
                print("[qc] known sentinel buckets:")
                print(look.to_string(index=False))
            else:
                print("[qc] sentinel CHEMBL IDs not present in canonical_map.csv (ok if not in dataset)")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(prog="isobind_pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("aggregate", help="aggregate profiles by bucket")
    a.add_argument("--profiles", required=True)
    a.add_argument("--members", required=True)
    a.add_argument("--out", default="bucket_profiles.csv")
    a.set_defaults(func=cmd_aggregate)

    r = sub.add_parser("ranks", help="build ranked target lists per bucket")
    r.add_argument("--bucket-profiles", required=True)
    r.add_argument("--out", default="ranked_targets.csv")
    r.set_defaults(func=cmd_ranks)

    r = sub.add_parser("rbo", help="compute RBO distances (SIGIR 2024 impl)")
    r.add_argument("--ranked", required=True)
    r.add_argument("--out", default="isobind_distances.csv")
    r.add_argument("--p", type=float, default=0.98)
    r.add_argument("--rbo-path", default="/home/cole/code/sigir2024-rbo/",
               help="Path containing the 'rbo' package (expects rbo.Python.rbo)")
    r.set_defaults(func=cmd_rbo)

    e = sub.add_parser("embed", help="SMACOF 2D embedding")
    e.add_argument("--dist", required=True)
    e.add_argument("--out", default="isobind_embedding2d.csv")
    e.set_defaults(func=cmd_embed)

    q = sub.add_parser("qc", help="quick sanity checks")
    q.add_argument("--dist", required=True)
    q.add_argument("--ranked", required=False)
    q.add_argument("--canon", required=False)
    q.set_defaults(func=cmd_qc)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

