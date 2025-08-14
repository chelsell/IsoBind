#!/usr/bin/env python3
import os, csv, argparse
from collections import defaultdict

def load_pairs(path, cols):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        out = []
        for row in r:
            out.append(tuple(row[c] for c in cols))
        return out

class UF:
    def __init__(self): self.p={}
    def find(self,x):
        self.p.setdefault(x,x)
        if self.p[x]!=x: self.p[x]=self.find(self.p[x])
        return self.p[x]
    def union(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra!=rb: self.p[rb]=ra

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--focus", nargs="+", default=["CHEMBL742"])
    args = ap.parse_args()

    out = args.outdir
    merge_map = load_pairs(os.path.join(out, "merge_map.csv"), ["dup_id","canonical_id"])
    auto_exact = load_pairs(os.path.join(out, "auto_exact.csv"), ["dup_id","canonical_id","rule","score"])
    fp_merges  = load_pairs(os.path.join(out, "fp_merges.csv"),  ["a","b","rule","fp"])
    nmr_merges = load_pairs(os.path.join(out, "nmr_merges.csv"), ["a","b","rule","sim13c","fp"])

    # Build UF graph from all logged edges (direction doesn’t matter)
    uf = UF()
    ids_seen = set()
    for a,b in [(d,c) for d,c in merge_map] + [(d,c) for d,c,_,_ in auto_exact] \
             + [(a,b) for a,b,_,_ in fp_merges] + [(a,b) for a,b,_,_,_ in nmr_merges]:
        uf.union(a,b)
        ids_seen.add(a); ids_seen.add(b)

    # Bucket by component
    comps = defaultdict(list)
    for cid in ids_seen:
        comps[uf.find(cid)].append(cid)

    # Build quick edge lookups (so we can cite rules/scores)
    exact_set = {(d,c):s for d,c,_,s in auto_exact}
    fp_set    = {(a,b):float(fp) for a,b,_,fp in fp_merges} | {(b,a):float(fp) for a,b,_,fp in fp_merges}
    nmr_set   = {(a,b):(float(sim), float(fp)) for a,b,_,sim,fp in nmr_merges} \
              | {(b,a):(float(sim), float(fp)) for a,b,_,sim,fp in nmr_merges}

    # Canonical for each member from merge_map, if present
    canon_of = {d:c for d,c in merge_map}

    def canonical_for(cid, group):
        # If not in merge_map (i.e., itself canonical), assume the minimal by ChEMBL number in its group
        def chembl_num(x):
            try: return int(x.replace("CHEMBL",""))
            except: return 10**12
        return canon_of.get(cid) or min(group, key=chembl_num)

    # Report
    for target in args.focus:
        if target not in ids_seen:
            print(f"[{target}] not present in the dedup logs. If it exists in the input but had no edges, it is canonical alone.")
            continue
        root = uf.find(target)
        members = sorted(comps[root], key=lambda x: int(x.replace("CHEMBL","")) if x.startswith("CHEMBL") else 10**12)
        canon = canonical_for(target, members)
        print(f"\n=== Component for {target} ===")
        print(f"Canonical: {canon}")
        print(f"Members ({len(members)}): {', '.join(members)}")

        # For each non-canonical, try to explain how it merged (prefer most specific evidence)
        print("\nHow each member merged:")
        for m in members:
            if m == canon:
                print(f"  {m}  <- canonical")
                continue
            why = []
            # exact?
            if (m, canon) in exact_set or (canon, m) in exact_set:
                s = exact_set.get((m, canon), exact_set.get((canon, m), "1.0"))
                why.append(f"Stage1: Exact InChIKey (score={s})")
            # direct FP/NMR edge to canon?
            if (m, canon) in fp_set:
                why.append(f"Stage2: FP Tanimoto={fp_set[(m, canon)]:.3f}")
            if (m, canon) in nmr_set:
                sim, fp = nmr_set[(m, canon)]
                why.append(f"Stage3: Sim13C={sim:.3f} (FP={fp:.3f})")
            # If no direct edge to canon, find any edge inside component that connected it
            if not why:
                # search any partner inside the component
                found = False
                for other in members:
                    if other == m: 
                        continue
                    if (m, other) in fp_set:
                        why.append(f"Stage2 via {other}: FP Tanimoto={fp_set[(m, other)]:.3f}")
                        found = True
                        break
                    if (m, other) in nmr_set:
                        sim, fp = nmr_set[(m, other)]
                        why.append(f"Stage3 via {other}: Sim13C={sim:.3f} (FP={fp:.3f})")
                        found = True
                        break
                if not found and ((m, canon) in canon_of or (canon, m) in canon_of):
                    why.append("Merged via earlier exact block edge")
            expl = " | ".join(why) if why else "No edge recorded (check logs); may be canonical swap within block"
            print(f"  {m}  -> {canon}  :: {expl}")

if __name__ == "__main__":
    main()

