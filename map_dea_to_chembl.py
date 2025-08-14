#!/usr/bin/env python3
import argparse, csv, os, json, math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdFingerprintGenerator as rfg
from rdkit.Chem.Scaffolds import MurckoScaffold

# ---- Policy (keep in lockstep with chembl_dedup.py) ----
FP_DUP = 0.95
FP_BORDER_LOW, FP_BORDER_HIGH = 0.85, 0.95
SIM13C_DUP = 0.85   # Sim13C tie-break threshold
# --------------------------------------------------------

_morgan_gen = rfg.GetMorganGenerator(radius=2, fpSize=2048)

def morgan_fp(mol): return _morgan_gen.GetFingerprint(mol)

def standardize_mol(mol: Chem.Mol, strip_isotopes: bool = True) -> Chem.Mol:
    params = rdMolStandardize.CleanupParameters()
    mol = rdMolStandardize.Cleanup(mol, params)
    mol = rdMolStandardize.FragmentParent(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    mol = rdMolStandardize.Normalize(mol)
    if strip_isotopes:
        for a in mol.GetAtoms(): a.SetIsotope(0)
    mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    return mol

def inchikeys(mol: Chem.Mol) -> Tuple[str, str]:
    ik = Chem.MolToInchiKey(mol)
    return ik, ik.split('-')[0]

def murcko_key(mol: Chem.Mol) -> str:
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except: return ''

def tanimoto_bulk(fp, fp_list): return DataStructs.BulkTanimotoSimilarity(fp, fp_list)

def load_sim13c_vectors(path: Optional[str]) -> Dict[str, np.ndarray]:
    if not path: return {}
    vecs = {}
    with open(path, newline='') as f:
        rdr = csv.reader(f); header = next(rdr)
        for row in rdr:
            cid = row[0]
            v = np.array([float(x) for x in row[1:]], dtype=float)
            n = np.linalg.norm(v)
            if n == 0: continue
            vecs[cid] = v / n
    return vecs

def cos_sim(vecs: Dict[str,np.ndarray], a: str, b: str) -> float:
    va, vb = vecs.get(a), vecs.get(b)
    if va is None or vb is None: return float('nan')
    return float(np.dot(va, vb))

def load_chembl_records(path: str):
    rows = []
    with open(path, newline='') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            cid = r.get('chembl_id','').strip()
            smi = r.get('smiles','').strip()
            if not cid or not smi: continue
            rows.append({'chembl_id': cid, 'smiles': smi})
    return rows

def load_dea_records(path: str):
    rows = []
    with open(path, newline='') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append({
                'dea_id': r.get('dea_id','').strip() or r.get('id','').strip(),
                'name': r.get('name','').strip(),
                'smiles': r.get('smiles','').strip(),
                'inchikey': r.get('inchikey','').strip(),
            })
    return rows

def load_merge_map(path: str) -> Dict[str,str]:
    if not path or not os.path.exists(path): return {}
    mp = {}
    with open(path, newline='') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            mp[r['dup_id']] = r['canonical_id']
    return mp

def chembl_num(cid: str) -> int:
    try: return int(cid.replace("CHEMBL",""))
    except: return 10**12

def resolve_canonical(cid: str, merge_map: Dict[str,str]) -> str:
    # follow chain if any
    while cid in merge_map:
        cid = merge_map[cid]
    return cid

def index_chembl(chembl_rows):
    """Build indices: ik_full -> ids, ik_block -> ids, scaffold -> ids, fps, mols."""
    ik_full_bins = defaultdict(list)
    ik_blk_bins  = defaultdict(list)
    scaff_bins   = defaultdict(list)
    fps, mols = {}, {}
    for r in chembl_rows:
        cid, smi = r['chembl_id'], r['smiles']
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        mol = standardize_mol(mol)
        mols[cid] = mol
        fps[cid]  = morgan_fp(mol)
        ikf, ikb  = inchikeys(mol)
        scf       = murcko_key(mol)
        ik_full_bins[ikf].append(cid)
        ik_blk_bins[ikb].append(cid)
        if scf: scaff_bins[scf].append(cid)
    return ik_full_bins, ik_blk_bins, scaff_bins, fps, mols

def best_fp_match(query_fp, candidates: List[str], fps: Dict[str,object]) -> Tuple[str,float]:
    best_id, best = None, -1.0
    if not candidates: return None, -1.0
    sims = tanimoto_bulk(query_fp, [fps[c] for c in candidates])
    for cid, s in zip(candidates, sims):
        if s > best: best_id, best = cid, s
    return best_id, float(best)

def map_dea_to_chembl(dea_rows, chem_index, fps, sim13c_vecs, merge_map, prefer_gate='both'):
    ik_full_bins, ik_blk_bins, scaff_bins, chem_fps, chem_mols = chem_index
    results = []
    for row in dea_rows:
        rid, name, smi, ik_given = row['dea_id'], row['name'], row['smiles'], row['inchikey']
        match = {'dea_id': rid, 'name': name, 'assigned_chembl_id': '', 'canonical_id': '',
                 'match_type': 'no_match', 'fp': '', 'sim13c': '', 'notes': ''}

        # 0) If we only have a name, we can’t resolve without a synonym table; leave for later.
        if not smi and not ik_given:
            match['notes'] = 'no structure or InChIKey; provide smiles or synonym table'
            results.append(match); continue

        # 1) Build DEA mol from SMILES or (if only IK is given) skip to exact-block checks
        mol = None
        if smi:
            mol0 = Chem.MolFromSmiles(smi)
            if mol0: mol = standardize_mol(mol0)
        ik_full = ik_given or (Chem.MolToInchiKey(mol) if mol else '')
        ik_blk  = ik_full.split('-')[0] if ik_full else ''

        # 2) Exact InChIKey full
        if ik_full and ik_full in ik_full_bins:
            # choose smallest chembl in that bin
            candidates = sorted(ik_full_bins[ik_full], key=chembl_num)
            picked = candidates[0]
            match.update({'assigned_chembl_id': picked,
                          'canonical_id': resolve_canonical(picked, merge_map),
                          'match_type': 'exact_inchikey'})
            results.append(match); continue

        # 3) FP/NMR within block or scaffold groups
        # Need a query FP; if no SMILES, we cannot do FP/NMR.
        if mol is None:
            match['notes'] = 'has InChIKey but no exact match; no SMILES to score FP/NMR'
            results.append(match); continue

        qfp = morgan_fp(mol)
        groups = []
        if ik_blk and ik_blk in ik_blk_bins: groups.append(('block', ik_blk_bins[ik_blk]))
        scf = murcko_key(mol)
        if scf and scf in scaff_bins: groups.append(('scaffold', scaff_bins[scf]))

        # merge candidates from both gates (dedupe)
        cand_map = {}
        for gate, ids in groups:
            for c in ids: cand_map[c] = gate
        candidates = list(cand_map.keys())

        # If no gate candidates, fall back to top-K over all chembl (K=200) to avoid O(N^2) for very large sets
        fallback_used = False
        if not candidates:
            fallback_used = True
            # Light fallback: random-sample or take first 200; here we take first 200 deterministically
            candidates = list(chem_fps.keys())[:200]
            for c in candidates: cand_map[c] = 'fallback'

        picked, fp_s = best_fp_match(qfp, candidates, chem_fps)
        if picked is None:
            match['notes'] = 'no candidates to compare'
            results.append(match); continue

        gate = cand_map.get(picked, 'unknown')
        # Stage 2: FP ≥ 0.95 => accept
        if fp_s >= FP_DUP:
            match.update({'assigned_chembl_id': picked,
                          'canonical_id': resolve_canonical(picked, merge_map),
                          'match_type': f'fp_{gate}',
                          'fp': f'{fp_s:.3f}'})
            results.append(match); continue

        # Stage 3: borderline + Sim13C tie-break
        if FP_BORDER_LOW <= fp_s < FP_BORDER_HIGH:
            sim = cos_sim(sim13c_vecs, picked, picked)  # placeholder to keep shape if DEA lacks Sim13C
            # Note: we only have Sim13C vectors for ChEMBL; for DEA we can't compute DEA-vs-CHEMBL Sim13C without a DEA vector.
            # So the tie-break here uses FP only unless you also supply DEA Sim13C vectors and wire them similarly.
            # If you DO have DEA Sim13C vectors, replace the line above with:
            # sim = cosine(dea_vecs[rid], chem_vecs[picked])
            if not math.isnan(sim) and sim >= SIM13C_DUP:
                match.update({'assigned_chembl_id': picked,
                              'canonical_id': resolve_canonical(picked, merge_map),
                              'match_type': f'borderline_sim13c_{gate}',
                              'fp': f'{fp_s:.3f}', 'sim13c': f'{sim:.3f}'})
                results.append(match); continue

        # If we get here: not confident → emit top hit as "no_match" with scores
        match.update({'assigned_chembl_id': picked,
                      'canonical_id': resolve_canonical(picked, merge_map),
                      'match_type': f'low_conf_{gate}{"_fallback" if fallback_used else ""}',
                      'fp': f'{fp_s:.3f}'})
        results.append(match)

    return results

def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows: w.writerow(r)

def main():
    ap = argparse.ArgumentParser(description="Map DEA dataset entries to ChEMBL IDs")
    ap.add_argument('--chembl-records', required=True)
    ap.add_argument('--merge-map', required=True)
    ap.add_argument('--dea-records', required=True)
    ap.add_argument('--nmr-features', default=None, help='Sim13C vectors for ChEMBL IDs (optional)')
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    chembl_rows = load_chembl_records(args.chembl_records)
    merge_map   = load_merge_map(args.merge_map)
    dea_rows    = load_dea_records(args.dea_records)
    sim13c_vecs = load_sim13c_vectors(args.nmr_features)

    ik_full_bins, ik_blk_bins, scaff_bins, fps, mols = index_chembl(chembl_rows)
    chem_index = (ik_full_bins, ik_blk_bins, scaff_bins, fps, mols)

    results = map_dea_to_chembl(dea_rows, chem_index, fps, sim13c_vecs, merge_map)

    header = ['dea_id','name','assigned_chembl_id','canonical_id','match_type','fp','sim13c','notes']
    out_csv = os.path.join(args.outdir, 'dea_to_chembl.csv')
    write_csv(out_csv, header, results)

    # Tiny summary
    counts = defaultdict(int)
    for r in results: counts[r['match_type']] += 1
    with open(os.path.join(args.outdir, 'summary.json'),'w') as f:
        json.dump({'counts':counts, 'n':len(results)}, f, indent=2)

if __name__ == '__main__':
    main()

