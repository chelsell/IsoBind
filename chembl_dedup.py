#!/usr/bin/env python3
# chembl_dedup.py
# Hierarchical ChEMBL dedup: InChIKey(full) → Morgan FP → Sim13C tie-breaker
# Policy:
#   1) Full InChIKey match            => merge
#   2) Same first-block + FP >= 0.95  => merge
#   3) Same first-block + 0.85<=FP<0.95 AND Sim13C >= 0.85 => merge
#   4) Else: no-merge (audit if FP<0.90 but NMR>=0.85)

import argparse, csv, json, os, math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# add with other imports
from tqdm.auto import tqdm

def pbar(iterable, enabled: bool, desc: str):
    return tqdm(iterable, desc=desc) if enabled else iterable


# Create a generator once, reuse for all molecules
_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

FP_DUP = 0.95
FP_BORDER_LOW, FP_BORDER_HIGH = 0.85, 0.95
SIM13C_DUP = 0.85         # raised because we're not using a fused metric
QUARANTINE_FP = 0.90      # audit if FP<0.90 but Sim13C>=0.85

def standardize_mol(mol: Chem.Mol, strip_isotopes: bool = True) -> Chem.Mol:
    params = rdMolStandardize.CleanupParameters()
    mol = rdMolStandardize.Cleanup(mol, params)
    # More robust parent selection than LargestFragmentChooser:
    mol = rdMolStandardize.FragmentParent(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    mol = rdMolStandardize.Normalize(mol)
    if strip_isotopes:
        for a in mol.GetAtoms():
            a.SetIsotope(0)
    return mol

def murcko_key(mol: Chem.Mol) -> str:
    # Bemis–Murcko scaffold as a SMILES key; empty string if fail
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ''

def tautomer_canonicalize(mol: Chem.Mol) -> Chem.Mol:
    te = rdMolStandardize.TautomerEnumerator()
    return te.Canonicalize(mol)

def inchikeys(mol: Chem.Mol) -> Tuple[str, str]:
    ik = Chem.MolToInchiKey(mol)
    return ik, ik.split('-')[0]

def morgan_fp(mol):
    return _morgan_gen.GetFingerprint(mol)

def tanimoto_bulk(fp, fp_list):
    return DataStructs.BulkTanimotoSimilarity(fp, fp_list)


def load_sim13c_vectors(path: str) -> Tuple[Dict[str, np.ndarray], int]:
    """
    CSV: chembl_id,f1,f2,...,fK
    Returns dict and dimensionality K.
    """
    if path is None:
        return {}, 0
    vecs = {}
    with open(path, newline='') as f:
        rdr = csv.reader(f)
        header = next(rdr)
        if len(header) < 2 or header[0].lower() != 'chembl_id':
            raise ValueError("sim13c_vectors.csv must start with 'chembl_id' then numeric columns.")
        for row in rdr:
            cid = row[0]
            v = np.array([float(x) for x in row[1:]], dtype=float)
            nrm = np.linalg.norm(v)
            if nrm == 0:
                continue
            vecs[cid] = v / nrm
    dim = len(next(iter(vecs.values()))) if vecs else 0
    return vecs, dim

def sim13c_cosine(cid_a: str, cid_b: str, vecs: Dict[str, np.ndarray]) -> float:
    va = vecs.get(cid_a); vb = vecs.get(cid_b)
    if va is None or vb is None:
        return float('nan')
    # cosine with pre-normalized vectors
    return float(np.dot(va, vb))

def load_records(path: str):
    rows = []
    with open(path, newline='') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if not r.get('chembl_id') or not r.get('smiles'):
                continue
            rows.append({
                'chembl_id': r['chembl_id'].strip(),
                'smiles': r['smiles'].strip(),
                'assay_count': int(r.get('assay_count', 0) or 0),
                'std_smiles': r.get('std_smiles', '').strip(),
            })
    return rows

def build_blocks(records, strip_isotopes=True):    
    exact_bins = defaultdict(list)
    block_bins = defaultdict(list)
    scaff_bins  = defaultdict(list)
    mols, fps = {}, {}
    ik_full_by_id, ik_blk_by_id = {}, {}

    for r in pbar(records, True, 'Standardize + block'):
        cid = r['chembl_id']
        mol = Chem.MolFromSmiles(r['smiles'])
        if mol is None:
            continue
        mol = standardize_mol(mol, strip_isotopes=strip_isotopes)
        mol = tautomer_canonicalize(mol)   # ON (confirmed)
        ik_full, ik_blk = inchikeys(mol)
        exact_bins[ik_full].append(cid)
        block_bins[ik_blk].append(cid)

        scaff = murcko_key(mol)
        if scaff:
            scaff_bins[scaff].append(cid)

        mols[cid] = mol
        fps[cid] = morgan_fp(mol)
        ik_full_by_id[cid] = ik_full
        ik_blk_by_id[cid] = ik_blk
    return exact_bins, block_bins, scaff_bins, mols, fps, ik_full_by_id, ik_blk_by_id

def iter_gate_groups(block_bins, scaff_bins, mode='both'):
    if mode == 'block':
        for g in block_bins.values():
            yield g
    elif mode == 'scaffold':
        for g in scaff_bins.values():
            yield g
    else:  # 'both' -> union of all groups, but avoid duplicates by using frozenset
        seen = set()
        for g in list(block_bins.values()) + list(scaff_bins.values()):
            fs = frozenset(g)
            if fs and fs not in seen:
                seen.add(fs)
                yield list(fs)


def fp_near_dups(groups, fps, thresh=FP_DUP):
    pairs = []
    for ids in groups:
        ids = [i for i in ids if i in fps]
        for i, a in enumerate(ids):
            sims = tanimoto_bulk(fps[a], [fps[b] for b in ids])
            for j, s in enumerate(sims):
                b = ids[j]
                if b == a:
                    continue
                if s >= thresh:
                    pairs.append((a,b,'FP',s))
    return pairs

def borderline_candidates(groups, fps, low=FP_BORDER_LOW, high=FP_BORDER_HIGH):
    cands = []
    for ids in groups:
        ids = [i for i in ids if i in fps]
        for i, a in enumerate(ids):
            sims = tanimoto_bulk(fps[a], [fps[b] for b in ids])
            for j, s in enumerate(sims):
                b = ids[j]
                if b == a:
                    continue
                if low <= s < high:
                    cands.append((a,b,s))
    return cands

class UF:
    def __init__(self): self.p={}
    def find(self,x):
        self.p.setdefault(x,x)
        if self.p[x]!=x: self.p[x]=self.find(self.p[x])
        return self.p[x]
    def union(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra!=rb: self.p[rb]=ra

def chembl_num(cid: str) -> int:
    try:
        return int(cid.replace('CHEMBL',''))
    except:
        return 10**12

def choose_canonical(ids: List[str], meta: Dict[str,dict]) -> str:
    def key(cid):
        return (chembl_num(cid), -meta.get(cid,{}).get('assay_count',0), meta.get(cid,{}).get('std_smiles',''))
    return sorted(ids, key=key)[0]


def nmr_tiebreak(cands, sim13c_vecs, fp_quarantine=QUARANTINE_FP, nmr_thresh=SIM13C_DUP):
    merges, quarantine = [], []
    for a,b,fp_s in pbar(cands, True, 'Sim13C tie-break'):    
        s = sim13c_cosine(a,b, sim13c_vecs)
        if math.isnan(s):
            continue
        if s >= nmr_thresh:
            merges.append((a,b,'Sim13C',s,fp_s))
            if fp_s < fp_quarantine:
                quarantine.append((a,b,fp_s,s))
    return merges, quarantine

def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser(description="ChEMBL dedup (FP + Sim13C tie-breaker)")
    ap.add_argument('--records', required=True, help='CSV with chembl_id,smiles[,assay_count,std_smiles]')
    ap.add_argument('--nmr-features', default=None, help='CSV of Sim13C vectors: chembl_id,f1,...,fK')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--no-progress', action='store_true', help='Disable tqdm progress bars')
    ap.add_argument('--gate-mode', choices=['block','scaffold','both'], default='both',
                help='Where to search for near-duplicates: by InChIKey first-block, scaffold, or both (default).')
    ap.add_argument('--keep-isotopes', action='store_true',
                help='If set, do NOT strip isotopes during standardization. Default is to strip.')

    args = ap.parse_args()
    show_pbar = not args.no_progress
    gate_mode = args.gate_mode
    strip_isotopes = not args.keep_isotopes
    records = load_records(args.records)
    meta = {r['chembl_id']: r for r in records}
    exact_bins, block_bins, skel_bins, mols, fps, ik_full_by_id, ik_blk_by_id = build_blocks(records)

    # Stage 1: exact merges
    uf = UF()
    exact_merges = []
    for ids in exact_bins.values():
        if len(ids) > 1:
            root = ids[0]
            for dup in ids[1:]:
                uf.union(root, dup)
                exact_merges.append((dup, root, 'InChIKeyFull', 1.0))

    # Stage 2: FP near-dups within first-block bins

    groups = list(iter_gate_groups(block_bins, skel_bins, gate_mode))

    fp_pairs = fp_near_dups(groups, fps, FP_DUP)
    for a,b,_t,s in fp_pairs:
        uf.union(a,b)

    # Stage 3: borderline → Sim13C
    border = borderline_candidates(groups, fps, FP_BORDER_LOW, FP_BORDER_HIGH)
    sim13c_vecs, dim = load_sim13c_vectors(args.nmr_features)
    nmr_pairs, quarantine = nmr_tiebreak(border, sim13c_vecs, QUARANTINE_FP, SIM13C_DUP)
    for a,b,_t,sim,fp_s in nmr_pairs:
        uf.union(a,b)

    # Build buckets per UF root (bounded to first-block membership implicitly via edges)
    buckets = defaultdict(list)
    for blk_ids in pbar(skel_bins.values(), show_pbar, 'Build UF buckets'):
        for cid in blk_ids:
            buckets[uf.find(cid)].append(cid)

    # Canonical selection + merge map
    merge_map_rows = []
    for root, ids in buckets.items():
        canon = choose_canonical(ids, meta)
        for cid in ids:
            if cid != canon:
                merge_map_rows.append((cid, canon))

    # Outputs
    outdir = args.outdir
    write_csv(os.path.join(outdir, 'merge_map.csv'), ['dup_id','canonical_id'], merge_map_rows)
    write_csv(os.path.join(outdir, 'auto_exact.csv'), ['dup_id','canonical_id','rule','score'], exact_merges)
    write_csv(os.path.join(outdir, 'fp_merges.csv'), ['a','b','rule','fp'], fp_pairs)
    write_csv(os.path.join(outdir, 'nmr_merges.csv'), ['a','b','rule','sim13c','fp'], nmr_pairs)
    write_csv(os.path.join(outdir, 'quarantine.csv'), ['a','b','fp','sim13c'], quarantine)

    stats = {
        'records_in': len(records),
        'unique_first_blocks': len(skel_bins),
        'exact_merge_edges': len(exact_merges),
        'fp_merge_edges': len(fp_pairs),
        'nmr_merge_edges': len(nmr_pairs),
        'quarantine_edges': len(quarantine),
        'merge_pairs_out': len(merge_map_rows),
        'sim13c_dim': dim,
        'policy': {
            'FP_DUP': FP_DUP,
            'FP_BORDER': [FP_BORDER_LOW, FP_BORDER_HIGH],
            'SIM13C_DUP': SIM13C_DUP,
            'QUARANTINE_FP': QUARANTINE_FP,
            'tautomer_canonicalization': True
        }
    }
    with open(os.path.join(outdir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

if __name__ == '__main__':
    main()

