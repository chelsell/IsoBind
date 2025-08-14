# 🟢 Safe to skim
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdFingerprintGenerator as rfg

smiles_by_id = {
    'CHEMBL1039': 'NC1(c2ccccc2Cl)CCCCC1=O',
    'CHEMBL1081808': '[11CH3]NC1(c2ccccc2)CCCCC1=O',
    'CHEMBL1714': 'CNC1(c2ccccc2Cl)CCCCC1=O.Cl',
    'CHEMBL2364609': 'CN[C@]1(c2ccccc2Cl)CCCCC1=O.Cl',
    'CHEMBL395091': 'CN[C@]1(c2ccccc2Cl)CCCCC1=O',
    'CHEMBL4650718': 'N[C@@]1(c2ccccc2Cl)CCC[C@@H](O)C1=O',
    'CHEMBL467126': 'N[C@]1(c2ccccc2Cl)CCCCC1=O',
    'CHEMBL467504': 'N[C@@]1(c2ccccc2Cl)CCCCC1=O',
    'CHEMBL467505': 'CN[C@@]1(c2ccccc2Cl)CCCCC1=O',
    'CHEMBL742': 'CNC1(c2ccccc2Cl)CCCCC1=O'
}

def std_parent(mol):
    params = rdMolStandardize.CleanupParameters()
    mol = rdMolStandardize.Cleanup(mol, params)
    # Use FragmentParent to be extra sure salts/solvents drop:
    mol = rdMolStandardize.FragmentParent(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    mol = rdMolStandardize.Normalize(mol)
    # Remove isotopes so [11C] doesn't split blocks:
    for a in mol.GetAtoms():
        a.SetIsotope(0)
    mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    return mol

def fp(m):
    gen = rfg.GetMorganGenerator(radius=2, fpSize=2048)
    return gen.GetFingerprint(m)

def ik_first(m):
    ik = Chem.MolToInchiKey(m)
    return ik.split('-')[0], ik

mols = {}
fps  = {}
for cid, smi in smiles_by_id.items():
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        continue
    mol = std_parent(mol)
    mols[cid] = mol
    fps[cid] = fp(mol)

ref = 'CHEMBL742'
ref_fp = fps[ref]
ref_blk, ref_full = ik_first(mols[ref])
print(f"REF {ref}: block={ref_blk}  full={ref_full}  std_smiles={Chem.MolToSmiles(mols[ref])}")

for cid in sorted(smiles_by_id):
    if cid == ref or cid not in mols: 
        continue
    blk, full = ik_first(mols[cid])
    sim = DataStructs.TanimotoSimilarity(ref_fp, fps[cid])
    print(f"{cid:>12}  block={blk}  FP={sim:.3f}  std_smiles={Chem.MolToSmiles(mols[cid])}")

