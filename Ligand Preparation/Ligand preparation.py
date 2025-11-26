from rdkit.Chem import AllChem
import os
from meeko import MoleculePreparation
input_smi_file = ""
output_dir = ""
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(input_smi_file, 'r') as file:
    lines = file.readlines()
for idx, line in enumerate(lines):
    parts = line.strip().split()
    if len(parts) == 0:
        continue
    smiles = parts[0]
    name = parts[1] if len(parts) > 1 else f"lig_{idx}"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Cannot parse SMILES for {smiles}, skipping.")
        continue
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(fragments) > 1:
        print(f"Multiple fragments in {name}, selecting the largest.")
        mol = max(fragments, key=lambda m: m.GetNumAtoms())
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
    except Exception as e:
        print(f"3D embedding failed for {name}: {e}")
        continue
    meeko_prep = MoleculePreparation()
    try:
        meeko_prep.prepare(mol)
        pdbqt_str = meeko_prep.write_pdbqt_string()
        out_path = os.path.join(output_dir, name + ".pdbqt")
        with open(out_path, 'w') as f:
            f.write(pdbqt_str)
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Failed to prepare {name}: {e}")
        continue
