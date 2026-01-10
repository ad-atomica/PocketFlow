"""Create a pocket PDB file from a protein structure and reference ligand.

This script extracts protein residues within a distance cutoff (default 10 Å) of a
reference ligand to define the binding pocket. The output includes surface atom
annotations for each pocket atom.

Example:
    Create a pocket PDB for CDK5c using a reference ligand::

        pixi run python create_pocket_pdb.py --protein data/proteins/cdk5c.pdb --ligand data/reference_ligands/cdk5c-reference.sdf

    This will create ``data/proteins/pocket.pdb`` containing only the pocket residues
    with surface annotations (atoms marked as "surf" or "inner").

pixi run python create_pocket_pdb.py --protein data/proteins/cdk10.pdb --ligand data/reference_ligands/cdk10-reference.sdf
"""

import argparse
from pathlib import Path

from pocket_flow.utils.parse_file import Ligand, Protein
from pocket_flow.utils.process_raw import SplitPocket

DIST_CUTOFF = 10


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein", type=str, required=True, help="PDB")
    parser.add_argument("--ligand", type=str, required=True, help="SDF")
    args = parser.parse_args()

    return args


def main():
    args = arguments()

    output_path = f"{Path(args.protein).parent}/pocket.pdb"

    pro = Protein(args.protein)
    lig = Ligand(args.ligand)

    pocket_block, _ligand_mol_block = SplitPocket._split_pocket_with_surface_atoms(pro, lig, DIST_CUTOFF)
    open(output_path, "w").write(pocket_block)


if __name__ == "__main__":
    main()
