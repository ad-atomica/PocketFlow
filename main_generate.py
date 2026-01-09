from __future__ import annotations

import argparse
import ast
import os
import time
from typing import TYPE_CHECKING

import torch

from pocket_flow import Generate, PocketFlow
from pocket_flow.utils.data import ComplexData, torchify_dict
from pocket_flow.utils.model_io import load_model_from_ckpt
from pocket_flow.utils.parse_file import Ligand, Protein
from pocket_flow.utils.time_utils import timewait
from pocket_flow.utils.transform import (
    AtomComposer,
    FeaturizeLigandAtom,
    FeaturizeProteinAtom,
    LigandCountNeighbors,
    RefineData,
)
from pocket_flow.utils.transform_utils import mask_node

if TYPE_CHECKING:
    from pocket_flow.utils.parse_file import ProteinAtomDict


def str2bool(v: str) -> bool:
    v_lower = v.lower()
    if v_lower in {"yes", "true", "t", "y", "1"}:
        return True
    if v_lower in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parameter() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pkt", "--pocket", type=str, default="None", help="the pdb file of pocket in receptor"
    )
    parser.add_argument(
        "--ckpt", type=str, default="./ckpt/ZINC-pretrained-255000.pt", help="the path of saved model"
    )
    parser.add_argument("-n", "--num_gen", type=int, default=100, help="the number of generateive molecule")
    parser.add_argument("--name", type=str, default="receptor", help="receptor name")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="cuda:x or cpu")
    parser.add_argument(
        "-at", "--atom_temperature", type=float, default=1.0, help="temperature for atom sampling"
    )
    parser.add_argument(
        "-bt", "--bond_temperature", type=float, default=1.0, help="temperature for bond sampling"
    )
    parser.add_argument("--max_atom_num", type=int, default=40, help="the max atom number for generation")
    parser.add_argument(
        "-ft", "--focus_threshold", type=float, default=0.5, help="the threshold of probility for focus atom"
    )
    parser.add_argument(
        "-cm",
        "--choose_max",
        type=str,
        default="1",
        help="whether choose the atom that has the highest prob as focus atom",
    )
    parser.add_argument(
        "--min_dist_inter_mol",
        type=float,
        default=3.0,
        help="inter-molecular dist cutoff between protein and ligand.",
    )
    parser.add_argument(
        "--bond_length_range",
        type=str,
        default="(1.0, 2.0)",
        help="the range of bond length for mol generation.",
    )
    parser.add_argument("-mdb", "--max_double_in_6ring", type=int, default=0, help="")
    parser.add_argument(
        "--with_print", type=str, default="1", help="whether print SMILES in generative process"
    )
    parser.add_argument(
        "--root_path", type=str, default="gen_results", help="the root path for saving results"
    )
    parser.add_argument(
        "--readme", "-rm", type=str, default="None", help="description of this genrative task"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=True,
        help="worker count for neighbor search",
    )
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parameter()
    if args.name == "receptor":
        args.name = args.pocket.split("/")[-1].split("-")[0]

    assert args.pocket != "None", "Please specify pocket !"
    assert args.ckpt != "None", "Please specify model !"

    device: torch.device = torch.device(args.device)
    print(f"Using device: {device.type}")
    if device.type == "cuda":
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    pdb_file: str = args.pocket
    args.choose_max = str2bool(args.choose_max)
    args.with_print = str2bool(args.with_print)

    pro_dict: ProteinAtomDict = Protein(pdb_file).get_atom_dict(get_surf=True)
    lig_dict = Ligand.empty_dict()
    data: ComplexData = ComplexData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pro_dict),
        ligand_dict=torchify_dict(lig_dict),
    )

    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom(atomic_numbers=[6, 7, 8, 9, 15, 16, 17, 35, 53])
    atom_composer = AtomComposer(
        knn=16,
        num_workers=args.num_workers,
        for_gen=True,
        use_protein_bond=True,
    )

    data = RefineData()(data)
    data = LigandCountNeighbors()(data)
    data = protein_featurizer(data)
    data = ligand_featurizer(data)
    node4mask: torch.Tensor = torch.arange(data.ligand_pos.size(0))
    data = mask_node(data, torch.empty([0], dtype=torch.long), node4mask, num_atom_type=9, y_pos_std=0.0)
    data = atom_composer.run(data)

    print("Loading model ...")
    model: PocketFlow = load_model_from_ckpt(PocketFlow, args.ckpt, device)
    print("Generating molecules ...")

    temperature: tuple[float, float] = (args.atom_temperature, args.bond_temperature)
    bond_length_range: tuple[float, float]
    if isinstance(args.bond_length_range, str):
        try:
            parsed = ast.literal_eval(args.bond_length_range)
        except (ValueError, SyntaxError) as e:
            raise ValueError(
                f"Failed to parse bond_length_range '{args.bond_length_range}': {e}. "
                "Expected a tuple of two floats, e.g., '(1.0, 2.0)'"
            ) from e

        if not isinstance(parsed, tuple):
            raise TypeError(f"bond_length_range must be a tuple, got {type(parsed).__name__}: {parsed}")
        if len(parsed) != 2:
            raise ValueError(
                f"bond_length_range must contain exactly 2 elements, got {len(parsed)}: {parsed}"
            )
        try:
            bond_length_range = (float(parsed[0]), float(parsed[1]))
        except (ValueError, TypeError) as e:
            raise TypeError(f"bond_length_range elements must be numeric, got {parsed}: {e}") from e
    else:
        bond_length_range = args.bond_length_range

    generate = Generate(
        model,
        atom_composer.run,  # type: ignore[arg-type]
        temperature=temperature,
        atom_type_map=[6, 7, 8, 9, 15, 16, 17, 35, 53],
        num_bond_type=4,
        max_atom_num=args.max_atom_num,
        focus_threshold=args.focus_threshold,
        max_double_in_6ring=args.max_double_in_6ring,
        min_dist_inter_mol=args.min_dist_inter_mol,
        bond_length_range=bond_length_range,
        choose_max=args.choose_max,
        device=device,
        num_workers=args.num_workers,
    )

    start: float = time.time()
    generate.generate(
        data, num_gen=args.num_gen, rec_name=args.name, with_print=args.with_print, root_path=args.root_path
    )
    os.system(f"cp {args.ckpt} {generate.out_dir}")

    gen_config: str = "\n".join([f"{k}: {v}" for k, v in args.__dict__.items()])
    with open(generate.out_dir + "/readme.txt", "w") as fw:
        fw.write(gen_config)

    end: float = time.time()
    print(f"Time: {timewait(end - start)}")


if __name__ == "__main__":
    main()
