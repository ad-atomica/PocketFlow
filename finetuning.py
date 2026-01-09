import torch

from pocket_flow.gdbp_model import PocketFlow, reset_parameters
from pocket_flow.utils import Experiment, LoadDataset, load_model_from_ckpt
from pocket_flow.utils.data import ComplexDataTrajectory

# from utils.parse_file import Protein, parse_sdf_to_dict
from pocket_flow.utils.transform import (
    AtomComposer,
    Combine,
    FeaturizeLigandAtom,
    FeaturizeProteinAtom,
    FocalMaker,
    LigandCountNeighbors,
    LigandTrajectory,
    PermType,
    RefineData,
    TrajCompose,
)
from pocket_flow.utils.transform_utils import GraphType

protein_featurizer = FeaturizeProteinAtom()
ligand_featurizer = FeaturizeLigandAtom(atomic_numbers=[6, 7, 8, 9, 15, 16, 17, 35, 53])
traj_fn = LigandTrajectory(perm_type=PermType.MIX, num_atom_type=9)
focal_masker = FocalMaker(r=4, num_work=16, atomic_numbers=[6, 7, 8, 9, 15, 16, 17, 35, 53])
atom_composer = AtomComposer(
    knn=16, num_workers=16, graph_type=GraphType.KNN, radius=10, use_protein_bond=True
)
combine = Combine(traj_fn, focal_masker, atom_composer)
transform = TrajCompose(
    [
        RefineData(),
        LigandCountNeighbors(),
        protein_featurizer,
        ligand_featurizer,
        combine,
        ComplexDataTrajectory.from_steps,
    ]
)

dataset = LoadDataset("./data/crossdocked_pocket10.lmdb", transform=transform)
print("Num data:", len(dataset))
train_set, valid_set = LoadDataset.split(dataset, val_num=100, shuffle=True, random_seed=0)
dataset[0]
## reset parameters
device = "cuda:0"
model = load_model_from_ckpt(PocketFlow, "../path/to/pretrained/ckpt.pt", device)
print(model.get_parameter_number())
keys = [
    "edge_flow.flow_layers.5",
    "atom_flow.flow_layers.5",
    "pos_predictor.mu_net",
    "pos_predictor.logsigma_net",
    "pos_predictor.pi_net",
    "focal_net.net.1",
]
model = reset_parameters(model, keys)
# model = freeze_parameters(model,key)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=2.0e-4, weight_decay=0, betas=(0.99, 0.999)
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, patience=10, min_lr=1.0e-5)

exp = Experiment(
    model,
    train_set,
    optimizer,
    valid_set=valid_set,
    scheduler=scheduler,
    device=device,
    use_amp=False,
)
exp.fit_step(
    1000000,
    valid_per_step=5000,
    train_batch_size=2,
    valid_batch_size=16,
    print_log=True,
    with_tb=True,
    logdir="./finetuning_log",
    schedule_key="loss",
    num_workers=8,
    pin_memory=False,
    follow_batch=[],
    exclude_keys=[],
    collate_fn=None,
    max_edge_num_in_batch=400000,
)
