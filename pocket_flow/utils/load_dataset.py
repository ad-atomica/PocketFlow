from __future__ import annotations

import os
import pickle
import random
from collections.abc import Callable
from multiprocessing import Pool
from pathlib import Path
from typing import cast

import lmdb
import numpy as np
import torch
from rdkit import Chem
from torch import Tensor
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .data import ComplexData, torchify_dict
from .dataset_types import SizedSubset
from .parse_file import Ligand, Protein


class LoadDataset(Dataset[ComplexData]):
    def __init__(
        self,
        dataset_path: str | os.PathLike[str],
        transform: Callable[[ComplexData], ComplexData] | None = None,
        map_size: int = 10 * (1024 * 1024 * 1024),
    ) -> None:
        super().__init__()
        self.dataset_path: Path = Path(dataset_path)
        self.transform = transform
        self.map_size = map_size
        self.db: lmdb.Environment | None = None
        self.keys: list[bytes] | None = None

    def _connect_db(self) -> None:
        db = lmdb.open(
            str(self.dataset_path),
            map_size=self.map_size,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.db = db
        with db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self) -> None:
        if self.db is not None:
            self.db.close()
        self.db = None
        self.keys = None

    def __len__(self) -> int:
        if self.db is None:
            self._connect_db()
        assert self.keys is not None
        return len(self.keys)

    def __getitem__(self, idx: int) -> ComplexData:
        if self.db is None:
            self._connect_db()
        db = self.db
        keys = self.keys
        assert db is not None
        assert keys is not None
        key = keys[idx]
        raw_data = cast(bytes | None, db.begin().get(key))
        assert raw_data is not None
        data: ComplexData = pickle.loads(raw_data)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def _connect_db_edit(self) -> None:
        db = lmdb.open(
            str(self.dataset_path),
            map_size=self.map_size,  # 10GB
            create=False,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.db = db
        with db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def remove(self, idx: int) -> None:
        if self.db is None:
            self._connect_db_edit()
        db = self.db
        keys = self.keys
        assert db is not None
        assert keys is not None
        txn = db.begin(write=True)
        txn.delete(keys[idx])
        txn.commit()
        self._close_db()

    @staticmethod
    def split(
        dataset: LoadDataset,
        val_num: int | None = None,
        shuffle: bool = True,
        random_seed: int = 0,
    ) -> tuple[SizedSubset[ComplexData], SizedSubset[ComplexData]]:
        index = list(range(len(dataset)))
        if shuffle:
            random.seed(random_seed)
            random.shuffle(index)
        index_tensor = torch.LongTensor(index)
        split_dic: dict[str, Tensor] = {
            "valid": index_tensor[:val_num],
            "train": index_tensor[val_num:],
        }
        subsets = {k: SizedSubset(dataset, indices=v.tolist()) for k, v in split_dic.items()}
        train_set, val_set = subsets["train"], subsets["valid"]
        return train_set, val_set

    @staticmethod
    def split_by_name(
        dataset: LoadDataset,
        test_key_set: set[tuple[str, str]],
        name2id_path: str | os.PathLike[str] | None = None,
    ) -> tuple[SizedSubset[ComplexData], SizedSubset[ComplexData]]:
        if name2id_path is None:
            name2id_dir = dataset.dataset_path.parent
            dataset_name = dataset.dataset_path.stem
            name2id_file = name2id_dir / f"{dataset_name}_name2id.pt"
            if name2id_file.exists():
                name2id: dict[tuple[str, str], int] = torch.load(str(name2id_file))
            else:
                name2id = {}
                for i in tqdm(range(len(dataset)), "Indexing Dataset"):
                    try:
                        data = dataset[i]
                    except AssertionError as e:
                        print(i, e)
                        continue
                    name = (
                        "/".join(data.protein_filename.split("/")[-2:]),
                        "/".join(data.ligand_filename.split("/")[-2:]),
                    )
                    name2id[name] = i
                torch.save(name2id, str(name2id_file))
        else:
            name2id_file = Path(name2id_path)
            name2id = torch.load(str(name2id_file))

        train_idx: list[int] = []
        test_idx: list[int] = []
        for k, v in tqdm(name2id.items(), "Spliting"):
            if k in test_key_set:
                test_idx.append(v)
            else:
                train_idx.append(v)
        split_dict: dict[str, Tensor] = {
            "valid": torch.LongTensor(test_idx),
            "train": torch.LongTensor(train_idx),
        }
        subsets = {k: SizedSubset(dataset, indices=v.tolist()) for k, v in split_dict.items()}
        train_set, val_set = subsets["train"], subsets["valid"]
        return train_set, val_set


class CrossDocked2020:
    def __init__(
        self,
        raw_path: str,
        index_path: str,
        unexpected_sample: tuple[str, ...] = (),
        atomic_numbers: tuple[int, ...] = (6, 7, 8, 9, 15, 16, 17, 35, 53),
    ) -> None:
        self.raw_path = raw_path
        self.file_dirname = os.path.dirname(raw_path)
        self.index_path = index_path
        self.unexpected_sample = unexpected_sample
        self.index = self.get_file(index_path, raw_path)
        self.atomic_numbers = set(atomic_numbers)
        self.only_backbone = False

    @staticmethod
    def get_file(index_dirname: str, index_path: str) -> list[list[str]]:
        with open(index_dirname, "rb") as f:
            index: list[tuple[str | None, str]] = pickle.load(f)
        file_list: list[list[str]] = []
        for i in index:
            if i[0] is None:
                continue
            else:
                pdb = os.path.join(index_path, i[0])
            sdf = os.path.join(index_path, i[1])
            file_list.append([pdb, sdf])
        return file_list

    def process(self, raw_file_info: list[str]) -> ComplexData | None:
        try:
            pocket_file, ligand_file = raw_file_info
            lig = Ligand(ligand_file, removeHs=True, sanitize=True)
            for a in lig.mol.GetAtoms():
                if a.GetAtomicNum() not in self.atomic_numbers:
                    return None
            else:
                ligand_dict = lig.to_dict()
            if self.only_backbone:
                pocket_dict = Protein(pocket_file).get_backbone_dict()
            else:
                pocket_dict = Protein(pocket_file).get_atom_dict()
            data = ComplexData.from_protein_ligand_dicts(
                protein_dict=torchify_dict(pocket_dict),
                ligand_dict=torchify_dict(ligand_dict),
            )
            data.protein_filename = "/".join(pocket_file.split("/")[-2:])
            data.ligand_filename = "/".join(ligand_file.split("/")[-2:])
            return data
        except Exception:
            return None

    def run(
        self,
        dataset_name: str = "crossdocked_pocket10_processed.lmdb",
        lmdb_path: str | None = None,
        max_ligand_atom: int = 50,
        only_backbone: bool = False,
        n_process: int = 16,
        interval: int = 2000,
    ) -> None:
        self.only_backbone = only_backbone
        if lmdb_path:
            lmdb_path = os.path.join(lmdb_path, dataset_name.split("/")[-1])
        else:
            lmdb_path = dataset_name
        if os.path.exists(lmdb_path):
            raise FileExistsError(lmdb_path + " has been existed !")
        db = lmdb.open(
            lmdb_path,
            map_size=200 * (1024 * 1024 * 1024),  # 200GB
            create=True,
            subdir=False,
            readonly=False,
        )
        data_ix_list: list[bytes] = []
        exception_list: list[list[str]] = []
        data_ix = 0
        for idx in tqdm(range(0, len(self.index), interval)):
            if idx + interval >= len(self.index):
                raw_files = self.index[idx:]
            else:
                raw_files = self.index[idx : idx + interval]
            val_raw_files: list[list[str]] = []
            for items in raw_files:
                if "ATOM" not in open(items[0]).read():
                    continue
                elif items[1] in self.unexpected_sample:
                    continue
                elif Chem.MolFromMolFile(items[1]) is not None:
                    val_raw_files.append(items)
                else:
                    exception_list.append(items)
            torch.multiprocessing.set_sharing_strategy("file_system")
            pool = Pool(processes=n_process)
            data_list: list[ComplexData | None] = pool.map(self.process, val_raw_files)
            pool.close()
            pool.join()
            with db.begin(write=True, buffers=True) as txn:
                for data in data_list:
                    if data is None:
                        continue
                    if data.protein_pos.size(0) < 50:
                        continue
                    if len(data.ligand_nbh_list) > max_ligand_atom:
                        continue
                    key = str(data_ix).encode()
                    txn.put(key=key, value=pickle.dumps(data))
                    data_ix_list.append(key)
                    data_ix += 1
        db.close()
        data_ix_arr = np.array(data_ix_list)
        np.save(lmdb_path.split(".")[0] + "_Keys", data_ix_arr)
        with open(lmdb_path.split(".")[0] + "_invalid.list", "w") as fw:
            fw.write(str(exception_list))
