from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar

from torch.utils.data import Dataset, Subset

T_co = TypeVar("T_co", covariant=True)


class SizedDataset(Dataset[T_co]):
    @abstractmethod
    def __len__(self) -> int: ...


class SizedSubset(Subset[T_co], SizedDataset[T_co]):
    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, idx: int) -> T_co:
        return super().__getitem__(idx)
