from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol

import torch
from torch import Tensor, nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from pocket_flow.utils.data import make_batch_collate
from pocket_flow.utils.dataset_types import SizedDataset
from pocket_flow.utils.file_utils import ensure_parent_dir_exists
from pocket_flow.utils.time_utils import timewait

if TYPE_CHECKING:
    from pocket_flow.gdbp_model.pocket_flow import TrainingBatch
    from pocket_flow.utils.data import ComplexData


def inf_iterator(iterable: DataLoader) -> Iterator[TrainingBatch]:
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def get_parameter_number(model: PocketFlowModel) -> dict[str, int]:
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num}


class PocketFlowModel(Protocol):
    config: Any

    def get_loss(self, data: TrainingBatch) -> dict[str, Tensor]: ...
    def train(self, mode: bool = True) -> nn.Module: ...
    def eval(self) -> nn.Module: ...
    def parameters(self) -> Iterator[nn.Parameter]: ...
    def state_dict(self) -> dict[str, Any]: ...


class Experiment:
    model: PocketFlowModel
    model_config: Any | None
    train_set: SizedDataset[ComplexData]
    valid_set: SizedDataset[ComplexData] | None
    optimizer: Optimizer
    scheduler: ReduceLROnPlateau | None
    num_train_data: int
    num_valid_data: int | None
    clip_grad: bool
    max_norm: float
    norm_type: float
    with_tb: bool
    device: str | torch.device
    device_type: Literal["cuda", "cpu", "mps"]
    pos_noise_std: float
    use_amp: bool
    grad_scaler: GradScaler
    train_loader: Iterator[TrainingBatch]
    valid_loader: Iterator[TrainingBatch]
    train_batch_size: int
    valid_batch_size: int
    n_iter_train: int
    n_iter_valid: int
    logdir: str
    writer: tensorboard.SummaryWriter

    def __init__(
        self,
        model: PocketFlowModel,
        train_set: SizedDataset[ComplexData],
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau | None = None,
        device: str | torch.device = "cuda",
        valid_set: SizedDataset[ComplexData] | None = None,
        clip_grad: bool = True,
        max_norm: float = 5.0,
        norm_type: float = 2.0,
        pos_noise_std: float = 0.1,
        use_amp: bool = False,
    ) -> None:
        self.model = model
        self.model_config = getattr(model, "config", None)
        self.train_set = train_set
        self.valid_set = valid_set
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_train_data = len(train_set)
        self.num_valid_data = len(valid_set) if valid_set else None

        self.clip_grad = clip_grad
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.with_tb = False
        self.device = device
        self.device_type = torch.device(device).type
        self.pos_noise_std = pos_noise_std
        self.use_amp = use_amp
        if self.use_amp:
            self.grad_scaler = GradScaler(device=self.device_type)

    @staticmethod
    def get_log(
        out_dict: dict[str, Tensor],
        key_word: str,
        it: int,
        time_gap: str | None = None,
    ) -> str:
        log: list[str] = []
        for key, value in out_dict.items():
            log.append(f" {key}:{value.item():.5f} |")
        log.insert(0, f"[{key_word} {it}]")
        if time_gap:
            log.append(f" Time: {time_gap}")
        return "".join(log)

    @staticmethod
    def write_summary(
        out_dict: dict[str, Tensor],
        writer: tensorboard.SummaryWriter,
        key_word: str,
        num_iter: int,
        scheduler: ReduceLROnPlateau | None = None,
        optimizer: Optimizer | None = None,
    ) -> None:
        for key, value in out_dict.items():
            writer.add_scalar(f"{key_word}/{key}", value, num_iter)
        if scheduler is not None:
            assert optimizer is not None
            writer.add_scalar(f"{key_word}/lr", optimizer.param_groups[0]["lr"], num_iter)
        writer.flush()

    @staticmethod
    def get_num_iter(num_data: int, batch_size: int) -> int:
        if num_data % batch_size == 0:
            n_iter = int(num_data / batch_size)
        else:
            n_iter = int(num_data / batch_size) + 1
        return n_iter

    @property
    def parameter_number(self) -> dict[str, int]:
        return get_parameter_number(self.model)

    def _train_step(self, batch: TrainingBatch, it: int = 0, print_log: bool = False) -> dict[str, Tensor]:
        start = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        if self.use_amp:
            with autocast(device_type=self.device_type):
                out_dict = self.model.get_loss(batch)
            self.grad_scaler.scale(out_dict["loss"]).backward()
        else:
            out_dict = self.model.get_loss(batch)
            out_dict["loss"].backward()

        if self.clip_grad:
            orig_grad_norm = clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.max_norm,
                norm_type=self.norm_type,
                error_if_nonfinite=True,
            )
        if self.use_amp:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

        if self.with_tb:
            if self.clip_grad:
                self.writer.add_scalar("train/step/grad", orig_grad_norm, it)
            self.write_summary(out_dict, self.writer, "train/step", it)

        end = time.time()
        time_gap = end - start
        log = self.get_log(out_dict, "Step", it, time_gap=f"{time_gap:.3f}")
        with open(self.logdir + "training.log", "a") as log_writer:
            log_writer.write(log + "\n")
        if print_log:
            print(log)
        return out_dict

    def validate(
        self,
        n_iter: int,
        n_epoch: int,
        print_log: bool = False,
        schedule_key: str = "loss",
    ) -> Tensor:
        start = time.time()
        log_dict: dict[str, Tensor] = {}
        with torch.no_grad():
            self.model.eval()
            for _ in range(n_iter):
                batch: TrainingBatch = next(self.valid_loader).to(self.device)
                out_dict = self.model.get_loss(batch)
                for key, value in out_dict.items():
                    if key not in log_dict:
                        log_dict[key] = value if "acc" in key else value * batch.num_graphs
                    else:
                        log_dict[key] += value if "acc" in key else value * batch.num_graphs

        assert self.num_valid_data is not None
        for key, value in log_dict.items():
            if "acc" in key:
                log_dict[key] = value / n_iter
            else:
                log_dict[key] = value / self.num_valid_data
        if self.scheduler:
            # ReduceLROnPlateau.step() takes metrics, not epoch
            self.scheduler.step(log_dict[schedule_key].item())
        if self.with_tb:
            self.write_summary(
                log_dict,
                self.writer,
                "val/epoch",
                n_epoch,
                scheduler=self.scheduler,
                optimizer=self.optimizer,
            )
        end = time.time()
        time_gap = timewait(end - start)
        log = self.get_log(log_dict, "Validate", n_epoch, time_gap=time_gap)
        with open(self.logdir + "training.log", "a") as log_writer:
            log_writer.write(log + "\n")
        if print_log:
            print(log)
        return log_dict["loss"]

    def fit_step(
        self,
        num_step: int,
        valid_per_step: int = 5000,
        train_batch_size: int = 4,
        valid_batch_size: int = 16,
        print_log: bool = True,
        with_tb: bool = True,
        logdir: str = "./training_log",
        schedule_key: str = "loss",
        *,
        num_workers: int,
        pin_memory: bool = False,
        follow_batch: Sequence[str] = (),
        exclude_keys: Sequence[str] = (),
        collate_fn: Callable[[Sequence[ComplexData]], Batch] | None = None,
        max_edge_num_in_batch: int = 900000,
    ) -> None:
        if collate_fn is None:
            collate_fn = make_batch_collate(
                follow_batch=follow_batch,
                exclude_keys=exclude_keys,
            )
        self.train_loader = inf_iterator(
            DataLoader(
                self.train_set,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
            )
        )
        self.train_batch_size = train_batch_size

        if self.valid_set:
            self.valid_loader = inf_iterator(
                DataLoader(
                    self.valid_set,
                    batch_size=valid_batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    collate_fn=collate_fn,
                )
            )
            self.valid_batch_size = valid_batch_size

        self.n_iter_train = self.get_num_iter(self.num_train_data, self.train_batch_size)
        if self.num_valid_data:
            self.n_iter_valid = self.get_num_iter(self.num_valid_data, self.valid_batch_size)
        date = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.logdir = logdir + "/" + date + "/"
        ensure_parent_dir_exists(self.logdir + "model_config.dir")
        open(self.logdir + "model_config.dir", "w").write(str(self.model_config))
        self.with_tb = with_tb
        if self.with_tb:
            self.writer = tensorboard.SummaryWriter(self.logdir)
        log_writer = open(self.logdir + "training.log", "w")
        log_writer.write(f"\n######## {self.parameter_number}; batch_size {train_batch_size} ########\n")
        log_writer.close()
        if print_log:
            print(f"\n######## {self.parameter_number} ########\n")
        for step in range(1, num_step + 1):
            batch: TrainingBatch = next(self.train_loader).to(self.device)
            cpx_noise = torch.randn_like(batch.cpx_pos) * self.pos_noise_std
            batch.cpx_pos = batch.cpx_pos + cpx_noise
            if batch.cpx_edge_index.size(1) > max_edge_num_in_batch:
                continue
            _out_dict = self._train_step(batch, it=step, print_log=print_log)
            if step % valid_per_step == 0 or step == num_step:
                if self.num_valid_data:
                    _val_loss = self.validate(
                        self.n_iter_valid, step, schedule_key=schedule_key, print_log=print_log
                    )
                ckpt_path = self.logdir + "/ckpt/"
                ensure_parent_dir_exists(ckpt_path + f"{step}.pt")
                torch.save(
                    {
                        "config": self.model_config,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                        "iteration": step,
                    },
                    ckpt_path + f"{step}.pt",
                )
