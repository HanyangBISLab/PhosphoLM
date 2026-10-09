import gc
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from phospholm.loss.focalloss import FocalLoss


PROTEINS_PER_UPDATE = 64
NUM_WORKERS = 16
SEED = 42
MAX_LR = 1e-4
MIN_LR = 1e-5
IGNORE_INDEX = -100


@dataclass(frozen=True)
class TokenTrainingConfig:
    name: str
    fold_dir: Path
    model_dir: Path
    folds: tuple[int, ...]
    checkpoint: str | Path
    dataset_class: type
    model_builder: object
    max_epochs: int = 100
    validation_fraction: float = 0.2
    early_stopping_patience: int = 3
    dataloader_num_workers: int = 16
    learning_rate: float = MAX_LR
    validation_steps: int | None = None
    token_mean_focal_loss: bool = False


def _split_dataset(dataset, fold, validation_fraction):

    validation_size = math.ceil(len(dataset) * validation_fraction)
    training_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(SEED + fold)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    return (
        Subset(dataset, indices[:training_size]),
        Subset(dataset, indices[training_size:]),
    )


@dataclass
class _EarlyStopping:
    patience: int
    best_loss: float = math.inf
    best_epoch: int | None = None
    bad_epochs: int = 0

    def update(self, validation_loss, epoch):
        improved = validation_loss < self.best_loss
        if improved:
            self.best_loss = float(validation_loss)
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self):
        return self.bad_epochs >= self.patience


def collate_one_protein(examples):
    example = examples[0]
    fields = ("input_ids", "attention_mask", "labels")
    return {field: torch.tensor([example[field]], dtype=torch.long) for field in fields}


def accumulation_steps(num_processes):
    return PROTEINS_PER_UPDATE // num_processes


def accumulation_group_size(step, dataset_size, steps_per_update, num_processes):

    group_start = step // steps_per_update * steps_per_update
    return min(PROTEINS_PER_UPDATE, dataset_size - group_start * num_processes)


def is_update_step(step, steps_on_rank, steps_per_update):
    return (step + 1) % steps_per_update == 0 or step + 1 == steps_on_rank


def _normalize_token_mean_gradients(
    accelerator, parameters, local_token_count, group_size
):
    total = accelerator.reduce(
        torch.tensor([local_token_count], device=accelerator.device, dtype=torch.int64),
        reduction="sum",
    )
    global_token_count = int(total[0])
    scale = group_size / global_token_count
    with torch.no_grad():
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
    return global_token_count


def _dataloader(dataset, fold, *, shuffle=True, num_workers=NUM_WORKERS):
    generator = torch.Generator().manual_seed(SEED + fold)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        collate_fn=collate_one_protein,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def _save_model(accelerator, model, tokenizer, path):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        path.parent.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(path, safe_serialization=True)
        tokenizer.save_pretrained(path)
        accelerator.print(f"Saved {path}")
    accelerator.wait_for_everyone()


def _validation_loss(
    accelerator,
    model,
    loader,
    loss_function,
    dataset_size,
    *,
    token_mean=False,
):

    was_training = model.training
    model.eval()
    local_loss = 0.0
    local_proteins = 0
    local_tokens = 0
    try:
        with torch.no_grad():
            for step, batch in enumerate(loader):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
                loss = loss_function(logits, batch["labels"])
                global_index = (
                    step * accelerator.num_processes + accelerator.process_index
                )
                if global_index < dataset_size:
                    token_count = int((batch["labels"] != IGNORE_INDEX).sum())
                    local_loss += float(loss) * (token_count if token_mean else 1)
                    local_proteins += 1
                    local_tokens += token_count
        values = [local_loss, local_proteins]
        if token_mean:
            values.append(local_tokens)
        totals = accelerator.reduce(
            torch.tensor(
                values,
                device=accelerator.device,
                dtype=torch.float64,
            ),
            reduction="sum",
        )
        validation_loss = float(totals[0] / totals[2 if token_mean else 1])
        return validation_loss
    finally:
        model.train(was_training)


def _train_fold(accelerator, config, fold):
    set_seed(SEED)
    fold_path = config.fold_dir / f"fold_{fold}_train.json"
    model, tokenizer = config.model_builder(config.checkpoint)
    model.set_attn_implementation("sdpa")

    dataset = config.dataset_class(fold_path, tokenizer)
    training_dataset, validation_dataset = _split_dataset(
        dataset, fold, config.validation_fraction
    )
    loader = _dataloader(
        training_dataset, fold, num_workers=config.dataloader_num_workers
    )
    validation_loader = _dataloader(
        validation_dataset,
        fold,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
    )

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    loss_function = FocalLoss(
        alpha=0.85,
        gamma=2.5,
        ignore_index=IGNORE_INDEX,
        class_balanced=not config.token_mean_focal_loss,
    )
    model, optimizer, loader, validation_loader = accelerator.prepare(
        model, optimizer, loader, validation_loader
    )

    steps_per_update = accumulation_steps(accelerator.num_processes)
    steps_on_rank = len(loader)
    updates_per_epoch = math.ceil(steps_on_rank / steps_per_update)

    total_updates = config.max_epochs * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=MIN_LR
    )
    optimizer.zero_grad(set_to_none=True)
    completed_updates = 0
    stopping = _EarlyStopping(config.early_stopping_patience)
    output = config.model_dir / f"fold_{fold}"
    best_step = None
    last_validation_step = None
    patience_unit = "epochs" if config.validation_steps is None else "validation_checks"
    loss_averaging = "supervised_tokens" if config.token_mean_focal_loss else "proteins"

    validation_schedule = (
        "at epoch end"
        if config.validation_steps is None
        else f"every {config.validation_steps} optimizer updates (plus the final update)"
    )
    accelerator.print(
        f"\n{config.name}, fold {fold}: {len(training_dataset):,} training / "
        f"{len(validation_dataset):,} validation proteins, "
        f"{accelerator.num_processes} process(es), at most {total_updates:,} updates; "
        f"{config.dataloader_num_workers} DataLoader workers per GPU per loader; "
        f"initial LR={config.learning_rate:g}; validation {validation_schedule}; "
        f"patience={config.early_stopping_patience} {patience_unit}; "
        f"focal alpha={'scalar' if config.token_mean_focal_loss else 'class-balanced'}, "
        f"loss averaged over {loss_averaging}"
    )
    progress = tqdm(
        total=config.max_epochs * steps_on_rank,
        desc=f"{config.name} fold {fold}",
        unit="protein",
        disable=not accelerator.is_main_process,
        dynamic_ncols=True,
    )

    def validate_and_checkpoint(epoch):
        nonlocal model, tokenizer, optimizer, best_step, last_validation_step
        nonlocal validation_loader, validation_dataset, loss_function

        validation_loss = _validation_loss(
            accelerator,
            model,
            validation_loader,
            loss_function,
            len(validation_dataset),
            **({"token_mean": True} if config.token_mean_focal_loss else {}),
        )
        improved = stopping.update(validation_loss, epoch)
        if improved:
            _save_model(accelerator, model, tokenizer, output / "final_model")
            best_step = completed_updates
        last_validation_step = completed_updates
        accelerator.print(
            f"Fold {fold}, epoch {epoch}, optimizer step {completed_updates}: "
            f"validation_loss={validation_loss:.6f}, improved={improved}, "
            f"best_step={best_step}, "
            f"patience={stopping.bad_epochs}/{stopping.patience} validation checks"
        )
        return validation_loss

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        local_loss = 0.0
        local_proteins = 0
        local_tokens = 0
        group_tokens = 0
        validation_loss = None
        for step, batch in enumerate(loader):
            token_count = int((batch["labels"] != IGNORE_INDEX).sum())

            update_now = is_update_step(step, steps_on_rank, steps_per_update)
            group_size = accumulation_group_size(
                step, len(training_dataset), steps_per_update, accelerator.num_processes
            )
            global_index = step * accelerator.num_processes + accelerator.process_index
            is_distributed_padding = global_index >= len(training_dataset)
            loss_scale = (
                0.0
                if is_distributed_padding
                else accelerator.num_processes / group_size
            )
            if config.token_mean_focal_loss:
                loss_scale *= token_count

            sync = nullcontext() if update_now else accelerator.no_sync(model)
            with sync:
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
                loss = loss_function(logits, batch["labels"])
                accelerator.backward(loss * loss_scale)

            if not is_distributed_padding:
                local_loss += float(loss.detach()) * (
                    token_count if config.token_mean_focal_loss else 1
                )
                local_proteins += 1
                local_tokens += token_count
                group_tokens += token_count
            if update_now:
                if config.token_mean_focal_loss:
                    _normalize_token_mean_gradients(
                        accelerator, parameters, group_tokens, group_size
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                completed_updates += 1
                group_tokens = 0

            progress.update()
            progress.set_postfix(loss=f"{float(loss.detach()):.6f}")
            if (
                update_now
                and config.validation_steps is not None
                and completed_updates % config.validation_steps == 0
            ):
                validation_loss = validate_and_checkpoint(epoch)
                if stopping.should_stop:
                    break

        values = [local_loss, local_proteins]
        if config.token_mean_focal_loss:
            values.append(local_tokens)
        totals = accelerator.reduce(
            torch.tensor(
                values,
                device=accelerator.device,
                dtype=torch.float64,
            ),
            reduction="sum",
        )
        training_loss = float(
            totals[0] / totals[2 if config.token_mean_focal_loss else 1]
        )
        if config.validation_steps is None or (
            epoch == config.max_epochs
            and not stopping.should_stop
            and last_validation_step != completed_updates
        ):
            validation_loss = validate_and_checkpoint(epoch)
        validation_display = (
            "not evaluated this epoch"
            if validation_loss is None
            else f"{validation_loss:.6f}"
        )
        accelerator.print(
            f"Fold {fold}, epoch {epoch}/{config.max_epochs}: "
            f"training_loss={training_loss:.6f}, last_validation_loss={validation_display}, "
            f"best_epoch={stopping.best_epoch}, "
            f"patience={stopping.bad_epochs}/{stopping.patience}"
        )
        if stopping.should_stop:
            accelerator.print(
                f"Early stopping fold {fold} at optimizer step {completed_updates} "
                f"in epoch {epoch}."
            )
            break

    progress.close()

    accelerator.print(
        f"Fold {fold}: retained epoch {stopping.best_epoch}, "
        f"optimizer step {best_step} with "
        f"validation_loss={stopping.best_loss:.6f} at {output / 'final_model'}"
    )
    del (
        validate_and_checkpoint,
        model,
        tokenizer,
        dataset,
        training_dataset,
        validation_dataset,
        loader,
        validation_loader,
        optimizer,
        scheduler,
        loss_function,
    )
    gc.collect()
    torch.cuda.empty_cache()
    accelerator.free_memory()


def train_token_folds(config):
    accelerator = Accelerator(mixed_precision="bf16")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    for fold in config.folds:
        _train_fold(accelerator, config, fold)
    accelerator.end_training()
