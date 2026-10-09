#!/usr/bin/env python3
import math
from pathlib import Path
import re

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, EsmForMaskedLM

from phospholm.mlm.data_collate import DataCollatorForPackedMasking
from phospholm.mlm.dataset import MLMDataset, ensure_fasta_index
from phospholm.tokenizer.phosphoLM_tokenizer import PTMTokenizer


ROOT = Path(__file__).resolve().parent
ARCHITECTURE = "facebook/esm2_t33_650M_UR50D"
TRAIN_FASTA = ROOT / "datasets/uniref50/uniref50.fasta"
TOKENIZER_PATH = ROOT / "output/phospholm_tokenizer"
OUTPUT_DIR = ROOT / "output/mlm_pretraining"

PER_DEVICE_BATCH_SIZE = 256
GRADIENT_ACCUMULATION = 1
LEARNING_RATE = 1e-4
MAX_STEPS = 1_150_000
SAVE_STEPS = 10
DATA_SEED = 42
RESUME = True
CHECKPOINT_PATH = None


def checkpoint_step(path):
    match = re.fullmatch(r"checkpoint-(\d+)", Path(path).name)
    return int(match.group(1)) if match else None


def latest_checkpoint(output_dir):
    candidates = [
        (step, path)
        for path in Path(output_dir).glob("checkpoint-*")
        if path.is_dir() and (step := checkpoint_step(path)) is not None
    ]
    return max(candidates, default=None)


def resume_checkpoint():
    if not RESUME:
        return None
    if CHECKPOINT_PATH:
        path = Path(CHECKPOINT_PATH).resolve()
        step = checkpoint_step(path)
        return step, path
    return latest_checkpoint(OUTPUT_DIR)


def save_checkpoint(accelerator, model, tokenizer, step):
    path = OUTPUT_DIR / f"checkpoint-{step}"
    accelerator.print(f"Saving {path}")
    accelerator.save_state(path)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(path)
        tokenizer.save_pretrained(path)
    accelerator.wait_for_everyone()


def create_accelerator():
    return Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        mixed_precision="bf16",
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True,
            data_seed=DATA_SEED,
        ),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    accelerator = create_accelerator()

    if accelerator.is_main_process:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ensure_fasta_index(TRAIN_FASTA)
    accelerator.wait_for_everyone()

    tokenizer = PTMTokenizer.from_pretrained(TOKENIZER_PATH)
    config = AutoConfig.from_pretrained(ARCHITECTURE)
    config.use_cache = False
    config.vocab_size = len(tokenizer)
    config.pad_token_id = tokenizer.pad_token_id
    config.mask_token_id = tokenizer.mask_token_id
    config.attn_implementation = "flash_attention_2"

    model = EsmForMaskedLM(config)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    dataset = MLMDataset(TRAIN_FASTA, tokenizer)
    collator = DataCollatorForPackedMasking(
        tokenizer=tokenizer,
        max_length=config.max_position_embeddings,
        residues=["S", "T", "Y"],
    )
    loader = DataLoader(
        dataset,
        batch_size=PER_DEVICE_BATCH_SIZE,
        collate_fn=collator,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=4,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    steps_per_epoch = len(loader)
    completed_steps = 0
    starting_epoch = 0
    batches_to_skip = 0
    checkpoint = resume_checkpoint()
    if checkpoint:
        completed_steps, path = checkpoint
        accelerator.print(f"Resuming from {path}")
        accelerator.load_state(path)
        completed_batches = completed_steps * GRADIENT_ACCUMULATION
        starting_epoch, batches_to_skip = divmod(completed_batches, steps_per_epoch)

    progress = tqdm(
        total=MAX_STEPS,
        initial=completed_steps,
        disable=not accelerator.is_local_main_process,
        desc="MLM training",
        unit="step",
    )
    epochs = math.ceil(MAX_STEPS / steps_per_epoch)
    model.train()
    for epoch in range(starting_epoch, epochs):
        epoch_loader = loader
        if epoch == starting_epoch and batches_to_skip:
            epoch_loader = accelerator.skip_first_batches(loader, batches_to_skip)

        if hasattr(epoch_loader, "set_epoch"):
            epoch_loader.set_epoch(epoch)
        for batch in epoch_loader:
            if completed_steps >= MAX_STEPS:
                break
            with accelerator.accumulate(model):
                loss = model(**batch).loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                completed_steps += 1
                progress.update()
                progress.set_postfix(loss=f"{float(loss.detach()):.6f}")
                if completed_steps % SAVE_STEPS == 0:
                    save_checkpoint(accelerator, model, tokenizer, completed_steps)

    progress.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = OUTPUT_DIR / "final_model"
        accelerator.unwrap_model(model).save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        accelerator.print(f"Saved {final_path}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
