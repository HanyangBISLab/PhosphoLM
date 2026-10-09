#!/usr/bin/env python3
"""Select phosphosite-proximal tokens for PhosphoLM hidden states 0–32."""

import argparse
from bisect import bisect_left
from collections import Counter
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformers import EsmModel

from phospholm.tokenizer.phosphoLM_tokenizer import PTMTokenizer


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "datasets/dbptm/dbptm.json"
CHECKPOINT = ROOT / "models/pretrained_phosphoLM"
OUTPUT = ROOT / "analysis/phosphosite_proximal_token_analysis/selected_tokens.csv"
HIDDEN_STATES = range(33)
RESIDUES = ("S", "T", "Y")
SEED = 42
RADIUS = 7
TOP_N = 5
MIN_COUNT = 20
FDR_CUTOFF = 0.05


@dataclass
class Protein:
    name: str
    sequence: str
    queries: list[tuple[int, int]]


def sample_queries(sequence, annotated, rng, counts):
    ordered = sorted(annotated)
    pools = {residue: [] for residue in RESIDUES}
    for site, residue in enumerate(sequence):
        if residue in pools and site not in annotated:
            pools[residue].append(site)
    preferred = {residue: [] for residue in RESIDUES}
    for residue, pool in pools.items():
        for site in pool:
            index = bisect_left(ordered, site)
            if (index == 0 or site - ordered[index - 1] > RADIUS) and (
                index == len(ordered) or ordered[index] - site > RADIUS
            ):
                preferred[residue].append(site)

    queries, used = [], set()
    for site in ordered:
        queries.append((site, 0))
        residue = sequence[site]
        pool = pools[residue]
        if not pool:
            counts["missing_controls"] += 1
            continue
        unused_preferred = [s for s in preferred[residue] if s not in used]
        unused_any = [s for s in pool if s not in used]
        if unused_preferred:
            background = rng.choice(unused_preferred)
        elif unused_any:
            background = rng.choice(unused_any)
            counts["radius_fallbacks"] += 1
        elif preferred[residue]:
            background = rng.choice(preferred[residue])
            counts["reused_controls"] += 1
        else:
            background = rng.choice(pool)
            counts["radius_fallbacks"] += 1
            counts["reused_controls"] += 1
        used.add(background)
        queries.append((background, 1))
    return queries


def load_proteins(path):
    with Path(path).open(encoding="utf-8") as handle:
        raw_records = json.load(handle)
    grouped, counts = {}, Counter()
    for raw in raw_records:
        name = str(raw["Protein"]).strip()
        sequence = str(raw["Seq"]).strip().upper()
        sites = set(raw.get("Site") or [])
        if sites:
            grouped.setdefault((name, sequence), set()).update(sites)
    rng = random.Random(SEED)
    proteins = [
        Protein(name, sequence, sample_queries(sequence, sites, rng, counts))
        for (name, sequence), sites in grouped.items()
    ]
    print(f"Loaded {len(proteins):,} proteins; control counts: {dict(counts)}")
    return proteins


def encode_protein(protein, tokenizer):
    encoded = tokenizer(
        protein.sequence,
        add_special_tokens=True,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    token_ids = encoded["input_ids"][0].tolist()
    offsets = encoded.pop("offset_mapping")[0].tolist()
    candidates, candidate_ids, atomic = [], [], {}
    for index, (token_id, (start, end)) in enumerate(zip(token_ids, offsets)):
        if token_id in tokenizer.structural_ids:
            continue
        token = tokenizer.convert_ids_to_tokens(token_id)
        if end - start == 1 and token in RESIDUES:
            atomic.setdefault(start, []).append(len(candidates))
        candidates.append(index)
        candidate_ids.append(token_id)
    queries = []
    for site, _ in protein.queries:
        matches = atomic.get(site, [])
        queries.append(matches[0])
    inputs = {key: encoded[key] for key in ("input_ids", "attention_mask")}
    return inputs, candidates, candidate_ids, queries


def load_model(checkpoint, device):
    tokenizer = PTMTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = EsmModel.from_pretrained(
        checkpoint,
        add_pooling_layer=False,
        attn_implementation="sdpa",
        local_files_only=True,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    return model.requires_grad_(False).eval().to(device), tokenizer


class AffinityAccumulator:

    def __init__(self, vocab_size):
        shape = (3, 2, vocab_size, 2 * RADIUS + 1)
        self.counts = np.zeros(shape, dtype=np.int64)
        self.sums = np.zeros(shape, dtype=np.float64)
        self.squared_sums = np.zeros(shape, dtype=np.float64)

    def add(self, hidden, candidates, candidate_ids, queries, residues, groups):
        vectors = F.normalize(hidden[candidates].float(), dim=-1)
        queries, candidate_ids, residues, groups = (
            np.asarray(values, dtype=np.int64)
            for values in (queries, candidate_ids, residues, groups)
        )
        for offset in range(-RADIUS, RADIUS + 1):
            if offset == 0:
                continue
            neighbors = queries + offset
            valid = (neighbors >= 0) & (neighbors < len(candidate_ids))
            query_indices = torch.as_tensor(queries[valid], device=hidden.device)
            neighbor_indices = torch.as_tensor(neighbors[valid], device=hidden.device)
            similarities = (
                (vectors[query_indices] * vectors[neighbor_indices])
                .sum(dim=-1)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            indices = (
                residues[valid],
                groups[valid],
                candidate_ids[neighbors[valid]],
                offset + RADIUS,
            )
            np.add.at(self.counts, indices, 1)
            np.add.at(self.sums, indices, similarities)
            np.add.at(self.squared_sums, indices, similarities**2)


def benjamini_hochberg(pvalues):
    pvalues = np.where(np.isfinite(pvalues), pvalues, 1.0)
    order = np.argsort(pvalues)
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    return adjusted


def compare_affinities(accumulator, token_ids, tokenizer):
    grid = pd.MultiIndex.from_product(
        [RESIDUES, token_ids, range(-RADIUS, RADIUS + 1)],
        names=["Site_AA", "token_id", "relative_position"],
    )
    summary = grid.to_frame(index=False)
    summary["token"] = tokenizer.convert_ids_to_tokens(summary["token_id"].tolist())
    counts, means, variances = [], [], []
    for group, prefix in enumerate(("foreground", "background")):
        n, total, squared = (
            np.take(array[:, group], token_ids, axis=1).ravel().astype(np.float64)
            for array in (
                accumulator.counts,
                accumulator.sums,
                accumulator.squared_sums,
            )
        )
        mean = np.full(len(n), np.nan)
        np.divide(total, n, out=mean, where=n > 0)
        variance = np.full(len(n), np.nan)
        valid = n > 1
        variance[valid] = np.maximum(
            (squared[valid] - total[valid] ** 2 / n[valid]) / (n[valid] - 1), 0
        )
        counts.append(n)
        means.append(mean)
        variances.append(variance)
        summary[f"{prefix}_count"] = n.astype(np.int64)

    delta = means[0] - means[1]
    pvalues = np.ones(len(summary))
    valid = (counts[0] > 1) & (counts[1] > 1)
    indices = np.flatnonzero(valid)
    a = variances[0][valid] / counts[0][valid]
    b = variances[1][valid] / counts[1][valid]
    se_squared = a + b
    nonzero = se_squared > 0
    df = se_squared[nonzero] ** 2 / (
        a[nonzero] ** 2 / (counts[0][valid][nonzero] - 1)
        + b[nonzero] ** 2 / (counts[1][valid][nonzero] - 1)
    )
    t = delta[indices[nonzero]] / np.sqrt(se_squared[nonzero])
    pvalues[indices[nonzero]] = 2 * student_t.sf(np.abs(t), df)
    pvalues[indices[~nonzero]] = np.isclose(delta[indices[~nonzero]], 0).astype(float)
    summary["affinity_delta"] = delta
    summary["fdr"] = benjamini_hochberg(pvalues)
    return summary


def select_tokens(summary):
    eligible = summary[
        np.isfinite(summary["affinity_delta"])
        & summary["affinity_delta"].gt(0)
        & summary["fdr"].lt(FDR_CUTOFF)
        & summary["foreground_count"].ge(MIN_COUNT)
        & summary["background_count"].ge(MIN_COUNT)
    ]
    ranked = eligible.groupby(["Site_AA", "token_id", "token"], as_index=False).agg(
        max_affinity_delta=("affinity_delta", "max"),
        min_fdr=("fdr", "min"),
        max_foreground_count=("foreground_count", "max"),
        max_background_count=("background_count", "max"),
    )
    ranked = ranked.sort_values(
        [
            "Site_AA",
            "max_affinity_delta",
            "min_fdr",
            "max_foreground_count",
            "max_background_count",
            "token",
            "token_id",
        ],
        ascending=[True, False, True, False, False, True, True],
    )
    return [
        ", ".join(ranked.loc[ranked["Site_AA"].eq(residue), "token"].head(TOP_N))
        for residue in RESIDUES
    ]


def save_table(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["Hidden state", "S tokens", "T tokens", "Y tokens"])
        writer = csv.writer(handle, lineterminator="\n", quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(rows)


def run_analysis(device):
    proteins = load_proteins(DATASET)
    torch.manual_seed(SEED)
    model, tokenizer = load_model(CHECKPOINT, device)
    accumulators = [AffinityAccumulator(len(tokenizer)) for _ in HIDDEN_STATES]
    observed_tokens = set()
    with torch.inference_mode():
        for protein in tqdm(
            proteins, desc="Phosphosite-proximal affinity", unit="protein"
        ):
            inputs, candidates, candidate_ids, queries = encode_protein(
                protein, tokenizer
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(
                **inputs, output_hidden_states=True, output_attentions=False
            )
            residues = [
                RESIDUES.index(protein.sequence[site]) for site, _ in protein.queries
            ]
            groups = [group for _, group in protein.queries]
            for state, accumulator in zip(HIDDEN_STATES, accumulators):
                hidden = outputs.hidden_states[state]
                accumulator.add(
                    hidden[0], candidates, candidate_ids, queries, residues, groups
                )
            observed_tokens.update(candidate_ids)
            del outputs, hidden, inputs
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    token_ids = sorted(observed_tokens)
    rows = [
        [state, *select_tokens(compare_affinities(accumulator, token_ids, tokenizer))]
        for state, accumulator in zip(HIDDEN_STATES, accumulators)
    ]
    save_table(rows, OUTPUT)
    print(f"Saved {len(rows)} hidden states to {OUTPUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="CUDA device (for example cuda:2) or cpu; FP16 on CUDA, FP32 on CPU.",
    )
    device = torch.device(parser.parse_args().device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    run_analysis(device)


if __name__ == "__main__":
    main()
