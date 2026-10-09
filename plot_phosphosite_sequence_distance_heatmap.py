#!/usr/bin/env python3
"""Plot sequence distances from the top ten structural tokens to dbPTM sites."""

import argparse
from bisect import bisect_left, bisect_right
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent
TOP_N = 10
FIRST_STATE = 15
RESIDUES = ("S", "T", "Y")
COUNT_COLUMNS = [
    "phosphosite_residue",
    "token",
    "relative_position",
    "direction",
    "raw_occurrence_pair_count",
    "unique_phosphosite_count",
    "unique_protein_count",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    paths = {
        "input-json": "datasets/dbptm/dbptm.json",
        "selected-tokens": "analysis/phosphosite_proximal_token_analysis/selected_tokens.csv",
        "distance-matrix": "analysis/phosphosite_token_distance/token_distance_matrix.csv",
        "analysis-dir": "analysis/phosphosite_sequence_distance",
        "figure-dir": "figures/phosphosite_sequence_distance",
    }
    for option, default in paths.items():
        parser.add_argument(f"--{option}", type=Path, default=Path(default))
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args(argv)
    for option in paths:
        name = option.replace("-", "_")
        setattr(args, name, (ROOT / getattr(args, name)).resolve())
    return args


def select_tokens(matrix_path, selected_path):
    matrix = pd.read_csv(matrix_path, keep_default_na=False)
    matrix["total_structures"] = pd.to_numeric(matrix["total_structures"])
    tokens = (
        matrix.sort_values(["total_structures", "token"], ascending=[False, True])[
            "token"
        ]
        .head(TOP_N)
        .tolist()
    )

    selected = pd.read_csv(selected_path, keep_default_na=False)
    states = pd.to_numeric(selected["Hidden state"], errors="raise")
    associations = {token: set() for token in tokens}
    for _, row in selected.loc[states >= FIRST_STATE].iterrows():
        for residue in RESIDUES:
            for token in str(row[f"{residue} tokens"]).split(","):
                token = token.strip()
                if token in associations:
                    associations[token].add(residue)
    return tokens, {
        token: next(iter(residues)) for token, residues in associations.items()
    }


def relative_token_edge_position(start, end, site):
    if end < site:
        return end - site
    if start > site:
        return start - site
    return 0


def analyze_records(records, tokens, associations, window=100, show_progress=True):
    shape = (len(tokens), 2 * window + 1)
    pairs, sites, proteins = (np.zeros(shape, dtype=np.int64) for _ in range(3))
    stats = {"proteins": 0, "sites": 0, "residues": 0, "longest_sequence": 0}
    for record in tqdm(
        records, desc="Scanning dbPTM", unit="protein", disable=not show_progress
    ):
        sequence, annotated = record["Seq"], record["Site"]
        sequence = sequence.strip().upper()
        unique_sites = sorted(set(annotated))
        by_residue = {
            residue: [site for site in unique_sites if sequence[site] == residue]
            for residue in RESIDUES
        }
        stats["proteins"] += 1
        stats["sites"] += len(unique_sites)
        stats["residues"] += len(sequence)
        stats["longest_sequence"] = max(stats["longest_sequence"], len(sequence))

        for index, token in enumerate(tokens):
            positions = by_residue[associations[token]]
            site_hits, protein_hits = set(), set()
            start = sequence.find(token)
            while start != -1:
                end = start + len(token) - 1
                first = bisect_left(positions, start - window)
                last = bisect_right(positions, end + window)
                for site in positions[first:last]:
                    distance = relative_token_edge_position(start, end, site)
                    pairs[index, distance + window] += 1
                    site_hits.add((distance, site))
                    protein_hits.add(distance)
                start = sequence.find(token, start + 1)
            for distance, _ in site_hits:
                sites[index, distance + window] += 1
            for distance in protein_hits:
                proteins[index, distance + window] += 1

    rows = []
    for index, token in enumerate(tokens):
        for distance in range(-window, window + 1):
            column = distance + window
            direction = (
                "N-terminal"
                if distance < 0
                else "C-terminal"
                if distance > 0
                else "overlap"
            )
            rows.append(
                [
                    associations[token],
                    token,
                    distance,
                    direction,
                    pairs[index, column],
                    sites[index, column],
                    proteins[index, column],
                ]
            )
    return pd.DataFrame(rows, columns=COUNT_COLUMNS), stats


def log_ticks(maximum):
    ticks, magnitude = [], 1
    while magnitude <= maximum:
        ticks.extend(value for m in (1, 2, 5) if (value := m * magnitude) <= maximum)
        magnitude *= 10
    if maximum not in ticks:
        if ticks and maximum / ticks[-1] < 1.5:
            ticks[-1] = maximum
        else:
            ticks.append(maximum)
    return ticks


def set_colorbar_ticks(figure, colorbar, ticks):
    ticks = list(ticks)
    while True:
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels(
            [f"{value / 1000:.1f}k" if value >= 1000 else str(value) for value in ticks]
        )
        figure.canvas.draw()
        boxes = [
            label.get_window_extent(figure.canvas.get_renderer())
            for label in colorbar.ax.get_yticklabels()
        ]
        collision = next(
            (i for i in range(1, len(boxes)) if boxes[i].y0 < boxes[i - 1].y1 + 1),
            None,
        )
        if collision is None or len(ticks) <= 2:
            return
        ticks.pop(collision - 1 if collision > 1 else collision)


def create_distance_figure(counts, tokens, window=100):
    positions = np.arange(-window, window + 1)
    values = (
        counts.pivot(
            index="token",
            columns="relative_position",
            values="unique_phosphosite_count",
        )
        .reindex(index=tokens, columns=positions)
        .to_numpy()
    )
    maximum = int(values.max())
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 7.0,
            "font.weight": "normal",
            "axes.labelsize": 7.0,
            "axes.labelweight": "normal",
            "axes.titlesize": 7.0,
            "axes.titleweight": "normal",
            "axes.linewidth": 0.75,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    height = 17.82 + len(tokens) * 7 * 25.4 / 72 + 5.5
    figure = plt.figure(figsize=(210 / 25.4, height / 25.4))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(45.0, 0.75),
        left=0.09,
        right=0.92,
        bottom=17.82 / height,
        top=1 - 5.5 / height,
        wspace=0.08,
    )
    axis, colorbar_axis = figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])
    normalization = LogNorm(vmin=1, vmax=max(2, maximum))
    colors = (
        ["#FFF8F8", "#FFF8F8"]
        if maximum == 1
        else ["#FFF8F8", "#FFD6D6", "#FF7373", "#FF0000"]
    )
    color_map = LinearSegmentedColormap.from_list(
        "white_below_threshold_to_full_red", colors
    )
    color_map.set_bad("white")
    image = axis.imshow(
        np.ma.masked_equal(values, 0),
        cmap=color_map,
        norm=normalization,
        aspect="auto",
        interpolation="none",
    )
    locations = np.arange(len(positions), dtype=float)
    major = (positions % 20 == 0) | (np.abs(positions) == window)
    labels = ["0" if p == 0 else f"{p:+d}".replace("-", "−") for p in positions[major]]
    axis.set_xticks(locations[major], labels)
    axis.set_xticks(locations[~major], minor=True)
    context = axis.secondary_xaxis("top")
    context.set_xticks(
        [
            locations[positions < 0].mean(),
            float(window),
            locations[positions > 0].mean(),
        ],
        ["upstream", "phosphosite", "downstream"],
    )
    context.tick_params(axis="x", top=True, bottom=False, length=0, width=0, pad=2.0)
    context.spines["top"].set_visible(True)
    context.spines["top"].set_linewidth(0.75)
    context.spines["top"].set_color("#222222")
    for side in ("left", "right", "bottom"):
        context.spines[side].set_visible(False)
    axis.set_yticks(np.arange(len(tokens)), tokens)
    axis.tick_params(
        axis="x",
        which="minor",
        bottom=True,
        top=False,
        length=2.0,
        width=0.35,
        direction="out",
    )
    axis.tick_params(
        axis="x",
        bottom=True,
        top=False,
        length=2.0,
        width=0.75,
        direction="out",
        pad=1.5,
        labelrotation=90,
    )
    axis.tick_params(
        axis="y",
        left=True,
        right=False,
        length=2.0,
        width=0.75,
        direction="out",
        pad=1.5,
    )
    axis.set_xlabel("Token position relative to phosphosite.", labelpad=4.0)
    axis.set_ylabel("Phosphosite-proximal\nPhosphoLM tokens")
    axis.set_yticks(np.arange(-0.5, len(tokens), 1.0), minor=True)
    axis.grid(which="minor", axis="y", color="white", linestyle="-", linewidth=0.35)
    axis.tick_params(axis="y", which="minor", left=False)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_linewidth(0.75)
        axis.spines[side].set_color("#222222")
    axis.axvline(window, color="#222222", linewidth=0.75, zorder=3)
    colorbar = figure.colorbar(image, cax=colorbar_axis, ticks=log_ticks(maximum))
    colorbar.set_label("Phosphosites", labelpad=2.0)
    colorbar.ax.tick_params(length=2.0, width=0.75, direction="out", pad=1.0)
    colorbar.outline.set_linewidth(0.75)
    colorbar.outline.set_edgecolor("#222222")
    set_colorbar_ticks(figure, colorbar, log_ticks(maximum))
    return figure


def save_outputs(counts, figure, analysis_dir, figure_dir, dpi=450):
    analysis_dir, figure_dir = Path(analysis_dir), Path(figure_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    destinations = (
        analysis_dir / "token_position_counts.csv",
        figure_dir / "token_phosphosite_distance_heatmap.png",
    )
    try:
        counts.to_csv(destinations[0], index=False)
        figure.savefig(destinations[1], dpi=dpi, facecolor="white", transparent=False)
    finally:
        plt.close(figure)
    return destinations


def main(argv=None):
    args = parse_args(argv)
    tokens, associations = select_tokens(args.distance_matrix, args.selected_tokens)
    print("Structural top ten: " + ", ".join(tokens))
    print(
        "Residue associations: " + ", ".join(f"{t}={associations[t]}" for t in tokens)
    )
    with args.input_json.open(encoding="utf-8") as handle:
        records = json.load(handle)
    counts, stats = analyze_records(records, tokens, associations, args.window)
    print("Complete dbPTM scan: " + ", ".join(f"{k}={v:,}" for k, v in stats.items()))
    print(
        f"Qualifying occurrence/site pairs: {counts.raw_occurrence_pair_count.sum():,}"
    )
    figure = create_distance_figure(counts, tokens, args.window)
    for path in save_outputs(
        counts, figure, args.analysis_dir, args.figure_dir, args.dpi
    ):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
