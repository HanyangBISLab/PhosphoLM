#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import auc, precision_recall_curve, roc_curve


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "figures"
TASKS = ("disease", "druggable", "phosphorylation", "functional", "ppi")
FOLDS_BY_TASK = {
    "disease": tuple(range(5)),
    "druggable": tuple(range(5)),
    "phosphorylation": tuple(range(10)),
    "functional": tuple(range(5)),
    "ppi": tuple(range(5)),
}
MODELS_BY_TASK = {
    "disease": ("PhosphoLM", "ESM2-650M", "PTM-Mamba"),
    "druggable": ("PhosphoLM", "ESM2-650M", "PTM-Mamba"),
    "phosphorylation": ("PhosphoLM", "ESM2-650M", "PTM-Mamba"),
    "functional": ("PhosphoLM", "ESM2-650M", "PTM-Mamba"),
    "ppi": ("PhosphoLM", "ESM2-650M", "PTM-Mamba"),
}
MODEL_COLORS = {
    "PhosphoLM": "#0072B2",
    "ESM2-650M": "#E69F00",
    "PTM-Mamba": "#009E73",
}
TASK_DIRECTORIES = {
    "disease": "disease",
    "druggable": "druggable",
    "phosphorylation": "phosphosite",
    "functional": "funcphos",
    "ppi": "ppi",
}
MODEL_DIRECTORIES = {
    "PhosphoLM": "phosphoLM",
    "ESM2-650M": "esm2_650M",
    "PTM-Mamba": "ptm_mamba",
}
PPI_LABELS = {"enhance": 1, "induce": 1, "inhibit": 0}
TARGET_RESIDUES = frozenset({"S", "T", "Y"})
TARGET_SPECIFICITIES = (0.90, 0.95)
BAR_METRICS = ("f1", "mcc", "precision", "recall", "specificity")
SPECIFICITY_BAR_METRICS = ("f1", "mcc", "precision", "recall")
CURVE_BAR_METRICS = ("auroc", "auprc")
PLOTTED_BAR_METRICS = (*SPECIFICITY_BAR_METRICS, *CURVE_BAR_METRICS)
BAR_LABELS = {
    "f1": "F1",
    "mcc": "MCC",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "auroc": "AUROC",
    "auprc": "AUPRC",
}
COMMON_CURVE_POINTS = 500
MM_PER_INCH = 25.4
FIGURE_WIDTH_MM = 91.44
BAR_FIGURE_WIDTH_MM = 2.0 * FIGURE_WIDTH_MM
FIGURE_HEIGHT_MM = 55.845
COMBINED_BAR_HEIGHT_SCALE = 0.90
DEFAULT_DPI = 450


@dataclass(frozen=True)
class PredictionSet:
    task: str
    model: str
    fold: int
    keys: tuple[Any, ...]
    labels: np.ndarray
    probabilities: np.ndarray


@dataclass(frozen=True)
class FoldCurves:
    fpr: np.ndarray
    tpr: np.ndarray
    recall: np.ndarray
    precision: np.ndarray
    prevalence: float
    auroc: float
    auprc: float


def prediction_path(task: str, model: str, fold: int) -> Path:
    family = MODEL_DIRECTORIES[model]
    if task == "phosphorylation" and model == "PTM-Mamba":
        family = "ptm-mamba"
    extension = "json" if task in {"phosphorylation", "functional"} else "csv"
    return (
        ROOT_DIR
        / "output"
        / TASK_DIRECTORIES[task]
        / family
        / "predictions"
        / f"fold_{fold}_test.{extension}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-task ROC/PR and specificity operating-point comparisons."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=list(TASKS),
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parser.parse_args(argv)


def set_publication_theme() -> None:
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
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_sequence_predictions(task: str, model: str, fold: int) -> PredictionSet:
    path = prediction_path(task, model, fold)
    data = pd.read_csv(path)
    labels = data["label"].to_numpy(dtype=np.int8)
    probabilities = data["probability"].to_numpy(dtype=np.float64)
    return PredictionSet(
        task, model, fold, tuple(data["row_index"].tolist()), labels, probabilities
    )


def load_ppi_predictions(model: str, fold: int) -> PredictionSet:
    path = prediction_path("ppi", model, fold)
    data = pd.read_csv(path)
    key_columns = ("target_uniprot", "binder_uniprot", "Effect", "Site")
    effects = data["Effect"].astype(str).str.strip().str.lower()
    mapped = effects.map(PPI_LABELS)
    labels = mapped.to_numpy(dtype=np.int8)
    probabilities = data["probability"].to_numpy(dtype=np.float64)
    keys = tuple(data.loc[:, key_columns].itertuples(index=False, name=None))
    return PredictionSet("ppi", model, fold, keys, labels, probabilities)


def load_phosphorylation_predictions(model: str, fold: int) -> PredictionSet:
    path = prediction_path("phosphorylation", model, fold)
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    keys, labels, probabilities = [], [], []
    for record in records:
        sites = set(record.get("Site", []))
        for index, residue in enumerate(record["Seq"]):
            if residue in TARGET_RESIDUES:
                keys.append((record["Protein"], index))
                labels.append(int(index in sites))
                probabilities.append(float(record["prob"][index]))
    return PredictionSet(
        "phosphorylation",
        model,
        fold,
        tuple(keys),
        np.asarray(labels, dtype=np.int8),
        np.asarray(probabilities, dtype=np.float64),
    )


def load_functional_predictions(model: str, fold: int) -> PredictionSet:
    path = prediction_path("functional", model, fold)
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    keys, labels, probabilities = [], [], []
    for record in records:
        for site, label in zip(record["Site"], record["regular"]):
            keys.append((record["Protein"], site))
            labels.append(label)
            probabilities.append(float(record["prob"][site]))
    return PredictionSet(
        "functional",
        model,
        fold,
        tuple(keys),
        np.asarray(labels, dtype=np.int8),
        np.asarray(probabilities, dtype=np.float64),
    )


def load_predictions(task: str, model: str, fold: int) -> PredictionSet:
    if task in {"disease", "druggable"}:
        return load_sequence_predictions(task, model, fold)
    loader = {
        "phosphorylation": load_phosphorylation_predictions,
        "functional": load_functional_predictions,
        "ppi": load_ppi_predictions,
    }[task]
    return loader(model, fold)


def missing_phosphorylation_counts(sets: dict[str, PredictionSet]) -> dict[str, int]:
    key_sets = {model: set(values.keys) for model, values in sets.items()}
    all_keys = set.union(*key_sets.values())
    return {model: len(all_keys - keys) for model, keys in key_sets.items()}


def build_fold_curves(values: PredictionSet) -> FoldCurves:
    fpr, tpr, _ = roc_curve(values.labels, values.probabilities)
    precision, recall, _ = precision_recall_curve(values.labels, values.probabilities)
    return FoldCurves(
        fpr=fpr,
        tpr=tpr,
        recall=recall,
        precision=precision,
        prevalence=float(values.labels.mean()),
        auroc=float(auc(fpr, tpr)),
        auprc=float(auc(recall, precision)),
    )


def interpolate_roc(
    curves: list[FoldCurves],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, COMMON_CURVE_POINTS)
    matrix = []
    for curve in curves:
        values = np.interp(x, curve.fpr, curve.tpr)
        values[0], values[-1] = 0.0, 1.0
        matrix.append(values)
    array = np.vstack(matrix)
    return x, array.mean(axis=0), array.std(axis=0, ddof=1 if len(array) > 1 else 0)


def interpolate_pr(
    curves: list[FoldCurves],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, COMMON_CURVE_POINTS)
    matrix = []
    for curve in curves:
        recall = curve.recall[::-1]
        precision = curve.precision[::-1]
        unique_recall, indices = np.unique(recall, return_index=True)
        matrix.append(np.interp(x, unique_recall, precision[indices]))
    array = np.vstack(matrix)
    return x, array.mean(axis=0), array.std(axis=0, ddof=1 if len(array) > 1 else 0)


def compute_mcc(tp: int, fp: int, tn: int, fn: int) -> float:
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denominator == 0 else ((tp * tn) - (fp * fn)) / denominator


def specificity_operating_point(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target: float,
) -> dict[str, float]:
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    sorted_labels = labels[order]
    tp_all = np.cumsum(sorted_labels, dtype=np.int64)
    fp_all = np.cumsum(1 - sorted_labels, dtype=np.int64)
    distinct = np.r_[
        np.flatnonzero(np.diff(sorted_probabilities) != 0),
        labels.size - 1,
    ]
    tp, fp = tp_all[distinct], fp_all[distinct]
    positives, negatives = int(labels.sum()), int(labels.size - labels.sum())
    fn, tn = positives - tp, negatives - fp
    tp = np.concatenate(([0], tp))
    fp = np.concatenate(([0], fp))
    tn = np.concatenate(([negatives], tn))
    fn = np.concatenate(([positives], fn))

    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp, dtype=np.float64),
        where=(tp + fp) != 0,
    )
    recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros_like(tp, dtype=np.float64),
        where=(tp + fn) != 0,
    )
    specificity = np.divide(
        tn,
        tn + fp,
        out=np.zeros_like(tp, dtype=np.float64),
        where=(tn + fp) != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp, dtype=np.float64),
        where=(precision + recall) != 0,
    )
    mcc = np.asarray(
        [
            compute_mcc(int(tp_i), int(fp_i), int(tn_i), int(fn_i))
            for tp_i, fp_i, tn_i, fn_i in zip(tp, fp, tn, fn)
        ],
        dtype=np.float64,
    )

    valid = np.flatnonzero(specificity >= target - 1e-12)
    chosen = valid
    best_recall = recall[chosen].max()
    chosen = chosen[np.isclose(recall[chosen], best_recall)]
    best_f1 = f1[chosen].max()
    chosen = chosen[np.isclose(f1[chosen], best_f1)]
    best_mcc = mcc[chosen].max()
    chosen = chosen[np.isclose(mcc[chosen], best_mcc)]
    specificity_gap = specificity[chosen] - target
    chosen = chosen[np.isclose(specificity_gap, specificity_gap.min())]
    index = int(chosen[-1])

    return {
        "f1": float(f1[index]),
        "mcc": float(mcc[index]),
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "specificity": float(specificity[index]),
    }


def collect_task(
    task: str,
) -> tuple[dict[str, list[FoldCurves]], pd.DataFrame, pd.DataFrame]:
    curves = {model: [] for model in MODELS_BY_TASK[task]}
    rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for fold in FOLDS_BY_TASK[task]:
        sets = {
            model: load_predictions(task, model, fold) for model in MODELS_BY_TASK[task]
        }
        skipped_by_model = {model: 0 for model in sets}
        if task == "phosphorylation":
            skipped_by_model = missing_phosphorylation_counts(sets)
        for model, values in sets.items():
            print(
                f"{task} fold {fold} {model}: {values.labels.size} candidates; "
                f"{skipped_by_model[model]} missing relative to model union",
                flush=True,
            )
            fold_curves = build_fold_curves(values)
            curves[model].append(fold_curves)
            curve_rows.append(
                {
                    "task": task,
                    "model": model,
                    "fold": fold,
                    "auroc": fold_curves.auroc,
                    "auprc": fold_curves.auprc,
                }
            )
            for target in TARGET_SPECIFICITIES:
                metrics = specificity_operating_point(
                    values.labels, values.probabilities, target
                )
                rows.append(
                    {
                        "task": task,
                        "model": model,
                        "fold": fold,
                        "n_candidates": int(values.labels.size),
                        "n_skipped_candidates": skipped_by_model[model],
                        "target_specificity": target,
                        **metrics,
                    }
                )
    return curves, pd.DataFrame(rows), pd.DataFrame(curve_rows)


def create_bar_figure(*, height_scale: float = 1.0) -> tuple[plt.Figure, plt.Axes]:
    figure, axis = plt.subplots(
        figsize=(
            BAR_FIGURE_WIDTH_MM / MM_PER_INCH,
            FIGURE_HEIGHT_MM * height_scale / MM_PER_INCH,
        )
    )
    figure.subplots_adjust(left=0.08, right=0.99, top=0.97, bottom=0.20)
    return figure, axis


def create_curve_figure() -> tuple[plt.Figure, plt.Axes]:
    figure, axis = plt.subplots(
        figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH)
    )
    figure.subplots_adjust(left=0.16, right=0.98, top=0.97, bottom=0.17)
    return figure, axis


def style_axis(axis: plt.Axes, *, categorical_x: bool = False) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#222222")
    axis.spines["bottom"].set_color("#222222")
    axis.tick_params(length=2.2, width=0.75, direction="out", pad=1.0)
    if not categorical_x:
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))


def save_figure(figure: plt.Figure, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output_path, dpi=dpi, facecolor="white", transparent=False)
    finally:
        plt.close(figure)


def plot_curves(
    curves_by_model: dict[str, list[FoldCurves]],
    kind: str,
    output_path: Path,
    dpi: int,
) -> None:
    figure, axis = create_curve_figure()
    handles: list[Line2D] = []
    lower_pr_bounds: list[float] = []
    for model, curves in curves_by_model.items():
        if kind == "roc":
            x, mean, std = interpolate_roc(curves)
            scores = np.asarray([curve.auroc for curve in curves], dtype=np.float64)
        else:
            x, mean, std = interpolate_pr(curves)
            scores = np.asarray([curve.auprc for curve in curves], dtype=np.float64)
            lower_pr_bounds.append(float(np.min(np.clip(mean - std, 0.0, 1.0))))
        color = MODEL_COLORS[model]
        axis.fill_between(
            x,
            np.clip(mean - std, 0, 1),
            np.clip(mean + std, 0, 1),
            color=color,
            alpha=0.16,
            linewidth=0,
            rasterized=True,
        )
        axis.plot(x, mean, color=color, linewidth=1.35)
        score_std = float(scores.std(ddof=1 if scores.size > 1 else 0))
        score_label = (
            f"{model} ({scores.mean() * 100.0:.1f}% ± {score_std * 100.0:.2f}%)"
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=1.6,
                label=score_label,
            )
        )
    if kind == "roc":
        axis.plot([0, 1], [0, 1], "--", color="#777777", linewidth=0.75)
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        y_limits = (0.0, 1.02)
    else:
        prevalence = float(
            np.mean([item.prevalence for item in next(iter(curves_by_model.values()))])
        )
        axis.axhline(prevalence, linestyle="--", color="#777777", linewidth=0.75)
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        lowest_value = min(prevalence, *lower_pr_bounds)
        y_min = max(
            0.0,
            float(np.floor((lowest_value - 0.05) / 0.05) * 0.05),
        )
        y_limits = (y_min, 1.01)
    axis.set_xlim(0, 1)
    axis.set_ylim(*y_limits)
    style_axis(axis)
    axis.legend(
        handles=handles,
        loc="best",
        ncol=1,
        frameon=True,
        facecolor="white",
        framealpha=0.82,
        edgecolor="none",
        borderaxespad=0.55,
        borderpad=0.25,
        handlelength=1.3,
        handletextpad=0.35,
        labelspacing=0.35,
    )
    save_figure(figure, output_path, dpi)


def summarize_metrics(per_fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, model, target), group in per_fold.groupby(
        ["task", "model", "target_specificity"], sort=False
    ):
        row: dict[str, Any] = {
            "task": task,
            "model": model,
            "target_specificity": target,
        }
        for metric in BAR_METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1 if len(values) > 1 else 0))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_curve_metrics(per_fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, model), group in per_fold.groupby(["task", "model"], sort=False):
        row: dict[str, Any] = {"task": task, "model": model}
        for metric in CURVE_BAR_METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1 if len(values) > 1 else 0))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_combined_specificity_bars(
    task: str,
    summary: pd.DataFrame,
    curve_summary: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    figure, axis = create_bar_figure(height_scale=COMBINED_BAR_HEIGHT_SCALE)
    models = list(MODELS_BY_TASK[task])
    x = np.arange(len(PLOTTED_BAR_METRICS), dtype=np.float64) * 1.25
    base_slot_width = 0.76 / len(models)
    slot_width = base_slot_width * 1.44
    pair_width = base_slot_width * 0.70
    target_gap = base_slot_width * 0.08
    bar_width = ((pair_width - target_gap) / 2.0) * 1.30 * 1.50
    annotation_tops: list[float] = []

    for model_index, model in enumerate(models):
        positions = x + (model_index - (len(models) - 1) / 2) * slot_width
        rows = {
            target: summary.loc[
                summary["model"].eq(model)
                & np.isclose(summary["target_specificity"], target)
            ].iloc[0]
            for target in TARGET_SPECIFICITIES
        }
        means = {
            target: np.asarray(
                [rows[target][f"{metric}_mean"] for metric in SPECIFICITY_BAR_METRICS]
            )
            for target in TARGET_SPECIFICITIES
        }
        stds = {
            target: np.asarray(
                [rows[target][f"{metric}_std"] for metric in SPECIFICITY_BAR_METRICS]
            )
            for target in TARGET_SPECIFICITIES
        }
        specificity_positions = positions[: len(SPECIFICITY_BAR_METRICS)]
        target_positions = {
            0.95: specificity_positions - (bar_width + target_gap) / 2.0,
            0.90: specificity_positions + (bar_width + target_gap) / 2.0,
        }

        axis.bar(
            target_positions[0.95],
            means[0.95],
            width=bar_width,
            color=MODEL_COLORS[model],
            alpha=0.58,
            edgecolor=MODEL_COLORS[model],
            linewidth=0.45,
            zorder=2,
        )
        axis.bar(
            target_positions[0.90],
            means[0.90],
            width=bar_width,
            facecolor="none",
            edgecolor=MODEL_COLORS[model],
            linewidth=0.70,
            hatch="////",
            zorder=3,
        )
        for target in (0.95, 0.90):
            axis.errorbar(
                target_positions[target],
                means[target],
                yerr=stds[target],
                fmt="none",
                ecolor="#333333",
                elinewidth=0.45,
                capsize=1.5,
                capthick=0.45,
                zorder=4,
            )
            for position, mean, std in zip(
                target_positions[target],
                means[target],
                stds[target],
                strict=True,
            ):
                label_y = float(mean + std + 0.012)
                annotation_tops.append(label_y)
                axis.text(
                    position,
                    label_y,
                    f"{mean:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.0,
                    rotation=90,
                    clip_on=False,
                    zorder=5,
                )

        curve_row = curve_summary.loc[curve_summary["model"].eq(model)].iloc[0]
        curve_means = np.asarray(
            [curve_row[f"{metric}_mean"] for metric in CURVE_BAR_METRICS]
        )
        curve_stds = np.asarray(
            [curve_row[f"{metric}_std"] for metric in CURVE_BAR_METRICS]
        )
        curve_positions = positions[len(SPECIFICITY_BAR_METRICS) :]
        curve_bar_width = base_slot_width * 0.70 * 1.30 * 1.50
        curve_bars = axis.bar(
            curve_positions,
            curve_means,
            width=curve_bar_width,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.4,
            yerr=curve_stds,
            capsize=1.5,
            error_kw={
                "elinewidth": 0.45,
                "capthick": 0.45,
                "ecolor": "#333333",
            },
            zorder=3,
        )
        for bar, mean, std in zip(curve_bars, curve_means, curve_stds):
            label_y = float(mean + std + 0.012)
            annotation_tops.append(label_y)
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                label_y,
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=5.0,
                rotation=90,
                clip_on=False,
                zorder=5,
            )

    axis.set_xticks(
        x,
        [BAR_LABELS[metric] for metric in PLOTTED_BAR_METRICS],
        rotation=20,
        ha="right",
        rotation_mode="anchor",
    )
    axis.set_ylabel("Score")
    axis.set_ylim(-0.05, max(1.05, max(annotation_tops, default=1.0) + 0.055))
    style_axis(axis, categorical_x=True)
    save_figure(figure, output_path, dpi)


def run_task(task: str, output_dir: Path, dpi: int) -> list[Path]:
    curves, per_fold, per_fold_curve_metrics = collect_task(task)
    summary = summarize_metrics(per_fold)
    curve_summary = summarize_curve_metrics(per_fold_curve_metrics)
    task_dir = output_dir / task
    paths = [
        task_dir / "roc_comparison.png",
        task_dir / "precision_recall_comparison.png",
        task_dir / "specificity_90_95_metrics_bar.png",
    ]
    plot_curves(curves, "roc", paths[0], dpi)
    plot_curves(curves, "pr", paths[1], dpi)
    plot_combined_specificity_bars(task, summary, curve_summary, paths[2], dpi)
    return paths


def main() -> None:
    args = parse_args()
    set_publication_theme()
    all_paths = []
    for task in args.tasks:
        paths = run_task(task, DEFAULT_OUTPUT_DIR, args.dpi)
        all_paths.extend(paths)
        print(f"{task}: saved {len(paths)} comparison figures", flush=True)
    print(f"Saved {len(all_paths)} figures under {DEFAULT_OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
