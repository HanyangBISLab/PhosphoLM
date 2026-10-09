#!/usr/bin/env python3
"""Plot the completed fixed-PDB token-distance analysis, including Pfam bars.

The statistical and rendering functions below are ported from the original
plot_phospho_motif_distance_distribution.py. This script does not load PyMOL.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Sequence
import math

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from calculate_phosphosite_token_distances import (  # noqa: E402
    ROOT,
    DEFAULT_ANALYSIS,
    DEFAULT_TOKENS,
    read_tokens,
)

PFAM_URL = "https://ftp.ebi.ac.uk/pub/databases/Pfam/mappings/pdb_pfam_mapping.txt"

UNIQUE_COLUMNS = ("token", "phospho_monomer", "pdb_id")


TOKEN_LOCATION_COLUMNS = (
    "token_chain_id",
    "token_residue_start_number",
    "token_residue_start_insertion_code",
)


DISTANCE_COLUMN = "min_distance_A"


COUNT_COLUMNS = (
    "token",
    "distance_lower_A",
    "distance_upper_A",
    "distance_midpoint_A",
    "structure_count",
)


PROTEIN_FAMILIES_COLOR = "#009E73"


HEATMAP_COLOR_STOPS = ("#FFF5F5", "#FFCCCC", "#FF6666", "#FF0000")


PFAM_SOURCE_COLUMNS = (
    "PDB",
    "CHAIN",
    "PDB_START",
    "PDB_END",
    "PFAM_ACCESSION",
    "PFAM_NAME",
    "AUTH_PDBRES_START",
    "AUTH_PDBRES_START_INS_CODE",
    "AUTH_PDBRES_END",
    "AUTH_PDBRES_END_INS_CODE",
)


PFAM_MAPPING_COLUMNS = (
    "pdb_id",
    "chain_id",
    "pfam_accession",
    "pfam_name",
    "auth_start_number",
    "auth_start_insertion_code",
    "auth_end_number",
    "auth_end_insertion_code",
    "mapping_method",
)


PFAM_COUNT_COLUMNS = (
    "token",
    "protein_family_count",
    "total_observation_count",
    "mapped_observation_count",
    "unmapped_observation_count",
)


PFAM_AUDIT_COLUMNS = (
    "observation_id",
    "token",
    "phospho_monomer",
    "pdb_id",
    "token_chain_id",
    "token_residue_start_number",
    "token_residue_start_insertion_code",
    "pfam_accession",
    "pfam_name",
    "pfam_auth_start_number",
    "pfam_auth_start_insertion_code",
    "pfam_auth_end_number",
    "pfam_auth_end_insertion_code",
    "mapping_status",
    "mapping_method",
    "unmatched_reason",
)


PFAM_CHUNK_SIZE = 250_000


MM_PER_INCH = 25.4


A4_WIDTH_MM = 210.0


A4_HEIGHT_MM = 297.0


HEATMAP_WIDTH_MM = A4_WIDTH_MM


HEATMAP_HEIGHT_MM = A4_HEIGHT_MM / 3.0


DEFAULT_DPI = 450


AXIS_LINE_COLOR = "#222222"


AXIS_LINE_WIDTH = 0.75


HEATMAP_SPINE_GAP_POINTS = 1.5


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
            "axes.linewidth": AXIS_LINE_WIDTH,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_and_deduplicate(input_csv: Path) -> pd.DataFrame:
    data = pd.read_csv(input_csv, keep_default_na=False)

    numeric_distances = pd.to_numeric(data[DISTANCE_COLUMN], errors="coerce")

    selected = data.loc[
        :,
        (*UNIQUE_COLUMNS, DISTANCE_COLUMN, *TOKEN_LOCATION_COLUMNS),
    ].copy()
    selected[DISTANCE_COLUMN] = numeric_distances
    selected["token_chain_id"] = selected["token_chain_id"].astype("string").str.strip()
    selected["token_chain_id"] = selected["token_chain_id"].mask(
        selected["token_chain_id"].eq("")
    )
    residue_start = pd.to_numeric(
        selected["token_residue_start_number"],
        errors="coerce",
    )
    valid_start = np.isfinite(residue_start) & np.isclose(
        residue_start,
        np.round(residue_start),
        rtol=0.0,
        atol=1e-12,
    )
    selected["token_residue_start_number"] = (
        residue_start.where(valid_start).round().astype("Int64")
    )
    selected["token_residue_start_insertion_code"] = (
        selected["token_residue_start_insertion_code"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
    )
    selected = selected.drop_duplicates(subset=list(UNIQUE_COLUMNS), keep="first")
    selected = selected.sort_values(list(UNIQUE_COLUMNS), kind="mergesort")
    selected = selected.reset_index(drop=True)
    return selected


def _parse_nullable_integer(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_values = numeric.to_numpy(dtype=np.float64)
    valid = np.isfinite(numeric_values) & np.isclose(
        numeric_values,
        np.round(numeric_values),
        rtol=0.0,
        atol=1e-12,
    )
    parsed = np.full(len(values), np.nan, dtype=np.float64)
    parsed[valid] = np.round(numeric_values[valid])
    return pd.Series(
        pd.array(parsed, dtype="Int64"),
        index=values.index,
    )


def _normalize_insertion_codes(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
        .replace({".": "", "?": ""})
    )


def _empty_pfam_mappings() -> pd.DataFrame:
    mappings = pd.DataFrame(columns=PFAM_MAPPING_COLUMNS)
    for column in (
        "auth_start_number",
        "auth_end_number",
    ):
        mappings[column] = mappings[column].astype("Int64")
    return mappings


def load_pfam_mappings(
    mapping_path: Path,
    relevant_pdb_ids: Iterable[object],
    chunk_size: int = PFAM_CHUNK_SIZE,
) -> pd.DataFrame:

    mapping_path = Path(mapping_path)

    relevant = {
        str(value).strip().lower()
        for value in relevant_pdb_ids
        if not pd.isna(value) and str(value).strip()
    }
    if not relevant:
        return _empty_pfam_mappings()

    selected_chunks: list[pd.DataFrame] = []
    chunks = pd.read_csv(
        mapping_path,
        sep="\t",
        comment="#",
        usecols=list(PFAM_SOURCE_COLUMNS),
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    )
    for chunk in chunks:
        normalized_pdb = (
            chunk["PDB"].astype("string").fillna("").str.strip().str.lower()
        )
        keep = normalized_pdb.isin(relevant)
        if not keep.any():
            continue
        selected = chunk.loc[keep, PFAM_SOURCE_COLUMNS].copy()
        selected["PDB"] = normalized_pdb.loc[keep]
        selected_chunks.append(selected)

    if not selected_chunks:
        return _empty_pfam_mappings()

    source = pd.concat(selected_chunks, ignore_index=True)
    pdb_start = _parse_nullable_integer(source["PDB_START"])
    pdb_end = _parse_nullable_integer(source["PDB_END"])
    author_start = _parse_nullable_integer(source["AUTH_PDBRES_START"])
    author_end = _parse_nullable_integer(source["AUTH_PDBRES_END"])
    start_insertion = _normalize_insertion_codes(source["AUTH_PDBRES_START_INS_CODE"])
    end_insertion = _normalize_insertion_codes(source["AUTH_PDBRES_END_INS_CODE"])

    label_span = pdb_end - pdb_start
    valid_span = label_span.notna() & label_span.ge(0)
    has_start = author_start.notna()
    has_end = author_end.notna()
    explicit = has_start & has_end
    infer_end = has_start & ~has_end & valid_span
    infer_start = ~has_start & has_end & valid_span

    resolved_start = author_start.copy()
    resolved_end = author_end.copy()
    resolved_end.loc[infer_end] = (
        author_start.loc[infer_end] + label_span.loc[infer_end]
    )
    resolved_start.loc[infer_start] = (
        author_end.loc[infer_start] - label_span.loc[infer_start]
    )
    start_insertion.loc[infer_start] = ""
    end_insertion.loc[infer_end] = ""

    mapping_method = pd.Series("", index=source.index, dtype="string")
    mapping_method.loc[explicit] = "explicit_author_bounds"
    mapping_method.loc[infer_end] = "inferred_author_end"
    mapping_method.loc[infer_start] = "inferred_author_start"

    resolved = mapping_method.ne("")
    same_number = resolved_start.eq(resolved_end).fillna(False)
    ordered_numbers = resolved_start.lt(resolved_end).fillna(False)
    ordered_insertions = start_insertion.le(end_insertion)
    ordered_ranges = resolved & (ordered_numbers | (same_number & ordered_insertions))
    mapping_method.loc[resolved & ~ordered_ranges] = ""

    mappings = pd.DataFrame(
        {
            "pdb_id": source["PDB"],
            "chain_id": (source["CHAIN"].astype("string").fillna("").str.strip()),
            "pfam_accession": (
                source["PFAM_ACCESSION"].astype("string").fillna("").str.strip()
            ),
            "pfam_name": (source["PFAM_NAME"].astype("string").fillna("").str.strip()),
            "auth_start_number": resolved_start,
            "auth_start_insertion_code": start_insertion,
            "auth_end_number": resolved_end,
            "auth_end_insertion_code": end_insertion,
            "mapping_method": mapping_method,
        }
    )
    mappings = mappings.loc[mappings["pfam_accession"].ne("")].copy()
    mappings = mappings.drop_duplicates(
        subset=list(PFAM_MAPPING_COLUMNS),
        keep="first",
    )
    mappings = mappings.sort_values(
        [
            "pdb_id",
            "chain_id",
            "pfam_accession",
            "auth_start_number",
            "auth_start_insertion_code",
            "auth_end_number",
            "auth_end_insertion_code",
            "pfam_name",
        ],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    return mappings.loc[:, PFAM_MAPPING_COLUMNS]


def _residue_position(number: object, insertion_code: object) -> tuple[int, str]:
    return int(number), str(insertion_code or "")


def _empty_audit_value() -> dict[str, object]:
    return {
        "pfam_accession": "",
        "pfam_name": "",
        "pfam_auth_start_number": pd.NA,
        "pfam_auth_start_insertion_code": "",
        "pfam_auth_end_number": pd.NA,
        "pfam_auth_end_insertion_code": "",
        "mapping_method": "",
    }


def map_observations_to_pfam(
    observations: pd.DataFrame,
    mappings: pd.DataFrame,
    token_order: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    normalized = observations.loc[
        :,
        (*UNIQUE_COLUMNS, *TOKEN_LOCATION_COLUMNS),
    ].copy()
    normalized.insert(
        0,
        "observation_id",
        np.arange(1, len(normalized) + 1, dtype=np.int64),
    )
    normalized["token"] = normalized["token"].astype("string").fillna("").astype(str)
    normalized["phospho_monomer"] = (
        normalized["phospho_monomer"].astype("string").fillna("").astype(str)
    )
    normalized["pdb_id"] = (
        normalized["pdb_id"].astype("string").fillna("").str.strip().str.lower()
    )
    normalized["token_chain_id"] = (
        normalized["token_chain_id"].astype("string").fillna("").str.strip()
    )
    normalized["token_residue_start_number"] = _parse_nullable_integer(
        normalized["token_residue_start_number"]
    )
    normalized["token_residue_start_insertion_code"] = _normalize_insertion_codes(
        normalized["token_residue_start_insertion_code"]
    )

    mapping_lookup: dict[tuple[str, str], list[object]] = {}
    for mapping in mappings.itertuples(index=False):
        key = (str(mapping.pdb_id), str(mapping.chain_id))
        mapping_lookup.setdefault(key, []).append(mapping)

    audit_rows: list[dict[str, object]] = []
    mapped_observation_ids: set[int] = set()
    token_pfams: dict[str, set[str]] = {}
    for observation in normalized.itertuples(index=False):
        observation_id = int(observation.observation_id)
        token = str(observation.token)
        chain_id = str(observation.token_chain_id)
        start_number = observation.token_residue_start_number
        start_insertion = str(observation.token_residue_start_insertion_code)
        audit_base = {
            "observation_id": observation_id,
            "token": token,
            "phospho_monomer": str(observation.phospho_monomer),
            "pdb_id": str(observation.pdb_id),
            "token_chain_id": chain_id,
            "token_residue_start_number": start_number,
            "token_residue_start_insertion_code": start_insertion,
        }

        if not chain_id or pd.isna(start_number):
            audit_rows.append(
                {
                    **audit_base,
                    **_empty_audit_value(),
                    "mapping_status": "missing_token_location",
                    "unmatched_reason": "missing_token_location",
                }
            )
            continue

        candidates = mapping_lookup.get(
            (str(observation.pdb_id), chain_id),
            [],
        )
        if not candidates:
            audit_rows.append(
                {
                    **audit_base,
                    **_empty_audit_value(),
                    "mapping_status": "no_pdb_chain_mapping",
                    "unmatched_reason": "no_pdb_chain_mapping",
                }
            )
            continue

        token_position = _residue_position(start_number, start_insertion)
        resolvable: list[object] = []
        matches: list[object] = []
        for candidate in candidates:
            if (
                not str(candidate.mapping_method)
                or pd.isna(candidate.auth_start_number)
                or pd.isna(candidate.auth_end_number)
            ):
                continue
            resolvable.append(candidate)
            interval_start = _residue_position(
                candidate.auth_start_number,
                candidate.auth_start_insertion_code,
            )
            interval_end = _residue_position(
                candidate.auth_end_number,
                candidate.auth_end_insertion_code,
            )
            if interval_start <= token_position <= interval_end:
                matches.append(candidate)

        if matches:
            distinct_matches: dict[str, object] = {}
            for match in matches:
                distinct_matches.setdefault(str(match.pfam_accession), match)
            mapped_observation_ids.add(observation_id)
            token_pfams.setdefault(token, set()).update(distinct_matches)
            for accession, match in distinct_matches.items():
                audit_rows.append(
                    {
                        **audit_base,
                        "pfam_accession": accession,
                        "pfam_name": str(match.pfam_name),
                        "pfam_auth_start_number": match.auth_start_number,
                        "pfam_auth_start_insertion_code": str(
                            match.auth_start_insertion_code
                        ),
                        "pfam_auth_end_number": match.auth_end_number,
                        "pfam_auth_end_insertion_code": str(
                            match.auth_end_insertion_code
                        ),
                        "mapping_status": "matched",
                        "mapping_method": str(match.mapping_method),
                        "unmatched_reason": "",
                    }
                )
            continue

        reason = "outside_pfam_ranges" if resolvable else "unresolvable_author_bounds"
        audit_rows.append(
            {
                **audit_base,
                **_empty_audit_value(),
                "mapping_status": reason,
                "unmatched_reason": reason,
            }
        )

    observed_tokens = normalized["token"].astype(str).tolist()
    if token_order is None:
        ordered_tokens = sorted(set(observed_tokens))
    else:
        ordered_tokens = [str(token) for token in token_order]

    total_counts = normalized.groupby("token", sort=False).size().to_dict()
    mapped_counts = (
        normalized.loc[normalized["observation_id"].isin(mapped_observation_ids)]
        .groupby("token", sort=False)
        .size()
        .to_dict()
    )
    count_rows = []
    for token in ordered_tokens:
        total = int(total_counts.get(token, 0))
        mapped = int(mapped_counts.get(token, 0))
        count_rows.append(
            {
                "token": token,
                "protein_family_count": len(token_pfams.get(token, set())),
                "total_observation_count": total,
                "mapped_observation_count": mapped,
                "unmapped_observation_count": total - mapped,
            }
        )
    pfam_counts = pd.DataFrame(count_rows, columns=PFAM_COUNT_COLUMNS)
    for column in PFAM_COUNT_COLUMNS[1:]:
        pfam_counts[column] = pfam_counts[column].astype(np.int64)

    audit = pd.DataFrame(audit_rows, columns=PFAM_AUDIT_COLUMNS)
    if not audit.empty:
        audit["observation_id"] = audit["observation_id"].astype(np.int64)
        audit["token_residue_start_number"] = audit[
            "token_residue_start_number"
        ].astype("Int64")
        audit["pfam_auth_start_number"] = audit["pfam_auth_start_number"].astype(
            "Int64"
        )
        audit["pfam_auth_end_number"] = audit["pfam_auth_end_number"].astype("Int64")
        audit = audit.sort_values(
            ["observation_id", "pfam_accession"],
            kind="mergesort",
        ).reset_index(drop=True)
    return pfam_counts, audit


def build_binned_counts(data: pd.DataFrame) -> pd.DataFrame:
    distances = data[DISTANCE_COLUMN].to_numpy(dtype=np.float64)
    distance_min = float(distances.min())
    distance_max = float(distances.max())
    first_lower = math.floor(distance_min)
    final_upper = math.ceil(distance_max)
    if final_upper <= first_lower:
        final_upper = first_lower + 1

    bin_lowers = np.arange(first_lower, final_upper, dtype=np.int64)
    assigned_lowers = np.floor(distances).astype(np.int64)
    final_boundary = np.isclose(distances, float(final_upper), rtol=0.0, atol=1e-12)
    assigned_lowers[final_boundary] = final_upper - 1

    working = data.assign(distance_lower_A=assigned_lowers)
    tokens = sorted(working["token"].astype(str).unique().tolist())
    full_index = pd.MultiIndex.from_product(
        [tokens, bin_lowers.tolist()],
        names=["token", "distance_lower_A"],
    )
    counts = (
        working.assign(token=working["token"].astype(str))
        .groupby(["token", "distance_lower_A"], sort=True)["pdb_id"]
        .nunique()
        .reindex(full_index, fill_value=0)
        .rename("structure_count")
        .reset_index()
    )
    counts["distance_upper_A"] = counts["distance_lower_A"] + 1
    counts["distance_midpoint_A"] = counts["distance_lower_A"] + 0.5
    counts["structure_count"] = counts["structure_count"].astype(np.int64)
    return counts.loc[:, COUNT_COLUMNS]


def build_token_distance_matrix(token_counts: pd.DataFrame) -> pd.DataFrame:
    pivot = token_counts.pivot(
        index="token",
        columns="distance_lower_A",
        values="structure_count",
    ).fillna(0)
    pivot = pivot.astype(np.int64)
    totals = pivot.sum(axis=1).astype(np.int64)
    order = (
        pd.DataFrame({"token": pivot.index, "total_structures": totals.to_numpy()})
        .sort_values(
            ["total_structures", "token"],
            ascending=[False, True],
            kind="mergesort",
        )["token"]
        .tolist()
    )
    pivot = pivot.loc[order]
    renamed_columns = {
        lower: f"{int(lower)}–{int(lower) + 1} Å" for lower in pivot.columns
    }
    matrix = pivot.rename(columns=renamed_columns).reset_index()
    matrix["total_structures"] = totals.loc[order].to_numpy(dtype=np.int64)
    return matrix


def create_token_heatmap(
    matrix: pd.DataFrame,
    pfam_counts: pd.DataFrame,
) -> tuple[
    plt.Figure,
    tuple[plt.Axes, plt.Axes, plt.Axes],
]:
    distance_columns = [
        column
        for column in matrix.columns
        if column not in {"token", "total_structures"}
    ]
    values = matrix[distance_columns].to_numpy(dtype=np.int64).T
    tokens = matrix["token"].astype(str).tolist()
    family_by_token = pfam_counts.assign(
        token=pfam_counts["token"].astype(str)
    ).set_index("token")["protein_family_count"]
    family_counts = family_by_token.loc[tokens].to_numpy(dtype=np.int64)

    figure = plt.figure(
        figsize=(
            HEATMAP_WIDTH_MM / MM_PER_INCH,
            HEATMAP_HEIGHT_MM / MM_PER_INCH,
        )
    )
    grid = figure.add_gridspec(
        3,
        2,
        width_ratios=(45.0, 0.75),
        height_ratios=(2.15, 5.30, 0.65),
        left=0.075,
        right=0.92,
        bottom=0.06,
        top=0.975,
        wspace=0.08,
        hspace=0.08,
    )
    family_axis = figure.add_subplot(grid[0, 0])
    heatmap_axis = figure.add_subplot(grid[1, 0], sharex=family_axis)
    colorbar_axis = figure.add_subplot(grid[1, 1])
    family_axis.set_zorder(heatmap_axis.get_zorder() + 1)
    family_axis.patch.set_alpha(0.0)

    maximum = int(values.max())
    color_map = LinearSegmentedColormap.from_list(
        "light_to_full_red",
        HEATMAP_COLOR_STOPS,
    )
    color_map.set_bad("white")
    normalization = LogNorm(vmin=1, vmax=maximum)
    image = heatmap_axis.imshow(
        np.ma.masked_equal(values, 0),
        cmap=color_map,
        norm=normalization,
        aspect="auto",
        interpolation="none",
    )
    heatmap_axis.set_xticks(np.arange(len(tokens)), tokens)
    heatmap_axis.set_yticks(
        np.arange(len(distance_columns)),
        distance_columns,
    )
    heatmap_axis.set_xlabel(
        "Phosphosite-proximal PhosphoLM token",
        labelpad=4.0,
    )
    heatmap_axis.set_ylabel("Minimum token–phosphoresidue distance (Å)")
    heatmap_axis.tick_params(
        axis="x",
        bottom=True,
        top=False,
        length=2.0,
        width=AXIS_LINE_WIDTH,
        direction="out",
        pad=1.5,
        labelbottom=True,
        labelrotation=90,
    )
    heatmap_axis.tick_params(
        axis="y",
        left=True,
        right=False,
        length=2.0,
        width=AXIS_LINE_WIDTH,
        direction="out",
        pad=1.5,
    )
    heatmap_axis.set_xticks(
        np.arange(-0.5, len(tokens), 1.0),
        minor=True,
    )
    heatmap_axis.set_yticks(
        np.arange(-0.5, len(distance_columns), 1.0),
        minor=True,
    )
    heatmap_axis.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=0.45,
    )
    heatmap_axis.tick_params(which="minor", bottom=False, left=False)
    for spine in heatmap_axis.spines.values():
        spine.set_visible(False)
    heatmap_axis.spines["left"].set_visible(True)
    heatmap_axis.spines["left"].set_color(AXIS_LINE_COLOR)
    heatmap_axis.spines["left"].set_position(("outward", HEATMAP_SPINE_GAP_POINTS))
    heatmap_axis.spines["bottom"].set_visible(True)
    heatmap_axis.spines["bottom"].set_color(AXIS_LINE_COLOR)
    heatmap_axis.spines["bottom"].set_position(("outward", HEATMAP_SPINE_GAP_POINTS))

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            count = int(values[row_index, column_index])
            if count == 0:
                continue
            text_color = "white" if normalization(count) >= 0.58 else "#111111"
            heatmap_axis.text(
                column_index,
                row_index,
                str(count),
                ha="center",
                va="center",
                color=text_color,
                fontsize=5.0,
            )

    positions = np.arange(len(tokens), dtype=np.float64)
    family_axis.bar(
        positions,
        family_counts,
        width=0.78,
        color=PROTEIN_FAMILIES_COLOR,
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )
    family_axis.set_xlim(-0.5, len(tokens) - 0.5)
    maximum_families = float(family_counts.max(initial=0))
    family_axis.set_ylim(0.0, max(1.0, maximum_families * 1.24))
    family_axis.set_xticks(positions, tokens)
    family_axis.set_xlabel("")
    family_axis.set_ylabel("Distinct protein families")
    family_axis.xaxis.set_ticks_position("bottom")
    family_axis.xaxis.set_label_position("bottom")
    family_axis.tick_params(
        axis="x",
        bottom=False,
        top=False,
        labelbottom=False,
        labeltop=False,
        length=0.0,
        width=AXIS_LINE_WIDTH,
        direction="out",
        pad=1.5,
        labelrotation=90,
    )
    family_axis.tick_params(
        axis="y",
        length=2.0,
        width=AXIS_LINE_WIDTH,
        direction="out",
        pad=1.0,
    )
    family_axis.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    family_axis.spines["top"].set_visible(False)
    family_axis.spines["right"].set_visible(False)
    family_axis.spines["left"].set_color(AXIS_LINE_COLOR)
    family_axis.spines["bottom"].set_visible(True)
    family_axis.spines["bottom"].set_color(AXIS_LINE_COLOR)
    family_axis.spines["bottom"].set_position(("outward", HEATMAP_SPINE_GAP_POINTS))
    family_label_offset = max(0.15, maximum_families * 0.025)
    for position, family_count in zip(
        positions,
        family_counts,
        strict=True,
    ):
        if family_count == 0:
            continue
        family_axis.text(
            position,
            float(family_count) + family_label_offset,
            str(int(family_count)),
            ha="center",
            va="bottom",
            fontsize=5.0,
            rotation=90,
            clip_on=False,
        )

    tick_candidates = np.asarray(
        [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
        dtype=np.int64,
    )
    colorbar_ticks = tick_candidates[tick_candidates <= maximum].tolist()
    if maximum not in colorbar_ticks:
        if (
            len(colorbar_ticks) > 1
            and math.log(maximum / colorbar_ticks[-1]) / math.log(maximum) < 0.05
        ):
            colorbar_ticks.pop()
        colorbar_ticks.append(maximum)
    colorbar = figure.colorbar(image, cax=colorbar_axis, ticks=colorbar_ticks)
    colorbar.set_label("Unique PDB entries", labelpad=2.0)
    colorbar.ax.tick_params(
        length=2.0,
        width=AXIS_LINE_WIDTH,
        pad=1.0,
    )
    colorbar.outline.set_linewidth(AXIS_LINE_WIDTH)
    colorbar.outline.set_edgecolor(AXIS_LINE_COLOR)
    colorbar.outline.set_snap(True)
    colorbar.ax.set_yticklabels([str(value) for value in colorbar_ticks])
    for axis, visible_sides in (
        (family_axis, ("left", "bottom")),
        (heatmap_axis, ("left", "bottom")),
    ):
        for side in visible_sides:
            spine = axis.spines[side]
            spine.set_linewidth(AXIS_LINE_WIDTH)
            spine.set_edgecolor(AXIS_LINE_COLOR)
            spine.set_snap(True)
            spine.set_zorder(5)
    heatmap_position = heatmap_axis.get_position()
    gap_x_fraction = (
        HEATMAP_SPINE_GAP_POINTS
        / 72.0
        / (figure.get_figwidth() * heatmap_position.width)
    )
    gap_y_fraction = (
        HEATMAP_SPINE_GAP_POINTS
        / 72.0
        / (figure.get_figheight() * heatmap_position.height)
    )
    connector_style = {
        "color": AXIS_LINE_COLOR,
        "linewidth": AXIS_LINE_WIDTH,
        "solid_capstyle": "butt",
        "clip_on": False,
        "transform": heatmap_axis.transAxes,
        "zorder": 5,
    }
    heatmap_axis.add_line(
        Line2D(
            [-gap_x_fraction, -gap_x_fraction],
            [-gap_y_fraction, 0.0],
            **connector_style,
        )
    )
    heatmap_axis.add_line(
        Line2D(
            [-gap_x_fraction, 0.0],
            [-gap_y_fraction, -gap_y_fraction],
            **connector_style,
        )
    )
    family_position = family_axis.get_position()
    family_gap_y_fraction = (
        HEATMAP_SPINE_GAP_POINTS
        / 72.0
        / (figure.get_figheight() * family_position.height)
    )
    family_axis.add_line(
        Line2D(
            [0.0, 0.0],
            [-family_gap_y_fraction, 0.0],
            color=AXIS_LINE_COLOR,
            linewidth=AXIS_LINE_WIDTH,
            solid_capstyle="butt",
            clip_on=False,
            transform=family_axis.transAxes,
            zorder=5,
        )
    )
    return figure, (
        heatmap_axis,
        family_axis,
        colorbar_axis,
    )


def ensure_pfam_mapping(path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(PFAM_URL, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for block in response.iter_content(1024 * 1024):
                handle.write(block)


def include_zero_hit_tokens(counts, tokens):
    index = pd.MultiIndex.from_product(
        [tokens, sorted(counts["distance_lower_A"].unique())],
        names=["token", "distance_lower_A"],
    )
    result = counts.set_index(["token", "distance_lower_A"])[["structure_count"]]
    result = result.reindex(index, fill_value=0).reset_index()
    result["distance_upper_A"] = result["distance_lower_A"] + 1
    result["distance_midpoint_A"] = result["distance_lower_A"] + 0.5
    return result.loc[:, COUNT_COLUMNS]


def run(args):
    tokens = read_tokens(args.tokens_csv)
    data = load_and_deduplicate(args.analysis_dir / "distances.csv")
    ensure_pfam_mapping(args.pfam_mapping)
    counts = include_zero_hit_tokens(build_binned_counts(data), tokens)
    matrix = build_token_distance_matrix(counts)
    mappings = load_pfam_mappings(args.pfam_mapping, data["pdb_id"])
    pfam_counts, audit = map_observations_to_pfam(
        data, mappings, matrix["token"].tolist()
    )
    set_publication_theme()
    figure, _ = create_token_heatmap(matrix, pfam_counts)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "token_distance_matrix.csv": matrix,
        "token_pfam_counts.csv": pfam_counts,
        "token_pfam_mapping_audit.csv": audit,
    }
    image_path = args.output_dir / "token_distance_heatmap.png"
    try:
        for name, frame in artifacts.items():
            frame.to_csv(args.analysis_dir / name, index=False)
        figure.savefig(image_path, dpi=args.dpi, facecolor="white", transparent=False)
    finally:
        plt.close(figure)
    print(f"Saved {image_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--tokens-csv", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument(
        "--pfam-mapping", type=Path, default=ROOT / "datasets/pfam/pdb_pfam_mapping.txt"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "figures/phosphosite_token_distance"
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, (ROOT / value).resolve())
    return args


if __name__ == "__main__":
    run(parse_args())
