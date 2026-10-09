#!/usr/bin/env python3
"""Measure selected tokens against a fixed PDB collection, without structure searches.

Run in the PyMOL environment. One independently owned structure is loaded at a
time. Source selection, coordinate rounding and nearest-occurrence semantics
are retained; bounded NumPy blocks accelerate the exact all-atom minimum.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
DEFAULT_ANALYSIS = ROOT / "analysis/phosphosite_token_distance"
DEFAULT_TOKENS = (
    ROOT / "analysis/phosphosite_proximal_token_analysis/selected_tokens.csv"
)
MONOMERS = ("SEP", "TPO", "PTR")
CUTOFF_A = 10.0
MIN_STATE = 15


class HeadlessClient:
    def __init__(self, cmd):
        self.cmd = cmd
        self.pdb_cache = {}
        self.model_cache = {}

    def get_pdbstr(self, selection):
        if selection not in self.pdb_cache:
            self.pdb_cache[selection] = self.cmd.get_pdbstr(selection, state=1)
        return self.pdb_cache[selection]

    def get_model_atoms(self, selection, state=1):
        key = (selection, state)
        if key not in self.model_cache:
            self.model_cache[key] = [
                vars(atom).copy()
                for atom in self.cmd.get_model(selection, state=state).atom
            ]
        return self.model_cache[key]


def minimum_distance(token_atoms, phospho_atoms):
    left = np.array([(a.x, a.y, a.z) for a in token_atoms], dtype=np.float64)
    right = np.array([(a.x, a.y, a.z) for a in phospho_atoms], dtype=np.float64)
    best = math.inf
    for start in range(0, len(left), 512):
        a = left[start : start + 512]
        for stop in range(0, len(right), 512):
            b = right[stop : stop + 512]
            dx = a[:, None, 0] - b[:, 0]
            dy = a[:, None, 1] - b[:, 1]
            dz = a[:, None, 2] - b[:, 2]
            best = min(best, float((dx * dx + dy * dy + dz * dz).min()))
    return math.sqrt(best)


PDB_COORDINATE_DECIMALS = 3


LOCATION_DISTANCE_TOLERANCE_A = 0.005


EXTRA_RESULT_FIELDS = [
    "within_cutoff",
    "distance_status",
    "token_atom_count",
    "phospho_atom_count",
    "min_distance_A",
    "token_chain_id",
    "token_residue_start",
    "token_residue_end",
    "token_residue_start_number",
    "token_residue_start_insertion_code",
    "token_residue_end_number",
    "token_residue_end_insertion_code",
    "token_occurrence_index",
    "token_occurrence_count",
]


THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "ASX": "B",
    "GLX": "Z",
    "UNK": "X",
    "SEC": "U",
    "PYL": "O",
    "MSE": "M",
    "SEP": "S",
    "TPO": "T",
    "PTR": "Y",
}


@dataclass(frozen=True)
class SelectedAtom:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class ResidueIdentity:
    model: str
    segment: str
    chain: str
    number: int
    insertion_code: str = ""

    @property
    def label(self) -> str:
        return f"{self.number}{self.insertion_code}"

    @property
    def polymer_key(self) -> Tuple[str, str, str]:
        return self.model, self.segment, self.chain


@dataclass(frozen=True)
class LocatedAtom(SelectedAtom):
    residue: ResidueIdentity
    residue_name: str


@dataclass(frozen=True)
class ProteinResidue:
    identity: ResidueIdentity
    residue_name: str
    one_letter: str


@dataclass(frozen=True)
class TokenOccurrence:
    occurrence_index: int
    residues: Tuple[ProteinResidue, ...]
    atoms: Tuple[LocatedAtom, ...]

    @property
    def chain_id(self) -> str:
        return self.residues[0].identity.chain

    @property
    def start(self) -> ResidueIdentity:
        return self.residues[0].identity

    @property
    def end(self) -> ResidueIdentity:
        return self.residues[-1].identity


@dataclass(frozen=True)
class OccurrenceDistanceResult:
    occurrence: TokenOccurrence
    phospho_atom_count: int
    min_distance: Optional[float]
    within_cutoff: bool
    status: str


@dataclass(frozen=True)
class DistanceResult:
    token_atom_count: int
    phospho_atom_count: int
    min_distance: Optional[float]
    within_cutoff: bool
    status: str
    nearest_occurrence: Optional[OccurrenceDistanceResult] = None
    occurrence_results: Tuple[OccurrenceDistanceResult, ...] = ()
    location_error: Optional[str] = None


def pymol_object_selection(object_name: str) -> str:
    return f"%{object_name}"


def parse_selected_atoms(pdb_text: str) -> List[SelectedAtom]:
    return [
        SelectedAtom(float(line[30:38]), float(line[38:46]), float(line[46:54]))
        for line in pdb_text.splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
    ]


def selected_atoms(client: HeadlessClient, selection: str) -> List[SelectedAtom]:
    return parse_selected_atoms(client.get_pdbstr(selection))


def parse_model_atoms(raw_atoms: Iterable[Mapping[str, Any]]) -> List[LocatedAtom]:
    atoms: List[LocatedAtom] = []
    for raw_atom in raw_atoms:
        coord = raw_atom["coord"]

        raw_number = raw_atom.get("resi_number")
        insertion_code = str(raw_atom.get("ins_code", "") or "").strip()
        if raw_number is None:
            raw_resi = str(raw_atom.get("resi", "") or "").strip()
            match = re.fullmatch(r"(-?\d+)(.*)", raw_resi)
            raw_number = match.group(1)
            if not insertion_code:
                insertion_code = match.group(2)

        residue = ResidueIdentity(
            model=str(raw_atom.get("model", "") or ""),
            segment=str(raw_atom.get("segi", "") or ""),
            chain=str(raw_atom.get("chain", "") or ""),
            number=int(raw_number),
            insertion_code=insertion_code,
        )
        atoms.append(
            LocatedAtom(
                x=round(float(coord[0]), PDB_COORDINATE_DECIMALS),
                y=round(float(coord[1]), PDB_COORDINATE_DECIMALS),
                z=round(float(coord[2]), PDB_COORDINATE_DECIMALS),
                residue=residue,
                residue_name=str(raw_atom.get("resn", "") or "").strip().upper(),
            )
        )
    return atoms


def selected_located_atoms(
    client: HeadlessClient,
    selection: str,
    state: int = 1,
) -> List[LocatedAtom]:
    return parse_model_atoms(client.get_model_atoms(selection, state=state))


def protein_residues_from_atoms(
    atoms: Iterable[LocatedAtom],
) -> Tuple[ProteinResidue, ...]:
    residues: List[ProteinResidue] = []
    seen = set()
    for atom in atoms:
        if atom.residue in seen:
            continue
        seen.add(atom.residue)
        residues.append(
            ProteinResidue(
                identity=atom.residue,
                residue_name=atom.residue_name,
                one_letter=THREE_TO_ONE.get(atom.residue_name, "?"),
            )
        )
    return tuple(residues)


def find_token_occurrences(
    protein_residues: Sequence[ProteinResidue],
    token_atoms: Sequence[LocatedAtom],
    token: str,
) -> Tuple[TokenOccurrence, ...]:
    if not token or not token_atoms:
        return ()

    selected_residue_keys = {atom.residue for atom in token_atoms}
    atoms_by_residue: "collections.OrderedDict[ResidueIdentity, List[LocatedAtom]]" = (
        collections.OrderedDict()
    )
    for atom in token_atoms:
        atoms_by_residue.setdefault(atom.residue, []).append(atom)

    residues_by_polymer: "collections.OrderedDict[Tuple[str, str, str], List[ProteinResidue]]" = collections.OrderedDict()
    for residue in protein_residues:
        residues_by_polymer.setdefault(residue.identity.polymer_key, []).append(residue)

    occurrences: List[TokenOccurrence] = []
    for residues in residues_by_polymer.values():
        sequence = "".join(residue.one_letter for residue in residues)
        start = sequence.find(token)
        while start >= 0:
            window = tuple(residues[start : start + len(token)])
            if len(window) == len(token) and all(
                residue.identity in selected_residue_keys for residue in window
            ):
                occurrence_atoms = tuple(
                    atom
                    for residue in window
                    for atom in atoms_by_residue.get(residue.identity, ())
                )
                if occurrence_atoms:
                    occurrences.append(
                        TokenOccurrence(
                            occurrence_index=len(occurrences) + 1,
                            residues=window,
                            atoms=occurrence_atoms,
                        )
                    )
            start = sequence.find(token, start + 1)

    return tuple(occurrences)


class StructureContext:
    def __init__(self, client: HeadlessClient, pdb_object: str):
        self.client = client
        self.pdb_object = pdb_object
        self._protein_residues: Optional[Tuple[ProteinResidue, ...]] = None
        self._full_protein_residues: Optional[Tuple[ProteinResidue, ...]] = None
        self._token_atoms: Dict[str, Tuple[LocatedAtom, ...]] = {}
        self._token_occurrences: Dict[str, Tuple[TokenOccurrence, ...]] = {}
        self._location_errors: Dict[str, str] = {}

    def token_atoms(self, token: str) -> Tuple[LocatedAtom, ...]:
        if token not in self._token_atoms:
            selection = (
                f"({pymol_object_selection(self.pdb_object)} and pepseq {token})"
            )
            self._token_atoms[token] = tuple(
                selected_located_atoms(self.client, selection)
            )
        return self._token_atoms[token]

    def protein_residues(self, full: bool = False) -> Tuple[ProteinResidue, ...]:
        if full:
            if self._full_protein_residues is None:
                selection = f"({pymol_object_selection(self.pdb_object)} and polymer.protein and alt +A)"
                atoms = selected_located_atoms(self.client, selection)
                self._full_protein_residues = protein_residues_from_atoms(atoms)
            return self._full_protein_residues

        if self._protein_residues is None:
            selection = (
                f"({pymol_object_selection(self.pdb_object)} "
                "and polymer.protein and guide and alt +A)"
            )
            atoms = selected_located_atoms(self.client, selection)
            self._protein_residues = protein_residues_from_atoms(atoms)
        return self._protein_residues

    def token_occurrences(
        self,
        token: str,
        token_atoms: Sequence[LocatedAtom],
    ) -> Tuple[Tuple[TokenOccurrence, ...], Optional[str]]:
        if token in self._token_occurrences:
            return self._token_occurrences[token], self._location_errors.get(token)
        residues = self.protein_residues()
        residue_keys = {residue.identity for residue in residues}
        token_residue_keys = {atom.residue for atom in token_atoms}
        if not token_residue_keys.issubset(residue_keys):
            residues = self.protein_residues(full=True)
        occurrences = find_token_occurrences(residues, token_atoms, token)
        covered_keys = {
            residue.identity
            for occurrence in occurrences
            for residue in occurrence.residues
        }
        error = None
        if token_atoms and not occurrences:
            error = "PyMOL selected token atoms, but no chain-specific token occurrence could be reconstructed"
        elif missing_keys := token_residue_keys - covered_keys:
            error = f"{len(missing_keys)} selected token residue(s) were not assigned to a complete occurrence"
        if error:
            occurrences = ()
            self._location_errors[token] = error
        self._token_occurrences[token] = occurrences
        return occurrences, error


def pair_distance_result(
    client: HeadlessClient,
    pdb_object: str,
    token: str,
    phospho_code: str,
    distance: float,
    structure_context: Optional[StructureContext] = None,
) -> DistanceResult:
    token_selection = f"({pymol_object_selection(pdb_object)} and pepseq {token})"
    phospho_selection = (
        f"({pymol_object_selection(pdb_object)} and resn {phospho_code})"
    )
    token_atoms = selected_atoms(client, token_selection)
    phospho_atoms = selected_atoms(client, phospho_selection)

    context = structure_context or StructureContext(client, pdb_object)

    if not token_atoms:
        return DistanceResult(
            token_atom_count=0,
            phospho_atom_count=len(phospho_atoms),
            min_distance=None,
            within_cutoff=False,
            status="no_token_atoms",
        )

    located_token_atoms = context.token_atoms(token)
    occurrences, location_error = context.token_occurrences(token, located_token_atoms)

    if not phospho_atoms:
        occurrence_results = tuple(
            OccurrenceDistanceResult(
                occurrence=occurrence,
                phospho_atom_count=0,
                min_distance=None,
                within_cutoff=False,
                status="no_phospho_atoms",
            )
            for occurrence in occurrences
        )
        return DistanceResult(
            token_atom_count=len(token_atoms),
            phospho_atom_count=0,
            min_distance=None,
            within_cutoff=False,
            status="no_phospho_atoms",
            occurrence_results=occurrence_results,
            location_error=location_error,
        )

    min_distance = minimum_distance(token_atoms, phospho_atoms)
    within_cutoff = min_distance <= distance

    occurrence_results: Tuple[OccurrenceDistanceResult, ...] = ()
    if occurrences:
        occurrence_results = tuple(
            OccurrenceDistanceResult(
                occurrence=occurrence,
                phospho_atom_count=len(phospho_atoms),
                min_distance=occurrence_distance,
                within_cutoff=occurrence_distance <= distance,
                status=(
                    "within_cutoff"
                    if occurrence_distance <= distance
                    else "outside_cutoff"
                ),
            )
            for occurrence in occurrences
            for occurrence_distance in (
                minimum_distance(occurrence.atoms, phospho_atoms),
            )
        )

    nearest_occurrence: Optional[OccurrenceDistanceResult] = None
    if occurrence_results:
        candidate = min(
            occurrence_results,
            key=lambda result: (
                result.min_distance if result.min_distance is not None else math.inf
            ),
        )
        if (
            candidate.min_distance is not None
            and abs(candidate.min_distance - min_distance)
            <= LOCATION_DISTANCE_TOLERANCE_A
        ):
            nearest_occurrence = candidate
        else:
            mismatch = (
                "closest reconstructed occurrence does not reproduce the global "
                f"pepseq distance ({candidate.min_distance!r} versus {min_distance!r})"
            )
            location_error = (
                f"{location_error}; {mismatch}" if location_error else mismatch
            )

    return DistanceResult(
        token_atom_count=len(token_atoms),
        phospho_atom_count=len(phospho_atoms),
        min_distance=min_distance,
        within_cutoff=within_cutoff,
        status="within_cutoff" if within_cutoff else "outside_cutoff",
        nearest_occurrence=nearest_occurrence,
        occurrence_results=occurrence_results,
        location_error=location_error,
    )


def result_row(row: Dict[str, str], result: DistanceResult) -> Dict[str, object]:
    nearest = result.nearest_occurrence
    start = nearest.occurrence.start if nearest is not None else None
    end = nearest.occurrence.end if nearest is not None else None
    out_row: Dict[str, object] = dict(row)
    out_row.update(
        {
            "within_cutoff": "true" if result.within_cutoff else "false",
            "distance_status": result.status,
            "token_atom_count": result.token_atom_count,
            "phospho_atom_count": result.phospho_atom_count,
            "min_distance_A": (
                "" if result.min_distance is None else f"{result.min_distance:.3f}"
            ),
            "token_chain_id": "" if nearest is None else nearest.occurrence.chain_id,
            "token_residue_start": "" if start is None else start.label,
            "token_residue_end": "" if end is None else end.label,
            "token_residue_start_number": "" if start is None else start.number,
            "token_residue_start_insertion_code": (
                "" if start is None else start.insertion_code
            ),
            "token_residue_end_number": "" if end is None else end.number,
            "token_residue_end_insertion_code": (
                "" if end is None else end.insertion_code
            ),
            "token_occurrence_index": (
                "" if nearest is None else nearest.occurrence.occurrence_index
            ),
            "token_occurrence_count": len(result.occurrence_results),
        }
    )
    return out_row


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def read_pdb_ids(path):
    payload = json.loads(Path(path).read_text())
    return [record["identifier"] for record in payload["result_set"]]


def read_tokens(path):
    tokens = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["Hidden state"]) < MIN_STATE:
                continue
            for name in ("S tokens", "T tokens", "Y tokens"):
                for token in row[name].split(","):
                    token = token.strip()
                    if token and token not in tokens:
                        tokens.append(token)
    return tokens


def prepare_structure(pdb_id, directory, local_directory, session):
    destination = directory / f"{pdb_id}.cif"
    if destination.exists():
        return destination
    if local_directory:
        for name in (f"{pdb_id.lower()}.cif", f"{pdb_id}.cif"):
            local = local_directory / name
            if local.is_file():
                shutil.copyfile(local, destination)
                return destination
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    with session.get(url, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for block in response.iter_content(1024 * 1024):
                handle.write(block)
    return destination


def analyze_structure(cmd, path, pdb_id, tokens):
    object_name = "phospholm_distance_structure"
    rows, warnings = [], []
    statuses = collections.Counter()
    try:
        cmd.load(str(path), object_name, format="cif", quiet=1)
        cmd.frame(1)
        client = HeadlessClient(cmd)
        context = StructureContext(client, object_name)
        for token in tokens:
            for monomer in MONOMERS:
                result = pair_distance_result(
                    client, object_name, token, monomer, CUTOFF_A, context
                )
                statuses[result.status] += 1
                if result.within_cutoff:
                    row = result_row(
                        {
                            "token": token,
                            "phospho_monomer": monomer,
                            "pdb_id": pdb_id,
                        },
                        result,
                    )
                    rows.append(row)
                    if result.location_error:
                        warnings.append(
                            {
                                "token": token,
                                "monomer": monomer,
                                "reason": result.location_error,
                            }
                        )
        return rows, dict(statuses), warnings
    finally:
        cmd.delete(object_name)


def run(args):
    import pymol2

    ids = read_pdb_ids(args.pdb_list)
    tokens = read_tokens(args.tokens_csv)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    args.structures_dir.mkdir(parents=True, exist_ok=True)
    progress = output / "structures"
    progress.mkdir(exist_ok=True)
    all_rows = []
    with pymol2.PyMOL() as instance, requests.Session() as session:
        cmd = instance.cmd
        cmd.set("cif_use_auth", 1)
        cmd.set("max_threads", 1)
        cmd.feedback("disable", "all", "everything")
        for index, pdb_id in enumerate(ids, 1):
            record_path = progress / f"{pdb_id}.json"
            record = json.loads(record_path.read_text()) if record_path.exists() else {}
            if not record.get("complete"):
                path = prepare_structure(
                    pdb_id, args.structures_dir, args.local_structures, session
                )
                rows, counts, location_warnings = analyze_structure(
                    cmd, path, pdb_id, tokens
                )
                record = {
                    "pdb_id": pdb_id,
                    "complete": True,
                    "rows": rows,
                    "statuses": counts,
                    "location_warnings": location_warnings,
                }
                save_json(record_path, record)
            all_rows.extend(record["rows"])
            if index % 25 == 0 or index == len(ids):
                print(
                    f"[{index}/{len(ids)}] qualifying_observations={len(all_rows)}",
                    flush=True,
                )
    fields = ["token", "phospho_monomer", "pdb_id"] + EXTRA_RESULT_FIELDS
    destination = output / "distances.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved {destination}; {len(all_rows)} observations", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdb-list", type=Path, default=ROOT / "datasets/pdb/SEP_TPO_PTR_PDBs.json"
    )
    parser.add_argument("--tokens-csv", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument(
        "--structures-dir", type=Path, default=ROOT / "datasets/pdb/structures"
    )
    parser.add_argument(
        "--local-structures",
        type=Path,
        help="Optional existing mmCIF directory to copy from",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS)
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, (ROOT / value).resolve())
    return args


if __name__ == "__main__":
    run(parse_args())
