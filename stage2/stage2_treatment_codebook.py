"""Centralized treatment codebook for Stage 2.

This module defines a single source of truth for:
- How raw dataset codes map to canonical numeric levels used by modeling.
- The ordinal upgrade direction (worst -> best).

Design goals:
- Never infer ordering from raw numeric codes (e.g., sorted unique values).
- Be stable across datasets and runs.
- Make downstream artifacts reproducible (write mapping metadata alongside outputs).

Notes:
- Canonical levels are always 0..K-1 where higher = "better" (more protective).
- Raw codes come from the iRAP-style codebook (see codes_filled_analysis.csv).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd


@dataclass(frozen=True)
class TreatmentSpec:
    name: str
    kind: str  # "binary" | "ordinal"
    raw_to_canonical: Dict[int, int]
    canonical_order: list[int]  # worst -> best; always increasing if mapping is correct


# Canonical convention used throughout Stage 2:
# - Higher canonical level = better / more protective.
# - Binary: 0 = not present, 1 = present.
# - Ordinal: 0 = worst, K-1 = best.
TREATMENT_SPECS: Dict[str, TreatmentSpec] = {
    "Centreline rumble strips": TreatmentSpec(
        name="Centreline rumble strips",
        kind="binary",
        # codes_filled_analysis.csv: Not present=1, Present=2
        raw_to_canonical={1: 0, 2: 1},
        canonical_order=[0, 1],
    ),
    "Street lighting": TreatmentSpec(
        name="Street lighting",
        kind="binary",
        # Not present=1, Present=2
        raw_to_canonical={1: 0, 2: 1},
        canonical_order=[0, 1],
    ),
    "Delineation": TreatmentSpec(
        name="Delineation",
        kind="binary",
        # Adequate=1, Poor=2. Canonical: 0=poor, 1=adequate
        raw_to_canonical={2: 0, 1: 1},
        canonical_order=[0, 1],
    ),
    "Road condition": TreatmentSpec(
        name="Road condition",
        kind="ordinal",
        # Good=1, Medium=2, Poor=3. Canonical: 0=poor, 1=medium, 2=good
        raw_to_canonical={3: 0, 2: 1, 1: 2},
        canonical_order=[0, 1, 2],
    ),
    "Paved shoulder - driver-side": TreatmentSpec(
        name="Paved shoulder - driver-side",
        kind="ordinal",
        # codes_filled_analysis.csv indicates:
        # Wide >=2.4m=1, Medium 1m-<2.4m=2, Narrow 0-<1m=3, blank/other=4.
        # Canonical: 0=none/unknown shoulder, 1=0-<1m, 2=1m-<2.4m, 3>=2.4m
        raw_to_canonical={4: 0, 3: 1, 2: 2, 1: 3},
        canonical_order=[0, 1, 2, 3],
    ),
    "Paved shoulder - passenger-side": TreatmentSpec(
        name="Paved shoulder - passenger-side",
        kind="ordinal",
        raw_to_canonical={4: 0, 3: 1, 2: 2, 1: 3},
        canonical_order=[0, 1, 2, 3],
    ),
}


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def codebook_fingerprint() -> str:
    payload = {
        name: {
            "kind": spec.kind,
            "raw_to_canonical": spec.raw_to_canonical,
            "canonical_order": spec.canonical_order,
        }
        for name, spec in TREATMENT_SPECS.items()
    }
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return digest[:12]


def get_spec(treatment_name: str) -> TreatmentSpec:
    try:
        return TREATMENT_SPECS[treatment_name]
    except KeyError as exc:
        raise KeyError(
            f"No canonical mapping defined for treatment '{treatment_name}'. "
            "Add it to stage2_treatment_codebook.TREATMENT_SPECS before modeling."
        ) from exc


def map_series_to_canonical(series: pd.Series, spec: TreatmentSpec, *, strict: bool) -> pd.Series:
    mapped = series.map(spec.raw_to_canonical)

    if strict:
        raw_non_null = series.dropna()
        unmapped_mask = raw_non_null.map(lambda v: int(v) if pd.notna(v) else v).map(
            lambda v: v not in spec.raw_to_canonical
        )
        if unmapped_mask.any():
            bad = sorted(set(raw_non_null[unmapped_mask].astype(int).tolist()))
            raise ValueError(
                f"Unmapped raw codes for '{spec.name}': {bad}. "
                "Update raw_to_canonical mapping before proceeding."
            )

    return mapped


def apply_canonical_treatment_mapping(
    df: pd.DataFrame,
    treatments: Iterable[str],
    *,
    strict: bool = True,
    keep_raw: bool = False,
) -> dict[str, Any]:
    """Map treatment columns in-place to canonical levels.

    Returns a mapping report suitable for writing as JSON.
    """

    report: dict[str, Any] = {
        "codebook_fingerprint": codebook_fingerprint(),
        "treatments": {},
    }

    for name in treatments:
        if name not in df.columns:
            continue
        spec = get_spec(name)

        raw_series = df[name]
        raw_unique = sorted({int(v) for v in raw_series.dropna().unique().tolist()})

        mapped = map_series_to_canonical(raw_series, spec, strict=strict)
        mapped_unique = sorted({int(v) for v in mapped.dropna().unique().tolist()})

        if keep_raw:
            df[f"raw_{name}"] = raw_series

        df[name] = mapped

        report["treatments"][name] = {
            "kind": spec.kind,
            "raw_unique": raw_unique,
            "raw_to_canonical": spec.raw_to_canonical,
            "mapped_unique": mapped_unique,
            "canonical_order": spec.canonical_order,
        }

    return report


def write_mapping_artifact(report: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_stable_json(report), encoding="utf-8")
