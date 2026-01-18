"""Create code-level upgrade table for Stage 2 prescriptions.

This table summarizes *actionable upgrades* where the top recommended treatment
represents a change from the current infrastructure level.

Source: rerun Stage-2 outputs in:
    stage2_outputs/.../stage2_cf_prescriptions/stage2_segment_prescriptions_*.csv

Method:
- Include all recommended segment-treatment rows in the Stage-2 prescriptions file.
- Keep only actionable upgrades (target_level > current_level).
- Aggregate counts by (treatment, current_level, target_level, labels).

Important:
- Levels in the prescriptions file are the canonical levels used by the pipeline.
- For manuscript-style reporting, we also display the *raw survey codes* (from the iRAP codebook).
    Note: for some treatments, "better" corresponds to a *lower* raw code (e.g., Delineation 2→1).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

import stage2_config as config
from stage2_treatment_codebook import get_spec


CODEBOOK_LABELS_CSV = Path(__file__).parent / "codes_filled_analysis.csv"


TREATMENT_ORDER = [
    "Centreline rumble strips",
    "Street lighting",
    "Delineation",
    "Road condition",
    "Paved shoulder - passenger-side",
    "Paved shoulder - driver-side",
]

TREATMENT_DISPLAY = {
    "Paved shoulder - passenger-side": "Paved shoulder – passenger side",
    "Paved shoulder - driver-side": "Paved shoulder – driver side",
}

# If True: restrict to the top-ranked treatment per segment.
# Manuscript Table (per your description) should generally be False (include all upgrades).
TOP_ONLY = False


def _runs_root() -> Path:
    return Path(__file__).resolve().parent / 'stage2_outputs' / 'runs'


def resolve_stage2_root(*, stage2_output_dir: str | None, run_id: str | None) -> Path:
    if stage2_output_dir:
        return Path(stage2_output_dir).expanduser().resolve()
    if run_id:
        return (_runs_root() / str(run_id)).resolve()
    return Path(config.OUTPUT_DIR).resolve()


def _latest_file(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")
    return matches[0]


def _latest_full_menu_prescriptions(stage2_cf_dir: Path) -> Path:
    """Return newest full MENU prescriptions file, excluding convenience exports."""
    prefix = "stage2_segment_prescriptions"
    candidates = list(stage2_cf_dir.glob(f"{prefix}_*.csv"))
    full_menu_re = re.compile(rf"^{re.escape(prefix)}_\d{{8}}_\d{{4}}$")
    candidates = [p for p in candidates if full_menu_re.match(p.stem)]
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No full MENU prescriptions found in {stage2_cf_dir}. "
            "Expected a file like stage2_segment_prescriptions_YYYYMMDD_HHMM.csv"
        )
    return candidates[0]


def _ensure_required(df: pd.DataFrame, cols: list[str], source: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")


def _canonical_to_raw_code(treatment: str, canonical_level: int) -> int:
    spec = get_spec(treatment)
    inv = {v: k for k, v in spec.raw_to_canonical.items()}
    if canonical_level not in inv:
        raise ValueError(
            f"Cannot invert canonical level {canonical_level} for '{treatment}'. "
            f"Known canonical levels: {sorted(inv.keys())}."
        )
    return int(inv[canonical_level])


def _load_codebook_labels() -> dict[tuple[str, int], str]:
    """Return mapping (Attribute Name, Code) -> Attribute Class Name (label).

    Note: codes_filled_analysis.csv contains some blank labels (e.g., paved shoulder code 4).
    Those will be treated as missing and handled by fallback text.
    """
    cb = pd.read_csv(CODEBOOK_LABELS_CSV)
    cb = cb.dropna(subset=["Attribute Name", "Code"])
    cb["Code"] = pd.to_numeric(cb["Code"], errors="coerce")
    cb = cb.dropna(subset=["Code"])
    cb["Code"] = cb["Code"].astype(int)
    out: dict[tuple[str, int], str] = {}
    for _, r in cb.iterrows():
        name = str(r["Attribute Name"])
        code = int(r["Code"])
        label = r["Attribute Class Name"]
        if pd.isna(label):
            continue
        label_str = str(label).strip()
        if not label_str:
            continue
        out[(name, code)] = label_str
    return out


def _codebook_label(
    labels: dict[tuple[str, int], str], treatment: str, raw_code: int
) -> str:
    # Handle known naming mismatch in codes_filled_analysis.csv
    if treatment == "Centreline rumble strips":
        candidates = [
            ("Centreline rumble strips", raw_code),
            ("Centre line rumble strips", raw_code),
        ]
    else:
        candidates = [(treatment, raw_code)]

    for key in candidates:
        if key in labels:
            return labels[key]

    # Fallbacks for codebook blanks / missing entries
    if treatment.startswith("Paved shoulder") and raw_code == 4:
        return "Missing / not recorded (code 4)"
    return f"Code {raw_code}"


def build_upgrade_table(segment_prescriptions_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(segment_prescriptions_csv)

    codebook_labels = _load_codebook_labels()

    required = [
        "segment_id",
        "treatment",
        "rank_within_segment",
        "current_level",
        "target_level",
        "current_level_label",
        "target_level_label",
    ]
    _ensure_required(df, required, segment_prescriptions_csv)

    if TOP_ONLY:
        subset = df[df["rank_within_segment"] == 1].copy()
        label = "top recommendations"
    else:
        subset = df.copy()
        label = "all recommendation rows"

    # Sanity check: prescriptions should be a subset of the 396 Stage 1 candidate hotspots.
    # The prescriptions file contains only segments with ≥1 recommended upgrade, so it will
    # typically be <396 (no-change hotspots are absent).
    n_unique = int(subset["segment_id"].nunique())
    if n_unique <= 0:
        raise ValueError(f"No segments found in {label} from {segment_prescriptions_csv}")

    # Upgrades only
    subset["current_level"] = pd.to_numeric(subset["current_level"], errors="raise")
    subset["target_level"] = pd.to_numeric(subset["target_level"], errors="raise")
    upgrades = subset[subset["target_level"] > subset["current_level"]].copy()

    # Aggregate
    grouped = (
        upgrades.groupby(
            [
                "treatment",
                "current_level",
                "target_level",
                "current_level_label",
                "target_level_label",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="N")
    )

    # Display formatting
    # 1) Canonical (internal) codes: shown as 1-based for readability.
    grouped["Codes (canonical)"] = (
        (grouped["current_level"].astype(int) + 1).astype(str)
        + " → "
        + (grouped["target_level"].astype(int) + 1).astype(str)
    )

    # 2) Raw survey codes: invert the codebook mapping back to original survey values.
    grouped["raw_current_code"] = grouped.apply(
        lambda r: _canonical_to_raw_code(str(r["treatment"]), int(r["current_level"])), axis=1
    )
    grouped["raw_target_code"] = grouped.apply(
        lambda r: _canonical_to_raw_code(str(r["treatment"]), int(r["target_level"])), axis=1
    )
    grouped["Codes"] = grouped["raw_current_code"].astype(str) + " → " + grouped["raw_target_code"].astype(str)

    # Scientific safety check: verify the displayed raw-code change corresponds to a true upgrade
    # under the canonical meaning (higher canonical = better).
    for _, row in grouped.iterrows():
        spec = get_spec(str(row["treatment"]))
        rc = int(row["raw_current_code"])
        rt = int(row["raw_target_code"])
        if spec.raw_to_canonical[rt] <= spec.raw_to_canonical[rc]:
            raise ValueError(
                "Detected non-upgrade after raw<->canonical mapping: "
                f"{row['treatment']}: raw {rc}->{rt} maps to canonical "
                f"{spec.raw_to_canonical[rc]}->{spec.raw_to_canonical[rt]}"
            )

    grouped["Treatment"] = grouped["treatment"].map(lambda t: TREATMENT_DISPLAY.get(t, t))

    # Labels in the prescriptions CSV are pipeline-generated and may use simplified wording.
    # For scientific reporting, derive labels directly from the raw iRAP codebook where possible.
    grouped["Current label"] = grouped.apply(
        lambda r: _codebook_label(codebook_labels, str(r["treatment"]), int(r["raw_current_code"])), axis=1
    )
    grouped["Prescribed change"] = grouped.apply(
        lambda r: _codebook_label(codebook_labels, str(r["treatment"]), int(r["raw_target_code"])), axis=1
    )

    # Keep the pipeline labels as audit columns.
    grouped["Current label (pipeline)"] = grouped["current_level_label"].astype(str)
    grouped["Prescribed change (pipeline)"] = grouped["target_level_label"].astype(str)

    # Ordering
    order_map = {t: i for i, t in enumerate(TREATMENT_ORDER)}
    grouped["_treat_order"] = grouped["treatment"].map(order_map).fillna(999).astype(int)

    grouped = grouped.sort_values(
        by=["_treat_order", "treatment", "current_level", "target_level"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )

    out = grouped[
        [
            "Treatment",
            "Codes",
            "Codes (canonical)",
            "Current label",
            "Prescribed change",
            "N",
            "Current label (pipeline)",
            "Prescribed change (pipeline)",
        ]
    ].reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a code-level upgrade table for Stage 2 prescriptions, counting only actionable upgrades "
            "(target != current)."
        )
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Stage 2 run id under stage2_outputs/runs/<run-id>. "
            "If provided, reads prescriptions from that run folder and writes to <run>/reports."
        ),
    )
    parser.add_argument(
        "--stage2-output-dir",
        default=None,
        help=(
            "Explicit Stage 2 output directory containing stage2_cf_prescriptions/. Overrides --run-id."
        ),
    )
    parser.add_argument(
        "--top-only",
        action="store_true",
        help="If set, restrict to the top-ranked treatment per segment (portfolio-style).",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help=(
            "Optional explicit output CSV path. If omitted, writes to <stage2-root>/reports/table_code_level_upgrades_v27.csv."
        ),
    )
    args = parser.parse_args()

    global TOP_ONLY
    TOP_ONLY = bool(args.top_only)

    stage2_root = resolve_stage2_root(stage2_output_dir=args.stage2_output_dir, run_id=args.run_id)
    stage2_cf_dir = (stage2_root / "stage2_cf_prescriptions").resolve()

    input_csv = _latest_full_menu_prescriptions(stage2_cf_dir)

    if args.out_csv:
        out_csv = Path(args.out_csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    else:
        reports_dir = (stage2_root / "reports").resolve() if (args.run_id or args.stage2_output_dir) else config.OUTPUT_SUBDIRS["reports"].resolve()
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_csv = reports_dir / "table_code_level_upgrades_v27.csv"

    print(f"Using input: {input_csv}")
    table = build_upgrade_table(input_csv)
    table.to_csv(out_csv, index=False)

    print(f"Wrote: {out_csv}")
    print(f"Rows: {len(table)}")


if __name__ == "__main__":
    main()
