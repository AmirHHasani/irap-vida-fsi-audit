"""Build Table 8: Treatment prevalence on the 396 Stage 1 candidate hotspots.

Source of hotspot list:
- Stage 2 run folder (preferred): stage2_outputs/runs/<run-id>/hierarchical_cf/hotspot_level/hotspot_segments_detailed.csv
- Legacy default (fallback): stage2_config.OUTPUT_DIR/hierarchical_cf/hotspot_level/hotspot_segments_detailed.csv

Source of raw treatment codes:
- input_data/segments_unique.csv (raw segment coding)

Important: Table 8 is a descriptive prevalence table and should use the *raw coded levels*
(1/2/3/4) with their original meanings, not any canonical/ordinal remapping used for modeling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import reporter_config as rcfg


def _format_share(count: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{(100.0 * count / total):.1f}%"


def build_table8(hotspot_segments_detailed_csv: Path, segments_unique_csv: Path) -> pd.DataFrame:
    hotspots = pd.read_csv(hotspot_segments_detailed_csv)
    if "Location ID" not in hotspots.columns:
        raise ValueError(
            f"Expected 'Location ID' column in {hotspot_segments_detailed_csv}, got: {list(hotspots.columns)[:20]} ..."
        )

    # Candidate hotspots are already 396 rows in this file (TP+FP by construction).
    hotspot_ids = hotspots[["Location ID"]].drop_duplicates()

    segments = pd.read_csv(
        segments_unique_csv,
        usecols=[
            "Location ID",
            "Centreline rumble strips",
            "Delineation",
            "Street lighting",
            "Paved shoulder - driver-side",
            "Paved shoulder - passenger-side",
            "Road condition",
        ],
    )

    merged = hotspot_ids.merge(segments, on="Location ID", how="left", validate="one_to_one")
    total = len(merged)
    if total != 396:
        # Keep going, but make it obvious something is off.
        raise ValueError(f"Expected 396 hotspot rows after merge, got {total}.")

    specs = [
        {
            "treatment": "Centreline rumble strips",
            "col": "Centreline rumble strips",
            "levels": [
                (1, "Absent (code 1)"),
                (2, "Present (code 2)"),
            ],
        },
        {
            "treatment": "Delineation",
            "col": "Delineation",
            "levels": [
                (1, "Adequate/good (code 1)"),
                (2, "Poor (code 2)"),
            ],
        },
        {
            "treatment": "Street lighting",
            "col": "Street lighting",
            "levels": [
                (1, "Not present (code 1)"),
                (2, "Present (code 2)"),
            ],
        },
        {
            "treatment": "Paved shoulder – driver side",
            "col": "Paved shoulder - driver-side",
            "levels": [
                (1, "Wide (≥ 2.4 m, code 1)"),
                (2, "Medium (1 m to < 2.4 m, code 2)"),
                (3, "Narrow (0 m to < 1 m, code 3)"),
                (4, "None (code 4)"),
            ],
        },
        {
            "treatment": "Paved shoulder – passenger side",
            "col": "Paved shoulder - passenger-side",
            "levels": [
                (1, "Wide (≥ 2.4 m, code 1)"),
                (2, "Medium (1 m to < 2.4 m, code 2)"),
                (3, "Narrow (0 m to < 1 m, code 3)"),
                (4, "None (code 4)"),
            ],
        },
        {
            "treatment": "Road condition",
            "col": "Road condition",
            "levels": [
                (1, "Good (code 1)"),
                (2, "Medium (code 2)"),
                (3, "Poor (code 3)"),
            ],
        },
    ]

    rows: list[dict[str, object]] = []
    for spec in specs:
        values = pd.to_numeric(merged[spec["col"]], errors="coerce")
        if values.isna().any():
            missing = int(values.isna().sum())
            raise ValueError(
                f"Found {missing} missing values for '{spec['col']}' after merge. "
                "Table 8 expects complete coding on the 396 hotspots."
            )

        values_int = values.astype(int)
        for code, label in spec["levels"]:
            count = int((values_int == code).sum())
            rows.append(
                {
                    "Treatment": spec["treatment"],
                    "Level (code meaning)": label,
                    "Count": count,
                    "Share": _format_share(count, total),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Table 8 prevalence for the 396 candidate hotspots.")
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Stage 2 run id under stage2_outputs/runs/<run-id>. "
            "If provided, hotspots are read from that run folder and output is written to <run-folder>/reports."
        ),
    )
    parser.add_argument(
        "--stage2-output-dir",
        default=None,
        help=(
            "Explicit Stage 2 output directory containing hierarchical_cf/... outputs. "
            "Overrides --run-id."
        ),
    )
    parser.add_argument(
        "--segments-unique-csv",
        default=None,
        help="Optional override for Phase 2 segments_unique.csv (raw coded levels).",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help=(
            "Optional explicit output CSV path. If omitted, writes to <stage2-output-dir>/reports/table8_prevalence_v27.csv."
        ),
    )
    args = parser.parse_args()

    stage2_dir = rcfg.resolve_stage2_root(stage2_output_dir=args.stage2_output_dir, run_id=args.run_id)
    hotspot_csv = stage2_dir / "hierarchical_cf" / "hotspot_level" / "hotspot_segments_detailed.csv"

    segments_csv = Path(args.segments_unique_csv).expanduser().resolve() if args.segments_unique_csv else rcfg.SEGMENTS_UNIQUE_CSV

    if args.out_csv:
        out_csv = Path(args.out_csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = rcfg.ensure_reports_dir(stage2_dir)
        out_csv = out_dir / "table8_prevalence_v27.csv"

    table8 = build_table8(hotspot_csv, segments_csv)
    table8.to_csv(out_csv, index=False)

    print(f"Hotspots: {hotspot_csv}")
    print(f"Raw codes: {segments_csv}")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
