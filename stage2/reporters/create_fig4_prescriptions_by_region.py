"""FIGURE 4. Distribution of Stage 2 prescriptions by reporting group and treatment type.

Uses the Stage 2 prescription summaries produced by the Stage 2 prescription step.
This figure is *not* based on raw treatment codes; it's based on the final prescription choice
(top-ranked treatment per segment; portfolio_top1), summarized per reporting group.

Input:
    stage2_outputs/.../(runs/<run-id>/)?/stage2_cf_prescriptions/stage2_region_treatment_summaries_*.csv
Output:
    stage2_outputs/.../(runs/<run-id>/)?/reports/fig4_prescriptions_by_region.(png|pdf)
"""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd

import stage2_config as config


def _runs_root() -> Path:
    return Path(__file__).resolve().parent / 'stage2_outputs' / 'runs'


def resolve_stage2_root(*, stage2_output_dir: str | None, run_id: str | None) -> Path:
    """Resolve the Stage 2 output root that contains stage2_cf_prescriptions/."""
    if stage2_output_dir:
        return Path(stage2_output_dir).expanduser().resolve()
    if run_id:
        return (_runs_root() / str(run_id)).resolve()
    return Path(config.OUTPUT_DIR).resolve()


TREATMENT_ORDER = [
    "Centreline rumble strips",
    "Delineation",
    "Paved shoulder - driver-side",
    "Paved shoulder - passenger-side",
    "Road condition",
    "Street lighting",
]

REGION_ORDER = [
    "EU Central/Adriatic",
    "Western Balkans (non-EU)",
    "EU Southeast Europe",
    "Eastern Europe",
]

REGION_COLORS = {
    "EU Central/Adriatic": "#4C72B0",  # blue
    "Western Balkans (non-EU)": "#C44E52",  # red
    "EU Southeast Europe": "#55A868",  # green
    "Eastern Europe": "#8172B3",  # purple
}


def _latest_file(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")
    return matches[0]


def _latest_region_treatment_summary(stage2_cf_dir: Path) -> Path:
    """Prefer the portfolio_top1 summary when multiple summaries exist."""
    stage2_cf_dir = Path(stage2_cf_dir).resolve()
    if not stage2_cf_dir.exists():
        raise FileNotFoundError(f"Stage 2 CF prescription directory not found: {stage2_cf_dir}")

    candidates = sorted(
        stage2_cf_dir.glob("stage2_region_treatment_summaries_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No files found for pattern 'stage2_region_treatment_summaries_*.csv' in {stage2_cf_dir}"
        )

    top1 = [p for p in candidates if "portfolio_top1" in p.name]
    return top1[0] if top1 else candidates[0]


def build_fig4(input_csv: Path, output_dir: Path) -> tuple[Path, Path]:
    df = pd.read_csv(input_csv)

    required = {"region", "treatment", "share_of_region_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {input_csv}: {sorted(missing)}")

    df["region"] = df["region"].astype(str)
    df["treatment"] = df["treatment"].astype(str)

    # Backward-compatible rename (older runs used 'Ukraine' as the grouped label).
    df["region"] = df["region"].replace({"Ukraine": "Eastern Europe"})

    # Keep only expected regions/treatments (guards against drift in upstream naming).
    seen_regions = set(df["region"].astype(str).unique().tolist())
    seen_treatments = set(df["treatment"].astype(str).unique().tolist())

    before = len(df)
    df = df[df["region"].isin(REGION_ORDER) & df["treatment"].isin(TREATMENT_ORDER)].copy()
    dropped = before - len(df)
    if dropped:
        unexpected_regions = sorted(seen_regions - set(REGION_ORDER))
        unexpected_treatments = sorted(seen_treatments - set(TREATMENT_ORDER))
        print(f"Note: dropped {dropped:,} rows with unexpected labels")
        if unexpected_regions:
            print(f"  Unexpected reporting groups: {unexpected_regions}")
        if unexpected_treatments:
            print(f"  Unexpected treatments: {unexpected_treatments}")

    pivot = (
        df.pivot_table(
            index="treatment", columns="region", values="share_of_region_pct", aggfunc="first"
        )
        .reindex(index=TREATMENT_ORDER, columns=REGION_ORDER)
        .fillna(0.0)
    )

    # Plot: grouped bars
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })

    fig, ax = plt.subplots(figsize=(13.5, 7.5))

    x = range(len(TREATMENT_ORDER))
    # Compute symmetric offsets so the grouped bars stay centered as region count changes.
    n_regions = len(REGION_ORDER)
    bar_width = 0.18 if n_regions >= 4 else 0.22
    centered = [(i - (n_regions - 1) / 2) * bar_width for i in range(n_regions)]
    offsets = {region: centered[i] for i, region in enumerate(REGION_ORDER)}

    for region in REGION_ORDER:
        y = pivot[region].values
        ax.bar(
            [i + offsets[region] for i in x],
            y,
            width=bar_width,
            label=region,
            color=REGION_COLORS.get(region, None),
            edgecolor="none",
        )

    ax.set_ylabel("Share of prescriptions (%)", fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [
            "Centreline rumble strips",
            "Delineation",
            "Paved shoulder – driver side",
            "Paved shoulder – passenger side",
            "Road condition",
            "Street lighting",
        ],
        rotation=25,
        ha="right",
    )

    # Force bold tick labels (matplotlib doesn't always apply rcParams to tick labels)
    for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        tick.set_fontweight("bold")

    ax.set_ylim(0, max(70, float(pivot.max().max()) + 5))
    legend = ax.legend(title="Reporting group", frameon=True)
    if legend is not None:
        try:
            legend.get_title().set_fontweight("bold")
            for t in legend.get_texts():
                t.set_fontweight("bold")
        except Exception:
            pass

    # Title intentionally removed (requested)
    ax.set_title("")

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "fig4_prescriptions_by_region.png"
    out_pdf = output_dir / "fig4_prescriptions_by_region.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    return out_png, out_pdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create Figure 4: distribution of Stage 2 prescriptions by reporting group and treatment type. "
            "Run-aware via --run-id (stage2_outputs/runs/<run-id>)."
        )
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Stage 2 run id under stage2_outputs/runs/<run-id>. If provided, reads/writes within that run folder.",
    )
    parser.add_argument(
        "--stage2-output-dir",
        default=None,
        help=(
            "Explicit Stage 2 output root directory containing stage2_cf_prescriptions/. Overrides --run-id."
        ),
    )
    parser.add_argument(
        "--in-csv",
        default=None,
        help="Optional explicit input CSV path. Overrides auto-detection.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory for figures. Defaults to <run>/reports (or config OUTPUT_SUBDIRS['reports']).",
    )
    args = parser.parse_args()

    stage2_root = resolve_stage2_root(stage2_output_dir=args.stage2_output_dir, run_id=args.run_id)
    stage2_cf_dir = (stage2_root / "stage2_cf_prescriptions").resolve()

    if args.in_csv:
        input_csv = Path(args.in_csv).expanduser().resolve()
    else:
        input_csv = _latest_region_treatment_summary(stage2_cf_dir)

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = (stage2_root / "reports") if (args.run_id or args.stage2_output_dir) else config.OUTPUT_SUBDIRS["reports"]

    print(f"Stage 2 root: {stage2_root}")
    print(f"Using input: {input_csv}")
    print(f"Last modified: {pd.Timestamp.fromtimestamp(input_csv.stat().st_mtime)}")

    out_png, out_pdf = build_fig4(input_csv, out_dir)

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()
