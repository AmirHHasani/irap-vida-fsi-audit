"""
stage1_publication_figures.py
Generate publication-quality figures for the Stage 1 results section.

Connects to stage1_config for paths, target transformation settings,
and output conventions. Reads the OOF predictions CSV produced by the
main pipeline and generates:

  1. Actual vs Predicted scatter (density-coloured, log-FSI scale)

Usage
-----
    python stage1_publication_figures.py [--run-dir <timestamped_run>]

If --run-dir is omitted the script auto-detects the latest run folder
inside stage1_outputs/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Connect to the project config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
import stage1_config as cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_latest_run(output_dir: Path) -> Path:
    """Return the most-recent timestamped run folder."""
    runs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    if not runs:
        raise FileNotFoundError(f"No run directories found in {output_dir}")
    return runs[-1]


def _load_oof_predictions(run_dir: Path) -> pd.DataFrame:
    """Load the out-of-fold segment predictions CSV."""
    oof_path = run_dir / "fold_results" / "oof_predictions_segments.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF predictions not found: {oof_path}")
    df = pd.read_csv(oof_path)
    for col in ("actual_risk", "predicted_risk"):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from {oof_path}")
    return df


# ---------------------------------------------------------------------------
# Figure 1: Actual vs Predicted (density-coloured scatter)
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(
    df: pd.DataFrame,
    out_path: Path,
    *,
    dpi: int = 300,
    figsize: tuple[float, float] = (8, 6),
) -> None:
    """Create a publication-quality actual-vs-predicted scatter.

    Matches the seaborn scatter style used elsewhere in
    stage1_visualizations.py (alpha-blended points, clean grid).
    Values are on the log-FSI scale (the model's training target),
    consistent with the performance metrics reported in the paper.
    """
    import seaborn as sns

    # Use log-FSI directly (model target scale)
    actual = df["actual_risk"].astype(float).values
    predicted = df["predicted_risk"].astype(float).values
    n = len(actual)

    # R² on the log scale (matches Table 4 in the paper)
    ss_res = np.sum((predicted - actual) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.scatterplot(x=actual, y=predicted, alpha=0.4, s=8, ax=ax,
                    linewidth=0, color="#1f77b4")

    # Identity line
    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    margin = (hi - lo) * 0.02
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "--", color="red", linewidth=1.2)

    ax.set_xlabel("Actual log-FSI", fontsize=12)
    ax.set_ylabel("Predicted log-FSI (out-of-fold)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.text(
        0.97, 0.05,
        f"$R^2 = {r2:.3f}$\n$n = {n:,}$",
        transform=ax.transAxes,
        fontsize=10,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[pub-fig] Saved actual-vs-predicted scatter → {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Residuals vs Predicted
# ---------------------------------------------------------------------------

def plot_residuals_vs_predicted(
    df: pd.DataFrame,
    out_path: Path,
    *,
    dpi: int = 300,
    figsize: tuple[float, float] = (8, 6),
) -> None:
    """Residual diagnostic: residuals vs predicted on log-FSI scale.

    Checks for heteroscedasticity and systematic bias.
    Matches the seaborn style of the actual-vs-predicted figure.
    """
    import seaborn as sns

    actual = df["actual_risk"].astype(float).values
    predicted = df["predicted_risk"].astype(float).values
    residuals = actual - predicted

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.scatterplot(x=predicted, y=residuals, alpha=0.4, s=8, ax=ax,
                    linewidth=0, color="#1f77b4")

    # Zero line
    ax.axhline(0, color="red", linestyle="--", linewidth=1.2)

    ax.set_xlabel("Predicted log-FSI (out-of-fold)", fontsize=12)
    ax.set_ylabel("Residual (actual \u2212 predicted)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Annotation: mean and std of residuals
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    ax.text(
        0.97, 0.95,
        f"$\\mu = {mu:.3f}$\n$\\sigma = {sigma:.3f}$",
        transform=ax.transAxes,
        fontsize=10,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[pub-fig] Saved residuals-vs-predicted → {out_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Stage 1 publication figures."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Timestamped run directory name inside stage1_outputs/ "
             "(auto-detects latest if omitted).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Where to write the figures. Defaults to <run-dir>/publication_figures/.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    # Resolve run directory
    base_output = Path(cfg.OUTPUT_DIR)
    if args.run_dir:
        run_dir = base_output / args.run_dir
    else:
        run_dir = _find_latest_run(base_output)
    print(f"[pub-fig] Using run directory: {run_dir}")

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = run_dir / "publication_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = _load_oof_predictions(run_dir)
    print(f"[pub-fig] Loaded {len(df):,} OOF predictions")
    print(f"[pub-fig] Target transformation: {cfg.TARGET_TRANSFORMATION}")
    print(f"[pub-fig] actual_risk range: [{df.actual_risk.min():.3f}, "
          f"{df.actual_risk.max():.3f}]")

    # --- Generate figures ---
    plot_actual_vs_predicted(
        df,
        out_path=out_dir / "oof_actual_vs_predicted.png",
        dpi=args.dpi,
    )

    plot_residuals_vs_predicted(
        df,
        out_path=out_dir / "oof_residuals_vs_predicted.png",
        dpi=args.dpi,
    )

    print(f"\n[pub-fig] All figures saved to {out_dir}")


if __name__ == "__main__":
    main()
