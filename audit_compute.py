import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def resolve_latest_stage2_run(project_root: Path) -> Path:
    stage2_outputs_root = project_root / "stage2" / "stage2_outputs"
    candidates: list[Path] = []
    for parent in stage2_outputs_root.iterdir():
        if not parent.is_dir():
            continue
        for child in parent.iterdir():
            if child.is_dir():
                candidates.append(child)
    if not candidates:
        raise FileNotFoundError(f"No Stage 2 run folders found under {stage2_outputs_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ad hoc audit summary for Stage 2 prescriptions.")
    parser.add_argument(
        "--stage2-run-dir",
        type=str,
        default=None,
        help="Path to a timestamped Stage 2 run directory. Defaults to the most recent run.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    stage2_run_dir = (
        Path(args.stage2_run_dir).expanduser().resolve()
        if args.stage2_run_dir
        else resolve_latest_stage2_run(project_root)
    )
    hcf = stage2_run_dir / "hierarchical_cf"

    hotspots = pd.read_csv(hcf / "hotspot_level" / "hotspot_segments_detailed.csv", low_memory=False)
    seg = pd.read_csv(project_root / "input_data" / "segments_unique.csv", low_memory=False)
    hotspots = hotspots.merge(
        seg[
            [
                "Location ID",
                "Centreline rumble strips",
                "Delineation",
                "Street lighting",
                "Paved shoulder - driver-side",
                "Paved shoulder - passenger-side",
                "Road condition",
            ]
        ],
        on="Location ID",
        how="left",
    )
    hotspots["baseline_fsi"] = np.exp(hotspots["predicted_risk"]) - 0.001

    contrasts = [
        ("CRS", "centreline_rumble_strips__0_to_1_cate", "Centreline rumble strips", 1),
        ("Del", "delineation__0_to_1_cate", "Delineation", 2),
        ("SL", "street_lighting__0_to_1_cate", "Street lighting", 1),
        ("PSD_01", "paved_shoulder_driver-side__0_to_1_cate", "Paved shoulder - driver-side", 4),
        ("PSD_12", "paved_shoulder_driver-side__1_to_2_cate", "Paved shoulder - driver-side", 3),
        ("PSD_23", "paved_shoulder_driver-side__2_to_3_cate", "Paved shoulder - driver-side", 2),
        ("PSP_01", "paved_shoulder_passenger-side__0_to_1_cate", "Paved shoulder - passenger-side", 4),
        ("PSP_12", "paved_shoulder_passenger-side__1_to_2_cate", "Paved shoulder - passenger-side", 3),
        ("PSP_23", "paved_shoulder_passenger-side__2_to_3_cate", "Paved shoulder - passenger-side", 2),
        ("RC_01", "road_condition__0_to_1_cate", "Road condition", 3),
        ("RC_12", "road_condition__1_to_2_cate", "Road condition", 2),
    ]

    a_threshold, r_threshold = 0.002, 5.0
    print("=" * 70)
    print("PER-CONTRAST BREAKDOWN (A=0.002, R=5%)")
    print("=" * 70)
    total_upgrades = 0
    total_all = 0
    for label, cate_col, treatment_col, eligible_raw in contrasts:
        valid_mask = hotspots[cate_col].notna().values
        idx = np.where(valid_mask)[0]
        n_eligible = int((hotspots[treatment_col].values == eligible_raw).sum())
        if len(idx) == 0:
            print(f"  {label:10s}  elig={n_eligible:4d}  valid={0:4d}  upgrades={0:4d}  total={0:4d}")
            continue
        cate = hotspots[cate_col].values[idx]
        fsi = hotspots["baseline_fsi"].values[idx]
        pred = hotspots["predicted_risk"].values[idx]
        delta_fsi = np.exp(pred + cate) - 0.001 - fsi
        abs_red = -delta_fsi
        rel_red = 100.0 * abs_red / np.maximum(fsi, 1e-10)
        prescribed = (abs_red >= a_threshold) & (rel_red >= r_threshold) & (cate < 0)
        is_eligible = hotspots[treatment_col].values[idx] == eligible_raw
        n_upgrades = int((prescribed & is_eligible).sum())
        n_total = int(prescribed.sum())
        total_upgrades += n_upgrades
        total_all += n_total
        print(
            f"  {label:10s}  elig={n_eligible:4d}  valid={len(idx):4d}  "
            f"upgrades={n_upgrades:4d}  total={n_total:4d}"
        )
    print(f"  {'TOTAL':10s}              upgrades={total_upgrades:4d}  total={total_all:4d}")


if __name__ == "__main__":
    main()
