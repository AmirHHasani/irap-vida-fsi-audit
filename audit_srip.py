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
    parser = argparse.ArgumentParser(description="Ad hoc SRIP agreement audit for Stage 2.")
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

    srip = pd.read_csv(hcf / "diagnostics" / "srip_agreement_segment_level.csv", low_memory=False)
    print(f"SRIP rows: {len(srip)}, unique segments: {srip['segment_id'].nunique()}")
    print(f"SRIP treatments: {sorted(srip['treatment'].unique())}")

    treatment_to_contrasts = {
        "Centreline rumble strips": [("centreline_rumble_strips__0_to_1_cate", "Centreline rumble strips", 1)],
        "Delineation": [("delineation__0_to_1_cate", "Delineation", 2)],
        "Street lighting": [("street_lighting__0_to_1_cate", "Street lighting", 1)],
        "Paved shoulder - driver-side": [
            ("paved_shoulder_driver-side__0_to_1_cate", "Paved shoulder - driver-side", 4),
            ("paved_shoulder_driver-side__1_to_2_cate", "Paved shoulder - driver-side", 3),
            ("paved_shoulder_driver-side__2_to_3_cate", "Paved shoulder - driver-side", 2),
        ],
        "Paved shoulder - passenger-side": [
            ("paved_shoulder_passenger-side__0_to_1_cate", "Paved shoulder - passenger-side", 4),
            ("paved_shoulder_passenger-side__1_to_2_cate", "Paved shoulder - passenger-side", 3),
            ("paved_shoulder_passenger-side__2_to_3_cate", "Paved shoulder - passenger-side", 2),
        ],
        "Road condition": [
            ("road_condition__0_to_1_cate", "Road condition", 3),
            ("road_condition__1_to_2_cate", "Road condition", 2),
        ],
    }

    a_threshold, r_threshold = 0.002, 5.0
    hotspot_by_seg = hotspots.set_index("segment_id")

    def has_thresholded_prescription(seg_id, treatment_name):
        if seg_id not in hotspot_by_seg.index:
            return False
        row = hotspot_by_seg.loc[seg_id]
        for cate_col, treatment_col, eligible_raw in treatment_to_contrasts[treatment_name]:
            cate = row[cate_col]
            if pd.isna(cate):
                continue
            current_level = row[treatment_col]
            if current_level != eligible_raw:
                continue
            fsi = row["baseline_fsi"]
            pred = row["predicted_risk"]
            delta_fsi = np.exp(pred + cate) - 0.001 - fsi
            abs_red = -delta_fsi
            rel_red = 100.0 * abs_red / max(fsi, 1e-10)
            if abs_red >= a_threshold and rel_red >= r_threshold and cate < 0:
                return True
        return False

    srip["cf_thresholded"] = srip.apply(
        lambda row: has_thresholded_prescription(row["segment_id"], row["treatment"]),
        axis=1,
    )

    print()
    print("=" * 70)
    print("SRIP AGREEMENT - THRESHOLDED (A=0.002, R=5%) vs CATE<0")
    print("=" * 70)

    srip_rec = srip["srip_recommends"].values.astype(bool)
    cf_orig = srip["cf_prescribes"].values.astype(bool)
    cf_thresh = srip["cf_thresholded"].values.astype(bool)

    for criterion_name, cf_vals in [("CATE<0", cf_orig), ("Thresholded", cf_thresh)]:
        tp = int((srip_rec & cf_vals).sum())
        fp = int((~srip_rec & cf_vals).sum())
        fn = int((srip_rec & ~cf_vals).sum())
        tn = int((~srip_rec & ~cf_vals).sum())
        total = tp + fp + fn + tn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        micro = (tp + tn) / total if total > 0 else 0
        p_o = (tp + tn) / total
        p_yes = ((tp + fp) / total) * ((tp + fn) / total)
        p_no = ((fn + tn) / total) * ((fp + tn) / total)
        p_e = p_yes + p_no
        kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0
        print(f"\n  === {criterion_name} ===")
        print(f"  Overlap hotspots: {srip['segment_id'].nunique()}")
        print(f"  Total pairs: {total}")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"  Precision = {precision:.3f}")
        print(f"  Micro accuracy = {micro:.3f}")
        print(f"  Cohen kappa = {kappa:.3f}")


if __name__ == "__main__":
    main()
