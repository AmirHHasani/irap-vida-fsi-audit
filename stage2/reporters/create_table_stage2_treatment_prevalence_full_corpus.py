"""Table: Full-corpus treatment prevalence (Stage 2 treatments).

Computes prevalence for the 6 validated Stage-2 treatments on the full eligible
corpus (expected N=147,466 segments).

IMPORTANT: Uses *raw coded levels* from input_data/segments_unique.csv.
Do NOT use canonical 0-based encodings here, because raw codes can be 1-based,
non-consecutive, and/or reversed (e.g., 1=best, 4=none).

Data source: input_data/segments_unique.csv
Output: stage2_outputs/.../reports/table_stage2_treatment_prevalence_full_corpus.(csv|tex)

Notes:
- Percentages are computed over non-missing observations per treatment.
- “Level” is the raw code as stored in segments_unique.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import stage2_config as config


EXPECTED_N = 147_466


CODEBOOK_CSV = Path(__file__).with_name("codes_filled_analysis.csv")

# Alias treatment names to codebook attribute names when they differ.
CODEBOOK_NAME_ALIASES: dict[str, str] = {
    # In codes_filled_analysis.csv the attribute is spelled with a space.
    "Centreline rumble strips": "Centre line rumble strips",
}


def _choose_best_label(candidates: list[str]) -> str:
    """Resolve duplicates in codes_filled_analysis.csv for the same (attribute, code)."""

    cleaned = [str(c).strip() for c in candidates if c is not None]
    cleaned = [c for c in cleaned if c != ""]
    if not cleaned:
        return "(blank)"

    # Prefer more specific labels over generic placeholders like 'Present'.
    def score(label: str) -> tuple[int, int, int]:
        lower = label.lower()
        is_generic = 1 if lower in {"present", "not present", "yes", "no"} else 0
        has_units = 1 if ("m" in lower or "km" in lower or "≥" in label or "<" in label) else 0
        return (has_units, -is_generic, len(label))

    return sorted(cleaned, key=score, reverse=True)[0]


def load_codebook_label_map(codebook_csv: Path) -> dict[str, dict[int, str]]:
    """Load Attribute Name + Code -> meaning from codes_filled_analysis.csv."""

    if not codebook_csv.exists():
        raise FileNotFoundError(
            f"codes_filled_analysis.csv not found at: {codebook_csv}. "
            "Cannot build level meanings without the codebook."
        )

    df = pd.read_csv(codebook_csv)
    required = {"Attribute Name", "Attribute Class Name", "Code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"codes_filled_analysis.csv missing columns: {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )

    wanted = set(config.ALL_TREATMENTS)
    wanted.update(CODEBOOK_NAME_ALIASES.values())

    df = df[df["Attribute Name"].astype(str).isin(wanted)].copy()
    if df.empty:
        return {}

    df["Code"] = pd.to_numeric(df["Code"], errors="coerce")
    df = df.dropna(subset=["Code"])
    df["Code"] = df["Code"].astype(int)

    # Collect candidates per (attribute, code)
    candidates: dict[str, dict[int, list[str]]] = {}
    for _, row in df.iterrows():
        attr = str(row["Attribute Name"]).strip()
        code = int(row["Code"])
        meaning = str(row.get("Attribute Class Name", "")).strip()
        candidates.setdefault(attr, {}).setdefault(code, []).append(meaning)

    out: dict[str, dict[int, str]] = {}
    for attr, code_map in candidates.items():
        out[attr] = {code: _choose_best_label(vals) for code, vals in code_map.items()}

    return out


def _format_pct(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{value:.1f}%"


def build_table(segments_unique_csv: Path) -> pd.DataFrame:
    if not segments_unique_csv.exists():
        raise FileNotFoundError(
            f"segments_unique.csv not found at: {segments_unique_csv}\n"
            "Expected raw segment coding at input_data/segments_unique.csv"
        )

    usecols = ["Location ID"] + list(config.ALL_TREATMENTS)
    df = pd.read_csv(segments_unique_csv, usecols=usecols, low_memory=False)

    n_rows = len(df)
    if n_rows != EXPECTED_N:
        print(
            f"Warning: expected {EXPECTED_N:,} rows but found {n_rows:,} in {segments_unique_csv}."
        )

    rows: list[dict[str, object]] = []

    codebook = load_codebook_label_map(CODEBOOK_CSV)
    for treatment in config.ALL_TREATMENTS:
        if treatment not in df.columns:
            raise ValueError(f"Missing treatment column in analysis dataset: {treatment}")

        values = pd.to_numeric(df[treatment], errors="coerce")
        non_missing = values.dropna().astype(int)
        n_non_missing = int(non_missing.shape[0])
        if n_non_missing == 0:
            rows.append(
                {
                    "Treatment": treatment,
                    "N_non_missing": 0,
                    "Level": np.nan,
                    "Count": 0,
                    "Percent_of_non_missing": "",
                }
            )
            continue

        # Prefer a stable, codebook-aligned ordering of levels when available.
        codebook_name = CODEBOOK_NAME_ALIASES.get(treatment, treatment)
        label_map = codebook.get(codebook_name, {})
        if label_map:
            ordered_levels = sorted(label_map.keys())
        else:
            ordered_levels = sorted(non_missing.unique().tolist())

        counts = non_missing.value_counts().to_dict()
        for level in ordered_levels:
            count = int(counts.get(level, 0))
            pct = 100.0 * float(count) / float(n_non_missing) if n_non_missing else float("nan")
            rows.append(
                {
                    "Treatment": treatment,
                    "N_non_missing": n_non_missing,
                    "Level": int(level),
                    "Level_meaning": label_map.get(int(level), ""),
                    "Count": count,
                    "Percent_of_non_missing": _format_pct(pct),
                }
            )

    out = pd.DataFrame(rows)

    # Nice ordering: match paper ordering for treatments, then ascending levels.
    out["Treatment"] = pd.Categorical(out["Treatment"], categories=config.ALL_TREATMENTS, ordered=True)
    out = out.sort_values(["Treatment", "Level"], ascending=[True, True]).reset_index(drop=True)
    out["Treatment"] = out["Treatment"].astype(str)

    return out


def main() -> None:
    segments_csv = config.SEGMENTS_DATA_CSV
    reports_dir = config.OUTPUT_SUBDIRS["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    table = build_table(segments_csv)

    out_csv = reports_dir / "table_stage2_treatment_prevalence_full_corpus.csv"
    out_tex = reports_dir / "table_stage2_treatment_prevalence_full_corpus.tex"

    table.to_csv(out_csv, index=False)
    try:
        table.to_latex(out_tex, index=False)
    except Exception as e:
        print(f"Warning: failed to write LaTeX table ({e}). CSV was written.")

    print(f"Wrote: {out_csv}")
    if out_tex.exists():
        print(f"Wrote: {out_tex}")


if __name__ == "__main__":
    main()
