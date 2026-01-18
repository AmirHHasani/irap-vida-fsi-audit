# stage1_countermeasure_comparison.py
import pandas as pd
import stage1_config as cfg
from stage1_utils import preferred_id_col

COUNTERMEASURE_ID_COL = getattr(cfg, 'COUNTERMEASURE_ID_COL', 'Location ID')
COUNTERMEASURE_DETAIL_COL = getattr(cfg, 'COUNTERMEASURE_DETAIL_COL', None)

def compare_with_countermeasures(high_risk_summary, countermeasure_df, id_col: str = None):
    """
    Retrieves and appends proposed countermeasures for each high-risk segment.

    This function takes the high-risk analysis table, finds all countermeasures
    for each Location ID, and adds them as a new column for manual validation.

    Args:
        high_risk_summary (pd.DataFrame): The DataFrame from analyze_high_risk_segments.
        countermeasure_df (pd.DataFrame): The DataFrame with all proposed countermeasures.

    Returns:
        pd.DataFrame: The high_risk_summary DataFrame with a new 'Proposed_Countermeasures' column.
    """
    # If countermeasure data isn't available, add a placeholder column and return
    if countermeasure_df is None or high_risk_summary is None:
        if high_risk_summary is not None:
            high_risk_summary['Proposed_Countermeasures'] = 'Data Not Available'
        return high_risk_summary

    # Determine which id column to use for matching (prefer canonical if omitted)
    id_col_use = id_col
    if id_col_use is None:
        try:
            id_col_use = preferred_id_col(high_risk_summary, prefer_canonical=True)
        except Exception:
            id_col_use = getattr(cfg, 'ID_COL', 'Location ID')

    # Determine id column in countermeasure_df (it may use a different configured name)
    cm_id_col = COUNTERMEASURE_ID_COL if COUNTERMEASURE_ID_COL in countermeasure_df.columns else None
    if cm_id_col is None:
        try:
            cm_id_col = preferred_id_col(countermeasure_df, prefer_canonical=True)
        except Exception:
            cm_id_col = COUNTERMEASURE_ID_COL

    # This helper function will be applied to each row of the high-risk table
    def get_countermeasures_for_segment(segment_id):
        # Filter the main countermeasure DataFrame for the specific segment ID
        try:
            proposed_cms = countermeasure_df[countermeasure_df[cm_id_col].astype(str) == str(segment_id)]
        except Exception:
            # Fallback to broad equality test on original configured column
            proposed_cms = countermeasure_df[countermeasure_df.get(COUNTERMEASURE_ID_COL, pd.Series(dtype=object)) == segment_id]

        if proposed_cms.empty:
            return "None Proposed"

        # Determine the best available detail/text column
        candidates = [COUNTERMEASURE_DETAIL_COL, getattr(cfg, 'COUNTERMEASURE_TEXT_COL', None), 'Countermeasure', 'countermeasure']
        chosen = None
        for c in candidates:
            if c and c in proposed_cms.columns:
                chosen = c
                break
        if chosen is None:
            # fallback to any column containing 'counter' (case-insensitive)
            for c in proposed_cms.columns:
                if 'counter' in c.lower():
                    chosen = c
                    break

        if chosen is None:
            # No useful detail column available
            return "; ".join(proposed_cms.astype(str).drop_duplicates().apply(lambda row: ' | '.join(row.values.astype(str)), axis=1).tolist()[:3])

        # Return a '; '-separated string of all unique countermeasure descriptions
        return "; ".join([str(x) for x in proposed_cms[chosen].dropna().unique()])

    # Create the new 'Proposed_Countermeasures' column by applying the function
    # Choose the best target column present in the high_risk_summary to look up segments
    if id_col_use in high_risk_summary.columns:
        target_col = id_col_use
    elif 'Location ID' in high_risk_summary.columns:
        target_col = 'Location ID'
    else:
        # Fallback to first column which is likely the identifier
        target_col = high_risk_summary.columns[0]

    high_risk_summary['Proposed_Countermeasures'] = high_risk_summary[target_col].apply(get_countermeasures_for_segment)
    
    print("   Countermeasure retrieval complete.")
    return high_risk_summary