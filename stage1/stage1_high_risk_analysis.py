# stage1_high_risk_analysis.py
import pandas as pd
import shap
from stage1_utils import preferred_id_col

def analyze_high_risk_segments(original_df, X_test, y_test, model, shap_summary_df, top_n=10, output_dir=None, id_col: str = None):
    """
    Identifies the highest-risk segments and creates a comprehensive analysis table
    that includes local drivers and their global importance rank.

    Args:
        original_df (pd.DataFrame): The original, unprocessed DataFrame to get metadata like Location ID.
        X_test (pd.DataFrame): The processed test set features.
        y_test (pd.Series): The test set target values.
        model: The trained model object (the best single model, used for local SHAP).
        shap_summary_df (pd.DataFrame): The global SHAP summary to get feature ranks.
        top_n (int): The number of top high-risk segments to analyze.

    Returns:
        pd.DataFrame: A comprehensive analysis table for the highest-risk segments.
    """
    if y_test is None:
        raise ValueError(
            "[ERROR] y_test is None in analyze_high_risk_segments. "
            "This usually means the model training or evaluation step failed. "
            f"Debug info: X_test shape: {None if X_test is None else X_test.shape}, model: {type(model)}, shap_summary_df empty: {shap_summary_df is None or shap_summary_df.empty}"
        )
    print("   Identifying top highest-risk segments and their drivers...")
    print("[DEBUG] X_test shape:", X_test.shape)
    print("[DEBUG] y_test shape:", y_test.shape)
    print("[DEBUG] X_test columns:", list(X_test.columns))
    print("[DEBUG] y_test head:\n", y_test.head())
    print("[DEBUG] X_test head:\n", X_test.head())
    
    # Combine test data to find the riskiest segments based on actual risk
    test_summary = X_test.copy()
    test_summary['Actual_FSI'] = y_test
    print("[DEBUG] After assigning y_test to test_summary['Actual_FSI']:")
    print(test_summary[['Actual_FSI']].head(10))
    test_summary['Actual_FSI'] = pd.to_numeric(test_summary['Actual_FSI'], errors='coerce')
    print("[DEBUG] After pd.to_numeric on Actual_FSI:")
    print(test_summary[['Actual_FSI']].head(10))
    print("[DEBUG] Number of non-NaN Actual_FSI:", test_summary['Actual_FSI'].notna().sum())
    highest_risk_segments_data = test_summary.nlargest(top_n, 'Actual_FSI')
    print("[DEBUG] highest_risk_segments_data indices:", highest_risk_segments_data.index.tolist())
    print("[DEBUG] highest_risk_segments_data['Actual_FSI']:")
    print(highest_risk_segments_data['Actual_FSI'])
    
    # --- Prepare for the detailed analysis ---
    # 1. Create a global rank mapping from the SHAP summary for easy lookup.
    #    The rank is the feature's position in the global importance list.
    global_rank_map = {row['feature']: i + 1 for i, row in shap_summary_df.iterrows()}
    
    # 2. Initialize a SHAP TreeExplainer for generating local explanations.
    #    This is done on the best single model for stability and clarity.
    explainer = shap.TreeExplainer(model)
    
    # 3. To be efficient, calculate local SHAP values ONLY for the top N segments we care about.
    local_shap_values = explainer(highest_risk_segments_data.drop(columns=['Actual_FSI']))
    
    # --- Loop through each high-risk segment to build the final table ---
    results = []
    # Determine ID column to use for mapping back to original_df
    id_col_use = id_col or preferred_id_col(original_df, prefer_canonical=True)

    for i, segment_index in enumerate(highest_risk_segments_data.index):
        # Get the original Location ID using the DataFrame index
        try:
            segment_id = original_df.loc[segment_index, id_col_use]
        except Exception:
            # Fallback to literal 'Location ID'
            segment_id = original_df.loc[segment_index, 'Location ID']
        actual_fsi = highest_risk_segments_data.loc[segment_index, 'Actual_FSI']
        
        # Get the local SHAP explanation for this specific segment
        shap_row = local_shap_values[i]
        
        # Create a DataFrame of this segment's drivers and sort by impact
        local_drivers_df = pd.DataFrame({
            'feature': shap_row.feature_names,
            'shap_value': shap_row.values
        }).sort_values(by='shap_value', ascending=False).head(3) # Get top 3 drivers
        
        # Format the driver string to include the global rank for context
        driver_strings = []
        for _, driver_row in local_drivers_df.iterrows():
            feature = driver_row['feature']
            shap_val = driver_row['shap_value']
            # Look up the global rank, default to 'N/A' if it's not a top global feature
            rank = global_rank_map.get(feature, 'N/A')
            driver_strings.append(f"{feature} (+{shap_val:.3f}, Global #{rank})")
            
        results.append({
            'Location ID': segment_id,
            'Actual_FSI': actual_fsi,
            'Top_Local_Drivers_With_Global_Rank': "; ".join(driver_strings)
        })
        
    print("   High-risk segment analysis complete.")
    return pd.DataFrame(results)