# stage1_hypothesis_testing.py
"""
Performs supplementary statistical hypothesis testing on the training data.

This module compares the findings of the ML model (via SHAP) with traditional
statistical tests to provide additional context. All tests are performed on the
training set to prevent data snooping.
"""
import pandas as pd
import numpy as np
from scipy import stats
import re # ADDED: Import the regular expression module

def _extract_feature_name(hypothesis_string):
    """
    Parses a raw hypothesis string to extract the clean feature name.
    It looks for text enclosed in double asterisks (**).
    
    Args:
        hypothesis_string (str): The raw hypothesis text.
        
    Returns:
        str or None: The cleaned feature name, or None if not found.
    """
    # WHY: This regex is robust. It finds content inside '**...**',
    # captures it, and handles potential single quotes inside.
    match = re.search(r"\*\*'(.+?)'\*\*", hypothesis_string)
    if match:
        return match.group(1) # Return the captured group (the feature name)
    
    # ADDED: Fallback for hypotheses that might not have single quotes
    match = re.search(r"\*\*(.+?)\*\*", hypothesis_string)
    if match:
        return match.group(1)

    return None

def perform_hypothesis_testing(X_train, y_train, shap_summary_df, hypotheses):
    """
    Tests pre-defined hypotheses using traditional statistical tests on the training data.
    This function is now robust and can handle raw hypothesis strings.

    Args:
        X_train (pd.DataFrame): The training feature matrix.
        y_train (pd.Series): The training target vector.
        shap_summary_df (pd.DataFrame): DataFrame of features ranked by SHAP importance.
        hypotheses (list): A list of raw hypothesis strings to test.

    Returns:
        pd.DataFrame: A tidy DataFrame summarizing the results of each hypothesis test.
    """
    if not hypotheses:
        print("   Skipping hypothesis testing: No hypotheses provided.")
        return pd.DataFrame()

    print("   Performing supplementary hypothesis testing on training data...")
    
    train_df = pd.concat([X_train, y_train], axis=1)
    target_col = y_train.name

    results = []
    top_shap_features = shap_summary_df['feature'].tolist()

    # CHANGED: The loop now robustly parses each hypothesis string
    for raw_hypothesis in hypotheses:
        # Use the new helper function to get the clean feature name
        feature = _extract_feature_name(raw_hypothesis)

        if feature is None:
            print(f"   Warning: Could not parse feature name from hypothesis: '{raw_hypothesis[:70]}...'. Skipping.")
            continue

        if feature not in train_df.columns:
            print(f"   Warning: Hypothesized feature '{feature}' not found in the final dataset. Skipping.")
            continue

        is_in_top_shap = feature in top_shap_features
        test_name = None
        p_value = np.nan
        
        if pd.api.types.is_categorical_dtype(train_df[feature]):
            test_name = "ANOVA F-test"
            categories = train_df[feature].dropna().unique()
            groups = [train_df[target_col][train_df[feature] == cat] for cat in categories]
            if len(groups) > 1:
                _, p_value = stats.f_oneway(*groups)
        
        elif pd.api.types.is_numeric_dtype(train_df[feature]):
            test_name = "Spearman Correlation"
            temp_df = train_df[[feature, target_col]].dropna()
            if len(temp_df) > 1:
                _, p_value = stats.spearmanr(temp_df[feature], temp_df[target_col])
        
        results.append({
            'Hypothesis (Feature)': feature,
            'In_Top_SHAP_Drivers': is_in_top_shap,
            'Statistical_Test': test_name,
            'P_Value': p_value
        })

    results_df = pd.DataFrame(results)
    print("   Hypothesis testing complete.")
    return results_df