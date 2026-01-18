# stage1_feature_engineering.py
"""
Functions for feature selection, preparation, and preprocessing.
"""
import pandas as pd
import re
from typing import Tuple, List, Optional

# sklearn preprocessing utilities (used by unified preprocessor)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import joblib

def prepare_features(df, target_col, metadata_cols, feature_exclusions):
    """
    Separates the DataFrame into features (X), target (y), and metadata.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target variable column.
        metadata_cols (list): A list of metadata columns to exclude from features.
        feature_exclusions (list): A list of other columns to exclude.

    Returns:
        pd.DataFrame: The feature matrix (X).
        pd.Series: The target vector (y).
        pd.DataFrame: A DataFrame containing only the essential metadata columns.
    """
    print("   Separating features (X), target (y), and metadata.")
    
    # --- NEW: Define essential metadata to preserve separately ---
    # These are columns needed for CV grouping, logging, mapping, and dataset-level analysis, but are not features.
    from stage1_config import ID_COL, ROAD_COLUMN_NAME, DATASET_ID_COL, INCLUDE_DATASET_ID_AS_FEATURE
    essential_metadata_cols = [ID_COL, ROAD_COLUMN_NAME, 'Latitude', 'Longitude']
    # Conditionally add Dataset ID to essential metadata (excluded from features) or allow as feature
    if not INCLUDE_DATASET_ID_AS_FEATURE:
        essential_metadata_cols.append(DATASET_ID_COL)
    # Ensure we only try to select columns that actually exist in the dataframe
    existing_essential_cols = [col for col in essential_metadata_cols if col in df.columns]
    # Always preserve Dataset ID in metadata_df regardless of feature inclusion
    metadata_preservation_cols = [ID_COL, ROAD_COLUMN_NAME, DATASET_ID_COL, 'Latitude', 'Longitude']
    metadata_df = df[[col for col in metadata_preservation_cols if col in df.columns]].copy()
    
    # Combine all columns to be dropped from the feature set
    cols_to_drop = set(metadata_cols) | set(feature_exclusions) | {target_col} | set(essential_metadata_cols)

    # Use .get() for safe access to the target column
    if df.get(target_col) is None:
        raise ValueError(f"Target column '{target_col}' not found in the DataFrame.")

    y = df[target_col]

    # Robust exclusion: also drop columns matching any exclusion pattern (substring match, case-insensitive, strip spaces)
    exclusion_patterns = [
        'star rating', 'srs', 'fatality estimation', 'fsi', 'policy target', 'smoothed'
    ]
    cols_to_drop_pattern = set()
    for col in df.columns:
        col_clean = col.lower().replace('_', ' ').strip()
        for pat in exclusion_patterns:
            if pat in col_clean:
                cols_to_drop_pattern.add(col)
                break

    all_cols_to_drop = set([col for col in cols_to_drop if col in df.columns]) | cols_to_drop_pattern
    # Only keep features that are in CATEGORICAL_FEATURES or NUMERICAL_FEATURES
    from stage1_config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
    feature_cols = [col for col in df.columns if col not in all_cols_to_drop and (col in CATEGORICAL_FEATURES or col in NUMERICAL_FEATURES)]
    X = df[feature_cols].copy()

    # --- NEW: Sanitize feature names to prevent encoding errors with SHAP ---
    def sanitize_col_name(name):
        # Convert to lowercase
        name = name.lower()
        # Replace special characters, parentheses, and spaces with a single underscore
        name = re.sub(r'[\s\(\)\-\/]+', '_', name)
        # Remove any other non-alphanumeric characters (except underscore)
        name = re.sub(r'[^a-z0-9_]+', '', name)
        # Collapse multiple underscores into one
        name = re.sub(r'[_]+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        # Ensure the name is ASCII
        return name.encode('ascii', 'ignore').decode('ascii')

    sanitized_columns = {col: sanitize_col_name(col) for col in X.columns}
    X.rename(columns=sanitized_columns, inplace=True)
    print("   [INFO] All feature names have been sanitized for model compatibility.")
    # --- End of sanitization ---

    print(f"   Feature set created with {X.shape[1]} columns. Dropped {len(all_cols_to_drop)} columns (pattern+exact).")
    print("   Included columns (features):")
    feature_list = list(X.columns)
    print(feature_list)
    # Save the feature list and a simple audit to files for reporting
    import os
    output_dir = 'stage1_outputs'
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    with open(os.path.join(output_dir, 'used_features.txt'), 'w', encoding='utf-8') as f:
        for col in feature_list:
            f.write(f"{col}\n")
    try:
        # Build a quick audit: why columns were dropped (metadata/exclusion/pattern)
        audit_rows = []
        cols_in_df = set(df.columns)
        exact_drop = set([c for c in (set(metadata_cols) | set(feature_exclusions) | {target_col}) if c in cols_in_df])
        pattern_drop = set([c for c in cols_in_df if c not in feature_list and c not in exact_drop and c not in essential_metadata_cols])
        for c in sorted(exact_drop):
            reason = 'metadata' if c in set(metadata_cols) else ('feature_exclusion' if c in set(feature_exclusions) else ('target' if c==target_col else 'exact'))
            audit_rows.append({'column': c, 'kept': False, 'reason': reason})
        for c in sorted(pattern_drop):
            # Only tag as pattern drop if it matched one of the patterns
            cl = c.lower().replace('_', ' ').strip()
            matched = any(pat in cl for pat in ['star rating','srs','fatality estimation','fsi','policy target','smoothed'])
            if matched:
                audit_rows.append({'column': c, 'kept': False, 'reason': 'pattern_exclusion'})
        for c in feature_list:
            audit_rows.append({'column': c, 'kept': True, 'reason': 'selected_feature'})
        import pandas as _pd
        _pd.DataFrame(audit_rows).to_csv(os.path.join(output_dir, 'feature_audit.csv'), index=False)
        print(f"   [INFO] Feature audit saved to {os.path.join(output_dir, 'feature_audit.csv')}")
    except Exception as _e_audit:
        print(f"   [WARN] Feature audit skipped: {_e_audit}")
    return X, y, metadata_df

def preprocess_data_split(df, numerical_features, categorical_features):
    """
    Applies preprocessing (imputation, type conversion) to a data split (train or test).

    Args:
        df (pd.DataFrame): The DataFrame split to process.
        numerical_features (list): List of numerical feature names.
        categorical_features (list): List of categorical feature names.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    df_processed = df.copy()

    # Process categorical features
    for col in categorical_features:
        if col in df_processed.columns:
            # Convert to category dtype
            df_processed[col] = df_processed[col].astype('category')
            # Add a 'Missing' category and fill NaNs
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].cat.add_categories(['Missing'])
                df_processed[col] = df_processed[col].fillna('Missing')

    # Process numerical features
    for col in numerical_features:
        if col in df_processed.columns:
            if df_processed[col].isnull().any():
                # Impute missing values with the median
                median_val = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_val)
    
    return df_processed


def build_preprocessor(numerical_features: List[str],
                       categorical_features: List[str],
                       for_tree_models: bool = True,
                       scale_numeric: bool = False):
    """Build an sklearn ColumnTransformer for the provided feature lists.

    This factory returns an unfitted ColumnTransformer. Callers should fit
    the transformer on training data only and reuse it for test/inference.
    """
    # numerical pipeline
    num_steps = [('impute', SimpleImputer(strategy='median'))]
    if scale_numeric:
        num_steps.append(('scale', StandardScaler()))
    num_pipeline = Pipeline(num_steps)

    # categorical pipeline
    # For tree models we use OrdinalEncoder with an explicit unknown value so trees get numeric codes.
    if for_tree_models:
        cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    else:
        cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    cat_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
        ('encode', cat_encoder)
    ])

    transformers = []
    if numerical_features:
        transformers.append(('num', num_pipeline, numerical_features))
    if categorical_features:
        transformers.append(('cat', cat_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', verbose_feature_names_out=False)
    return preprocessor


def fit_transform_preprocessor(X_train: pd.DataFrame,
                               X_test: pd.DataFrame,
                               numerical_features: List[str],
                               categorical_features: List[str],
                               for_tree_models: bool = True,
                               scale_numeric: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer, List[str]]:
    """Build, fit on X_train, and transform both X_train and X_test.

    Returns transformed DataFrames with preserved indices and a stable feature name list.
    """
    # Utility: sanitize a configured feature list to match sanitized column names in X_train
    def _sanitize_config_list(config_list, available_columns):
        # Use the same sanitation logic as prepare_features to normalize names
        def _sanitize(name):
            name = str(name).lower()
            name = re.sub(r'[\s\(\)\-\/]+', '_', name)
            name = re.sub(r'[^a-z0-9_]+', '', name)
            name = re.sub(r'[_]+', '_', name)
            return name.strip('_')

        sanitized = [_sanitize(n) for n in config_list]
        # keep only those that exist in the available columns
        return [s for s in sanitized if s in available_columns]

    # Filter feature lists to those present in the (sanitized) data columns
    num_feats = _sanitize_config_list(numerical_features, list(X_train.columns))
    cat_feats = _sanitize_config_list(categorical_features, list(X_train.columns))

    preprocessor = build_preprocessor(num_feats, cat_feats, for_tree_models=for_tree_models, scale_numeric=scale_numeric)
    # Make a safe copy and coerce categorical columns to object dtype so
    # SimpleImputer(fill_value='__MISSING__') can safely insert the string
    # sentinel. This prevents errors when columns currently have an integer
    # dtype but are intended to be treated as categorical.
    X_train = X_train.copy()
    for col in cat_feats:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('object')

    preprocessor.fit(X_train)

    # Transform. Ensure X_test has compatible dtypes for categorical columns
    # (coerce to object) before transforming to avoid casting issues.
    X_train_t = preprocessor.transform(X_train)
    # Handle X_test possibly being None (caller may only want a fitted preprocessor)
    if X_test is None:
        # Use X_train as fallback to produce a valid transformed frame (caller should be aware)
        X_test = X_train.copy()
    else:
        X_test = X_test.copy()
        for col in cat_feats:
            if col in X_test.columns:
                X_test[col] = X_test[col].astype('object')

    X_test_t = preprocessor.transform(X_test)

    # Retrieve feature names in a robust way
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback: build names from inputs
        feature_names = []
        if num_feats:
            feature_names.extend(num_feats)
        if cat_feats:
            # Best-effort: if encoder has get_feature_names_out, use it
            try:
                enc = preprocessor.named_transformers_['cat'].named_steps['encode']
                cat_names = list(enc.get_feature_names_out(cat_feats))
                feature_names.extend(cat_names)
            except Exception:
                # Default to categorical column names
                feature_names.extend(cat_feats)

    # Wrap transformed arrays into DataFrames and preserve indices
    X_train_proc = pd.DataFrame(X_train_t, index=X_train.index, columns=feature_names)
    X_test_proc = pd.DataFrame(X_test_t, index=X_test.index, columns=feature_names)

    return X_train_proc, X_test_proc, preprocessor, feature_names

def preprocess_for_cv_fold(X_train, X_test, numerical_features, categorical_features):
    """
    Preprocesses training and testing splits for a CV fold robustly.

    This function learns all encodings and imputations from the training set ONLY
    and applies the same transformations to the test set to prevent data leakage
    and handle unseen categories gracefully.

    Returns:
        (pd.DataFrame, pd.DataFrame): Processed X_train and X_test.
    """
    X_train_proc = X_train.copy()
    X_test_proc = X_test.copy()

    # --- 1. Handle Numerical Features (Impute based on train set) ---
    for col in numerical_features:
        if col in X_train_proc.columns:
            median_val = X_train_proc[col].median()
            X_train_proc[col] = X_train_proc[col].fillna(median_val)
            X_test_proc[col] = X_test_proc[col].fillna(median_val)

    # --- 2. Handle Categorical Features (Learn from train, apply to both) ---
    for col in categorical_features:
        if col in X_train_proc.columns:
            # Use 'object' for consistent NaN handling before converting
            X_train_proc[col] = X_train_proc[col].astype('object')
            X_test_proc[col] = X_test_proc[col].astype('object')

            # Learn categories and mode from the training set
            train_mode = X_train_proc[col].mode()[0]
            all_categories = list(X_train_proc[col].dropna().unique())

            # Create a consistent categorical type based on training data
            cat_type = pd.CategoricalDtype(categories=all_categories, ordered=False)
            X_train_proc[col] = X_train_proc[col].astype(cat_type)
            X_test_proc[col] = X_test_proc[col].astype(cat_type)

            # Impute NaNs using the training set's mode (now always on category dtype, no warning)
            X_train_proc[col] = X_train_proc[col].fillna(train_mode)
            X_test_proc[col] = X_test_proc[col].fillna(train_mode)

    return X_train_proc, X_test_proc