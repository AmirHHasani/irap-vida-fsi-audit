def generate_summary_maps(master_results_df, output_dir, top_n_hotspots=1000):
    """
    Generate and save two maps:
    1. Comprehensive FSI map (all segments, colored by actual FSI, highest FSI = most colored)
    2. Top N dangerous segments per road (by FSI)
    """
    import plotly.express as px
    import plotly.io as pio
    import os
    import numpy as np
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import stage1_config as cfg

    def transform_risk_to_size(risk_values, min_size=6, max_size=20):
        """Transform risk values to valid marker sizes, handling negative values from log transformation"""
        risk_values = np.array(risk_values)
        
        # Handle negative values from log transformation
        risk_min, risk_max = risk_values.min(), risk_values.max()
        if risk_min < 0:
            # Shift to positive range
            risk_shifted = risk_values - risk_min + 0.001
        else:
            risk_shifted = risk_values.copy()
        
        # Avoid division by zero
        if risk_shifted.max() == risk_shifted.min():
            return np.full_like(risk_shifted, (min_size + max_size) / 2)
        
        # Min-max scaling to size range
        risk_normalized = (risk_shifted - risk_shifted.min()) / (risk_shifted.max() - risk_shifted.min())
        sizes = min_size + risk_normalized * (max_size - min_size)
        
        # Validation - ensure all sizes are positive and within range
        sizes = np.clip(sizes, min_size, max_size)
        return sizes

    # --- Rename columns for clarity ---
    df = master_results_df.copy()
    if 'actual_risk' in df.columns:
        df = df.rename(columns={'actual_risk': 'actual_fsi'})
    if 'predicted_risk' in df.columns:
        df = df.rename(columns={'predicted_risk': 'predicted_fsi'})

    # --- Map 1: Comprehensive FSI Map (No Mapbox, Classy Geo) ---
    fsi_min = df['actual_fsi'].min()
    fsi_max = df['actual_fsi'].max()
    marker_sizes = transform_risk_to_size(df['actual_fsi'], min_size=6, max_size=14)
    fig1 = px.scatter_geo(
        df,
        lat="latitude", lon="longitude",
        color="actual_fsi",
        size=marker_sizes,
        hover_name="segment_id",
        hover_data={"road_id":True, "actual_fsi":True, "predicted_fsi":True, "fold_number":True},
        color_continuous_scale="YlOrRd",
        range_color=[fsi_min, fsi_max],
        size_max=12,
        title="Comprehensive FSI Map (All Segments)"
    )
    fig1.update_layout(
        geo=dict(
            showland=True, landcolor="white",
            showcountries=True, countrycolor="lightgray",
            showlakes=True, lakecolor="lightblue",
            showframe=False, showcoastlines=False,
            projection_type="equirectangular"
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=14),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig1.write_html(str(output_dir / "comprehensive_fsi_map.html"))
    print(f"[INFO] Comprehensive FSI map saved: {output_dir / 'comprehensive_fsi_map.html'}")
    if getattr(cfg, 'SAVE_MAPS_AS_IMAGES', False):
        try:
            fig1.write_image(str(output_dir / "comprehensive_fsi_map.png"), width=1200, height=800, scale=2)
            print(f"[INFO] Comprehensive FSI map image saved: {output_dir / 'comprehensive_fsi_map.png'}")
        except Exception as e:
            print(f"[WARNING] Failed to save PNG map. Exception: {e}")

    # --- Map 2: Top N Dangerous Segments Per Road (No Mapbox, Classy Geo) ---
    top_n = top_n_hotspots
    top_per_road = df.sort_values(['road_id', 'actual_fsi'], ascending=[True, False]) \
        .groupby('road_id').head(top_n)
    top_marker_sizes = transform_risk_to_size(top_per_road['actual_fsi'], min_size=6, max_size=14)
    fig2 = px.scatter_geo(
        top_per_road,
        lat="latitude", lon="longitude",
        color="actual_fsi",
        size=top_marker_sizes,
        hover_name="segment_id",
        hover_data={"road_id":True, "actual_fsi":True, "predicted_fsi":True, "fold_number":True},
        color_continuous_scale="YlOrRd",
        range_color=[fsi_min, fsi_max],
        size_max=12,
        title=f"Top {top_n} Dangerous Segments Per Road (by FSI)"
    )
    fig2.update_layout(
        geo=dict(
            showland=True, landcolor="white",
            showcountries=True, countrycolor="lightgray",
            showlakes=True, lakecolor="lightblue",
            showframe=False, showcoastlines=False,
            projection_type="equirectangular"
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=14),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig2.write_html(str(output_dir / f"top_{top_n}_per_road_fsi_map.html"))
    print(f"[INFO] Per-road FSI map saved: {output_dir / f'top_{top_n}_per_road_fsi_map.html'}")
    if getattr(cfg, 'SAVE_MAPS_AS_IMAGES', False):
        try:
            fig2.write_image(str(output_dir / f"top_{top_n}_per_road_fsi_map.png"), width=1200, height=800, scale=2)
            print(f"[INFO] Per-road FSI map image saved: {output_dir / f'top_{top_n}_per_road_fsi_map.png'}")
        except Exception as e:
            print(f"[WARNING] Failed to save PNG map. Exception: {e}")
# stage1_visualizations.py
"""
Generates supplementary diagnostic and comparison plots for model evaluation.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import the utility function for saving plots from the config file
from stage1_config import save_plot

def plot_model_comparison(model_results, output_dir, split_strategy=None):
    """
    Generates and saves bar plots comparing all trained models by R², MAE, and RMSE.

    Args:
        model_results (list): The list of dictionaries containing results for each model.
        output_dir (Path): The directory to save the plots.
        split_strategy (str, optional): The split strategy used for the models. Default is None.
    """
    print("   Generating model performance comparison plots...")
    # Convert the results into a pandas DataFrame for easy plotting
    import numpy as _np

    # Determine which metrics keys are available and prefer linear MAE if present
    rows = []
    for r in model_results:
        metrics = r.get('metrics', {})
        # R2 preference: prefer aggregated mean for CV, else single-value
        r2 = metrics.get('Test R2 Mean', metrics.get('Test R2', metrics.get('R2', _np.nan)))

        # MAE preference order: linear aggregated, plain MAE, log-MAE
        mae = _np.nan
        for key in ['Test MAE (linear)', 'Test MAE Mean', 'Test MAE', 'Test MAE (log)']:
            if key in metrics and metrics.get(key) is not None:
                try:
                    mae = float(metrics.get(key))
                    break
                except Exception:
                    continue

        # RMSE preference order: linear aggregated, plain RMSE, log-RMSE
        rmse = _np.nan
        for key in ['Test RMSE (linear)', 'Test RMSE Mean', 'Test RMSE', 'Test RMSE (log)']:
            if key in metrics and metrics.get(key) is not None:
                try:
                    rmse = float(metrics.get(key))
                    break
                except Exception:
                    continue

        rows.append({'Model': r.get('name', 'unknown'), 'R² Score': r2, 'MAE': mae, 'RMSE': rmse})
    perf_df = pd.DataFrame(rows)

    # --- R² Score Plot ---
    plt.figure(figsize=(10, 6))
    # FIXED: Assign y variable to hue to avoid deprecation warning
    sns.barplot(x='R² Score', y='Model', hue='Model', data=perf_df.sort_values('R² Score', ascending=False), 
                palette='viridis', legend=False)
    plt.title('Model Comparison: Test R² Score (Higher is Better)')
    plt.xlabel('Test R² Score')
    plt.ylabel('Model')
    save_plot(plt.gcf(), 'model_comparison_test_r2.png', directory=output_dir)
    plt.close()

    # --- MAE Plot ---
    plt.figure(figsize=(10, 6))
    # If MAE values are not available (all NaN) produce a clear placeholder image
    if perf_df['MAE'].isna().all():
        fig = plt.gcf()
        fig.set_size_inches(10, 3)
        plt.text(0.5, 0.5, 'No MAE values available for the provided results', ha='center', va='center')
        plt.axis('off')
        save_plot(fig, 'model_comparison_test_mae.png', directory=output_dir)
        plt.close()
    else:
        # FIXED: Assign y variable to hue to avoid deprecation warning
        sns.barplot(x='MAE', y='Model', hue='Model', data=perf_df.sort_values('MAE', ascending=True), 
                    palette='plasma', legend=False)
        plt.title('Model Comparison: Test MAE (Lower is Better)')
        plt.xlabel('Test Mean Absolute Error (MAE)')
        plt.ylabel('Model')
        save_plot(plt.gcf(), 'model_comparison_test_mae.png', directory=output_dir)
        plt.close()

    # --- RMSE Plot ---
    plt.figure(figsize=(10, 6))
    # If RMSE values are not available (all NaN) produce a clear placeholder image
    if perf_df['RMSE'].isna().all():
        fig = plt.gcf()
        fig.set_size_inches(10, 3)
        plt.text(0.5, 0.5, 'No RMSE values available for the provided results', ha='center', va='center')
        plt.axis('off')
        save_plot(fig, 'model_comparison_test_rmse.png', directory=output_dir)
        plt.close()
    else:
        # FIXED: Assign y variable to hue to avoid deprecation warning
        sns.barplot(x='RMSE', y='Model', hue='Model', data=perf_df.sort_values('RMSE', ascending=True), 
                    palette='coolwarm', legend=False)
        plt.title('Model Comparison: Test RMSE (Lower is Better)')
        plt.xlabel('Test Root Mean Squared Error (RMSE)')
        plt.ylabel('Model')
        save_plot(plt.gcf(), 'model_comparison_test_rmse.png', directory=output_dir)
        plt.close()
    
    print("   Comparison plots saved.")



def plot_oof_residual_analysis(master_pred_df, output_dir):
    """
    Generate residual diagnostics using out-of-fold (OOF) predictions collected during group CV.

    This provides a global residuals-vs-predicted and residual-distribution view when there
    is no single holdout split (BY_ROAD strategy).
    """
    print('   Generating OOF residual analysis from OOF predictions...')
    from pathlib import Path as _Path
    import numpy as _np
    import scipy.stats as _stats

    out_dir = _Path(output_dir)
    if master_pred_df is None or master_pred_df.empty:
        print('   [INFO] master_pred_df missing or empty; skipping OOF residual analysis.')
        return

    df = master_pred_df.copy()
    # Ensure required columns present
    if not {'predicted_risk', 'actual_risk'}.issubset(df.columns):
        print('   [WARN] master_pred_df missing required columns for residual analysis.')
        return

    # Back-transform to original linear scale when log1p was applied
    try:
        actual_lin = _np.expm1(df['actual_risk'].astype(float).values)
        pred_lin = _np.expm1(df['predicted_risk'].astype(float).values)
    except Exception:
        # Fall back to using provided values if they are already linear
        actual_lin = df['actual_risk'].astype(float).values
        pred_lin = df['predicted_risk'].astype(float).values

    residuals = actual_lin - pred_lin

    import matplotlib.pyplot as _plt
    import seaborn as _sns

    # Plot 1: Residuals vs Predicted (linear)
    fig1, ax1 = _plt.subplots(1, 1, figsize=(8, 6))
    _sns.scatterplot(x=pred_lin, y=residuals, alpha=0.4, ax=ax1)
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_title('OOF Residuals vs Predicted (linear scale)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted (linear)', fontsize=12)
    ax1.set_ylabel('Residual (actual - predicted)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    _plt.tight_layout()
    save_plot(fig1, 'oof_residuals_vs_predicted.png', directory=out_dir)
    _plt.close(fig1)

    # Plot 2: Residual distribution (linear)
    fig2, ax2 = _plt.subplots(1, 1, figsize=(8, 6))
    
    # Remove extreme outliers for better visualization (keep 99% of data)
    residuals_clean = residuals[~_np.isnan(residuals)]
    if len(residuals_clean) > 0:
        q1, q99 = _np.percentile(residuals_clean, [1, 99])
        residuals_plot = residuals_clean[(residuals_clean >= q1) & (residuals_clean <= q99)]
        
        # Use more bins for better resolution
        n_bins = min(50, max(20, len(residuals_plot) // 100))
        _sns.histplot(residuals_plot, kde=True, bins=n_bins, ax=ax2)
        
        # Add statistics text
        mean_resid = _np.mean(residuals_clean)
        median_resid = _np.median(residuals_clean)
        std_resid = _np.std(residuals_clean)
        
        stats_text = f'Mean: {mean_resid:.3f}\nMedian: {median_resid:.3f}\nStd: {std_resid:.3f}\n(1-99% percentile shown)'
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'No valid residuals to plot', ha='center', va='center')
    
    ax2.set_title('Distribution of OOF Residuals (linear)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Residual', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    _plt.tight_layout()
    save_plot(fig2, 'oof_residuals_distribution.png', directory=out_dir)
    _plt.close(fig2)

    # Plot 3: QQ-plot for normality check
    fig3, ax3 = _plt.subplots(1, 1, figsize=(8, 6))
    try:
        _stats.probplot(residuals, dist='norm', plot=ax3)
        ax3.set_title('QQ-Plot of Residuals (linear)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    except Exception as e:
        ax3.text(0.5, 0.5, f'QQ-plot unavailable\n{str(e)}', ha='center', va='center')
        ax3.set_axis_off()
    _plt.tight_layout()
    save_plot(fig3, 'oof_residuals_qqplot.png', directory=out_dir)
    _plt.close(fig3)
    
    print('   OOF residual plots saved (3 separate files).')


def plot_residual_analysis(y_true, y_pred, model_name, output_dir):
    """
    Generates and saves residual analysis plots for the best model.

    Args:
        y_true (pd.Series): The true target values.
        y_pred (np.array): The model's predicted values.
        model_name (str): The name of the model being analyzed.
        output_dir (Path): The directory to save the plots.
    """
    print("   Generating residual analysis plots for the best model...")
    residuals = y_true - y_pred

    # Create a figure with two subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Residual Analysis for {model_name}', fontsize=16)

    # --- Plot 1: Residuals vs. Predicted Values ---
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, ax=axes[0])
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_title('Residuals vs. Predicted Values')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals (Actual - Predicted)')

    # --- Plot 2: Distribution of Residuals ---
    sns.histplot(residuals, kde=True, ax=axes[1])
    axes[1].set_title('Distribution of Residuals')
    axes[1].set_xlabel('Residual Value')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    save_plot(fig, 'residual_analysis_best_model.png', directory=output_dir)
    plt.close()
    print("   Residual plots saved.")


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be safe for use in filenames.
    
    Args:
        name: String to sanitize
        
    Returns:
        Sanitized string safe for filenames
    """
    import re
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', str(name))
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized


def generate_road_comparison_map(road_df: pd.DataFrame, 
                                road_id: str, 
                                output_dir: Path,
                                top_k: int = None) -> str:
    """
    Generate side-by-side comparison maps for a single road:
    Map A: Ground truth risk (all segments colored by actual risk)
    Map B: Model predictions (all segments + highlighted top-K predicted hotspots)
    
    Args:
        road_df: DataFrame containing all segments for this road
        road_id: Identifier for the road
        output_dir: Directory to save the map
        top_k: Number of top predicted segments to highlight (default from config)
        
    Returns:
        Path to the generated HTML file
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        print("[ERROR] Plotly not available for map generation")
        return None
        
    if top_k is None:
        import stage1_config as cfg
        top_k = getattr(cfg, 'VALIDATION_MAP_TOP_K', None) or getattr(cfg, 'HOTSPOT_K', 3)
    
    # Ensure required columns exist
    required_cols = ['latitude', 'longitude', 'actual_risk', 'predicted_risk']
    missing_cols = [col for col in required_cols if col not in road_df.columns]
    if missing_cols:
        print(f"[ERROR] Missing columns for road {road_id}: {missing_cols}")
        return None
    
    if len(road_df) == 0:
        print(f"[WARN] No segments found for road {road_id}")
        return None
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize road_id for filename
    sanitized_road_id = sanitize_filename(road_id)
    
    # Prepare data
    df = road_df.copy()
    
    # Identify top-K predicted hotspots
    top_k_segments = df.nlargest(min(top_k, len(df)), 'predicted_risk')
    df['is_predicted_hotspot'] = df.index.isin(top_k_segments.index)
    
    # Calculate color ranges for consistency across both maps
    actual_min, actual_max = df['actual_risk'].min(), df['actual_risk'].max()
    pred_min, pred_max = df['predicted_risk'].min(), df['predicted_risk'].max()
    
    # Create subplots with shared coordinates
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Ground Truth Risk - {road_id}",
            f"Model Predictions (Top {top_k} Highlighted) - {road_id}"
        ],
    specs=[[{"type": "scattermap"}, {"type": "scattermap"}]],
        horizontal_spacing=0.05
    )
    
    # Map A: Ground Truth (Left)
    fig.add_trace(
    go.Scattermap(
            lat=df['latitude'],
            lon=df['longitude'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['actual_risk'],
                colorscale='YlOrRd',
                cmin=actual_min,
                cmax=actual_max,
                colorbar=dict(
                    title="Actual Risk (FSI)",
                    x=0.45,  # Position colorbar between subplots
                    len=0.5
                )
            ),
            text=[f"Segment: {idx}<br>Actual Risk: {risk:.3f}" 
                  for idx, risk in zip(df.index, df['actual_risk'])],
            hovertemplate='<b>%{text}</b><br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>',
            name="Actual Risk"
        ),
        row=1, col=1
    )
    
    # Map B: Model Predictions (Right)
    # FIXED: Implement proper layering strategy for hotspot visibility
    
    # Layer 1: Non-hotspot segments (background) - neutral color for de-emphasis
    non_hotspot_df = df[~df['is_predicted_hotspot']]
    if len(non_hotspot_df) > 0:
        fig.add_trace(
            go.Scattermap(
                lat=non_hotspot_df['latitude'],
                lon=non_hotspot_df['longitude'],
                mode='markers',
                marker=dict(
                    size=5,
                    color='lightgray',  # Neutral background color
                    opacity=0.6,        # Reduced opacity for background
                    symbol='circle'
                ),
                text=[f"Segment: {idx}<br>Predicted Risk: {pred:.3f}<br>Actual Risk: {actual:.3f}" 
                      for idx, pred, actual in zip(non_hotspot_df.index, non_hotspot_df['predicted_risk'], non_hotspot_df['actual_risk'])],
                hovertemplate='<b>%{text}</b><br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>',
                name="Other Segments",
                showlegend=True
            ),
            row=1, col=2
        )

    # Layer 2: Top-K predicted hotspots (foreground) - rendered last for visibility
    hotspot_df = df[df['is_predicted_hotspot']]
    if len(hotspot_df) > 0:
        fig.add_trace(
            go.Scattermap(
                lat=hotspot_df['latitude'],
                lon=hotspot_df['longitude'],
                mode='markers',
                marker=dict(
                    size=16,  # Larger size for emphasis
                    color=hotspot_df['predicted_risk'],
                    colorscale='YlOrRd',
                    cmin=actual_min,
                    cmax=actual_max,
                    symbol='diamond',  # Distinct symbol
                    opacity=1.0,       # Full opacity for maximum visibility
                    colorbar=dict(title='Predicted Risk (FSI)', x=0.95, len=0.5)
                    # FIXED: Removed invalid 'line' property - not supported in scattermapbox
                ),
                text=[f"<b>HOTSPOT (Rank {rank})</b><br>Segment: {idx}<br>Predicted Risk: {pred:.3f}<br>Actual Risk: {actual:.3f}" 
                      for rank, (idx, pred, actual) in enumerate(
                          zip(hotspot_df.index, hotspot_df['predicted_risk'], hotspot_df['actual_risk']), 1)],
                hovertemplate='%{text}<br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>',
                name=f"Top {top_k} Predicted Hotspots"
            ),
            row=1, col=2
        )
    
    # Calculate center coordinates for map positioning
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    # Update layout for both subplots
    fig.update_layout(
        title=f"Road Validation Map: {road_id}",
        mapbox1=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=12
        ),
        mapbox2=dict(
            style="open-street-map", 
            center=dict(lat=center_lat, lon=center_lon),
            zoom=12
        ),
        height=600,
        width=1200,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left", 
            x=1.02
        )
    )
    
    # Save the map
    filename = f"validation_map_road_{sanitized_road_id}.html"
    filepath = output_dir / filename
    
    fig.write_html(str(filepath))
    print(f"[INFO] Road validation map saved: {filepath}")
    
    # Optionally save as PNG if configured
    import stage1_config as cfg
    if getattr(cfg, 'SAVE_MAPS_AS_IMAGES', False):
        try:
            png_filename = f"validation_map_road_{sanitized_road_id}.png"
            png_filepath = output_dir / png_filename
            fig.write_image(str(png_filepath), width=1200, height=600, scale=2)
            print(f"[INFO] Road validation map image saved: {png_filepath}")
        except Exception as e:
            print(f"[WARN] Failed to save PNG map (install kaleido: pip install kaleido): {e}")
    
    return str(filepath)
