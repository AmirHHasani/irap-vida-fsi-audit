# stage1_reporting.py
import pandas as pd
from pathlib import Path

def generate_final_report(metrics_results, shap_summary_df, countermeasure_summary, hypothesis_results, config_params, output_dir, filename):
    """
    Assembles all analysis results into a comprehensive markdown report.

    Args:
        metrics_results (list): A list of dictionaries, where each dict contains the
                                results for one trained model.
        shap_summary_df (pd.DataFrame): DataFrame of features ranked by SHAP importance.
        countermeasure_summary (pd.DataFrame): The comprehensive high-risk analysis table.
        hypothesis_results (pd.DataFrame): DataFrame with results of statistical tests.
        config_params (dict): A dictionary of the configuration parameters used for the run.
        output_dir (Path): The directory to save the report in.
        filename (str): The name of the final markdown file.
    """
    report_content = "# Stage 1: Interpretable Road Risk Model Analysis Report\n\n"
    report_content += "This report summarizes the results of the end-to-end road risk modeling and interpretation pipeline.\n\n"

    # --- Section 1: Model Performance Comparison ---
    # WHY: This new section creates a table comparing all trained models,
    # which is why we now accept 'metrics_results'.
    report_content += "## 1. Model Performance Comparison\n\n"
    report_content += "The following table compares the performance of all trained models on the test set.\n\n"
    
    perf_data = []
    for result in metrics_results:
        # Format numeric values properly, leave as 'N/A' if missing
        r2 = result['metrics'].get('Test R2', None)
        mae = result['metrics'].get('Test MAE', None)
        rmse = result['metrics'].get('Test RMSE', None)
        
        perf_data.append({
            'Model': result['name'],
            'R² Score': f"{r2:.6f}" if r2 is not None else 'N/A',
            'MAE': f"{mae:.6f}" if mae is not None else 'N/A',
            'RMSE': f"{rmse:.6f}" if rmse is not None else 'N/A',
            'Split Strategy': result['metrics'].get('Split Strategy', 'N/A')
        })
    
    perf_df = pd.DataFrame(perf_data)
    report_content += perf_df.to_markdown(index=False, tablefmt="grid")
    report_content += "\n\n---\n\n"

    # --- Section 2: Global / Hotspot Risk Drivers (SHAP Analysis) ---
    report_content += "## 2. Risk Driver Importance (Global / Hotspot-Focused)\n\n"
    report_content += ("This section surfaces feature importance derived from SHAP. In the BY_ROAD hotspot pipeline, "
                       "global SHAP may be skipped in favor of hotspot-focused OOF SHAP importance (restricted to predicted high-risk segments).\n\n")
    hotspot_shap_path = output_dir / 'hotspot_shap' / 'hotspot_shap_feature_importance.csv'
    hotspot_shap_df = None
    if hotspot_shap_path.exists():
        try:
            hotspot_shap_df = pd.read_csv(hotspot_shap_path)
        except Exception:
            hotspot_shap_df = None
    if hotspot_shap_df is not None and not hotspot_shap_df.empty:
        report_content += "**Hotspot-Focused SHAP Importance (Mean |SHAP| Across Predicted Hotspots)**\n\n"
        top_hotspot = hotspot_shap_df.head(25)
        report_content += top_hotspot.to_markdown(index=False, tablefmt="grid") + "\n\n"
    if shap_summary_df is not None and not shap_summary_df.empty:
        report_content += "**Standard SHAP Summary (Global)**\n\n"
        report_content += shap_summary_df.to_markdown(index=False, tablefmt="grid") + "\n\n"
    if (hotspot_shap_df is None or hotspot_shap_df.empty) and (shap_summary_df is None or shap_summary_df.empty):
        report_content += "No SHAP-based driver information was available for this run.\n\n"
    report_content += "---\n\n"

    # --- Section 3: Hotspot Ranking Performance ---
    report_content += "## 3. Hotspot Ranking Performance\n\n"
    report_content += ("Evaluation of predicted vs. actual high-risk (Top-K) segments per road. Metrics include overlap@K, "
                       "precision@K, recall@K, reciprocal rank (RR), and nDCG@K across configured K values.\n\n")
    ranking_long_path = output_dir / 'fold_results' / 'per_road_hotspot_metrics_long.csv'
    ranking_agg_path = output_dir / 'fold_results' / 'hotspot_ranking_metrics_aggregated.json'
    if ranking_agg_path.exists():
        try:
            import json
            with open(ranking_agg_path, 'r') as f:
                agg_metrics = json.load(f)
            # Flatten for display
            flat_rows = []
            for k, vals in agg_metrics.items():
                d = {'K_or_Aggregate': k}
                d.update(vals)
                flat_rows.append(d)
            agg_df = pd.DataFrame(flat_rows)
            report_content += "**Aggregated Ranking Metrics**\n\n"
            report_content += agg_df.to_markdown(index=False, tablefmt="grid") + "\n\n"
        except Exception:
            report_content += "Could not parse aggregated ranking metrics file.\n\n"
    if ranking_long_path.exists():
        try:
            rl_df = pd.read_csv(ranking_long_path)
            sample_rl = rl_df.head(50)
            report_content += "**Sample Per-Road Ranking Metrics (first 50 rows)**\n\n"
            report_content += sample_rl.to_markdown(index=False, tablefmt="grid") + "\n\n"
        except Exception:
            report_content += "Could not load detailed per-road ranking metrics.\n\n"
    report_content += "---\n\n"

    # --- Section 4: Countermeasure Coverage & Hotspot Overlay ---
    report_content += "## 4. Countermeasure Coverage & Hotspot Overlay\n\n"
    report_content += ("This section links predicted hotspots with existing countermeasures. Coverage metrics quantify the "
                       "percentage of predicted hotspots (TP+FP) already treated, and gaps (FN) lacking interventions.\n\n")
    overlay_path = output_dir / 'fold_results' / 'hotspot_prediction_overlay.csv'
    coverage_path = output_dir / 'fold_results' / 'countermeasure_coverage_summary.csv'
    freq_path = output_dir / 'fold_results' / 'countermeasure_occurrence_counts.csv'
    if coverage_path.exists():
        try:
            cov_df = pd.read_csv(coverage_path)
            report_content += "**Countermeasure Coverage Metrics**\n\n"
            report_content += cov_df.head(25).to_markdown(index=False, tablefmt="grid") + "\n\n"
        except Exception:
            report_content += "Coverage metrics file present but could not be read.\n\n"
    if freq_path.exists():
        try:
            freq_df = pd.read_csv(freq_path)
            report_content += "**Most Frequent Countermeasures on Hotspots (Top 30)**\n\n"
            report_content += freq_df.head(30).to_markdown(index=False, tablefmt="grid") + "\n\n"
        except Exception:
            report_content += "Countermeasure frequency file present but could not be read.\n\n"
    if overlay_path.exists():
        report_content += ("Full TP/FP/FN overlay saved as `hotspot_prediction_overlay.csv`. "
                           "Interactive map (if generated) is in the maps directory.\n\n")
    else:
        report_content += "No hotspot overlay file was generated for this run.\n\n"
    report_content += "---\n\n"

    # --- Section 5: Legacy High-Risk Segment Detail (If Available) ---
    report_content += "## 5. Legacy High-Risk Segment Detail (Optional)\n\n"
    report_content += ("If the earlier style high-risk segment local SHAP analysis was executed, those results appear here. "
                       "In the new hotspot workflow this may be empty.\n\n")
    if countermeasure_summary is not None and not countermeasure_summary.empty:
        report_cols = ['Location ID', 'Actual_FSI', 'Top_Local_Drivers_With_Global_Rank', 'Proposed_Countermeasures']
        display_summary = countermeasure_summary[[col for col in report_cols if col in countermeasure_summary.columns]]
        report_content += display_summary.head(30).to_markdown(index=False, tablefmt="grid") + "\n\n"
    else:
        report_content += "No legacy high-risk local explanation table was produced.\n\n"
    report_content += "---\n\n"

    # --- Section 6: Supplementary Hypothesis Testing ---
    report_content += "## 6. Supplementary Hypothesis Testing\n\n"
    report_content += "This table shows the results of traditional statistical tests performed on the **training data** for pre-defined hypotheses, compared against the model's SHAP findings.\n\n"
    if hypothesis_results is not None and not hypothesis_results.empty:
        report_content += hypothesis_results.to_markdown(index=False, tablefmt="grid")
    else:
        report_content += "No hypothesis tests were performed.\n"

    # --- Appendix: Key Configuration Parameters ---
    report_content += "---\n\n## Appendix: Key Configuration Parameters\n\n"
    try:
        cfg_df = (pd.DataFrame(sorted(config_params.items()), columns=['Parameter','Value'])
                  .query("Parameter not in ['__doc__','__annotations__']"))
        report_content += cfg_df.to_markdown(index=False, tablefmt='grid') + "\n\n"
    except Exception:
        report_content += "Could not render configuration parameters.\n\n"

    # --- Save the final report ---
    try:
        report_path = output_dir / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"   Final report saved successfully to: {report_path}")
    except Exception as e:
        print(f"   [ERROR] Failed to save the final report: {e}")