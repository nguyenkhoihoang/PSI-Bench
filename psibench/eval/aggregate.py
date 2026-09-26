"""
Aggregate multiple evaluation metrics into a single score.
Reads CSV files for different metrics and extracts final_score columns.

The affect metric is supplied with either --emo or --valence (exactly one).
The chosen one names the affect column in the output table, so an emotion run
and a valence run can be placed side by side.

# Emotion variant -> output/aggregate/aggregate_scores.csv
python psibench/eval/aggregate.py \
  --mtld output/lexical_diversity_normalized/wasserstein_distances.csv \
  --emo output/emotion_analysis/emotion_js_divergence_average.csv \
  --ptc output/ptc_analysis/ptc_js_divergence_average.csv \
  --verbosity output/length_comparison/hf/comprehensive_metrics.csv \
  --depressive output/depressive_markers/depressive_distance.csv \
  --output-dir output/aggregate

# Valence variant -> output/aggregate_valence/aggregate_scores_valence.csv
python psibench/eval/aggregate.py \
  --mtld output/lexical_diversity_normalized/wasserstein_distances.csv \
  --valence output/valence_analysis/valence_js_divergence_average.csv \
  --ptc output/ptc_analysis/ptc_js_divergence_average.csv \
  --verbosity output/length_comparison/hf/comprehensive_metrics.csv \
  --depressive output/depressive_markers/depressive_distance.csv \
  --output-dir output/aggregate_valence

"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from psibench.eval.utils import shorten_backend_name, PSI_ABBREV


# Label variants that refer to the same backend. Metric CSVs written at different
# times can disagree: MTLD and verbosity were generated when
# Qwen3-30B-A3B-Instruct-2507 shortened to 'qwen3-30b-i', while
# BACKEND_LABEL_MAPPING in utils.py now yields 'qwen3-30b-a3b-i'. Both name the
# same model, so they are merged onto the right-hand (canonical) form here.
BACKEND_ALIASES = {
    'qwen3-30b-i': 'qwen3-30b-a3b-i',
}


def apply_backend_alias(name: str) -> str:
    """Map an alias of a short name (PS-backend / RD-backend) onto its canonical form."""
    for prefix in ('PS-', 'RD-'):
        if name.startswith(prefix):
            backend = name[len(prefix):]
            return prefix + BACKEND_ALIASES.get(backend, backend)
    return name


def normalize_dataset_name(name: str) -> str:
    """Normalize dataset names to standard format (PS-backend or RD-backend).
    
    Handles various naming conventions:
    - "patientpsi-Llama-3.1-8B-Instruct" -> "PS-llama3.1-8b-i"
    - "patientpsi_Llama-3.1-8B-Instruct" -> "PS-llama3.1-8b-i"
    - "PS-llama3.1-8b-i" -> "PS-llama3.1-8b-i" (already normalized)
    
    Args:
        name: Dataset name in any format
        
    Returns:
        Normalized name in format "PS-backend" or "RD-backend"
    """
    if not name or name == 'Real':
        return name
    
    # If already in short format (PS- or RD-), only resolve aliases
    if name.startswith('PS-') or name.startswith('RD-'):
        return apply_backend_alias(name)
    
    # Handle full PSI names with either dash or underscore
    for psi_full, psi_abbrev in PSI_ABBREV.items():
        # Try with dash separator
        if name.startswith(f"{psi_full}-"):
            backend_part = name[len(psi_full)+1:]  # +1 for the dash
            backend_short = shorten_backend_name(backend_part)
            return apply_backend_alias(f"{psi_abbrev}-{backend_short}")
        # Try with underscore separator
        elif name.startswith(f"{psi_full}_"):
            backend_part = name[len(psi_full)+1:]  # +1 for the underscore
            backend_short = shorten_backend_name(backend_part)
            return apply_backend_alias(f"{psi_abbrev}-{backend_short}")
    
    # If no match, return original
    return name


def aggregate_metrics(
    mtld_csv_path: str,
    ptc_csv_path: str,
    depressive_csv_path: str,
    verbosity_csv_path: str,
    emo_csv_path: Optional[str] = None,
    valence_csv_path: Optional[str] = None,
    verbosity_weight: float = 0.5,
    output_dir: str = "output",
    output_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Aggregate multiple evaluation metrics into a single score.
    
    Args:
        mtld_csv_path: Path to CSV with lexical diversity (expects 'final_score' column)
        ptc_csv_path: Path to CSV with PTC (expects 'final_score' column)
        depressive_csv_path: Path to CSV with depressive markers (expects 'final_score' column)
        verbosity_csv_path: Path to CSV with verbosity metrics (expects 'final_score' or 'combined_score' column)
        emo_csv_path: Path to CSV with emotion JS divergence (expects 'final_score' column)
        valence_csv_path: Path to CSV with valence JS divergence (expects 'final_score' column).
            Mutually exclusive with emo_csv_path; exactly one must be given. The one
            chosen names the affect column in the output ('emotion' or 'valence').
        verbosity_weight: Weight for verbosity metric (default: 0.5, others have weight 1.0)
        output_dir: Directory to save output CSV files
        output_name: Output CSV filename. Defaults to 'aggregate_scores.csv' for an
            emotion run and 'aggregate_scores_valence.csv' for a valence run.
        
    Returns:
        DataFrame with aggregated scores
    """
    
    # Resolve which affect metric is in play — exactly one of emotion / valence
    if (emo_csv_path is None) == (valence_csv_path is None):
        raise ValueError(
            "Provide exactly one affect metric: emo_csv_path (--emo) or "
            "valence_csv_path (--valence), not both and not neither."
        )
    affect_col = 'valence' if valence_csv_path is not None else 'emotion'
    affect_csv_path = valence_csv_path if valence_csv_path is not None else emo_csv_path
    
    # Read CSV files and extract final_score columns
    print("\n[INFO] Reading metric files...")
    print(f"[INFO] Affect metric: {affect_col}  ({affect_csv_path})")
    
    # Helper function to read and extract final_score
    def read_final_scores(csv_path: str, metric_name: str, use_combined_for_verbosity: bool = False) -> Dict[str, float]:
        df = pd.read_csv(csv_path)
        scores = {}
        
        # Determine score column name
        score_col = None
        if 'final_score' in df.columns:
            score_col = 'final_score'
        elif use_combined_for_verbosity and 'combined_score' in df.columns:
            score_col = 'combined_score'
        else:
            print(f"[WARNING] 'final_score' column not found in {csv_path}")
            return scores
        
        # Get dataset identifier column (try multiple possible names)
        dataset_col = None
        for col_name in ['dataset', 'display_name', 'model']:
            if col_name in df.columns:
                dataset_col = col_name
                break
        
        if dataset_col is None:
            dataset_col = df.columns[0]
        
        for _, row in df.iterrows():
            dataset_name = row[dataset_col]
            score_value = row[score_col]
            
            # Skip Real dataset and rows with missing scores
            if dataset_name and dataset_name != 'Real' and pd.notna(score_value):
                # Normalize the dataset name to standard format
                normalized_name = normalize_dataset_name(str(dataset_name))
                scores[normalized_name] = float(score_value)
        
        print(f"  ✓ {metric_name}: {len(scores)} datasets loaded")
        return scores
    
    # Read all metrics
    mtld_scores = read_final_scores(mtld_csv_path, "MTLD")
    affect_scores = read_final_scores(affect_csv_path, affect_col.capitalize())
    ptc_scores = read_final_scores(ptc_csv_path, "PTC")
    depressive_scores = read_final_scores(depressive_csv_path, "Depressive")
    verbosity_scores = read_final_scores(verbosity_csv_path, "Verbosity", use_combined_for_verbosity=True)
    
    # Get all unique datasets across all metrics
    all_datasets = set()
    all_datasets.update(mtld_scores.keys())
    all_datasets.update(affect_scores.keys())
    all_datasets.update(ptc_scores.keys())
    all_datasets.update(depressive_scores.keys())
    all_datasets.update(verbosity_scores.keys())
    
    print(f"\n[INFO] Found {len(all_datasets)} unique datasets total")
    
    # Coverage check — a dataset missing from a metric silently drops out of that
    # metric's weight, so report it rather than letting it show up as a bare NaN.
    per_metric = {
        'mtld': mtld_scores,
        affect_col: affect_scores,
        'ptc': ptc_scores,
        'depressive': depressive_scores,
        'verbosity': verbosity_scores,
    }
    missing = {m: sorted(all_datasets - set(sc.keys())) for m, sc in per_metric.items()}
    if any(missing.values()):
        print("\n[WARNING] Metric coverage is incomplete — these datasets are missing:")
        for metric, names in missing.items():
            if names:
                print(f"  {metric:<12} missing {len(names)}: {', '.join(names)}")
        print("  Rows missing a metric are averaged over fewer metrics (see 'metrics_used'),")
        print("  so their aggregate_score is not directly comparable to a complete row.")
    else:
        print("[INFO] All metrics cover all datasets.")
    
    # Create results dataframe
    results = []
    
    for dataset in sorted(all_datasets):
        mtld_score = mtld_scores.get(dataset, np.nan)
        affect_score = affect_scores.get(dataset, np.nan)
        ptc_score = ptc_scores.get(dataset, np.nan)
        depressive_score = depressive_scores.get(dataset, np.nan)
        verbosity_score = verbosity_scores.get(dataset, np.nan)
        
        # Compute weighted aggregate score
        # mtld, affect, ptc, depressive have weight 1.0; verbosity has weight specified by user
        scores = []
        weights = []
        
        if not np.isnan(mtld_score):
            scores.append(mtld_score)
            weights.append(1.0)
        
        if not np.isnan(affect_score):
            scores.append(affect_score)
            weights.append(1.0)
        
        if not np.isnan(ptc_score):
            scores.append(ptc_score)
            weights.append(1.0)

        if not np.isnan(depressive_score):
            scores.append(depressive_score)
            weights.append(1.0)
        
        if not np.isnan(verbosity_score):
            scores.append(verbosity_score)
            weights.append(verbosity_weight)
        
        # Calculate weighted average
        if scores and weights:
            aggregate_score = np.average(scores, weights=weights)
        else:
            aggregate_score = np.nan
        
        results.append({
            'dataset': dataset,
            'mtld': mtld_score,
            affect_col: affect_score,
            'ptc': ptc_score,
            'depressive': depressive_score,
            'verbosity': verbosity_score,
            'metrics_used': len(scores),
            'aggregate_score': aggregate_score
        })
    
    df = pd.DataFrame(results)
    
    # Sort by aggregate score (descending - higher is better)
    df = df.sort_values('aggregate_score', ascending=False, na_position='last')
    
    # Round all scores to 2 decimal places
    for col in ['mtld', affect_col, 'ptc', 'depressive', 'verbosity', 'aggregate_score']:
        df[col] = df[col].round(2)
    
    # Save to CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if output_name is None:
        output_name = "aggregate_scores.csv" if affect_col == 'emotion' else f"aggregate_scores_{affect_col}.csv"
    output_csv = output_path / output_name
    df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Saved aggregated results to: {output_csv}")
    
    # Display results
    print("\n" + "="*100)
    print(f"AGGREGATED EVALUATION SCORES — affect metric: {affect_col.upper()} (sorted by aggregate_score, descending)")
    print("="*100)
    print(f"Weights: MTLD=1.0, {affect_col.capitalize()}=1.0, PTC=1.0, Depressive=1.0, Verbosity={verbosity_weight}")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate evaluation metrics")
    parser.add_argument("--mtld", required=True, help="Path to MTLD CSV file (wasserstein_distances.csv)")
    affect_group = parser.add_mutually_exclusive_group(required=True)
    affect_group.add_argument("--emo", help="Path to emotion JS divergence CSV file (affect column named 'emotion')")
    affect_group.add_argument("--valence", help="Path to valence JS divergence CSV file (affect column named 'valence')")
    parser.add_argument("--ptc", required=True, help="Path to PTC CSV file")
    parser.add_argument("--depressive", required=True, help="Path to depressive marker CSV file")
    parser.add_argument("--verbosity", nargs='+', required=True, 
                        help="Path to verbosity CSV file, optionally followed by weight (default: 0.5)")
    parser.add_argument("--output-dir", default="output/aggregate", help="Output directory")
    parser.add_argument("--output-name", default=None,
                        help="Output CSV filename (default: aggregate_scores.csv for --emo, "
                             "aggregate_scores_valence.csv for --valence)")
    
    args = parser.parse_args()
    
    # Parse verbosity path and weight
    verbosity_path = args.verbosity[0]
    verbosity_weight = 1.0  # Default weight
    if len(args.verbosity) > 1:
        try:
            verbosity_weight = float(args.verbosity[1])
        except ValueError:
            print(f"[WARNING] Invalid verbosity weight '{args.verbosity[1]}', using default 0.5")
    
    aggregate_metrics(
        mtld_csv_path=args.mtld,
        emo_csv_path=args.emo,
        valence_csv_path=args.valence,
        ptc_csv_path=args.ptc,
        depressive_csv_path=args.depressive,
        verbosity_csv_path=verbosity_path,
        verbosity_weight=verbosity_weight,
        output_dir=args.output_dir,
        output_name=args.output_name
    )
