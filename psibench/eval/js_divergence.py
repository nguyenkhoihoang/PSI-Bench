"""
Calculate Jensen-Shannon divergence between categorical distributions (emotions, PTC codes, etc.) across turns.

For each synthetic dataset, compares the distribution at each turn (0-15)
with the real distribution at that same turn, then averages the JS divergence across all turns.

Usage (emotions):
    python psibench/eval/js_divergence.py \
        --csv-file output/emotion_analysis/emotion_percentages_by_turn_t16_no_neutral.csv \
        --turn-threshold 16 \
        --output-dir output/emotion_analysis \
        --label-column emotion \
        --label-type emotion

Usage (PTC):
    python psibench/eval/js_divergence.py \
        --csv-file output/ptc_analysis_turn0/ptc_percentages_by_turn_t16_no_filler.csv \
        --turn-threshold 16 \
        --output-dir output/ptc_analysis_turn0 \
        --label-column category \
        --label-type ptc
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Tuple

from psibench.eval.utils import (
    extract_base_model, 
    get_model_family_priority, 
    extract_model_size,
    PSI_ABBREV,
    shorten_backend_name
)


def get_label_distribution(df: pd.DataFrame, dataset: str, turn_index: int, 
                          labels: List[str], label_column: str) -> np.ndarray:
    """Get label distribution for a specific dataset and turn.
    
    Args:
        df: DataFrame with columns [dataset, turn_index, <label_column>, percentage]
        dataset: Dataset name (e.g., 'real', 'patientpsi_Llama-3.1-8B-Instruct')
        turn_index: Turn index (0-15)
        labels: List of label names in desired order
        label_column: Name of the column containing labels (e.g., 'emotion', 'ptc_category')
        
    Returns:
        Numpy array of percentages (sums to 100) in the order of labels list.
        Returns zeros if data not found.
    """
    subset = df[(df['dataset'] == dataset) & (df['turn_index'] == turn_index)]
    
    # Create distribution vector
    distribution = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        label_row = subset[subset[label_column] == label]
        if not label_row.empty:
            distribution[idx] = label_row.iloc[0]['percentage']
    
    return distribution


def calculate_js_per_turn(df: pd.DataFrame, synthetic_dataset: str, 
                          turn_threshold: int, labels: List[str], label_column: str) -> Tuple[List[float], float]:
    """Calculate Jensen-Shannon divergence for each turn and average.
    
    Args:
        df: DataFrame with label percentages
        synthetic_dataset: Name of synthetic dataset
        turn_threshold: Maximum turn to analyze (exclusive)
        labels: List of label names
        label_column: Name of the column containing labels
        
    Returns:
        Tuple of (per_turn_js_list, average_js)
    """
    js_divergences = []
    
    for turn_idx in range(turn_threshold):
        # Get distributions for this turn
        real_dist = get_label_distribution(df, 'real', turn_idx, labels, label_column)
        synth_dist = get_label_distribution(df, synthetic_dataset, turn_idx, labels, label_column)
        
        # Check if we have valid distributions
        if real_dist.sum() == 0 or synth_dist.sum() == 0:
            print(f"[WARNING] Missing data for {synthetic_dataset} at turn {turn_idx}")
            continue
        
        # Convert percentages to probabilities (divide by 100)
        real_prob = real_dist / 100.0
        synth_prob = synth_dist / 100.0
        
        # Calculate Jensen-Shannon divergence
        js = jensenshannon(real_prob, synth_prob)
        js_divergences.append(js)
    
    # Calculate average
    avg_js = np.mean(js_divergences) if js_divergences else float('nan')
    
    return js_divergences, avg_js


def sort_key_by_base_model(dataset_name: str) -> Tuple[int, int]:
    """Sort key: by base model family, then model size."""
    backend_part = dataset_name
    for psi in ['patientpsi', 'roleplaydoh']:
        if dataset_name.startswith(psi + '_'):
            backend_part = dataset_name[len(psi) + 1:]  # +1 for underscore
            break
    
    base_model = extract_base_model(backend_part)
    model_priority = get_model_family_priority(base_model)
    model_size = extract_model_size(backend_part)
    
    return (model_priority, model_size)


def get_display_name(dataset_name: str) -> str:
    """Convert dataset name to display name with PSI abbreviation (Persimmon has no backend)."""
    if dataset_name.lower().startswith('persimmon'):
        return 'Persimmon'
    for psi, abbrev in PSI_ABBREV.items():
        if dataset_name.startswith(psi + '_'):
            backend_part = dataset_name[len(psi) + 1:]  # +1 for underscore
            backend_short = shorten_backend_name(backend_part)
            return f"{abbrev}-{backend_short}"
    return dataset_name


def main():
    parser = argparse.ArgumentParser(
        description='Calculate Jensen-Shannon divergence between categorical distributions across turns.'
    )
    parser.add_argument(
        '--csv-file',
        type=str,
        required=True,
        help='Path to percentages CSV file'
    )
    parser.add_argument(
        '--turn-threshold',
        type=int,
        default=16,
        help='Maximum turn index to analyze (default: 16)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/emotion_analysis',
        help='Output directory for results (default: output/emotion_analysis)'
    )
    parser.add_argument(
        '--label-column',
        type=str,
        default='emotion',
        help='Name of column containing labels (default: emotion)'
    )
    parser.add_argument(
        '--label-type',
        type=str,
        default='emotion',
        help='Type of labels for output filenames (default: emotion)'
    )
    
    args = parser.parse_args()
    
    # Load CSV
    print(f"[INFO] Loading data from {args.csv_file}")
    df = pd.read_csv(args.csv_file)
    
    # Verify label column exists
    if args.label_column not in df.columns:
        print(f"[ERROR] Column '{args.label_column}' not found in CSV. Available columns: {df.columns.tolist()}")
        return
    
    # Get list of unique labels (in consistent order)
    labels = sorted(df[args.label_column].unique())
    print(f"[INFO] Found {len(labels)} {args.label_type}s: {labels}")
    
    # Get all synthetic datasets (exclude 'real')
    all_datasets = df['dataset'].unique()
    synthetic_datasets = [d for d in all_datasets if d != 'real']
    print(f"[INFO] Found {len(synthetic_datasets)} synthetic datasets")
    
    # Calculate JS divergence for each dataset
    results = []
    per_turn_results = []
    
    for dataset in synthetic_datasets:
        print(f"\n[INFO] Processing {dataset}...")
        js_per_turn, avg_js = calculate_js_per_turn(df, dataset, args.turn_threshold, labels, args.label_column)
        
        # Store average result
        results.append({
            'dataset': dataset,
            'display_name': get_display_name(dataset),
            'average_js_divergence': avg_js,
            'num_turns': len(js_per_turn)
        })
        
        # Store per-turn results
        for turn_idx, js_value in enumerate(js_per_turn):
            per_turn_results.append({
                'dataset': dataset,
                'display_name': get_display_name(dataset),
                'turn_index': turn_idx,
                'js_divergence': js_value
            })
    
    # Create DataFrames
    results_df = pd.DataFrame(results)
    per_turn_df = pd.DataFrame(per_turn_results)
    
    # Add final score column: (1 - divergence) * 100
    results_df['final_score'] = (1 - results_df['average_js_divergence']) * 100
    
    # Sort by base model
    results_df = results_df.loc[results_df['dataset'].map(sort_key_by_base_model).sort_values().index]
    per_turn_df = per_turn_df.loc[per_turn_df['dataset'].map(sort_key_by_base_model).sort_values().index]
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    avg_csv = output_dir / f'{args.label_type}_js_divergence_average.csv'
    per_turn_csv = output_dir / f'{args.label_type}_js_divergence_per_turn.csv'
    
    results_df.to_csv(avg_csv, index=False)
    per_turn_df.to_csv(per_turn_csv, index=False)
    
    print(f"\n[SAVED] Average JS divergence: {avg_csv}")
    print(f"[SAVED] Per-turn JS divergence: {per_turn_csv}")
    
    # Display results
    print("\n" + "="*100)
    print(f"JENSEN-SHANNON DIVERGENCE - AVERAGE ACROSS TURNS ({args.label_type.upper()})")
    print("="*100)
    display_df = results_df[['display_name', 'average_js_divergence', 'final_score', 'num_turns']].copy()
    display_df.columns = ['Dataset', 'Avg JS Divergence', 'Final Score', 'Num Turns']
    print(display_df.to_string(index=False))
    print()
    
    # Show per-turn statistics
    print("\n" + "="*100)
    print(f"JENSEN-SHANNON DIVERGENCE - PER TURN STATISTICS ({args.label_type.upper()})")
    print("="*100)
    
    # Pivot table for easier reading
    pivot_df = per_turn_df.pivot(index='display_name', columns='turn_index', values='js_divergence')
    pivot_df.columns = [f'Turn_{i}' for i in pivot_df.columns]
    print(pivot_df.to_string())
    print()
    
    print(f"\n[DONE] Analysis complete. Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
