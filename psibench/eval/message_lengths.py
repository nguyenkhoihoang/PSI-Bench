"""Compare conversation lengths between synthetic (from HuggingFace) and real datasets.

Usage:
  python -m psibench.eval.message_lengths \
    --config configs/default.yaml \
    --output-dir output/length_comparison
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from psibench.eval.utils import (
    extract_patient_messages_by_turn, 
    safe_dir_name, 
    get_all_psi_backend_pairs, 
    extract_model_size, 
    shorten_backend_name,
    sort_key_by_psi_and_size,
    sort_key_by_base_model,
    extract_base_model,
    get_model_family_priority,
    PSI_ABBREV,
    is_persimmon_label,
    PERSIMMON_COLOR,
    assign_backend_markers,
)

from psibench.data_loader.main_loader import (
    load_real_dataset,
    load_synthetic_hf_to_df,
)
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()


def count_words(text: str) -> int:
    """Count words in a text string."""
    return len(text.split())


def count_tokens_approx(text: str) -> int:
    """Approximate token count (roughly 1 token = 4 characters)."""
    return len(text) // 4


def count_characters(text: str) -> int:
    """Count characters in a text string (excluding spaces)."""
    return len(text.replace(' ', ''))


def count_sentences(text: str) -> int:
    """Count sentences in a text string using NLTK sentence tokenizer."""
    import nltk
    try:
        # Try finding 'punkt_tab' first as it's the newer, safer version
        nltk.data.find('tokenizers/punkt_tab')
    except:
        # If not found or if the older error is raised, download 'punkt_tab'
        nltk.download('punkt_tab')
    
    sentences = nltk.sent_tokenize(text)
    return max(1, len(sentences))  # At least 1 sentence


def calculate_average_lengths(messages_by_turn: Dict[int, List[str]]) -> pd.DataFrame:
    """Calculate average word and token counts for each turn.
    
    Args:
        messages_by_turn: Dictionary mapping turn_index -> list of messages
        
    Returns:
        DataFrame with columns: turn, count, avg_words, std_words, avg_tokens, std_tokens,
                                avg_chars_per_word, avg_words_per_sentence
    """
    results = []
    
    for turn_idx in sorted(messages_by_turn.keys()):
        messages = messages_by_turn[turn_idx]
        
        if not messages:
            continue
        
        word_counts = [count_words(msg) for msg in messages]
        token_counts = [count_tokens_approx(msg) for msg in messages]
        char_counts = [count_characters(msg) for msg in messages]
        sentence_counts = [count_sentences(msg) for msg in messages]
        
        # Calculate chars per word
        chars_per_word = [c / max(1, w) for c, w in zip(char_counts, word_counts)]
        
        # Calculate words per sentence
        words_per_sentence = [w / max(1, s) for w, s in zip(word_counts, sentence_counts)]
        
        results.append({
            'turn': turn_idx,
            'count': len(messages),
            'avg_words': np.mean(word_counts),
            'std_words': np.std(word_counts),
            'avg_tokens': np.mean(token_counts),
            'std_tokens': np.std(token_counts),
            'avg_chars_per_word': np.mean(chars_per_word),
            'avg_words_per_sentence': np.mean(words_per_sentence),
        })
    
    return pd.DataFrame(results)


def load_real_conversations(dataset_type: str) -> List[Dict]:
    """Load real conversations from Eeyore dataset.
    
    Args:
        dataset_type: Dataset type (e.g., 'esc', 'hope', 'annomi')
        
    Returns:
        List of conversation dictionaries
    """
    df = load_real_dataset(dataset_type)
    conversations = []
    
    for _, row in df.iterrows():
        conversations.append({
            'messages': row['messages']
        })
    
    return conversations



def compare_conversation_sets(synthetic_convs: List[Dict], real_convs: List[Dict], dataset_type: str, max_turns: int = None):
    """Shared comparison pipeline between synthetic and real conversations."""

    if len(synthetic_convs) != len(real_convs):
        print(f"\n⚠️  WARNING: Conversation counts do not match!")
        print(f"   Synthetic: {len(synthetic_convs)}, Real: {len(real_convs)}")
    else:
        print(f"\n✓ Conversation counts match: {len(synthetic_convs)}")

    print(f"\nAnalyzing patient messages by turn (max_turns: {max_turns or 'all'})...")
    synthetic_by_turn = extract_patient_messages_by_turn(synthetic_convs, max_turns)
    real_by_turn = extract_patient_messages_by_turn(real_convs, max_turns)

    synthetic_df = calculate_average_lengths(synthetic_by_turn)
    real_df = calculate_average_lengths(real_by_turn)

    comparison_data = []
    all_turns = sorted(set(synthetic_df['turn'].tolist() + real_df['turn'].tolist()))

    for turn in all_turns:
        synth_row = synthetic_df[synthetic_df['turn'] == turn]
        real_row = real_df[real_df['turn'] == turn]

        synth_words = synth_row['avg_words'].values[0] if not synth_row.empty else np.nan
        real_words = real_row['avg_words'].values[0] if not real_row.empty else np.nan
        synth_tokens = synth_row['avg_tokens'].values[0] if not synth_row.empty else np.nan
        real_tokens = real_row['avg_tokens'].values[0] if not real_row.empty else np.nan
        synth_count = synth_row['count'].values[0] if not synth_row.empty else 0
        real_count = real_row['count'].values[0] if not real_row.empty else 0

        comparison_data.append({
            'turn': turn,
            'synthetic_count': synth_count,
            'real_count': real_count,
            'synthetic_avg_words': synth_words,
            'real_avg_words': real_words,
            'words_diff': synth_words - real_words if not np.isnan(synth_words) and not np.isnan(real_words) else np.nan,
            'synthetic_avg_tokens': synth_tokens,
            'real_avg_tokens': real_tokens,
            'tokens_diff': synth_tokens - real_tokens if not np.isnan(synth_tokens) and not np.isnan(real_tokens) else np.nan,
        })

    comparison_df = pd.DataFrame(comparison_data)

    return synthetic_df, real_df, comparison_df


def compare_all_hf_pairs(config_path: str, output_dir: str, sort_method: str = 'psi-size',
                         synthetic_dir: str = None):
    """Compare all available HF psi/backend pairs against all real datasets combined (train split).
    
    Args:
        config_path: Path to configuration file
        output_dir: Output directory for results
        sort_method: Sorting method - 'psi-size' or 'base-model'
    """

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    max_turns = config.get('patient').get('max_turns')

    output_root = Path(output_dir)

    # Load all real data once (combined esc, hope, annomi)
    print("Loading all real conversations (esc, hope, annomi combined)...")
    all_real_convs = []
    for dataset in ['esc', 'hope', 'annomi']:
        try:
            real_convs = load_real_conversations(dataset)
            all_real_convs.extend(real_convs)
            print(f"  ✓ Loaded {len(real_convs)} from {dataset}")
        except Exception as e:
            print(f"  ⚠ Error loading {dataset}: {e}")
    
    print(f"Total real conversations: {len(all_real_convs)}")
    real_by_turn = extract_patient_messages_by_turn(all_real_convs, max_turns)
    real_df = calculate_average_lengths(real_by_turn)
    print(f"Real data: {len(real_df)} turns analyzed\n")

    # Get all unique (psi, backend_llm) pairs
    print("Loading all unique PSI-backend pairs...")
    dataset_name = config.get('eval', {}).get('hf_dataset', 'hknguyen20/psibench-data')
    all_pairs = get_all_psi_backend_pairs(dataset_name=dataset_name, synthetic_dir=synthetic_dir)
    print(f"Found {len(all_pairs)} unique (psi, backend_llm) pairs\n")

    # Load all synthetic data
    synthetic_data = {}
    
    for psi, backend_llm in sorted(all_pairs):
        label = f"{psi}-{safe_dir_name(backend_llm)}"
        print(f"Loading {label}...")
        
        # Load all data for this psi/backend pair
        synthetic_df_all = load_synthetic_hf_to_df(
            psi=psi,
            backend_llm=backend_llm,
            dataset_name=dataset_name,
            synthetic_dir=synthetic_dir,
        )

        if synthetic_df_all.empty:
            print(f"  ⚠ No synthetic rows for {psi}/{backend_llm}")
            continue

        # Combine all datasets for this PSI/backend combination
        synthetic_convs = synthetic_df_all.to_dict('records')
        print(f"  ✓ Loaded {len(synthetic_convs)} conversations (all datasets)")
        
        # Calculate statistics
        synth_by_turn = extract_patient_messages_by_turn(synthetic_convs, max_turns)
        synth_df = calculate_average_lengths(synth_by_turn)
        synthetic_data[label] = synth_df
        print(f"  ✓ {len(synth_df)} turns analyzed")

    if synthetic_data:
        out_dir = output_root / "hf"
        plot_multiple_psi_comparison(real_df, synthetic_data, out_dir, sort_method)
        create_comprehensive_metrics_csv(real_df, synthetic_data, out_dir, sort_method)
    else:
        print(f"  ⚠ No synthetic data found")


def calculate_mean_differences(real_df: pd.DataFrame, synthetic_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate mean differences between synthetic and real average word counts.
    
    Mean difference = average(synthetic_avg_words) - average(real_avg_words)
    
    Args:
        real_df: DataFrame with real data statistics
        synthetic_data: Dictionary mapping variant name -> DataFrame with synthetic statistics
    
    Returns:
        DataFrame with columns: dataset, display_name, mean_synthetic_words, mean_real_words, mean_difference
    """
    results = []
    
    # Calculate overall mean for real data
    mean_real_words = real_df['avg_words'].mean()
    
    for variant_name, synth_df in synthetic_data.items():
        # Calculate overall mean for this synthetic variant
        mean_synth_words = synth_df['avg_words'].mean()
        
        # Calculate difference (synthetic - real)
        mean_diff = mean_synth_words - mean_real_words
        
        # Create display label following emo_detection.py style
        display_label = 'Persimmon' if is_persimmon_label(variant_name) else variant_name
        for psi in PSI_ABBREV.keys():
            if variant_name.startswith(psi):
                backend_part = variant_name[len(psi):].lstrip('-_')
                backend_part = shorten_backend_name(backend_part)
                display_label = f"{PSI_ABBREV[psi]}-{backend_part}"
                break
        
        results.append({
            'dataset': variant_name,
            'display_name': display_label,
            'mean_synthetic_words': mean_synth_words,
            'mean_real_words': mean_real_words,
            'mean_difference': mean_diff
        })
    
    return pd.DataFrame(results)


def create_comprehensive_metrics_csv(real_df: pd.DataFrame, synthetic_data: Dict[str, pd.DataFrame], output_path: Path, sort_method: str = 'psi-size'):
    """Create a comprehensive CSV with all metrics across all models.
    
    Args:
        real_df: DataFrame with real data statistics
        synthetic_data: Dictionary mapping variant name -> DataFrame with synthetic statistics
        output_path: Path to save the CSV
        sort_method: Sorting method - 'psi-size' or 'base-model'
    """
    results = []
    
    # Calculate overall means for real data
    real_avg_words = real_df['avg_words'].mean()
    real_avg_chars_per_word = real_df['avg_chars_per_word'].mean()
    real_avg_words_per_sentence = real_df['avg_words_per_sentence'].mean()
    
    real_metrics = {
        'model': 'Real',
        'display_name': 'Real',
        'avg_words': real_avg_words,
        'avg_chars_per_word': real_avg_chars_per_word,
        'avg_words_per_sentence': real_avg_words_per_sentence,
    }
    results.append(real_metrics)
    
    # Calculate overall means for each synthetic variant
    for variant_name, synth_df in synthetic_data.items():
        # Create display label
        display_label = 'Persimmon' if is_persimmon_label(variant_name) else variant_name
        for psi in PSI_ABBREV.keys():
            if variant_name.startswith(psi):
                backend_part = variant_name[len(psi):].lstrip('-_')
                backend_part = shorten_backend_name(backend_part)
                display_label = f"{PSI_ABBREV[psi]}-{backend_part}"
                break
        
        synth_metrics = {
            'model': variant_name,
            'display_name': display_label,
            'avg_words': synth_df['avg_words'].mean(),
            'avg_chars_per_word': synth_df['avg_chars_per_word'].mean(),
            'avg_words_per_sentence': synth_df['avg_words_per_sentence'].mean(),
        }
        results.append(synth_metrics)
    
    # Create DataFrame and sort
    df = pd.DataFrame(results)
    
    # Calculate log-ratio similarity scores
    # Score formula: 100 * exp(-|log(x / real)|)
    # Score = 100 (perfect match) when x = real, decreases as x diverges from real
    
    def length_score(x, real):
        """Calculate log-ratio similarity score."""
        d = np.abs(np.log(x / real))
        return 100 * np.exp(-d)
    
    df['words_score'] = np.nan
    df['wps_score'] = np.nan
    df['final_score'] = np.nan
    
    for idx, row in df.iterrows():
        if row['model'] == 'Real':
            # Real has perfect scores of 100
            df.at[idx, 'words_score'] = 100.0
            df.at[idx, 'wps_score'] = 100.0
            df.at[idx, 'final_score'] = 100.0
        else:
            # Calculate individual scores
            words_score = length_score(row['avg_words'], real_avg_words)
            wps_score = length_score(row['avg_words_per_sentence'], real_avg_words_per_sentence)
            
            df.at[idx, 'words_score'] = words_score
            df.at[idx, 'wps_score'] = wps_score
            
            # Combined score: average of both scores
            df.at[idx, 'final_score'] = (words_score + wps_score) / 2
    
    # Sort: Real first, then by selected method
    real_row = df[df['model'] == 'Real']
    synth_rows = df[df['model'] != 'Real'].copy()
    
    if sort_method == 'base-model':
        synth_rows['sort_key'] = synth_rows['model'].apply(sort_key_by_base_model)
    else:
        synth_rows['sort_key'] = synth_rows['model'].apply(sort_key_by_psi_and_size)
    
    synth_rows = synth_rows.sort_values('sort_key').drop(columns=['sort_key'])
    
    df = pd.concat([real_row, synth_rows], ignore_index=True)
    
    # Save to CSV
    csv_path = output_path / 'comprehensive_metrics.csv'
    df.to_csv(csv_path, index=False, float_format='%.3f')
    
    print(f"\n{'='*100}")
    print("Comprehensive Metrics Across All Models")
    print(f"{'='*100}")
    print(f"\n{'Model':<30s} {'Avg Words':>10s} {'Chars/Word':>11s} {'Words/Sent':>11s} {'Words Score':>12s} {'WPS Score':>10s} {'Combined':>10s}")
    print(f"{'-'*30} {'-'*10} {'-'*11} {'-'*11} {'-'*12} {'-'*10} {'-'*10}")
    
    for _, row in df.iterrows():
        print(f"{row['display_name']:<30s} {row['avg_words']:>10.2f} {row['avg_chars_per_word']:>11.2f} "
              f"{row['avg_words_per_sentence']:>11.2f} {row['words_score']:>12.2f} "
              f"{row['wps_score']:>10.2f} {row['final_score']:>10.2f}")
    
    print(f"\n[CSV SAVED] {csv_path}\n")


def plot_metric_comparison(real_df: pd.DataFrame, synthetic_data: Dict[str, pd.DataFrame], 
                          output_path: Path, metric_col: str, ylabel: str, filename: str, sort_method: str = 'psi-size'):
    """Create a line graph comparing a specific metric across multiple PSI simulators.
    
    Args:
        real_df: DataFrame with real data statistics
        synthetic_data: Dictionary mapping variant name (psi-backend) -> DataFrame with synthetic statistics
        output_path: Path to save the plot
        metric_col: Column name to plot (e.g., 'avg_words', 'avg_chars_per_word', 'avg_words_per_sentence')
        ylabel: Y-axis label for the plot
        filename: Output filename for the plot
        sort_method: Sorting method - 'psi-size' or 'base-model'
    """
    # Filter data to show only turns 0-15
    real_df_filtered = real_df[real_df['turn'] <= 15].copy()
    synthetic_data_filtered = {k: v[v['turn'] <= 15].copy() for k, v in synthetic_data.items()}
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot real data (black line)
    ax.plot(real_df_filtered['turn'], real_df_filtered[metric_col], 
            linewidth=2, label='Real', alpha=0.9, linestyle='-', color='black', marker='|', markersize=5)
    
    # Define colors for each PSI type
    psi_colors = {
        'patientpsi': '#1f77b4',      # blue
        'roleplaydoh': '#ff7f0e',     # orange
    }
    
    # Define markers for each backend_llm type
    unique_backends = set()
    for variant_name in synthetic_data_filtered.keys():
        for psi in psi_colors.keys():
            if variant_name.startswith(psi):
                backend_part = variant_name[len(psi):].lstrip('-_')
                if backend_part:
                    unique_backends.add(backend_part)
                break
    
    # Family-aware markers shared with the other eval plots
    backend_markers = assign_backend_markers(list(unique_backends))
    
    # Track which PSI types and backends are actually used
    used_psi_types = set()
    used_backends = set()
    has_persimmon = False
    
    # Plot synthetic data for each variant
    for variant_name, synth_df in sorted(synthetic_data_filtered.items()):
        if is_persimmon_label(variant_name):
            # Single method without a backend LLM: styled like the human line, in green.
            has_persimmon = True
            ax.plot(synth_df['turn'], synth_df[metric_col],
                    linewidth=2, alpha=0.9, color=PERSIMMON_COLOR, linestyle='-', marker='|', markersize=5)
            continue

        psi_type = None
        backend_llm = None
        
        for psi in psi_colors.keys():
            if variant_name.startswith(psi):
                psi_type = psi
                backend_part = variant_name[len(psi):].lstrip('-_')
                backend_llm = backend_part if backend_part else 'unknown'
                break
        
        if not psi_type:
            parts = variant_name.split('-')
            psi_type = parts[0] if parts else 'unknown'
            backend_llm = '-'.join(parts[1:]) if len(parts) > 1 else 'unknown'
        
        color = psi_colors.get(psi_type, '#7f7f7f')
        
        marker = 'o'
        backend_name = None
        for backend, marker_char in backend_markers.items():
            if backend.lower() in backend_llm.lower() or backend_llm.lower() in backend.lower():
                marker = marker_char
                backend_name = backend
                break
        
        used_psi_types.add(psi_type)
        if backend_name:
            used_backends.add(backend_name)
        
        ax.plot(synth_df['turn'], synth_df[metric_col], 
            linewidth=2, alpha=0.8, color=color, linestyle='-', marker=marker, markersize=5)
    
    # Create two legends
    from matplotlib.lines import Line2D
    
    color_legend_elements = [Line2D([0], [0], color='black', linewidth=2, label='Real')]
    if has_persimmon:
        color_legend_elements.append(Line2D([0], [0], color=PERSIMMON_COLOR, linewidth=2, label='Persimmon'))
    for psi_type in sorted(used_psi_types):
        if psi_type in psi_colors:
            color_legend_elements.append(
                Line2D([0], [0], color=psi_colors[psi_type], linewidth=2, label=psi_type)
            )
    
    marker_legend_elements = []
    # Sort backends for legend based on sort_method
    if sort_method == 'base-model':
        def sort_by_base_model_and_size(backend_name):
            base_model = extract_base_model(backend_name)
            model_priority = get_model_family_priority(base_model)
            model_size = extract_model_size(backend_name)
            return (model_priority, model_size)
        sorted_legend_backends = sorted(used_backends, key=sort_by_base_model_and_size)
    else:
        sorted_legend_backends = sorted(used_backends, key=lambda b: extract_model_size(b))
    
    for backend in sorted_legend_backends:
        if backend in backend_markers:
            marker_legend_elements.append(
                Line2D([0], [0], color='gray', marker=backend_markers[backend], 
                       linestyle='None', markersize=8, label=shorten_backend_name(backend))
            )
    
    # ACL-style formatting
    ax.set_xlabel('Turn Index', fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.grid(False)
    ax.set_xticks([0, 5, 10, 15])
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Place color legend on top (first row)
    legend1 = ax.legend(handles=color_legend_elements, fontsize=16, 
                        bbox_to_anchor=(0.5, 1.05), loc='lower center', ncol=max(3, len(color_legend_elements)),
                        frameon=False, handletextpad=0.2, columnspacing=1.0)
    ax.add_artist(legend1)
    
    # Place marker legend on top (second row)
    legend2 = ax.legend(handles=marker_legend_elements, fontsize=16, 
                        bbox_to_anchor=(0.5, 0.98), loc='lower center', ncol=max(2, len(marker_legend_elements)),
                        frameon=False, handletextpad=0.2, columnspacing=1.0)
    ax.add_artist(legend2)
    
    plt.tight_layout()
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight', 
                bbox_extra_artists=[legend1, legend2], pad_inches=0.1)
    print(f"✓ {ylabel} graph saved to: {output_path / filename}")
    plt.close()


def plot_multiple_psi_comparison(real_df: pd.DataFrame, synthetic_data: Dict[str, pd.DataFrame], output_path: Path, sort_method: str = 'psi-size'):
    """Create line graphs comparing multiple PSI simulators against real data for all metrics.
    
    Args:
        real_df: DataFrame with real data statistics
        synthetic_data: Dictionary mapping variant name (psi-backend) -> DataFrame with synthetic statistics
        output_path: Path to save the plots
        sort_method: Sorting method - 'psi-size' or 'base-model'
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate and save mean differences
    mean_diff_df = calculate_mean_differences(real_df, synthetic_data)
    
    # Sort by selected method
    if sort_method == 'base-model':
        mean_diff_df['sort_key'] = mean_diff_df['dataset'].apply(sort_key_by_base_model)
    else:
        mean_diff_df['sort_key'] = mean_diff_df['dataset'].apply(sort_key_by_psi_and_size)
    mean_diff_df = mean_diff_df.sort_values('sort_key').drop(columns=['sort_key'])
    
    mean_diff_csv = output_path / 'mean_word_count_differences.csv'
    mean_diff_df.to_csv(mean_diff_csv, index=False)
    
    # Print formatted output following emo_detection.py style
    print(f"\n{'='*70}")
    print("Mean Word Count Differences (Synthetic - Real)")
    print(f"{'='*70}")
    print(f"Real Mean: {mean_diff_df['mean_real_words'].iloc[0]:.2f} words\n")
    
    for _, row in mean_diff_df.iterrows():
        sign = '+' if row['mean_difference'] >= 0 else ''
        print(f"  {row['display_name']:30s}: {row['mean_synthetic_words']:6.2f} - {row['mean_real_words']:6.2f} = {sign}{row['mean_difference']:6.2f}")
    
    print(f"\n[CSV SAVED] {mean_diff_csv}\n")
    
    # Generate plots for all three metrics
    print(f"\n{'='*70}")
    print("Generating Comparison Plots")
    print(f"{'='*70}\n")
    
    plot_metric_comparison(
        real_df, synthetic_data, output_path,
        metric_col='avg_words',
        ylabel='Average Word Count',
        filename='message_lengths.png',
        sort_method=sort_method
    )
    
    plot_metric_comparison(
        real_df, synthetic_data, output_path,
        metric_col='avg_chars_per_word',
        ylabel='Average Characters per Word',
        filename='avg_chars_per_word.png',
        sort_method=sort_method
    )
    
    plot_metric_comparison(
        real_df, synthetic_data, output_path,
        metric_col='avg_words_per_sentence',
        ylabel='Average Words per Sentence',
        filename='avg_words_per_sentence.png',
        sort_method=sort_method
    )



def main():
    """Main function to run conversation length comparison."""
    parser = argparse.ArgumentParser(
        description='Compare conversation lengths between synthetic (HuggingFace) and real datasets'
    )
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file (default: configs/default.yaml)')
    parser.add_argument('--output-dir', type=str, default='output/length_comparison',
                       help='Output directory for results (default: output/length_comparison)')
    parser.add_argument(
        '--sort-method',
        type=str,
        choices=['psi-size', 'base-model'],
        default='psi-size',
        help="Sorting method for results: 'psi-size' (sort by PSI then model size) or 'base-model' (sort by base model family, then size) (default: psi-size)"
    )
    
    parser.add_argument('--synthetic-dir', type=str, default=None,
                        help='Local folder tree of session_*.json (<psi>/<model>/<dataset>/...) used instead of the HF synthetic data; psi and model are read from the folder names')

    args = parser.parse_args()
    
    compare_all_hf_pairs(
        synthetic_dir=args.synthetic_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        sort_method=args.sort_method,
    )


if __name__ == '__main__':
    main()
