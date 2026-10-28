"""
Calculate and compare Lexical Diversity (MTLD, MATTR) between real and synthetic conversations
using strict preprocessing and length control strategies.

Boxplots group by model name, with two side-by-side boxes per model (PS and RD) differentiated by color.

Sort by base model instead of PSI:

python psibench/eval/lexical_diversity.py \
    --output-dir output/lexical_diversity_normalized \
    --min-tokens 100 \
    --sort-method base-model
    
Edit and redraw graphs without rerunning analysis:

python psibench/eval/lexical_diversity.py \
    --from-csv output/lexical_diversity_gpt41mini/lexical_diversity_all_pairs.csv \
    --output-dir output/lexical_diversity_gpt41mini/ \
    --min-tokens 100 \
    --sort-method base-model

Redraw to calculate Wasserstein distances without rerunning analysis:
Sort by wasserstein distance between MTLD distributions of real vs synthetic datasets:
python psibench/eval/lexical_diversity.py \
    --output-dir output/lexical_diversity3 \
    --min-tokens 100 \
    --from-csv output/lexical_diversity2/lexical_diversity_all_pairs.csv \
    --sort-method wasserstein

Sort by base model family (Llama, Qwen, GPT, etc.):
python psibench/eval/lexical_diversity.py \
    --output-dir output/lexical_diversity3 \
    --min-tokens 100 \
    --from-csv output/lexical_diversity_normalized/lexical_diversity_all_pairs.csv \
        --output-dir output/lexical_diversity_normalized \
    --sort-method base-model
    

"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lexicalrichness import LexicalRichness
from scipy.stats import wasserstein_distance

from psibench.data_loader.main_loader import load_real_dataset, load_synthetic_hf_to_df
from psibench.eval.utils import (
    get_all_psi_backend_pairs, safe_dir_name, PSI_ABBREV, PSI_LABELS, extract_model_size, 
    shorten_backend_name, sort_key_by_psi_and_size, sort_key_by_base_model, 
    sort_key_by_psi_base_model, sort_summary_key, get_assistant_messages, aggregate_messages,
    extract_base_model, get_model_family_priority, PSI_COLORS, is_persimmon_label
)
from psibench.data_loader.utils import normalize_backend_name

def calculate_bidirectional_mtld(text: str, threshold: float = 0.72) -> float:
    """
    Calculate bidirectional MTLD (Forward + Backward) / 2.
    """
    if not text or len(text.split()) < 1: # Basic sanity check
        return float('nan')

    try:
        lex = LexicalRichness(text)
        
        # Forward MTLD
        mtld_fwd = lex.mtld(threshold=threshold)
        
        # Backward MTLD (reverse the word list)
        words = text.split()
        reversed_text = " ".join(words[::-1])
        lex_rev = LexicalRichness(reversed_text)
        mtld_bwd = lex_rev.mtld(threshold=threshold)
        
        if math.isnan(mtld_fwd) or math.isnan(mtld_bwd):
            return float('nan')
            
        return (mtld_fwd + mtld_bwd) / 2.0
        
    except Exception:
        return float('nan')

def truncate_to_length(text: str, target_length: int) -> str:
    """Truncate text to the first `target_length` tokens."""
    words = text.split()
    if len(words) <= target_length:
        return text
    return " ".join(words[:target_length])


def analyze_session(
    real_msgs: List[str],
    synth_msgs: List[str],
    session_id: int,
    min_tokens: int = 100,
    synth_turn_msgs: Optional[List[str]] = None
) -> List[Dict]:
    """
    Analyze a single session using Raw, Matched, Cumulative, and Turn-Level strategies.
    """
    results = []
    
    # 1. Aggregation (messages are already preprocessed from get_assistant_messages)
    real_text_raw = aggregate_messages(real_msgs)
    synth_text_raw = aggregate_messages(synth_msgs)
    
    real_len = len(real_text_raw.split())
    synth_len = len(synth_text_raw.split())
    
    # --- Strategy A: Raw Aggregated ---
    results.append({
        'session_id': session_id,
        'speaker_type': 'Real',
        'analysis_type': 'Raw',
        'turn_index': -1, # N/A
        'token_count': real_len,
        'mtld': calculate_bidirectional_mtld(real_text_raw),
        'meets_threshold': real_len >= min_tokens
    })
        
    results.append({
        'session_id': session_id,
        'speaker_type': 'Synthetic',
        'analysis_type': 'Raw',
        'turn_index': -1, # N/A
        'token_count': synth_len,
        'mtld': calculate_bidirectional_mtld(synth_text_raw),
        'meets_threshold': synth_len >= min_tokens
    })

    # --- Strategy B: Matched Truncated ---
    # For matched, we still need some content to work with. 
    # If either is 0, we can't really match. But let's assume > 0 for now or handle gracefully.
    if real_len > 0 and synth_len > 0:
        target_len = min(real_len, synth_len)
        
        real_text_trunc = truncate_to_length(real_text_raw, target_len)
        synth_text_trunc = truncate_to_length(synth_text_raw, target_len)
        
        # Matched usually implies we want to compare "fairly". 
        # If the common length is < min_tokens, we flag it.
        meets_threshold = target_len >= min_tokens
        
        results.append({
            'session_id': session_id,
            'speaker_type': 'Real',
            'analysis_type': 'Matched',
            'turn_index': -1,
            'token_count': len(real_text_trunc.split()),
            'mtld': calculate_bidirectional_mtld(real_text_trunc),

            'meets_threshold': meets_threshold
        })
        
        results.append({
            'session_id': session_id,
            'speaker_type': 'Synthetic',
            'analysis_type': 'Matched',
            'turn_index': -1,
            'token_count': len(synth_text_trunc.split()),
            'mtld': calculate_bidirectional_mtld(synth_text_trunc),

            'meets_threshold': meets_threshold
        })
        
    # --- Strategy C: Cumulative Analysis ---
    # Real (messages already preprocessed)
    cum_text = ""
    for i, msg in enumerate(real_msgs):
        if not msg: continue
        cum_text += " " + msg
        
        current_len = len(cum_text.split())
        results.append({
            'session_id': session_id,
            'speaker_type': 'Real',
            'analysis_type': 'Cumulative',
            'turn_index': i + 1,
            'token_count': current_len,
            'mtld': calculate_bidirectional_mtld(cum_text),

            'meets_threshold': current_len >= min_tokens
        })

    # Synthetic (messages already preprocessed)
    cum_text = ""
    for i, msg in enumerate(synth_msgs):
        if not msg: continue
        cum_text += " " + msg
        
        current_len = len(cum_text.split())
        results.append({
            'session_id': session_id,
            'speaker_type': 'Synthetic',
            'analysis_type': 'Cumulative',
            'turn_index': i + 1,
            'token_count': current_len,
            'mtld': calculate_bidirectional_mtld(cum_text),

            'meets_threshold': current_len >= min_tokens
        })

    # --- Strategy D: Turn-Level Analysis ---
    # Real (messages already preprocessed)
    for i, msg in enumerate(real_msgs):
        if not msg: continue
        
        current_len = len(msg.split())
        results.append({
            'session_id': session_id,
            'speaker_type': 'Real',
            'analysis_type': 'Turn-Level',
            'turn_index': i + 1,
            'token_count': current_len,
            'mtld': calculate_bidirectional_mtld(msg),

            'meets_threshold': current_len >= min_tokens
        })
            
    # Synthetic - ONLY if synth_turn_msgs is provided (messages already preprocessed)
    if synth_turn_msgs:
        for i, msg in enumerate(synth_turn_msgs):
            if not msg: continue
            
            current_len = len(msg.split())
            results.append({
                'session_id': session_id,
                'speaker_type': 'Synthetic',
                'analysis_type': 'Turn-Level',
                'turn_index': i + 1,
                'token_count': current_len,
                'mtld': calculate_bidirectional_mtld(msg),
    
                'meets_threshold': current_len >= min_tokens
            })
            
    return results


def compare_all_hf_pairs(output_dir: str, min_tokens: int = 100, dataset_name: str = 'hknguyen20/psibench-conv', sort_method: str = 'psi-size',
                         synthetic_dir: str = None):
    """Compare all available HF psi/backend pairs against all real datasets combined.
    
    Args:
        output_dir: Output directory path
        min_tokens: Minimum token threshold
        dataset_name: HuggingFace dataset name
        sort_method: Sorting method - 'psi-size', 'base-model', or 'psi-base-model'
    """
    
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Load all real data once (combined esc, hope, annomi)
    print("\n[INFO] Loading all real conversations (esc, hope, annomi combined)...")
    try:
        real_df = load_real_dataset("all")
        real_conversations = real_df.to_dict('records')
        print(f"[INFO] Loaded {len(real_conversations)} real conversations")
    except Exception as e:
        print(f"[ERROR] Could not load real data: {e}")
        return

    # Get all unique (psi, backend_llm) pairs
    print("\n[INFO] Loading all unique PSI-backend pairs...")
    all_pairs = get_all_psi_backend_pairs(dataset_name=dataset_name, synthetic_dir=synthetic_dir)
    print(f"[INFO] Found {len(all_pairs)} unique (psi, backend_llm) pairs")

    # Store concatenated texts for each dataset
    dataset_concatenated_texts = {}
    
    # Analyze real data
    print("\n[INFO] Analyzing real conversations...")
    all_results = []
    real_all_msgs = []
    for idx, conv in enumerate(real_conversations):
        real_msgs = get_assistant_messages(conv.get('messages', []))
        if not real_msgs:
            continue
        
        # Collect for concatenation
        real_all_msgs.extend(real_msgs)
        
        # Raw aggregated analysis
        real_text_raw = aggregate_messages(real_msgs)
        real_len = len(real_text_raw.split())
        
        all_results.append({
            'dataset': 'Real',
            'session_id': idx,
            'speaker_type': 'Real',
            'analysis_type': 'Raw',
            'turn_index': -1,
            'token_count': real_len,
            'mtld': calculate_bidirectional_mtld(real_text_raw),
            'meets_threshold': real_len >= min_tokens
        })
        
        # Cumulative analysis
        cum_text = ""
        for turn_idx, msg in enumerate(real_msgs):
            if not msg:
                continue
            cum_text += " " + msg
            current_len = len(cum_text.split())
            
            all_results.append({
                'dataset': 'Real',
                'session_id': idx,
                'speaker_type': 'Real',
                'analysis_type': 'Cumulative',
                'turn_index': turn_idx + 1,
                'token_count': current_len,
                'mtld': calculate_bidirectional_mtld(cum_text),
                'meets_threshold': current_len >= min_tokens
            })
    
    # Store concatenated text for Real
    dataset_concatenated_texts['Real'] = aggregate_messages(real_all_msgs)
    
    # Analyze each PSI-backend pair
    for psi, backend_llm in sorted(all_pairs):
        normalized_backend = normalize_backend_name(backend_llm)
        label = f"{psi}-{safe_dir_name(normalized_backend)}"
        
        print(f"\n[INFO] Loading synthetic data for {label}...")
        
        try:
            # Load all data for this psi/backend pair
            synthetic_df = load_synthetic_hf_to_df(psi=psi, backend_llm=normalized_backend, dataset_name=dataset_name, synthetic_dir=synthetic_dir)
            
            if synthetic_df.empty:
                print(f"[WARNING] No data found for {label}")
                continue
            
            synthetic_conversations = synthetic_df.to_dict('records')
            print(f"[INFO] Analyzing {len(synthetic_conversations)} synthetic conversations ({label})...")
            
            # Analyze synthetic data
            synth_all_msgs = []
            for idx, conv in enumerate(synthetic_conversations):
                synth_msgs = get_assistant_messages(conv.get('messages', []))
                if not synth_msgs:
                    continue
                
                # Collect for concatenation
                synth_all_msgs.extend(synth_msgs)
                
                # Raw aggregated analysis
                synth_text_raw = aggregate_messages(synth_msgs)
                synth_len = len(synth_text_raw.split())
                
                all_results.append({
                    'dataset': label,
                    'session_id': idx,
                    'speaker_type': 'Synthetic',
                    'analysis_type': 'Raw',
                    'turn_index': -1,
                    'token_count': synth_len,
                    'mtld': calculate_bidirectional_mtld(synth_text_raw),
                    'meets_threshold': synth_len >= min_tokens
                })
                
                # Cumulative analysis
                cum_text = ""
                for turn_idx, msg in enumerate(synth_msgs):
                    if not msg:
                        continue
                    cum_text += " " + msg
                    current_len = len(cum_text.split())
                    
                    all_results.append({
                        'dataset': label,
                        'session_id': idx,
                        'speaker_type': 'Synthetic',
                        'analysis_type': 'Cumulative',
                        'turn_index': turn_idx + 1,
                        'token_count': current_len,
                        'mtld': calculate_bidirectional_mtld(cum_text),
                        'meets_threshold': current_len >= min_tokens
                    })
            
            # Store concatenated text for this dataset
            dataset_concatenated_texts[label] = aggregate_messages(synth_all_msgs)
                
        except Exception as e:
            print(f"[ERROR] Failed to load {label}: {e}")
            continue
    
    if not all_results:
        print("[ERROR] No results generated.")
        return
    
    # Save results
    df = pd.DataFrame(all_results)
    csv_path = output_root / 'lexical_diversity_all_pairs.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n[CSV SAVED] {csv_path}")
    
    # Create comparison visualizations
    create_comparison_visualizations(df, output_root, min_tokens, dataset_concatenated_texts, sort_method)
    print(f"\n[DONE] Analysis complete. Results saved to: {output_root}")


def create_comparison_visualizations(df: pd.DataFrame, output_dir: Path, min_tokens: int, dataset_concatenated_texts: Dict[str, str] = None, sort_method: str = 'psi-size'):
    """Generate comparison plots across all PSI-backend pairs.
    
    Creates grouped boxplots where each model appears once with two side-by-side boxes
    (one for patientpsi, one for roleplaydoh) differentiated by color.
    
    Creates:
    1. Grouped boxplot of raw MTLD scores (by model, with PS/RD hue)
    2. Summary statistics table
    3. Cumulative MTLD progression plot
    
    Args:
        df: DataFrame with analysis results
        output_dir: Output directory path
        min_tokens: Minimum token threshold
        dataset_concatenated_texts: Optional dict of concatenated texts per dataset
        sort_method: Sorting method - 'psi-size' (sort by model size), 
                     'base-model' (sort by base model family then size), 
                     'psi-base-model' (same as psi-size), 
                     or 'wasserstein' (sort by average Wasserstein distance)
    """
    sns.set_theme(style="whitegrid")
    
    # Filter to only Raw analysis type and sessions that meet threshold
    raw_df = df[(df['analysis_type'] == 'Raw') & (df['meets_threshold'] == True)].copy()
    
    if raw_df.empty:
        print("[WARNING] No data meeting threshold for visualization")
        return
    
    # Extract PSI type and backend name for each dataset
    raw_df['psi_type'] = None
    raw_df['backend_name'] = None
    
    for idx, row in raw_df.iterrows():
        dataset = row['dataset']
        if dataset == 'Real':
            raw_df.at[idx, 'psi_type'] = 'Real'
            raw_df.at[idx, 'backend_name'] = 'human'
        elif is_persimmon_label(dataset):
            raw_df.at[idx, 'psi_type'] = 'persimmon'
            raw_df.at[idx, 'backend_name'] = 'persimmon'
        else:
            for psi in ['patientpsi', 'roleplaydoh']:
                if dataset.startswith(psi):
                    raw_df.at[idx, 'psi_type'] = psi
                    backend_part = dataset[len(psi):].lstrip('-_')
                    backend_short = shorten_backend_name(backend_part)
                    raw_df.at[idx, 'backend_name'] = backend_short
                    break
    
    # Get unique backends (excluding human)
    synthetic_backends = sorted([b for b in raw_df['backend_name'].unique() if b not in ('human', 'persimmon')])
    
    # Sort backends based on method
    if sort_method == 'wasserstein':
        # Load wasserstein distances and sort by average distance across both PSIs
        wasserstein_csv = output_dir / 'wasserstein_distances.csv'
        if wasserstein_csv.exists():
            wd_df = pd.read_csv(wasserstein_csv)
            # Create mapping from backend to average wasserstein distance
            backend_wd = {}
            for backend in synthetic_backends:
                distances = []
                for psi_abbrev in ['PS', 'RD']:
                    dataset_label = f"{psi_abbrev}-{backend}"
                    match = wd_df[wd_df['dataset'] == dataset_label]
                    if not match.empty:
                        distances.append(match.iloc[0]['wasserstein_distance'])
                if distances:
                    backend_wd[backend] = sum(distances) / len(distances)
            
            synthetic_backends = sorted(synthetic_backends, key=lambda x: backend_wd.get(x, float('inf')))
            print(f"[INFO] Sorting by average Wasserstein distance (loaded from {wasserstein_csv})")
        else:
            print(f"[WARNING] Wasserstein CSV not found at {wasserstein_csv}, falling back to size sorting")
            synthetic_backends = sorted(synthetic_backends, key=lambda x: extract_model_size(x))
    elif sort_method == 'base-model':
        # Sort by base model family, then by model size
        def sort_by_base_model_and_size(backend_name):
            base_model = extract_base_model(backend_name)
            model_priority = get_model_family_priority(base_model)
            model_size = extract_model_size(backend_name)
            return (model_priority, model_size)
        
        synthetic_backends = sorted(synthetic_backends, key=sort_by_base_model_and_size)
        print(f"[INFO] Sorting by base model family and size")
    else:
        # Default: sort by model size
        synthetic_backends = sorted(synthetic_backends, key=lambda x: extract_model_size(x))
    
    has_persimmon = (raw_df['backend_name'] == 'persimmon').any()
    # Persimmon is a single method without a backend LLM, so it gets its own row right below Human.
    backend_order = ['human'] + (['persimmon'] if has_persimmon else []) + synthetic_backends
    
    # 1. Boxplot comparison across all datasets - GROUPED BY MODEL
    # Use a wider, flatter aspect ratio to reduce vertical stretching.
    num_backends = len(backend_order)
    fig_height = max(7.5, num_backends * 0.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    # Define colors for each PSI type
    psi_colors = {
        'Real': '#4d4d4d',
        **PSI_COLORS,
    }
    
    # Create grouped boxplot with hue by PSI type
    sns.boxplot(
        data=raw_df, 
        y='backend_name', 
        x='mtld', 
        hue='psi_type',
        order=backend_order,
        hue_order=['Real'] + (['persimmon'] if has_persimmon else []) + ['patientpsi', 'roleplaydoh'],
        palette=psi_colors,
        linewidth=1.4,
        fliersize=2.5,
        saturation=0.95,
        ax=ax, 
        gap=0.1
    )
    
    # Add a reference line at the median human MTLD to compare synthetic models.
    human_mtld = raw_df[(raw_df['backend_name'] == 'human') & (raw_df['psi_type'] == 'Real')]['mtld'].dropna()
    if not human_mtld.empty:
        human_median_mtld = human_mtld.median()
        ax.axvline(
            x=human_median_mtld,
            color=psi_colors['Real'],
            linestyle='--',
            linewidth=1.2,
            alpha=0.8,
            zorder=0
        )

    # Create legend for PSI types
    from matplotlib.patches import Patch
    
    # Check which PSI types are actually present
    used_psi_types = raw_df['psi_type'].unique()
    color_legend_elements = []
    for psi_type in ['Real'] + sorted([p for p in used_psi_types if p != 'Real']):
        if psi_type in psi_colors:
            color_legend_elements.append(
                Patch(facecolor=psi_colors[psi_type], label=PSI_LABELS[psi_type])
            )
    # Add legend above the plot
    legend = ax.legend(handles=color_legend_elements, fontsize=22, 
                      bbox_to_anchor=(0.5, 1.0), loc='lower center', 
                      ncol=len(color_legend_elements),
                      frameon=False, handletextpad=0.2, columnspacing=1.0)
    
    # ACL-style formatting
    ax.set_ylabel('', fontsize=24)  # Remove y-axis label
    ax.set_xlabel("MTLD", fontsize=24)
    ax.tick_params(axis='y', which='major', labelsize=21)
    ax.tick_params(axis='x', which='major', labelsize=22)
    ax.grid(axis='x', color='#d9d9d9', linewidth=0.7, alpha=0.6)
    ax.grid(axis='y', visible=False)
    
    plt.tight_layout()
    # Save as both PNG and PDF
    plt.savefig(output_dir / 'boxplot_mtld_all_pairs.png', dpi=300, bbox_inches='tight',
                bbox_extra_artists=[legend], pad_inches=0.0)
    plt.savefig(output_dir / 'boxplot_mtld_all_pairs.pdf', bbox_inches='tight',
                bbox_extra_artists=[legend], pad_inches=0.0)
    plt.close()
    print(f"✓ Boxplot saved: {output_dir / 'boxplot_mtld_all_pairs.png'}")
    print(f"✓ Boxplot saved: {output_dir / 'boxplot_mtld_all_pairs.pdf'}")
    
    # 2. Summary statistics table
    summary_stats = raw_df.groupby('dataset')['mtld'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).round(2)
    
    # Calculate MTLD on concatenated text for each dataset
    if dataset_concatenated_texts:
        mtld_concat_values = {}
        for dataset, concat_text in dataset_concatenated_texts.items():
            mtld_concat = calculate_bidirectional_mtld(concat_text)
            mtld_concat_values[dataset] = round(mtld_concat, 2) if not math.isnan(mtld_concat) else float('nan')
        
        # Add as a new column
        summary_stats['mtld_concatenated'] = summary_stats.index.map(mtld_concat_values)
    else:
        summary_stats['mtld_concatenated'] = pd.Series(dtype=float)

    display_names = {}
    for dataset in summary_stats.index:
        if dataset == 'Real':
            display_names[dataset] = 'Real'
        else:
            if is_persimmon_label(dataset):
                display_names[dataset] = 'Persimmon'
                continue
            for psi, abbrev in PSI_ABBREV.items():
                if dataset.startswith(psi):
                    backend_part = dataset[len(psi):].lstrip('-_')
                    backend_short = shorten_backend_name(backend_part)
                    display_names[dataset] = f"{abbrev}-{backend_short}"
                    break
            if dataset not in display_names:
                display_names[dataset] = dataset
    
    summary_stats.index = summary_stats.index.map(display_names)
    
    # Sort by PSI type, then model size
    summary_stats = summary_stats.loc[sorted(summary_stats.index, key=sort_summary_key)]
    
    # Reorder columns: mtld_concatenated, mean, median, std, min, max, count
    summary_stats = summary_stats[['mtld_concatenated', 'mean', 'median', 'std', 'min', 'max', 'count']]
    
    summary_csv = output_dir / 'mtld_summary_statistics.csv'
    summary_stats.to_csv(summary_csv)
    print(f"✓ Summary statistics saved: {summary_csv}")
    
    # Print summary to console
    print("\n" + "="*100)
    print("MTLD SUMMARY STATISTICS")
    print("="*100)
    print(summary_stats.to_string())
    print()

def calculate_wasserstein_distances(df: pd.DataFrame, output_dir: Path, min_tokens: int = 100, sort_method: str = 'psi-size'):
    """Calculate Wasserstein distances between real and synthetic datasets.
    
    Args:
        df: DataFrame with lexical diversity results
        output_dir: Directory to save the results
        min_tokens: Minimum token threshold for filtering
        sort_method: Sorting method - 'psi-size', 'base-model', or 'psi-base-model'
    """
    print("\n[INFO] Calculating Wasserstein distances...")
    
    # Filter to Raw analysis type and sessions that meet threshold
    raw_df = df[(df['analysis_type'] == 'Raw') & (df['meets_threshold'] == True)].copy()
    
    if raw_df.empty:
        print("[WARNING] No data meeting threshold for Wasserstein distance calculation")
        return
    
    # Get real MTLD values
    real_mtld = raw_df[raw_df['dataset'] == 'Real']['mtld'].dropna().values
    
    if len(real_mtld) == 0:
        print("[WARNING] No real data found for comparison")
        return
    
    print(f"[INFO] Real dataset: {len(real_mtld)} samples")
    
    # Calculate Wasserstein distance for each synthetic dataset
    results = []
    
    # Select sort function based on method
    sort_funcs = {
        'psi-size': sort_key_by_psi_and_size,
        'base-model': sort_key_by_base_model,
        'psi-base-model': sort_key_by_psi_base_model
    }
    sort_func = sort_funcs.get(sort_method, sort_key_by_psi_and_size)
    
    synthetic_datasets = sorted([d for d in raw_df['dataset'].unique() if d != 'Real'], 
                                key=sort_func)
    
    for dataset in synthetic_datasets:
        synth_mtld = raw_df[raw_df['dataset'] == dataset]['mtld'].dropna().values
        
        if len(synth_mtld) == 0:
            print(f"[WARNING] No data for {dataset}")
            continue
        
        # Calculate Wasserstein distance
        wd = wasserstein_distance(real_mtld, synth_mtld)
        
        if is_persimmon_label(dataset):
            results.append({
                'dataset': 'Persimmon',
                'n_samples': len(synth_mtld),
                'wasserstein_distance': round(wd, 4),
                'mean_mtld': round(synth_mtld.mean(), 2),
                'std_mtld': round(synth_mtld.std(), 2)
            })
            print(f"  - {dataset}: WD={wd:.4f}, n={len(synth_mtld)}")
            continue

        # Extract PSI and backend for better labeling
        backend_name = None
        
        for psi in ['patientpsi', 'roleplaydoh']:
            if dataset.startswith(psi):
                backend_part = dataset[len(psi):].lstrip('-_')
                backend_name = shorten_backend_name(backend_part)
                break
        
        results.append({
            'dataset': f"{PSI_ABBREV[psi]}-{safe_dir_name(backend_name)}",
            'n_samples': len(synth_mtld),
            'wasserstein_distance': round(wd, 4),
            'mean_mtld': round(synth_mtld.mean(), 2),
            'std_mtld': round(synth_mtld.std(), 2)
        })
        
        print(f"  - {dataset}: WD={wd:.4f}, n={len(synth_mtld)}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Add real dataset info as reference
    real_info = pd.DataFrame([{
        'dataset': 'Real',
        'n_samples': len(real_mtld),
        'wasserstein_distance': 0.0,
        'mean_mtld': round(real_mtld.mean(), 2),
        'std_mtld': round(real_mtld.std(), 2)
    }])
    
    results_df = pd.concat([real_info, results_df], ignore_index=True)
    
    # Keep the previous raw score for reference.
    results_df['un-normalized'] = (100 - results_df['wasserstein_distance']).clip(lower=0, upper=100)

    # Min-max normalize Wasserstein distance into a 0-100 similarity score.
    wd_min = results_df['wasserstein_distance'].min()
    wd_max = results_df['wasserstein_distance'].max()
    if wd_max == wd_min:
        results_df['final_score'] = 100.0
    else:
        results_df['final_score'] = 100 * (1 - (results_df['wasserstein_distance'] - wd_min) / (wd_max - wd_min))
    
    # Save to CSV
    csv_path = output_dir / 'wasserstein_distances.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n[CSV SAVED] {csv_path}")
    
    # Print summary table
    print("\n" + "="*100)
    print("WASSERSTEIN DISTANCES (comparing to Real dataset)")
    print("="*100)
    print(results_df.to_string(index=False))
    print()
    
    return results_df


def regenerate_plots_from_csv(csv_path: str, output_dir: str, min_tokens: int = 100, sort_method: str = 'psi-size'):
    """Regenerate visualizations from previously saved CSV results.
    
    Args:
        csv_path: Path to the saved lexical_diversity_all_pairs.csv file
        output_dir: Directory to save the regenerated plots
        min_tokens: Minimum token threshold (should match original analysis)
        sort_method: Sorting method - 'psi-size' (sort by model size), 
                     'base-model' (sort by base model family then size), 
                     'psi-base-model' (same as psi-size), 
                     or 'wasserstein' (sort by average Wasserstein distance)
    """
    print(f"\n[INFO] Loading results from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} rows")
    
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Calculate Wasserstein distances
    calculate_wasserstein_distances(df, output_root, min_tokens, sort_method)
    
    # Regenerate visualizations (without concatenated text data)
    create_comparison_visualizations(df, output_root, min_tokens, dataset_concatenated_texts=None, sort_method=sort_method)
    print(f"\n[DONE] Plots regenerated and saved to: {output_root}")



def main():
    parser = argparse.ArgumentParser(
        description='Lexical Diversity Analysis - Compare all PSI-backend pairs against real data'
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/lexical_diversity",
        help="Output directory (default: output/lexical_diversity)"
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=100,
        help="Minimum token count threshold for analysis (default: 100)"
    )
    parser.add_argument(
        "--from-csv",
        type=str,
        help="Path to existing CSV file to regenerate plots from (skips data loading and analysis)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--sort-method",
        type=str,
        choices=['psi-size', 'base-model', 'psi-base-model', 'wasserstein'],
        default='psi-size',
        help="Sorting method for results: 'psi-size' (sort by model size), 'base-model' (sort by base model family, then size), 'psi-base-model' (same as psi-size), or 'wasserstein' (sort by Wasserstein distance, requires wasserstein_distances.csv) (default: psi-size)"
    )
    
    parser.add_argument(
        "--synthetic-dir",
        type=str,
        default=None,
        help="Local folder tree of session_*.json (<psi>/<model>/<dataset>/...) used instead of the HF synthetic data; psi and model are read from the folder names"
    )

    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    dataset_name = config.get('eval', {}).get('hf_dataset', 'hknguyen20/psibench-conv')
    
    if args.from_csv:
        # Regenerate plots from existing CSV
        regenerate_plots_from_csv(
            csv_path=args.from_csv,
            output_dir=args.output_dir,
            min_tokens=args.min_tokens,
            sort_method=args.sort_method
        )
    else:
        # Run full comparison
        compare_all_hf_pairs(
            output_dir=args.output_dir,
            min_tokens=args.min_tokens,
            dataset_name=dataset_name,
            sort_method=args.sort_method,
            synthetic_dir=args.synthetic_dir,
        )


if __name__ == "__main__":
    main()
