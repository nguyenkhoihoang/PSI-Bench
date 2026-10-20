"""
Analyze and compare depressive linguistic markers between real and synthetic conversations.

This script compares the frequency of depressive linguistic markers in patient responses
across real depressive patients and different LLM-based patient simulators.

Frequencies are calculated as:
- Raw count: Total occurrences of each marker
- Normalized by tokens: Count per configurable token window (default 100 tokens)
- Normalized by utterances: Average count per patient turn
- Normalized by unique speakers: Average count per conversation

#!/bin/bash
# Example script to run depressive linguistic marker analysis
# Automatically loads all real data (ESC, HOPE, AnnoMI) and all synthetic PSI-backend pairs

# Basic usage - load all data with all metrics
python psibench/eval/depressive_linguistic_markers.py \
    --output-dir output/depressive_markers \
    --metrics all \
    --token-scale 1000 \
    --combined

# Redraw
    python3 -m psibench.eval.depressive_linguistic_markers --csv-file output/depressive_markers --output-dir output/depressive_markers

# Load with specific metric only
python psibench/eval/depressive_linguistic_markers.py \
    --output-dir output/depressive_markers/per_tokens \
    --metrics per_tokens

# Analyze only first 10 turns for all pairs
python psibench/eval/depressive_linguistic_markers.py \
    --max-turns 10 \
    --output-dir output/depressive_markers/first_10_turns \
    --metrics per_tokens

"""

import argparse
import json
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate
import matplotlib.pyplot as plt

from psibench.data_loader.main_loader import load_real_dataset, load_synthetic_hf_to_df
from psibench.eval.utils import (
    extract_patient_messages_by_turn,
    get_all_psi_backend_pairs,
    safe_dir_name,
    PSI_ABBREV,
    PSI_LABELS,
    PSI_COLORS,
    PERSIMMON_COLOR,
    is_persimmon_label,
    sort_key_by_psi_and_size,
    sort_key_by_backend_family_and_size,
    create_display_label,
    shorten_backend_name,
    extract_backend_name_from_label,
    assign_backend_markers,
)
from psibench.data_loader.utils import normalize_backend_name
import yaml

# =============================================================================
# DOTPLOT STYLE DEFAULTS
# =============================================================================

DOTPLOT_FIGSIZE = (18, 6.0)
DOTPLOT_MARKER_SIZE = 500
DOTPLOT_TITLE_FONTSIZE = 29
DOTPLOT_XTICK_LABELSIZE = 25
DOTPLOT_XTICK_LENGTH = 6
DOTPLOT_XTICK_WIDTH = 1.2
DOTPLOT_LEGEND_MARKERSIZE = 18
DOTPLOT_LEGEND_FONTSIZE = 27
DOTPLOT_BACKEND_LEGEND_BBOX_Y = 1.35
DOTPLOT_SIMULATOR_LEGEND_BBOX_Y = 1.15
DOTPLOT_BACKEND_LEGEND_ROWS = 2

# =============================================================================
# DEPRESSIVE LINGUISTIC MARKER TAXONOMY
# =============================================================================
# Based on validated taxonomies from depression research literature

DEPRESSIVE_MARKERS = {
    "absolutist": {
        "patterns": [
            r"\b(absolutely|all|always|complete|completely|constant|constantly)\b",
            r"\b(definitely|entire|ever|every|everyone|everything|full)\b",
            r"\b(must|never|nothing|totally|whole)\b"
        ],
        "case_sensitive": False
    },
    "depressive_words": {
        "patterns": [
            r"\b(depression|collapse|stress|suicide|apastia|anxious|sad|tired)\b",
            r"\b(death|lonely|insomnia|bad|desperate|give up|low|leave)\b",
            r"\b(fear|danger|close|sensitive|lost|shadow|destroy|suspect)\b",
            r"\b(crash|dark|helpless|guilt|negative|frustration|nervous)\b",
            r"\b(melancholy|rubbish|jump|forget|goodbye|cut wrist|edge|haze)\b",
            r"\b(antidepressant)\b"
        ],
        "case_sensitive": False
    },
    "non_fluencies": {
        "patterns": [
            r"\b(uh|um|er|ah|eh|oh|hmm|mm|mmm|hm|huh|mhm)\b",
            r"\b(you know|yknow|y'know|i mean|lets see|let's see)\b",
            r"\.\.\.|…"
        ],
        "case_sensitive": False
    },
    "first_person_singular": {
        "patterns": [
            r"\bI\b", r"\bme\b", r"\bmine\b", r"\bmy\b", r"\bmyself\b"
        ],
        "case_sensitive": False
    },
    "social_pronouns": {
        "patterns": [
            r"\bwe\b", r"\bus\b", r"\bour\b", r"\bours\b", r"\bourselves\b"
        ],
        "case_sensitive": False
    }
}


# =============================================================================
# TEXT PROCESSING & MARKER DETECTION
# =============================================================================

def count_words(text: str) -> int:
    """Count total words in text."""
    return len(text.split())


def count_tokens(text: str) -> int:
    """Approximate token count (roughly 1 token = 4 characters)."""
    return len(text) // 4


def compile_marker_patterns(markers_dict: Dict) -> Dict[str, List[re.Pattern]]:
    """Compile regex patterns for each marker category."""
    compiled = {}
    for marker_name, marker_info in markers_dict.items():
        flags = 0 if marker_info.get("case_sensitive", False) else re.IGNORECASE
        compiled[marker_name] = [
            re.compile(pattern, flags) 
            for pattern in marker_info["patterns"]
        ]
    return compiled


def detect_markers(text: str, compiled_patterns: Dict[str, List[re.Pattern]]) -> Dict[str, int]:
    """
    Detect all markers in text and return counts for each category.
    
    Args:
        text: Input text to analyze
        compiled_patterns: Dictionary of compiled regex patterns
        
    Returns:
        Dictionary mapping marker name to count
    """
    counts = {}
    for marker_name, patterns in compiled_patterns.items():
        total_count = 0
        for pattern in patterns:
            matches = pattern.findall(text)
            total_count += len(matches)
        counts[marker_name] = total_count
    return counts


def extract_patient_messages(messages: List[Dict]) -> List[str]:
    """Extract content of messages from the 'assistant' role (patient)."""
    return [
        msg.get('content', '').strip()
        for msg in messages 
        if msg.get('role') == 'assistant' and msg.get('content', '').strip()
    ]


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_conversations(
    conversations: List[Dict],
    compiled_patterns: Dict[str, List[re.Pattern]],
    max_turns: Optional[int] = None
) -> Dict:
    """
    Analyze all conversations and compute marker statistics.
    
    Args:
        conversations: List of conversation dictionaries
        compiled_patterns: Compiled regex patterns for markers
        max_turns: Optional maximum number of turns to analyze
        
    Returns:
        Dictionary containing:
        - raw_counts: Total count of each marker across all conversations
        - word_counts: Total words analyzed
        - utterance_counts: Total patient utterances
        - conversation_count: Total conversations
        - per_conversation_counts: List of marker counts per conversation
        - marker_details: Detailed breakdown per conversation
    """
    total_raw_counts = Counter()
    total_words = 0
    total_tokens = 0
    total_utterances = 0
    total_messages_with_markers = Counter()
    total_messages_with_any_combined_marker = 0
    conversation_count = len(conversations)
    per_conversation_counts = []
    marker_details = []
    
    for conv_idx, conv in enumerate(conversations):
        patient_messages = extract_patient_messages(conv.get('messages', []))
        
        # Calculate actual patient turns and apply limit
        actual_patient_turns = len(patient_messages)
        if max_turns is not None:
            effective_max_turns = min(max_turns, actual_patient_turns)
            patient_messages = patient_messages[:effective_max_turns]
        
        if not patient_messages:
            continue
        
        # Aggregate all patient messages for this conversation
        full_text = " ".join(patient_messages)
        
        # Count markers in full conversation
        conv_marker_counts = detect_markers(full_text, compiled_patterns)
        
        # Count messages containing each marker type
        conv_messages_with_markers = Counter()
        conv_messages_with_any_combined_marker = 0
        combined_marker_names = ['absolutist', 'depressive_words', 'non_fluencies']
        for message in patient_messages:
            # Note: message is already preprocessed from extract_patient_messages
            message_markers = detect_markers(message, compiled_patterns)
            for marker_name, count in message_markers.items():
                if count > 0:
                    conv_messages_with_markers[marker_name] += 1
            if any(message_markers.get(marker, 0) > 0 for marker in combined_marker_names):
                conv_messages_with_any_combined_marker += 1
        
        # Count words, tokens, and utterances
        conv_words = count_words(full_text)
        conv_tokens = count_tokens(full_text)
        conv_utterances = len(patient_messages)
        
        # Update totals
        total_raw_counts.update(conv_marker_counts)
        total_words += conv_words
        total_tokens += conv_tokens
        total_utterances += conv_utterances
        total_messages_with_markers.update(conv_messages_with_markers)
        total_messages_with_any_combined_marker += conv_messages_with_any_combined_marker
        
        # Store per-conversation data
        per_conversation_counts.append(conv_marker_counts)
        marker_details.append({
            'conversation_id': conv_idx,
            'words': conv_words,
            'utterances': conv_utterances,
            'markers': conv_marker_counts
        })
    
    return {
        'raw_counts': dict(total_raw_counts),
        'word_counts': total_words,
        'token_counts': total_tokens,
        'utterance_counts': total_utterances,
        'messages_with_markers': dict(total_messages_with_markers),
        'messages_with_any_combined_marker': total_messages_with_any_combined_marker,
        'conversation_count': conversation_count,
        'per_conversation_counts': per_conversation_counts,
        'marker_details': marker_details
    }


def compute_normalized_metrics(
    analysis_results: Dict,
    token_scale: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    Compute normalized metrics from analysis results.
    
    Returns:
        Dictionary with keys:
        - 'per_tokens': Markers per token_scale tokens
        - 'percentage_messages': Percentage of patient messages containing each marker
    """
    raw_counts = analysis_results['raw_counts']
    token_count = analysis_results['token_counts']
    utterance_count = analysis_results['utterance_counts']
    messages_with_markers = analysis_results['messages_with_markers']
    
    metrics = {}
    combined_marker_names = ['absolutist', 'depressive_words', 'non_fluencies']
    combined_raw_count = sum(raw_counts.get(marker, 0) for marker in combined_marker_names)
    
    # Per token_scale tokens (exclude first_person_singular and social_pronouns from display)
    if token_count > 0:
        metrics['per_tokens'] = {
            marker: (count / token_count) * token_scale
            for marker, count in raw_counts.items()
            if marker not in ['first_person_singular', 'social_pronouns']
        }
        metrics['per_tokens']['combined_ptk'] = (combined_raw_count / token_count) * token_scale
    else:
        metrics['per_tokens'] = {
            marker: 0.0 for marker in raw_counts 
            if marker not in ['first_person_singular', 'social_pronouns']
        }
        metrics['per_tokens']['combined_ptk'] = 0.0
    
    # Calculate self-focus ratio (first_person_singular / social_pronouns)
    first_person = raw_counts.get('first_person_singular', 0)
    social = raw_counts.get('social_pronouns', 0)
    
    if social > 0:
        self_focus_ratio = first_person / social
    elif first_person > 0:
        self_focus_ratio = float('inf')  # Only self-focus, no social
    else:
        self_focus_ratio = 0.0  # Neither present
    
    metrics['per_tokens']['self_focus_ratio'] = self_focus_ratio
    
    # Percentage of messages containing each marker (exclude first_person_singular and social_pronouns)
    if utterance_count > 0:
        metrics['percentage_messages'] = {
            marker: (messages_with_markers.get(marker, 0) / utterance_count) * 100
            for marker in raw_counts.keys()
            if marker not in ['first_person_singular', 'social_pronouns']
        }
        combined_messages_count = analysis_results.get('messages_with_any_combined_marker', 0)
        metrics['percentage_messages']['combined_msg'] = (
            combined_messages_count / utterance_count
        ) * 100
    else:
        metrics['percentage_messages'] = {
            marker: 0.0 for marker in raw_counts
            if marker not in ['first_person_singular', 'social_pronouns']
        }
        metrics['percentage_messages']['combined_msg'] = 0.0
    
    # Add self-focus ratio to percentage_messages as well for consistency
    metrics['percentage_messages']['self_focus_ratio'] = self_focus_ratio
    
    return metrics


def compute_statistical_measures(per_conversation_counts: List[Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    """
    Compute statistical measures (mean, std, median) for each marker across conversations.
    
    Args:
        per_conversation_counts: List of marker counts per conversation
        
    Returns:
        Dictionary mapping marker name to statistics (mean, std, median, min, max)
    """
    if not per_conversation_counts:
        return {}
    
    # Get all marker names
    all_markers = set()
    for counts in per_conversation_counts:
        all_markers.update(counts.keys())
    
    stats = {}
    for marker in all_markers:
        values = [counts.get(marker, 0) for counts in per_conversation_counts]
        stats[marker] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    return stats


# =============================================================================
# COMPARISON & VISUALIZATION
# =============================================================================

def create_comparison_dataframe(
    results_dict: Dict[str, Dict],
    metric_type: str = 'per_tokens',
    token_scale: int = 100
) -> pd.DataFrame:
    """
    Create a comparison DataFrame for visualization.
    
    Args:
        results_dict: Dictionary mapping dataset name to analysis results
        metric_type: Type of metric ('per_tokens', 'percentage_messages')
        
    Returns:
        DataFrame with markers as rows and datasets as columns
    """
    data = {}
    for dataset_name, results in results_dict.items():
        metrics = compute_normalized_metrics(results, token_scale)
        data[dataset_name] = metrics[metric_type]
    
    df = pd.DataFrame(data)
    return df


def create_depressive_distance_summary(corr_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Create summary CSV with human and synthetic distribution stats.

    Rows:
    - ptk_all
    - msg_all
    - ptk_dist
    - msg_diff
    - combined_dist
    """
    required_cols = ['dataset', 'combined_ptk', 'combined_msg', 'ptk_distance', 'msg_difference']
    missing_cols = [col for col in required_cols if col not in corr_df.columns]
    if missing_cols:
        raise ValueError(
            "Missing required columns for summary CSV: "
            + ", ".join(missing_cols)
            + ". Run with --combined to generate distance columns."
        )

    if 'dataset' not in corr_df.columns or 'Real' not in set(corr_df['dataset']):
        raise ValueError("Summary CSV requires a Real row in depressive distance data.")

    real_row = corr_df[corr_df['dataset'] == 'Real'].iloc[0]
    synth_df = corr_df[corr_df['dataset'] != 'Real'].copy()

    synth_df['combined_dist'] = (synth_df['ptk_distance'] + synth_df['msg_difference']) / 2.0

    row_specs = [
        ('ptk_all', 'combined_ptk', float(real_row['combined_ptk'])),
        ('msg_all', 'combined_msg', float(real_row['combined_msg'])),
        ('ptk_dist', 'ptk_distance', 0.0),
        ('msg_diff', 'msg_difference', 0.0),
        ('combined_dist', 'combined_dist', 0.0),
    ]

    summary_rows = []
    for metric_name, column_name, human_value in row_specs:
        values = pd.to_numeric(synth_df[column_name], errors='coerce').dropna()
        summary_rows.append(
            {
                'metric': metric_name,
                'Human': human_value,
                'Synthetic_Min': float(values.min()) if len(values) else np.nan,
                'Synthetic_Mean': float(values.mean()) if len(values) else np.nan,
                'Synthetic_Median': float(values.median()) if len(values) else np.nan,
                'Synthetic_Max': float(values.max()) if len(values) else np.nan,
                'Synthetic_STD': float(values.std(ddof=0)) if len(values) else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path, index=False)
    return summary_df


def visualize_combined_dot_plots(corr_df: pd.DataFrame, output_dir: Path) -> None:
    """Create horizontal 1D dot plots for combined_ptk and combined_msg.

    - Panel 1: per-1000-token combined marker rate (x-axis)
    - Panel 2: message prevalence (%) (x-axis)
    Identity is encoded via color/marker + legend; all points are placed at y=0.
    """
    required_cols = ['dataset', 'display_name', 'combined_ptk', 'combined_msg']
    missing_cols = [col for col in required_cols if col not in corr_df.columns]
    if missing_cols:
        raise ValueError(
            "Missing required columns for combined dot plots: "
            + ", ".join(missing_cols)
        )

    plot_df = corr_df.copy()
    real_df = plot_df[plot_df['dataset'] == 'Real']
    synth_df = plot_df[plot_df['dataset'] != 'Real'].copy()
    synth_df = synth_df.sort_values('dataset', key=lambda x: x.map(sort_key_by_psi_and_size))
    plot_df = pd.concat([real_df, synth_df], ignore_index=True)

    backend_by_dataset = {
        row['dataset']: extract_backend_name_from_label(row['dataset'])
        for _, row in synth_df.iterrows()
        if not is_persimmon_label(row['dataset'])
    }
    sorted_backends = sorted(set(backend_by_dataset.values()), key=sort_key_by_backend_family_and_size)
    backend_markers = assign_backend_markers(sorted_backends)

    fig, axes = plt.subplots(2, 1, figsize=DOTPLOT_FIGSIZE)
    metric_specs = [
        ('combined_ptk', 'Marker Rates', [5, 10, 15, 20, 25]),
        ('combined_msg', 'Marker Prevalences (%)', [35, 50, 65, 80, 95]),
        # ('combined_ptk', 'Number of markers per 1000 tokens',''),
        # ('combined_msg', 'Percentage of Messages with marker(s)',''),
    ]

    used_psi_types = set()
    used_backends = set()

    for ax, (metric_col, title, x_ticks) in zip(axes, metric_specs):

        values = plot_df[metric_col].astype(float).to_numpy()
        x_min = float(np.min(values))
        x_max = float(np.max(values))
        pad = max((x_max - x_min) * 0.12, 0.05)
        left = x_min - pad
        right = x_max + pad

        ticks = np.array(x_ticks, dtype=int)
        # Fixed ticks keep the paper layout; extend them only if a point (e.g. Persimmon) falls outside.
        if ticks.size > 1:
            step = int(ticks[1] - ticks[0])
            slack = 0.05 * float(ticks[-1] - ticks[0])  # the padding the axis already has
            while x_min < ticks[0] - slack:
                ticks = np.insert(ticks, 0, ticks[0] - step)
            while x_max > ticks[-1] + slack:
                ticks = np.append(ticks, ticks[-1] + step)
        if ticks.size == 0:
            tick_step = max(1, int(np.ceil((right - left) / 4.0)))
            ticks = np.arange(int(np.floor(left)), int(np.ceil(right)) + 1, tick_step)

        real_value = float(real_df.iloc[0][metric_col]) if not real_df.empty else None
        # if real_value is not None:
        #     ax.axvline(real_value, color='black', linestyle='--', linewidth=1.5, alpha=0.45)

        ax.axhline(0.0, color='black', linewidth=1.8, alpha=0.75, zorder=1)

        for idx, row in plot_df.iterrows():
            dataset = row['dataset']
            value = row[metric_col]

            if dataset == 'Real':
                color = 'black'
                marker = 'o'
                size = DOTPLOT_MARKER_SIZE
            elif is_persimmon_label(dataset):
                color = PERSIMMON_COLOR
                marker = 'o'
                size = DOTPLOT_MARKER_SIZE
                used_psi_types.add('persimmon')
            else:
                psi_type = 'patientpsi' if str(dataset).startswith('patientpsi') else 'roleplaydoh'
                backend = backend_by_dataset.get(dataset, 'unknown')
                color = PSI_COLORS.get(psi_type, '#808080')
                marker = backend_markers.get(backend, 'o')
                size = DOTPLOT_MARKER_SIZE
                used_psi_types.add(psi_type)
                used_backends.add(backend)

            ax.scatter(value, 0.0, color=color, marker=marker, s=size, alpha=0.9, zorder=3)

        ax.set_title(title, fontsize=DOTPLOT_TITLE_FONTSIZE, fontweight='bold')
        ax.set_xlabel('', fontsize=20)
        tick_range = float(ticks[-1] - ticks[0])
        tick_padding = tick_range * 0.05
        ax.set_xlim(float(ticks[0]) - tick_padding, float(ticks[-1]) + tick_padding)
        ax.set_xticks(ticks.astype(int))
        ax.tick_params(
            axis='x',
            labelsize=DOTPLOT_XTICK_LABELSIZE,
            length=DOTPLOT_XTICK_LENGTH,
            width=DOTPLOT_XTICK_WIDTH,
        )
        ax.set_yticks([])
        # ax.set_ylim(-0.001, 0.001)
        # ax.set_ylim(-1, 1)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.grid(axis='x', alpha=0.8, linewidth=1.5)

    from matplotlib.lines import Line2D

    backend_legend_elements = []
    for backend in sorted(used_backends, key=sort_key_by_backend_family_and_size):
        backend_legend_elements.append(
            Line2D([0], [0], color='gray', marker=backend_markers.get(backend, 'o'),
                   linestyle='None', markersize=DOTPLOT_LEGEND_MARKERSIZE,
                   label=shorten_backend_name(backend))
        )

    simulator_legend_elements = [
        Line2D([0], [0], color='black', marker='o', linestyle='None',
               markersize=DOTPLOT_LEGEND_MARKERSIZE, label=PSI_LABELS['Real'])
    ]
    for psi_type in sorted(used_psi_types):
        simulator_legend_elements.append(
            Line2D([0], [0], color=PSI_COLORS.get(psi_type, '#808080'), marker='o',
                   linestyle='None', markersize=DOTPLOT_LEGEND_MARKERSIZE,
                   label=PSI_LABELS.get(psi_type, psi_type))
        )

    extra_artists = []
    if backend_legend_elements:
        backend_legend_columns = max(
            1,
            (len(backend_legend_elements) + DOTPLOT_BACKEND_LEGEND_ROWS - 1)
            // DOTPLOT_BACKEND_LEGEND_ROWS,
        )
        backend_legend = fig.legend(
            handles=backend_legend_elements,
            bbox_to_anchor=(0.5, DOTPLOT_BACKEND_LEGEND_BBOX_Y),
            loc='upper center',
            ncol=backend_legend_columns,
            frameon=False,
            fontsize=DOTPLOT_LEGEND_FONTSIZE,
        )
        extra_artists.append(backend_legend)

    simulator_legend = fig.legend(
        handles=simulator_legend_elements,
        bbox_to_anchor=(0.5, DOTPLOT_SIMULATOR_LEGEND_BBOX_Y),
        loc='upper center',
        ncol=max(2, len(simulator_legend_elements)),
        frameon=False,
        fontsize=DOTPLOT_LEGEND_FONTSIZE,
    )
    extra_artists.append(simulator_legend)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.subplots_adjust(hspace=0.8)

    output_png = output_dir / 'depressive_combined_dotplot.png'
    output_pdf = output_dir / 'depressive_combined_dotplot.pdf'
    fig.savefig(output_png, dpi=300, bbox_inches='tight', bbox_extra_artists=extra_artists)
    fig.savefig(output_pdf, bbox_inches='tight', bbox_extra_artists=extra_artists)
    plt.close(fig)

    print(f"[PLOT SAVED] {output_png}")
    print(f"[PLOT SAVED] {output_pdf}")


def compute_profile_vectors(results: Dict, token_scale: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rate and prevalence vectors for a patient type.
    
    Rate vector (ptk): [absolutist_ptk, depressive_ptk, nonfluency_ptk, combined_ptk]
    Prevalence vector (msg): [absolutist_msg, depressive_msg, nonfluency_msg, combined_msg]
    
    Args:
        results: Analysis results for a dataset
        token_scale: Scale factor for token normalization
        
    Returns:
        Tuple of (rate_vector, prevalence_vector) as numpy arrays
    """
    metrics = compute_normalized_metrics(results, token_scale)
    
    # Extract relevant markers (excluding self_focus_ratio)
    per_tokens = metrics['per_tokens']
    per_messages = metrics['percentage_messages']
    
    # Rate vector: per-token metrics
    rate_vector = np.array([
        per_tokens.get('absolutist', 0.0),
        per_tokens.get('depressive_words', 0.0),
        per_tokens.get('non_fluencies', 0.0),
        per_tokens.get('combined_ptk', 0.0)
    ])
    
    # Prevalence vector: percentage of messages metrics
    prevalence_vector = np.array([
        per_messages.get('absolutist', 0.0),
        per_messages.get('depressive_words', 0.0),
        per_messages.get('non_fluencies', 0.0),
        per_messages.get('combined_msg', 0.0)
    ])
    
    return rate_vector, prevalence_vector


def compute_depressive_distance(
    results_dict: Dict[str, Dict],
    token_scale: int = 100,
    combined_only: bool = False,
) -> pd.DataFrame:
    """
    Compute profile similarity metrics between synthetic and real profiles.
    
    Default mode computes:
    - Pearson correlations for rate vector (ptk) and prevalence vector (msg)
    - Euclidean distances for rate vector (ptk) and prevalence vector (msg)
    - Final score: average of the two Euclidean distances

    Combined-only mode (`combined_only=True`) computes:
    - ptk_distance: % distance of combined_ptk from human (scaled to 0-100)
    - msg_difference: absolute percentage-point difference in combined_msg from human (0-100)
    - final_score: 100 - average(ptk_distance, msg_difference)
    
    Args:
        results_dict: Dictionary mapping dataset name to analysis results
        token_scale: Scale factor for token normalization
        
    Returns:
        DataFrame with profile metrics and final scores for each synthetic dataset
    """
    if 'Real' not in results_dict:
        raise ValueError("Real dataset not found in results_dict")
    
    # Compute real patient profile vectors
    real_rate_vector, real_prevalence_vector = compute_profile_vectors(results_dict['Real'], token_scale)
    real_combined_ptk = float(real_rate_vector[3])
    real_combined_msg = float(real_prevalence_vector[3])

    def pct_distance_from_human(synth_value: float, human_value: float) -> float:
        if human_value <= 0:
            return 0.0 if synth_value <= 0 else 100.0
        return min(100.0, abs(synth_value - human_value) / human_value * 100.0)
    
    correlations = []
    rate_names = ['absolutist_ptk', 'depressive_ptk', 'nonfluency_ptk', 'combined_ptk']
    prevalence_names = ['absolutist_msg', 'depressive_msg', 'nonfluency_msg', 'combined_msg']
    
    for dataset_name, results in results_dict.items():
        if dataset_name == 'Real':
            continue
        
        # Compute synthetic profile vectors
        synth_rate_vector, synth_prevalence_vector = compute_profile_vectors(results, token_scale)
        
        if combined_only:
            synth_combined_ptk = float(synth_rate_vector[3])
            synth_combined_msg = float(synth_prevalence_vector[3])

            ptk_distance = pct_distance_from_human(synth_combined_ptk, real_combined_ptk)
            msg_difference = abs(synth_combined_msg - real_combined_msg)
            final_score = 100.0 - ((ptk_distance + msg_difference) / 2.0)

            correlations.append({
                'dataset': dataset_name,
                'ptk_distance': ptk_distance,
                'msg_difference': msg_difference,
                'final_score': final_score,
                'combined_ptk': synth_combined_ptk,
                'combined_msg': synth_combined_msg,
                **{name: val for name, val in zip(rate_names, synth_rate_vector)},
                **{name: val for name, val in zip(prevalence_names, synth_prevalence_vector)},
            })
        else:
            # Compute Pearson correlations for each vector type
            rate_correlation, _ = stats.pearsonr(real_rate_vector, synth_rate_vector)
            prevalence_correlation, _ = stats.pearsonr(real_prevalence_vector, synth_prevalence_vector)

            # Compute average correlation
            avg_correlation = (rate_correlation + prevalence_correlation) / 2

            # Compute Euclidean distances
            rate_distance = np.linalg.norm(synth_rate_vector - real_rate_vector)
            prevalence_distance = np.linalg.norm(synth_prevalence_vector - real_prevalence_vector)

            # Compute final score (average of two distances)
            final_score = (rate_distance + prevalence_distance) / 2

            correlations.append({
                'dataset': dataset_name,
                'pearson_r_ptk': rate_correlation,
                'pearson_r_msg': prevalence_correlation,
                'pearson_r_avg': avg_correlation,
                'euclidean_dist_ptk': rate_distance,
                'euclidean_dist_msg': prevalence_distance,
                'final_score': final_score,
                **{name: val for name, val in zip(rate_names, synth_rate_vector)},
                **{name: val for name, val in zip(prevalence_names, synth_prevalence_vector)}
            })
    
    # Create DataFrame and sort by PSI type and model size (not by correlation)
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('dataset', key=lambda x: x.map(sort_key_by_psi_and_size))
    
    # Add real profile as reference row
    if combined_only:
        real_row = {
            'dataset': 'Real',
            'ptk_distance': 0.0,
            'msg_difference': 0.0,
            'final_score': 100.0,
            'combined_ptk': real_combined_ptk,
            'combined_msg': real_combined_msg,
        }
        real_row.update({name: val for name, val in zip(rate_names, real_rate_vector)})
        real_row.update({name: val for name, val in zip(prevalence_names, real_prevalence_vector)})
    else:
        real_row = {
            'dataset': 'Real',
            'pearson_r_ptk': 1.0,
            'pearson_r_msg': 1.0,
            'pearson_r_avg': 1.0,
            'euclidean_dist_ptk': 0.0,
            'euclidean_dist_msg': 0.0,
            'final_score': 0.0
        }
        real_row.update({name: val for name, val in zip(rate_names, real_rate_vector)})
        real_row.update({name: val for name, val in zip(prevalence_names, real_prevalence_vector)})

    real_df = pd.DataFrame([real_row])
    
    # Combine with real at top
    result_df = pd.concat([real_df, corr_df], ignore_index=True)
    
    return result_df


def redraw_from_csv(csv_file: Path, output_dir: Path) -> None:
    """Redraw combined depressive marker plot from saved CSV without recomputation.

    Args:
        csv_file: Path to `depressive_distance.csv` or directory containing it
        output_dir: Directory to save regenerated plots
    """
    csv_path = csv_file
    if csv_file.is_dir():
        csv_path = csv_file / "depressive_distance.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV file: {csv_path}")

    corr_df = pd.read_csv(csv_path)

    if 'dataset' not in corr_df.columns:
        raise ValueError("CSV must include 'dataset' column for redraw mode")

    if 'display_name' not in corr_df.columns:
        corr_df['display_name'] = corr_df['dataset'].apply(
            lambda x: 'Real' if x == 'Real' else create_display_label(x)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    visualize_combined_dot_plots(corr_df, output_dir)

    if {'combined_ptk', 'combined_msg', 'ptk_distance', 'msg_difference', 'combined_dist'}.issubset(corr_df.columns):
        summary_csv_path = output_dir / "depressive_distance_summary.csv"
        create_depressive_distance_summary(corr_df, summary_csv_path)
        print(f"[CSV SAVED] {summary_csv_path}")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze depressive linguistic markers in patient conversations. "
                    "Automatically loads all real data and all synthetic PSI-backend pairs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/depressive_markers"),
        help="Directory to save output files"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum number of patient turns to analyze per conversation (default: 20)"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        choices=["per_tokens", "percentage_messages", "all"],
        default=["per_tokens", "percentage_messages"],
        help="Normalization metrics to compute and visualize"
    )
    parser.add_argument(
        "--token-scale",
        type=int,
        default=100,
        help="Scale factor for token-normalized metrics (e.g., 100 for per-100 tokens, 1000 for per-1000 tokens)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--synthetic-dir",
        type=str,
        default=None,
        help="Local folder tree of session_*.json (<psi>/<model>/<dataset>/...) used instead of the HF synthetic data; psi and model are read from the folder names"
    )
    parser.add_argument(
        "--csv-file",
        type=Path,
        default=None,
        help=(
            "Redraw combined plots from saved depressive_distance.csv without recomputation. "
            "Can be a CSV file path or a directory containing depressive_distance.csv."
        )
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help=(
            "Use combined-marker-only profile scoring in depressive_distance.csv: "
            "ptk_distance is human-relative %%, msg_difference is absolute %%-point gap, "
            "and final_score = 100 - avg(distance)."
        )
    )
    args = parser.parse_args()

    # Redraw-only mode from saved CSV
    if args.csv_file is not None:
        redraw_from_csv(args.csv_file, args.output_dir)
        print(f"\n[DONE] Redraw complete. Results saved to: {args.output_dir}")
        return
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    dataset_name = config.get('eval', {}).get('hf_dataset', 'hknguyen20/psibench-data')
    
    # Expand "all" metrics
    if "all" in args.metrics:
        args.metrics = ["per_tokens", "percentage_messages"]
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compile patterns
    print("[INFO] Compiling marker patterns...")
    compiled_patterns = compile_marker_patterns(DEPRESSIVE_MARKERS)
    
    # Load and analyze data
    results_dict = {}
    
    # Load all real data
    print("\n[INFO] Loading all real data (ESC, HOPE, AnnoMI)...")
    try:
        df = load_real_dataset("all")
        real_conversations = df.to_dict('records')
        print(f"[INFO] Loaded {len(real_conversations)} real conversations")
    except Exception as e:
        print(f"[ERROR] Could not load real data: {e}")
        return
    
    print(f"[INFO] Analyzing {len(real_conversations)} real conversations...")
    results_dict['Real'] = analyze_conversations(real_conversations, compiled_patterns, args.max_turns)
    
    # Load all synthetic PSI-backend pairs
    print("\n[INFO] Loading all available PSI-backend pairs...")
    all_pairs = get_all_psi_backend_pairs(dataset_name=dataset_name, synthetic_dir=args.synthetic_dir)
    print(f"[INFO] Found {len(all_pairs)} unique PSI-backend pairs")
    
    # Load synthetic data for each pair
    for psi, backend_llm in sorted(all_pairs):
        # Normalize backend_llm name to match HF dataset
        normalized_backend = normalize_backend_name(backend_llm)
        label = f"{psi}-{safe_dir_name(normalized_backend)}"
        
        print(f"\n[INFO] Loading synthetic data for {label}...")
        
        # Load synthetic conversations
        try:
            df = load_synthetic_hf_to_df(psi=psi, backend_llm=normalized_backend, dataset_name=dataset_name, synthetic_dir=args.synthetic_dir)
            if df.empty:
                print(f"[WARNING] No data found for {label}")
                continue
            
            synth_conversations = df.to_dict('records')
            print(f"[INFO] Analyzing {len(synth_conversations)} synthetic conversations ({label})...")
            results_dict[label] = analyze_conversations(synth_conversations, compiled_patterns, args.max_turns)
        except Exception as e:
            print(f"[ERROR] Failed to load {label}: {e}")
            continue

    # Compute and display results for each metric
    for metric_type in args.metrics:        

        comparison_df = create_comparison_dataframe(results_dict, metric_type, args.token_scale)
        
        # Sort columns: Real first, then by PSI type and model size
        sorted_columns = ['Real'] + sorted(
            [col for col in comparison_df.columns if col != 'Real'],
            key=sort_key_by_psi_and_size
        )
        comparison_df = comparison_df[sorted_columns]
        
        # Create display labels for column names
        column_mapping = {'Real': 'Real'}
        for col in sorted_columns:
            if col != 'Real':
                column_mapping[col] = create_display_label(col)
        comparison_df = comparison_df.rename(columns=column_mapping)
        
        if metric_type == "per_tokens":
            csv_path = args.output_dir / f"markers_per_{args.token_scale}_tokens.csv"
        else:
            csv_path = args.output_dir / f"markers_{metric_type}.csv"
        comparison_df.to_csv(csv_path)
        print(f"[CSV SAVED] {csv_path}")
    
    # Save raw counts
    raw_counts_data = {
        dataset: results['raw_counts']
        for dataset, results in results_dict.items()
    }
    raw_df = pd.DataFrame(raw_counts_data)
    
    # Sort columns: Real first, then by PSI type and model size
    sorted_columns = ['Real'] + sorted(
        [col for col in raw_df.columns if col != 'Real'],
        key=sort_key_by_psi_and_size
    )
    raw_df = raw_df[sorted_columns]
    
    # Create display labels for column names
    column_mapping = {'Real': 'Real'}
    for col in sorted_columns:
        if col != 'Real':
            column_mapping[col] = create_display_label(col)
    raw_df = raw_df.rename(columns=column_mapping)
    
    raw_csv_path = args.output_dir / "markers_raw_counts.csv"
    raw_df.to_csv(raw_csv_path)
    print(f"[CSV SAVED] {raw_csv_path}")
    
    # Compute and save profile correlations
    print("\n[INFO] Computing profile correlations...")
    corr_df = compute_depressive_distance(
        results_dict,
        args.token_scale,
        combined_only=args.combined,
    )
    
    # Add display labels
    corr_df['display_name'] = corr_df['dataset'].apply(
        lambda x: 'Real' if x == 'Real' else create_display_label(x)
    )
    # Reorder columns to have display_name first
    cols = ['display_name'] + [col for col in corr_df.columns if col != 'display_name']
    corr_df = corr_df[cols]
    
    corr_csv_path = args.output_dir / "depressive_distance.csv"
    corr_df.to_csv(corr_csv_path, index=False)
    print(f"[CSV SAVED] {corr_csv_path}")

    if args.combined:
        summary_csv_path = args.output_dir / "depressive_distance_summary.csv"
        create_depressive_distance_summary(corr_df, summary_csv_path)
        print(f"[CSV SAVED] {summary_csv_path}")
        visualize_combined_dot_plots(corr_df, args.output_dir)
    
    print(f"\n[DONE] Analysis complete. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
