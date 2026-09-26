"""
Valence Classification: Classify patient turns as positive, negative, or neutral.

This module uses an LLM-judge to classify each patient turn in therapy conversations
according to emotional valence (positive / negative / neutral).

Resume/skip behavior mirrors emotion_classification.py:
- Real data: skips reclassification only if real_valence_detailed.json exists and
    its conversation count matches the currently eligible real conversations.
- Synthetic per (psi, backend) pair: skips only if the pair's
    *_valence_detailed.json exists and its conversation count matches expected
    count for this run; otherwise that whole pair is reprocessed.

Usage:
# Analyze all HF pairs
python -m psibench.eval.valence_classification \\
  --hf \\
  --batch-size 32 \\
  --turn-threshold 16 \\
  --config configs/default.yaml

# Redraw from saved CSV:
python -m psibench.eval.valence_classification \\
  --csv-file output/valence_analysis \\
  --turn-threshold 16
"""

import argparse
import json
import os
import yaml
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

from tqdm import tqdm
from dotenv import load_dotenv

from litellm import batch_completion

from psibench.prompts.judge_prompt import create_valence_judge_prompt
from psibench.data_loader.main_loader import load_real_dataset, load_synthetic_hf_to_df
from psibench.eval.utils import (
    get_all_psi_backend_pairs,
    sort_key_by_psi_and_size,
    sort_key_by_backend_family_and_size,
    PSI_ABBREV,
    PSI_LABELS,
    shorten_backend_name,
    safe_dir_name,
    extract_model_size,
    get_model_opacity,
    extract_backend_name_from_label,
    assign_backend_markers,
    PSI_COLORS,
    build_conversation_id,
)
from psibench.data_loader.utils import normalize_backend_name

load_dotenv()

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

VALENCE_CATEGORIES = ['positive', 'negative', 'neutral']


class ValenceClassifier:
    """Judge for valence classification (positive / negative / neutral)."""

    def __init__(self, config: Dict[str, Any], debug: bool = False):
        self.config = config
        self.debug = debug
        self.prompt_template = create_valence_judge_prompt()

        judge_config = config.get("eval", {}).get("valence_classifier", {})
        self.model = judge_config.get('model')
        self.temperature = judge_config.get('temperature')

        if judge_config.get("api_base"):
            self.api_base = judge_config.get("api_base")
            self.api_key = "sk-no-key-required"
        else:
            self.api_base = os.getenv("OPENAI_BASE_URL")
            self.api_key = os.getenv("OPENAI_API_KEY")

        if self.debug:
            print(f"[ValenceClassifier] Initialized with model: {self.model}")
            print(f"[ValenceClassifier] Temperature: {self.temperature}")

    def _format_history(self, history: list[Dict[str, str]], num_messages: int = None) -> str:
        formatted = []
        for msg in history:
            if not msg.get("content", "").strip():
                continue
            role = "THERAPIST" if msg["role"] == "user" else "PATIENT"
            content = msg["content"]
            formatted.append(f"{role}: {content}")
            if num_messages is not None and len(formatted) >= num_messages:
                break
        return "\n".join(formatted)

    def classify_turns_batch(self, conversations: List[List[Dict[str, str]]], num_messages: int = 4) -> List[List[Dict[str, Any]]]:
        """Classify valence for all patient turns across multiple conversations in parallel."""
        all_tasks = []
        task_metadata = []

        for conv_idx, conversation in enumerate(conversations):
            patient_turns = []
            for i, msg in enumerate(conversation):
                role = msg.get('role', '').lower()
                if role in ('patient', 'assistant'):
                    history = conversation[:i]
                    patient_turns.append({
                        'turn_index': len(patient_turns),
                        'content': msg['content'],
                        'history': history
                    })

            for turn in patient_turns:
                formatted_history = self._format_history(turn['history'], num_messages=num_messages)

                messages = self.prompt_template.format_messages(
                    history=formatted_history,
                    current_message=turn['content']
                )

                litellm_messages = []
                for i, msg in enumerate(messages):
                    if hasattr(msg, 'type'):
                        role = msg.type
                        if role == 'human':
                            role = 'user'
                        elif role not in ['system', 'assistant', 'user', 'function', 'tool', 'developer']:
                            role = 'system' if i == 0 else 'user'
                    else:
                        role = 'system' if i == 0 else 'user'

                    litellm_messages.append({"role": role, "content": msg.content})

                all_tasks.append(litellm_messages)
                task_metadata.append({
                    'conv_idx': conv_idx,
                    'turn_index': turn['turn_index'],
                    'content': turn['content']
                })

        if not all_tasks:
            return [[] for _ in conversations]

        if self.debug:
            print(f"[DEBUG] Running {len(all_tasks)} valence classification tasks in parallel...")

        responses = batch_completion(
            model=self.model,
            messages=all_tasks,
            temperature=self.temperature,
            api_key=self.api_key,
            api_base=self.api_base,
        )

        results_by_conv = {i: [] for i in range(len(conversations))}

        for idx, (response, metadata) in enumerate(zip(responses, task_metadata)):
            try:
                valence = response.choices[0].message.content.strip().lower()

                if valence not in VALENCE_CATEGORIES:
                    if self.debug:
                        print(f"[WARNING] Invalid valence '{valence}' at task {idx}, defaulting to 'neutral'")
                    valence = 'neutral'

                results_by_conv[metadata['conv_idx']].append({
                    'turn_index': metadata['turn_index'],
                    'content': metadata['content'],
                    'valence': valence
                })

            except Exception as e:
                if self.debug:
                    print(f"[ERROR] Failed to parse response at task {idx}: {e}")

                results_by_conv[metadata['conv_idx']].append({
                    'turn_index': metadata['turn_index'],
                    'content': metadata['content'],
                    'valence': 'neutral'
                })

        return [results_by_conv[i] for i in range(len(conversations))]


def compare_all_hf_pairs(config: Dict[str, Any], output_dir: Path, batch_size: int = 1,
                         num_messages: int = 4, exact_turns: int = None, turn_threshold: int = 12,
                         debug: bool = False):
    """Compare valence distributions across all PSI-backend pairs from HuggingFace against real data."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    judge = ValenceClassifier(config, debug=debug)

    print("\n" + "="*70)
    print("Loading all real conversations (esc, hope, annomi combined)...")
    print("="*70)
    all_real_convs = []
    valid_indices = []

    real_df = load_real_dataset(dataset_type='all')
    for idx, row in real_df.iterrows():
        messages = row["messages"]
        patient_turns = sum(1 for msg in messages if msg.get('role', '').lower() == 'assistant')
        if exact_turns is None or patient_turns == exact_turns:
            all_real_convs.append(messages)
            valid_indices.append(idx)

    print(f"\nTotal real conversations: {len(all_real_convs)}")
    if exact_turns:
        print(f"(Filtered to conversations with exactly {exact_turns} patient turns)")

    real_json = output_root / 'real_valence_detailed.json'
    real_results = []
    if real_json.exists():
        try:
            with open(real_json, 'r', encoding='utf-8') as f:
                existing_real_data = json.load(f)
            if len(existing_real_data) == len(all_real_convs):
                print(f"[SKIP] Real conversations already processed with {len(existing_real_data)} conversations")
                real_results = existing_real_data
            else:
                print(f"[WARNING] Existing real file has {len(existing_real_data)} conversations, expected {len(all_real_convs)}. Reprocessing...")
        except Exception as e:
            print(f"[WARNING] Failed to load existing real file: {e}. Reprocessing...")

    if not real_results:
        print("\n[INFO] Analyzing real conversations...")
        for batch_start in tqdm(range(0, len(all_real_convs), batch_size), desc="Classifying real"):
            batch_end = min(batch_start + batch_size, len(all_real_convs))
            batch_conversations = all_real_convs[batch_start:batch_end]
            batch_ids = valid_indices[batch_start:batch_end]

            batch_results = judge.classify_turns_batch(batch_conversations, num_messages=num_messages)

            for conv_id, conv, classifications in zip(batch_ids, batch_conversations, batch_results):
                total_patient_turns = sum(1 for msg in conv if msg.get('role', '').lower() == 'assistant')

                if total_patient_turns == 0:
                    print(f"[WARNING] Conversation {conv_id} has 0 patient turns, skipping")
                    continue

                valence_counts = Counter(c['valence'] for c in classifications)
                valence_percentages = {v: valence_counts.get(v, 0) / total_patient_turns
                                       for v in VALENCE_CATEGORIES}

                result = {
                    'conversation_id': build_conversation_id(conv_id, is_real=True),
                    'total_patient_turns': total_patient_turns,
                    'classifications': classifications,
                }
                for v in VALENCE_CATEGORIES:
                    result[f'{v}_count'] = valence_counts.get(v, 0)
                    result[f'{v}_pct'] = valence_percentages[v]

                real_results.append(result)

    real_df_out = pd.DataFrame(real_results)
    print(f"[INFO] Analyzed {len(real_df_out)} real conversations")

    dataset_name = config.get('eval', {}).get('hf_dataset', 'hknguyen20/psibench-data')

    print(f"[INFO] Loading all unique PSI-backend pairs from {dataset_name}")
    all_pairs = get_all_psi_backend_pairs(dataset_name=dataset_name)
    print(f"[INFO] Found {len(all_pairs)} unique (psi, backend_llm) pairs")

    all_synthetic_results = {}
    all_synthetic_results_raw = {}

    for psi, backend_llm in sorted(all_pairs, key=lambda x: sort_key_by_psi_and_size(f"{x[0]}-{x[1]}")):
        label = f"{psi}_{backend_llm}"
        normalized_backend = normalize_backend_name(backend_llm)
        safe_label = safe_dir_name(label)

        print(f"\n{'='*70}")
        print(f"Processing: {psi} + {backend_llm}")
        print(f"{'='*70}")

        synth_json = output_root / f'{safe_label}_valence_detailed.json'
        if synth_json.exists():
            try:
                with open(synth_json, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if len(existing_data) == len(all_real_convs):
                    print(f"[SKIP] {label} already processed with {len(existing_data)} conversations")
                    synth_df = pd.DataFrame(existing_data)
                    all_synthetic_results[label] = synth_df
                    all_synthetic_results_raw[label] = existing_data
                    continue
                else:
                    print(f"[WARNING] Existing file has {len(existing_data)} conversations, expected {len(all_real_convs)}. Reprocessing...")
            except Exception as e:
                print(f"[WARNING] Failed to load existing file: {e}. Reprocessing...")

        try:
            df_hf = load_synthetic_hf_to_df(psi=psi, backend_llm=normalized_backend, dataset_name=dataset_name)

            if df_hf.empty:
                print(f"[WARNING] No data found for {label}, skipping")
                continue

            synthetic_convs = []
            if exact_turns is not None and valid_indices:
                for _, row in df_hf.iterrows():
                    session_id = row.get('session_id', None)
                    if session_id in valid_indices:
                        messages = row['messages']
                        synthetic_convs.append((session_id, messages))
                print(f"[INFO] Loaded {len(synthetic_convs)} conversations (filtered by session_id)")
            else:
                for _, row in df_hf.iterrows():
                    conv = row['messages']
                    session_id = row.get('session_id', len(synthetic_convs))
                    synthetic_convs.append((session_id, conv))
                print(f"[INFO] Loaded {len(synthetic_convs)} conversations")

            if not synthetic_convs:
                print(f"[WARNING] No conversations after filtering for {label}, skipping")
                continue

            print(f"[INFO] Analyzing {len(synthetic_convs)} synthetic conversations...")

            synthetic_results = []
            for batch_start in tqdm(range(0, len(synthetic_convs), batch_size),
                                    desc=f"Classifying {label}", leave=False):
                batch_end = min(batch_start + batch_size, len(synthetic_convs))
                batch_items = synthetic_convs[batch_start:batch_end]
                batch_session_ids = [item[0] for item in batch_items]
                batch_conversations = [item[1] for item in batch_items]

                batch_results = judge.classify_turns_batch(batch_conversations, num_messages=num_messages)

                for session_id, conv, classifications in zip(batch_session_ids, batch_conversations, batch_results):
                    total_patient_turns = sum(1 for msg in conv if msg.get('role', '').lower() == 'assistant')

                    if total_patient_turns == 0:
                        print(f"[WARNING] Conversation {build_conversation_id(session_id, psi=psi, backend_llm=normalized_backend)} has 0 patient turns, skipping")
                        continue

                    valence_counts = Counter(c['valence'] for c in classifications)
                    valence_percentages = {v: valence_counts.get(v, 0) / total_patient_turns
                                           for v in VALENCE_CATEGORIES}

                    result = {
                        'conversation_id': build_conversation_id(session_id, psi=psi, backend_llm=normalized_backend),
                        'total_patient_turns': total_patient_turns,
                        'classifications': classifications,
                    }
                    for v in VALENCE_CATEGORIES:
                        result[f'{v}_count'] = valence_counts.get(v, 0)
                        result[f'{v}_pct'] = valence_percentages[v]

                    synthetic_results.append(result)

            synth_df = pd.DataFrame(synthetic_results)
            all_synthetic_results[label] = synth_df
            all_synthetic_results_raw[label] = synthetic_results
            print(f"[INFO] Completed {label}: {len(synth_df)} conversations")

            synth_csv = output_root / f'{safe_label}_valence_summary.csv'
            synth_df.drop(columns=['classifications']).to_csv(synth_csv, index=False)
            print(f"[CSV SAVED] {synth_csv}")

            synth_json = output_root / f'{safe_label}_valence_detailed.json'
            with open(synth_json, 'w', encoding='utf-8') as f:
                json.dump(synthetic_results, f, indent=2, ensure_ascii=False)
            print(f"[JSON SAVED] {synth_json}")

        except Exception as e:
            print(f"[ERROR] Failed to process {label}: {e}")
            continue

    print(f"\n{'='*70}")
    print("Saving real data results...")
    print(f"{'='*70}")

    real_csv = output_root / 'real_valence_summary.csv'
    real_df_out.drop(columns=['classifications']).to_csv(real_csv, index=False)
    print(f"[CSV SAVED] {real_csv}")

    real_json = output_root / 'real_valence_detailed.json'
    with open(real_json, 'w', encoding='utf-8') as f:
        json.dump(real_results, f, indent=2, ensure_ascii=False)
    print(f"[JSON SAVED] {real_json}")

    print("\n[INFO] Creating visualizations...")
    visualize_valence_percentages_by_turn(real_df_out, all_synthetic_results, output_root, turn_threshold=turn_threshold, exact_turns=exact_turns)
    visualize_valence_distributions(real_df_out, all_synthetic_results, output_root, exact_turns=exact_turns)

    print(f"\n[DONE] Multi-pair valence analysis complete. Results saved to: {output_root}")
    print(f"       Analyzed {len(all_synthetic_results)} synthetic datasets")


def get_turn_valence_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract turn-by-turn valence data from analysis results."""
    turn_data = []
    for _, row in df.iterrows():
        conv_id = row['conversation_id']
        classifications = row['classifications']
        for classification in classifications:
            turn_data.append({
                'conversation_id': conv_id,
                'turn_index': classification['turn_index'],
                'valence': classification['valence']
            })
    return pd.DataFrame(turn_data)


def visualize_valence_percentages_by_turn(real_df: pd.DataFrame, all_synthetic_results: Dict[str, pd.DataFrame],
                                          output_dir: Path, turn_threshold: int = 16, exact_turns: int = None,
                                          pre_calculated_percentages: Dict[str, Dict[str, pd.DataFrame]] = None):
    """Create line plots showing percentage of each valence category across turns.

    Layout: 1 row × 3 columns (negative, positive, neutral).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    VALENCE_TO_PLOT = ['negative', 'positive', 'neutral']

    if pre_calculated_percentages is not None:
        print("[INFO] Using pre-calculated percentages from saved file")
        real_percentages = pre_calculated_percentages['real']
        synthetic_percentages = {k: v for k, v in pre_calculated_percentages.items() if k != 'real'}
    else:
        real_turns = get_turn_valence_data(real_df)
        real_turns = real_turns[real_turns['turn_index'] < turn_threshold].copy()

        real_percentages = {}
        for v in VALENCE_TO_PLOT:
            vbt = real_turns.groupby('turn_index').apply(
                lambda x: (x['valence'] == v).sum() / len(x) * 100
            ).reset_index(name='percentage')
            real_percentages[v] = vbt

        synthetic_percentages = {}
        for label in sorted(all_synthetic_results.keys(), key=sort_key_by_psi_and_size):
            synth_df = all_synthetic_results[label]
            synth_turns = get_turn_valence_data(synth_df)
            synth_turns = synth_turns[synth_turns['turn_index'] < turn_threshold].copy()

            synthetic_percentages[label] = {}
            for v in VALENCE_TO_PLOT:
                vbt = synth_turns.groupby('turn_index').apply(
                    lambda x: (x['valence'] == v).sum() / len(x) * 100
                ).reset_index(name='percentage')
                synthetic_percentages[label][v] = vbt

        save_valence_percentages(real_percentages, synthetic_percentages, output_dir, turn_threshold)

    fig, axes = plt.subplots(1, 3, figsize=(21, 4.8))
    axes = axes.flatten()

    backend_by_label = {label: extract_backend_name_from_label(label) for label in all_synthetic_results.keys()}
    sorted_backends = sorted(set(backend_by_label.values()), key=sort_key_by_backend_family_and_size)
    backend_markers = assign_backend_markers(sorted_backends)
    all_model_sizes = [extract_model_size(b) for b in sorted_backends if extract_model_size(b) > 0]

    for idx, v in enumerate(VALENCE_TO_PLOT):
        ax = axes[idx]

        real_data = real_percentages[v]
        ax.plot(real_data['turn_index'], real_data['percentage'],
                linewidth=3.2, alpha=0.9, color='black', marker='o', markersize=8)

        for label in sorted(all_synthetic_results.keys(), key=sort_key_by_psi_and_size):
            synth_data = synthetic_percentages[label][v]
            psi_type = 'patientpsi' if 'patientpsi' in label else 'roleplaydoh'
            color = PSI_COLORS.get(psi_type, '#808080')
            backend = backend_by_label.get(label, 'unknown')
            marker = backend_markers.get(backend, 'x')
            alpha = get_model_opacity(backend, all_model_sizes)

            ax.plot(synth_data['turn_index'], synth_data['percentage'],
                    linewidth=3.0, alpha=alpha, color=color, marker=marker, markersize=8)

        ax.set_xlabel('Turn Index', fontsize=30)
        ax.set_ylabel('Percentage', fontsize=30)
        ax.set_title(v.capitalize(), fontsize=34, fontweight='bold', pad=16)
        ax.set_xlim(-0.5, turn_threshold - 0.5)
        ax.set_ylim(0, 100)
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.grid(True, alpha=0.3)

        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.subplots_adjust(left=0.05, right=0.995, bottom=0.10, top=0.90, wspace=0.32)
    plt.savefig(output_dir / 'valence_percentages_by_turn.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'valence_percentages_by_turn.pdf', bbox_inches='tight')
    plt.close()

    print(f"[PLOT SAVED] {output_dir / 'valence_percentages_by_turn.png'}")
    print(f"[PLOT SAVED] {output_dir / 'valence_percentages_by_turn.pdf'}")


def save_valence_percentages(real_percentages: Dict[str, pd.DataFrame],
                             synthetic_percentages: Dict[str, Dict[str, pd.DataFrame]],
                             output_dir: Path, turn_threshold: int):
    """Save calculated valence percentages to CSV for faster redraw.

    Saves two versions:
    1. Original with all valence categories including neutral
    2. Without neutral, with positive/negative rescaled to sum to 100%
    """
    all_data = []

    for v, df in real_percentages.items():
        for _, row in df.iterrows():
            all_data.append({'dataset': 'real', 'turn_index': int(row['turn_index']),
                             'valence': v, 'percentage': float(row['percentage'])})

    for label, valence_dict in synthetic_percentages.items():
        for v, df in valence_dict.items():
            for _, row in df.iterrows():
                all_data.append({'dataset': label, 'turn_index': int(row['turn_index']),
                                 'valence': v, 'percentage': float(row['percentage'])})

    percentages_df = pd.DataFrame(all_data)
    output_file = output_dir / f'valence_percentages_by_turn_t{turn_threshold}.csv'
    percentages_df.to_csv(output_file, index=False)
    print(f"[CSV SAVED] Valence percentages (with neutral): {output_file}")

    no_neutral_df = percentages_df[percentages_df['valence'] != 'neutral'].copy()
    rescaled_data = []
    for (dataset, turn_idx), group in no_neutral_df.groupby(['dataset', 'turn_index']):
        total = group['percentage'].sum()
        if total > 0:
            scale_factor = 100.0 / total
            for _, row in group.iterrows():
                rescaled_data.append({
                    'dataset': row['dataset'], 'turn_index': int(row['turn_index']),
                    'valence': row['valence'], 'percentage': float(row['percentage'] * scale_factor)
                })
        else:
            for _, row in group.iterrows():
                rescaled_data.append({
                    'dataset': row['dataset'], 'turn_index': int(row['turn_index']),
                    'valence': row['valence'], 'percentage': float(row['percentage'])
                })

    rescaled_df = pd.DataFrame(rescaled_data)
    output_file_no_neutral = output_dir / f'valence_percentages_by_turn_t{turn_threshold}_no_neutral.csv'
    rescaled_df.to_csv(output_file_no_neutral, index=False)
    print(f"[CSV SAVED] Valence percentages (no neutral, rescaled): {output_file_no_neutral}")


def load_valence_percentages(csv_dir: Path, turn_threshold: int) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load pre-calculated valence percentages from CSV."""
    percentages_file = csv_dir / f'valence_percentages_by_turn_t{turn_threshold}.csv'

    if not percentages_file.exists():
        return None

    try:
        df = pd.read_csv(percentages_file)
        required_cols = ['dataset', 'turn_index', 'valence', 'percentage']
        if not all(col in df.columns for col in required_cols):
            print(f"[WARNING] Invalid percentages file format, missing required columns")
            return None

        result = {}
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            result[dataset] = {}
            for v in dataset_df['valence'].unique():
                valence_df = dataset_df[dataset_df['valence'] == v][['turn_index', 'percentage']]
                result[dataset][v] = valence_df.reset_index(drop=True)

        print(f"[INFO] Loaded pre-calculated percentages from {percentages_file}")
        return result

    except Exception as e:
        print(f"[WARNING] Failed to load percentages file: {e}")
        return None


def visualize_valence_distributions(real_df: pd.DataFrame, all_synthetic_results: Dict[str, pd.DataFrame],
                                    output_dir: Path, exact_turns: int = None):
    """Create horizontal stacked bar chart comparing valence distributions across all datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_stats = []

    real_percentages = {}
    for v in VALENCE_CATEGORIES:
        count_col = f'{v}_count'
        real_percentages[v] = real_df[count_col].sum() / real_df['total_patient_turns'].sum()
    dataset_stats.append({'label': 'real', 'display_name': 'Real', **real_percentages})

    for label in sorted(all_synthetic_results.keys(), key=sort_key_by_psi_and_size):
        synth_df = all_synthetic_results[label]
        synth_percentages = {}
        for v in VALENCE_CATEGORIES:
            count_col = f'{v}_count'
            synth_percentages[v] = synth_df[count_col].sum() / synth_df['total_patient_turns'].sum()

        psi_type = 'patientpsi' if 'patientpsi' in label else 'roleplaydoh'
        backend = label.split('_', 1)[1] if '_' in label else label
        short_backend = shorten_backend_name(backend)
        psi_abbrev = PSI_ABBREV.get(psi_type, psi_type)
        display_name = f"{psi_abbrev}-{short_backend}"

        dataset_stats.append({'label': label, 'display_name': display_name, **synth_percentages})

    fig_height = max(6, len(dataset_stats) * 0.4)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    dataset_labels = [d['display_name'] for d in dataset_stats][::-1]

    valence_colors = {
        'positive': '#2ca02c',   # Green
        'negative': '#d62728',   # Red
        'neutral':  '#7f7f7f',   # Gray
    }

    left = [0] * len(dataset_labels)
    for v in VALENCE_CATEGORIES:
        values = [d[v] for d in dataset_stats][::-1]
        ax.barh(dataset_labels, values, left=left, label=v.capitalize(),
                color=valence_colors[v], alpha=0.85)
        left = [l + val for l, val in zip(left, values)]

    for i, dataset in enumerate(reversed(dataset_stats)):
        left_pos = 0
        for v in VALENCE_CATEGORIES:
            value = dataset[v]
            if value >= 0.05:
                ax.text(left_pos + value/2, i, f'{value*100:.0f}%',
                        ha='center', va='center', fontsize=10, fontweight='bold', color='white')
            left_pos += value

    ax.set_xlabel('Proportion', fontsize=18)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), fontsize=14, ncol=3, title='Valence')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'valence_distribution_all_pairs.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'valence_distribution_all_pairs.pdf', bbox_inches='tight')
    plt.close()
    print(f"[PLOT SAVED] {output_dir / 'valence_distribution_all_pairs.png'}")
    print(f"[PLOT SAVED] {output_dir / 'valence_distribution_all_pairs.pdf'}")


def redraw_from_csv(csv_dir: str, output_dir: str, turn_threshold: int = 12, exact_turns: int = None):
    """Redraw visualizations from existing CSV and JSON files."""
    csv_path = Path(csv_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Loading data from {csv_path}...")

    real_json = csv_path / 'real_valence_detailed.json'
    if not real_json.exists():
        raise FileNotFoundError(f"Real data file not found: {real_json}")

    with open(real_json, 'r', encoding='utf-8') as f:
        real_results = json.load(f)

    real_df = pd.DataFrame(real_results)
    print(f"[INFO] Loaded real data: {len(real_df)} conversations")

    all_synthetic_results = {}
    synthetic_csvs = list(csv_path.glob('*_valence_summary.csv'))
    synthetic_csvs = [f for f in synthetic_csvs if not f.name.startswith('real_')]

    for synth_csv in synthetic_csvs:
        label = synth_csv.stem.replace('_valence_summary', '')
        synth_json = csv_path / f'{label}_valence_detailed.json'
        if synth_json.exists():
            with open(synth_json, 'r', encoding='utf-8') as f:
                synth_results = json.load(f)
            synth_df = pd.DataFrame(synth_results)
        else:
            synth_df = pd.read_csv(synth_csv)
            print(f"[WARNING] No detailed JSON found for {label}, using summary CSV only")

        all_synthetic_results[label] = synth_df
        print(f"[INFO] Loaded {label}: {len(synth_df)} conversations")

    print(f"\n[INFO] Found {len(all_synthetic_results)} synthetic datasets")

    pre_calculated = load_valence_percentages(csv_path, turn_threshold)

    if pre_calculated is not None:
        print("[INFO] Creating visualizations from pre-calculated percentages...")
        no_neutral_csv = csv_path / f'valence_percentages_by_turn_t{turn_threshold}_no_neutral.csv'
        if not no_neutral_csv.exists():
            real_percentages = pre_calculated.get('real', {})
            synthetic_percentages = {k: v for k, v in pre_calculated.items() if k != 'real'}
            save_valence_percentages(real_percentages, synthetic_percentages, csv_path, turn_threshold)

        visualize_valence_percentages_by_turn(real_df, all_synthetic_results, output_root,
                                              turn_threshold=turn_threshold, exact_turns=exact_turns,
                                              pre_calculated_percentages=pre_calculated)
    else:
        print("[INFO] Pre-calculated percentages not found, calculating from detailed data...")
        datasets_with_classifications = {
            label: df for label, df in all_synthetic_results.items()
            if 'classifications' in df.columns
        }
        if datasets_with_classifications:
            visualize_valence_percentages_by_turn(real_df, datasets_with_classifications, output_root,
                                                  turn_threshold=turn_threshold, exact_turns=exact_turns)
        else:
            print("[WARNING] No datasets with detailed classifications found, skipping by-turn visualization")

    visualize_valence_distributions(real_df, all_synthetic_results, output_root, exact_turns=exact_turns)

    print(f"\n[DONE] Visualizations regenerated in: {output_root}")


def main():
    """Main function to run valence classification analysis."""
    parser = argparse.ArgumentParser(description='Classify patient turns by emotional valence (positive/negative/neutral)')
    parser.add_argument('--output-dir', type=str, default='output/valence_analysis', help='Output directory')
    parser.add_argument('--hf', action='store_true', help='Load all psi/backend pairs from HF dataset')
    parser.add_argument('--csv-file', type=str, default=None,
                        help='Directory containing CSV/JSON files to redraw graphs from')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file (default: configs/default.yaml)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Number of parallel tasks to run (default: 1)')
    parser.add_argument('--turn-threshold', type=int, default=12,
                        help='Maximum turn index for line plots (default: 12)')
    parser.add_argument('--exact-turns', type=int, default=None,
                        help='Only analyze conversations with exactly this many patient turns')
    parser.add_argument('--num-messages', type=int, default=4,
                        help='Number of previous messages for context (default: 4)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dir_parts = []
    if args.exact_turns:
        dir_parts.append(f"exact_turns_{args.exact_turns}")

    if dir_parts:
        output_dir = Path(args.output_dir) / "_".join(dir_parts)
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    if args.csv_file:
        redraw_from_csv(
            csv_dir=args.csv_file,
            output_dir=output_dir,
            turn_threshold=args.turn_threshold,
            exact_turns=args.exact_turns,
        )
    elif args.hf:
        compare_all_hf_pairs(
            config=config,
            output_dir=output_dir,
            batch_size=args.batch_size,
            num_messages=args.num_messages,
            exact_turns=args.exact_turns,
            turn_threshold=args.turn_threshold,
            debug=args.debug,
        )
    else:
        print("Error: Please provide --hf or --csv-file")
        return

    elapsed = time.time() - start_time
    print(f"\nResults saved to {output_dir}")
    print(f"Total time taken: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
