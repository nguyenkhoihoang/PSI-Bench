"""
Emotion Classification: Classify patient turns using Plutchik's 8 basic emotions.

This module uses an LLM-judge to classify each patient turn in therapy conversations
according to Plutchik's 8 basic emotions (anger, disgust, fear, joy, sadness, surprise, 
anticipation, trust) plus neutral.

Resume/skip behavior:
- Real data: skips reclassification only if real_emotion_detailed.json exists and
    its conversation count matches the currently eligible real conversations.
- Synthetic per (psi, backend) pair: skips only if the pair's
    *_emotion_detailed.json exists and its conversation count matches expected
    count for this run; otherwise that whole pair is reprocessed.
- No partial resume within a pair/conversation. Skip is dataset-level.

Usage:
# Analyze all HF pairs
python -m psibench.eval.emotion_classification \
  --hf \
  --batch-size 32 \
    --exact-turns 12 \
  --config configs/default.yaml

# Classify all, but limit the visualization:
python -m psibench.eval.emotion_classification \
    --hf \
    --batch-size 384 \
    --turn-threshold 16 \
    --config configs/default.yaml
    
# Redraw from saved CSV:
python -m psibench.eval.emotion_classification \
  --csv-file output/emotion_analysis \
  --turn-threshold 16

python -m psibench.eval.emotion_classification --csv-file output/emotion_analysis --turn-threshold 16 --ptc-plot-file output/ptc_analysis_turn0/ptc_percentages_by_turn_no_filler.png
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
from json_repair import repair_json

from psibench.prompts.judge_prompt import create_emotion_judge_prompt
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

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Plutchik's 8 basic emotions + neutral
EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'anticipation', 'trust', 'neutral']


class EmotionClassifier:
    """Judge for emotion classification using Plutchik's 8 basic emotions."""
    
    def __init__(self, config: Dict[str, Any], debug: bool = False):
        """Initialize the emotion classifier.
        
        Args:
            config: Configuration dictionary with judge_model settings
            debug: Enable debug logging
        """
        self.config = config
        self.debug = debug
        self.prompt_template = create_emotion_judge_prompt()
        
        # Get judge model settings
        judge_config = config.get("eval", {}).get("emotion_classifier", {})
        self.model = judge_config.get('model')
        self.temperature = judge_config.get('temperature')
        
        # Get API settings
        if judge_config.get("api_base"):
            self.api_base = judge_config.get("api_base")
            self.api_key = "sk-no-key-required"
        else:
            self.api_base = os.getenv("OPENAI_BASE_URL")
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        if self.debug:
            print(f"[EmotionClassifier] Initialized with model: {self.model}")
            print(f"[EmotionClassifier] Temperature: {self.temperature}")
            
    def _format_history(self, history: list[Dict[str, str]], num_messages: int = None) -> str:
        """Format conversation history for the prompt.
        Switch to Therapist/Assistant labels, makes life easier for judges
        
        Args:
            history: List of conversation messages
            num_messages: Maximum number of messages to include (optional)
            
        Returns:
            Formatted conversation history string
        """
        formatted = []
        for msg in history:
            # Skip empty messages
            if not msg.get("content", "").strip():
                continue
            role = "THERAPIST" if msg["role"] == "user" else "PATIENT"
            content = msg["content"]
            formatted.append(f"{role}: {content}")
            if num_messages is not None and len(formatted) >= num_messages:
                break
            
        return "\n".join(formatted)

    def classify_turns_batch(self, conversations: List[List[Dict[str, str]]], num_messages: int = 4) -> List[List[Dict[str, Any]]]:
        """Classify emotions for all patient turns across multiple conversations in parallel.
        
        Args:
            conversations: List of conversations, each is a list of message dicts
            num_messages: Number of previous messages to include as history (default: 4)
            
        Returns:
            List of classification results, one per conversation. Each result is a list of dicts with:
            - turn_index: Index of the patient turn
            - content: Patient message content
            - emotion: Classified emotion
        """
        # Build all tasks across all conversations
        all_tasks = []
        task_metadata = []  # Track which conversation and turn each task belongs to
        
        for conv_idx, conversation in enumerate(conversations):
            patient_turns = []
            for i, msg in enumerate(conversation):
                # Handle both 'patient' (synthetic) and 'assistant' (real) roles
                role = msg.get('role', '').lower()
                if role in ('patient', 'assistant'):
                    # Get history (all messages before this one)
                    history = conversation[:i]
                    patient_turns.append({
                        'turn_index': len(patient_turns),
                        'content': msg['content'],
                        'history': history
                    })
            
            # Create classification tasks for each patient turn
            for turn in patient_turns:
                formatted_history = self._format_history(turn['history'], num_messages=num_messages)
                
                messages = self.prompt_template.format_messages(
                    history=formatted_history,
                    current_message=turn['content']
                )
                
                # Convert to litellm format - map 'human' to 'user' for OpenAI compatibility
                litellm_messages = []
                for i, msg in enumerate(messages):
                    if hasattr(msg, 'type'):
                        role = msg.type
                        # Fix: Map 'human' to 'user' for OpenAI API
                        if role == 'human':
                            role = 'user'
                        elif role not in ['system', 'assistant', 'user', 'function', 'tool', 'developer']:
                            role = 'system' if i == 0 else 'user'
                    else:
                        role = 'system' if i == 0 else 'user'
                    
                    litellm_messages.append({
                        "role": role,
                        "content": msg.content
                    })
                
                all_tasks.append(litellm_messages)
                task_metadata.append({
                    'conv_idx': conv_idx,
                    'turn_index': turn['turn_index'],
                    'content': turn['content']
                })
        
        if not all_tasks:
            return [[] for _ in conversations]
        
        # Run all classifications in parallel
        if self.debug:
            print(f"[DEBUG] Running {len(all_tasks)} emotion classification tasks in parallel...")
        
        responses = batch_completion(
            model=self.model,
            messages=all_tasks,
            temperature=self.temperature,
            api_key=self.api_key,
            api_base=self.api_base,
        )
        
        # Parse responses and group by conversation
        results_by_conv = {i: [] for i in range(len(conversations))}
        
        for idx, (response, metadata) in enumerate(zip(responses, task_metadata)):
            try:
                emotion = response.choices[0].message.content.strip().lower()
                
                # Validate emotion
                if emotion not in EMOTIONS:
                    if self.debug:
                        print(f"[WARNING] Invalid emotion '{emotion}' at task {idx}, defaulting to 'neutral'")
                    emotion = 'neutral'
                
                results_by_conv[metadata['conv_idx']].append({
                    'turn_index': metadata['turn_index'],
                    'content': metadata['content'],
                    'emotion': emotion
                })
                
            except Exception as e:
                if self.debug:
                    print(f"[ERROR] Failed to parse response at task {idx}: {e}")
                    print(f"[ERROR] Response: {response}")
                
                # Default to neutral on error
                results_by_conv[metadata['conv_idx']].append({
                    'turn_index': metadata['turn_index'],
                    'content': metadata['content'],
                    'emotion': 'neutral'
                })
        
        # Convert dict to list maintaining conversation order
        return [results_by_conv[i] for i in range(len(conversations))]


def compare_all_hf_pairs(config: Dict[str, Any], output_dir: Path, batch_size: int = 1,
                         num_messages: int = 4, exact_turns: int = None, turn_threshold: int = 12,
                         debug: bool = False, ptc_plot_file: str = None, synthetic_dir: str = None):
    """Compare emotion distributions across all PSI-backend pairs from HuggingFace against real data.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save results
        batch_size: Number of conversations to process in parallel
        num_messages: Number of previous messages for context
        exact_turns: If specified, only include conversations with exactly this many patient turns
        turn_threshold: Maximum turn index to display in visualizations (default: 12)
        debug: Enable debug logging
    """
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Initialize emotion judge
    judge = EmotionClassifier(config, debug=debug)
    
    # Load all real data once (combined esc, hope, annomi)
    print("\n" + "="*70)
    print("Loading all real conversations (esc, hope, annomi combined)...")
    print("="*70)
    all_real_convs = []
    valid_indices = []
    
    # Load all conversations from combined dataset
    real_df = load_real_dataset(dataset_type='all')
    for idx, row in real_df.iterrows():
        messages = row["messages"]
        # Real data uses 'assistant' role for patient
        patient_turns = sum(1 for msg in messages if msg.get('role', '').lower() == 'assistant')
        if exact_turns is None or patient_turns == exact_turns:
            all_real_convs.append(messages)
            valid_indices.append(idx)
    
    print(f"\nTotal real conversations: {len(all_real_convs)}")
    if exact_turns:
        print(f"(Filtered to conversations with exactly {exact_turns} patient turns)")
    
    # Check if real data already processed
    real_json = output_root / 'real_emotion_detailed.json'
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
    
    # Analyze real data if not already processed
    if not real_results:
        print("\n[INFO] Analyzing real conversations...")
        for batch_start in tqdm(range(0, len(all_real_convs), batch_size), desc="Classifying real"):
            batch_end = min(batch_start + batch_size, len(all_real_convs))
            batch_conversations = all_real_convs[batch_start:batch_end]
            batch_ids = valid_indices[batch_start:batch_end]
            
            batch_results = judge.classify_turns_batch(batch_conversations, num_messages=num_messages)
            
            for conv_id, conv, classifications in zip(batch_ids, batch_conversations, batch_results):
                # Real data uses 'assistant' role for patient
                total_patient_turns = sum(1 for msg in conv if msg.get('role', '').lower() == 'assistant')
                
                # Skip conversations with no patient turns (shouldn't happen with exact_turns filter)
                if total_patient_turns == 0:
                    print(f"[WARNING] Conversation {conv_id} has 0 patient turns, skipping")
                    continue
                
                emotion_counts = Counter(c['emotion'] for c in classifications)
                emotion_percentages = {emotion: emotion_counts.get(emotion, 0) / total_patient_turns 
                                      for emotion in EMOTIONS}
                
                result = {
                    'conversation_id': build_conversation_id(conv_id, is_real=True),
                    'total_patient_turns': total_patient_turns,
                    'classifications': classifications,
                }
                
                for emotion in EMOTIONS:
                    result[f'{emotion}_count'] = emotion_counts.get(emotion, 0)
                    result[f'{emotion}_pct'] = emotion_percentages[emotion]
                
                real_results.append(result)
    
    real_df = pd.DataFrame(real_results)
    print(f"[INFO] Analyzed {len(real_df)} real conversations")
    
    # Get HF dataset name from config
    dataset_name = config.get('eval', {}).get('hf_dataset', 'hknguyen20/psibench-data')
    
    # Get all PSI-backend pairs
    print("\n[INFO] Loading all unique PSI-backend pairs...")
    all_pairs = get_all_psi_backend_pairs(dataset_name=dataset_name, synthetic_dir=synthetic_dir)
    print(f"[INFO] Found {len(all_pairs)} unique (psi, backend_llm) pairs")
    print(f"[INFO] Loading synthetic data from HF dataset: {dataset_name}\n")
    
    # Analyze all synthetic pairs
    all_synthetic_results = {}
    all_synthetic_results_raw = {}  # Store raw results for JSON export
    
    for psi, backend_llm in sorted(all_pairs, key=lambda x: sort_key_by_psi_and_size(f"{x[0]}-{x[1]}")):
        label = f"{psi}_{backend_llm}"
        normalized_backend = normalize_backend_name(backend_llm)
        safe_label = safe_dir_name(label)
        
        print(f"\n{'='*70}")
        print(f"Processing: {psi} + {backend_llm}")
        print(f"{'='*70}")
        
        # Check if detailed JSON already exists with correct count
        synth_json = output_root / f'{safe_label}_emotion_detailed.json'
        if synth_json.exists():
            try:
                with open(synth_json, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if len(existing_data) == len(all_real_convs):  # Should match real conv count (1398)
                    print(f"[SKIP] {label} already processed with {len(existing_data)} conversations")
                    # Load existing data for visualization
                    synth_df = pd.DataFrame(existing_data)
                    all_synthetic_results[label] = synth_df
                    all_synthetic_results_raw[label] = existing_data
                    continue
                else:
                    print(f"[WARNING] Existing file has {len(existing_data)} conversations, expected {len(all_real_convs)}. Reprocessing...")
            except Exception as e:
                print(f"[WARNING] Failed to load existing file: {e}. Reprocessing...")
        
        try:
            # Load from HF
            df_hf = load_synthetic_hf_to_df(psi=psi, backend_llm=normalized_backend, dataset_name=dataset_name, synthetic_dir=synthetic_dir)
            
            if df_hf.empty:
                print(f"[WARNING] No data found for {label}, skipping")
                continue
            
            # Convert to conversation format
            synthetic_convs = []
            # Filter synthetic data by session_id to match valid real conversation indices
            if exact_turns is not None and valid_indices:
                # Filter by session_id matching valid_indices (conversations with exact_turns patient turns)
                for _, row in df_hf.iterrows():
                    session_id = row.get('session_id', None)
                    if session_id in valid_indices:
                        messages = row['messages']
                        synthetic_convs.append((session_id, messages))
                print(f"[INFO] Loaded {len(synthetic_convs)} conversations (filtered by session_id to match real conversations)")
            else:
                # No filtering by session_id, use all conversations
                for _, row in df_hf.iterrows():
                    conv = row['messages']
                    session_id = row.get('session_id', len(synthetic_convs))
                    synthetic_convs.append((session_id, conv))
                print(f"[INFO] Loaded {len(synthetic_convs)} conversations")
            
            if not synthetic_convs:
                print(f"[WARNING] No conversations after filtering for {label}, skipping")
                continue
            
            print(f"[INFO] Analyzing {len(synthetic_convs)} synthetic conversations...")
            
            # Classify in batches
            synthetic_results = []
            for batch_start in tqdm(range(0, len(synthetic_convs), batch_size), 
                                   desc=f"Classifying {label}", leave=False):
                batch_end = min(batch_start + batch_size, len(synthetic_convs))
                batch_items = synthetic_convs[batch_start:batch_end]
                batch_session_ids = [item[0] for item in batch_items]
                batch_conversations = [item[1] for item in batch_items]
                
                batch_results = judge.classify_turns_batch(batch_conversations, num_messages=num_messages)
                
                for session_id, conv, classifications in zip(batch_session_ids, batch_conversations, batch_results):
                    # Synthetic data may use 'patient' or 'assistant' role
                    total_patient_turns = sum(1 for msg in conv if msg.get('role', '').lower() == 'assistant')
                    
                    # Skip conversations with no patient turns
                    if total_patient_turns == 0:
                        print(f"[WARNING] Conversation {build_conversation_id(session_id, psi=psi, backend_llm=normalized_backend)} has 0 patient turns, skipping")
                        continue
                    
                    emotion_counts = Counter(c['emotion'] for c in classifications)
                    emotion_percentages = {emotion: emotion_counts.get(emotion, 0) / total_patient_turns 
                                          for emotion in EMOTIONS}
                    
                    result = {
                        'conversation_id': build_conversation_id(session_id, psi=psi, backend_llm=normalized_backend),
                        'total_patient_turns': total_patient_turns,
                        'classifications': classifications,
                    }
                    
                    for emotion in EMOTIONS:
                        result[f'{emotion}_count'] = emotion_counts.get(emotion, 0)
                        result[f'{emotion}_pct'] = emotion_percentages[emotion]
                    
                    synthetic_results.append(result)
            
            synth_df = pd.DataFrame(synthetic_results)
            all_synthetic_results[label] = synth_df
            all_synthetic_results_raw[label] = synthetic_results  # Store raw results for JSON export
            print(f"[INFO] Completed {label}: {len(synth_df)} conversations")
            
            # Save immediately after processing
            print(f"[INFO] Saving {label} results...")
            
            # Save summary CSV (without classifications)
            synth_csv = output_root / f'{safe_label}_emotion_summary.csv'
            synth_summary = synth_df.drop(columns=['classifications'])
            synth_summary.to_csv(synth_csv, index=False)
            print(f"[CSV SAVED] {synth_csv}")
            
            # Save detailed JSON (with classifications)
            synth_json = output_root / f'{safe_label}_emotion_detailed.json'
            with open(synth_json, 'w', encoding='utf-8') as f:
                json.dump(synthetic_results, f, indent=2, ensure_ascii=False)
            print(f"[JSON SAVED] {synth_json}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {label}: {e}")
            continue
    
    # Save real data results
    print(f"\n{'='*70}")
    print("Saving real data results...")
    print(f"{'='*70}")
    
    # Save real data
    real_csv = output_root / 'real_emotion_summary.csv'
    real_summary = real_df.drop(columns=['classifications'])
    real_summary.to_csv(real_csv, index=False)
    print(f"[CSV SAVED] {real_csv}")
    
    real_json = output_root / 'real_emotion_detailed.json'
    with open(real_json, 'w', encoding='utf-8') as f:
        json.dump(real_results, f, indent=2, ensure_ascii=False)
    print(f"[JSON SAVED] {real_json}")
    
    # Synthetic datasets are now saved immediately after processing (no need to save again here)
    
    # Create visualizations
    print("\n[INFO] Creating visualizations...")
    visualize_emotion_percentages_by_turn(real_df, all_synthetic_results, output_root, turn_threshold=turn_threshold, exact_turns=exact_turns)
    visualize_emotion_distributions(real_df, all_synthetic_results, output_root, exact_turns=exact_turns)

    if ptc_plot_file:
        merge_ptc_and_emotion_plots(
            ptc_plot_file=Path(ptc_plot_file),
            emotion_plot_file=output_root / 'emotion_percentages_by_turn.png',
            output_dir=output_root,
        )
    
    print(f"\n[DONE] Multi-pair emotion analysis complete. Results saved to: {output_root}")
    print(f"       Analyzed {len(all_synthetic_results)} synthetic datasets")


def get_turn_emotion_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract turn-by-turn emotion data from analysis results.
    
    Args:
        df: DataFrame with conversation analysis including classifications
        
    Returns:
        DataFrame with columns: conversation_id, turn_index, emotion
    """
    turn_data = []
    
    for _, row in df.iterrows():
        conv_id = row['conversation_id']
        classifications = row['classifications']
        
        for classification in classifications:
            turn_data.append({
                'conversation_id': conv_id,
                'turn_index': classification['turn_index'],
                'emotion': classification['emotion']
            })
    
    return pd.DataFrame(turn_data)


def visualize_emotion_percentages_by_turn(real_df: pd.DataFrame, all_synthetic_results: Dict[str, pd.DataFrame],
                                          output_dir: Path, turn_threshold: int = 16, exact_turns: int = None,
                                          pre_calculated_percentages: Dict[str, Dict[str, pd.DataFrame]] = None):
    """Create line plots showing percentage of each emotion across turns.
    
    All emotions are shown in subplots within a single figure.
    
    Args:
        real_df: DataFrame with real conversation analysis
        all_synthetic_results: Dictionary mapping label -> DataFrame with synthetic results
        output_dir: Directory to save plots
        turn_threshold: Maximum turn index to display
        exact_turns: If provided, label plots to reflect exact patient-turn filtering
        pre_calculated_percentages: Optional pre-calculated percentages to use instead of computing from data.
                                   Format: {'real': {emotion: DataFrame}, label: {emotion: DataFrame}}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Requested emotions: remove anger, keep neutral in its place, and keep
    # trust/joy/anticipation.
    # Layout:
    # Row 1: fear, sadness, neutral
    # Row 2: trust, joy, anticipation
    EMOTIONS_TO_PLOT = ['fear', 'sadness', 'neutral', 'trust', 'joy', 'anticipation']
    
    # Use pre-calculated percentages if provided, otherwise calculate from data
    if pre_calculated_percentages is not None:
        print("[INFO] Using pre-calculated percentages from saved file")
        real_percentages = pre_calculated_percentages['real']
        synthetic_percentages = {k: v for k, v in pre_calculated_percentages.items() if k != 'real'}
    else:
        # Get turn-by-turn data for real
        real_turns = get_turn_emotion_data(real_df)
        real_turns = real_turns[real_turns['turn_index'] < turn_threshold].copy()
        
        # Calculate percentage of each emotion at each turn for real data
        real_percentages = {}
        for emotion in EMOTIONS_TO_PLOT:
            emotion_by_turn = real_turns.groupby('turn_index').apply(
                lambda x: (x['emotion'] == emotion).sum() / len(x) * 100
            ).reset_index(name='percentage')
            real_percentages[emotion] = emotion_by_turn
        
        # Calculate for each synthetic dataset
        synthetic_percentages = {}
        for label in sorted(all_synthetic_results.keys(), key=sort_key_by_psi_and_size):
            synth_df = all_synthetic_results[label]
            synth_turns = get_turn_emotion_data(synth_df)
            synth_turns = synth_turns[synth_turns['turn_index'] < turn_threshold].copy()
            
            synthetic_percentages[label] = {}
            for emotion in EMOTIONS_TO_PLOT:
                emotion_by_turn = synth_turns.groupby('turn_index').apply(
                    lambda x: (x['emotion'] == emotion).sum() / len(x) * 100
                ).reset_index(name='percentage')
                synthetic_percentages[label][emotion] = emotion_by_turn
        
        # Save calculated percentages for faster redraw
        save_emotion_percentages(real_percentages, synthetic_percentages, output_dir, turn_threshold)
    
    # Match PTC by-turn subplot typography and line styling.
    fig, axes = plt.subplots(2, 3, figsize=(21, 11))
    axes = axes.flatten()
    
    # Build family-aware marker map once and reuse across plots/legend.
    backend_by_label = {label: extract_backend_name_from_label(label) for label in all_synthetic_results.keys()}
    sorted_backends = sorted(set(backend_by_label.values()), key=sort_key_by_backend_family_and_size)
    backend_markers = assign_backend_markers(sorted_backends)

    # Track what appears in the figure so the legend only contains used entries.
    used_psi_types = set()
    used_backends = set()
    
    # Collect all model sizes for opacity normalization
    all_model_sizes = [extract_model_size(backend) for backend in sorted_backends if extract_model_size(backend) > 0]
    
    # Plot each emotion
    for idx, emotion in enumerate(EMOTIONS_TO_PLOT):
        ax = axes[idx]
        
        # Plot real data
        real_data = real_percentages[emotion]
        ax.plot(real_data['turn_index'], real_data['percentage'],
            linewidth=3.2, alpha=0.9, color='black', marker='o', markersize=8)
        
        # Plot synthetic data
        for label in sorted(all_synthetic_results.keys(), key=sort_key_by_psi_and_size):
            synth_data = synthetic_percentages[label][emotion]
            
            # Determine PSI type for color
            psi_type = 'patientpsi' if 'patientpsi' in label else 'roleplaydoh'
            color = PSI_COLORS.get(psi_type, '#808080')
            used_psi_types.add(psi_type)
            
            # Extract backend for marker
            backend = backend_by_label.get(label, 'unknown')
            marker = backend_markers.get(backend, 'x')
            if backend in backend_markers:
                used_backends.add(backend)
            
            # Determine alpha based on model size (smaller models more faint)
            alpha = get_model_opacity(backend, all_model_sizes)
            
            ax.plot(synth_data['turn_index'], synth_data['percentage'],
                 linewidth=3.0, alpha=alpha, color=color, marker=marker, markersize=8)
        
        ax.set_xlabel('Turn Index', fontsize=30)
        ax.set_ylabel('Percentage', fontsize=30)
        ax.set_title(emotion.capitalize(), fontsize=34, fontweight='bold', pad=16)
        ax.set_xlim(-0.5, turn_threshold - 0.5)
        ax.set_ylim(0, 100)
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.grid(True, alpha=0.3)

        # Force integer x ticks to match PTC style.
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Create PTC-style split legend: PSI colors and backend markers.
    # from matplotlib.lines import Line2D

    # color_legend_elements = [Line2D([0], [0], color='black', linewidth=2.5, label=PSI_LABELS['Real'])]
    # for psi_type in sorted(used_psi_types):
    #     if psi_type in PSI_COLORS:
    #         color_legend_elements.append(
    #             Line2D([0], [0], color=PSI_COLORS[psi_type], linewidth=2, label=PSI_LABELS.get(psi_type, psi_type))
    #         )

    # marker_legend_elements = []
    # for backend in sorted(used_backends, key=sort_key_by_backend_family_and_size):
    #     if backend in backend_markers:
    #         marker_legend_elements.append(
    #             Line2D([0], [0], color='gray', marker=backend_markers[backend],
    #                    linestyle='None', markersize=8, label=shorten_backend_name(backend))
    #         )

    # legend1 = fig.legend(handles=color_legend_elements, fontsize=16,
    #                      bbox_to_anchor=(0.5, 1.03), loc='upper center',
    #                      ncol=max(3, len(color_legend_elements)), frameon=False,
    #                      handletextpad=0.4, columnspacing=1.1)

    # legend2 = None
    # if marker_legend_elements:
    #     legend2 = fig.legend(handles=marker_legend_elements, fontsize=14,
    #                          bbox_to_anchor=(0.5, 0.98), loc='upper center',
    #                          ncol=max(2, len(marker_legend_elements)), frameon=False,
    #                          handletextpad=0.4, columnspacing=1.0)
        
    fig.subplots_adjust(left=0.05, right=0.995, bottom=0.10, top=0.90, wspace=0.32, hspace=0.45)
    # extra_artists = [legend1, legend2] if legend2 else [legend1]
    plt.savefig(output_dir / 'emotion_percentages_by_turn.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'emotion_percentages_by_turn.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"[PLOT SAVED] {output_dir / 'emotion_percentages_by_turn.png'}")
    print(f"[PLOT SAVED] {output_dir / 'emotion_percentages_by_turn.pdf'}")


def merge_ptc_and_emotion_plots(ptc_plot_file: Path, emotion_plot_file: Path, output_dir: Path):
    """Merge PTC and emotion by-turn plots into one figure with two stacked subplots.

    The merged figure intentionally relies on the legend already present in the
    PTC figure, so emotion should be generated without its own legend.
    """
    if not ptc_plot_file.exists():
        print(f"[WARNING] PTC plot not found, skipping merge: {ptc_plot_file}")
        return

    if not emotion_plot_file.exists():
        print(f"[WARNING] Emotion plot not found, skipping merge: {emotion_plot_file}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    ptc_img = plt.imread(ptc_plot_file)
    emotion_img = plt.imread(emotion_plot_file)

    # PTC on top, emotion under it, with ~2x height so each emotion subplot
    # is close in size to each single PTC subplot.
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(20, 15),
        gridspec_kw={'height_ratios': [1, 2]},
    )
    axes[0].imshow(ptc_img)
    axes[0].axis('off')
    axes[1].imshow(emotion_img)
    axes[1].axis('off')

    plt.tight_layout(pad=0.5)
    merged_png = output_dir / 'ptc_emotion_percentages_by_turn_merged.png'
    merged_pdf = output_dir / 'ptc_emotion_percentages_by_turn_merged.pdf'
    plt.savefig(merged_png, dpi=300, bbox_inches='tight')
    plt.savefig(merged_pdf, bbox_inches='tight')
    plt.close()

    print(f"[PLOT SAVED] {merged_png}")
    print(f"[PLOT SAVED] {merged_pdf}")


def save_emotion_percentages(real_percentages: Dict[str, pd.DataFrame], 
                            synthetic_percentages: Dict[str, Dict[str, pd.DataFrame]],
                            output_dir: Path, turn_threshold: int):
    """Save calculated emotion percentages to CSV for faster redraw.
    
    Saves two versions:
    1. Original with all emotions including neutral
    2. Without neutral, with remaining emotions rescaled to sum to 100%
    
    Args:
        real_percentages: Dict mapping emotion -> DataFrame with [turn_index, percentage]
        synthetic_percentages: Dict mapping label -> {emotion: DataFrame with [turn_index, percentage]}
        output_dir: Directory to save the file
        turn_threshold: Maximum turn index included in the data
    """
    # Combine all data into a single DataFrame
    all_data = []
    
    # Add real data
    for emotion, df in real_percentages.items():
        for _, row in df.iterrows():
            all_data.append({
                'dataset': 'real',
                'turn_index': int(row['turn_index']),
                'emotion': emotion,
                'percentage': float(row['percentage'])
            })
    
    # Add synthetic data
    for label, emotions_dict in synthetic_percentages.items():
        for emotion, df in emotions_dict.items():
            for _, row in df.iterrows():
                all_data.append({
                    'dataset': label,
                    'turn_index': int(row['turn_index']),
                    'emotion': emotion,
                    'percentage': float(row['percentage'])
                })
    
    # Save original CSV with all emotions
    percentages_df = pd.DataFrame(all_data)
    output_file = output_dir / f'emotion_percentages_by_turn_t{turn_threshold}.csv'
    percentages_df.to_csv(output_file, index=False)
    print(f"[CSV SAVED] Emotion percentages (with neutral): {output_file}")
    
    # Create version without neutral, rescaled to sum to 100%
    # Filter out neutral
    no_neutral_df = percentages_df[percentages_df['emotion'] != 'neutral'].copy()
    
    # Group by dataset and turn_index to rescale
    rescaled_data = []
    for (dataset, turn_idx), group in no_neutral_df.groupby(['dataset', 'turn_index']):
        total = group['percentage'].sum()
        if total > 0:
            # Rescale so remaining emotions sum to 100%
            scale_factor = 100.0 / total
            for _, row in group.iterrows():
                rescaled_data.append({
                    'dataset': row['dataset'],
                    'turn_index': int(row['turn_index']),
                    'emotion': row['emotion'],
                    'percentage': float(row['percentage'] * scale_factor)
                })
        else:
            # If total is 0, keep as is (shouldn't happen in practice)
            for _, row in group.iterrows():
                rescaled_data.append({
                    'dataset': row['dataset'],
                    'turn_index': int(row['turn_index']),
                    'emotion': row['emotion'],
                    'percentage': float(row['percentage'])
                })
    
    # Save rescaled CSV without neutral
    rescaled_df = pd.DataFrame(rescaled_data)
    output_file_no_neutral = output_dir / f'emotion_percentages_by_turn_t{turn_threshold}_no_neutral.csv'
    rescaled_df.to_csv(output_file_no_neutral, index=False)
    print(f"[CSV SAVED] Emotion percentages (no neutral, rescaled): {output_file_no_neutral}")


def load_emotion_percentages(csv_dir: Path, turn_threshold: int) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load pre-calculated emotion percentages from CSV.
    
    Args:
        csv_dir: Directory containing the saved percentages file
        turn_threshold: Maximum turn index to load
        
    Returns:
        Dict mapping dataset -> {emotion: DataFrame with [turn_index, percentage]}
        Returns None if file doesn't exist or is invalid
    """
    percentages_file = csv_dir / f'emotion_percentages_by_turn_t{turn_threshold}.csv'
    
    if not percentages_file.exists():
        return None
    
    try:
        df = pd.read_csv(percentages_file)
        
        # Validate required columns
        required_cols = ['dataset', 'turn_index', 'emotion', 'percentage']
        if not all(col in df.columns for col in required_cols):
            print(f"[WARNING] Invalid percentages file format, missing required columns")
            return None
        
        # Convert to nested dict structure
        result = {}
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            result[dataset] = {}
            
            for emotion in dataset_df['emotion'].unique():
                emotion_df = dataset_df[dataset_df['emotion'] == emotion][['turn_index', 'percentage']]
                result[dataset][emotion] = emotion_df.reset_index(drop=True)
        
        print(f"[INFO] Loaded pre-calculated percentages from {percentages_file}")
        return result
        
    except Exception as e:
        print(f"[WARNING] Failed to load percentages file: {e}")
        return None


def visualize_emotion_distributions(real_df: pd.DataFrame, all_synthetic_results: Dict[str, pd.DataFrame], 
                                    output_dir: Path, exact_turns: int = None):
    """Create horizontal stacked bar chart comparing emotion distributions across all datasets.
    
    Args:
        real_df: DataFrame with real conversation analysis
        all_synthetic_results: Dictionary mapping label -> DataFrame with synthetic results
        output_dir: Directory to save plots
        exact_turns: If provided, label plots to reflect exact patient-turn filtering
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate emotion percentages for all datasets
    dataset_stats = []
    
    # Real data
    real_percentages = {}
    for emotion in EMOTIONS:
        count_col = f'{emotion}_count'
        real_percentages[emotion] = real_df[count_col].sum() / real_df['total_patient_turns'].sum()
    
    dataset_stats.append({
        'label': 'real',
        'display_name': 'Real',
        **real_percentages
    })
    
    # Synthetic data - sorted by PSI and model size
    for label in sorted(all_synthetic_results.keys(), key=sort_key_by_psi_and_size):
        synth_df = all_synthetic_results[label]
        
        synth_percentages = {}
        for emotion in EMOTIONS:
            count_col = f'{emotion}_count'
            synth_percentages[emotion] = synth_df[count_col].sum() / synth_df['total_patient_turns'].sum()
        
        # Create display label
        psi_type = 'patientpsi' if 'patientpsi' in label else 'roleplaydoh'
        backend = label.split('_', 1)[1] if '_' in label else label
        short_backend = shorten_backend_name(backend)
        psi_abbrev = PSI_ABBREV.get(psi_type, psi_type)
        display_name = f"{psi_abbrev}-{short_backend}"
        
        dataset_stats.append({
            'label': label,
            'display_name': display_name,
            **synth_percentages
        })
    
    # Create horizontal stacked bar chart
    fig_height = max(6, len(dataset_stats) * 0.4)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    # Extract data for plotting (reversed for top-to-bottom display)
    dataset_labels = [d['display_name'] for d in dataset_stats][::-1]
    
    # Define colors for emotions (using a colorblind-friendly palette)
    emotion_colors = {
        'anger': '#d62728',      # Red
        'disgust': '#8c564b',    # Brown
        'fear': '#9467bd',       # Purple
        'joy': '#2ca02c',        # Green
        'sadness': '#1f77b4',    # Blue
        'surprise': '#ff7f0e',   # Orange
        'anticipation': '#e377c2', # Pink
        'trust': '#17becf',      # Cyan
        'neutral': '#7f7f7f'     # Gray
    }
    
    # Plot horizontal stacked bars
    left = [0] * len(dataset_labels)
    
    for emotion in EMOTIONS:
        values = [d[emotion] for d in dataset_stats][::-1]
        ax.barh(dataset_labels, values, left=left, label=emotion.capitalize(), 
                color=emotion_colors[emotion], alpha=0.85)
        left = [l + v for l, v in zip(left, values)]
    
    # Add percentage labels for segments >= 5%
    for i, dataset in enumerate(reversed(dataset_stats)):
        left_pos = 0
        for emotion in EMOTIONS:
            value = dataset[emotion]
            if value >= 0.05:  # Only label if >= 5%
                ax.text(left_pos + value/2, i, f'{value*100:.0f}%', 
                       ha='center', va='center', fontsize=10, fontweight='bold', color='white')
            left_pos += value
    
    ax.set_xlabel('Proportion', fontsize=18)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), fontsize=14, ncol=4, title='Emotions')
    ax.grid(axis='x', alpha=0.3)
    
    title_suffix = f" (exactly {exact_turns} patient turns)" if exact_turns else ""
    # fig.suptitle(f'Emotion Distribution{title_suffix}', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'emotion_distribution_all_pairs.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'emotion_distribution_all_pairs.pdf', bbox_inches='tight')
    plt.close()
    print(f"[PLOT SAVED] {output_dir / 'emotion_distribution_all_pairs.png'}")
    print(f"[PLOT SAVED] {output_dir / 'emotion_distribution_all_pairs.pdf'}")


def redraw_from_csv(csv_dir: str, output_dir: str, turn_threshold: int = 12,
                    exact_turns: int = None, ptc_plot_file: str = None):
    """Redraw visualizations from existing CSV and JSON files.
    
    Args:
        csv_dir: Directory containing saved CSV and JSON files
        output_dir: Output directory for regenerated plots
        turn_threshold: Maximum turn index for line plots
        exact_turns: If provided, label plots to reflect exact patient-turn filtering
        ptc_plot_file: Optional existing PTC by-turn PNG path to merge with emotion plot
    """
    csv_path = Path(csv_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] Loading data from {csv_path}...")
    
    # Load real data
    real_json = csv_path / 'real_emotion_detailed.json'
    if not real_json.exists():
        raise FileNotFoundError(f"Real data file not found: {real_json}")
    
    with open(real_json, 'r', encoding='utf-8') as f:
        real_results = json.load(f)
    
    real_df = pd.DataFrame(real_results)
    print(f"[INFO] Loaded real data: {len(real_df)} conversations")
    
    # Load all synthetic data
    all_synthetic_results = {}
    synthetic_csvs = list(csv_path.glob('*_emotion_summary.csv'))
    synthetic_csvs = [f for f in synthetic_csvs if not f.name.startswith('real_')]
    
    for synth_csv in synthetic_csvs:
        label = synth_csv.stem.replace('_emotion_summary', '')
        
        # Load summary CSV
        synth_summary = pd.read_csv(synth_csv)
        
        # Try to load detailed JSON if available
        synth_json = csv_path / f'{label}_emotion_detailed.json'
        if synth_json.exists():
            with open(synth_json, 'r', encoding='utf-8') as f:
                synth_results = json.load(f)
            synth_df = pd.DataFrame(synth_results)
        else:
            # If detailed JSON not available, use summary CSV
            # Note: This won't have 'classifications' field, so some visualizations may not work
            synth_df = synth_summary
            print(f"[WARNING] No detailed JSON found for {label}, using summary CSV only")
        
        all_synthetic_results[label] = synth_df
        print(f"[INFO] Loaded {label}: {len(synth_df)} conversations")
    
    print(f"\n[INFO] Found {len(all_synthetic_results)} synthetic datasets")
    
    # Try to load pre-calculated percentages first
    pre_calculated = load_emotion_percentages(csv_path, turn_threshold)
    
    if pre_calculated is not None:
        # Use pre-calculated percentages (much faster)
        print("[INFO] Creating visualizations from pre-calculated percentages...")
        
        # Also generate the no_neutral CSV if it doesn't exist
        no_neutral_csv = csv_path / f'emotion_percentages_by_turn_t{turn_threshold}_no_neutral.csv'
        if not no_neutral_csv.exists():
            print("[INFO] Generating no_neutral CSV from loaded data...")
            # Extract real and synthetic percentages from pre_calculated
            real_percentages = pre_calculated.get('real', {})
            synthetic_percentages = {k: v for k, v in pre_calculated.items() if k != 'real'}
            # Save both versions (including the no_neutral version)
            save_emotion_percentages(real_percentages, synthetic_percentages, csv_path, turn_threshold)
        
        visualize_emotion_percentages_by_turn(real_df, all_synthetic_results, output_root,
                                             turn_threshold=turn_threshold, exact_turns=exact_turns,
                                             pre_calculated_percentages=pre_calculated)
    else:
        # Fall back to calculating from detailed data
        print("[INFO] Pre-calculated percentages not found, calculating from detailed data...")
        # Filter datasets that have classifications for by-turn visualization
        datasets_with_classifications = {}
        for label, df in all_synthetic_results.items():
            if 'classifications' in df.columns:
                datasets_with_classifications[label] = df
            else:
                print(f"[WARNING] Skipping {label} for by-turn visualization (missing detailed classifications)")
        
        # Create visualizations
        print("\n[INFO] Creating visualizations...")
        if datasets_with_classifications:
            visualize_emotion_percentages_by_turn(real_df, datasets_with_classifications, output_root, 
                                                 turn_threshold=turn_threshold, exact_turns=exact_turns)
        else:
            print("[WARNING] No datasets with detailed classifications found, skipping by-turn visualization")
    
    visualize_emotion_distributions(real_df, all_synthetic_results, output_root, exact_turns=exact_turns)

    if ptc_plot_file:
        merge_ptc_and_emotion_plots(
            ptc_plot_file=Path(ptc_plot_file),
            emotion_plot_file=output_root / 'emotion_percentages_by_turn.png',
            output_dir=output_root,
        )
    
    print(f"\n[DONE] Visualizations regenerated in: {output_root}")


def main():
    """Main function to run emotion classification analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Classify conversations using Plutchik emotion framework')
    parser.add_argument('--output-dir', type=str, default='output/emotion_analysis',
                       help='Output directory')
    parser.add_argument('--hf', action='store_true',
                       help='Load all psi/backend pairs from HF dataset')
    parser.add_argument('--synthetic-dir', type=str, default=None,
                       help='Local folder tree of session_*.json (<psi>/<model>/<dataset>/...) used instead of the HF synthetic data; psi and model are read from the folder names')
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
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--ptc-plot-file', type=str, default=None,
                       help='Optional path to an existing PTC by-turn PNG to merge with emotion by-turn plot')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Create descriptive output directory name
    dir_parts = []
    
    if args.csv_file:
        csv_path = Path(args.csv_file)
        original_id = csv_path.name
        # dir_parts.append(f"redraw_{original_id}")
    
    if args.exact_turns:
        dir_parts.append(f"exact_turns_{args.exact_turns}")
    
    # Use subdirectory if dir_parts exist, otherwise use output_dir directly
    if dir_parts:
        dir_name = "_".join(dir_parts)
        output_dir = Path(args.output_dir) / dir_name
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    if args.csv_file:
        # Redraw mode
        redraw_from_csv(
            csv_dir=args.csv_file,
            output_dir=output_dir,
            turn_threshold=args.turn_threshold,
            exact_turns=args.exact_turns,
            ptc_plot_file=args.ptc_plot_file,
        )
    elif args.hf or args.synthetic_dir:
        # Multi-dataset analysis mode (synthetic side from HF or --synthetic-dir)
        judge = EmotionClassifier(config, debug=args.debug)
        compare_all_hf_pairs(
            config=config,
            output_dir=output_dir,
            batch_size=args.batch_size,
            num_messages=args.num_messages,
            exact_turns=args.exact_turns,
            turn_threshold=args.turn_threshold,
            debug=args.debug,
            ptc_plot_file=args.ptc_plot_file,
            synthetic_dir=args.synthetic_dir,
        )
    else:
        print("Error: Please provide --hf, --synthetic-dir or --csv-file")
        return
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nResults saved to {output_dir}")
    print(f"Total time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
