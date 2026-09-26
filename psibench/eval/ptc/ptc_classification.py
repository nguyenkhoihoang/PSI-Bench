"""
PTC Classification: Classify patient turns as Problem, Transition, or Change.

This module uses an LLM-judge to classify each patient turn in therapy conversations
according to the Problem-Transition-Change framework.

Resume/skip behavior:
- Real data: skips reclassification only if real_ptc_detailed.json exists and
    its conversation count matches the currently eligible real conversations.
- Synthetic per (psi, backend) pair: skips only if the pair's
    *_ptc_detailed.json exists and its conversation count matches expected count
    for this run; otherwise that whole pair is reprocessed.
- No partial resume within a pair/conversation. Skip is dataset-level.

# Classify all turns but display only first 16 in visualizations and ignore filler from summaries:
nohup python -m psibench.eval.ptc.ptc_classification \
  --hf \
  --batch-size 384 \
  --single-turn \
  --no-filler \
  --turn-threshold 16 \
    --output-dir output/ptc_analysis_turn0 \
  --config configs/default.yaml > ptc.out 2>&1 &


python -m psibench.eval.ptc.ptc_classification \
  --hf \
  --batch-size 384 \
  --single-turn \
  --exact-turns 12 \
  --config configs/default.yaml


# Redraw from saved CSV with custom turn display threshold:
python -m psibench.eval.ptc.ptc_classification \
  --csv-file output/ptc_analysis_turn0 \
  --turn-threshold 16 \
  --no-filler

"""

import json
import os
import ast
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import yaml
from litellm import batch_completion
from json_repair import repair_json
from scipy.spatial import distance

from psibench.prompts.judge_prompt import create_ptc_judge_conversation_prompt
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


class PTCClassifier:
    """Judge for Problem-Transition-Change framework classification."""
    
    def __init__(self, config: Dict[str, Any], debug: bool = False):
        """Initialize the PTC judge.
        
        Args:
            config: Configuration dictionary containing eval.ptc_classifier settings
            debug: Enable debug logging (default: False)
        """
        # Store model settings for batch completion
        self.debug = debug
        judge_config = config.get("eval", {}).get("ptc_classifier", {})
        self.model_name = judge_config.get("model")
        self.temperature = judge_config.get("temperature", 0.3)
        self.history_limit = judge_config.get("history_limit", 20)
        
        # Get API settings
        if judge_config.get("api_base"):
            self.api_base = judge_config.get("api_base")
            self.api_key = "sk-no-key-required"
        else:
            self.api_base = os.getenv("OPENAI_BASE_URL")
            self.api_key = os.getenv("OPENAI_API_KEY")
            
    def _format_history(self, history: list[Dict[str, str]]) -> str:
        """Format conversation history for the prompt.
        Switch to Therapist/Assistant labels, makes life easier for judges
        
        Args:
            history: List of conversation messages
            
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
            if len(formatted) == self.history_limit:
                break
            
        return "\n".join(formatted)

    def classify_conversations_batch(self, conversations: List[List[Dict[str, str]]]) -> List[List[Dict[str, Any]]]:
        """Batch classify multiple conversations using litellm.batch_completion.
        
        Args:
            conversations: List of conversations, each being a list of messages
            
        Returns:
            List of classification results for each conversation
        """
        # Get the prompt template
        conversation_prompt = create_ptc_judge_conversation_prompt()
        
        # Build messages for each conversation
        all_messages = []
        for messages in conversations:
            conversation_str = self._format_history(messages)
            # Debug: print approximate context size
            if self.debug:
                approx_tokens = max(1, len(conversation_str) // 4)
                print(f"Judge prompt context: ~{approx_tokens} tokens, {len(conversation_str)} chars")
            
            # Format prompt messages
            formatted_prompt = conversation_prompt.format_messages(conversation=conversation_str)
            # Convert to litellm format - map 'human' to 'user' for OpenAI compatibility
            litellm_messages = []
            for i, msg in enumerate(formatted_prompt):
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
            all_messages.append(litellm_messages)
        if self.debug:
            print(f"Prepared {len(all_messages)} judge calls; first message roles: {[m['role'] for m in all_messages[0]] if all_messages else []}")
        
        # Batch completion call with error handling
        try:
            responses = batch_completion(
                model=self.model_name,
                messages=all_messages,
                temperature=self.temperature,
                api_key=self.api_key,
                api_base=self.api_base,
            )
        except Exception as e:
            print(f"Error in batch_completion call: {e}")
            return [[] for _ in conversations]
        
        # Parse all responses
        results = []
        for i, response in enumerate(responses):
            try:
                # Handle different response formats
                if isinstance(response, str):
                    response_text = response.strip()
                elif hasattr(response, 'choices') and len(getattr(response, 'choices', [])) > 0:
                    content = response.choices[0].message.content
                    if content is None:
                        if self.debug:
                            print(f"Batch {i}: choices[0].message.content is None. Full response: {getattr(response, 'dict', lambda: response)() if hasattr(response, 'dict') else response}")
                        results.append([])
                        continue
                    response_text = content.strip()
                elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                    content = response.message.content
                    if content is None:
                        if self.debug:
                            print(f"Batch {i}: message.content is None. Full response: {getattr(response, 'dict', lambda: response)() if hasattr(response, 'dict') else response}")
                        results.append([])
                        continue
                    response_text = content.strip()
                elif hasattr(response, 'content'):
                    content = response.content
                    if content is None:
                        if self.debug:
                            print(f"Batch {i}: content is None. Full response: {getattr(response, 'dict', lambda: response)() if hasattr(response, 'dict') else response}")
                        results.append([])
                        continue
                    response_text = content.strip()
                else:
                    raise ValueError(f"Unexpected response format: {type(response)}")
                
                classifications = repair_json(response_text, return_objects=True)
                # Normalize common shapes: dict with key, list of dicts, list of strings
                if isinstance(classifications, dict):
                    # Try common keys
                    for key in ('classifications', 'turns', 'labels'):
                        if key in classifications and isinstance(classifications[key], list):
                            classifications = classifications[key]
                            break
                if isinstance(classifications, list):
                    # If list of strings, convert to dicts
                    if all(isinstance(x, str) for x in classifications):
                        classifications = [{'classification': x} for x in classifications]
                    # If list-of-lists, unwrap first
                    elif len(classifications) > 0 and all(isinstance(x, list) for x in classifications):
                        classifications = classifications[0]
                
                # Filter invalid items and log, and add turn_index
                valid_items = []
                for idx, item in enumerate(classifications if isinstance(classifications, list) else []):
                    if isinstance(item, dict) and 'classification' in item and isinstance(item['classification'], str):
                        # Ensure turn_index is present
                        if 'turn_index' not in item:
                            item['turn_index'] = idx
                        valid_items.append(item)
                    else:
                        if self.debug:
                            print(f"Batch {i}: Dropping invalid item at pos {idx}: {item}")
                classifications = valid_items
                results.append(classifications)
            except Exception as e:
                print(f"Error parsing batch response {i}: {e}")
                print(f"Response type: {type(response)}")
                results.append([])  # Empty list on error
        
        return results

    def classify_turns_batch(self, conversations: List[List[Dict[str, str]]], num_messages: int = 6) -> List[List[Dict[str, Any]]]:
        """Batch classify individual patient turns with limited history context.
        
        Args:
            conversations: List of conversations, each being a list of messages
            num_messages: Number of previous messages to include as history (default: 6)
            
        Returns:
            List of classification results for each conversation, where each result is a list of 
            {"content": str, "classification": str, "turn_index": int} for each patient turn
        """
        from psibench.prompts.judge_prompt import create_ptc_judge_single_turn_prompt
        
        # Get the prompt template
        single_turn_prompt = create_ptc_judge_single_turn_prompt()
        
        # Build all classification tasks across all conversations
        all_messages = []
        task_metadata = []  # Track which conversation and turn each task belongs to
        
        for conv_idx, messages in enumerate(conversations):
            # Find all patient turns (role == 'assistant')
            patient_indices = [i for i, msg in enumerate(messages) 
                             if msg.get('role') == 'assistant' and msg.get('content', '').strip()]
            
            for patient_turn_idx, patient_msg_idx in enumerate(patient_indices):
                # Get history: up to num_messages before this patient turn
                history_start = max(0, patient_msg_idx - num_messages)
                history_messages = messages[history_start:patient_msg_idx]
                history_str = self._format_history(history_messages) if history_messages else "(No previous history)"
                
                # Current patient message
                current_message = messages[patient_msg_idx]['content']
                
                # Format prompt
                formatted_prompt = single_turn_prompt.format_messages(
                    history=history_str,
                    current_message=current_message
                )
                
                # Convert to litellm format
                litellm_messages = []
                for i, msg in enumerate(formatted_prompt):
                    if hasattr(msg, 'type'):
                        role = msg.type
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
                
                all_messages.append(litellm_messages)
                task_metadata.append({
                    'conv_idx': conv_idx,
                    'turn_idx': patient_turn_idx,
                    'message_idx': patient_msg_idx,
                    'content': current_message
                })
        
        if self.debug:
            print(f"Prepared {len(all_messages)} single-turn classification tasks across {len(conversations)} conversations")
            # Log first few request structures for debugging
            for task_idx in range(min(2, len(all_messages))):
                print(f"\n--- Sample request {task_idx} ---")
                print(f"Number of messages: {len(all_messages[task_idx])}")
                for msg_idx, msg in enumerate(all_messages[task_idx]):
                    content_preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                    print(f"  Message {msg_idx} (role={msg['role']}): {content_preview}")
        
        # Batch completion call
        try:
            if self.debug:
                print(f"\n--- Sending batch completion request ---")
                print(f"Model: {self.model_name}")
                print(f"Temperature: {self.temperature}")
                print(f"API base: {self.api_base}")
            
            responses = batch_completion(
                model=self.model_name,
                messages=all_messages,
                temperature=self.temperature,
                api_key=self.api_key,
                api_base=self.api_base,
            )
            
            if self.debug:
                print(f"Batch completion succeeded. Received {len(responses)} responses")
        except Exception as e:
            print(f"Error in batch_completion call: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            return [[] for _ in conversations]
        
        # Parse responses and organize by conversation
        conversation_results = [[] for _ in conversations]
        
        for i, (response, metadata) in enumerate(zip(responses, task_metadata)):
            try:
                # Extract response text
                if isinstance(response, str):
                    response_text = response.strip()
                elif hasattr(response, 'choices') and len(getattr(response, 'choices', [])) > 0:
                    content = response.choices[0].message.content
                    if content is None:
                        if self.debug:
                            print(f"Task {i}: choices[0].message.content is None; skipping")
                        continue
                    else:
                        response_text = content.strip()
                elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                    content = response.message.content
                    if content is None:
                        if self.debug:
                            print(f"Task {i}: message.content is None; skipping")
                        continue
                    else:
                        response_text = content.strip()
                elif hasattr(response, 'content'):
                    content = response.content
                    if content is None:
                        if self.debug:
                            print(f"Task {i}: content is None; skipping")
                        continue
                    else:
                        response_text = content.strip()
                else:
                    raise ValueError(f"Unexpected response format: {type(response)}")
                
                # Parse classification (should be single letter)
                if 'response_text' in locals():
                    classification = response_text.upper().strip()
                    # Extract just the classification letter if there's extra text
                    for letter in ['P', 'T', 'C', 'F']:
                        if letter in classification:
                            classification = letter
                            break
                    else:
                        if self.debug:
                            print(f"Task {i}: Could not find valid classification in '{response_text}'; skipping")
                        continue
                
                # Add to appropriate conversation's results
                conv_idx = metadata['conv_idx']
                conversation_results[conv_idx].append({
                    'content': metadata['content'],
                    'classification': classification,
                    'turn_index': metadata['turn_idx'],
                    'message_index': metadata['message_idx']
                })
                
            except Exception as e:
                if self.debug:
                    print(f"Error parsing task {i} response: {e}; skipping")
                continue
        
        return conversation_results


def analyze_conversations(
    judge: PTCClassifier,
    output_dir: Path,
    batch_size: int = 1,
    dataset: str = None,
    indices: List[int] = None,
    data_dir: Path = None,
    session_files: List[Path] = None,
    single_turn: bool = False,
    num_messages: int = 6,
    exact_turns: int = None,) -> pd.DataFrame:
    """Analyze conversations from either real dataset or synthetic files.
    
    Args:
        judge: PTCClassifier instance
        output_dir: Directory to save results
        batch_size: Number of conversations to process in parallel
        dataset: Dataset type for real conversations (e.g., 'esc', 'hope'). If provided, loads from real conversation dataset.
        indices: List of conversation indices (required if dataset is provided)
        data_dir: Directory containing synthetic conversation JSON files (alternative to dataset/indices)
        single_turn: If True, use single-turn classification with limited history (default: False)
        num_messages: Number of previous messages to include as history for single-turn mode (default: 6)
        exact_turns: If specified, only include conversations with exactly this many patient turns (default: None)
        
    Returns:
        DataFrame with analysis results
    """
    results = []
    all_conversations = []
    conversation_ids = []  # Store identifiers (indices or filenames)
    mismatch_count = 0  # Track classification mismatches
    
    # Load conversations based on source
    if dataset is not None and indices is not None:
        # Load from real dataset
        real_df = load_real_dataset(dataset_type=dataset, indices=indices)
        print(f"Loaded {len(real_df)} real conversations from {dataset} dataset")
        
        for idx, row in real_df.iterrows():
            messages = row["messages"]
            # Filter by exact patient turn count if specified
            if exact_turns is not None:
                patient_turn_count = sum(1 for msg in messages 
                                       if msg.get('role') == 'assistant' and msg.get('content', '').strip())
                if patient_turn_count != exact_turns:
                    continue
            all_conversations.append(messages)
            conversation_ids.append(('real', idx))
        
        print(f"Analyzing {len(all_conversations)} conversations (filtered to {exact_turns} patient turns)" if exact_turns else f"Analyzing {len(all_conversations)} conversations")
            
    elif data_dir is not None:
        # Load from synthetic files
        # If a specific list of session_files is provided, use that; otherwise glob from data_dir
        if session_files is None:
            session_files = sorted(data_dir.glob('session_*.json'))
        print(f"Loaded {len(session_files)} conversations from {data_dir}")
        
        for session_file in session_files:
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                messages = data.get('messages', [])
                # Filter by exact patient turn count if specified
                if exact_turns is not None:
                    patient_turn_count = sum(1 for msg in messages 
                                           if msg.get('role') == 'assistant' and msg.get('content', '').strip())
                    if patient_turn_count != exact_turns:
                        continue
                all_conversations.append(messages)
                conversation_ids.append(('synthetic', session_file))
            except Exception as e:
                print(f"Error loading {session_file}: {e}")
                continue
        
        print(f"Analyzing {len(all_conversations)} conversations (filtered to {exact_turns} patient turns)" if exact_turns else f"Analyzing {len(all_conversations)} conversations")
    else:
        raise ValueError("Must provide either (dataset and indices) or data_dir")
    
    print(f"Batch size: {batch_size}")
    
    # Process in batches
    for batch_start in tqdm(range(0, len(all_conversations), batch_size), desc="Classifying conversations"):
        batch_end = min(batch_start + batch_size, len(all_conversations))
        batch_conversations = all_conversations[batch_start:batch_end]
        batch_ids = conversation_ids[batch_start:batch_end]
        
        try:
            if single_turn:
                classifications_list = judge.classify_turns_batch(batch_conversations, num_messages=num_messages)
            else:
                classifications_list = judge.classify_conversations_batch(batch_conversations)
            
            # Process results
            for conv_id, classifications, messages in zip(batch_ids, classifications_list, batch_conversations):
                # Count expected non-empty patient turns
                expected_patient_turns = min(judge.history_limit // 2 ,sum(1 for msg in messages 
                                              if msg.get('role') == 'assistant' and msg.get('content', '').strip()))
                
                # Validate classification count
                actual_classifications = len(classifications)
                if conv_id[0] == 'real':
                    conv_identifier = build_conversation_id(conv_id[1], is_real=True)
                else:
                    conv_identifier = build_conversation_id(conv_id[1].stem)
                has_mismatch = actual_classifications != expected_patient_turns
                if has_mismatch:
                    mismatch_count += 1
                    print(f"WARNING: Classification mismatch for {conv_identifier}: expected {expected_patient_turns} patient turns, got {actual_classifications} classifications")
                    # Save debug output
                    debug_filename = f"debug_{safe_dir_name(conv_identifier)}.json"
                    debug_path = output_dir / "debug" / debug_filename
                    debug_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    debug_output = {
                        "conversation_id": conv_identifier,
                        "expected_patient_turns": expected_patient_turns,
                        "actual_classifications": actual_classifications,
                        "input_messages": messages,
                        "output_classifications": classifications
                    }
                    
                    with open(debug_path, 'w', encoding='utf-8') as f:
                        json.dump(debug_output, f, indent=2, ensure_ascii=False)
                    
                    print(f"Debug output saved to {debug_path}")
                
                ptc_counts = Counter([c['classification'] for c in classifications])
                total = len(classifications)
                non_filler_total = sum([ptc_counts.get(cat, 0) for cat in ['P', 'T', 'C']])
                
                # Build result dict based on source type
                result = { 
                    'conversation_id': conv_identifier,
                        'total_patient_turns': total,
                        'non_filler_turns': non_filler_total,
                        'P_count': ptc_counts.get('P', 0),
                        'T_count': ptc_counts.get('T', 0),
                        'C_count': ptc_counts.get('C', 0),
                        'F_count': ptc_counts.get('F', 0),
                        'P_ratio': ptc_counts.get('P', 0) / non_filler_total if non_filler_total > 0 else 0,
                        'T_ratio': ptc_counts.get('T', 0) / non_filler_total if non_filler_total > 0 else 0,
                        'C_ratio': ptc_counts.get('C', 0) / non_filler_total if non_filler_total > 0 else 0,
                        'F_ratio': ptc_counts.get('F', 0) / total if total > 0 else 0,
                        'classifications': classifications,
                        'has_mismatch': has_mismatch
                        }
                detail_filename = f"{safe_dir_name(conv_identifier)}_ptc.json"
                
                results.append(result)
                
                # Save detailed classification for this conversation
                detail_path = output_dir / detail_filename
                detail_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(detail_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                    
        except Exception as e:
            print(f"Error processing batch starting at {batch_start}: {e}")
            continue
    
    total_conversations = len(results)
    print(f"\n{'='*60}")
    print(f"Classification Summary: {mismatch_count}/{total_conversations} conversations had mismatches")
    print(f"{'='*60}")
    # Write summary to text file
    summary_path = output_dir / "classification_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Total conversations: {total_conversations}\n")
        f.write(f"Conversations with mismatches: {mismatch_count}\n")
        f.write(f"Mismatch rate: {mismatch_count/total_conversations*100:.1f}%\n")
    print(f"Summary saved to {summary_path}")
    return pd.DataFrame(results)


def compare_all_hf_pairs(config: Dict[str, Any], output_dir: Path, batch_size: int = 1, 
                         single_turn: bool = False, num_messages: int = 6, exact_turns: int = None, 
                         turn_threshold: int = 16, debug: bool = False):
    """Compare PTC distributions across all PSI-backend pairs from HuggingFace against real data.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save results
        batch_size: Number of conversations to process in parallel
        single_turn: If True, use single-turn classification
        num_messages: Number of previous messages for single-turn mode
        exact_turns: If specified, only include conversations with exactly this many patient turns
        turn_threshold: Maximum turn index to display in visualizations (default: 16)
        debug: Enable debug logging
    """
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Initialize PTC judge
    judge = PTCClassifier(config, debug=debug)
    
    # Load all real data once (combined esc, hope, annomi)
    print("\n" + "="*70)
    print("Loading all real conversations (esc, hope, annomi combined)...")
    print("="*70)
    all_real_convs = []
    valid_indices = []  # Track which conversation indices pass the filter
    
    for dataset in ['esc', 'hope', 'annomi']:
        try:
            real_df = load_real_dataset(dataset_type=dataset)
            for idx, row in real_df.iterrows():
                messages = row["messages"]
                if exact_turns is not None:
                    patient_turn_count = sum(1 for msg in messages 
                                           if msg.get('role') == 'assistant' and msg.get('content', '').strip())
                    if patient_turn_count != exact_turns:
                        continue
                all_real_convs.append(messages)
                valid_indices.append(idx)  # Store the index of conversations that pass the filter
            print(f"  ✓ Loaded {len([m for _, r in real_df.iterrows() for m in [r['messages']]])} from {dataset}")
        except Exception as e:
            print(f"  ⚠ Error loading {dataset}: {e}")
    
    print(f"\nTotal real conversations: {len(all_real_convs)}")
    if exact_turns:
        print(f"Filtered to conversations with exactly {exact_turns} patient turns")
        print(f"Valid conversation indices: {sorted(set(valid_indices))}\n")
    
    # Check if real data already processed
    real_json = output_root / 'real_ptc_detailed.json'
    real_results = []
    if real_json.exists():
        try:
            with open(real_json, 'r', encoding='utf-8') as f:
                existing_real_data = json.load(f)
            if len(existing_real_data) == len(all_real_convs):
                print(f"[SKIP] Real conversations already processed with {len(existing_real_data)} conversations")
                real_results = existing_real_data
            else:
                print(
                    f"[WARNING] Existing real file has {len(existing_real_data)} conversations, "
                    f"expected {len(all_real_convs)}. Reprocessing..."
                )
        except Exception as e:
            print(f"[WARNING] Failed to load existing real file: {e}. Reprocessing...")

    # Analyze real data if not already processed
    if not real_results:
        print("\n[INFO] Analyzing real conversations...")
        for batch_start in tqdm(range(0, len(all_real_convs), batch_size), desc="Classifying real"):
            batch_convs = all_real_convs[batch_start:batch_start + batch_size]

            if single_turn:
                batch_classifications = judge.classify_turns_batch(batch_convs, num_messages=num_messages)
            else:
                batch_classifications = judge.classify_conversations_batch(batch_convs)

            for idx, classifications in enumerate(batch_classifications):
                conv_idx = batch_start + idx
                session_id = valid_indices[conv_idx] if conv_idx < len(valid_indices) else conv_idx
                p_count = sum(1 for c in classifications if c['classification'] == 'P')
                t_count = sum(1 for c in classifications if c['classification'] == 'T')
                c_count = sum(1 for c in classifications if c['classification'] == 'C')
                f_count = sum(1 for c in classifications if c['classification'] == 'F')
                total_patient_turns = len(classifications)
                non_filler_turns = p_count + t_count + c_count

                real_results.append({
                    'conversation_id': build_conversation_id(session_id, is_real=True),
                    'P_count': p_count,
                    'T_count': t_count,
                    'C_count': c_count,
                    'F_count': f_count,
                    'total_patient_turns': total_patient_turns,
                    'non_filler_turns': non_filler_turns,
                    'P_ratio': p_count / non_filler_turns if non_filler_turns > 0 else 0,
                    'T_ratio': t_count / non_filler_turns if non_filler_turns > 0 else 0,
                    'C_ratio': c_count / non_filler_turns if non_filler_turns > 0 else 0,
                    'F_ratio': f_count / total_patient_turns if total_patient_turns > 0 else 0,
                    'classifications': classifications
                })
    
    real_df = pd.DataFrame(real_results)
    print(f"[INFO] Analyzed {len(real_df)} real conversations")
    
    # Get all PSI-backend pairs
    dataset_name = config.get('eval', {}).get('hf_dataset', 'hknguyen20/psibench-data')
    print(f"[INFO] Loading all unique PSI-backend pairs from {dataset_name}")
    all_pairs = get_all_psi_backend_pairs(dataset_name=dataset_name)
    print(f"[INFO] Found {len(all_pairs)} unique (psi, backend_llm) pairs\n")
    
    # Analyze all synthetic pairs
    all_synthetic_results = {}
    
    for psi, backend_llm in sorted(all_pairs, key=lambda x: sort_key_by_psi_and_size(f"{x[0]}-{x[1]}")):
        normalized_backend = normalize_backend_name(backend_llm)
        label = f"{psi}-{safe_dir_name(normalized_backend)}"
        synth_json = output_root / f'{label}_ptc_detailed.json'
        
        print(f"\n{'='*70}")
        print(f"Loading {label}...")
        print(f"{'='*70}")

        # Skip pair if detailed JSON already exists with expected conversation count
        if synth_json.exists():
            try:
                with open(synth_json, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if len(existing_data) == len(all_real_convs):
                    print(f"[SKIP] {label} already processed with {len(existing_data)} conversations")
                    all_synthetic_results[label] = pd.DataFrame(existing_data)
                    continue
                else:
                    print(
                        f"[WARNING] Existing file has {len(existing_data)} conversations, "
                        f"expected {len(all_real_convs)}. Reprocessing..."
                    )
            except Exception as e:
                print(f"[WARNING] Failed to load existing file: {e}. Reprocessing...")
        
        try:
            synthetic_df_hf = load_synthetic_hf_to_df(psi=psi, backend_llm=normalized_backend, dataset_name=dataset_name)
            if synthetic_df_hf.empty:
                print(f"  ⚠ No synthetic data found for {psi} + {normalized_backend}")
                continue
            
            synthetic_convs = []
            # Filter synthetic data to only include session_ids that match valid real conversation indices
            if exact_turns is not None and valid_indices:
                # Filter by session_id matching valid_indices
                for _, row in synthetic_df_hf.iterrows():
                    session_id = row.get('session_id', None)
                    if session_id in valid_indices:
                        messages = row['messages']
                        synthetic_convs.append((session_id, messages))
                print(f"  ✓ Loaded {len(synthetic_convs)} conversations (filtered by session_id to match real conversations)")
            else:
                # No filtering by session_id, use all conversations
                for _, row in synthetic_df_hf.iterrows():
                    messages = row['messages']
                    session_id = row.get('session_id', len(synthetic_convs))
                    if exact_turns is not None:
                        patient_turn_count = sum(1 for msg in messages 
                                               if msg.get('role') == 'assistant' and msg.get('content', '').strip())
                        if patient_turn_count != exact_turns:
                            continue
                    synthetic_convs.append((session_id, messages))
                print(f"  ✓ Loaded {len(synthetic_convs)} conversations")
            
            if not synthetic_convs:
                print(f"  ⚠ No conversations after filtering")
                continue
            
            # Analyze synthetic data
            synthetic_results = []
            for batch_start in tqdm(range(0, len(synthetic_convs), batch_size), desc=f"Classifying {label}"):
                batch_items = synthetic_convs[batch_start:batch_start + batch_size]
                batch_session_ids = [item[0] for item in batch_items]
                batch_convs = [item[1] for item in batch_items]
                
                if single_turn:
                    batch_classifications = judge.classify_turns_batch(batch_convs, num_messages=num_messages)
                else:
                    batch_classifications = judge.classify_conversations_batch(batch_convs)
                
                for session_id, classifications in zip(batch_session_ids, batch_classifications):
                    p_count = sum(1 for c in classifications if c['classification'] == 'P')
                    t_count = sum(1 for c in classifications if c['classification'] == 'T')
                    c_count = sum(1 for c in classifications if c['classification'] == 'C')
                    f_count = sum(1 for c in classifications if c['classification'] == 'F')
                    total_patient_turns = len(classifications)
                    non_filler_turns = p_count + t_count + c_count
                    
                    synthetic_results.append({
                        'conversation_id': build_conversation_id(session_id, psi=psi, backend_llm=normalized_backend),
                        'P_count': p_count,
                        'T_count': t_count,
                        'C_count': c_count,
                        'F_count': f_count,
                        'total_patient_turns': total_patient_turns,
                        'non_filler_turns': non_filler_turns,
                        'P_ratio': p_count / non_filler_turns if non_filler_turns > 0 else 0,
                        'T_ratio': t_count / non_filler_turns if non_filler_turns > 0 else 0,
                        'C_ratio': c_count / non_filler_turns if non_filler_turns > 0 else 0,
                        'F_ratio': f_count / total_patient_turns if total_patient_turns > 0 else 0,
                        'classifications': classifications
                    })
            
            synthetic_df = pd.DataFrame(synthetic_results)
            all_synthetic_results[label] = synthetic_df
            print(f"  ✓ Analyzed {len(synthetic_df)} conversations")
            
            # Save immediately after processing
            print(f"  [INFO] Saving {label} results...")
            
            # Save summary CSV (without classifications)
            synth_csv = output_root / f'{label}_ptc_summary.csv'
            synth_summary = synthetic_df.drop(columns=['classifications'])
            synth_summary.to_csv(synth_csv, index=False)
            print(f"  [CSV SAVED] {synth_csv}")
            
            # Save detailed JSON (with classifications)
            with open(synth_json, 'w', encoding='utf-8') as f:
                json.dump(synthetic_results, f, indent=2, ensure_ascii=False)
            print(f"  [JSON SAVED] {synth_json}")
            
        except Exception as e:
            print(f"  ✗ Error analyzing {psi} + {backend_llm}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save real data results
    print(f"\n{'='*70}")
    print("Saving real data results...")
    print(f"{'='*70}")
    
    # Save real data - both summary CSV and detailed JSON
    real_csv = output_root / 'real_ptc_summary.csv'
    real_summary = real_df.drop(columns=['classifications'])
    real_summary.to_csv(real_csv, index=False)
    print(f"[CSV SAVED] {real_csv}")
    
    with open(real_json, 'w', encoding='utf-8') as f:
        json.dump(real_results, f, indent=2, ensure_ascii=False)
    print(f"[JSON SAVED] {real_json}")
    
    # Synthetic datasets are now saved immediately after processing (no need to save again here)
    
    # Create multi-pair visualization
    print("\n[INFO] Creating multi-pair visualization...")
    visualize_all_distributions(real_df, all_synthetic_results, output_root, exact_turns=exact_turns)
    
    # Create progression visualization
    print("\n[INFO] Creating progression visualization...")
    visualize_all_progressions(real_df, all_synthetic_results, output_root, turn_threshold=turn_threshold, exact_turns=exact_turns)
    
    # Create PTC percentage by turn visualizations
    print("\n[INFO] Creating PTC percentage by turn visualizations...")
    visualize_ptc_percentages_by_turn(real_df, all_synthetic_results, output_root, turn_threshold=turn_threshold, exact_turns=exact_turns, no_filler=False)
    
    # Also create no-filler version
    print("\n[INFO] Creating PTC percentage by turn visualizations (no filler)...")
    visualize_ptc_percentages_by_turn(real_df, all_synthetic_results, output_root, turn_threshold=turn_threshold, exact_turns=exact_turns, no_filler=True)
    
    # Create comparison summary
    print("\n[INFO] Creating comparison summary...")
    summary_data = []
    
    # Add real data
    real_stats = {
        'dataset': 'Real',
        'display_name': 'Real',
        'P_mean': real_df['P_ratio'].mean(),
        'T_mean': real_df['T_ratio'].mean(),
        'C_mean': real_df['C_ratio'].mean(),
        'F_mean': real_df['F_ratio'].mean(),
        'num_conversations': len(real_df)
    }
    summary_data.append(real_stats)
    
    # Add synthetic data
    for label in sorted(all_synthetic_results.keys(), key=sort_key_by_psi_and_size):
        synth_df = all_synthetic_results[label]
        
        # Create display label
        display_label = label
        for psi in PSI_ABBREV.keys():
            if label.startswith(psi):
                backend_part = label[len(psi):].lstrip('-_')
                backend_part = shorten_backend_name(backend_part)
                display_label = f"{PSI_ABBREV[psi]}-{backend_part}"
                break
        
        synth_stats = {
            'dataset': label,
            'display_name': display_label,
            'P_mean': synth_df['P_ratio'].mean(),
            'T_mean': synth_df['T_ratio'].mean(),
            'C_mean': synth_df['C_ratio'].mean(),
            'F_mean': synth_df['F_ratio'].mean(),
            'num_conversations': len(synth_df)
        }
        summary_data.append(synth_stats)
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = output_root / 'ptc_summary_all_pairs.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"[CSV SAVED] {summary_csv}")
    
    # Print summary table
    print(f"\n{'='*70}")
    print("PTC Distribution Summary (All Pairs)")
    print(f"{'='*70}")
    print(summary_df[['display_name', 'P_mean', 'T_mean', 'C_mean', 'F_mean', 'num_conversations']].to_string(index=False))
    print()
    
    # Calculate Jensen-Shannon distances for all synthetic datasets
    print("\n[INFO] Calculating Jensen-Shannon distances...")
    js_distances = []
    
    # Get real PTCF ratios
    real_ptcf = {
        'P': real_df['P_count'].sum() / real_df['total_patient_turns'].sum(),
        'T': real_df['T_count'].sum() / real_df['total_patient_turns'].sum(),
        'C': real_df['C_count'].sum() / real_df['total_patient_turns'].sum(),
        'F': real_df['F_count'].sum() / real_df['total_patient_turns'].sum()
    }
    
    for label in sorted(all_synthetic_results.keys(), key=sort_key_by_psi_and_size):
        synth_df = all_synthetic_results[label]
        
        # Get synthetic PTCF ratios
        synth_ptcf = {
            'P': synth_df['P_count'].sum() / synth_df['total_patient_turns'].sum(),
            'T': synth_df['T_count'].sum() / synth_df['total_patient_turns'].sum(),
            'C': synth_df['C_count'].sum() / synth_df['total_patient_turns'].sum(),
            'F': synth_df['F_count'].sum() / synth_df['total_patient_turns'].sum()
        }
        
        # Calculate JS distance
        js_dist = calculate_jensenshannon_distance(real_ptcf, synth_ptcf)
        
        # Get display name
        display_name = label
        for psi in PSI_ABBREV.keys():
            if label.startswith(psi):
                backend = label[len(psi)+1:]  # +1 for the dash
                display_name = f"{PSI_ABBREV[psi]}-{shorten_backend_name(backend)}"
                break
        
        js_distances.append({
            'dataset': label,
            'display_name': display_name,
            'js_distance': js_dist
        })
        
        print(f"  {display_name}: {js_dist:.4f}")
    
    # Save Jensen-Shannon distances to CSV
    if js_distances:
        js_df = pd.DataFrame(js_distances)
        js_csv_path = output_root / 'jensenshannon_distances.csv'
        js_df.to_csv(js_csv_path, index=False)
        print(f"\n[CSV SAVED] {js_csv_path}")
    
    print(f"\n[DONE] Multi-pair PTC analysis complete. Results saved to: {output_root}")
    print(f"       Analyzed {len(all_synthetic_results)} synthetic datasets")


def calculate_jensenshannon_distance(real_ptcf: Dict[str, float], synthetic_ptcf: Dict[str, float], no_filler: bool = False) -> float:
    """Calculate Jensen-Shannon distance between two PTCF distributions.
    
    Args:
        real_ptcf: Real data PTCF ratios as dict (e.g., {'P': 0.3, 'T': 0.2, 'C': 0.4, 'F': 0.1})
        synthetic_ptcf: Synthetic data PTCF ratios as dict
        no_filler: If True, exclude filler from the comparison and renormalize
    
    Returns:
        Jensen-Shannon distance (0 = identical, 1 = completely different)
    """
    # Get all categories
    if no_filler:
        all_categories = sorted(['P', 'T', 'C'])
    else:
        all_categories = sorted(['P', 'T', 'C', 'F'])
    
    # Convert to probability distributions
    real_dist = np.array([real_ptcf.get(cat, 0) for cat in all_categories])
    synthetic_dist = np.array([synthetic_ptcf.get(cat, 0) for cat in all_categories])
    
    # Normalize to ensure they sum to 1.0 (handle floating point errors)
    real_dist = real_dist / real_dist.sum() if real_dist.sum() > 0 else real_dist
    synthetic_dist = synthetic_dist / synthetic_dist.sum() if synthetic_dist.sum() > 0 else synthetic_dist
    
    # Calculate Jensen-Shannon distance
    js_distance = distance.jensenshannon(real_dist, synthetic_dist)
    
    return float(js_distance)


def compare_distributions(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, output_dir: Path, judge: PTCClassifier = None):
    """Compare PTC distributions between real and synthetic data.
    
    Args:
        real_df: DataFrame with real conversation analysis
        synthetic_df: DataFrame with synthetic conversation analysis
        output_dir: Directory to save comparison results
        judge: PTCClassifier instance for model settings
    """
    # Calculate aggregate statistics (P, T, C averaged over non-filler turns only)
    real_stats = {
        'P_mean': real_df['P_ratio'].mean(),  # Already calculated over non-filler turns
        'T_mean': real_df['T_ratio'].mean(),
        'C_mean': real_df['C_ratio'].mean(),
        'F_mean': real_df['F_ratio'].mean(),  # Calculated over all turns
        'P_std': real_df['P_ratio'].std(),
        'T_std': real_df['T_ratio'].std(),
        'C_std': real_df['C_ratio'].std(),
        'F_std': real_df['F_ratio'].std(),
        'total_turns': real_df['total_patient_turns'].sum(),
        'non_filler_turns': real_df['non_filler_turns'].sum()
    }
    
    synthetic_stats = {
        'P_mean': synthetic_df['P_ratio'].mean(),  # Already calculated over non-filler turns
        'T_mean': synthetic_df['T_ratio'].mean(),
        'C_mean': synthetic_df['C_ratio'].mean(),
        'F_mean': synthetic_df['F_ratio'].mean(),  # Calculated over all turns
        'P_std': synthetic_df['P_ratio'].std(),
        'T_std': synthetic_df['T_ratio'].std(),
        'C_std': synthetic_df['C_ratio'].std(),
        'F_std': synthetic_df['F_ratio'].std(),
        'total_turns': synthetic_df['total_patient_turns'].sum(),
        'non_filler_turns': synthetic_df['non_filler_turns'].sum()
    }
    
    # Create comparison DataFrame (P, T, C are proportions of non-filler turns)
    comparison = pd.DataFrame({
        'Real_Mean': [real_stats['P_mean'], real_stats['T_mean'], real_stats['C_mean']],
        'Real_Std': [real_stats['P_std'], real_stats['T_std'], real_stats['C_std']],
        'Synthetic_Mean': [synthetic_stats['P_mean'], synthetic_stats['T_mean'], synthetic_stats['C_mean']],
        'Synthetic_Std': [synthetic_stats['P_std'], synthetic_stats['T_std'], synthetic_stats['C_std']],
        'Difference': [
            synthetic_stats['P_mean'] - real_stats['P_mean'],
            synthetic_stats['T_mean'] - real_stats['T_mean'],
            synthetic_stats['C_mean'] - real_stats['C_mean']
        ]
    }, index=['Problem (P)', 'Transition (T)', 'Change (C)'])
    
    with open(output_dir / 'ptc_comparison.txt', 'w') as f:
        f.write("PTC Distribution Comparison: Real vs Synthetic\n")
        f.write("=" * 70 + "\n\n")
        
        # Write model settings if judge is provided
        if judge:
            f.write("--- Model Settings ---\n")
            f.write(f"Model: {judge.model_name}\n")
            f.write(f"Temperature: {judge.temperature}\n")
            f.write("\n")
        
        f.write("Note: P, T, C ratios are calculated over non-filler turns only\n")
        f.write("=" * 70 + "\n\n")
        f.write(comparison.to_string())
        f.write(f"\n\n--- Turn Statistics ---")
        f.write(f"\nTotal Real Turns: {real_stats['total_turns']}")
        f.write(f"\nNon-Filler Real Turns: {real_stats['non_filler_turns']}")
        f.write(f"\nReal Filler Ratio: {real_stats['F_mean']:.3f}")
        f.write(f"\n\nTotal Synthetic Turns: {synthetic_stats['total_turns']}")
        f.write(f"\nNon-Filler Synthetic Turns: {synthetic_stats['non_filler_turns']}")
        f.write(f"\nSynthetic Filler Ratio: {synthetic_stats['F_mean']:.3f}\n")
    
    print("\nPTC Distribution Comparison:")
    print(comparison)
    
    return comparison


def get_turn_classifications(df: pd.DataFrame) -> pd.DataFrame:
    """Extract turn-by-turn classifications from analysis results.
    
    Args:
        df: DataFrame with conversation analysis including classifications
        
    Returns:
        DataFrame with columns: conversation_id, turn_index, classification
    """
    turn_data = []
    
    for idx, row in df.iterrows():
        classifications = row['classifications']
        # Parse JSON/literal string if needed (when loading from CSV)
        if isinstance(classifications, str):
            try:
                classifications = ast.literal_eval(classifications)
            except (ValueError, SyntaxError):
                classifications = json.loads(classifications)
        
        for turn_idx, turn in enumerate(classifications):
            # Use turn_index from classification if available, otherwise use enumerate index
            actual_turn_idx = turn.get('turn_index', turn_idx) if isinstance(turn, dict) else turn_idx
            turn_data.append({
                'conversation_id': row.get('conversation_id', idx),
                'turn_index': actual_turn_idx,
                'classification': turn['classification']
            })
    
    return pd.DataFrame(turn_data)


def visualize_all_distributions(real_df: pd.DataFrame, all_synthetic_results: Dict[str, pd.DataFrame], 
                                output_dir: Path, exact_turns: int = None, no_filler: bool = False):
    """Create horizontal stacked bar chart comparing PTC distributions across all datasets.
    
    Args:
        real_df: DataFrame with real conversation analysis
        all_synthetic_results: Dictionary mapping label -> DataFrame with synthetic results
        output_dir: Directory to save plots
        exact_turns: If provided, label plots to reflect exact patient-turn filtering
        no_filler: If True, exclude filler from graph and renormalize P, T, C to 100%
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate PTCF ratios for all datasets
    dataset_stats = []
    
    # Real data
    real_p = real_df['P_count'].sum() / real_df['total_patient_turns'].sum()
    real_t = real_df['T_count'].sum() / real_df['total_patient_turns'].sum()
    real_c = real_df['C_count'].sum() / real_df['total_patient_turns'].sum()
    real_f = real_df['F_count'].sum() / real_df['total_patient_turns'].sum()
    
    if no_filler:
        # Renormalize P, T, C to add up to 100% (excluding filler)
        ptc_sum = real_p + real_t + real_c
        if ptc_sum > 0:
            dataset_stats.append({
                'label': 'Real',
                'display_name': 'Real',
                'P': real_p / ptc_sum,
                'T': real_t / ptc_sum,
                'C': real_c / ptc_sum,
                'F': 0.0
            })
        else:
            dataset_stats.append({
                'label': 'Real',
                'display_name': 'Real',
                'P': 0.0,
                'T': 0.0,
                'C': 0.0,
                'F': 0.0
            })
    else:
        dataset_stats.append({
            'label': 'Real',
            'display_name': 'Real',
            'P': real_p,
            'T': real_t,
            'C': real_c,
            'F': real_f
        })
    
    # Synthetic data - sorted by PSI and model size
    for label in sorted(all_synthetic_results.keys(), key=sort_key_by_psi_and_size):
        synth_df = all_synthetic_results[label]
        
        synth_p = synth_df['P_count'].sum() / synth_df['total_patient_turns'].sum()
        synth_t = synth_df['T_count'].sum() / synth_df['total_patient_turns'].sum()
        synth_c = synth_df['C_count'].sum() / synth_df['total_patient_turns'].sum()
        synth_f = synth_df['F_count'].sum() / synth_df['total_patient_turns'].sum()
        
        # Create display label
        display_label = label
        for psi in PSI_ABBREV.keys():
            if label.startswith(psi):
                backend_part = label[len(psi):].lstrip('-_')
                backend_part = shorten_backend_name(backend_part)
                display_label = f"{PSI_ABBREV[psi]}-{backend_part}"
                break
        
        if no_filler:
            # Renormalize P, T, C to add up to 100% (excluding filler)
            ptc_sum = synth_p + synth_t + synth_c
            if ptc_sum > 0:
                dataset_stats.append({
                    'label': label,
                    'display_name': display_label,
                    'P': synth_p / ptc_sum,
                    'T': synth_t / ptc_sum,
                    'C': synth_c / ptc_sum,
                    'F': 0.0
                })
            else:
                dataset_stats.append({
                    'label': label,
                    'display_name': display_label,
                    'P': 0.0,
                    'T': 0.0,
                    'C': 0.0,
                    'F': 0.0
                })
        else:
            dataset_stats.append({
                'label': label,
                'display_name': display_label,
                'P': synth_p,
                'T': synth_t,
                'C': synth_c,
                'F': synth_f
            })
    
    # Create horizontal stacked bar chart
    fig_height = max(6, len(dataset_stats) * 0.4)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    # Extract data for plotting (reversed for top-to-bottom display)
    dataset_labels = [d['display_name'] for d in dataset_stats][::-1]
    p_values = [d['P'] for d in dataset_stats][::-1]
    t_values = [d['T'] for d in dataset_stats][::-1]
    c_values = [d['C'] for d in dataset_stats][::-1]
    f_values = [d['F'] for d in dataset_stats][::-1]
    
    # Define colors matching the vertical chart style
    colors = {'P': '#e74c3c', 'T': '#f39c12', 'C': '#27ae60', 'F': '#95a5a6'}
    
    # Plot horizontal stacked bars
    left = [0] * len(dataset_labels)
    
    # Problem
    ax.barh(dataset_labels, p_values, left=left, label='Problem', 
            color=colors['P'], alpha=0.85)
    left = [l + p for l, p in zip(left, p_values)]
    
    # Transition
    ax.barh(dataset_labels, t_values, left=left, label='Transition', 
            color=colors['T'], alpha=0.85)
    left = [l + t for l, t in zip(left, t_values)]
    
    # Change
    ax.barh(dataset_labels, c_values, left=left, label='Change', 
            color=colors['C'], alpha=0.85)
    left = [l + c for l, c in zip(left, c_values)]
    
    # Filler (only if not excluded)
    if not no_filler:
        ax.barh(dataset_labels, f_values, left=left, label='Filler', 
                color=colors['F'], alpha=0.85)
    
    # Add percentage labels for segments >= 5%
    for i, (p, t, c, f) in enumerate(zip(p_values, t_values, c_values, f_values)):
        y_pos = i
        # Problem
        if p >= 0.05:
            ax.text(p/2, y_pos, f'{p*100:.1f}%', ha='center', va='center', 
                   fontsize=12, color='white', fontweight='bold')
        # Transition
        if t >= 0.05:
            ax.text(p + t/2, y_pos, f'{t*100:.1f}%', ha='center', va='center', 
                   fontsize=12, color='white', fontweight='bold')
        # Change
        if c >= 0.05:
            ax.text(p + t + c/2, y_pos, f'{c*100:.1f}%', ha='center', va='center', 
                   fontsize=12, color='white', fontweight='bold')
        # Filler (only if not excluded)
        if not no_filler and f >= 0.05:
            ax.text(p + t + c + f/2, y_pos, f'{f*100:.1f}%', ha='center', va='center', 
                   fontsize=12, color='white', fontweight='bold')
    
    ax.set_xlabel('Ratio', fontsize=18)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='x', labelsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16, title='Categories')
    ax.grid(axis='x', alpha=0.3)
    
    title_suffix = f" (exactly {exact_turns} patient turns)" if exact_turns else ""
    # ax.set_title(f'PTC Distribution {title_suffix}', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    filename_suffix = '_no_filler' if no_filler else ''
    plt.savefig(output_dir / f'ptc_distribution_all_pairs{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT SAVED] {output_dir / f'ptc_distribution_all_pairs{filename_suffix}.png'}")


def visualize_all_progressions(real_df: pd.DataFrame, all_synthetic_results: Dict[str, pd.DataFrame],
                                output_dir: Path, turn_threshold: int = 12, exact_turns: int = None):
    """Create multi-line progression plot comparing all datasets.
    
    Args:
        real_df: DataFrame with real conversation analysis
        all_synthetic_results: Dictionary mapping label -> DataFrame with synthetic results
        output_dir: Directory to save plots
        turn_threshold: Maximum turn index to display
        exact_turns: If provided, label plots to reflect exact patient-turn filtering
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get turn-by-turn data for real
    real_turns = get_turn_classifications(real_df)
    real_turns = real_turns[real_turns['turn_index'] < turn_threshold].copy()
    
    # Map classifications to numeric values: P=1, T=2, C=3, F=NaN
    ptc_map = {'P': 1, 'T': 2, 'C': 3, 'F': np.nan}
    real_turns['ptc_numeric'] = real_turns['classification'].map(ptc_map)
    real_avg = real_turns.groupby('turn_index')['ptc_numeric'].mean().reset_index()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Plot real data
    ax.step(real_avg['turn_index'], real_avg['ptc_numeric'],
            where='mid', label='Real', linewidth=2.5, alpha=0.9, color='black', marker='|', markersize=5)
    
    # Build family-aware marker map once and reuse across plots/legend.
    backend_by_label = {label: extract_backend_name_from_label(label) for label in all_synthetic_results.keys()}
    sorted_backends = sorted(set(backend_by_label.values()), key=sort_key_by_backend_family_and_size)
    backend_markers = assign_backend_markers(sorted_backends)
    
    # Collect all model sizes for opacity normalization
    all_model_sizes = [extract_model_size(backend) for backend in sorted_backends if extract_model_size(backend) > 0]
    
    used_psi_types = set()
    used_backends = set()
    
    # Plot synthetic data for each variant
    for label in sorted(all_synthetic_results.keys(), key=sort_key_by_psi_and_size):
        synth_df = all_synthetic_results[label]
        
        # Get turn-by-turn data
        synth_turns = get_turn_classifications(synth_df)
        synth_turns = synth_turns[synth_turns['turn_index'] < turn_threshold].copy()
        synth_turns['ptc_numeric'] = synth_turns['classification'].map(ptc_map)
        synth_avg = synth_turns.groupby('turn_index')['ptc_numeric'].mean().reset_index()
        
        # Determine PSI type and backend
        psi_type = None
        backend_llm = backend_by_label.get(label, 'unknown')
        for psi in PSI_COLORS.keys():
            if label.startswith(psi):
                psi_type = psi
                break
        
        if not psi_type:
            psi_type = 'unknown'
            backend_llm = label
        
        # Get color and marker
        color = PSI_COLORS.get(psi_type, '#7f7f7f')
        marker = backend_markers.get(backend_llm, 'o')
        backend_name = backend_llm if backend_llm in backend_markers else None
        alpha = get_model_opacity(backend_llm, all_model_sizes)  # Get opacity based on model size
        
        used_psi_types.add(psi_type)
        if backend_name:
            used_backends.add(backend_name)
        
        # Plot line
        ax.step(synth_avg['turn_index'], synth_avg['ptc_numeric'],
                where='mid', linewidth=2, alpha=alpha, color=color, marker=marker, markersize=5)
    
    # Create legends
    from matplotlib.lines import Line2D
    
    # Legend for PSI types (colors)
    color_legend_elements = [Line2D([0], [0], color='black', linewidth=2, label=PSI_LABELS['Real'])]
    for psi_type in sorted(used_psi_types):
        if psi_type in PSI_COLORS:
            color_legend_elements.append(
                Line2D([0], [0], color=PSI_COLORS[psi_type], linewidth=2, label=PSI_LABELS.get(psi_type, psi_type))
            )
    
    # Legend for backends (markers)
    marker_legend_elements = []
    for backend in sorted(used_backends, key=sort_key_by_backend_family_and_size):
        if backend in backend_markers:
            marker_legend_elements.append(
                Line2D([0], [0], color='gray', marker=backend_markers[backend],
                       linestyle='None', markersize=8, label=shorten_backend_name(backend))
            )
    
    # ACL-style formatting
    ax.set_xlabel('Turn Index', fontsize=18)
    # ax.set_ylabel('PTC Classification', fontsize=18)
    title_suffix = f" (exactly {exact_turns} patient turns)" if exact_turns else ""
    # ax.set_title(f'PTC Progression {title_suffix}', fontsize=18, fontweight='bold')
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Problem', 'Transition', 'Change'], fontsize=20)
    ax.set_ylim([0.5, 3.5])
    ax.set_xlim([-0.5, turn_threshold - 0.5])
    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.grid(False)
    
    # Force x-axis to show only integer values
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Place legends
    legend1 = ax.legend(handles=color_legend_elements, fontsize=20,
                        bbox_to_anchor=(0.5, 1.05), loc='lower center', ncol=max(3, len(color_legend_elements)),
                        frameon=False, handletextpad=0.2, columnspacing=1.0)
    ax.add_artist(legend1)
    
    if marker_legend_elements:
        legend2 = ax.legend(handles=marker_legend_elements, fontsize=20,
                            bbox_to_anchor=(0.5, 0.98), loc='lower center', ncol=max(2, len(marker_legend_elements)),
                            frameon=False, handletextpad=0.2, columnspacing=1.0)
        ax.add_artist(legend2)
        extra_artists = [legend1, legend2]
    else:
        extra_artists = [legend1]
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ptc_progression_all_pairs.png', dpi=300, bbox_inches='tight',
                bbox_extra_artists=extra_artists, pad_inches=0.1)
    plt.close()
    print(f"[PLOT SAVED] {output_dir / 'ptc_progression_all_pairs.png'}")


def visualize_ptc_percentages_by_turn(real_df: pd.DataFrame, all_synthetic_results: Dict[str, pd.DataFrame],
                                       output_dir: Path, turn_threshold: int = 12, exact_turns: int = None,
                                       pre_calculated_percentages: Dict[str, Dict[int, Dict[str, float]]] = None,
                                       no_filler: bool = False):
    """Create one figure with three subplots showing percentage of P, T, and C at each turn for all models.
    
    Args:
        real_df: DataFrame with real conversation analysis
        all_synthetic_results: Dictionary mapping label -> DataFrame with synthetic results
        output_dir: Directory to save plots
        turn_threshold: Maximum turn index to display
        exact_turns: If provided, label plots to reflect exact patient-turn filtering
        pre_calculated_percentages: Optional pre-calculated percentages to use instead of computing from data.
                                   Format: {dataset: {turn_index: {'P': pct, 'T': pct, 'C': pct}}}
        no_filler: If True, rescale P, T, C to sum to 100% (excluding filler)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build family-aware marker map once and reuse across plots/legend.
    backend_by_label = {label: extract_backend_name_from_label(label) for label in all_synthetic_results.keys()}
    sorted_backends = sorted(set(backend_by_label.values()), key=sort_key_by_backend_family_and_size)
    backend_markers = assign_backend_markers(sorted_backends)
    
    # Collect all model sizes for opacity normalization
    all_model_sizes = [extract_model_size(backend) for backend in sorted_backends if extract_model_size(backend) > 0]
    
    # Calculate percentage data for each category (P, T, C)
    categories = ['P', 'T', 'C']
    category_colors = {'P': '#e74c3c', 'T': '#f39c12', 'C': '#27ae60'}
    category_names = {'P': 'Problem', 'T': 'Transition', 'C': 'Change'}
    
    # Use pre-calculated percentages if provided, otherwise calculate from data
    if pre_calculated_percentages is not None:
        print("[INFO] Using pre-calculated percentages from saved file")
        all_percentages = pre_calculated_percentages
        # Rescale if no_filler is True
        if no_filler:
            for dataset in all_percentages:
                for turn_idx in all_percentages[dataset]:
                    ptc_sum = all_percentages[dataset][turn_idx].get('P', 0) + \
                              all_percentages[dataset][turn_idx].get('T', 0) + \
                              all_percentages[dataset][turn_idx].get('C', 0)
                    if ptc_sum > 0:
                        all_percentages[dataset][turn_idx] = {
                            'P': (all_percentages[dataset][turn_idx].get('P', 0) / ptc_sum) * 100,
                            'T': (all_percentages[dataset][turn_idx].get('T', 0) / ptc_sum) * 100,
                            'C': (all_percentages[dataset][turn_idx].get('C', 0) / ptc_sum) * 100
                        }
    else:
        # Calculate percentages from raw data
        all_percentages = {}
        
        # Calculate for real data
        real_turns = get_turn_classifications(real_df)
        real_turns = real_turns[real_turns['turn_index'] < turn_threshold].copy()
        all_percentages['real'] = {}
        for turn_idx in range(turn_threshold):
            turn_data = real_turns[real_turns['turn_index'] == turn_idx]
            total = len(turn_data)
            if total > 0:
                p_count = (turn_data['classification'] == 'P').sum()
                t_count = (turn_data['classification'] == 'T').sum()
                c_count = (turn_data['classification'] == 'C').sum()
                
                if no_filler:
                    # Rescale to exclude filler: only count P, T, C in denominator
                    ptc_total = p_count + t_count + c_count
                    if ptc_total > 0:
                        all_percentages['real'][turn_idx] = {
                            'P': (p_count / ptc_total) * 100,
                            'T': (t_count / ptc_total) * 100,
                            'C': (c_count / ptc_total) * 100
                        }
                    else:
                        all_percentages['real'][turn_idx] = {'P': 0, 'T': 0, 'C': 0}
                else:
                    # Include filler in denominator
                    all_percentages['real'][turn_idx] = {
                        'P': (p_count / total) * 100,
                        'T': (t_count / total) * 100,
                        'C': (c_count / total) * 100
                    }
            else:
                all_percentages['real'][turn_idx] = {'P': 0, 'T': 0, 'C': 0}
        
        # Calculate for synthetic data
        for label in sorted(all_synthetic_results.keys(), key=sort_key_by_psi_and_size):
            synth_df = all_synthetic_results[label]
            synth_turns = get_turn_classifications(synth_df)
            synth_turns = synth_turns[synth_turns['turn_index'] < turn_threshold].copy()
            all_percentages[label] = {}
            for turn_idx in range(turn_threshold):
                turn_data = synth_turns[synth_turns['turn_index'] == turn_idx]
                total = len(turn_data)
                if total > 0:
                    p_count = (turn_data['classification'] == 'P').sum()
                    t_count = (turn_data['classification'] == 'T').sum()
                    c_count = (turn_data['classification'] == 'C').sum()
                    
                    if no_filler:
                        # Rescale to exclude filler: only count P, T, C in denominator
                        ptc_total = p_count + t_count + c_count
                        if ptc_total > 0:
                            all_percentages[label][turn_idx] = {
                                'P': (p_count / ptc_total) * 100,
                                'T': (t_count / ptc_total) * 100,
                                'C': (c_count / ptc_total) * 100
                            }
                        else:
                            all_percentages[label][turn_idx] = {'P': 0, 'T': 0, 'C': 0}
                    else:
                        # Include filler in denominator
                        all_percentages[label][turn_idx] = {
                            'P': (p_count / total) * 100,
                            'T': (t_count / total) * 100,
                            'C': (c_count / total) * 100
                        }
                else:
                    all_percentages[label][turn_idx] = {'P': 0, 'T': 0, 'C': 0}
        
        # Save calculated percentages for faster redraw
        save_ptc_percentages(all_percentages, output_dir, turn_threshold, no_filler=no_filler)
    
    # Create figure with 3 subplots (1 row, 3 columns).
    # Target layout: wider figure with moderate height.
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    
    used_psi_types = set()
    used_backends = set()
    
    for idx, category in enumerate(categories):
        ax = axes[idx]
        
        # Plot real data using pre-calculated percentages
        real_percentages = [all_percentages['real'][turn_idx][category] 
                           for turn_idx in range(turn_threshold) 
                           if turn_idx in all_percentages['real']]
        
        # Plot real data
        ax.plot(range(len(real_percentages)), real_percentages, 
            linewidth=3.2, alpha=0.9, color='black', marker='o', markersize=8)
        
        # Plot synthetic data using pre-calculated percentages
        for label in sorted(all_synthetic_results.keys(), key=sort_key_by_psi_and_size):
            if label not in all_percentages:
                continue
            
            synth_percentages = [all_percentages[label][turn_idx][category] 
                               for turn_idx in range(turn_threshold) 
                               if turn_idx in all_percentages[label]]
            
            # Determine PSI type and backend
            parts = label.split('-')
            psi_type = parts[0] if len(parts) > 0 else 'unknown'
            backend = backend_by_label.get(label, 'unknown')
            
            used_psi_types.add(psi_type)
            used_backends.add(backend)
            
            # Get color and marker
            color = PSI_COLORS.get(psi_type, '#95a5a6')
            marker = backend_markers.get(backend, 'o')
            
            # Determine alpha based on model size
            alpha = get_model_opacity(backend, all_model_sizes)
            
            # Plot synthetic data (no label to avoid duplicate legends)
            ax.plot(range(len(synth_percentages)), synth_percentages,
                    linewidth=3.0, alpha=alpha, color=color, 
                    marker=marker, markersize=8)
        
        # Subplot formatting
        ax.set_title(f'{category_names[category]}', fontsize=34, fontweight='bold', pad=16)
        ax.set_xlabel('Turn Index', fontsize=30)
        ax.set_ylabel('Percentage', fontsize=30)
        ax.set_ylim([0, 100])
        ax.set_xlim([-0.5, turn_threshold - 0.5])
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.grid(True, alpha=0.3)
        
        # Force x-axis to show only integer values
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Create legends (only once for the entire figure)
    from matplotlib.lines import Line2D
    
    # Legend for PSI types (colors)
    color_legend_elements = [Line2D([0], [0], color='black', linewidth=4, label=PSI_LABELS['Real'])]
    for psi_type in sorted(used_psi_types):
        psi_label = PSI_LABELS.get(psi_type, psi_type)
        color_legend_elements.append(
            Line2D([0], [0], color=PSI_COLORS.get(psi_type, '#95a5a6'), linewidth=4, label=psi_label)
        )
    
    # Legend for backends (markers)
    marker_legend_elements = []
    for backend in sorted(used_backends, key=sort_key_by_backend_family_and_size):
        short_name = shorten_backend_name(backend)
        marker_legend_elements.append(
            Line2D([0], [0], color='gray', marker=backend_markers.get(backend, 'o'), 
                   linestyle='None', markersize=15, label=short_name)
        )
    
    # Place legends above the figure
    legend1 = fig.legend(handles=color_legend_elements, fontsize=32,
                        bbox_to_anchor=(0.5, 1.00), loc='lower center', ncol=max(3, len(color_legend_elements)),
                        frameon=False, handletextpad=0.4, columnspacing=1.8)
    
    if marker_legend_elements:
        marker_ncol = max(1, (len(marker_legend_elements) + 1) // 2)
        extra_artists = [legend1, fig.legend(handles=marker_legend_elements, fontsize=32,
                                             bbox_to_anchor=(0.5, 1.12), loc='lower center', 
                                             ncol=marker_ncol,
                                             frameon=False, handletextpad=0.4, columnspacing=1.8,
                                             labelspacing=0.6)]
    else:
        extra_artists = [legend1]
    
    # Keep subplots readable at the shorter height while preserving legend space.
    fig.subplots_adjust(left=0.05, right=0.995, bottom=0.16, top=0.9, wspace=0.32)
    filename_suffix = '_no_filler' if no_filler else ''
    output_filename = output_dir / f'ptc_percentages_by_turn{filename_suffix}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight',
                bbox_extra_artists=extra_artists, pad_inches=0.1)
    plt.savefig(output_dir / f'ptc_percentages_by_turn{filename_suffix}.pdf', bbox_inches='tight')
    plt.close()
    print(f"[PLOT SAVED] {output_filename}")


def save_ptc_percentages(all_percentages: Dict[str, Dict[int, Dict[str, float]]],
                        output_dir: Path, turn_threshold: int, no_filler: bool = False):
    """Save calculated PTC percentages to CSV for faster redraw.
    
    Args:
        all_percentages: Dict mapping dataset -> {turn_index: {'P': pct, 'T': pct, 'C': pct}}
        output_dir: Directory to save the file
        turn_threshold: Maximum turn index included in the data
        no_filler: If True, percentages are rescaled excluding filler
    """
    # Combine all data into a single DataFrame
    all_data = []
    
    for dataset, turn_data in all_percentages.items():
        for turn_idx, categories in turn_data.items():
            for category, percentage in categories.items():
                all_data.append({
                    'dataset': dataset,
                    'turn_index': int(turn_idx),
                    'category': category,
                    'percentage': float(percentage)
                })
    
    # Save to CSV
    percentages_df = pd.DataFrame(all_data)
    filename_suffix = '_no_filler' if no_filler else ''
    output_file = output_dir / f'ptc_percentages_by_turn_t{turn_threshold}{filename_suffix}.csv'
    percentages_df.to_csv(output_file, index=False)
    print(f"[CSV SAVED] PTC percentages: {output_file}")


def load_ptc_percentages(csv_dir: Path, turn_threshold: int, no_filler: bool = False) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Load pre-calculated PTC percentages from CSV.
    
    Args:
        csv_dir: Directory containing the saved percentages file
        turn_threshold: Maximum turn index to load
        no_filler: If True, look for file with no_filler suffix
        
    Returns:
        Dict mapping dataset -> {turn_index: {'P': pct, 'T': pct, 'C': pct}}
        Returns None if file doesn't exist or is invalid
    """
    filename_suffix = '_no_filler' if no_filler else ''
    percentages_file = csv_dir / f'ptc_percentages_by_turn_t{turn_threshold}{filename_suffix}.csv'
    
    if not percentages_file.exists():
        return None
    
    try:
        df = pd.read_csv(percentages_file)
        
        # Validate required columns
        required_cols = ['dataset', 'turn_index', 'category', 'percentage']
        if not all(col in df.columns for col in required_cols):
            print(f"[WARNING] Invalid percentages file format, missing required columns")
            return None
        
        # Convert to nested dict structure
        result = {}
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            result[dataset] = {}
            
            for turn_idx in dataset_df['turn_index'].unique():
                turn_df = dataset_df[dataset_df['turn_index'] == turn_idx]
                result[dataset][int(turn_idx)] = {}
                
                for _, row in turn_df.iterrows():
                    result[dataset][int(turn_idx)][row['category']] = float(row['percentage'])
        
        print(f"[INFO] Loaded pre-calculated percentages from {percentages_file}")
        return result
        
    except Exception as e:
        print(f"[WARNING] Failed to load percentages file: {e}")
        return None


def visualize_turn_counts_by_model(real_df: pd.DataFrame, all_synthetic_results: Dict[str, pd.DataFrame],
                                     output_dir: Path, turn_threshold: int = 12, exact_turns: int = None):
    """Create separate bar charts for each model showing PTC counts per turn.
    
    Args:
        real_df: DataFrame with real conversation analysis
        all_synthetic_results: Dictionary mapping label -> DataFrame with synthetic results
        output_dir: Directory to save plots
        turn_threshold: Maximum turn index to display
        exact_turns: If provided, label plots to reflect exact patient-turn filtering
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count_plots_dir = output_dir / 'turn_count_plots'
    count_plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Define colors for P, T, C
    colors = {'P': '#e74c3c', 'T': '#f39c12', 'C': '#27ae60'}
    
    # Process all datasets (real + synthetic), Real first
    all_datasets = [('Real', real_df)] + sorted([(label, df) for label, df in all_synthetic_results.items()], 
                                                  key=lambda x: sort_key_by_psi_and_size(x[0]))
    
    # Prepare data for all models (for collage)
    all_turn_counts = []
    all_display_labels = []
    
    for dataset_name, df in all_datasets:
        # Get turn-by-turn classifications
        turn_data = get_turn_classifications(df)
        turn_data = turn_data[turn_data['turn_index'] < turn_threshold].copy()
        
        # Count P, T, C for each turn
        turn_counts = turn_data.groupby(['turn_index', 'classification']).size().unstack(fill_value=0)
        
        # Ensure we have all categories (P, T, C)
        for cat in ['P', 'T', 'C']:
            if cat not in turn_counts.columns:
                turn_counts[cat] = 0
        
        # Select only P, T, C columns
        turn_counts = turn_counts[['P', 'T', 'C']]
        
        # Create display label for title
        if dataset_name == 'Real':
            display_label = 'Real'
        else:
            display_label = dataset_name
            for psi in PSI_ABBREV.keys():
                if dataset_name.startswith(psi):
                    backend_part = dataset_name[len(psi):].lstrip('-_')
                    backend_part = shorten_backend_name(backend_part)
                    display_label = f"{PSI_ABBREV[psi]}-{backend_part}"
                    break
        
        all_turn_counts.append(turn_counts)
        all_display_labels.append(display_label)
        
        # Create individual bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot grouped bars
        x = turn_counts.index
        width = 0.25
        x_pos = np.arange(len(x))
        
        ax.bar(x_pos - width, turn_counts['P'], width, label='Problem', 
               color=colors['P'], alpha=0.85)
        ax.bar(x_pos, turn_counts['T'], width, label='Transition', 
               color=colors['T'], alpha=0.85)
        ax.bar(x_pos + width, turn_counts['C'], width, label='Change', 
               color=colors['C'], alpha=0.85)
        
        # Formatting
        ax.set_xlabel('Turn Index', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        
        title_suffix = f" (exactly {exact_turns} patient turns)" if exact_turns else ""
        ax.set_title(f'PTC Count per Turn: {display_label}{title_suffix}', 
                    fontsize=16, fontweight='bold')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x, fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save with safe filename
        safe_name = dataset_name.replace('/', '_').replace(' ', '_')
        filename = count_plots_dir / f'turn_counts_{safe_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"[PLOT SAVED] {len(all_datasets)} turn count plots in {count_plots_dir}")
    
    # Create collaged figure with all models
    n_models = len(all_datasets)
    # Arrange in grid: 3 columns
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols  # Ceiling division
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    
    # Flatten axes array for easier indexing
    if nrows == 1:
        axes = [axes] if ncols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, (turn_counts, display_label) in enumerate(zip(all_turn_counts, all_display_labels)):
        ax = axes[idx]
        
        # Plot grouped bars
        x = turn_counts.index
        width = 0.25
        x_pos = np.arange(len(x))
        
        ax.bar(x_pos - width, turn_counts['P'], width, label='Problem', 
               color=colors['P'], alpha=0.85)
        ax.bar(x_pos, turn_counts['T'], width, label='Transition', 
               color=colors['T'], alpha=0.85)
        ax.bar(x_pos + width, turn_counts['C'], width, label='Change', 
               color=colors['C'], alpha=0.85)
        
        # Formatting
        ax.set_xlabel('Turn Index', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{display_label}', fontsize=12, fontweight='bold')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x, fontsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
    
    # Hide extra subplots if any
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    title_suffix = f" (exactly {exact_turns} patient turns)" if exact_turns else ""
    fig.suptitle(f'PTC Count per Turn - All Models{title_suffix}', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    collage_filename = output_dir / 'turn_counts_all_models_collage.png'
    plt.savefig(collage_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[PLOT SAVED] Collaged turn count plot: {collage_filename}")



def visualize_distributions(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, 
                           output_dir: Path, turn_threshold: int = 16, exact_turns: int = None):
    """Create visualizations comparing PTC distributions.
    
    Args:
        real_df: DataFrame with real conversation analysis
        synthetic_df: DataFrame with synthetic conversation analysis
        output_dir: Directory to save plots
        turn_threshold: Maximum turn index to display in progression plots (default: 16)
        exact_turns: If provided, label plots to reflect exact patient-turn filtering
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate average counts as ratios of total turns (including fillers)
    real_p_ratio = real_df['P_count'].sum() / real_df['total_patient_turns'].sum()
    real_t_ratio = real_df['T_count'].sum() / real_df['total_patient_turns'].sum()
    real_c_ratio = real_df['C_count'].sum() / real_df['total_patient_turns'].sum()
    real_f_ratio = real_df['F_count'].sum() / real_df['total_patient_turns'].sum()
    
    synthetic_p_ratio = synthetic_df['P_count'].sum() / synthetic_df['total_patient_turns'].sum()
    synthetic_t_ratio = synthetic_df['T_count'].sum() / synthetic_df['total_patient_turns'].sum()
    synthetic_c_ratio = synthetic_df['C_count'].sum() / synthetic_df['total_patient_turns'].sum()
    synthetic_f_ratio = synthetic_df['F_count'].sum() / synthetic_df['total_patient_turns'].sum()
    
    real_means = [real_p_ratio, real_t_ratio, real_c_ratio]
    synthetic_means = [synthetic_p_ratio, synthetic_t_ratio, synthetic_c_ratio]
    real_filler_mean = real_f_ratio
    synthetic_filler_mean = synthetic_f_ratio
    
    # 1. Stacked bar chart of average P, T, C, F ratios
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    colors = ['#e74c3c', '#f39c12', '#27ae60', '#95a5a6']
    labels = ['Problem', 'Transition', 'Change', 'Filler']
    
    # Real stacked bar
    bottom = 0
    real_values = real_means + [real_filler_mean]
    for i, (value, color, label) in enumerate(zip(real_values, colors, labels)):
        axes[0].bar(0, value, bottom=bottom, label=label, alpha=0.8, color=color, width=0.5)
        # Add value label
        if value > 0.02:  # Only show label if segment is large enough
            axes[0].text(0, bottom + value/2, f'{value:.2f}', 
                    ha='center', va='center', fontsize=12, fontweight='bold')
        bottom += value
    
    axes[0].set_xlim([-0.5, 0.5])
    axes[0].set_ylim([0, 1])
    axes[0].set_ylabel('Ratio', fontsize=12)
    axes[0].set_title('Real Conversations', fontsize=14, fontweight='bold')
    axes[0].set_xticks([])
    axes[0].legend(loc='upper right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Synthetic stacked bar
    bottom = 0
    synthetic_values = synthetic_means + [synthetic_filler_mean]
    for i, (value, color, label) in enumerate(zip(synthetic_values, colors, labels)):
        axes[1].bar(0, value, bottom=bottom, label=label, alpha=0.8, color=color, width=0.5)
        # Add value label
        if value > 0.02:  # Only show label if segment is large enough
            axes[1].text(0, bottom + value/2, f'{value:.2f}', 
                        ha='center', va='center', fontsize=12, fontweight='bold')
        bottom += value
    
    axes[1].set_xlim([-0.5, 0.5])
    axes[1].set_ylim([0, 1])
    axes[1].set_ylabel('Ratio', fontsize=12)
    axes[1].set_title('Synthetic Conversations', fontsize=14, fontweight='bold')
    axes[1].set_xticks([])
    axes[1].legend(loc='upper right')
    axes[1].grid(axis='y', alpha=0.3)
    
    subtitle_suffix = f" (exactly {exact_turns} patient turns)" if exact_turns else ""
    plt.suptitle(f'Average PTCF Distribution{subtitle_suffix}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'ptc_stacked_average.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Step line plot showing PTC progression through conversation turns
    # Use all conversations, turn_threshold only controls display range
    if exact_turns:
        print(f"\nAnalyzing {len(real_df)} real and {len(synthetic_df)} synthetic conversations with exactly {exact_turns} patient turns")
    else:
        print(f"\nAnalyzing {len(real_df)} real and {len(synthetic_df)} synthetic conversations (displaying up to turn {turn_threshold})")
    
    # Get turn-by-turn data
    real_turns = get_turn_classifications(real_df)
    synthetic_turns = get_turn_classifications(synthetic_df)
    
    # Filter to first 16 turns only
    real_turns = real_turns[real_turns['turn_index'] < turn_threshold].copy()
    synthetic_turns = synthetic_turns[synthetic_turns['turn_index'] < turn_threshold].copy()
    
    # Map classifications to numeric values: P=1, T=2, C=3, F=NaN
    ptc_map = {'P': 1, 'T': 2, 'C': 3, 'F': np.nan}
    real_turns['ptc_numeric'] = real_turns['classification'].map(ptc_map)
    synthetic_turns['ptc_numeric'] = synthetic_turns['classification'].map(ptc_map)
    
    # Average PTC value per turn index across all conversations (F will be NaN and excluded)
    real_avg = real_turns.groupby('turn_index')['ptc_numeric'].mean().reset_index()
    synthetic_avg = synthetic_turns.groupby('turn_index')['ptc_numeric'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot step lines for non-filler turns only
    ax.step(real_avg['turn_index'], real_avg['ptc_numeric'], 
            where='mid', label='Real', linewidth=2.5, alpha=0.8, color='steelblue')
    ax.step(synthetic_avg['turn_index'], synthetic_avg['ptc_numeric'], 
            where='mid', label='Synthetic', linewidth=2.5, alpha=0.8, color='coral')
    
    # ACL-style formatting matching message_lengths.py
    ax.set_xlabel('Turn Index', fontsize=18)
    ax.set_ylabel('PTC Classification', fontsize=18)
    if exact_turns:
        title_text = f'Average PTC Progression (First {turn_threshold} Turns, Conversations with exactly {exact_turns} patient turns)'
    else:
        title_text = f'Average PTC Progression (First {turn_threshold} Turns, Conversations with ≥{turn_threshold} Turns)'
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Problem (P)', 'Transition (T)', 'Change (C)'], fontsize=14)
    ax.set_ylim([0.5, 3.5])
    ax.set_xlim([-0.5, turn_threshold - 0.5])
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Force x-axis to show only integer values
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.legend(fontsize=16, loc='best')
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ptc_progression_stepline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization plots saved to {output_dir}/")
    print("  - ptc_stacked_average.png: Stacked bar chart of average P/T/C/F ratios (all conversations)")
    print(f"  - ptc_progression_stepline.png: Step line plot showing PTC progression (first {turn_threshold} turns, conversations ≥{turn_threshold} turns)")


def redraw_from_csv(csv_dir: str, output_dir: str, min_turn: int = 16, turn_threshold: int = 12, 
                    exact_turns: int = None, no_filler: bool = False):
    """Redraw visualizations from existing CSV/JSON files.
    
    Args:
        csv_dir: Directory containing saved CSV and JSON files
        output_dir: Output directory for regenerated plots
        min_turn: Minimum number of patient turns required for progression analysis (default: 12)
        turn_threshold: Maximum turn index to display in visualizations (default: 16)
        exact_turns: If provided, label plots to reflect exact patient-turn filtering
        no_filler: If True, exclude filler from graphs and renormalize P, T, C to 100%
    """
    csv_path = Path(csv_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] Loading data from {csv_path}...")
    
    # Load real data - try JSON first, then fall back to CSV
    real_json = csv_path / 'real_ptc_detailed.json'
    real_csv = csv_path / 'real_ptc_summary.csv'
    real_csv_legacy = csv_path / 'real_ptc_analysis.csv'  # Legacy filename
    
    if real_json.exists():
        with open(real_json, 'r', encoding='utf-8') as f:
            real_results = json.load(f)
        real_df = pd.DataFrame(real_results)
        print(f"[INFO] Loaded real data from JSON: {len(real_df)} conversations")
    elif real_csv.exists():
        real_df = pd.read_csv(real_csv)
        print(f"[INFO] Loaded real data from summary CSV: {len(real_df)} conversations")
        print(f"[WARNING] No classifications column available (detailed JSON not found)")
    elif real_csv_legacy.exists():
        real_df = pd.read_csv(real_csv_legacy)
        print(f"[INFO] Loaded real data from legacy CSV: {len(real_df)} conversations")
        # Parse classifications if stored as string
        if 'classifications' in real_df.columns:
            real_df['classifications'] = real_df['classifications'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
    else:
        print(f"[ERROR] Real data not found in {csv_path}")
        return
    
    # Load all synthetic data - try JSON first, then fall back to CSV
    all_synthetic_results = {}
    synthetic_jsons = list(csv_path.glob('*_ptc_detailed.json'))
    synthetic_jsons = [f for f in synthetic_jsons if not f.name.startswith('real_')]
    
    # Load from JSON if available
    for synth_json in synthetic_jsons:
        label = synth_json.stem.replace('_ptc_detailed', '')
        with open(synth_json, 'r', encoding='utf-8') as f:
            synth_results = json.load(f)
        synth_df = pd.DataFrame(synth_results)
        all_synthetic_results[label] = synth_df
        print(f"[INFO] Loaded {label} from JSON: {len(synth_df)} conversations")
    
    # Load from summary CSV if JSON not available
    synthetic_csvs = list(csv_path.glob('*_ptc_summary.csv'))
    synthetic_csvs = [f for f in synthetic_csvs if not f.name.startswith('real_')]
    
    for synth_csv in synthetic_csvs:
        label = synth_csv.stem.replace('_ptc_summary', '')
        if label not in all_synthetic_results:  # Only load if not already loaded from JSON
            synth_df = pd.read_csv(synth_csv)
            all_synthetic_results[label] = synth_df
            print(f"[INFO] Loaded {label} from summary CSV: {len(synth_df)} conversations")
            print(f"[WARNING] No classifications column available for {label} (detailed JSON not found)")
    
    # Also check for legacy _ptc_analysis.csv files
    legacy_csvs = list(csv_path.glob('*_ptc_analysis.csv'))
    legacy_csvs = [f for f in legacy_csvs if f.name != 'real_ptc_analysis.csv']
    
    for synth_csv in legacy_csvs:
        label = synth_csv.stem.replace('_ptc_analysis', '')
        if label not in all_synthetic_results:  # Only load if not already loaded
            synth_df = pd.read_csv(synth_csv)
            # Parse classifications if stored as string
            if 'classifications' in synth_df.columns:
                synth_df['classifications'] = synth_df['classifications'].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
            all_synthetic_results[label] = synth_df
            print(f"[INFO] Loaded {label} from legacy CSV: {len(synth_df)} conversations")
    
    print(f"\n[INFO] Found {len(all_synthetic_results)} synthetic datasets")
    
    # Try to load pre-calculated percentages first
    pre_calculated = load_ptc_percentages(csv_path, turn_threshold, no_filler=no_filler)
    
    if pre_calculated is not None:
        # Use pre-calculated percentages (much faster)
        print("[INFO] Creating visualizations from pre-calculated percentages...")
        
        # Create multi-pair visualization (doesn't need classifications)
        print("\n[INFO] Creating multi-pair visualization...")
        visualize_all_distributions(real_df, all_synthetic_results, output_root, exact_turns=exact_turns, no_filler=no_filler)
        
        # We still need classifications for progression visualization
        datasets_with_classifications = {}
        real_has_classifications = 'classifications' in real_df.columns
        for label, df in all_synthetic_results.items():
            if 'classifications' in df.columns:
                datasets_with_classifications[label] = df
        
        # Create progression visualization (needs classifications)
        if real_has_classifications and datasets_with_classifications:
            print("\n[INFO] Creating progression visualization...")
            visualize_all_progressions(real_df, datasets_with_classifications, output_root, turn_threshold=turn_threshold, exact_turns=exact_turns)
        else:
            print(f"[SKIP] Progression visualization requires classifications data")
        
        # Create PTC percentage by turn visualizations using pre-calculated data
        print("\n[INFO] Creating PTC percentage by turn visualizations...")
        visualize_ptc_percentages_by_turn(real_df, all_synthetic_results, output_root,
                                         turn_threshold=turn_threshold, exact_turns=exact_turns,
                                         pre_calculated_percentages=pre_calculated, no_filler=no_filler)
        
        # Create turn count visualizations (needs classifications)
        if real_has_classifications and datasets_with_classifications:
            print("\n[INFO] Creating turn count visualizations for each model...")
            visualize_turn_counts_by_model(real_df, datasets_with_classifications, output_root, turn_threshold=turn_threshold, exact_turns=exact_turns)
        else:
            print(f"[SKIP] Turn count visualization requires classifications data")
    else:
        # Fall back to calculating from detailed data
        print("[INFO] Pre-calculated percentages not found, calculating from detailed data...")
        
        # Check which datasets have classifications for turn-based visualizations
        datasets_with_classifications = {}
        if 'classifications' in real_df.columns:
            real_has_classifications = True
        else:
            real_has_classifications = False
            print(f"[WARNING] Real data missing classifications column, skipping turn-based visualizations")
        
        for label, df in all_synthetic_results.items():
            if 'classifications' in df.columns:
                datasets_with_classifications[label] = df
            else:
                print(f"[WARNING] {label} missing classifications column, skipping from turn-based visualizations")
        
        # Create multi-pair visualization (doesn't need classifications)
        print("\n[INFO] Creating multi-pair visualization...")
        visualize_all_distributions(real_df, all_synthetic_results, output_root, exact_turns=exact_turns, no_filler=no_filler)
        
        # Create progression visualization (needs classifications)
        if real_has_classifications and datasets_with_classifications:
            print("\n[INFO] Creating progression visualization...")
            visualize_all_progressions(real_df, datasets_with_classifications, output_root, turn_threshold=turn_threshold, exact_turns=exact_turns)
        else:
            print(f"[SKIP] Progression visualization requires classifications data")
        
        # Create PTC percentage by turn visualizations (needs classifications)
        if real_has_classifications and datasets_with_classifications:
            print("\n[INFO] Creating PTC percentage by turn visualizations...")
            visualize_ptc_percentages_by_turn(real_df, datasets_with_classifications, output_root, turn_threshold=turn_threshold, exact_turns=exact_turns, no_filler=no_filler)
        else:
            print(f"[SKIP] PTC percentage by turn visualization requires classifications data")
        
        # Create turn count visualizations (needs classifications)
        if real_has_classifications and datasets_with_classifications:
            print("\n[INFO] Creating turn count visualizations for each model...")
            visualize_turn_counts_by_model(real_df, datasets_with_classifications, output_root, turn_threshold=turn_threshold, exact_turns=exact_turns)
        else:
            print(f"[SKIP] Turn count visualization requires classifications data")
    
    # Create comparison summary
    print("\n[INFO] Creating comparison summary...")
    summary_data = []
    
    # Add real data
    real_stats = {
        'dataset': 'Real',
        'display_name': 'Real',
        'P_mean': real_df['P_ratio'].mean(),
        'T_mean': real_df['T_ratio'].mean(),
        'C_mean': real_df['C_ratio'].mean(),
        'F_mean': real_df['F_ratio'].mean(),
        'num_conversations': len(real_df)
    }
    summary_data.append(real_stats)
    
    # Add synthetic data
    for label in sorted(all_synthetic_results.keys(), key=sort_key_by_psi_and_size):
        synth_df = all_synthetic_results[label]
        
        # Create display label
        display_label = label
        for psi in PSI_ABBREV.keys():
            if label.startswith(psi):
                backend_part = label[len(psi):].lstrip('-_')
                backend_part = shorten_backend_name(backend_part)
                display_label = f"{PSI_ABBREV[psi]}-{backend_part}"
                break
        
        synth_stats = {
            'dataset': label,
            'display_name': display_label,
            'P_mean': synth_df['P_ratio'].mean(),
            'T_mean': synth_df['T_ratio'].mean(),
            'C_mean': synth_df['C_ratio'].mean(),
            'F_mean': synth_df['F_ratio'].mean(),
            'num_conversations': len(synth_df)
        }
        summary_data.append(synth_stats)
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = output_root / 'ptc_summary_all_pairs.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"[CSV SAVED] {summary_csv}")
    
    # Print summary table
    print(f"\n{'='*70}")
    print("PTC Distribution Summary (All Pairs)")
    print(f"{'='*70}")
    print(summary_df[['display_name', 'P_mean', 'T_mean', 'C_mean', 'F_mean', 'num_conversations']].to_string(index=False))
    print()
    
    # Calculate Jensen-Shannon distances for all synthetic datasets
    print("\n[INFO] Calculating Jensen-Shannon distances...")
    js_distances = []
    
    # Get real PTCF ratios
    real_ptcf = {
        'P': real_df['P_count'].sum() / real_df['total_patient_turns'].sum(),
        'T': real_df['T_count'].sum() / real_df['total_patient_turns'].sum(),
        'C': real_df['C_count'].sum() / real_df['total_patient_turns'].sum(),
        'F': real_df['F_count'].sum() / real_df['total_patient_turns'].sum()
    }
    
    for label in sorted(all_synthetic_results.keys(), key=sort_key_by_psi_and_size):
        synth_df = all_synthetic_results[label]
        
        # Get synthetic PTCF ratios
        synth_ptcf = {
            'P': synth_df['P_count'].sum() / synth_df['total_patient_turns'].sum(),
            'T': synth_df['T_count'].sum() / synth_df['total_patient_turns'].sum(),
            'C': synth_df['C_count'].sum() / synth_df['total_patient_turns'].sum(),
            'F': synth_df['F_count'].sum() / synth_df['total_patient_turns'].sum()
        }
        
        # Calculate JS distance
        js_dist = calculate_jensenshannon_distance(real_ptcf, synth_ptcf, no_filler=no_filler)
        
        # Get display name
        display_name = label
        for psi in PSI_ABBREV.keys():
            if label.startswith(psi):
                backend = label[len(psi)+1:]  # +1 for the dash
                display_name = f"{PSI_ABBREV[psi]}-{shorten_backend_name(backend)}"
                break
        
        js_distances.append({
            'dataset': label,
            'display_name': display_name,
            'js_distance': js_dist
        })
        
        print(f"  {display_name}: {js_dist:.4f}")
    
    # Save Jensen-Shannon distances to CSV
    if js_distances:
        js_df = pd.DataFrame(js_distances)
        filename_suffix = '_no_filler' if no_filler else ''
        js_csv_path = output_root / f'jensenshannon_distances{filename_suffix}.csv'
        js_df.to_csv(js_csv_path, index=False)
        print(f"\n[CSV SAVED] {js_csv_path}")
    
    print(f"\n[DONE] Visualizations regenerated in: {output_root}")
    print(f"       Based on {len(all_synthetic_results)} synthetic datasets from CSVs")


def main():
    """Main function to run PTC classification analysis."""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='Classify conversations using PTC framework')
    parser.add_argument('--dataset', type=str, default='esc', 
                       help='Dataset type for real conversations (default: esc)')
    parser.add_argument('--synthetic-dir', type=str, help='Directory with synthetic conversations')
    parser.add_argument('--output-dir', type=str, default='output/ptc_analysis',
                       help='Output directory')
    parser.add_argument('--compare', action='store_true',
                       help='Compare real vs synthetic')
    parser.add_argument('--hf', action='store_true',
                       help='Load all psi/backend pairs from HF dataset (hknguyen20/psibench-data)')
    parser.add_argument('--csv-file', type=str, default=None,
                       help='Directory containing CSV files to redraw graphs from (skips classification)')
    parser.add_argument('--no-filler', action='store_true',
                       help='Exclude filler from percentage graphs and Jensen-Shannon calculation, renormalize P, T, C to 100%%')
    parser.add_argument('--N', type=int, default=None,
                       help='Number of conversations to classify (default: all available samples)')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file (default: configs/default.yaml)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Number of parallel tasks to run (default: 1)')
    parser.add_argument('--turn-threshold', type=int, default=16,
                       help='Maximum turn index to display in visualizations (default: 16)')
    parser.add_argument('--exact-turns', type=int, default=None,
                       help='Only analyze conversations with exactly this many patient turns (default: None, analyze all)')
    parser.add_argument('--single-turn', action='store_true',
                       help='Use single-turn classification with limited history instead of full conversation')
    parser.add_argument('--num-messages', type=int, default=4,
                       help='Number of previous messages to include as history for single-turn mode (default: 4)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Clean string arguments
    args.dataset = args.dataset.strip().lower() if args.dataset else args.dataset
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize PTC judge
    judge = PTCClassifier(config, debug=args.debug)
    
    start_time = time.time()
    
    if args.csv_file:
        # Redraw mode: load from CSV and regenerate plots
        redraw_from_csv(
            csv_dir=args.csv_file,
            output_dir=output_dir,
            turn_threshold=args.turn_threshold,
            exact_turns=args.exact_turns,
            no_filler=args.no_filler
        )
    elif args.hf:
        # Multi-dataset HF analysis mode
        compare_all_hf_pairs(
            config=config,
            output_dir=output_dir,
            batch_size=args.batch_size,
            single_turn=args.single_turn,
            num_messages=args.num_messages,
            exact_turns=args.exact_turns,
            turn_threshold=args.turn_threshold,
            debug=args.debug
        )
    elif args.compare and args.synthetic_dir:
        # Compare mode - need to get indices from synthetic data
        synthetic_dir = Path(args.synthetic_dir)
        
        # Extract indices from synthetic session files
        session_files = sorted(synthetic_dir.glob('session_*.json'))
        indices = [int(f.stem.split('_')[1]) for f in session_files]
        
        # Limit number of conversations if specified
        if args.N:
            indices = indices[:args.N]
            session_files = session_files[:args.N]
        
        print(f"Found {len(indices)} conversations to analyze")
        print(f"Batch size: {args.batch_size}")
        print(f"Classification mode: {'Single-turn' if args.single_turn else 'Full conversation'}")
        if args.single_turn:
            print(f"History window: {args.num_messages} messages")
        if args.exact_turns:
            print(f"Filtering to conversations with exactly {args.exact_turns} patient turns")
        print(f"Analyzing corresponding real conversations from {args.dataset} dataset...")
        real_df = analyze_conversations(judge, output_dir / 'real', batch_size=args.batch_size, 
                                        dataset=args.dataset, indices=indices,
                                        single_turn=args.single_turn, num_messages=args.num_messages,
                                        exact_turns=args.exact_turns)
        
        print("\nAnalyzing synthetic conversations...")
        synthetic_df = analyze_conversations(
            judge,
            output_dir / 'synthetic',
            batch_size=args.batch_size,
            data_dir=synthetic_dir,
            session_files=session_files,
            single_turn=args.single_turn,
            num_messages=args.num_messages,
            exact_turns=args.exact_turns,
        )
        
        print("\nComparing distributions...")
        compare_distributions(real_df, synthetic_df, output_dir, judge)
        
        print("\nGenerating visualizations...")
        visualize_distributions(real_df, synthetic_df, output_dir, 
                               turn_threshold=args.turn_threshold, exact_turns=args.exact_turns)
        
    elif args.synthetic_dir:
        # Analyze synthetic only
        print("Analyzing synthetic conversations...")
        print(f"Classification mode: {'Single-turn' if args.single_turn else 'Full conversation'}")
        if args.single_turn:
            print(f"History window: {args.num_messages} messages")
        if args.exact_turns:
            print(f"Filtering to conversations with exactly {args.exact_turns} patient turns")
        synthetic_dir = Path(args.synthetic_dir)
        # If N specified, pick the first N session files
        selected_files = None
        if args.N:
            all_files = sorted(synthetic_dir.glob('session_*.json'))
            selected_files = all_files[:args.N]
        df = analyze_conversations(
            judge,
            output_dir,
            batch_size=args.batch_size,
            data_dir=synthetic_dir,
            session_files=selected_files,
            single_turn=args.single_turn,
            num_messages=args.num_messages,
            exact_turns=args.exact_turns,
        )
    else:
        print("Error: Please provide --real-dir and/or --synthetic-dir")
        return
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nResults saved to {output_dir}")
    print(f"Total time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()