import os
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any
from datasets import load_dataset
import re

PSI_ABBREV = {
        'patientpsi': 'PS',
        'roleplaydoh': 'RD'
    }

# Labels for plotting PSI types in legends (including LaTeX formatting)
PSI_LABELS = {
    'Real': 'Human',
    'patientpsi': r'PATIENT-$\Psi$', 
    'roleplaydoh': 'Roleplay-doh'
}

# Marker styles for plotting different backend models
# BACKEND_MARKER_STYLES = ['o', '^', 's', 'D', 'P', 'x', '1', 'p', '*', 'h']
BACKEND_MARKER_STYLES = ['o']

# Family-aware marker pools. Models in the same family get visually related markers.
FAMILY_MARKER_STYLES = {
    'GPT': ['v', '^', '<', '>'],
    'Llama': ['+', 'P'],
    'Qwen': ['d', 'D'],
    'Unknown': BACKEND_MARKER_STYLES,
}

# Colors for different PSI types
PSI_COLORS = {
    'patientpsi': '#1f77b4',
    'roleplaydoh': '#ff7f0e',
}

def get_assistant_messages(messages: List[Dict], preprocess_text: bool = False) -> List[str]:
    """Extract content of messages from the 'assistant' role.
    
    Args:
        messages: List of message dictionaries
        preprocess_text: If True, apply preprocessing (lowercase, remove punctuation)
    
    Returns:
        List of message content strings
    """
    if preprocess_text:
        return [preprocess(msg.get('content', '')) for msg in messages 
                if msg.get('role') == 'assistant' and msg.get('content', '').strip()]
    else:
        return [msg.get('content', '') for msg in messages 
                if msg.get('role') == 'assistant' and msg.get('content', '').strip()]


def aggregate_messages(messages: List[str]) -> str:
    """Concatenate all messages into a single string."""
    return " ".join(messages)

def preprocess(text: str) -> str:
    """
    Preprocess text: lowercase, remove punctuation.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize whitespace
    words = text.split()

    return " ".join(words)

def extract_patient_messages_by_turn(conversations: List[Dict], max_turns: int = None) -> Dict[int, List[str]]:
    """Extract patient messages organized by turn index.
    
    Args:
        conversations: List of conversation dictionaries
        max_turns: Maximum turn to analyze (None = analyze all)
        
    Returns:
        Dictionary mapping turn_index -> list of patient messages at that turn
    """
    messages_by_turn = defaultdict(list)
    
    for conv in conversations:
        messages = conv.get('messages', [])
        patient_turn_idx = 0
        
        for msg in messages:
            # Patient messages have role 'assistant'
            if msg.get('role') == 'assistant':
                content = msg.get('content', '').strip()
                if content:  # Only count non-empty messages
                    if max_turns is None or patient_turn_idx < max_turns:
                        messages_by_turn[patient_turn_idx].append(content)
                    patient_turn_idx += 1
    
    return messages_by_turn

def get_all_psi_backend_pairs(token: Optional[str] = None, dataset_name: str = "hknguyen20/psibench-data") -> List[Tuple[str, str]]:
    """Get all unique PSI-backend LLM pairs from HuggingFace dataset.
    
    Args:
        token: Optional HuggingFace token
        dataset_name: HuggingFace dataset name (default: "hknguyen20/psibench-data")
        
    Returns:
        List of (psi, backend_llm) tuples
    """
    hf_token = token or os.getenv("HF_TOKEN")
    dataset = load_dataset(dataset_name, split="train", token=hf_token)
    df = dataset.to_pandas()
    print(f"Loaded dataset {dataset_name}")
    
    # Get unique pairs
    unique_pairs = df[['psi', 'backend_llm']].drop_duplicates()
    pairs = [(row['psi'], row['backend_llm']) for _, row in unique_pairs.iterrows()]
    
    return sorted(pairs)


def safe_dir_name(name: str) -> str:
    """Convert a string to a safe directory/label name."""
    return name.replace('/', '_').replace(':', '_').replace(' ', '_')


def format_session_id(session_id: Any) -> str:
    """Format session id as zero-padded 4-digit string when possible."""
    session_str = str(session_id)
    if session_str.startswith("session_"):
        session_str = session_str.split("session_", 1)[1]
    session_str = session_str.split(".")[0]

    if session_str.isdigit():
        return f"{int(session_str):04d}"

    digits = "".join(ch for ch in session_str if ch.isdigit())
    if digits:
        return f"{int(digits):04d}"

    return session_str


def build_conversation_id(session_id: Any, psi: str = None, backend_llm: str = None, is_real: bool = False) -> str:
    """Build standardized conversation id.

    Format:
      - Real: conv_profile_{session_id}_real
      - Synthetic: conv_profile_{session_id}_{patient_sim}-{backend_LLM}
    """
    formatted_session_id = format_session_id(session_id)

    if is_real:
        return f"conv_profile_{formatted_session_id}_real"

    if psi and backend_llm:
        psi_tag = PSI_ABBREV.get(psi, psi)
        backend_tag = safe_dir_name(backend_llm)
        return f"conv_profile_{formatted_session_id}_{psi_tag}-{backend_tag}"

    return f"conv_profile_{formatted_session_id}_synthetic"


# Backend name shortening for graph labels
BACKEND_LABEL_MAPPING = {
    'Qwen3-30B-A3B-Instruct-2507': 'qwen3-30b-a3b-i',
    'Llama-3.3-70B-Instruct': 'llama3.3-70b-i',
    'Llama-3.1-8B-Instruct': 'llama3.1-8b-i',
    'Qwen2.5-72B-Instruct': 'qwen2.5-72b-i',
}
def create_display_label(dataset):
    """Convert 'patientpsi-backend' to 'PS-shortened_backend'"""
    for psi in PSI_ABBREV.keys():
        if dataset.startswith(psi):
            # Extract backend part after PSI prefix
            backend_part = dataset[len(psi):].lstrip('-_')
            # Shorten the backend name
            backend_part = shorten_backend_name(backend_part)
            return f"{PSI_ABBREV[psi]}-{backend_part}"
    return dataset

def shorten_backend_name(backend_name: str) -> str:
    """Shorten backend name for compact graph labels.
    
    Args:
        backend_name: Full backend name
        
    Returns:
        Shortened backend name if mapping exists, otherwise original name
    """
    return BACKEND_LABEL_MAPPING.get(backend_name, backend_name)


def extract_model_size(model_name: str) -> int:
    """Extract parameter size (in billions) from model name.
    
    Args:
        model_name: Model name containing size (e.g., "Llama-3.3-70B-Instruct", "gpt-oss-120b")
        
    Returns:
        Parameter size in billions as integer. Returns 0 if not found.
        
    Examples:
        >>> extract_model_size("Llama-3.3-70B-Instruct")
        70
        >>> extract_model_size("Qwen3-30B-A3B-Instruct-2507")
        30
        >>> extract_model_size("gpt-oss-120b")
        120
    """
    # Look for patterns like "-70B", "-30b", "-120B", etc. (requires dash before number)
    match = re.search(r'-(\d+)[bB]', model_name)
    if match:
        return int(match.group(1))
    return 0


def sort_key_by_psi_and_size(dataset_name: str) -> Tuple[str, int]:
    """Sort key function for dataset names: sorts by PSI type, then model size.
    
    Args:
        dataset_name: Dataset name like "patientpsi-Llama-3.3-70B-Instruct" or "roleplaydoh-gpt4o"
        
    Returns:
        Tuple of (psi_name, model_size) for sorting
        
    Examples:
        >>> sort_key_by_psi_and_size("patientpsi-Llama-3.3-70B-Instruct")
        ('patientpsi', 70)
        >>> sort_key_by_psi_and_size("roleplaydoh-gpt4o")
        ('roleplaydoh', 0)
    """
    for psi in ['patientpsi', 'roleplaydoh']:
        if dataset_name.startswith(psi):
            backend_part = dataset_name[len(psi):].lstrip('-_')
            return (psi, extract_model_size(backend_part))
    return (dataset_name, 0)


def sort_summary_key(display_name: str) -> Tuple[str, int]:
    """Sort key function for display names with PS-/RD- prefixes.
    
    Args:
        display_name: Display name like "PS-llama3.3-70b-i" or "RD-gpt4o" or "Real"
        
    Returns:
        Tuple of (psi_name, model_size) for sorting. Real comes first with ('', 0)
        
    Examples:
        >>> sort_summary_key("Real")
        ('', 0)
        >>> sort_summary_key("PS-llama3.3-70b-i")
        ('patientpsi', 70)
        >>> sort_summary_key("RD-gpt4o")
        ('roleplaydoh', 0)
    """
    if display_name == 'Real':
        return ('', 0)
    if display_name.startswith('PS-'):
        backend_part = display_name[3:]  # Remove 'PS-'
        return ('patientpsi', extract_model_size(backend_part))
    elif display_name.startswith('RD-'):
        backend_part = display_name[3:]  # Remove 'RD-'
        return ('roleplaydoh', extract_model_size(backend_part))
    return (display_name, 0)


def extract_base_model(model_name: str) -> str:
    """Extract base model family name from a model name.
    
    Args:
        model_name: Model name (e.g., "Llama-3.3-70B-Instruct", "Qwen3-30B-A3B-Instruct-2507", "gpt-oss-120b")
        
    Returns:
        Base model family name (e.g., "Llama", "Qwen", "GPT"). Returns "Unknown" if not matched.
        
    Examples:
        >>> extract_base_model("Llama-3.3-70B-Instruct")
        'Llama'
        >>> extract_base_model("Qwen3-30B-A3B-Instruct-2507")
        'Qwen'
        >>> extract_base_model("gpt-oss-120b")
        'GPT'
        >>> extract_base_model("Qwen2.5-72B-Instruct")
        'Qwen'
    """
    model_name_lower = model_name.lower()
    
    # Check for common model families (case-insensitive)
    if 'llama' in model_name_lower:
        return 'Llama'
    elif 'qwen' in model_name_lower:
        return 'Qwen'
    elif 'gpt' in model_name_lower:
        return 'GPT'
    elif 'mistral' in model_name_lower:
        return 'Mistral'
    elif 'gemma' in model_name_lower:
        return 'Gemma'
    elif 'falcon' in model_name_lower:
        return 'Falcon'
    elif 'phi' in model_name_lower:
        return 'Phi'
    else:
        return 'Unknown'


def sort_key_by_backend_family_and_size(backend_name: str) -> Tuple[int, int, str]:
    """Sort key for backend names: family priority, then size, then name."""
    base_model = extract_base_model(backend_name)
    return (
        get_model_family_priority(base_model),
        extract_model_size(backend_name),
        backend_name.lower(),
    )


def extract_backend_name_from_label(label: str) -> str:
    """Extract backend name from dataset labels used across eval modules.

    Supports:
    - patientpsi-<backend>, roleplaydoh-<backend>
    - patientpsi_<backend>, roleplaydoh_<backend>
    - generic <prefix>-<backend> / <prefix>_<backend>
    """
    if not label:
        return 'unknown'

    for psi in PSI_ABBREV.keys():
        if label.startswith(psi):
            backend = label[len(psi):].lstrip('-_')
            return backend if backend else 'unknown'

    if '_' in label:
        prefix, backend = label.split('_', 1)
        if backend:
            return backend

    if '-' in label:
        prefix, backend = label.split('-', 1)
        if backend:
            return backend

    return label


def assign_backend_markers(backends: List[str]) -> Dict[str, str]:
    """Assign markers so same-family backends look similar but remain distinct.

    Within each family, markers are assigned in ascending model-size order.
    """
    unique_backends = sorted(set(backends), key=sort_key_by_backend_family_and_size)
    grouped: Dict[str, List[str]] = defaultdict(list)
    for backend in unique_backends:
        family = extract_base_model(backend)
        grouped[family].append(backend)

    marker_map: Dict[str, str] = {}
    for family, family_backends in grouped.items():
        marker_pool = FAMILY_MARKER_STYLES.get(family, BACKEND_MARKER_STYLES)
        ordered_family_backends = sorted(family_backends, key=sort_key_by_backend_family_and_size)
        for i, backend in enumerate(ordered_family_backends):
            marker_map[backend] = marker_pool[i % len(marker_pool)]

    return marker_map


def get_model_family_priority(base_model: str) -> int:
    """Get sort priority for model family.
    
    Args:
        base_model: Base model family name (e.g., "Llama", "Qwen", "GPT")
        
    Returns:
        Integer priority (lower = earlier in sort order)
    """
    priority_order = {
        'Llama': 0,
        'Qwen': 1,
        'GPT': 2,
        'Mistral': 3,
        'Gemma': 4,
        'Falcon': 5,
        'Phi': 6,
        'Unknown': 999
    }
    return priority_order.get(base_model, 999)


def sort_key_by_base_model(dataset_name: str) -> Tuple[int, int]:
    """Sort key function for dataset names: sorts by base model family, then model size.
    
    Groups results by base model (Llama, Qwen, GPT, etc.) in priority order,
    and within each group, sorts by model size (smaller to larger).
    
    Args:
        dataset_name: Dataset name like "patientpsi-Llama-3.3-70B-Instruct" or "roleplaydoh-gpt-oss-120b"
        
    Returns:
        Tuple of (model_family_priority, model_size) for sorting
        
    Examples:
        >>> sort_key_by_base_model("patientpsi-Llama-3.3-70B-Instruct")
        (0, 70)
        >>> sort_key_by_base_model("roleplaydoh-Qwen3-30B-A3B-Instruct-2507")
        (1, 30)
        >>> sort_key_by_base_model("patientpsi-gpt-oss-120b")
        (2, 120)
    """
    # Extract backend part (remove PSI prefix if present)
    backend_part = dataset_name
    for psi in ['patientpsi', 'roleplaydoh']:
        if dataset_name.startswith(psi):
            backend_part = dataset_name[len(psi):].lstrip('-_')
            break
    
    base_model = extract_base_model(backend_part)
    model_priority = get_model_family_priority(base_model)
    model_size = extract_model_size(backend_part)
    
    return (model_priority, model_size)


def sort_key_by_psi_base_model(dataset_name: str) -> Tuple[str, int, int]:
    """Sort key function: sorts by PSI type, then base model family, then model size.
    
    This function groups results first by PSI simulator type, then by base model within each PSI
    (in priority order: Llama, Qwen, GPT, ...), and finally by model size within each base model group.
    
    Args:
        dataset_name: Dataset name like "patientpsi-Llama-3.3-70B-Instruct" or "roleplaydoh-gpt-oss-120b"
        
    Returns:
        Tuple of (psi_name, model_family_priority, model_size) for sorting
        
    Examples:
        >>> sort_key_by_psi_base_model("patientpsi-Llama-3.3-70B-Instruct")
        ('patientpsi', 0, 70)
        >>> sort_key_by_psi_base_model("roleplaydoh-Qwen3-30B-A3B-Instruct-2507")
        ('roleplaydoh', 1, 30)
    """
    # Extract PSI type
    psi_type = 'unknown'
    backend_part = dataset_name
    
    for psi in ['patientpsi', 'roleplaydoh']:
        if dataset_name.startswith(psi):
            psi_type = psi
            backend_part = dataset_name[len(psi):].lstrip('-_')
            break
    
    base_model = extract_base_model(backend_part)
    model_priority = get_model_family_priority(base_model)
    model_size = extract_model_size(backend_part)
    
    return (psi_type, model_priority, model_size)


def get_model_opacity(model_name: str, all_model_sizes: List[int]) -> float:
    """Calculate opacity for a model based on its size relative to other models.
    
    Smaller models get lower opacity (more transparent), larger models get higher opacity.
    Opacity ranges from 0.35 (smallest) to 1.0 (largest).
    
    Args:
        model_name: Model name (e.g., "Llama-3.3-70B-Instruct")
        all_model_sizes: List of all model sizes in the dataset for normalization
        
    Returns:
        Opacity value between 0.35 and 1.0
        
    Examples:
        >>> get_model_opacity("Llama-3.1-8B-Instruct", [8, 30, 70, 120])
        0.35
        >>> get_model_opacity("gpt-oss-120b", [8, 30, 70, 120])
        1.0
    """
    model_size = extract_model_size(model_name)
    
    if not all_model_sizes or len(all_model_sizes) == 0:
        return 0.85  # Default opacity if no sizes available
    
    # If model size is 0 or not in list, use default
    if model_size == 0:
        return 0.65
    
    # Get min and max sizes for normalization
    min_size = min(all_model_sizes)
    max_size = max(all_model_sizes)
    
    # If all models are the same size, return default
    if min_size == max_size:
        return 0.85
    
    # Normalize to 0-1 range, then map to 0.35-1.0 opacity range
    normalized = (model_size - min_size) / (max_size - min_size)
    opacity = 0.35 + (normalized * 0.65)  # Maps [0,1] to [0.35, 1.0]
    
    return opacity
