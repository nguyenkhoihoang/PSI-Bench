import os
from typing import Literal, Optional
from datasets import load_dataset
from collections import defaultdict
from psibench.data_loader.utils import merge_consecutive_messages, normalize_backend_name
import pandas as pd
from pathlib import Path
import json

_DEFAULT_REAL_CONV_HF_DATASET = "liusiyang/eeyore_profile"

def load_real_from_hf(dataset_type: Literal["ESC","HOPE", "AnnoMI"], indices: list = None, hf_dataset: str = None):
    """Load specific real conversation datasets from HuggingFace."""
    repo = hf_dataset or _DEFAULT_REAL_CONV_HF_DATASET
    data = load_dataset(repo, split='train', token=os.getenv("HF_TOKEN"))
    df = data.to_pandas()
    df['messages'] = df['messages'].apply(merge_consecutive_messages)
    if indices:
        matched_df = df.loc[indices].copy()
        return matched_df[matched_df['source'] == dataset_type].copy()

    return df[df['source'] == dataset_type]

def load_all_real(indices: list = None, hf_dataset: str = None):
    """Load all real datasets (ESC, HOPE, AnnoMI) from HuggingFace."""
    repo = hf_dataset or _DEFAULT_REAL_CONV_HF_DATASET
    data = load_dataset(repo, split='train', token=os.getenv("HF_TOKEN"))
    df = data.to_pandas()
    df['messages'] = df['messages'].apply(merge_consecutive_messages)

    if indices:
        matched_df = df.loc[indices].copy()
        return matched_df[matched_df['source'].isin(["ESC", "HOPE", "AnnoMI"])].copy()

    return df[df['source'].isin(["ESC", "HOPE", "AnnoMI"])]

def load_real_dataset(dataset_type: str, indices: list = None, hf_dataset: str = None):
    """Load real conversation dataset by type.

    Args:
        dataset_type: One of "esc", "hope", "annomi", or "all"
        indices: Optional list of specific indices to load
        hf_dataset: HuggingFace repo ID; reads from config['data']['real_conv_hf_dataset'] by default

    Returns:
        DataFrame containing the loaded dataset

    Raises:
        ValueError: If dataset_type is not supported
    """
    match dataset_type:
        case "esc":
            return load_real_from_hf(dataset_type="ESC", indices=indices, hf_dataset=hf_dataset)
        case "hope":
            return load_real_from_hf(dataset_type="HOPE", indices=indices, hf_dataset=hf_dataset)
        case "annomi":
            return load_real_from_hf(dataset_type="AnnoMI", indices=indices, hf_dataset=hf_dataset)
        case "all":
            return load_all_real(indices=indices, hf_dataset=hf_dataset)
        case _:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")


def load_synthetic_data_to_df(data_dir: str):
    """Load synthetic session data from directory 
        Return into a DataFrame."""
    data_dir = Path(data_dir)
    sessions = []
    
    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return pd.DataFrame()
    
    # Load all session JSON files
    for session_file in sorted(data_dir.glob('session_*.json')):
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
                sessions.append(session_data)
        except Exception as e:
            print(f"[WARNING] Failed to load {session_file}: {e}")
            continue
    
    if not sessions:
        print(f"[WARNING] No session files found in {data_dir}")
        return pd.DataFrame()
    
    return pd.DataFrame(sessions)


def _safe_json_load(value):
    """Parse JSON strings back to Python objects when possible."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def load_synthetic_hf_to_df(psi: str, backend_llm: str, split: str = "train", session_id: int = None, token: Optional[str] = None, dataset_name: str = "hknguyen20/psibench-conv", synthetic_dir: Optional[str] = None):
    """Load PSI-bench synthetic data from HuggingFace and filter by simulator and backend.

    Args:
        psi: Name of the PSI simulator (e.g., "patientpsi" or "roleplaydoh").
        backend_llm: Backend LLM identifier (e.g., "hosted_vllm_openai_gpt-oss-120b").
        split: HF split to load (default: "train").
        session_id: Optional session ID to filter by.
        token: Optional HF token; falls back to HF_TOKEN env var.
        dataset_name: HuggingFace dataset name (default: "hknguyen20/psibench-conv").
        synthetic_dir: If set, read local session_*.json under this root instead of HF.

    Returns:
        Pandas DataFrame filtered to the requested psi and backend_llm.
    """

    if synthetic_dir:
        df = load_synthetic_local_to_df(synthetic_dir, psi, backend_llm)
        return df if session_id is None or df.empty else df[df["session_id"] == session_id].copy()

    hf_token = token or os.getenv("HF_TOKEN")
    dataset = load_dataset(dataset_name, split=split, token=hf_token)
    df = dataset.to_pandas()

    # Filter by session_id first if provided
    if session_id is not None:
        df = df[df["session_id"] == session_id].copy()

    # Filter by psi and backend (case-insensitive)
    filtered = df[
        df["psi"].str.lower() == psi.lower()
    ]
    filtered = filtered[
        filtered["backend_llm"].str.lower() == backend_llm.lower()
    ].copy()

    if filtered.empty:
        print(f"[WARNING] No rows found for psi='{psi}' and backend_llm='{backend_llm}'")
        return filtered

    # Convert JSON-like strings back to Python objects for convenience
    for col in ("messages", "profile", "ccd"):
        if col in filtered.columns:
            filtered[col] = filtered[col].apply(_safe_json_load)

    return filtered




# Folder names that sit below <psi>/<model>/ in the generation output layout
# (generate_conversations.py writes <output_dir>/<psi>/<model>/<dataset>/session_*.json).
LOCAL_DATASET_DIRS = {"all", "esc", "hope", "annomi"}


def discover_local_pairs(root: str):
    """Recurse through root and map (psi, backend_llm) -> [dirs containing session_*.json].

    Mirrors the HF dataset's psi/backend_llm columns using the folder layout
    <psi>/<model>/<dataset>/session_*.json. The psi and model are read from the path of each
    directory that holds sessions, so root may be the whole data tree (data/synthetic), one psi,
    or a single <psi>/<model>/all folder. backend_llm is normalized like the HF values.
    If a pair has an 'all' directory only that one is used (the others would overlap it).
    """
    root = Path(root).resolve()
    dirs = sorted({f.parent for f in root.rglob("session_*.json")})
    pairs = defaultdict(list)
    for d in dirs:
        model_dir = d.parent if d.name in LOCAL_DATASET_DIRS else d
        pairs[(model_dir.parent.name, normalize_backend_name(model_dir.name))].append(d)
    for key, ds in pairs.items():
        all_dirs = [d for d in ds if d.name == "all"]
        if all_dirs:
            pairs[key] = all_dirs
    return dict(pairs)


def get_local_psi_backend_pairs(root: str):
    return sorted(discover_local_pairs(root))


def load_synthetic_local_to_df(root: str, psi: str, backend_llm: str):
    """Local counterpart of load_synthetic_hf_to_df, reading session_*.json under root."""
    dirs = discover_local_pairs(root).get((psi, normalize_backend_name(backend_llm)), [])
    rows = []
    for d in dirs:
        for f in sorted(d.glob("session_*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    row = json.load(fh)
            except Exception as e:
                print(f"[WARNING] Failed to load {f}: {e}")
                continue
            sid = f.stem.split("_", 1)[1]
            row["session_id"] = int(sid) if sid.isdigit() else sid
            row["psi"] = psi
            row["backend_llm"] = backend_llm
            rows.append(row)
    if not rows:
        print(f"[WARNING] No local sessions for psi='{psi}' and backend_llm='{backend_llm}' under {root}")
    return pd.DataFrame(rows)
