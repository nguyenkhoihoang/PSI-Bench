
# PSI-bench


## 1) Environment setup

```bash
conda create -n psibench python=3.11
conda activate psibench
pip install -e .
```

Create `.env` inside `psibench/` with:
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `HF_TOKEN`

## 2) Repository structure (what goes where)

- `psibench/generate_conversations.py`: generate full synthetic conversations
- `psibench/eval/`: evaluation modules
- `data/`: where generated conversations will be stored
	test_data_quality
- `output/`: evaluation results (csv/json/plots)
	- `output/human_annotation/`
	- `output/depressive_markers/`
	- `output/valence_analysis/`
	- `output/length_comparison/`
	- `output/lexical_diversity/`

Generated synthetic conversations are written under:
- `data/synthetic/{psi}/{dataset}/` (example: `data/synthetic/patientpsi/all/`)

## 3) Generate synthetic data

Common options:
- `--psi`: `patientpsi`(default), `roleplaydoh`

If using offline vLLM models, follow `vllm_serve/README.md` first.

```bash
# single conversation
python -m psibench.generate_conversations --psi patientpsi --N 1 --config configs/default.yaml

# 10 conversations, parallel batch size 5
python -m psibench.generate_conversations --psi roleplaydoh --N 10 --batch-size 5 --config configs/default.yaml
```

## 4) Evaluate conversations

### 4a) All-in-one script

`run_eval.sh` runs the full pipeline end-to-end. It pulls synthetic conversations from the
HuggingFace dataset specified in `eval.hf_dataset` of the config file (no local data
generation required).

```bash
bash run_eval.sh \
  --config configs/default.yaml \
  --output-dir output/eval_run \
  --batch-size 384 \
  --turn-threshold 16
```

Pipeline steps:

| Step | What happens | Notes |
|------|-------------|-------|
| 1 | LLM classifies **valence** and **PTC** labels on HF data | Both run in parallel; LLM model/API set in config |
| 2 | **JS divergence** computed from classification outputs | Sequential; waits for Step 1 |
| 3 | **Depressive markers**, **message lengths**, **lexical diversity** | All three run in parallel |
| 4 | **Aggregate** score combining all five metrics | Writes summary to `<output-dir>/aggregate/` |

Each step writes a `.out` log file inside `--output-dir` for debugging.

### 4b) Run individual metrics

Use per-metric evaluation scripts under `psibench/eval/`.
Each metric file includes runnable command examples in the top comment/docstring.

Metric modules:
- `psibench/eval/ptc/ptc_classification.py`
- `psibench/eval/valence_classification.py`
- `psibench/eval/js_divergence.py`
- `psibench/eval/message_lengths.py`
- `psibench/eval/depressive_linguistic_markers.py`
- `psibench/eval/lexical_diversity.py`
After running all metrics, aggregate multiple evaluation metrics into a single score:
- `psibench/eval/aggregate.py`

## 5) Key outputs to check

- PTC analysis: `output/ptc_analysis/`
- Valence analysis results: `output/valence_analysis/`
- Depressive marker analysis: `output/depressive_markers/`
- Length comparison: `output/length_comparison/`
- Lexical diversity: `output/lexical_diversity/`
- Aggregate score: `output/aggregate/`
