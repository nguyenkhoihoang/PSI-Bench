# Valence Classification

Classifies each patient turn in therapy conversations as **positive**, **negative**, or **neutral** emotional valence using an LLM judge. Mirrors `emotion_classification.py` in structure and output format.

## Categories

| Label | When to use |
|-------|-------------|
| `positive` | Hope, relief, gratitude, optimism, acceptance, positive affect |
| `negative` | Sadness, fear, anger, shame, hopelessness, distress, negative affect |
| `neutral` | No clear affect: factual statements, procedural responses |

---

## 1. Classify conversations

Classifies every patient turn via `ValenceClassifier` and saves per-turn detail (JSON) and summary statistics (CSV).

```bash
# Classify all HF PSI-backend pairs against real data
python -m psibench.eval.valence_classification \
  --hf \
  --batch-size 32 \
  --turn-threshold 16 \
  --config configs/default.yaml \
  --output-dir output/valence_analysis


**Key arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--hf` | — | Load all PSI/backend pairs from HuggingFace |
| `--batch-size N` | 1 | Parallel LLM calls per batch (384 recommended for speed) |
| `--turn-threshold N` | 12 | Max turn index shown in line plots |
| `--exact-turns N` | — | Only include convs with exactly N patient turns |
| `--config PATH` | `configs/default.yaml` | Config with `eval.valence_classifier` block |
| `--output-dir DIR` | `output/valence_analysis` | Where to write results |
| `--csv-file DIR` | — | Redraw plots from existing outputs (skips LLM calls) |

**Output files** (in `output/valence_analysis/`):

```
real_valence_detailed.json                      # per-turn labels for all real convs
real_valence_summary.csv                        # per-conv counts & percentages
<psi>_<backend>_valence_detailed.json           # per-turn labels per synthetic pair
<psi>_<backend>_valence_summary.csv             # per-conv summary per pair
valence_percentages_by_turn_t16.csv             # % per valence per turn (all datasets)
valence_percentages_by_turn_t16_no_neutral.csv  # same, neutral excluded & rescaled
valence_percentages_by_turn.png/.pdf            # 1×3 line plots (positive/negative/neutral)
valence_distribution_all_pairs.png/.pdf         # stacked bar chart across all datasets
```

**Redraw plots without re-running the LLM:**

```bash
python -m psibench.eval.valence_classification \
  --csv-file output/valence_analysis \
  --turn-threshold 16
```

---

## 2. Jensen-Shannon divergence across turns

Reuses `js_divergence.py` directly (no changes needed). Pass the no-neutral CSV for a distribution over only positive/negative, or the full CSV to include neutral.

```bash
# JS divergence (excluding neutral — recommended for cleaner signal)
python psibench/eval/js_divergence.py \
  --csv-file output/valence_analysis/valence_percentages_by_turn_t16_no_neutral.csv \
  --turn-threshold 16 \
  --output-dir output/valence_analysis \
  --label-column valence \
  --label-type valence

# JS divergence (all three categories including neutral)
python psibench/eval/js_divergence.py \
  --csv-file output/valence_analysis/valence_percentages_by_turn_t16.csv \
  --turn-threshold 16 \
  --output-dir output/valence_analysis \
  --label-column valence \
  --label-type valence
```

**Output files:**

```
valence_js_divergence_average.csv   # avg JS divergence per synthetic dataset + final_score
valence_js_divergence_per_turn.csv  # per-turn JS divergence values
```

`final_score = (1 - avg_js_divergence) × 100` — higher is better (closer to real distribution).

---

## 3. LLM agreement with annotators

Uses the human annotation pipeline in `psibench/eval/human_annotation/`, which pulls conversations and human labels from Supabase. Valence is compared against human Plutchik emotion labels mapped to valence groups (positive/negative/neutral) using the same `_normalize_emotion_group` logic as the existing emotion pipeline.

**Step A — classify (generates LLM JSON with `emotion`, `progress`, and `valence` fields):**

```bash
python -m psibench.eval.human_annotation.classification_match \
  --config configs/default.yaml \
  --mode classify \
  --output-file output/human_annotation/classification_match_llm.json
```

**Step B — analyze valence agreement (strict majority, ties skipped):**

```bash
python -m psibench.eval.human_annotation.classification_match_valence \
  --llm-input-file output/human_annotation/classification_match_llm.json \
  --offline
```

**Or run both steps together and get all metrics (emotion + progress + valence):**

```bash
python -m psibench.eval.human_annotation.classification_match \
  --config configs/default.yaml \
  --mode all \
  --output-file output/human_annotation/classification_match_llm.json
```

**Metrics reported** (all at strict-majority, ties skipped):

- `strict_match_percentage` — % where LLM valence matches human majority valence group
- `individual_human_vs_llm_match_percentage_median` — median per-annotator match %
- `cohens_kappa_majority_human_vs_llm` — Cohen's κ (majority vote vs LLM)
- `cohens_kappa_individual_human_vs_llm_median` — median per-annotator Cohen's κ
- `fleiss_kappa_human_human` — inter-annotator Fleiss' κ on valence groups

**Human label mapping:**
```
positive: joy, trust, anticipation, surprise
negative: sadness, disgust, anger, fear
neutral:  neutral
```

---

## Configuration

The `valence_classifier` block in `configs/default.yaml` controls which LLM is used:

```yaml
eval:
  valence_classifier:
    temperature: 0.2
    model: "hosted_vllm/openai/gpt-oss-120b"
    api_base: "http://convai-srv-03.cs.illinois.edu:9003/v1"
```

The same judge model serves all three tasks. Make sure `vllm_serve/launch_judge.sh` is running before starting any classification step.

---

## Full pipeline (example)

```bash
# Step 1 — classify all HF PSI-backend pairs
python -m psibench.eval.valence_classification \
  --hf --batch-size 384 --turn-threshold 16 \
  --config configs/default.yaml --output-dir output/valence_analysis

# Step 2 — JS divergence across turns
python psibench/eval/js_divergence.py \
  --csv-file output/valence_analysis/valence_percentages_by_turn_t16_no_neutral.csv \
  --turn-threshold 16 --output-dir output/valence_analysis \
  --label-column valence --label-type valence

# Step 3a — classify human-annotation conversations (adds valence alongside emotion + progress)
python -m psibench.eval.human_annotation.classification_match \
  --config configs/default.yaml --mode classify \
  --output-file output/human_annotation/classification_match_llm.json

# Step 3b — compute valence vs annotator agreement (strict majority)
python -m psibench.eval.human_annotation.classification_match_valence \
  --llm-input-file output/human_annotation/classification_match_llm.json \
  --offline
```
