# AfroXLMR / XLM-R Mechanistic Audit

This project implements a **coarse, publishable, laptop-feasible mechanistic audit** of three multilingual transformer backbones:

- `xlm-roberta-base`
- `Davlan/afro-xlmr-large`
- `dsfsi/afro-xlmr-comet`

The intended research use is to study **false localisation**: whether locally relevant behaviour in an African language model reflects deep representational change or shallow, brittle adjustment.

## Design choices

This code intentionally does **not** attempt full sparse circuit discovery. Instead, it implements methods that are:
- feasible on a 16 GB Apple Silicon laptop,
- defensible in a paper,
- robust to cross-model differences in depth and checkpoint packaging.

Implemented methods:
1. **Shared classifier wrapper** over all backbones.
2. **Relative-depth layer alignment** for models with different numbers of layers.
3. **Layerwise pooled representation extraction**.
4. **Representation similarity** using mean-vector cosine similarity and **linear CKA**.
5. **Layer ablation** as a coarse causal intervention.
6. **Within-model paired activation patching** on local/base counterfactual pairs.
7. **Optional dynamic quantisation** for a compression realism check.
8. **False-localisation index** combining late-layer dependence and weak early/mid-layer divergence.

## Dataset schema

Expected CSV columns:

- `id`
- `text`
- `label`
- `language`
- `is_local_task`
- `pair_id`
- `pair_role`
- `split`

### Column meanings

- `id`: unique example id
- `text`: input text
- `label`: integer class label
- `language`: language tag
- `is_local_task`: 1 for local evaluation items, else 0
- `pair_id`: shared id for counterfactual pairs; blank if not paired
- `pair_role`: `base`, `local`, or blank
- `split`: `train`, `validation`, or `test`

For patching, each non-empty `pair_id` should usually have:
- one `base` row
- one `local` row

See `data/example_schema.csv`.

## Installation

```bash
python -m pip install -U pip
python -m pip install torch transformers pandas numpy scikit-learn pyarrow tqdm pytest
```

## Usage

```bash
python run_afroxlmr_comparison.py \
  --csv data/example_schema.csv \
  --output-dir outputs/run1 \
  --max-length 128 \
  --epochs 1 \
  --batch-size 8
```

## Outputs

The pipeline saves intermediate artefacts throughout:
- validated dataset copy
- predictions and metrics for each model
- per-depth pooled representations
- cosine similarity tables
- linear CKA tables
- layer ablation tables
- patching tables
- quantised Comet metrics
- false-localisation summary JSON

This redundancy is intentional.

## Why these methods?

### Shared classifier wrapper
The comparison should be about **backbone representations**, not arbitrary shipped classification heads.

### Relative-depth alignment
The three models do not have equal depth. Relative alignment is the most defensible coarse comparison.

### Linear CKA
CKA is a standard representation-similarity measure that is more robust than plain cosine similarity for hidden-state comparison.

### Layer ablation
Head-level or neuron-level interventions are possible, but much heavier. Layer-level ablation is stable and feasible on laptop hardware.

### Patching
This is a coarse layer-level implementation inspired by activation patching / causal tracing. It supports claims about **where local behaviour becomes causally effective**.

## Layout

```text
afroxlmr_mechint_recreated/
  README.md
  run_afroxlmr_comparison.py
  data/
    example_schema.csv
  src/
    afroxlmr_mechint/
      __init__.py
      pipeline.py
  tests/
    test_pipeline.py
```
