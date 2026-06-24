# AfroXLMR / XLM-R Localisation Analysis

This repository implements a laptop-feasible analysis of how localisation is represented inside African-adapted multilingual transformer encoder models.

The project compares upstream XLM-R models with African-adapted models on an institutional-validity task. The aim is to test whether local South African institutional distinctions are merely available at the classifier output, or whether they appear in the internal representations of the models.

The analysis focuses on two localisation pathways:

- `xlm-roberta-large` -> `Davlan/afro-xlmr-large` -> `local_models/afro-xlmr-comet`
- `xlm-roberta-base` -> `Davlan/afro-xlmr-small`

The main scripts also support additional models where available, but the primary interpretation should be based on these documented lineages.

## Research task

The task is binary classification of institutional validity.

Each example is a sentence containing South African institutional, document or public-service references. Valid examples preserve the original institutional relation. Invalid examples are created by swapping institutions or documents so that the sentence remains grammatical but becomes contextually implausible.

Labels are:

- `1`: valid institutional sentence
- `0`: invalid swapped institutional sentence

The classifier is not intended as a deployed classifier. It is used as a readout over frozen model representations so that the internal representations of different models can be compared.

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

Column meanings:

- `id`: unique example identifier
- `text`: input sentence
- `label`: integer class label
- `language`: language tag
- `is_local_task`: retained for compatibility with earlier analyses
- `pair_id`: shared identifier for paired valid and invalid examples
- `pair_role`: role in the pair, usually `base` for valid and `local` for invalid in older files
- `split`: `train`, `validation` or `test`

The current main dataset is:

```text
 data/classifier_dataset_llm_pipeline.csv
```

The optional metadata file is:

```text
 data/classifier_dataset.csv
```

The same split column is used for all models. Pair-level split leakage should be checked before running the main analyses.

## Models

Primary models:

- `xlm_r_large`: `xlm-roberta-large`
- `afroxlmr_large`: `Davlan/afro-xlmr-large`
- `afroxlmr_comet`: `local_models/afro-xlmr-comet`
- `xlm_r_base`: `xlm-roberta-base`
- `afroxlmr_small`: `Davlan/afro-xlmr-small`

Secondary or optional models may include:

- `afroxlmr_base`: `Davlan/afro-xlmr-base`
- `zabantu_xlmr`: `dsfsi/zabantu-xlm-roberta`

AfroXLMR-Comet may require a local Hugging Face model directory with a reconstructed `config.json` before loading.

## Installation

```bash
python -m pip install -U pip
python -m pip install torch transformers pandas numpy scikit-learn pyarrow tqdm matplotlib pytest
```

If using `uv`, install dependencies through the project environment and run scripts with `uv run`.

## Main pipeline

Run the main representation-analysis pipeline:

```bash
uv run python pipeline_with_layerwise_probes_merged.py \
  --csv data/classifier_dataset_llm_pipeline.csv \
  --metadata-csv data/classifier_dataset.csv \
  --output-dir outputs/institutional_swap_run_llm_probes_crosspatch \
  --max-length 128 \
  --epochs 10 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --seed 13
```

Generate figures:

```bash
uv run python plot_research_progress_merged.py \
  --output-dir outputs/institutional_swap_run_llm_probes_crosspatch
```

## Classifier training

For each model, the transformer encoder is frozen. Only a downstream classifier head is trained.

The classifier uses:

- mean-pooled hidden states;
- dropout;
- a two-layer multilayer perceptron classifier.

Default hyperparameters:

- epochs: `10`
- batch size: `8`
- learning rate: `1e-4`
- optimiser: `AdamW`
- weight decay: `0.01`
- maximum sequence length: `128`
- random seed: `13`

Freezing the encoder ensures that differences between models reflect pretrained and localised representations rather than task-specific encoder fine-tuning.

## Main analyses

### Classifier performance

The pipeline reports test accuracy, macro F1, confusion matrices and per-example predictions for each model.

Outputs:

```text
<output_dir>/<model_name>/test_metrics.json
<output_dir>/<model_name>/test_predictions.csv
```

### Debiased linear CKA

Representational similarity is measured using debiased linear centred kernel alignment.

Representations are extracted at aligned relative depths:

- 0%
- 25%
- 50%
- 75%
- 100%

Outputs:

```text
<output_dir>/cross_model_linear_cka.csv
```

The plotting script displays within-family comparisons only:

- XLM-R Large, AfroXLMR Large and AfroXLMR Comet
- XLM-R Base and AfroXLMR Small

### Noise ablation

Layerwise noise ablation replaces one encoder layer output at a time with variance-scaled Gaussian noise during inference. The classifier is then evaluated again.

This estimates whether task performance depends on structured information from each layer.

Outputs:

```text
<output_dir>/<model_name>/layer_noise_ablation.csv
```

### Within-model activation patching

Within-model paired activation patching uses valid/invalid sentence pairs.

For each pair, the invalid sentence representation at a selected depth is patched into the valid sentence forward pass within the same model. The prediction-change rate is recorded.

Outputs:

```text
<output_dir>/<model_name>/within_model_activation_patching.csv
```

### Cross-model activation patching

Cross-model patching tests whether source-model representations can substitute for target-model representations.

Implemented direct patching pairs:

- `xlm_r_large` -> `afroxlmr_large`
- `xlm_r_base` -> `afroxlmr_small`

Only hidden-size-compatible pairs are patched directly. AfroXLMR-Comet is excluded from direct cross-model patching because its hidden size differs from AfroXLMR-Large.

Outputs:

```text
<output_dir>/cross_model_activation_patching.csv
```

### Layerwise compatibility probes

Layerwise compatibility probes test whether the valid/invalid distinction is linearly decodable from each layer.

For each model and layer, a logistic-regression probe is trained on frozen layer representations and evaluated on the held-out test split.

Outputs:

```text
<output_dir>/<model_name>/layerwise_compatibility_probe.csv
```

Probe results should be interpreted as decodability results. They show that information is recoverable from a representation, not that the model necessarily uses that information causally.

## Feature analysis

A separate script analyses valid-minus-invalid contrast directions.

Run:

```bash
uv run python feature_analysis_pipeline.py \
  --csv data/classifier_dataset_llm_pipeline.csv \
  --output-dir outputs/feature_analysis_llm \
  --max-length 128 \
  --batch-size 8
```

Run unit tests:

```bash
uv run python feature_analysis_pipeline.py --run-tests
```

The feature analysis computes, for each model and layer, a direction from average invalid representations towards average valid representations. It then measures how well this direction separates valid and invalid held-out examples.

Main outputs:

```text
outputs/feature_analysis_llm/contrast_feature_strength_all_models.csv
outputs/feature_analysis_llm/contrast_feature_transfer.csv
outputs/feature_analysis_llm/contrast_feature_intervention_all_models.csv
outputs/feature_analysis_llm/feature_analysis_captions.txt
outputs/feature_analysis_llm/figures/01_contrast_feature_strength_by_layer.png
outputs/feature_analysis_llm/figures/02_contrast_feature_location_heatmap.png
outputs/feature_analysis_llm/figures/03_contrast_feature_transfer_heatmap.png
outputs/feature_analysis_llm/figures/04_contrast_feature_intervention.png
```

### Feature separation

Feature separation measures how well the valid-minus-invalid contrast direction separates valid and invalid test examples. The reported score is orientation-invariant ROC-AUC.

### Feature transfer

Feature transfer tests whether a contrast direction learned in a source model remains useful as a readout direction in a target model.

This is not activation patching. No representations are inserted into the target model.

Implemented transfer pairs:

- `xlm_r_large` -> `afroxlmr_large`
- `xlm_r_base` -> `afroxlmr_small`

### Probe-level contrast intervention

The feature intervention shifts saved test representations along the learned contrast direction and records how often the probe prediction changes.

This is a probe-level intervention, not full transformer causal tracing.

## Shallow localisation index

The shallow localisation index is an exploratory summary metric. It combines:

- how much ablation sensitivity is concentrated in late layers;
- how similar early and middle representations remain to the upstream reference model.

Higher values indicate a stronger shallow-localisation pattern: late-layer dependence combined with early and middle representational similarity to the reference model.

This index is a heuristic diagnostic and should not be treated as a standard benchmark.

Recommended lineage-specific summaries:

- `afroxlmr_large` relative to `xlm_r_large`
- `afroxlmr_comet` relative to `afroxlmr_large`
- `afroxlmr_small` relative to `xlm_r_base`

## Expected outputs from the main pipeline

The main pipeline saves:

```text
<output_dir>/validated_dataset.csv
<output_dir>/cross_model_linear_cka.csv
<output_dir>/cross_model_activation_patching.csv
<output_dir>/false_localisation_summary_by_family.json
<output_dir>/figures/
<output_dir>/<model_name>/test_metrics.json
<output_dir>/<model_name>/test_predictions.csv
<output_dir>/<model_name>/layer_noise_ablation.csv
<output_dir>/<model_name>/within_model_activation_patching.csv
<output_dir>/<model_name>/layerwise_compatibility_probe.csv
```

Some files are created only when the corresponding analysis is run and the required data are available.

## Interpretation guide

The analyses answer different questions:

- classifier performance: whether frozen representations support the task;
- CKA: whether two models have similar representational geometry;
- noise ablation: whether a layer is needed for task performance;
- compatibility probes: whether the valid/invalid distinction is decodable at a layer;
- within-model patching: whether valid/invalid paired representations affect predictions;
- cross-model patching: whether upstream representations can substitute for adapted representations;
- feature separation: whether a contrast direction separates valid from invalid examples;
- feature transfer: whether that contrast direction is partly preserved across localisation.

The main empirical interpretation is that localisation is partial and distributed. The adapted models preserve substantial inherited structure from their upstream XLM-R models, but institutional-validity information can become more clearly distinguishable and less functionally interchangeable at later depths.

## Repository

Implementation repository:

```text
https://github.com/graymanicom/afroxlmr_comparison
```
