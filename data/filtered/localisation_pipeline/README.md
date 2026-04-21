# Localisation Pipeline

This project rewrites the earlier single-sentence pipeline around a **paired-sentence design**:

- each row keeps the English sentence,
- the aligned non-English sentence,
- the non-English language code,
- and the data source.

It supports two corpora:

- **Autshumato** from local sentence-aligned files on disk
- **Vuk'uzenzele** from the Hugging Face dataset `dsfsi/vukuzenzele-sentence-aligned`

It also implements the revised filtering and lexicon workflow:

1. **paired ingestion** into one dataframe and CSV
2. **high-precision English filtering** using
   - rule-based screening first
   - optional OpenAI LLM adjudication for borderline cases
3. **paired-language validation** on the aligned non-English sentence
4. **lexicon induction** from both Autshumato and Vuk'uzenzele
5. **lexicon comparison** between the two corpora

---

## Project layout

```text
localisation_pipeline/
├── config/
│   ├── seed_lexicon.yaml
│   └── settings.example.yaml
├── src/
│   ├── config.py
│   ├── pipeline.py
│   ├── ingest/
│   │   ├── autshumato.py
│   │   └── vukuzenzele.py
│   ├── filtering/
│   │   ├── llm_filter.py
│   │   ├── pipeline.py
│   │   └── rules.py
│   ├── lexicon/
│   │   ├── induce.py
│   │   └── schema.py
│   └── utils/
│       ├── io.py
│       └── text.py
├── main.py
├── pyproject.toml
└── README.md
```

---

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management
- local access to your Autshumato files
- network access when loading the Hugging Face Vuk'uzenzele dataset
- an `OPENAI_API_KEY` if you enable the LLM filter

---

## Installation with `uv`

From the project root:

```bash
uv sync
```

If you want the optional fastText language-id dependency too:

```bash
uv sync --extra langid
```

To enable development tooling:

```bash
uv sync --extra dev
```

---

## Configuration

Copy the example settings file:

```bash
cp config/settings.example.yaml config/settings.yaml
```

Then edit:

- the local Autshumato file paths
- the Vuk'uzenzele subsets you want to ingest
- filtering thresholds
- whether the OpenAI LLM filter is enabled

The shipped example config is already expanded to your initial nine-language set: Afrikaans, isiXhosa, isiZulu, Sepedi, Sesotho, Setswana, Siswati, Tshivenda, and Xitsonga.

### Important configuration fields

#### `autshumato.pairs`

Each entry must give:

- `language`: the non-English ISO-style code you want to use in outputs
- `english_path`: local path to the English aligned file
- `other_path`: local path to the aligned non-English file

Example:

```yaml
autshumato:
  enabled: true
  pairs:
    - language: xho
      english_path: afroxlmr_comparison/data/filtered/raw/Autshumato/isiXhosa/BilingualCorpus.SADiLaR.English-isiXhosa.2.0.0.CAM.2025-06-10.en.txt
      other_path: afroxlmr_comparison/data/filtered/raw/Autshumato/isiXhosa/BilingualCorpus.SADiLaR.English-isiXhosa.2.0.0.CAM.2025-06-10.xh.txt
```

#### `vukuzenzele.subsets`

Use only subsets that include English. For your current starting setup, the example config includes:

- `eng-afr`
- `eng-xho`
- `eng-zul`
- `eng-nso`
- `eng-sot`
- `eng-tsn`
- `eng-ssw`
- `eng-ven`
- `eng-tso`

The configured splits are concatenated internally, so `train`, `eval`, and `test` can be loaded together.

#### `filtering.english.llm.enabled`

Set this to `true` if you want to use the OpenAI API for borderline English cases.

Then set your API key in the shell:

```bash
export OPENAI_API_KEY="your_key_here"
```

```bash
export HF_TOKEN="your_token_here"
```

or in a local `.env` file:

```env
OPENAI_API_KEY=your_key_here
```

---

## How the pipeline works

### 1. Ingestion

The ingestion stage creates **one row per aligned pair** with columns:

- `pair_id`
- `english`
- `other_sentence`
- `language`
- `source`
- `source_doc_id`
- `source_row_id`
- optional alignment metadata for Vuk'uzenzele

The pair-oriented CSV is written to:

```text
outputs/paired_sentences.csv
```

### 2. Lexicon induction

Lexicon induction starts from the **English seed lexicon** in `config/seed_lexicon.yaml`.

For each English seed item:

1. find aligned rows whose English side contains the seed term
2. collect target-language n-grams from the aligned non-English side
3. score them with a simple PMI-like measure
4. keep candidates that satisfy configurable count and score thresholds

The code induces lexicons separately from:

- `autshumato`
- `vukuzenzele_hf`

and then produces:

- `outputs/autshumato_induced.csv`
- `outputs/vukuzenzele_induced.csv`
- `outputs/combined.csv`
- `outputs/comparison.csv`

The comparison table helps you inspect:

- which terms appear in both induced lexicons
- which terms are only induced from one corpus
- relative counts and scores by corpus

### 3. Revised filtering

Filtering is pair-aware.

#### Step A: High-precision English screening

The English side is screened first using rules:

- length bounds
- metadata-like exclusion
- single-sentence shape heuristic
- anaphor penalty
- temporal brittleness exclusion
- institution lexicon hit
- action lexicon hit

#### Step B: Optional LLM adjudication

If enabled, the OpenAI API is only called for **borderline rule-based cases**.

The LLM returns structured judgments for:

- institution present
- action present
- self-contained
- temporally brittle
- keep / do not keep
- confidence 1–5

This is designed so the LLM is a **precision helper**, not the sole source of truth.

#### Step C: Non-English validation

The aligned non-English sentence is then validated using:

- non-empty check
- length range
- target-language lexicon hits
- optional relation-window heuristic between institution and action hits

The final keep decision is:

```text
keep = english_pass AND paired_language_pass
```

---

## Running the pipeline

### Ingest only

```bash
uv run python main.py ingest --config-path config/settings.yaml
```

### Lexicon induction only

```bash
uv run python main.py lexicon --config-path config/settings.yaml
```

### Filtering only

```bash
uv run python main.py filter --config-path config/settings.yaml
```

### Full pipeline

```bash
uv run python main.py all --config-path config/settings.yaml
```

---

## Output files

Typical outputs are:

- `outputs/paired_sentences.csv`
- `outputs/seed.csv`
- `outputs/autshumato_induced.csv`
- `outputs/autshumato_stats.csv`
- `outputs/vukuzenzele_induced.csv`
- `outputs/vukuzenzele_stats.csv`
- `outputs/combined.csv`
- `outputs/comparison.csv`
- `outputs/filtered_pairs.csv`

---

## Notes on the OpenAI API integration

The LLM filter is implemented with the modern OpenAI Python SDK and the **Responses API**.
In this codebase, the LLM layer is intentionally conservative:

- it only runs on borderline cases
- it is easy to switch off
- its raw output is stored for auditability

If you want completely deterministic runs, leave it disabled and rely on the rule-based English filter.

---

## Suggested workflow for your project

A sensible order for development is:

1. start with a few languages such as `xho` and `zul`
2. run `ingest`
3. inspect `paired_sentences.csv`
4. run `lexicon`
5. inspect the induced lexicons and the comparison table
6. refine the seed lexicon
7. run `filter`
8. manually audit a sample of kept and rejected pairs
9. only then enable the LLM filter for borderline recovery

---

## Important limitations

- Lexicon induction is **auditable and practical**, not magically perfect.
  It still needs manual review and iteration.
- The rule-based relation check in low-resource languages is a fallback heuristic.
- Vuk'uzenzele ingestion depends on Hugging Face dataset availability.
- The OpenAI filter improves recall for borderline English cases, but it should not be treated as ground truth.

---

## Extending this code

Natural next additions would be:

- stronger morphological handling for Bantu languages
- explicit provenance tracking for every induced lexicon entry
- pair-family construction for base/local contrasts
- perturbation generation and minimality scoring
- leakage probes and split logic from the earlier design

