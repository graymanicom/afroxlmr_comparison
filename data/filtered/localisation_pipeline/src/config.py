from __future__ import annotations
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field
from src.utils.io import load_yaml
from dataclasses import dataclass, field
from typing import Dict


class AutshumatoPairConfig(BaseModel):
    language: str
    english_path: str
    other_path: str


class AutshumatoConfig(BaseModel):
    enabled: bool = True
    pairs: list[AutshumatoPairConfig] = Field(default_factory=list)


class VukuzenzeleConfig(BaseModel):
    enabled: bool = True
    repo_id: str = "dsfsi/vukuzenzele-sentence-aligned"
    subsets: list[str] = Field(default_factory=list)
    splits: list[str] = Field(default_factory=lambda: ["train", "eval", "test"])
    min_alignment_score: float = 0.0


class LLMFilterConfig(BaseModel):
    enabled: bool = False
    model: str = "gpt-5.4"
    batch_size: int = 20
    reasoning_effort: str = "low"
    temperature: float = 0.0
    max_output_tokens: int = 300
    low_confidence_rule_band: tuple[int, int] = (2, 4)


class EnglishFilteringConfig(BaseModel):
    min_words: int = 8
    max_words: int = 32
    min_chars: int = 25
    max_chars: int = 320
    max_digit_ratio: float = 0.25
    max_upper_ratio_short: float = 0.85
    short_text_upper_ratio_len: int = 12
    allow_sources: list[str] = Field(default_factory=list)
    require_institution_hit: bool = True
    require_action_hit: bool = True
    llm: LLMFilterConfig = Field(default_factory=LLMFilterConfig)


class PairedLanguageFilteringConfig(BaseModel):
    min_words: int = 4
    max_words: int = 40
    require_nonempty: bool = True
    relation_window: int = 8
    min_validation_score: int = 1
    fasttext_model_path: str | None = None


class FilteringConfig(BaseModel):
    english: EnglishFilteringConfig = Field(default_factory=EnglishFilteringConfig)
    paired_language: PairedLanguageFilteringConfig = Field(default_factory=PairedLanguageFilteringConfig)

@dataclass
class LexiconSourceThresholds:
    min_candidate_count: int = 2
    min_candidate_score: float = 0.1
    top_k_per_seed_lang: int = 50

@dataclass
class LexiconPruneConfig:
    enabled: bool = True
    min_tokens_by_kind: Dict[str, int] = field(default_factory=lambda: {
        "institution": 2,
        "action": 1,
        "document": 1,
        "locality": 1,
        "argument": 1,
        "access_frame": 1,
    })

    max_tokens_by_kind: Dict[str, int] = field(default_factory=lambda: {
        "institution": 5,
        "action": 4,
        "document": 4,
        "locality": 4,
        "argument": 4,
        "access_frame": 4,
    })

    drop_leading_stopword: bool = True
    drop_trailing_stopword: bool = True
    suppress_subphrases: bool = True
    subphrase_score_ratio: float = 0.9

class LexiconConfig(BaseModel):
    seed_yaml: str = "config/seed_lexicon.yaml"
    max_ngram: int = 4
    min_candidate_count: int = 2
    min_candidate_score: float = 1.5
    min_target_token_chars: int = 2
    stopword_files: dict[str, str] = Field(default_factory=dict)
    source_thresholds: dict[str, LexiconSourceThresholds] = field(default_factory=dict)
    prune: LexiconPruneConfig = field(default_factory=LexiconPruneConfig)
    compare_output: str = "outputs/lexicon_comparison.csv"

class ProjectConfig(BaseModel):
    project_name: str = "localisation-pipeline"
    output_dir: str = "outputs"
    english_column: str = "english"
    other_column: str = "other_sentence"
    language_column: str = "language"
    source_column: str = "source"
    autshumato: AutshumatoConfig = Field(default_factory=AutshumatoConfig)
    vukuzenzele: VukuzenzeleConfig = Field(default_factory=VukuzenzeleConfig)
    filtering: FilteringConfig = Field(default_factory=FilteringConfig)
    lexicon: LexiconConfig = Field(default_factory=LexiconConfig)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)
    
def load_config(path: str | Path) -> ProjectConfig:
    raw: dict[str, Any] = load_yaml(path)
    lex_raw = raw["lexicon"]

    source_thresholds = {
        name: LexiconSourceThresholds(**vals)
        for name, vals in lex_raw.get("source_thresholds", {}).items()
    }

    prune_cfg = LexiconPruneConfig(**lex_raw.get("prune", {}))
    lexicon_cfg = LexiconConfig(
        seed_yaml=lex_raw["seed_yaml"],
        stopword_files=lex_raw["stopword_files"],
        max_ngram=lex_raw.get("max_ngram", 4),
        min_target_token_chars=lex_raw.get("min_target_token_chars", 2),
        source_thresholds=source_thresholds,
        prune=prune_cfg,
        compare_output=lex_raw.get(
            "compare_output",
            "outputs/lexicon/lexicon_comparison.csv",
        ),
    )

    raw["lexicon"] = lexicon_cfg

    return ProjectConfig.model_validate(raw)

