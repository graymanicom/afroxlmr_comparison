from __future__ import annotations

import os
from dataclasses import asdict

import pandas as pd
from dotenv import load_dotenv

from src.config import ProjectConfig
from src.filtering.llm_filter import OpenAILLMFilter
from src.filtering.rules import build_lexicon_lookup, score_english_candidate, validate_paired_language


class PairFilterPipeline:
    """Apply the revised filtering strategy requested by the user.

    1. High-precision English filter.
       - Rule-based screen first.
       - Optional OpenAI LLM adjudication for borderline cases.
    2. Lightweight validation of the paired non-English sentence.
    """

    def __init__(self, config: ProjectConfig, lexicon_df: pd.DataFrame):
        self.config = config
        self.lexicon_df = lexicon_df
        self.lookup = build_lexicon_lookup(lexicon_df)
        self.english_lookup = self.lookup.get("eng", {})

        load_dotenv()
        self.llm = None
        llm_cfg = config.filtering.english.llm
        if llm_cfg.enabled:
            self.llm = OpenAILLMFilter(api_key=os.getenv("OPENAI_API_KEY"), config=llm_cfg)

    def filter_pairs(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        low_band_start, low_band_end = self.config.filtering.english.llm.low_confidence_rule_band

        for record in pairs_df.to_dict(orient="records"):
            english_result = score_english_candidate(
                english_text=str(record["english"]),
                source=str(record["source"]),
                english_lookup=self.english_lookup,
                config=self.config.filtering.english,
            )

            llm_used = False
            llm_keep = None
            llm_confidence = None
            llm_explanation = None

            english_keep = english_result.passed
            # Only escalate to the LLM for borderline rule scores.
            if self.llm and low_band_start <= english_result.score <= low_band_end:
                decision = self.llm.screen_sentence(str(record["english"]))
                llm_used = True
                llm_keep = decision.keep
                llm_confidence = decision.confidence
                llm_explanation = decision.explanation
                english_keep = english_keep or (
                    decision.keep
                    and decision.confidence >= 4
                    and decision.institution_present
                    and decision.action_present
                    and decision.self_contained
                    and not decision.temporally_brittle
                )

            paired_result = validate_paired_language(
                other_text=str(record["other_sentence"]),
                language=str(record["language"]),
                lexicon_lookup=self.lookup,
                config=self.config.filtering.paired_language,
            )

            keep = english_keep and paired_result.passed
            rows.append(
                {
                    **record,
                    "keep": keep,
                    "english_rule_pass": english_result.passed,
                    "english_rule_score": english_result.score,
                    "english_rule_reasons": "|".join(english_result.reasons),
                    "paired_validation_pass": paired_result.passed,
                    "paired_validation_score": paired_result.score,
                    "paired_validation_reasons": "|".join(paired_result.reasons),
                    "llm_used": llm_used,
                    "llm_keep": llm_keep,
                    "llm_confidence": llm_confidence,
                    "llm_explanation": llm_explanation,
                }
            )

        return pd.DataFrame(rows)
