from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ProjectConfig
from src.filtering.pipeline import PairFilterPipeline
from src.ingest.autshumato import ingest_autshumato
from src.ingest.vukuzenzele import ingest_vukuzenzele
from src.lexicon.induce import combine_lexicons, compare_lexicon_sources, induce_lexicon_from_pairs, top_candidates_by_language_and_kind
from src.lexicon.schema import load_seed_lexicon
from src.utils.io import ensure_dir


class LocalisationPipeline:
    def __init__(self, config: ProjectConfig):
        self.config = config
        ensure_dir(config.output_dir)

    def ingest(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        if self.config.autshumato.enabled:
            frames.append(ingest_autshumato(self.config.autshumato))
        if self.config.vukuzenzele.enabled:
            frames.append(ingest_vukuzenzele(self.config.vukuzenzele))
        if not frames:
            raise ValueError("No data sources are enabled.")
        pairs_df = pd.concat(frames, ignore_index=True)
        pairs_df = pairs_df.drop_duplicates(subset=["english", "other_sentence", "language", "source"])
        return pairs_df

    def save_ingested_pairs(self, pairs_df: pd.DataFrame, filename: str = "paired_sentences.csv") -> Path:
        out_path = Path(self.config.output_dir) / filename
        pairs_df.to_csv(out_path, index=False)
        return out_path

    def induce_lexicons(self, pairs_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        seed_df = load_seed_lexicon(self.config.lexicon.seed_yaml)

        aut_df, aut_stats = induce_lexicon_from_pairs(
            pairs_df=pairs_df,
            seed_df=seed_df,
            config=self.config.lexicon,
            source_name="autshumato",
        )
        vuk_df, vuk_stats = induce_lexicon_from_pairs(
            pairs_df=pairs_df,
            seed_df=seed_df,
            config=self.config.lexicon,
            source_name="vukuzenzele_hf",
        )

        aut_top50 = top_candidates_by_language_and_kind(aut_df, top_n=50)
        vuk_top50 = top_candidates_by_language_and_kind(vuk_df, top_n=50)

        combined = combine_lexicons(seed_df, [aut_df, vuk_df])
        comparison = compare_lexicon_sources(
            aut_df,
            vuk_df,
            output_path=self.config.lexicon.compare_output,
        )

        return {
            "seed": seed_df,
            "autshumato_induced": aut_df,
            "autshumato_stats": aut_stats,
            "autshumato_top50_by_lang_kind": aut_top50,
            "vukuzenzele_induced": vuk_df,
            "vukuzenzele_stats": vuk_stats,
            "vukuzenzele_top50_by_lang_kind": vuk_top50,
            "combined": combined,
            "comparison": comparison,
        }

    def save_lexicon_outputs(self, lexicons: dict[str, pd.DataFrame]) -> dict[str, Path]:
        saved: dict[str, Path] = {}
        for name, df in lexicons.items():
            out_path = Path(self.config.output_dir) / f"{name}.csv"
            df.to_csv(out_path, index=False)
            saved[name] = out_path
        return saved

    def filter_pairs(self, pairs_df: pd.DataFrame, combined_lexicon_df: pd.DataFrame) -> pd.DataFrame:
        filter_pipeline = PairFilterPipeline(self.config, combined_lexicon_df)
        return filter_pipeline.filter_pairs(pairs_df)

    def save_filtered_pairs(self, filtered_df: pd.DataFrame, filename: str = "filtered_pairs.csv") -> Path:
        out_path = Path(self.config.output_dir) / filename
        filtered_df.to_csv(out_path, index=False)
        return out_path
