from __future__ import annotations
from pathlib import Path
import typer
from src.config import load_config
from src.pipeline import LocalisationPipeline

import os
print("Running from:", os.getcwd())

app = typer.Typer(add_completion=False, help="Multilingual localisation pair pipeline.")

@app.command()
def ingest(
    config_path: str = typer.Option("config/settings.yaml", help="Path to YAML config file."),
) -> None:
    """Ingest Autshumato and Vuk'uzenzele into one paired-sentence CSV."""
    config = load_config(config_path)
    pipeline = LocalisationPipeline(config)
    pairs_df = pipeline.ingest()
    out_path = pipeline.save_ingested_pairs(pairs_df)
    typer.echo(f"Saved {len(pairs_df):,} rows to {out_path}")


@app.command()
def lexicon(
    config_path: str = typer.Option("config/settings.yaml", help="Path to YAML config file."),
) -> None:
    """Induce lexicons from both corpora and write comparison outputs."""
    config = load_config(config_path)
    pipeline = LocalisationPipeline(config)
    pairs_df = pipeline.ingest()
    lexicons = pipeline.induce_lexicons(pairs_df)
    saved = pipeline.save_lexicon_outputs(lexicons)
    for name, path in saved.items():
        typer.echo(f"[{name}] -> {path}")


@app.command()
def filter(
    config_path: str = typer.Option("config/settings.yaml", help="Path to YAML config file."),
) -> None:
    """Run ingestion, lexicon induction, and the revised pair-filtering stage."""
    config = load_config(config_path)
    pipeline = LocalisationPipeline(config)

    pairs_df = pipeline.ingest()
    pipeline.save_ingested_pairs(pairs_df)

    lexicons = pipeline.induce_lexicons(pairs_df)
    pipeline.save_lexicon_outputs(lexicons)

    filtered_df = pipeline.filter_pairs(pairs_df, lexicons["combined"])
    out_path = pipeline.save_filtered_pairs(filtered_df)

    kept = int(filtered_df["keep"].sum()) if "keep" in filtered_df else 0
    typer.echo(f"Saved {len(filtered_df):,} rows to {out_path}; kept {kept:,} rows.")


@app.command()
def all(
    config_path: str = typer.Option("config/settings.yaml", help="Path to YAML config file."),
) -> None:
    """Run the full pipeline end-to-end.

    This is the main entrypoint most users will want. It performs:
      1. Ingestion of aligned pairs into one dataframe and CSV.
      2. Lexicon induction from Autshumato and Vuk'uzenzele separately.
      3. Lexicon comparison and combined lexicon export.
      4. Revised filtering: English high-precision screen + paired-language validation.
    """
    filter(config_path=config_path)


if __name__ == "__main__":
    app()
