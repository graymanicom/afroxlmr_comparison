from pathlib import Path
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

"""
uv run python src/prepare_classifier_dataset_with_llm.py \
  --classifier-csv /Users/graym/localisation_data/outputs/ins_doc_filtered_replacement_capped_v2/substitutions/classifier_dataset.csv \
  --audit-csv /Users/graym/localisation_data/outputs/ins_doc_filtered_replacement_capped_v2/audit/audit_items_llm_openai.csv \
  --output-csv data/classifier_dataset_llm_pipeline.csv
"""


def prepare_dataset(
    classifier_path: Path,
    audit_path: Path,
    output_path: Path,
    random_state: int = 13,
    balance_labels: bool = True,
) -> None:
    classifier_df = pd.read_csv(classifier_path)
    audit_df = pd.read_csv(audit_path)

    print("Classifier rows:", len(classifier_df))
    print("Audit rows:", len(audit_df))
    print("Audit columns:", audit_df.columns.tolist())

    required_classifier = {"pair_id", "text", "label", "language"}
    missing_classifier = required_classifier - set(classifier_df.columns)
    if missing_classifier:
        raise ValueError(f"classifier_dataset missing columns: {missing_classifier}")

    required_audit = {
        "audit_id",
        "pair_id",
        "language",
        "llm_repaired_sentence",
        "llm_repair_status",
    }
    missing_audit = required_audit - set(audit_df.columns)
    if missing_audit:
        raise ValueError(f"audit_items_llm_openai missing columns: {missing_audit}")

    # ------------------------------------------------------------------
    # 1. Valid examples: original valid statements from classifier dataset
    # ------------------------------------------------------------------
    valid_df = classifier_df.loc[classifier_df["label"].astype(int) == 1].copy()

    # If the valid originals appear more than once per pair_id, keep one.
    valid_df = (
        valid_df.sort_values(["pair_id", "text"])
        .drop_duplicates(subset=["pair_id"], keep="first")
        .copy()
    )

    valid_out = pd.DataFrame(
        {
            "source_row_id": valid_df.get("id", valid_df.index.astype(str)),
            "pair_id": valid_df["pair_id"].astype(str),
            "text": valid_df["text"].astype(str),
            "label": 1,
            "language": valid_df["language"].astype(str),
            "is_local_task": 1,
            "pair_role": "base",
            "example_source": "classifier_original_valid",
        }
    )

    # ------------------------------------------------------------------
    # 2. Invalid examples: LLM-repaired swapped statements from audit file
    # ------------------------------------------------------------------
    invalid_df = audit_df.copy()

    invalid_df = invalid_df.dropna(
        subset=["pair_id", "language", "llm_repaired_sentence"]
    ).copy()

    # Keep only successfully repaired items, if the status column is present.
    invalid_df = invalid_df.loc[
        invalid_df["llm_repair_status"].astype(str).str.lower().eq("ok")
    ].copy()

    invalid_out = pd.DataFrame(
        {
            "source_row_id": invalid_df["audit_id"].astype(str),
            "pair_id": invalid_df["pair_id"].astype(str),
            "text": invalid_df["llm_repaired_sentence"].astype(str),
            "label": 0,
            "language": invalid_df["language"].astype(str),
            "is_local_task": 1,
            "pair_role": "local",  # kept for compatibility with existing patching code
            "example_source": "llm_repaired_invalid_swap",
        }
    )

    # ------------------------------------------------------------------
    # 3. Keep only pair_ids that have both valid and invalid examples
    # ------------------------------------------------------------------
    valid_pairs = set(valid_out["pair_id"])
    invalid_pairs = set(invalid_out["pair_id"])
    usable_pairs = valid_pairs & invalid_pairs

    if not usable_pairs:
        raise ValueError("No overlapping pair_id values between valid and invalid examples.")

    valid_out = valid_out.loc[valid_out["pair_id"].isin(usable_pairs)].copy()
    invalid_out = invalid_out.loc[invalid_out["pair_id"].isin(usable_pairs)].copy()

    # ------------------------------------------------------------------
    # 4. Optional label balancing
    #
    # If balance_labels=True, keep one invalid example per valid pair_id.
    # This prevents the classifier from learning an invalid-majority bias.
    # ------------------------------------------------------------------
    if balance_labels:
        invalid_out = (
            invalid_out.groupby("pair_id", group_keys=False)
            .sample(n=1, random_state=random_state)
            .reset_index(drop=True)
        )

    df = pd.concat([valid_out, invalid_out], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. Split by pair_id to avoid leakage
    # ------------------------------------------------------------------
    pair_ids = pd.Series(df["pair_id"].drop_duplicates())

    train_pairs, temp_pairs = train_test_split(
        pair_ids,
        test_size=0.30,
        random_state=random_state,
    )

    val_pairs, test_pairs = train_test_split(
        temp_pairs,
        test_size=0.50,
        random_state=random_state,
    )

    train_pairs = set(train_pairs)
    val_pairs = set(val_pairs)

    def assign_split(pair_id: str) -> str:
        if pair_id in train_pairs:
            return "train"
        if pair_id in val_pairs:
            return "validation"
        return "test"

    df["split"] = df["pair_id"].apply(assign_split)

    # Stable final IDs.
    df["id"] = [f"llmclf_{i:05d}" for i in range(len(df))]

    out_cols = [
        "id",
        "text",
        "label",
        "language",
        "is_local_task",
        "pair_id",
        "pair_role",
        "split",
    ]

    out = df[out_cols].copy()
    out.to_csv(output_path, index=False)

    print(f"\nWrote: {output_path}")
    print("\nFinal row count:", len(out))
    print("\nLabel counts:")
    print(out["label"].value_counts().sort_index())
    print("\nSplit/label counts:")
    print(out.groupby(["split", "label"]).size())
    print("\nSplit/language counts:")
    print(out.groupby(["split", "language"]).size())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare classifier dataset using valid originals and LLM-repaired invalid swaps."
    )
    parser.add_argument(
        "--classifier-csv",
        type=Path,
        default=Path("data/classifier_dataset.csv"),
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("data/audit_items_llm_openai.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/classifier_dataset_llm_pipeline.csv"),
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Use all LLM-repaired invalid rows instead of sampling one invalid per valid pair.",
    )
    args = parser.parse_args()

    prepare_dataset(
        classifier_path=args.classifier_csv,
        audit_path=args.audit_csv,
        output_path=args.output_csv,
        balance_labels=not args.no_balance,
    )


if __name__ == "__main__":
    main()