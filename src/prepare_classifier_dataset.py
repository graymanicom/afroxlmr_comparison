from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = Path("/Users/graym/localisation_data/outputs/ins_doc_filtered_replacement_capped_v2/substitutions/classifier_dataset.csv")
OUTPUT = Path("data/classifier_dataset_pipeline.csv")
RANDOM_STATE = 13

df = pd.read_csv(INPUT)

print("Input columns:")
print(df.columns.tolist())

required = {"text", "label", "language", "pair_id"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=["text", "label", "language", "pair_id"]).copy()
df["label"] = df["label"].astype(int)

# ---------------------------------------------------------------------
# Balance the dataset while preserving original/swapped pair structure.
#
# Assumption:
#   label = 1 means original valid sentence
#   label = 0 means invalid swapped sentence
#
# We keep one valid and one invalid example per pair_id where possible.
# This prevents the classifier from learning "everything is invalid".
# ---------------------------------------------------------------------

valid = df[df["label"] == 1].copy()
invalid = df[df["label"] == 0].copy()

valid_pair_ids = set(valid["pair_id"])
invalid_pair_ids = set(invalid["pair_id"])

usable_pair_ids = sorted(valid_pair_ids & invalid_pair_ids)

if not usable_pair_ids:
    raise ValueError("No pair_id has both a valid and invalid example.")

valid_balanced = (
    valid[valid["pair_id"].isin(usable_pair_ids)]
    .groupby("pair_id", group_keys=False)
    .sample(n=1, random_state=RANDOM_STATE)
)

invalid_balanced = (
    invalid[invalid["pair_id"].isin(usable_pair_ids)]
    .groupby("pair_id", group_keys=False)
    .sample(n=1, random_state=RANDOM_STATE)
)

df = pd.concat([valid_balanced, invalid_balanced], ignore_index=True)

# Shuffle after balancing.
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print("\nBalanced label counts:")
print(df["label"].value_counts())

# Create stable row IDs after balancing.
df["id"] = [f"clf_{i:05d}" for i in range(len(df))]

# All examples are localisation-relevant South African institutional examples.
df["is_local_task"] = 1

# Keep compatibility with existing activation patching code.
# Conceptual meaning:
#   base  = original valid sentence
#   local = swapped invalid sentence
df["pair_role"] = df["label"].map({
    1: "base",
    0: "local",
})

# Split by pair_id to avoid leakage.
pair_ids = pd.Series(df["pair_id"].drop_duplicates())

train_pairs, temp_pairs = train_test_split(
    pair_ids,
    test_size=0.30,
    random_state=RANDOM_STATE,
)

val_pairs, test_pairs = train_test_split(
    temp_pairs,
    test_size=0.50,
    random_state=RANDOM_STATE,
)

train_pairs = set(train_pairs)
val_pairs = set(val_pairs)

def assign_split(pair_id):
    if pair_id in train_pairs:
        return "train"
    if pair_id in val_pairs:
        return "validation"
    return "test"

df["split"] = df["pair_id"].apply(assign_split)

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
out.to_csv(OUTPUT, index=False)

print(f"\nWrote {OUTPUT}")

print("\nSplit/label counts:")
print(out.groupby(["split", "label"]).size())

print("\nSplit/language counts:")
print(out.groupby(["split", "language"]).size())

print("\nPair role counts:")
print(out.groupby(["split", "pair_role"]).size())