# Improved semantic-validity dataset

This version is less templated, more institutionally grounded, and includes stronger local-vs-generic test pairs.

Use it with the existing pipeline unchanged because the CSV schema is identical.

Suggested README/code changes:
- replace sentiment wording with semantic validity or institutional plausibility
- keep is_local_task as an analysis tag, not the training target
- add patching fields such as patched_matches_local_label or patched_moves_toward_local_label

Summary:
```json
{
  "rows": 500,
  "split_counts": {
    "train": 300,
    "validation": 100,
    "test": 100
  },
  "label_counts": {
    "1": 250,
    "0": 250
  },
  "language_counts": {
    "eng": 349,
    "swa": 71,
    "zul": 80
  },
  "local_counts": {
    "1": 302,
    "0": 198
  },
  "paired_test_rows": 60
}
```
