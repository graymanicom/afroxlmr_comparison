import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path("outputs/institutional_swap_run_llm_probes_crosspatch_unbiasedCKA")

def compute_false_localisation_index(late_layer_importance_share, early_mid_divergence):
    return float(
        max(
            0.0,
            min(
                1.0,
                0.5 * late_layer_importance_share
                + 0.5 * (1.0 - early_mid_divergence),
            ),
        )
    )

def summarise_late_layer_share(ablation_df):
    work = ablation_df.loc[ablation_df["layer"].astype(str) != "baseline"].copy()
    if work.empty:
        return 0.0

    baseline = ablation_df.loc[ablation_df["layer"].astype(str) == "baseline"]
    if baseline.empty:
        return 0.0

    baseline_acc = float(baseline["overall_accuracy"].iloc[0])
    work["drop"] = baseline_acc - work["overall_accuracy"].astype(float)

    n = len(work)
    cutoff = max(1, math.ceil(2 * n / 3))

    late_drop = work.iloc[cutoff:]["drop"].clip(lower=0).sum()
    total_drop = work["drop"].clip(lower=0).sum()

    return 0.0 if total_drop <= 0 else float(late_drop / total_drop)

def summarise_early_mid_divergence(cka_df):
    if cka_df.empty:
        return 0.0

    metric_col = "linear_cka_debiased" if "linear_cka_debiased" in cka_df.columns else "linear_cka"

    work = cka_df.copy()
    work["pct"] = work["aligned_depth"].str.rstrip("%").astype(int)
    early_mid = work.loc[work["pct"] <= 50]

    if early_mid.empty:
        early_mid = work

    mean_similarity = float(early_mid[metric_col].mean())
    return max(0.0, min(1.0, 1.0 - mean_similarity))

def get_pair_cka(all_cka_df, target_model, reference_model):
    return all_cka_df.loc[
        (
            (all_cka_df["model_a"] == reference_model)
            & (all_cka_df["model_b"] == target_model)
        )
        |
        (
            (all_cka_df["model_a"] == target_model)
            & (all_cka_df["model_b"] == reference_model)
        )
    ].copy()

def compute_summary(target_model, reference_models):
    ablation_path = OUTPUT_DIR / target_model / "layer_noise_ablation.csv"
    if not ablation_path.exists():
        raise FileNotFoundError(ablation_path)

    all_cka_df = pd.read_csv(OUTPUT_DIR / "cross_model_linear_cka.csv")
    ablation_df = pd.read_csv(ablation_path)

    late_share = summarise_late_layer_share(ablation_df)

    divergences = []
    refs_used = []

    for ref in reference_models:
        cka_ref = get_pair_cka(all_cka_df, target_model, ref)
        if cka_ref.empty:
            continue
        divergences.append(summarise_early_mid_divergence(cka_ref))
        refs_used.append(ref)

    if not divergences:
        raise RuntimeError(f"No CKA references found for {target_model}")

    early_mid_divergence = float(np.mean(divergences))
    fli = compute_false_localisation_index(late_share, early_mid_divergence)

    return {
        "target_model": target_model,
        "reference_models_used": refs_used,
        "late_layer_importance_share": late_share,
        "early_mid_divergence": early_mid_divergence,
        "false_localisation_index": fli,
    }

summaries = {
    "afroxlmr_large_from_xlm_r_large": compute_summary(
        "afroxlmr_large",
        ["xlm_r_large"],
    ),
    "afroxlmr_comet_from_afroxlmr_large": compute_summary(
        "afroxlmr_comet",
        ["afroxlmr_large"],
    ),
    "afroxlmr_small_from_xlm_r_base": compute_summary(
        "afroxlmr_small",
        ["xlm_r_base"],
    ),
}

out_path = OUTPUT_DIR / "false_localisation_summary_by_family.json"
out_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

print(json.dumps(summaries, indent=2))
print(f"\nWrote {out_path}")