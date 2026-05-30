from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

REQUIRED_COLUMNS = ["id", "text", "label", "split"]
DEPTH_POSITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEPTH_LABELS = [f"{int(round(100 * p))}%" for p in DEPTH_POSITIONS]


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_name: str


@dataclasses.dataclass
class DirectionInfo:
    layer: int
    scaler: StandardScaler
    direction: np.ndarray
    raw_auc: float
    strength_auc: float
    orientation: int


DEFAULT_MODELS = [
    ModelSpec("xlm_r_large", "xlm-roberta-large"),
    ModelSpec("afroxlmr_large", "Davlan/afro-xlmr-large"),
    ModelSpec("afroxlmr_comet", "local_models/afro-xlmr-comet"),
    ModelSpec("xlm_r_base", "xlm-roberta-base"),
    ModelSpec("afroxlmr_small", "Davlan/afro-xlmr-small"),
]

# These are the localisation lineage pairs for which raw feature-direction transfer is meaningful.
# They have matching hidden sizes and documented upstream -> adapted relationships.
DEFAULT_TRANSFER_PAIRS = [
    ("xlm_r_large", "afroxlmr_large"),
    ("xlm_r_base", "afroxlmr_small"),
]


class TextDataset(Dataset):
    def __init__(self, encodings: Mapping[str, Sequence[Any]], labels: Sequence[int], ids: Sequence[str]):
        self.encodings = encodings
        self.labels = list(labels)
        self.ids = list(ids)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        item["example_ids"] = self.ids[idx]
        return item


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def save_json(data: Mapping[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def nice_model_name(name: str) -> str:
    mapping = {
        "xlm_r_large": "XLM-R Large",
        "afroxlmr_large": "AfroXLMR Large",
        "afroxlmr_comet": "AfroXLMR Comet",
        "xlm_r_base": "XLM-R Base",
        "afroxlmr_small": "AfroXLMR Small",
    }
    return mapping.get(name, name.replace("_", " ").title())


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = df.copy()
    out["id"] = out["id"].astype(str)
    out["text"] = out["text"].astype(str)
    out["label"] = out["label"].astype(int)
    out["split"] = out["split"].astype(str)
    allowed = {"train", "validation", "test"}
    bad_splits = set(out["split"].unique()) - allowed
    if bad_splits:
        raise ValueError(f"Unexpected split values: {sorted(bad_splits)}")
    if set(out["label"].unique()) - {0, 1}:
        raise ValueError("Feature analysis currently expects binary labels 0/1")
    if out["id"].duplicated().any():
        raise ValueError("Duplicate ids found in dataset")
    return out


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "example_ids": [item["example_ids"] for item in batch],
    }


def masked_mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (hidden_state * mask).sum(dim=1) / denom


def layer_indices(num_layers: int) -> list[int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if num_layers == 1:
        return [0 for _ in DEPTH_POSITIONS]
    return [max(0, min(num_layers - 1, int(round(p * (num_layers - 1))))) for p in DEPTH_POSITIONS]


def tokenise_dataframe(df: pd.DataFrame, tokenizer, max_length: int) -> TextDataset:
    encodings = tokenizer(
        df["text"].astype(str).tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    return TextDataset(encodings, df["label"].astype(int).tolist(), df["id"].astype(str).tolist())


def load_backbone_and_tokenizer(spec: ModelSpec, device: torch.device):
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    config = AutoConfig.from_pretrained(spec.hf_name)
    model = AutoModel.from_pretrained(spec.hf_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name, use_fast=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    if device.type == "mps":
        model = model.to(device=device, dtype=torch.float32)
    else:
        model = model.to(device)
    return model, tokenizer


@torch.no_grad()
def extract_layerwise_representations(model, data_loader: DataLoader, device: torch.device) -> tuple[dict[int, np.ndarray], list[str], np.ndarray]:
    model.eval()
    num_layers = int(getattr(model.config, "num_hidden_layers"))
    buffers = {layer: [] for layer in range(num_layers)}
    ids: list[str] = []
    labels: list[int] = []

    for batch in tqdm(data_loader, desc="extract layerwise representations", leave=False):
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        ids.extend([str(x) for x in batch["example_ids"]])
        labels.extend([int(x) for x in batch["labels"].tolist()])
        for layer in range(num_layers):
            pooled = masked_mean_pool(hidden_states[layer + 1], batch["attention_mask"].to(device))
            buffers[layer].append(pooled.detach().cpu().numpy())

    stacked = {layer: np.concatenate(chunks, axis=0) for layer, chunks in buffers.items() if chunks}
    return stacked, ids, np.asarray(labels, dtype=int)


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def fit_contrast_direction(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, layer: int) -> DirectionInfo:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    valid = x_train_s[y_train == 1]
    invalid = x_train_s[y_train == 0]
    if len(valid) == 0 or len(invalid) == 0:
        raise ValueError("Both valid and invalid examples are required")

    direction = valid.mean(axis=0) - invalid.mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        direction = np.zeros_like(direction)
    scores = x_test_s @ direction
    raw_auc = _safe_auc(y_test, scores)
    if math.isnan(raw_auc):
        strength_auc = float("nan")
        orientation = 1
    elif raw_auc >= 0.5:
        strength_auc = raw_auc
        orientation = 1
    else:
        strength_auc = 1.0 - raw_auc
        orientation = -1

    return DirectionInfo(
        layer=layer,
        scaler=scaler,
        direction=direction,
        raw_auc=float(raw_auc),
        strength_auc=float(strength_auc),
        orientation=orientation,
    )


def analyse_contrast_features(
    model_name: str,
    train_reprs: Mapping[int, np.ndarray],
    y_train: np.ndarray,
    test_reprs: Mapping[int, np.ndarray],
    y_test: np.ndarray,
    model_dir: Path,
    top_k: int = 25,
) -> tuple[pd.DataFrame, dict[int, DirectionInfo]]:
    ensure_dir(model_dir)
    rows = []
    top_rows = []
    directions: dict[int, DirectionInfo] = {}

    for layer in sorted(train_reprs):
        info = fit_contrast_direction(train_reprs[layer], y_train, test_reprs[layer], y_test, layer)
        directions[layer] = info
        rows.append({
            "model": model_name,
            "layer": int(layer),
            "raw_auc": info.raw_auc,
            "strength_auc": info.strength_auc,
            "orientation": int(info.orientation),
            "hidden_size": int(info.direction.shape[0]),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        })

        top_idx = np.argsort(np.abs(info.direction))[::-1][:top_k]
        for rank, dim in enumerate(top_idx, start=1):
            top_rows.append({
                "model": model_name,
                "layer": int(layer),
                "rank": int(rank),
                "dimension": int(dim),
                "weight": float(info.direction[dim]),
                "abs_weight": float(abs(info.direction[dim])),
            })

        np.save(model_dir / f"contrast_direction_layer_{layer:02d}.npy", info.direction)

    strength_df = pd.DataFrame(rows)
    top_df = pd.DataFrame(top_rows)
    save_dataframe(strength_df, model_dir / "contrast_feature_strength.csv")
    save_dataframe(top_df, model_dir / "top_contrast_dimensions.csv")
    return strength_df, directions


def nearest_layer_for_depth(num_layers: int, depth_label: str) -> int:
    mapping = dict(zip(DEPTH_LABELS, layer_indices(num_layers)))
    return mapping[depth_label]


def compute_feature_transfer(
    source_name: str,
    target_name: str,
    source_train_reprs: Mapping[int, np.ndarray],
    source_y_train: np.ndarray,
    source_test_reprs: Mapping[int, np.ndarray],
    source_y_test: np.ndarray,
    target_test_reprs: Mapping[int, np.ndarray],
    target_y_test: np.ndarray,
) -> pd.DataFrame:
    source_layers = sorted(source_train_reprs)
    target_layers = sorted(target_test_reprs)
    source_depth_layers = dict(zip(DEPTH_LABELS, layer_indices(len(source_layers))))
    target_depth_layers = dict(zip(DEPTH_LABELS, layer_indices(len(target_layers))))

    rows = []
    for depth in DEPTH_LABELS:
        source_layer = source_depth_layers[depth]
        target_layer = target_depth_layers[depth]
        x_source_train = source_train_reprs[source_layer]
        x_source_test = source_test_reprs[source_layer]
        x_target_test = target_test_reprs[target_layer]

        if x_source_train.shape[1] != x_target_test.shape[1]:
            rows.append({
                "source_model": source_name,
                "target_model": target_name,
                "aligned_depth": depth,
                "source_layer": int(source_layer),
                "target_layer": int(target_layer),
                "transfer_auc": np.nan,
                "transfer_strength_auc": np.nan,
                "source_self_strength_auc": np.nan,
                "skip_reason": f"hidden_size_mismatch:{x_source_train.shape[1]}!={x_target_test.shape[1]}",
            })
            continue

        info = fit_contrast_direction(x_source_train, source_y_train, x_source_test, source_y_test, source_layer)
        x_target_s = info.scaler.transform(x_target_test)
        scores = x_target_s @ info.direction
        raw_transfer_auc = _safe_auc(target_y_test, scores)
        transfer_strength = np.nan if math.isnan(raw_transfer_auc) else max(raw_transfer_auc, 1.0 - raw_transfer_auc)

        rows.append({
            "source_model": source_name,
            "target_model": target_name,
            "aligned_depth": depth,
            "source_layer": int(source_layer),
            "target_layer": int(target_layer),
            "transfer_auc": float(raw_transfer_auc),
            "transfer_strength_auc": float(transfer_strength),
            "source_self_strength_auc": float(info.strength_auc),
            "skip_reason": "",
        })

    return pd.DataFrame(rows)


def compute_feature_interventions(
    model_name: str,
    train_reprs: Mapping[int, np.ndarray],
    y_train: np.ndarray,
    test_reprs: Mapping[int, np.ndarray],
    y_test: np.ndarray,
    directions: Mapping[int, DirectionInfo],
    alphas: Sequence[float],
) -> pd.DataFrame:
    strength_by_layer = {layer: info.strength_auc for layer, info in directions.items()}
    finite_layers = [layer for layer, score in strength_by_layer.items() if not math.isnan(score)]
    if not finite_layers:
        return pd.DataFrame()
    peak_layer = max(finite_layers, key=lambda layer: strength_by_layer[layer])
    info = directions[peak_layer]

    x_train_s = info.scaler.transform(train_reprs[peak_layer])
    x_test_s = info.scaler.transform(test_reprs[peak_layer])
    oriented_direction = info.orientation * info.direction
    norm = np.linalg.norm(oriented_direction)
    if norm <= 1e-12:
        unit_direction = oriented_direction
    else:
        unit_direction = oriented_direction / norm

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=13)
    clf.fit(x_train_s, y_train)
    base_proba = clf.predict_proba(x_test_s)[:, 1]
    base_pred = (base_proba >= 0.5).astype(int)

    rows = []
    for alpha in alphas:
        shifted = x_test_s + float(alpha) * unit_direction
        proba = clf.predict_proba(shifted)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "model": model_name,
            "layer": int(peak_layer),
            "alpha": float(alpha),
            "prediction_change_rate": float(np.mean(pred != base_pred)),
            "mean_valid_probability": float(np.mean(proba)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "baseline_mean_valid_probability": float(np.mean(base_proba)),
            "baseline_accuracy": float(accuracy_score(y_test, base_pred)),
            "baseline_macro_f1": float(f1_score(y_test, base_pred, average="macro", zero_division=0)),
        })
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_strength(strength_df: pd.DataFrame, fig_dir: Path) -> None:
    if strength_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for model_name, sdf in strength_df.groupby("model"):
        sdf = sdf.sort_values("layer")
        ax.plot(sdf["layer"], sdf["strength_auc"], marker="o", label=nice_model_name(model_name))
    ax.axhline(0.5, linestyle=":", linewidth=1, label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUC-ROC of valid/invalid separation")
    ax.set_ylim(0.45, 1.0)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, fig_dir / "01_contrast_feature_strength_by_layer.png")


def plot_feature_location_heatmap(strength_df: pd.DataFrame, fig_dir: Path) -> None:
    if strength_df.empty:
        return
    df = strength_df.copy()
    df["model_label"] = df["model"].map(nice_model_name)
    pivot = df.pivot(index="model_label", columns="layer", values="strength_auc")
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    fig, ax = plt.subplots(figsize=(9.0, max(3.0, 0.55 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0.5, vmax=1.0)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Model")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Feature separation score")
    save_figure(fig, fig_dir / "02_contrast_feature_location_heatmap.png")


def plot_feature_transfer_heatmap(transfer_df: pd.DataFrame, fig_dir: Path) -> None:
    if transfer_df.empty:
        return
    work = transfer_df.loc[transfer_df["skip_reason"].fillna("").astype(str) == ""].copy()
    if work.empty:
        return
    work["pair"] = work["source_model"].map(nice_model_name) + " → " + work["target_model"].map(nice_model_name)
    pivot = work.pivot(index="pair", columns="aligned_depth", values="transfer_strength_auc")
    pivot = pivot[[c for c in DEPTH_LABELS if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(8.5, max(3.0, 0.8 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0.5, vmax=1.0)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_xlabel("Aligned depth")
    ax.set_ylabel("Transferred feature direction")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Transfer separation score")
    save_figure(fig, fig_dir / "03_contrast_feature_transfer_heatmap.png")


def plot_feature_intervention(intervention_df: pd.DataFrame, fig_dir: Path) -> None:
    if intervention_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for model_name, sdf in intervention_df.groupby("model"):
        sdf = sdf.sort_values("alpha")
        ax.plot(sdf["alpha"], sdf["prediction_change_rate"], marker="o", label=nice_model_name(model_name))
    ax.axvline(0, linestyle=":", linewidth=1)
    ax.set_xlabel("Intervention strength along contrast direction")
    ax.set_ylabel("Probe prediction change rate")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, fig_dir / "04_contrast_feature_intervention.png")


def write_captions(out_dir: Path) -> None:
    captions = {
        "01_contrast_feature_strength_by_layer.png": (
            "Layerwise contrast-feature strength. For each model and layer, a valid-minus-invalid "
            "activation direction is estimated on the training split and evaluated on held-out test examples. "
            "The y-axis is orientation-invariant AUC, so 0.5 means chance separation and values closer to 1 "
            "mean that the direction cleanly separates valid institutional sentences from invalid swapped variants. "
            "Peaks indicate layers where the institutional-validity contrast is most linearly salient."
        ),
        "02_contrast_feature_location_heatmap.png": (
            "Feature-location heatmap. Each cell reports the held-out separation score for the contrast direction "
            "at one model layer. Brighter cells indicate layers where the valid/invalid institutional contrast is more "
            "strongly encoded. A shift in the brightest region across a localisation lineage suggests that the contrast "
            "has changed location or salience during adaptation."
        ),
        "03_contrast_feature_transfer_heatmap.png": (
            "Cross-model feature-transfer heatmap. A contrast direction is learned in the source model and tested on "
            "the target model at aligned depths. High scores suggest that the source model's feature direction remains "
            "functionally usable in the localised model. Low scores suggest that localisation may have reorganised the "
            "feature or changed the coordinate basis. This plot is only produced for hidden-size-compatible lineage pairs."
        ),
        "04_contrast_feature_intervention.png": (
            "Feature-direction intervention. For each model, the strongest layerwise contrast direction is used as an "
            "intervention direction on held-out representations, and a probe's prediction change rate is measured as "
            "representations are shifted along that direction. This is a causal test of the probe-level representation, "
            "not a full transformer causal-tracing experiment. Larger changes imply that the contrast direction is more "
            "consequential for the probe's classification boundary."
        ),
        "methodological_note": (
            "This script identifies contrastive activation directions, not fully isolated human-interpretable features. "
            "The term feature should therefore be read as a candidate localisation-relevant direction in activation space. "
            "These analyses complement CKA, probes and patching by asking whether the valid/invalid institutional contrast "
            "becomes stronger, shifts layer location or transfers across localisation lineages."
        ),
    }
    save_json(captions, out_dir / "feature_analysis_captions.json")
    lines = []
    for key, value in captions.items():
        lines.append(f"{key}\n{value}\n")
    (out_dir / "feature_analysis_captions.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_model_args(model_args: Sequence[str] | None) -> list[ModelSpec]:
    if not model_args:
        return list(DEFAULT_MODELS)
    specs = []
    for item in model_args:
        if "=" not in item:
            raise ValueError("Model arguments must have form name=hf_or_local_path")
        name, hf_name = item.split("=", 1)
        specs.append(ModelSpec(name.strip(), hf_name.strip()))
    return specs


def run_feature_analysis(
    csv_path: Path,
    output_dir: Path,
    max_length: int = 128,
    batch_size: int = 8,
    seed: int = 13,
    device_name: str | None = None,
    model_args: Sequence[str] | None = None,
) -> None:
    set_seed(seed)
    ensure_dir(output_dir)
    fig_dir = output_dir / "figures"
    ensure_dir(fig_dir)

    if device_name is None:
        if torch.backends.mps.is_available():
            device_name = "mps"
        elif torch.cuda.is_available():
            device_name = "cuda"
        else:
            device_name = "cpu"
    device = torch.device(device_name)

    df = validate_dataset(pd.read_csv(csv_path))
    train_df = df.loc[df["split"] == "train"].reset_index(drop=True)
    test_df = df.loc[df["split"] == "test"].reset_index(drop=True)
    if train_df.empty or test_df.empty:
        raise ValueError("Both train and test splits are required")

    save_dataframe(df, output_dir / "feature_analysis_dataset_snapshot.csv")

    specs = parse_model_args(model_args)
    all_strength = []
    all_transfer = []
    all_interventions = []
    repr_store: dict[str, dict[str, Any]] = {}

    for spec in specs:
        print(f"\n=== Feature analysis: {spec.name} ===")
        model_dir = output_dir / spec.name
        ensure_dir(model_dir)
        try:
            model, tokenizer = load_backbone_and_tokenizer(spec, device)
        except Exception as exc:
            print(f"Skipping {spec.hf_name} due to load failure: {type(exc).__name__}: {exc}")
            continue

        train_ds = tokenise_dataframe(train_df, tokenizer, max_length)
        test_ds = tokenise_dataframe(test_df, tokenizer, max_length)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

        train_reprs, train_ids, y_train = extract_layerwise_representations(model, train_loader, device)
        test_reprs, test_ids, y_test = extract_layerwise_representations(model, test_loader, device)
        strength_df, directions = analyse_contrast_features(
            model_name=spec.name,
            train_reprs=train_reprs,
            y_train=y_train,
            test_reprs=test_reprs,
            y_test=y_test,
            model_dir=model_dir,
        )
        all_strength.append(strength_df)

        intervention_df = compute_feature_interventions(
            model_name=spec.name,
            train_reprs=train_reprs,
            y_train=y_train,
            test_reprs=test_reprs,
            y_test=y_test,
            directions=directions,
            alphas=[-3, -2, -1, 0, 1, 2, 3],
        )
        if not intervention_df.empty:
            save_dataframe(intervention_df, model_dir / "contrast_feature_intervention.csv")
            all_interventions.append(intervention_df)

        repr_store[spec.name] = {
            "train_reprs": train_reprs,
            "test_reprs": test_reprs,
            "y_train": y_train,
            "y_test": y_test,
            "hidden_size": next(iter(train_reprs.values())).shape[1],
        }

    if all_strength:
        strength_all = pd.concat(all_strength, ignore_index=True)
    else:
        raise RuntimeError("No model feature-strength results were produced")

    for source_name, target_name in DEFAULT_TRANSFER_PAIRS:
        if source_name not in repr_store or target_name not in repr_store:
            continue
        transfer_df = compute_feature_transfer(
            source_name=source_name,
            target_name=target_name,
            source_train_reprs=repr_store[source_name]["train_reprs"],
            source_y_train=repr_store[source_name]["y_train"],
            source_test_reprs=repr_store[source_name]["test_reprs"],
            source_y_test=repr_store[source_name]["y_test"],
            target_test_reprs=repr_store[target_name]["test_reprs"],
            target_y_test=repr_store[target_name]["y_test"],
        )
        all_transfer.append(transfer_df)

    transfer_all = pd.concat(all_transfer, ignore_index=True) if all_transfer else pd.DataFrame()
    intervention_all = pd.concat(all_interventions, ignore_index=True) if all_interventions else pd.DataFrame()

    save_dataframe(strength_all, output_dir / "contrast_feature_strength_all_models.csv")
    if not transfer_all.empty:
        save_dataframe(transfer_all, output_dir / "contrast_feature_transfer.csv")
    if not intervention_all.empty:
        save_dataframe(intervention_all, output_dir / "contrast_feature_intervention_all_models.csv")

    plot_feature_strength(strength_all, fig_dir)
    plot_feature_location_heatmap(strength_all, fig_dir)
    plot_feature_transfer_heatmap(transfer_all, fig_dir)
    plot_feature_intervention(intervention_all, fig_dir)
    write_captions(output_dir)
    print(f"\nSaved feature analysis to: {output_dir}")


def run_unit_tests() -> None:
    rng = np.random.default_rng(13)
    n_train = 200
    n_test = 100
    dim = 16
    y_train = np.array([0] * (n_train // 2) + [1] * (n_train // 2))
    y_test = np.array([0] * (n_test // 2) + [1] * (n_test // 2))
    x_train = rng.normal(size=(n_train, dim))
    x_test = rng.normal(size=(n_test, dim))
    x_train[y_train == 1, 0] += 2.0
    x_train[y_train == 0, 0] -= 2.0
    x_test[y_test == 1, 0] += 2.0
    x_test[y_test == 0, 0] -= 2.0

    info = fit_contrast_direction(x_train, y_train, x_test, y_test, layer=0)
    assert info.strength_auc > 0.95, f"Expected strong separation, got {info.strength_auc}"
    assert info.direction.shape == (dim,)

    train_reprs = {0: x_train, 1: x_train + rng.normal(scale=0.01, size=x_train.shape)}
    test_reprs = {0: x_test, 1: x_test + rng.normal(scale=0.01, size=x_test.shape)}
    strength_df, directions = analyse_contrast_features("synthetic", train_reprs, y_train, test_reprs, y_test, Path("/tmp/feature_analysis_test"))
    assert set(strength_df.columns).issuperset({"model", "layer", "strength_auc"})
    assert len(directions) == 2

    transfer_df = compute_feature_transfer("a", "b", train_reprs, y_train, test_reprs, y_test, test_reprs, y_test)
    assert len(transfer_df) == len(DEPTH_LABELS)
    assert transfer_df["transfer_strength_auc"].dropna().mean() > 0.95

    mismatch_reprs = {0: rng.normal(size=(n_test, dim + 1)), 1: rng.normal(size=(n_test, dim + 1))}
    mismatch_df = compute_feature_transfer("a", "c", train_reprs, y_train, test_reprs, y_test, mismatch_reprs, y_test)
    assert mismatch_df["skip_reason"].astype(str).str.contains("hidden_size_mismatch").any()

    intervention_df = compute_feature_interventions("synthetic", train_reprs, y_train, test_reprs, y_test, directions, [-1, 0, 1])
    assert not intervention_df.empty
    assert set(intervention_df.columns).issuperset({"alpha", "prediction_change_rate", "mean_valid_probability"})
    print("All unit tests passed.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Contrastive feature analysis for localisation-relevant activation directions.")
    parser.add_argument("--csv", type=Path, default=None, help="Classifier dataset with train/test splits")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/feature_analysis"))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model", action="append", default=None, help="Optional model spec in form name=hf_or_local_path. Can be repeated.")
    parser.add_argument("--run-tests", action="store_true", help="Run lightweight unit tests and exit")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.run_tests:
        run_unit_tests()
        return
    if args.csv is None:
        raise ValueError("--csv is required unless --run-tests is used")
    run_feature_analysis(
        csv_path=args.csv,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
        device_name=args.device,
        model_args=args.model,
    )


if __name__ == "__main__":
    main()
