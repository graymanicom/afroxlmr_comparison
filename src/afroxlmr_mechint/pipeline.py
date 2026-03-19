from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

REQUIRED_COLUMNS = [
    "id", "text", "label", "language", "is_local_task", "pair_id", "pair_role", "split"
]


class DatasetSchemaError(ValueError):
    """Raised when the dataset CSV does not match the expected schema."""


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Mapping[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RelativeDepthAligner:
    """
    Align layers across models of different depth using relative depth.
    This is more defensible than raw-index comparison for heterogeneous backbones.
    """
    def __init__(self, relative_positions: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)):
        if not relative_positions:
            raise ValueError("relative_positions must be non-empty")
        for pos in relative_positions:
            if pos < 0 or pos > 1:
                raise ValueError("relative positions must lie in [0, 1]")
        self.relative_positions = tuple(relative_positions)

    def layer_indices(self, num_layers: int) -> List[int]:
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_layers == 1:
            return [0 for _ in self.relative_positions]
        return [max(0, min(num_layers - 1, int(round(pos * (num_layers - 1))))) for pos in self.relative_positions]

    def labelled_positions(self) -> List[str]:
        return [f"{int(round(100 * pos))}%" for pos in self.relative_positions]


def validate_dataset_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetSchemaError(f"Missing required columns: {missing}")

    out = df.copy()
    if out["id"].duplicated().any():
        dupes = out.loc[out["id"].duplicated(), "id"].tolist()
        raise DatasetSchemaError(f"Duplicate ids found: {dupes[:10]}")

    out["pair_role"] = out["pair_role"].fillna("").astype(str)
    if not set(out["pair_role"].unique()).issubset({"", "base", "local"}):
        raise DatasetSchemaError("pair_role must be base, local, or blank")

    out["pair_id"] = out["pair_id"].fillna("").astype(str)
    if not set(out["split"].astype(str).unique()).issubset({"train", "validation", "test"}):
        raise DatasetSchemaError("split must only contain train, validation, test")

    try:
        out["label"] = out["label"].astype(int)
        out["is_local_task"] = out["is_local_task"].astype(int)
    except Exception as exc:
        raise DatasetSchemaError("label and is_local_task must be castable to int") from exc

    grouped = out.loc[out["pair_id"] != ""].groupby("pair_id")
    for pair_id, grp in grouped:
        roles = set(grp["pair_role"].tolist())
        if "base" not in roles or "local" not in roles:
            raise DatasetSchemaError(f"Pair {pair_id} must contain both base and local rows")
    return out


class TextClassificationDataset(Dataset):
    """Dataset wrapper for tokenised classification inputs."""
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


@dataclasses.dataclass
class ModelSpec:
    name: str
    hf_name: str
    num_labels: int


class SharedClassifierWrapper(nn.Module):
    """
    Shared classifier over arbitrary encoder backbones.

    This avoids comparing inconsistent shipped heads and keeps the comparison focused
    on backbone representations.
    """
    def __init__(self, backbone: nn.Module, hidden_size: int, num_labels: int, dropout_prob: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None, output_hidden_states=False):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        pooled = masked_mean_pool(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {
            "loss": loss,
            "logits": logits,
            "pooled_output": pooled,
            "hidden_states": outputs.hidden_states if output_hidden_states else None,
        }


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool over non-padding tokens."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    masked = last_hidden_state * mask
    denom = mask.sum(dim=1).clamp(min=1.0)
    return masked.sum(dim=1) / denom


def pooled_cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine similarity between the mean vectors of two representation matrices."""
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    xn = np.linalg.norm(x_mean)
    yn = np.linalg.norm(y_mean)
    if xn == 0 or yn == 0:
        return 0.0
    return float(np.dot(x_mean, y_mean) / (xn * yn))


def linear_cka(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """
    Linear CKA for comparing hidden representations.

    CKA is preferred here to naive cosine similarity because it is less sensitive
    to isotropic scaling and orthogonal transforms of feature spaces.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    x_ty = x.T @ y
    x_tx = x.T @ x
    y_ty = y.T @ y
    numerator = float(np.sum(x_ty ** 2))
    denom = math.sqrt(float(np.sum(x_tx ** 2)) * float(np.sum(y_ty ** 2))) + eps
    return numerator / denom


def compute_false_localisation_index(late_layer_importance_share: float, early_mid_divergence: float) -> float:
    """
    Heuristic synthesis metric.

    High values indicate:
    - local behaviour depends heavily on late layers, and
    - early/mid representations remain too similar to reference models.
    """
    raw = 0.5 * late_layer_importance_share + 0.5 * (1.0 - early_mid_divergence)
    return float(max(0.0, min(1.0, raw)))


def build_pair_mapping(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    pairs = []
    for pair_id, grp in df.loc[df["pair_id"] != ""].groupby("pair_id"):
        base_row = grp.loc[grp["pair_role"] == "base"]
        local_row = grp.loc[grp["pair_role"] == "local"]
        if len(base_row) == 1 and len(local_row) == 1:
            pairs.append((pair_id, str(base_row.iloc[0]["id"]), str(local_row.iloc[0]["id"])))
    return pairs


def split_dataframe(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {split: sdf.reset_index(drop=True) for split, sdf in df.groupby("split")}


def _lazy_import_transformers():
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    return AutoConfig, AutoModel, AutoTokenizer


def load_model_and_tokenizer(spec: ModelSpec, device: torch.device):
    AutoConfig, AutoModel, AutoTokenizer = _lazy_import_transformers()
    config = AutoConfig.from_pretrained(spec.hf_name)
    backbone = AutoModel.from_pretrained(spec.hf_name, config=config)
    hidden_size = int(getattr(config, "hidden_size"))
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name, use_fast=True)
    model = SharedClassifierWrapper(
        backbone=backbone,
        hidden_size=hidden_size,
        num_labels=spec.num_labels,
    )

    # On Apple MPS, keep all floating model parameters in float32 to avoid
    # Metal dtype-mismatch failures during matrix multiplication.
    if device.type == "mps":
        model = model.to(device=device, dtype=torch.float32)
    else:
        model = model.to(device)

    return model, tokenizer

def tokenise_dataframe(df: pd.DataFrame, tokenizer, max_length: int) -> TextClassificationDataset:
    encodings = tokenizer(
        df["text"].astype(str).tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    return TextClassificationDataset(encodings=encodings, labels=df["label"].tolist(), ids=df["id"].tolist())


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "example_ids": [item["example_ids"] for item in batch],
    }


def train_classifier(model, train_loader, val_loader, device, epochs, learning_rate, output_dir: Path):
    """Light fine-tuning suitable for laptop hardware."""
    ensure_dir(output_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"train epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad(set_to_none=True)
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                output_hidden_states=False,
            )
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses))}
        if val_loader is not None:
            val_metrics = evaluate_classifier(model, val_loader, device)
            record.update({f"val_{k}": v for k, v in val_metrics.items() if k != "predictions"})
        history.append(record)
        save_json({"history": history}, output_dir / "training_history.json")


@torch.no_grad()
def evaluate_classifier(model, data_loader, device):
    model.eval()
    y_true, y_pred, rows = [], [], []
    for batch in tqdm(data_loader, desc="evaluate", leave=False):
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=None,
            output_hidden_states=False,
        )
        preds = out["logits"].argmax(dim=-1).detach().cpu().tolist()
        true = batch["labels"].tolist()
        for ex_id, t, p in zip(batch["example_ids"], true, preds):
            rows.append({"id": ex_id, "label": t, "prediction": p})
        y_true.extend(true)
        y_pred.extend(preds)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "predictions": pd.DataFrame(rows),
    }


@torch.no_grad()
def extract_aligned_representations(model, data_loader, device, aligner: RelativeDepthAligner):
    """
    Extract pooled sequence representations at aligned relative depths.

    The hidden state list includes embeddings at index 0, then layer outputs.
    Therefore aligned layer k corresponds to hidden_states[k + 1].
    """
    model.eval()
    num_layers = int(getattr(model.backbone.config, "num_hidden_layers"))
    aligned_indices = aligner.layer_indices(num_layers)
    aligned_labels = aligner.labelled_positions()
    buffers = {lab: [] for lab in aligned_labels}
    meta_rows = []

    for batch in tqdm(data_loader, desc="extract representations", leave=False):
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=None,
            output_hidden_states=True,
        )
        hidden_states = out["hidden_states"]
        for ex_id in batch["example_ids"]:
            meta_rows.append({"id": ex_id})
        for lab, idx in zip(aligned_labels, aligned_indices):
            hs = hidden_states[idx + 1]
            pooled = masked_mean_pool(hs, batch["attention_mask"].to(device)).detach().cpu().numpy()
            buffers[lab].append(pooled)

    stacked = {lab: np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1)) for lab, chunks in buffers.items()}
    return stacked, pd.DataFrame(meta_rows)


def compute_similarity_tables(repr_a, repr_b, model_a: str, model_b: str):
    cosine_rows, cka_rows = [], []
    for k in [k for k in repr_a.keys() if k in repr_b]:
        x, y = repr_a[k], repr_b[k]
        common_dim = min(x.shape[1], y.shape[1])
        x_use, y_use = x[:, :common_dim], y[:, :common_dim]
        cosine_rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "aligned_depth": k,
            "cosine_mean_similarity": pooled_cosine_similarity(x_use, y_use),
        })
        cka_rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "aligned_depth": k,
            "linear_cka": linear_cka(x_use, y_use),
        })
    return pd.DataFrame(cosine_rows), pd.DataFrame(cka_rows)


def _zero_layer_output_hook(module, inputs, output):
    """Zero the primary transformer-block output tensor while preserving tuple structure."""
    if isinstance(output, tuple):
        return (torch.zeros_like(output[0]),) + output[1:]
    return torch.zeros_like(output)


@torch.no_grad()
def evaluate_with_layer_zero_ablation(model, data_loader, device, local_ids: set):
    """
    Coarse causal intervention:
    zero one encoder layer at a time and measure the performance drop.
    """
    layers = model.backbone.encoder.layer
    baseline = evaluate_classifier(model, data_loader, device)
    pred_df = baseline["predictions"].copy()
    pred_df["is_local_task"] = pred_df["id"].isin(local_ids).astype(int)

    def subset_acc(frame):
        return float((frame["label"] == frame["prediction"]).mean()) if len(frame) else np.nan

    records = [{
        "layer": "baseline",
        "overall_accuracy": baseline["accuracy"],
        "overall_macro_f1": baseline["macro_f1"],
        "local_accuracy": subset_acc(pred_df.loc[pred_df["is_local_task"] == 1]),
        "global_accuracy": subset_acc(pred_df.loc[pred_df["is_local_task"] == 0]),
    }]

    for idx, layer in enumerate(layers):
        handle = layer.register_forward_hook(_zero_layer_output_hook)
        ablated = evaluate_classifier(model, data_loader, device)
        handle.remove()
        pred_df = ablated["predictions"].copy()
        pred_df["is_local_task"] = pred_df["id"].isin(local_ids).astype(int)
        records.append({
            "layer": idx,
            "overall_accuracy": ablated["accuracy"],
            "overall_macro_f1": ablated["macro_f1"],
            "local_accuracy": subset_acc(pred_df.loc[pred_df["is_local_task"] == 1]),
            "global_accuracy": subset_acc(pred_df.loc[pred_df["is_local_task"] == 0]),
        })
    return pd.DataFrame(records)


@torch.no_grad()
def paired_activation_patching(model, tokenizer, pairs_df: pd.DataFrame, device, aligner: RelativeDepthAligner, max_length: int):
    """
    Coarse within-model layer patching.

    For each pair:
    - cache the layer output for the 'local' input,
    - patch it into the corresponding layer for the 'base' input,
    - observe whether the prediction changes.
    """
    model.eval()
    num_layers = int(getattr(model.backbone.config, "num_hidden_layers"))
    aligned_indices = aligner.layer_indices(num_layers)
    aligned_labels = aligner.labelled_positions()
    id_to_row = {str(row["id"]): row for _, row in pairs_df.iterrows()}
    rows = []

    for pair_id, base_id, local_id in tqdm(build_pair_mapping(pairs_df), desc="patch pairs", leave=False):
        base_row, local_row = id_to_row[base_id], id_to_row[local_id]
        base_inputs = tokenizer([str(base_row["text"])], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        local_inputs = tokenizer([str(local_row["text"])], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        base_out = model(
            input_ids=base_inputs["input_ids"].to(device),
            attention_mask=base_inputs["attention_mask"].to(device),
            labels=None,
            output_hidden_states=False,
        )
        base_pred = int(base_out["logits"].argmax(dim=-1).item())

        for lab, layer_idx in zip(aligned_labels, aligned_indices):
            cache = {}
            def save_local(module, inputs, output):
                cache["tensor"] = output[0].detach() if isinstance(output, tuple) else output.detach()
                return output

            h1 = model.backbone.encoder.layer[layer_idx].register_forward_hook(save_local)
            _ = model(
                input_ids=local_inputs["input_ids"].to(device),
                attention_mask=local_inputs["attention_mask"].to(device),
                labels=None,
                output_hidden_states=False,
            )
            h1.remove()

            def patch_base(module, inputs, output):
                patch = cache["tensor"]
                if isinstance(output, tuple):
                    patch = patch.to(output[0].dtype)
                    return (patch,) + output[1:]
                patch = patch.to(output.dtype)
                return patch

            h2 = model.backbone.encoder.layer[layer_idx].register_forward_hook(patch_base)
            patched = model(
                input_ids=base_inputs["input_ids"].to(device),
                attention_mask=base_inputs["attention_mask"].to(device),
                labels=None,
                output_hidden_states=False,
            )
            h2.remove()
            patched_pred = int(patched["logits"].argmax(dim=-1).item())
            rows.append({
                "pair_id": pair_id,
                "base_id": base_id,
                "local_id": local_id,
                "aligned_depth": lab,
                "base_prediction": base_pred,
                "patched_prediction": patched_pred,
                "prediction_changed": int(base_pred != patched_pred),
            })
    return pd.DataFrame(rows)


def maybe_dynamic_quantise_linear_layers(model):
    """
    CPU-friendly dynamic quantisation over linear layers.
    """
    return torch.quantization.quantize_dynamic(copy.deepcopy(model).cpu(), {nn.Linear}, dtype=torch.qint8)


def summarise_late_layer_share(ablation_df: pd.DataFrame) -> float:
    """
    Estimate how much local-task degradation is concentrated in the latest third of layers.
    """
    work = ablation_df.loc[ablation_df["layer"] != "baseline"].copy()
    if work.empty:
        return 0.0
    baseline_local = float(ablation_df.loc[ablation_df["layer"] == "baseline", "local_accuracy"].iloc[0])
    work["local_drop"] = baseline_local - work["local_accuracy"].astype(float)
    n = len(work)
    cutoff = max(1, math.ceil(2 * n / 3))
    late = work.iloc[cutoff:]["local_drop"].sum()
    total = work["local_drop"].clip(lower=0).sum()
    return 0.0 if total <= 0 else float(late / total)


def summarise_early_mid_divergence(cka_df: pd.DataFrame) -> float:
    """
    Convert early/mid-layer CKA similarity into a divergence measure.
    Higher divergence means stronger representational change away from the reference model.
    """
    if cka_df.empty:
        return 0.0
    work = cka_df.copy()
    work["pct"] = work["aligned_depth"].str.rstrip("%").astype(int)
    early_mid = work.loc[work["pct"] <= 50]
    if early_mid.empty:
        early_mid = work
    mean_similarity = float(early_mid["linear_cka"].mean())
    return max(0.0, min(1.0, 1.0 - mean_similarity))


def _compute_target_false_localisation_summary(
    output_dir: Path,
    all_cka_df: pd.DataFrame,
    loaded_names: list[str],
    target_model: str,
    reference_models: list[str],
) -> dict:
    """
    Compute the false-localisation summary for a chosen target model.

    The summary combines:
    - late-layer concentration of local-task dependence
    - weak early/mid-layer divergence from reference models
    """
    if target_model not in loaded_names:
        raise RuntimeError(f"Target model '{target_model}' was not loaded.")

    target_ablation_path = output_dir / target_model / "layer_ablation.csv"
    if not target_ablation_path.exists():
        raise RuntimeError(f"Missing ablation results for target model: {target_ablation_path}")

    target_ablation = pd.read_csv(target_ablation_path)
    late_share = summarise_late_layer_share(target_ablation)

    reference_divergences = []

    for ref_model in reference_models:
        if ref_model not in loaded_names:
            continue

        cka_ref = all_cka_df.loc[
            (
                (all_cka_df["model_a"] == ref_model)
                & (all_cka_df["model_b"] == target_model)
            )
            |
            (
                (all_cka_df["model_a"] == target_model)
                & (all_cka_df["model_b"] == ref_model)
            )
        ]

        if not cka_ref.empty:
            reference_divergences.append(summarise_early_mid_divergence(cka_ref))

    if not reference_divergences:
        raise RuntimeError(
            f"No valid reference-model CKA comparisons were available for target model '{target_model}'."
        )

    early_mid_div = float(np.mean(reference_divergences))
    false_idx = compute_false_localisation_index(late_share, early_mid_div)

    summary = {
        "target_model": target_model,
        "reference_models_used": [m for m in reference_models if m in loaded_names],
        "late_layer_importance_share": late_share,
        "early_mid_divergence": early_mid_div,
        "false_localisation_index": false_idx,
    }
    return summary


def run_full_comparison(
    csv_path: Path,
    output_dir: Path,
    max_length: int = 128,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    seed: int = 13,
    device_name: str | None = None,
):
    """
    End-to-end runner tailored to:
    - xlm-roberta-base
    - Davlan/afro-xlmr-large
    - Davlan/afro-xlmr-small
    - dsfsi/afro-xlmr-comet (optional; skipped if unavailable)

    This version is designed to remain usable even when some candidate models
    are unavailable or non-standard on Hugging Face.
    """
    set_seed(seed)
    ensure_dir(output_dir)

    if device_name is None:
        if torch.backends.mps.is_available():
            device_name = "mps"
        elif torch.cuda.is_available():
            device_name = "cuda"
        else:
            device_name = "cpu"
    device = torch.device(device_name)

    df = validate_dataset_schema(pd.read_csv(csv_path))
    save_dataframe(df, output_dir / "validated_dataset.parquet")

    splits = split_dataframe(df)
    train_df = splits.get("train", pd.DataFrame(columns=df.columns))
    val_df = splits.get("validation", pd.DataFrame(columns=df.columns))
    test_df = splits.get("test", pd.DataFrame(columns=df.columns))

    local_ids = set(df.loc[df["is_local_task"] == 1, "id"].astype(str).tolist())
    num_labels = int(df["label"].nunique())

    specs = [
        ModelSpec("xlm_r_base", "xlm-roberta-base", num_labels),
        ModelSpec("afroxlmr_large", "Davlan/afro-xlmr-large", num_labels),
        ModelSpec("afroxlmr_small", "Davlan/afro-xlmr-small", num_labels),
        ModelSpec("afroxlmr_comet", "dsfsi/afro-xlmr-comet", num_labels),
    ]

    aligner = RelativeDepthAligner()
    test_reprs: dict[str, dict[str, np.ndarray]] = {}

    # Load only models that are actually available.
    loaded_specs: list[tuple[ModelSpec, object, object]] = []
    for spec in specs:
        try:
            model, tokenizer = load_model_and_tokenizer(spec, device)
            loaded_specs.append((spec, model, tokenizer))
        except Exception as e:
            print(f"Skipping {spec.hf_name} due to load failure: {type(e).__name__}: {e}")
            continue

    if len(loaded_specs) < 2:
        raise RuntimeError(
            "Fewer than two models could be loaded. "
            "At least two are required for cross-model comparisons."
        )

    # Run the per-model analysis.
    for spec, model, tokenizer in loaded_specs:
        model_dir = output_dir / spec.name
        ensure_dir(model_dir)

        train_ds = tokenise_dataframe(train_df, tokenizer, max_length)
        val_ds = tokenise_dataframe(val_df, tokenizer, max_length)
        test_ds = tokenise_dataframe(test_df, tokenizer, max_length)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_batch,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )

        if len(train_ds) > 0:
            train_classifier(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader if len(val_ds) > 0 else None,
                device=device,
                epochs=epochs,
                learning_rate=learning_rate,
                output_dir=model_dir,
            )

        metrics = evaluate_classifier(model, test_loader, device)
        save_dataframe(metrics["predictions"], model_dir / "test_predictions.csv")
        save_json(
            {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            },
            model_dir / "test_metrics.json",
        )

        reprs, meta_df = extract_aligned_representations(
            model=model,
            data_loader=test_loader,
            device=device,
            aligner=aligner,
        )
        test_reprs[spec.name] = reprs
        save_dataframe(meta_df, model_dir / "representation_metadata.csv")

        for depth_label, mat in reprs.items():
            np.save(model_dir / f"repr_{depth_label.replace('%', 'pct')}.npy", mat)

        ablation_df = evaluate_with_layer_zero_ablation(
            model=model,
            data_loader=test_loader,
            device=device,
            local_ids=local_ids,
        )
        save_dataframe(ablation_df, model_dir / "layer_ablation.csv")

        if len(test_df.loc[test_df["pair_id"].fillna("") != ""]) > 0:
            patch_df = paired_activation_patching(
                model=model,
                tokenizer=tokenizer,
                pairs_df=test_df,
                device=device,
                aligner=aligner,
                max_length=max_length,
            )
            save_dataframe(patch_df, model_dir / "activation_patching.csv")

        # Compression realism check: run on models that are small enough to make sense here.
        if spec.name in {"afroxlmr_small", "afroxlmr_comet"}:
            q_model = maybe_dynamic_quantise_linear_layers(model)
            q_metrics = evaluate_classifier(q_model, test_loader, torch.device("cpu"))
            save_dataframe(q_metrics["predictions"], model_dir / "quantised_test_predictions.csv")
            save_json(
                {
                    "accuracy": q_metrics["accuracy"],
                    "macro_f1": q_metrics["macro_f1"],
                },
                model_dir / "quantised_test_metrics.json",
            )

    loaded_names = [spec.name for spec, _, _ in loaded_specs]
    pairings = list(combinations(loaded_names, 2))

    cos_all = []
    cka_all = []

    for a, b in pairings:
        if a not in test_reprs or b not in test_reprs:
            continue
        cos_df, cka_df = compute_similarity_tables(test_reprs[a], test_reprs[b], a, b)
        cos_all.append(cos_df)
        cka_all.append(cka_df)

    if not cos_all or not cka_all:
        raise RuntimeError("No cross-model similarity tables could be computed.")

    all_cos_df = pd.concat(cos_all, ignore_index=True)
    all_cka_df = pd.concat(cka_all, ignore_index=True)

    save_dataframe(all_cos_df, output_dir / "cross_model_cosine_similarity.csv")
    save_dataframe(all_cka_df, output_dir / "cross_model_linear_cka.csv")

    # Compute the false-localisation summary for AfroXLMR-Small.
    summary = _compute_target_false_localisation_summary(
        output_dir=output_dir,
        all_cka_df=all_cka_df,
        loaded_names=loaded_names,
        target_model="afroxlmr_small",
        reference_models=["xlm_r_base", "afroxlmr_large"],
    )
    save_json(summary, output_dir / "false_localisation_summary.json")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run the AfroXLMR/XLM-R comparison pipeline.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_full_comparison(
        csv_path=args.csv,
        output_dir=args.output_dir,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
