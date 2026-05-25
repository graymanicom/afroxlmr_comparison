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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels),
        )

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


def _debiased_dot_product_similarity(
    dot_product_similarity: float,
    sum_squared_rows_x: np.ndarray,
    sum_squared_rows_y: np.ndarray,
    squared_norm_x: float,
    squared_norm_y: float,
    n: int,
) -> float:
    """
    Unbiased estimator used for debiased linear CKA.

    This follows the debiasing correction discussed by Kornblith et al. (2019)
    and used in Murphy, Zylberberg and Fyshe's biased CKA correction paper.
    """
    return float(
        dot_product_similarity
        - (n / (n - 2.0)) * np.dot(sum_squared_rows_x, sum_squared_rows_y)
        + (squared_norm_x * squared_norm_y) / ((n - 1.0) * (n - 2.0))
    )


def linear_cka(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """
    Debiased linear CKA for comparing hidden representations.

    x and y must have the same number of examples/rows, but may have
    different feature dimensions.

    Unlike the standard biased linear CKA estimator, this version corrects
    for inflated similarity in low-sample, high-dimensional settings.
    Debiased CKA can be negative.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")

    n = x.shape[0]
    if n <= 2:
        raise ValueError("debiased linear CKA requires more than 2 examples")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Centre features across examples.
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    # Squared row norms are the diagonals of the linear Gram matrices.
    sum_squared_rows_x = np.sum(x ** 2, axis=1)
    sum_squared_rows_y = np.sum(y ** 2, axis=1)

    squared_norm_x = float(np.sum(sum_squared_rows_x))
    squared_norm_y = float(np.sum(sum_squared_rows_y))

    # Frobenius squared dot-product similarities.
    x_y_similarity = float(np.sum((x.T @ y) ** 2))
    x_x_similarity = float(np.sum((x.T @ x) ** 2))
    y_y_similarity = float(np.sum((y.T @ y) ** 2))

    # Debias numerator and normalisation terms.
    debiased_xy = _debiased_dot_product_similarity(
        x_y_similarity,
        sum_squared_rows_x,
        sum_squared_rows_y,
        squared_norm_x,
        squared_norm_y,
        n,
    )

    debiased_xx = _debiased_dot_product_similarity(
        x_x_similarity,
        sum_squared_rows_x,
        sum_squared_rows_x,
        squared_norm_x,
        squared_norm_x,
        n,
    )

    debiased_yy = _debiased_dot_product_similarity(
        y_y_similarity,
        sum_squared_rows_y,
        sum_squared_rows_y,
        squared_norm_y,
        squared_norm_y,
        n,
    )

    denom = np.sqrt(max(debiased_xx, 0.0) * max(debiased_yy, 0.0)) + eps
    return float(debiased_xy / denom)


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
        
def safe_accuracy(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    return float((df["label"] == df["prediction"]).mean())

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

    # freeze the backbone and only train the classifier head, to keep the comparison focused on representational differences.
    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {num_trainable}/{num_total}")

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
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=0.01,
    )
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
def evaluate_classifier(model, data_loader, device, report_name: str | None = None):
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

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    pred_df = pd.DataFrame(rows)

    if report_name is not None:
        print(f"\n=== Evaluation report: {report_name} ===")
        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, zero_division=0))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predictions": pred_df,
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



@torch.no_grad()
def extract_layerwise_probe_representations(model, data_loader, device):
    """
    Extract pooled sequence representations from every encoder layer.

    The hidden state list includes embeddings at index 0, then layer outputs.
    Therefore encoder layer k corresponds to hidden_states[k + 1].
    """
    model.eval()
    num_layers = int(getattr(model.backbone.config, "num_hidden_layers"))
    buffers = {str(idx): [] for idx in range(num_layers)}
    ids = []

    for batch in tqdm(data_loader, desc="extract layerwise probe representations", leave=False):
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=None,
            output_hidden_states=True,
        )
        hidden_states = out["hidden_states"]
        ids.extend([str(ex_id) for ex_id in batch["example_ids"]])
        for idx in range(num_layers):
            hs = hidden_states[idx + 1]
            pooled = masked_mean_pool(hs, batch["attention_mask"].to(device)).detach().cpu().numpy()
            buffers[str(idx)].append(pooled)

    stacked = {layer: np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1)) for layer, chunks in buffers.items()}
    return stacked, ids


def _first_non_empty(values: pd.Series):
    values = values.dropna().astype(str)
    values = values.loc[values.str.strip() != ""]
    if values.empty:
        return np.nan
    return values.value_counts().index[0]


def build_probe_label_frame(df: pd.DataFrame, metadata_csv: Path | None = None) -> pd.DataFrame:
    """
    Build labels for layerwise probes.

    Compatibility labels use the dataset's valid/invalid label.
    Semantic-type labels are taken from existing metadata where available. If
    metadata_csv is supplied, semantic types are propagated by pair_id.
    """
    label_df = df[["id", "label", "pair_id", "pair_role"]].copy()
    label_df["id"] = label_df["id"].astype(str)
    label_df["pair_id"] = label_df["pair_id"].astype(str)
    label_df["compatibility_label"] = label_df["label"].astype(int)
    label_df["semantic_type_label"] = np.nan

    if "original_semantic_type" in df.columns:
        if "replacement_semantic_type" in df.columns:
            semantic_values = np.where(
                df["pair_role"].astype(str).eq("local") & df["replacement_semantic_type"].notna(),
                df["replacement_semantic_type"].astype(str),
                df["original_semantic_type"].astype(str),
            )
            label_df["semantic_type_label"] = semantic_values
        else:
            label_df["semantic_type_label"] = df["original_semantic_type"]

    if metadata_csv is not None and metadata_csv.exists():
        metadata_df = pd.read_csv(metadata_csv)
        if {"pair_id", "original_semantic_type"}.issubset(metadata_df.columns):
            pair_semantic = (
                metadata_df.assign(pair_id=metadata_df["pair_id"].astype(str))
                .groupby("pair_id")["original_semantic_type"]
                .agg(_first_non_empty)
                .rename("metadata_original_semantic_type")
                .reset_index()
            )
            label_df = label_df.merge(pair_semantic, on="pair_id", how="left")
            label_df["semantic_type_label"] = label_df["semantic_type_label"].where(
                label_df["semantic_type_label"].notna(),
                label_df["metadata_original_semantic_type"],
            )
            label_df = label_df.drop(columns=["metadata_original_semantic_type"])

    label_df["semantic_type_label"] = label_df["semantic_type_label"].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return label_df


def _labels_for_ids(ids: Sequence[str], label_map: Mapping[str, Any]) -> np.ndarray:
    return np.array([label_map.get(str(ex_id), np.nan) for ex_id in ids], dtype=object)


def _fit_layerwise_probe(
    train_reprs: Mapping[str, np.ndarray],
    train_ids: Sequence[str],
    val_reprs: Mapping[str, np.ndarray],
    val_ids: Sequence[str],
    test_reprs: Mapping[str, np.ndarray],
    test_ids: Sequence[str],
    label_df: pd.DataFrame,
    target_column: str,
    probe_name: str,
) -> pd.DataFrame:
    label_map = dict(zip(label_df["id"].astype(str), label_df[target_column]))
    y_train_raw = _labels_for_ids(train_ids, label_map)

    train_mask = pd.notna(y_train_raw)
    y_train_series = pd.Series(y_train_raw[train_mask]).astype(str)
    class_counts = y_train_series.value_counts()
    usable_classes = set(class_counts.loc[class_counts >= 2].index.tolist())

    if len(usable_classes) < 2:
        return pd.DataFrame()

    train_mask = train_mask & np.array([str(y) in usable_classes for y in y_train_raw])
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(pd.Series(y_train_raw[train_mask]).astype(str))

    y_val_raw = _labels_for_ids(val_ids, label_map)
    val_mask = pd.notna(y_val_raw) & np.array([str(y) in set(encoder.classes_) for y in y_val_raw])
    y_test_raw = _labels_for_ids(test_ids, label_map)
    test_mask = pd.notna(y_test_raw) & np.array([str(y) in set(encoder.classes_) for y in y_test_raw])

    rows = []
    for layer in sorted(train_reprs.keys(), key=lambda x: int(x)):
        x_train = train_reprs[layer][train_mask]
        if len(x_train) == 0:
            continue

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=13),
        )
        clf.fit(x_train, y_train)

        record = {
            "probe": probe_name,
            "layer": int(layer),
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_classes": int(len(encoder.classes_)),
            "classes": "|".join(encoder.classes_.tolist()),
        }

        if val_mask.sum() > 0:
            y_val = encoder.transform(pd.Series(y_val_raw[val_mask]).astype(str))
            pred_val = clf.predict(val_reprs[layer][val_mask])
            record["val_accuracy"] = float(accuracy_score(y_val, pred_val))
            record["val_macro_f1"] = float(f1_score(y_val, pred_val, average="macro", zero_division=0))
        else:
            record["val_accuracy"] = np.nan
            record["val_macro_f1"] = np.nan

        if test_mask.sum() > 0:
            y_test = encoder.transform(pd.Series(y_test_raw[test_mask]).astype(str))
            pred_test = clf.predict(test_reprs[layer][test_mask])
            record["test_accuracy"] = float(accuracy_score(y_test, pred_test))
            record["test_macro_f1"] = float(f1_score(y_test, pred_test, average="macro", zero_division=0))
        else:
            record["test_accuracy"] = np.nan
            record["test_macro_f1"] = np.nan

        rows.append(record)

    return pd.DataFrame(rows)


def run_layerwise_probes(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    probe_label_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    train_reprs, train_ids = extract_layerwise_probe_representations(model, train_loader, device)
    val_reprs, val_ids = extract_layerwise_probe_representations(model, val_loader, device)
    test_reprs, test_ids = extract_layerwise_probe_representations(model, test_loader, device)

    compatibility_df = _fit_layerwise_probe(
        train_reprs=train_reprs,
        train_ids=train_ids,
        val_reprs=val_reprs,
        val_ids=val_ids,
        test_reprs=test_reprs,
        test_ids=test_ids,
        label_df=probe_label_df,
        target_column="compatibility_label",
        probe_name="compatibility",
    )

    semantic_label_df = probe_label_df.loc[probe_label_df["pair_role"].astype(str) == "base"].copy()
    semantic_df = _fit_layerwise_probe(
        train_reprs=train_reprs,
        train_ids=train_ids,
        val_reprs=val_reprs,
        val_ids=val_ids,
        test_reprs=test_reprs,
        test_ids=test_ids,
        label_df=semantic_label_df,
        target_column="semantic_type_label",
        probe_name="semantic_type",
    )

    return {
        "compatibility": compatibility_df,
        "semantic_type": semantic_df,
    }


def compute_similarity_tables(repr_a, repr_b, model_a: str, model_b: str):
    cosine_rows, cka_rows = [], []

    for k in [k for k in repr_a.keys() if k in repr_b]:
        x, y = repr_a[k], repr_b[k]

        # Cosine similarity requires equal hidden dimensions, so this remains
        # a truncated diagnostic when model sizes differ.
        common_dim = min(x.shape[1], y.shape[1])
        x_cos, y_cos = x[:, :common_dim], y[:, :common_dim]

        cosine_rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "aligned_depth": k,
            "cosine_mean_similarity": pooled_cosine_similarity(x_cos, y_cos),
        })

        # Linear CKA can compare matrices with different feature dimensions,
        # as long as they have the same examples/rows.
        cka_rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "aligned_depth": k,
            "linear_cka": linear_cka(x, y),
        })

    return pd.DataFrame(cosine_rows), pd.DataFrame(cka_rows)

def _alpha_layer_output_hook(module, inputs, output):
    """use to diminish a layer's contribution to the final prediction by multiple alpha.
    keeps shape of signal/information structure but reduces magnitude. 
    Basically destroys transformers if alpha=0"""
    alpha = 0.3
    if isinstance(output, tuple):
        scaled = output[0] * alpha
        return (scaled,) + output[1:]
    return output * alpha


def _noise_layer_output_hook(module, inputs, output):
    """Replace structured layer output with same-scale noise."""
    if isinstance(output, tuple):
        h = output[0]
        scale = h.std().clamp(min=1e-6)
        noise = torch.randn_like(h) * scale
        return (noise,) + output[1:]
    h = output
    scale = h.std().clamp(min=1e-6)
    noise = torch.randn_like(h) * scale
    return noise

@torch.no_grad()
def evaluate_with_layer_noise_ablation( model, data_loader, device, local_ids: set | None = None,) -> pd.DataFrame:
    """
    Coarse causal intervention:
    replace one encoder layer's output with same-scale noise and measure
    the resulting change in classifier performance.

    This tests whether the classifier depends on structured information
    in a layer, rather than merely on activation magnitude.

    Notes
    -----
    In the current institutional-validity task, all examples may be
    localisation-relevant. In that case, `local_ids` can be None or can
    contain all example ids. The function therefore reports optional
    local/non-local accuracies without assuming both subsets exist.
    """
    layers = model.backbone.encoder.layer
    local_ids = local_ids or set()

    def subset_acc(frame: pd.DataFrame) -> float:
        if frame.empty:
            return float("nan")
        return float((frame["label"] == frame["prediction"]).mean())

    def build_record(layer_name, metrics):
        pred_df = metrics["predictions"].copy()

        if local_ids:
            pred_df["is_local_task"] = pred_df["id"].isin(local_ids).astype(int)
        else:
            pred_df["is_local_task"] = 1

        local_df = pred_df.loc[pred_df["is_local_task"] == 1]
        global_df = pred_df.loc[pred_df["is_local_task"] == 0]

        return {
            "layer": layer_name,
            "overall_accuracy": metrics["accuracy"],
            "overall_macro_f1": metrics["macro_f1"],
            "local_accuracy": subset_acc(local_df),
            "global_accuracy": subset_acc(global_df),
            "n_examples": int(len(pred_df)),
            "n_local_examples": int(len(local_df)),
            "n_global_examples": int(len(global_df)),
        }

    records = []

    baseline = evaluate_classifier(model, data_loader, device)
    records.append(build_record("baseline", baseline))

    for idx, layer in enumerate(layers):
        handle = layer.register_forward_hook(_noise_layer_output_hook)
        try:
            ablated = evaluate_classifier(model, data_loader, device)
        finally:
            handle.remove()

        records.append(build_record(idx, ablated))

    return pd.DataFrame(records)


@torch.no_grad()
def within_model_paired_activation_patching(model, tokenizer, pairs_df: pd.DataFrame, device, aligner: RelativeDepthAligner, max_length: int):
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

    for pair_id, valid_id, invalid_id in tqdm(build_pair_mapping(pairs_df), desc="patch pairs", leave=False):
        valid_row, invalid_row = id_to_row[valid_id], id_to_row[invalid_id]
        valid_inputs = tokenizer([str(valid_row["text"])], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        invalid_inputs = tokenizer([str(invalid_row["text"])], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        valid_out = model(
            input_ids=valid_inputs["input_ids"].to(device),
            attention_mask=valid_inputs["attention_mask"].to(device),
            labels=None,
            output_hidden_states=False,
        )
        valid_pred = int(valid_out["logits"].argmax(dim=-1).item())

        for lab, layer_idx in zip(aligned_labels, aligned_indices):
            cache = {}
            def save_invalid_activation(module, inputs, output):
                cache["tensor"] = output[0].detach() if isinstance(output, tuple) else output.detach()
                return output

            h1 = model.backbone.encoder.layer[layer_idx].register_forward_hook(save_invalid_activation)
            _ = model(
                input_ids=invalid_inputs["input_ids"].to(device),
                attention_mask=invalid_inputs["attention_mask"].to(device),
                labels=None,
                output_hidden_states=False,
            )
            h1.remove()

            def patch_valid_activation(module, inputs, output):
                patch = cache["tensor"]
                if isinstance(output, tuple):
                    patch = patch.to(output[0].dtype)
                    return (patch,) + output[1:]
                patch = patch.to(output.dtype)
                return patch

            h2 = model.backbone.encoder.layer[layer_idx].register_forward_hook(patch_valid_activation)
            patched = model(
                input_ids=valid_inputs["input_ids"].to(device),
                attention_mask=valid_inputs["attention_mask"].to(device),
                labels=None,
                output_hidden_states=False,
            )
            h2.remove()
            patched_valid_pred = int(patched["logits"].argmax(dim=-1).item())
            rows.append({
                "pair_id": pair_id,
                "valid_id": valid_id,
                "invalid_id": invalid_id,
                "aligned_depth": lab,
                "valid_prediction": valid_pred,
                "patched_valid_prediction": patched_valid_pred,
                "prediction_changed": int(valid_pred != patched_valid_pred),
            })
    return pd.DataFrame(rows)


@torch.no_grad()
def cross_model_activation_patching(
    source_model,
    source_tokenizer,
    source_name: str,
    target_model,
    target_tokenizer,
    target_name: str,
    examples_df: pd.DataFrame,
    device,
    aligner: RelativeDepthAligner,
    max_length: int,
) -> pd.DataFrame:
    source_model.eval()
    target_model.eval()
    source_layers = source_model.backbone.encoder.layer
    target_layers = target_model.backbone.encoder.layer
    source_indices = aligner.layer_indices(int(getattr(source_model.backbone.config, "num_hidden_layers")))
    target_indices = aligner.layer_indices(int(getattr(target_model.backbone.config, "num_hidden_layers")))
    aligned_labels = aligner.labelled_positions()
    rows = []

    source_hidden = int(getattr(source_model.backbone.config, "hidden_size"))
    target_hidden = int(getattr(target_model.backbone.config, "hidden_size"))
    if source_hidden != target_hidden:
        return pd.DataFrame([{
            "source_model": source_name,
            "target_model": target_name,
            "aligned_depth": "skipped",
            "skip_reason": f"hidden_size_mismatch:{source_hidden}!={target_hidden}",
        }])

    for _, example_row in tqdm(examples_df.iterrows(), total=len(examples_df), desc=f"cross-patch {source_name} -> {target_name}", leave=False):
        text_value = str(example_row["text"])
        source_inputs = source_tokenizer([text_value], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        target_inputs = target_tokenizer([text_value], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")

        target_out = target_model(
            input_ids=target_inputs["input_ids"].to(device),
            attention_mask=target_inputs["attention_mask"].to(device),
            labels=None,
            output_hidden_states=False,
        )
        target_pred = int(target_out["logits"].argmax(dim=-1).item())

        for lab, source_idx, target_idx in zip(aligned_labels, source_indices, target_indices):
            cache = {}

            def save_source_activation(module, inputs, output):
                cache["tensor"] = output[0].detach() if isinstance(output, tuple) else output.detach()
                return output

            h1 = source_layers[source_idx].register_forward_hook(save_source_activation)
            _ = source_model(
                input_ids=source_inputs["input_ids"].to(device),
                attention_mask=source_inputs["attention_mask"].to(device),
                labels=None,
                output_hidden_states=False,
            )
            h1.remove()

            def patch_target_activation(module, inputs, output):
                patch = cache["tensor"]
                if isinstance(output, tuple):
                    if patch.shape != output[0].shape:
                        raise ValueError(f"Patch shape {tuple(patch.shape)} does not match target shape {tuple(output[0].shape)}")
                    patch = patch.to(output[0].dtype)
                    return (patch,) + output[1:]
                if patch.shape != output.shape:
                    raise ValueError(f"Patch shape {tuple(patch.shape)} does not match target shape {tuple(output.shape)}")
                patch = patch.to(output.dtype)
                return patch

            h2 = target_layers[target_idx].register_forward_hook(patch_target_activation)
            try:
                patched = target_model(
                    input_ids=target_inputs["input_ids"].to(device),
                    attention_mask=target_inputs["attention_mask"].to(device),
                    labels=None,
                    output_hidden_states=False,
                )
                patched_pred = int(patched["logits"].argmax(dim=-1).item())
                rows.append({
                    "source_model": source_name,
                    "target_model": target_name,
                    "id": str(example_row["id"]),
                    "label": int(example_row["label"]),
                    "aligned_depth": lab,
                    "source_layer": int(source_idx),
                    "target_layer": int(target_idx),
                    "target_prediction": target_pred,
                    "patched_prediction": patched_pred,
                    "prediction_changed": int(target_pred != patched_pred),
                    "skip_reason": "",
                })
            finally:
                h2.remove()

    return pd.DataFrame(rows)


def maybe_dynamic_quantise_linear_layers(model):
    """
    CPU-friendly dynamic quantisation over linear layers.

    On Apple Silicon / ARM CPUs, PyTorch quantised ops should use QNNPACK.
    On x86 CPUs, use x86/fbgemm when available.

    If no quantisation engine is available, return None and let the caller skip
    the quantised comparison gracefully.
    """
    model_cpu = copy.deepcopy(model).cpu().eval()

    supported = torch.backends.quantized.supported_engines
    if not supported:
        print("Skipping quantisation: no quantised backends are available.")
        return None

    if "qnnpack" in supported:
        torch.backends.quantized.engine = "qnnpack"
    elif "x86" in supported:
        torch.backends.quantized.engine = "x86"
    elif "fbgemm" in supported:
        torch.backends.quantized.engine = "fbgemm"
    else:
        print(f"Skipping quantisation: unsupported quantised engines {supported}")
        return None

    try:
        q_model = torch.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear},
            dtype=torch.qint8,
        )
        return q_model
    except Exception as e:
        print(f"Skipping quantisation due to failure: {type(e).__name__}: {e}")
        return None


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

    target_ablation_path = output_dir / target_model / "layer_noise_ablation.csv"
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
    metadata_csv: Path | None = None,
):
    """
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
    probe_label_df = build_probe_label_frame(df, metadata_csv=metadata_csv)
    save_dataframe(probe_label_df, output_dir / "probe_labels.csv")

    splits = split_dataframe(df)
    train_df = splits.get("train", pd.DataFrame(columns=df.columns))
    val_df = splits.get("validation", pd.DataFrame(columns=df.columns))
    test_df = splits.get("test", pd.DataFrame(columns=df.columns))

    local_ids = set(df.loc[df["is_local_task"] == 1, "id"].astype(str).tolist())
    num_labels = int(df["label"].nunique())

    specs = [
        # Primary lineage. 
        # xlm_r_large is the main reference. 
        # afroxlmr_large is the main synthesis model using Multilingual Adaptive Fine-tuning (MAFT).
        # afroxlmr_comet is distilled from afroxlmr_large 
        ModelSpec("xlm_r_large", "xlm-roberta-large", num_labels),
        ModelSpec("afroxlmr_large", "Davlan/afro-xlmr-large", num_labels),
        ModelSpec("afroxlmr_comet", "local_models/afro-xlmr-comet", num_labels),

        # Secondary AfroXLMR base/small lineage
        ModelSpec("xlm_r_base", "xlm-roberta-base", num_labels),
        ModelSpec("afroxlmr_base", "Davlan/afro-xlmr-base", num_labels),
        ModelSpec("afroxlmr_small", "Davlan/afro-xlmr-small", num_labels),

        # Regional from-scratch contrast
        ModelSpec("zabantu_xlmr", "dsfsi/zabantu-xlm-roberta", num_labels),
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

        metrics = evaluate_classifier(model, test_loader, device, report_name=spec.name)
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

        ablation_df = evaluate_with_layer_noise_ablation(
            model=model,
            data_loader=test_loader,
            device=device,
            local_ids=local_ids,
        )
        save_dataframe(ablation_df, model_dir / "layer_noise_ablation.csv")

        probe_dfs = run_layerwise_probes(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            probe_label_df=probe_label_df,
        )
        if not probe_dfs["compatibility"].empty:
            save_dataframe(probe_dfs["compatibility"], model_dir / "layerwise_compatibility_probe.csv")
        if not probe_dfs["semantic_type"].empty:
            save_dataframe(probe_dfs["semantic_type"], model_dir / "layerwise_semantic_type_probe.csv")

        if len(test_df.loc[test_df["pair_id"].fillna("") != ""]) > 0:
            patch_df = within_model_paired_activation_patching(
                model=model,
                tokenizer=tokenizer,
                pairs_df=test_df,
                device=device,
                aligner=aligner,
                max_length=max_length,
            )
            save_dataframe(patch_df, model_dir / "within_model_activation_patching.csv")

        # Compression realism check: run on models that are small enough to make sense here.
        if spec.name in {"afroxlmr_small", "afroxlmr_comet"}:
            q_model = maybe_dynamic_quantise_linear_layers(model)
            if q_model is not None:
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

    loaded_by_name = {spec.name: (model, tokenizer) for spec, model, tokenizer in loaded_specs}
    cross_patch_specs = [
        ("xlm_r_large", "afroxlmr_large"),
        ("xlm_r_base", "afroxlmr_small"),
    ]
    cross_patch_dfs = []
    for source_name, target_name in cross_patch_specs:
        if source_name not in loaded_by_name or target_name not in loaded_by_name:
            continue
        source_model, source_tokenizer = loaded_by_name[source_name]
        target_model, target_tokenizer = loaded_by_name[target_name]
        cross_patch_dfs.append(cross_model_activation_patching(
            source_model=source_model,
            source_tokenizer=source_tokenizer,
            source_name=source_name,
            target_model=target_model,
            target_tokenizer=target_tokenizer,
            target_name=target_name,
            examples_df=test_df,
            device=device,
            aligner=aligner,
            max_length=max_length,
        ))
    if cross_patch_dfs:
        save_dataframe(pd.concat(cross_patch_dfs, ignore_index=True), output_dir / "cross_model_activation_patching.csv")

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
    parser.add_argument("--metadata-csv", type=Path, default=None)
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
        metadata_csv=args.metadata_csv,
    )


if __name__ == "__main__":
    main()
