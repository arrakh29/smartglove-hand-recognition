import argparse
import json
import pickle
import random
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


ANGLE_COLUMNS = [
    "Thumb_MCP", "Thumb_IP",
    "Index_MCP", "Index_PIP", "Index_DIP",
    "Middle_MCP", "Middle_PIP", "Middle_DIP",
    "Ring_MCP", "Ring_PIP", "Ring_DIP",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP",
]

STAT_SUFFIXES = ["_mean", "_std", "_min", "_max"]

METADATA_COLUMNS = [
    "file_name", "label", "subject", "sample_id",
    "rep_id", "window_id", "group_id",
    "source_system", "hand_side", "feature_mode",
    "num_frames_total_file", "file_duration_s", "num_frames_window",
    "window_start_time_s", "window_end_time_s", "window_duration_s",
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_hand_adjacency() -> np.ndarray:
    n = len(ANGLE_COLUMNS)
    A = np.zeros((n, n), dtype=np.float32)

    edges = [
        (0, 1),           # Thumb_MCP - Thumb_IP
        (2, 3), (3, 4),   # Index
        (5, 6), (6, 7),   # Middle
        (8, 9), (9, 10),  # Ring
        (11, 12), (12, 13),  # Pinky
        (0, 2),           # Thumb to Index root
        (2, 5),           # Index to Middle root
        (5, 8),           # Middle to Ring root
        (8, 11),          # Ring to Pinky root
    ]

    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0

    for i in range(n):
        A[i, i] = 1.0

    deg = np.sum(A, axis=1)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-8))
    A_norm = deg_inv_sqrt @ A @ deg_inv_sqrt
    return A_norm.astype(np.float32)


def discover_feature_columns(df: pd.DataFrame) -> List[str]:
    feature_cols = []
    for c in df.columns:
        if c in METADATA_COLUMNS:
            continue
        if any(c.endswith(s) for s in STAT_SUFFIXES):
            feature_cols.append(c)

    if not feature_cols:
        raise ValueError("Tidak ditemukan kolom fitur statistik (_mean/_std/_min/_max).")

    return feature_cols


def discover_base_feature_names(feature_columns: List[str]) -> List[str]:
    base_names = set()
    for c in feature_columns:
        found = False
        for s in STAT_SUFFIXES:
            if c.endswith(s):
                base_names.add(c[:-len(s)])
                found = True
                break
        if not found:
            raise ValueError(f"Kolom fitur tidak valid: {c}")
    return sorted(base_names)


def validate_feature_layout(base_feature_names: List[str], feature_columns: List[str]):
    expected = []
    for base in base_feature_names:
        for s in STAT_SUFFIXES:
            expected.append(f"{base}{s}")

    missing = [c for c in expected if c not in feature_columns]
    if missing:
        raise ValueError(f"Kolom statistik tidak lengkap. Missing: {missing[:20]}")


def ordered_stat_columns(base_names: List[str]) -> List[str]:
    cols = []
    for base in base_names:
        for s in STAT_SUFFIXES:
            cols.append(f"{base}{s}")
    return cols


def build_angle_node_features_from_row(row: np.ndarray, num_angle_nodes: int, num_stats: int) -> np.ndarray:
    return row.reshape(num_angle_nodes, num_stats).astype(np.float32)


class HybridPoseDataset(Dataset):
    def __init__(
        self,
        X_angle: np.ndarray,
        X_extra: np.ndarray,
        y: np.ndarray,
        num_angle_nodes: int,
        num_stats_per_angle: int,
    ):
        self.X_angle = X_angle.astype(np.float32)
        self.X_extra = X_extra.astype(np.float32) if X_extra.size > 0 else np.zeros((len(X_angle), 0), dtype=np.float32)
        self.y = y.astype(np.int64)
        self.num_angle_nodes = num_angle_nodes
        self.num_stats_per_angle = num_stats_per_angle

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        angle_x = build_angle_node_features_from_row(
            self.X_angle[idx],
            num_angle_nodes=self.num_angle_nodes,
            num_stats=self.num_stats_per_angle,
        )
        extra_x = self.X_extra[idx]
        y = self.y[idx]
        return (
            torch.tensor(angle_x, dtype=torch.float32),
            torch.tensor(extra_x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = self.linear(x)
        return x


class HybridGCNMLPClassifier(nn.Module):
    def __init__(
        self,
        angle_in_features: int,
        extra_in_features: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.5,
    ):
        super().__init__()

        # GCN branch for 14 angle joints
        self.gcn1 = GCNLayer(angle_in_features, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn3 = GCNLayer(hidden_dim, hidden_dim)

        # MLP branch for extra/global features
        self.has_extra = extra_in_features > 0
        if self.has_extra:
            mlp_hidden = max(64, hidden_dim)
            self.extra_mlp = nn.Sequential(
                nn.Linear(extra_in_features, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, hidden_dim),
                nn.ReLU(),
            )
        else:
            self.extra_mlp = None

        fusion_in = hidden_dim + (hidden_dim if self.has_extra else 0)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, angle_x, extra_x, adj):
        # GCN branch
        xg = self.gcn1(angle_x, adj)
        xg = F.relu(xg)
        xg = self.dropout(xg)

        xg = self.gcn2(xg, adj)
        xg = F.relu(xg)
        xg = self.dropout(xg)

        xg = self.gcn3(xg, adj)
        xg = F.relu(xg)
        xg = xg.mean(dim=1)
        xg = self.dropout(xg)

        # MLP branch
        if self.has_extra:
            xm = self.extra_mlp(extra_x)
            x = torch.cat([xg, xm], dim=1)
        else:
            x = xg

        logits = self.classifier(x)
        return logits


def train_one_epoch(model, loader, optimizer, criterion, device, adj):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for angle_x, extra_x, yb in loader:
        angle_x = angle_x.to(device)
        extra_x = extra_x.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(angle_x, extra_x, adj)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * angle_x.size(0)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(yb.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, adj):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for angle_x, extra_x, yb in loader:
        angle_x = angle_x.to(device)
        extra_x = extra_x.to(device)
        yb = yb.to(device)

        logits = model(angle_x, extra_x, adj)
        loss = criterion(logits, yb)

        total_loss += loss.item() * angle_x.size(0)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(yb.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, acc, np.array(all_targets), np.array(all_preds)


def save_confusion_matrix(cm: np.ndarray, class_names, out_png: Path, out_csv: Path):
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(out_csv)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def grouped_split_by_group_id(df: pd.DataFrame, seed: int, test_size: float, val_size: float):
    rng = random.Random(seed)
    group_df = df[["group_id", "label"]].drop_duplicates().reset_index(drop=True)

    train_group_ids = []
    val_group_ids = []
    test_group_ids = []

    for label, sub in group_df.groupby("label"):
        groups = sub["group_id"].tolist()
        rng.shuffle(groups)

        n = len(groups)
        if n < 3:
            raise ValueError(
                f"Label '{label}' punya group terlalu sedikit ({n}). "
                f"Minimal 3 group supaya train/val/test masing-masing kebagian."
            )

        n_test = max(1, round(n * test_size))
        n_val = max(1, round(n * val_size))

        while n_test + n_val >= n:
            if n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break

        test_groups = groups[:n_test]
        val_groups = groups[n_test:n_test + n_val]
        train_groups = groups[n_test + n_val:]

        train_group_ids.extend(train_groups)
        val_group_ids.extend(val_groups)
        test_group_ids.extend(test_groups)

    train_df = df[df["group_id"].isin(train_group_ids)].copy()
    val_df = df[df["group_id"].isin(val_group_ids)].copy()
    test_df = df[df["group_id"].isin(test_group_ids)].copy()

    return train_df, val_df, test_df


def parse_args():
    p = argparse.ArgumentParser(description="Train Hybrid GCN + MLP for angle + discriminative features")
    p.add_argument("--input_csv", type=str, required=True, help="Path ke CSV fitur hasil ekstraksi")
    p.add_argument("--output_dir", type=str, required=True, help="Folder output training")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--val_size", type=float, default=0.15)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = out_dir / "models"
    metric_dir = out_dir / "metrics"
    figure_dir = out_dir / "figures"
    pred_dir = out_dir / "predictions"

    for d in [model_dir, metric_dir, figure_dir, pred_dir]:
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    required_meta = ["file_name", "label", "subject", "sample_id", "rep_id", "window_id", "group_id"]
    missing_meta = [c for c in required_meta if c not in df.columns]
    if missing_meta:
        raise ValueError(f"Kolom metadata hilang: {missing_meta}")

    feature_columns = discover_feature_columns(df)
    base_feature_names = discover_base_feature_names(feature_columns)
    validate_feature_layout(base_feature_names, feature_columns)

    angle_base_names = [c for c in ANGLE_COLUMNS if c in base_feature_names]
    if len(angle_base_names) != len(ANGLE_COLUMNS):
        missing_angles = [c for c in ANGLE_COLUMNS if c not in angle_base_names]
        raise ValueError(f"Kolom angle dasar tidak lengkap. Missing: {missing_angles}")

    extra_base_names = [c for c in base_feature_names if c not in ANGLE_COLUMNS]

    angle_feature_columns = ordered_stat_columns(angle_base_names)
    extra_feature_columns = ordered_stat_columns(extra_base_names)

    num_angle_nodes = len(angle_base_names)
    num_stats_per_angle = len(STAT_SUFFIXES)
    extra_input_dim = len(extra_feature_columns)

    print("=== DATA INFO ===")
    print(f"Total rows        : {len(df)}")
    print(f"Total file unik   : {df['file_name'].nunique()}")
    print(f"Total group unik  : {df['group_id'].nunique()}")
    print(f"Total label unik  : {df['label'].nunique()}")
    print(f"Angle base fitur  : {len(angle_base_names)}")
    print(f"Extra base fitur  : {len(extra_base_names)}")
    print(f"Extra input dim   : {extra_input_dim}")
    print("\nGroup per label:")
    print(df[["group_id", "label"]].drop_duplicates()["label"].value_counts())

    train_df, val_df, test_df = grouped_split_by_group_id(
        df=df,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    print("\n=== SPLIT INFO ===")
    print(f"Train rows  : {len(train_df)}")
    print(f"Val rows    : {len(val_df)}")
    print(f"Test rows   : {len(test_df)}")
    print(f"Train groups: {train_df['group_id'].nunique()}")
    print(f"Val groups  : {val_df['group_id'].nunique()}")
    print(f"Test groups : {test_df['group_id'].nunique()}")

    print("\nTrain label counts:")
    print(train_df["label"].value_counts())

    print("\nVal label counts:")
    print(val_df["label"].value_counts())

    print("\nTest label counts:")
    print(test_df["label"].value_counts())

    label_encoder = LabelEncoder()
    label_encoder.fit(df["label"].values)

    y_train = label_encoder.transform(train_df["label"].values)
    y_val = label_encoder.transform(val_df["label"].values)
    y_test = label_encoder.transform(test_df["label"].values)
    class_names = label_encoder.classes_

    X_train_angle = train_df[angle_feature_columns].values
    X_val_angle = val_df[angle_feature_columns].values
    X_test_angle = test_df[angle_feature_columns].values

    if extra_feature_columns:
        X_train_extra = train_df[extra_feature_columns].values
        X_val_extra = val_df[extra_feature_columns].values
        X_test_extra = test_df[extra_feature_columns].values
    else:
        X_train_extra = np.zeros((len(train_df), 0), dtype=np.float32)
        X_val_extra = np.zeros((len(val_df), 0), dtype=np.float32)
        X_test_extra = np.zeros((len(test_df), 0), dtype=np.float32)

    angle_scaler = StandardScaler()
    X_train_angle = angle_scaler.fit_transform(X_train_angle)
    X_val_angle = angle_scaler.transform(X_val_angle)
    X_test_angle = angle_scaler.transform(X_test_angle)

    if extra_feature_columns:
        extra_scaler = StandardScaler()
        X_train_extra = extra_scaler.fit_transform(X_train_extra)
        X_val_extra = extra_scaler.transform(X_val_extra)
        X_test_extra = extra_scaler.transform(X_test_extra)
    else:
        extra_scaler = None

    with open(model_dir / "angle_scaler.pkl", "wb") as f:
        pickle.dump(angle_scaler, f)

    if extra_scaler is not None:
        with open(model_dir / "extra_scaler.pkl", "wb") as f:
            pickle.dump(extra_scaler, f)

    with open(model_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({str(i): cls for i, cls in enumerate(class_names)}, f, indent=2, ensure_ascii=False)

    with open(model_dir / "angle_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(angle_feature_columns, f, indent=2, ensure_ascii=False)

    with open(model_dir / "extra_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(extra_feature_columns, f, indent=2, ensure_ascii=False)

    model_config = {
        "model_name": "hybrid_gcn_mlp",
        "feature_mode": "hybrid_gcn_mlp_angle_plus_extra",
        "angle_in_features": int(num_stats_per_angle),
        "extra_in_features": int(extra_input_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_classes": int(len(class_names)),
        "dropout": float(args.dropout),
        "angle_feature_columns": angle_feature_columns,
        "extra_feature_columns": extra_feature_columns,
        "class_names": list(class_names),
        "stat_suffixes": list(STAT_SUFFIXES),
        "angle_columns": list(ANGLE_COLUMNS),
        "seed": int(args.seed),
    }

    with open(model_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)

    with open(model_dir / "feature_scaler.pkl", "wb") as f:
        pickle.dump({
            "angle_scaler": angle_scaler,
            "extra_scaler": extra_scaler,
            "angle_feature_columns": angle_feature_columns,
            "extra_feature_columns": extra_feature_columns,
        }, f)

    with open(model_dir / "label_names.json", "w", encoding="utf-8") as f:
        json.dump(list(class_names), f, indent=2, ensure_ascii=False)

    train_ds = HybridPoseDataset(
        X_angle=X_train_angle,
        X_extra=X_train_extra,
        y=y_train,
        num_angle_nodes=num_angle_nodes,
        num_stats_per_angle=num_stats_per_angle,
    )
    val_ds = HybridPoseDataset(
        X_angle=X_val_angle,
        X_extra=X_val_extra,
        y=y_val,
        num_angle_nodes=num_angle_nodes,
        num_stats_per_angle=num_stats_per_angle,
    )
    test_ds = HybridPoseDataset(
        X_angle=X_test_angle,
        X_extra=X_test_extra,
        y=y_test,
        num_angle_nodes=num_angle_nodes,
        num_stats_per_angle=num_stats_per_angle,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj = torch.tensor(build_hand_adjacency(), dtype=torch.float32, device=device)

    model = HybridGCNMLPClassifier(
        angle_in_features=num_stats_per_angle,
        extra_in_features=extra_input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=len(class_names),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_loss = float("inf")
    best_epoch = -1
    wait = 0

    best_model_path = model_dir / "best_hybrid_gcn_mlp_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, adj
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, adj
        )[:2]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            wait += 1

        if wait >= args.patience:
            print(f"Early stopping pada epoch {epoch}. Best epoch: {best_epoch}")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(metric_dir / "epoch_metrics.csv", index=False)

    # =========================================
    # SAVE TRAINING CURVES
    # =========================================
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_dir / "loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_acc"], label="Train Accuracy")
    plt.plot(history_df["epoch"], history_df["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_dir / "accuracy_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc, y_true, y_pred = evaluate(
        model, test_loader, criterion, device, adj
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    save_confusion_matrix(
        cm,
        class_names=class_names,
        out_png=figure_dir / "confusion_matrix.png",
        out_csv=metric_dir / "confusion_matrix.csv",
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    with open(metric_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    summary = {
        "feature_mode": "hybrid_gcn_mlp_angle_plus_extra",
        "split_mode": "group_id",
        "best_epoch": int(best_epoch),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_precision_macro": float(precision),
        "test_recall_macro": float(recall),
        "test_f1_macro": float(f1),
        "num_classes": int(len(class_names)),
        "num_train_rows": int(len(train_df)),
        "num_val_rows": int(len(val_df)),
        "num_test_rows": int(len(test_df)),
        "num_train_groups": int(train_df["group_id"].nunique()),
        "num_val_groups": int(val_df["group_id"].nunique()),
        "num_test_groups": int(test_df["group_id"].nunique()),
        "num_angle_base_features": int(len(angle_base_names)),
        "num_extra_base_features": int(len(extra_base_names)),
        "extra_input_dim": int(extra_input_dim),
        "seed": int(args.seed),
    }


    bundle_payload = {
        "state_dict": model.state_dict(),
        "model_config": model_config,
        "label_mapping": {int(i): cls for i, cls in enumerate(class_names)},
        "angle_feature_columns": angle_feature_columns,
        "extra_feature_columns": extra_feature_columns,
        "angle_columns": list(ANGLE_COLUMNS),
        "stat_suffixes": list(STAT_SUFFIXES),
    }
    torch.save(bundle_payload, model_dir / "model_bundle.pth")

    with open(metric_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pred_df = pd.DataFrame({
        "file_name": test_df["file_name"].values,
        "rep_id": test_df["rep_id"].values,
        "window_id": test_df["window_id"].values,
        "group_id": test_df["group_id"].values,
        "true_label_id": y_true,
        "pred_label_id": y_pred,
        "true_label": label_encoder.inverse_transform(y_true),
        "pred_label": label_encoder.inverse_transform(y_pred),
    })
    pred_df.to_csv(pred_dir / "predictions.csv", index=False)

    print("\n=== HASIL TEST ===")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"\nBest model: {best_model_path}")
    print(f"Model bundle: {model_dir / 'model_bundle.pth'}")
    print(f"Model config: {model_dir / 'model_config.json'}")
    print(f"Feature scaler bundle: {model_dir / 'feature_scaler.pkl'}")
    print(f"Label names: {model_dir / 'label_names.json'}")
    print(f"Loss curve: {figure_dir / 'loss_curve.png'}")
    print(f"Accuracy curve: {figure_dir / 'accuracy_curve.png'}")
    print(f"Confusion matrix PNG: {figure_dir / 'confusion_matrix.png'}")
    print(f"Classification report: {metric_dir / 'classification_report.txt'}")


if __name__ == "__main__":
    main()