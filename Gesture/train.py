import argparse
import json
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from gesture_stgcn_common import GestureSTGCN, build_adjacency


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GestureDataset(Dataset):
    def __init__(self, X_joint, X_feat, y):
        self.X_joint = X_joint.astype(np.float32)
        self.X_feat = X_feat.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_joint[idx], dtype=torch.float32),
            torch.tensor(self.X_feat[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def grouped_split(group_ids, y, seed=42, test_size=0.15, val_size=0.15):
    rng = random.Random(seed)
    df = pd.DataFrame({"group_id": group_ids, "y": y}).drop_duplicates()

    train_groups, val_groups, test_groups = [], [], []

    for cls, sub in df.groupby("y"):
        groups = sub["group_id"].tolist()
        rng.shuffle(groups)

        n = len(groups)
        if n < 3:
            raise ValueError(f"Class {cls} has only {n} groups; need at least 3")

        n_test = max(1, round(n * test_size))
        n_val = max(1, round(n * val_size))

        while n_test + n_val >= n:
            if n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break

        test_groups.extend(groups[:n_test])
        val_groups.extend(groups[n_test:n_test + n_val])
        train_groups.extend(groups[n_test + n_val:])

    train_idx = [i for i, g in enumerate(group_ids) if g in train_groups]
    val_idx = [i for i, g in enumerate(group_ids) if g in val_groups]
    test_idx = [i for i, g in enumerate(group_ids) if g in test_groups]

    return train_idx, val_idx, test_idx


def train_one_epoch(model, loader, optimizer, criterion, device, A):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for joint_x, feat_x, yb in loader:
        joint_x = joint_x.to(device)
        feat_x = feat_x.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(joint_x, feat_x, A)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * joint_x.size(0)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(yb.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, A):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    for joint_x, feat_x, yb in loader:
        joint_x = joint_x.to(device)
        feat_x = feat_x.to(device)
        yb = yb.to(device)

        logits = model(joint_x, feat_x, A)
        loss = criterion(logits, yb)
        probs = torch.softmax(logits, dim=1)

        total_loss += loss.item() * joint_x.size(0)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(yb.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)

    return avg_loss, acc, np.array(all_targets), np.array(all_preds), np.array(all_probs)


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
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def save_training_curves(history, output_dir: Path):
    history_df = pd.DataFrame(history)

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_acc"], label="Train Accuracy")
    plt.plot(history_df["epoch"], history_df["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="Train gesture ST-GCN + temporal feature branch")
    p.add_argument("--dataset_npz", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--feat_hidden_dim", type=int, default=128)
    p.add_argument("--stgcn_dropout", type=float, default=0.3)
    p.add_argument("--feat_dropout", type=float, default=0.3)
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

    data = np.load(args.dataset_npz, allow_pickle=True)
    X_joint = data["X_joint"]
    X_feat = data["X_feat"]
    y = data["y"]
    file_names = data["file_names"]
    subjects = data["subjects"]
    rep_ids = data["rep_ids"]
    group_ids = data["group_ids"]
    label_names = data["label_names"].tolist()
    feature_names = data["feature_names"].tolist()

    print("=== DATA INFO ===")
    print("X_joint:", X_joint.shape)
    print("X_feat :", X_feat.shape)
    print("y      :", y.shape)
    print("labels :", label_names)

    train_idx, val_idx, test_idx = grouped_split(
        group_ids=group_ids,
        y=y,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    X_train_joint, X_val_joint, X_test_joint = X_joint[train_idx], X_joint[val_idx], X_joint[test_idx]
    X_train_feat, X_val_feat, X_test_feat = X_feat[train_idx], X_feat[val_idx], X_feat[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    feat_dim = X_feat.shape[1]
    scaler = StandardScaler()

    train_feat_2d = np.transpose(X_train_feat, (0, 2, 1)).reshape(-1, feat_dim)
    val_feat_2d = np.transpose(X_val_feat, (0, 2, 1)).reshape(-1, feat_dim)
    test_feat_2d = np.transpose(X_test_feat, (0, 2, 1)).reshape(-1, feat_dim)

    train_feat_2d = scaler.fit_transform(train_feat_2d)
    val_feat_2d = scaler.transform(val_feat_2d)
    test_feat_2d = scaler.transform(test_feat_2d)

    T = X_feat.shape[2]
    X_train_feat = np.transpose(train_feat_2d.reshape(len(X_train_feat), T, feat_dim), (0, 2, 1))
    X_val_feat = np.transpose(val_feat_2d.reshape(len(X_val_feat), T, feat_dim), (0, 2, 1))
    X_test_feat = np.transpose(test_feat_2d.reshape(len(X_test_feat), T, feat_dim), (0, 2, 1))

    train_ds = GestureDataset(X_train_joint, X_train_feat, y_train)
    val_ds = GestureDataset(X_val_joint, X_val_feat, y_val)
    test_ds = GestureDataset(X_test_joint, X_test_feat, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    A = torch.tensor(build_adjacency(), dtype=torch.float32, device=device)

    model = GestureSTGCN(
        num_classes=len(label_names),
        joint_in_channels=X_joint.shape[1],
        feature_in_dim=X_feat.shape[1],
        hidden_dim=args.hidden_dim,
        feat_hidden_dim=args.feat_hidden_dim,
        stgcn_dropout=args.stgcn_dropout,
        feat_dropout=args.feat_dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_model_path = model_dir / "best_gesture_stgcn.pth"
    best_val_loss = float("inf")
    best_epoch = -1
    wait = 0
    history = []

    print("\n=== TRAINING START ===")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, A)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device, A)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

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
            print(f"Early stopping at epoch {epoch}, best_epoch={best_epoch}")
            break

    # Save epoch metrics
    history_df = pd.DataFrame(history)
    history_df.to_csv(metric_dir / "epoch_metrics.csv", index=False)

    # Save training curves
    save_training_curves(history, figure_dir)

    # Save best bundle
    torch.save(
        {
            "model_state_dict": torch.load(best_model_path, map_location="cpu"),
            "joint_in_channels": int(X_joint.shape[1]),
            "feature_in_dim": int(X_feat.shape[1]),
            "num_classes": int(len(label_names)),
            "hidden_dim": int(args.hidden_dim),
            "feat_hidden_dim": int(args.feat_hidden_dim),
            "stgcn_dropout": float(args.stgcn_dropout),
            "feat_dropout": float(args.feat_dropout),
        },
        model_dir / "best_gesture_stgcn_bundle.pth",
    )

    # Load best model for testing
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc, y_true, y_pred, probs = evaluate(
        model, test_loader, criterion, device, A
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(label_names)),
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_names)))
    save_confusion_matrix(
        cm,
        label_names,
        figure_dir / "confusion_matrix.png",
        metric_dir / "confusion_matrix.csv",
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(label_names)),
        target_names=label_names,
        digits=4,
        zero_division=0,
    )

    with open(metric_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    summary = {
        "best_epoch": int(best_epoch),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_precision_macro": float(precision),
        "test_recall_macro": float(recall),
        "test_f1_macro": float(f1),
        "num_samples": int(len(y)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_test": int(len(test_idx)),
        "label_names": label_names,
        "feature_names": feature_names,
        "seed": int(args.seed),
    }

    with open(metric_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pred_df = pd.DataFrame(
        {
            "file_name": file_names[test_idx],
            "subject": subjects[test_idx],
            "rep_id": rep_ids[test_idx],
            "group_id": group_ids[test_idx],
            "true_label_id": y_true,
            "pred_label_id": y_pred,
            "true_label": [label_names[i] for i in y_true],
            "pred_label": [label_names[i] for i in y_pred],
            "confidence": probs.max(axis=1),
        }
    )
    pred_df.to_csv(pred_dir / "predictions.csv", index=False)

    joblib.dump(scaler, model_dir / "feature_scaler.pkl")
    with open(model_dir / "label_names.json", "w", encoding="utf-8") as f:
        json.dump(label_names, f, indent=2)

    print("\n=== TEST RESULT ===")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Best epoch: {best_epoch}")

    print("\n=== OUTPUT FILES ===")
    print("Model bundle        :", model_dir / "best_gesture_stgcn_bundle.pth")
    print("Scaler              :", model_dir / "feature_scaler.pkl")
    print("Label names         :", model_dir / "label_names.json")
    print("Epoch metrics CSV   :", metric_dir / "epoch_metrics.csv")
    print("Metrics summary JSON:", metric_dir / "metrics_summary.json")
    print("Classification rep. :", metric_dir / "classification_report.txt")
    print("Confusion matrix CSV:", metric_dir / "confusion_matrix.csv")
    print("Predictions CSV     :", pred_dir / "predictions.csv")
    print("Accuracy curve PNG  :", figure_dir / "accuracy_curve.png")
    print("Loss curve PNG      :", figure_dir / "loss_curve.png")
    print("Confusion matrix PNG:", figure_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()