
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from gesture_stgcn_common import (
    LABELS,
    FRAME_FEATURE_NAMES,
    extract_right_hand_points,
    extract_frame_feature_vector,
    parse_jsonl_filename,
    points_to_relative_joint_array,
    resample_sequence,
)


def parse_args():
    p = argparse.ArgumentParser(description="Extract ST-GCN gesture dataset from Rokoko JSONL files")
    p.add_argument("--input_dir", type=str, required=True, help="Folder containing .jsonl recordings")
    p.add_argument("--output_dir", type=str, required=True, help="Output folder")
    p.add_argument("--seq_len", type=int, default=30, help="Resampled sequence length")
    return p.parse_args()


def load_jsonl(file_path: Path) -> List[Dict]:
    packets = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                packets.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"{file_path.name} line {line_no}: {e}")
    if not packets:
        raise ValueError(f"Empty file: {file_path.name}")
    return packets


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files in: {input_dir}")

    X_joint = []
    X_feat = []
    y = []
    file_names = []
    subjects = []
    rep_ids = []
    group_ids = []
    labels = []
    errors = []

    for fp in jsonl_files:
        try:
            label, subject, rep_id = parse_jsonl_filename(fp.stem)
            if label not in LABELS:
                raise ValueError(f"Label '{label}' not in supported labels: {LABELS}")

            packets = load_jsonl(fp)

            joint_seq = []
            feat_seq = []

            for packet in packets:
                pts = extract_right_hand_points(packet)
                joint_frame = points_to_relative_joint_array(pts)
                feat_frame = extract_frame_feature_vector(pts)
                joint_seq.append(joint_frame)
                feat_seq.append(feat_frame)

            joint_seq = np.stack(joint_seq, axis=0)
            feat_seq = np.stack(feat_seq, axis=0)

            joint_seq = resample_sequence(joint_seq, args.seq_len)
            feat_seq = resample_sequence(feat_seq, args.seq_len)

            joint_seq = np.transpose(joint_seq, (2, 0, 1)).astype(np.float32)
            feat_seq = np.transpose(feat_seq, (1, 0)).astype(np.float32)

            X_joint.append(joint_seq)
            X_feat.append(feat_seq)
            y.append(LABELS.index(label))
            file_names.append(fp.name)
            subjects.append(subject)
            rep_ids.append(rep_id)
            group_ids.append(fp.stem)
            labels.append(label)

            print(f"[OK] {fp.name} -> label={label}, frames={len(packets)}, seq_len={args.seq_len}")

        except Exception as e:
            msg = f"[ERROR] {fp.name}: {e}"
            print(msg)
            errors.append(msg)

    if not X_joint:
        raise RuntimeError("No valid samples extracted")

    X_joint = np.stack(X_joint, axis=0)
    X_feat = np.stack(X_feat, axis=0)
    y = np.array(y, dtype=np.int64)

    np.savez_compressed(
        output_dir / "gesture_stgcn_dataset.npz",
        X_joint=X_joint,
        X_feat=X_feat,
        y=y,
        labels=np.array(labels),
        label_names=np.array(LABELS),
        file_names=np.array(file_names),
        subjects=np.array(subjects),
        rep_ids=np.array(rep_ids),
        group_ids=np.array(group_ids),
        feature_names=np.array(FRAME_FEATURE_NAMES),
    )

    meta = {
        "num_samples": int(len(y)),
        "seq_len": int(args.seq_len),
        "joint_shape": list(X_joint.shape),
        "feat_shape": list(X_feat.shape),
        "labels": LABELS,
        "num_errors": len(errors),
    }

    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if errors:
        with open(output_dir / "extract_errors.log", "w", encoding="utf-8") as f:
            for line in errors:
                f.write(line + "\n")

    print("\nDone.")
    print(f"Saved dataset: {output_dir / 'gesture_stgcn_dataset.npz'}")


if __name__ == "__main__":
    main()
