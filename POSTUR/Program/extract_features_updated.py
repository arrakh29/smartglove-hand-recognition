import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================
ANGLE_COLUMNS = [
    "Thumb_MCP", "Thumb_IP",
    "Index_MCP", "Index_PIP", "Index_DIP",
    "Middle_MCP", "Middle_PIP", "Middle_DIP",
    "Ring_MCP", "Ring_PIP", "Ring_DIP",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP",
]

ANGLE_MEAN_COLUMNS = [f"{c}_mean" for c in ANGLE_COLUMNS]
ANGLE_STD_COLUMNS = [f"{c}_std" for c in ANGLE_COLUMNS]
ANGLE_MIN_COLUMNS = [f"{c}_min" for c in ANGLE_COLUMNS]
ANGLE_MAX_COLUMNS = [f"{c}_max" for c in ANGLE_COLUMNS]

MCP_JOINTS = {"Thumb_MCP", "Index_MCP", "Middle_MCP", "Ring_MCP", "Pinky_MCP"}

ANGLE_DEFINITIONS = {
    "Thumb_MCP": ("wrist", "thumb_prox", "thumb_med"),
    "Thumb_IP": ("thumb_prox", "thumb_med", "thumb_dist"),

    "Index_MCP": ("wrist", "index_prox", "index_med"),
    "Index_PIP": ("index_prox", "index_med", "index_dist"),
    "Index_DIP": ("index_med", "index_dist", "index_tip"),

    "Middle_MCP": ("wrist", "middle_prox", "middle_med"),
    "Middle_PIP": ("middle_prox", "middle_med", "middle_dist"),
    "Middle_DIP": ("middle_med", "middle_dist", "middle_tip"),

    "Ring_MCP": ("wrist", "ring_prox", "ring_med"),
    "Ring_PIP": ("ring_prox", "ring_med", "ring_dist"),
    "Ring_DIP": ("ring_med", "ring_dist", "ring_tip"),

    "Pinky_MCP": ("wrist", "pinky_prox", "pinky_med"),
    "Pinky_PIP": ("pinky_prox", "pinky_med", "pinky_dist"),
    "Pinky_DIP": ("pinky_med", "pinky_dist", "pinky_tip"),
}

EXTRA_FEATURE_COLUMNS = [
    "thumb_index_tip_dist",
    "thumb_middle_tip_dist",
    "thumb_ring_tip_dist",
    "thumb_pinky_tip_dist",
    "thumb_tip_to_palm",
    "index_tip_to_palm",
    "middle_tip_to_palm",
    "ring_tip_to_palm",
    "pinky_tip_to_palm",
    "index_middle_tip_dist",
    "middle_ring_tip_dist",
    "ring_pinky_tip_dist",
    "index_pinky_tip_dist",
    "spread_ratio_index_pinky_vs_index_middle",
    "spread_ratio_thumb_index_vs_thumb_pinky",
    "index_extension",
    "middle_extension",
    "ring_extension",
    "pinky_extension",
    "thumb_extension",
    "angle_thumb_index",
    "angle_index_middle",
    "angle_middle_ring",
    "angle_ring_pinky",
    "angle_index_pinky",
    "pinch_index",
    "pinch_middle",
    "pinch_ring",
    "pinch_pinky",
    "mean_four_finger_extension",
]

EXTRA_MEAN_COLUMNS = [f"{c}_mean" for c in EXTRA_FEATURE_COLUMNS]
EXTRA_STD_COLUMNS = [f"{c}_std" for c in EXTRA_FEATURE_COLUMNS]
EXTRA_MIN_COLUMNS = [f"{c}_min" for c in EXTRA_FEATURE_COLUMNS]
EXTRA_MAX_COLUMNS = [f"{c}_max" for c in EXTRA_FEATURE_COLUMNS]


# =========================================================
# ARGUMENTS
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(description="Ekstraksi fitur sudut + fitur diskriminatif dari JSONL Rokoko 3 detik.")
    p.add_argument("--input_dir", type=str, required=True, help="Folder berisi file .jsonl hasil recorder.")
    p.add_argument("--output_dir", type=str, required=True, help="Folder output.")
    p.add_argument("--num_windows", type=int, default=15, help="Jumlah window per file. Default 15.")
    p.add_argument("--save_frame_features", action="store_true", help="Simpan fitur per frame.")
    p.add_argument("--frame_features_dir_name", type=str, default="frame_features", help="Subfolder frame features.")
    p.add_argument("--output_name", type=str, default="features_jsonl_xyz_to_angle_v2.csv", help="Nama file CSV output fitur.")
    p.add_argument("--save_schema", action="store_true", help="Simpan schema fitur dan ringkasan ekstraksi.")
    return p.parse_args()


# =========================================================
# HELPERS
# =========================================================
def normalize_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0

    cos_theta = np.dot(v1, v2) / (n1 * n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def project_to_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    return v - np.dot(v, n) * n


def signed_angle_in_plane(v1: np.ndarray, v2: np.ndarray, plane_normal: np.ndarray) -> float:
    v1n = normalize_vec(v1)
    v2n = normalize_vec(v2)
    pn = normalize_vec(plane_normal)

    dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
    cross = np.cross(v1n, v2n)
    s = np.dot(cross, pn)
    return float(np.degrees(np.arctan2(abs(s), dot)))


def aggregate_stats(series: pd.Series) -> Tuple[float, float, float, float]:
    arr = pd.to_numeric(series, errors="coerce")
    return (
        float(arr.mean(skipna=True)),
        float(arr.std(skipna=True, ddof=0)),
        float(arr.min(skipna=True)),
        float(arr.max(skipna=True)),
    )


def vec3(node: Dict) -> np.ndarray:
    p = node["position"]
    return np.array([p["x"], p["y"], p["z"]], dtype=np.float64)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = _unit(v1)
    v2 = _unit(v2)
    cos_val = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


# =========================================================
# PACKET PARSING
# =========================================================
def extract_right_hand_points(packet: Dict) -> Dict[str, np.ndarray]:
    body = packet["scene"]["actors"][0]["body"]

    return {
        "wrist": vec3(body["rightHand"]),

        "thumb_prox": vec3(body["rightThumbProximal"]),
        "thumb_med": vec3(body["rightThumbMedial"]),
        "thumb_dist": vec3(body["rightThumbDistal"]),
        "thumb_tip": vec3(body["rightThumbTip"]),

        "index_prox": vec3(body["rightIndexProximal"]),
        "index_med": vec3(body["rightIndexMedial"]),
        "index_dist": vec3(body["rightIndexDistal"]),
        "index_tip": vec3(body["rightIndexTip"]),

        "middle_prox": vec3(body["rightMiddleProximal"]),
        "middle_med": vec3(body["rightMiddleMedial"]),
        "middle_dist": vec3(body["rightMiddleDistal"]),
        "middle_tip": vec3(body["rightMiddleTip"]),

        "ring_prox": vec3(body["rightRingProximal"]),
        "ring_med": vec3(body["rightRingMedial"]),
        "ring_dist": vec3(body["rightRingDistal"]),
        "ring_tip": vec3(body["rightRingTip"]),

        "pinky_prox": vec3(body["rightLittleProximal"]),
        "pinky_med": vec3(body["rightLittleMedial"]),
        "pinky_dist": vec3(body["rightLittleDistal"]),
        "pinky_tip": vec3(body["rightLittleTip"]),
    }


def compute_palm_plane_normal(pts: Dict[str, np.ndarray]) -> np.ndarray:
    wrist = pts["wrist"]
    idx = pts["index_prox"]
    mid = pts["middle_prox"]
    pky = pts["pinky_prox"]

    v1 = mid - wrist
    v2 = pky - idx
    n = np.cross(v1, v2)
    return normalize_vec(n)


def compute_mcp_plane_flexion(pts: Dict[str, np.ndarray], joint_name: str) -> float:
    a_name, b_name, c_name = ANGLE_DEFINITIONS[joint_name]
    pa = pts[a_name]
    pb = pts[b_name]
    pc = pts[c_name]

    incoming = pa - pb
    outgoing = pc - pb

    palm_n = compute_palm_plane_normal(pts)
    incoming_proj = project_to_plane(incoming, palm_n)
    outgoing_proj = project_to_plane(outgoing, palm_n)

    plane_ang = signed_angle_in_plane(incoming_proj, outgoing_proj, palm_n)
    return 180.0 - plane_ang


def extract_right_hand_frame_angles(packet: Dict) -> Dict[str, float]:
    pts = extract_right_hand_points(packet)
    angles = {}

    for joint_name in ANGLE_COLUMNS:
        if joint_name in MCP_JOINTS:
            angles[joint_name] = compute_mcp_plane_flexion(pts, joint_name)
        else:
            a_name, b_name, c_name = ANGLE_DEFINITIONS[joint_name]
            raw = angle_3pts(pts[a_name], pts[b_name], pts[c_name])
            angles[joint_name] = 180.0 - raw

    return angles


def extract_discriminative_features_from_points(pts: Dict[str, np.ndarray]) -> Dict[str, float]:
    wrist = pts["wrist"]
    thumb_tip = pts["thumb_tip"]
    index_tip = pts["index_tip"]
    middle_tip = pts["middle_tip"]
    ring_tip = pts["ring_tip"]
    pinky_tip = pts["pinky_tip"]

    thumb_mcp = pts["thumb_med"]
    index_mcp = pts["index_prox"]
    middle_mcp = pts["middle_prox"]
    ring_mcp = pts["ring_prox"]
    pinky_mcp = pts["pinky_prox"]

    palm_center = np.mean(
        [wrist, index_mcp, middle_mcp, ring_mcp, pinky_mcp],
        axis=0
    )

    hand_scale = (
        _dist(wrist, middle_mcp) +
        _dist(wrist, index_mcp) +
        _dist(wrist, pinky_mcp)
    ) / 3.0 + 1e-8

    v_thumb = thumb_tip - palm_center
    v_index = index_tip - palm_center
    v_middle = middle_tip - palm_center
    v_ring = ring_tip - palm_center
    v_pinky = pinky_tip - palm_center

    features = {}

    features["thumb_index_tip_dist"] = _dist(thumb_tip, index_tip) / hand_scale
    features["thumb_middle_tip_dist"] = _dist(thumb_tip, middle_tip) / hand_scale
    features["thumb_ring_tip_dist"] = _dist(thumb_tip, ring_tip) / hand_scale
    features["thumb_pinky_tip_dist"] = _dist(thumb_tip, pinky_tip) / hand_scale

    features["thumb_tip_to_palm"] = _dist(thumb_tip, palm_center) / hand_scale
    features["index_tip_to_palm"] = _dist(index_tip, palm_center) / hand_scale
    features["middle_tip_to_palm"] = _dist(middle_tip, palm_center) / hand_scale
    features["ring_tip_to_palm"] = _dist(ring_tip, palm_center) / hand_scale
    features["pinky_tip_to_palm"] = _dist(pinky_tip, palm_center) / hand_scale

    features["index_middle_tip_dist"] = _dist(index_tip, middle_tip) / hand_scale
    features["middle_ring_tip_dist"] = _dist(middle_tip, ring_tip) / hand_scale
    features["ring_pinky_tip_dist"] = _dist(ring_tip, pinky_tip) / hand_scale
    features["index_pinky_tip_dist"] = _dist(index_tip, pinky_tip) / hand_scale

    features["spread_ratio_index_pinky_vs_index_middle"] = (
        features["index_pinky_tip_dist"] / (features["index_middle_tip_dist"] + 1e-8)
    )
    features["spread_ratio_thumb_index_vs_thumb_pinky"] = (
        features["thumb_index_tip_dist"] / (features["thumb_pinky_tip_dist"] + 1e-8)
    )

    features["index_extension"] = _dist(index_tip, index_mcp) / hand_scale
    features["middle_extension"] = _dist(middle_tip, middle_mcp) / hand_scale
    features["ring_extension"] = _dist(ring_tip, ring_mcp) / hand_scale
    features["pinky_extension"] = _dist(pinky_tip, pinky_mcp) / hand_scale
    features["thumb_extension"] = _dist(thumb_tip, thumb_mcp) / hand_scale

    features["angle_thumb_index"] = _angle_between(v_thumb, v_index)
    features["angle_index_middle"] = _angle_between(v_index, v_middle)
    features["angle_middle_ring"] = _angle_between(v_middle, v_ring)
    features["angle_ring_pinky"] = _angle_between(v_ring, v_pinky)
    features["angle_index_pinky"] = _angle_between(v_index, v_pinky)

    features["pinch_index"] = features["thumb_index_tip_dist"]
    features["pinch_middle"] = features["thumb_middle_tip_dist"]
    features["pinch_ring"] = features["thumb_ring_tip_dist"]
    features["pinch_pinky"] = features["thumb_pinky_tip_dist"]

    features["mean_four_finger_extension"] = float(np.mean([
        features["index_extension"],
        features["middle_extension"],
        features["ring_extension"],
        features["pinky_extension"],
    ]))

    return features


def extract_right_hand_frame_features(packet: Dict) -> Dict[str, float]:
    pts = extract_right_hand_points(packet)
    angles = extract_right_hand_frame_angles(packet)
    extra = extract_discriminative_features_from_points(pts)

    row = {}
    row.update(angles)
    row.update(extra)
    return row


# =========================================================
# FILE NAME PARSING
# format: pose_subject_rep01.jsonl
# =========================================================
def parse_jsonl_filename(stem: str) -> Tuple[str, str, str]:
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Nama file tidak sesuai format: {stem}. "
            "Contoh: radialdigital_person1_rep01.jsonl"
        )

    rep_token = parts[-1]
    subject = parts[-2]
    label = "_".join(parts[:-2])

    if not rep_token.lower().startswith("rep"):
        raise ValueError(f"Token repetition tidak valid pada nama file: {stem}")

    rep_id = rep_token[3:].zfill(2)
    return label, subject, rep_id


# =========================================================
# LOAD JSONL
# =========================================================
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
                raise ValueError(f"Gagal parse JSON di line {line_no}: {e}")
    if not packets:
        raise ValueError("File JSONL kosong.")
    return packets


def build_frame_features(packets: List[Dict]) -> pd.DataFrame:
    rows = []
    total_frames = len(packets)

    for i, packet in enumerate(packets):
        feats = extract_right_hand_frame_features(packet)
        row = {"frame": i}
        row.update(feats)
        rows.append(row)

    df = pd.DataFrame(rows)

    if total_frames > 1:
        df["time_s"] = np.linspace(0.0, 3.0, total_frames)
    else:
        df["time_s"] = 0.0

    return df


# =========================================================
# WINDOWING
# =========================================================
def split_into_windows(df: pd.DataFrame, num_windows: int) -> List[pd.DataFrame]:
    n_frames = len(df)
    if n_frames == 0:
        return []

    index_splits = np.array_split(np.arange(n_frames), num_windows)
    windows = []
    for idxs in index_splits:
        if len(idxs) == 0:
            continue
        windows.append(df.iloc[idxs].copy())
    return windows


def aggregate_window_features(
    window_df: pd.DataFrame,
    label: str,
    subject: str,
    rep_id: str,
    source_file: str,
    window_id: int,
    total_frames_file: int,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "file_name": source_file,
        "label": label,
        "subject": subject,
        "sample_id": rep_id,
        "rep_id": rep_id,
        "window_id": f"{window_id:02d}",
        "group_id": f"{Path(source_file).stem}",
        "source_system": "rokoko_jsonl",
        "hand_side": "right",
        "feature_mode": "xyz_to_angle_jsonl_plus_discriminative",
        "num_frames_total_file": int(total_frames_file),
        "file_duration_s": float(3.0),
        "num_frames_window": int(len(window_df)),
    }

    if len(window_df) > 0:
        row["window_start_time_s"] = float(window_df["time_s"].iloc[0])
        row["window_end_time_s"] = float(window_df["time_s"].iloc[-1])
        row["window_duration_s"] = float(window_df["time_s"].iloc[-1] - window_df["time_s"].iloc[0])
    else:
        row["window_start_time_s"] = 0.0
        row["window_end_time_s"] = 0.0
        row["window_duration_s"] = 0.0

    for col in ANGLE_COLUMNS:
        mean_v, std_v, min_v, max_v = aggregate_stats(window_df[col])
        row[f"{col}_mean"] = mean_v
        row[f"{col}_std"] = std_v
        row[f"{col}_min"] = min_v
        row[f"{col}_max"] = max_v

    for col in EXTRA_FEATURE_COLUMNS:
        mean_v, std_v, min_v, max_v = aggregate_stats(window_df[col])
        row[f"{col}_mean"] = mean_v
        row[f"{col}_std"] = std_v
        row[f"{col}_min"] = min_v
        row[f"{col}_max"] = max_v

    return row


# =========================================================
# MAIN
# =========================================================
def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_features_dir = output_dir / args.frame_features_dir_name
    if args.save_frame_features:
        frame_features_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"Tidak ada file JSONL di folder: {input_dir}")

    all_rows: List[Dict[str, object]] = []
    error_logs: List[str] = []

    for file_path in jsonl_files:
        try:
            label, subject, rep_id = parse_jsonl_filename(file_path.stem)

            packets = load_jsonl(file_path)
            frame_df = build_frame_features(packets)

            if args.save_frame_features:
                out_frame = frame_features_dir / f"{file_path.stem}_frame_features.csv"
                frame_df.to_csv(out_frame, index=False)

            total_frames_file = len(frame_df)
            windows = split_into_windows(frame_df, args.num_windows)

            kept_windows = 0
            for win_id, win_df in enumerate(windows, start=1):
                row = aggregate_window_features(
                    window_df=win_df,
                    label=label,
                    subject=subject,
                    rep_id=rep_id,
                    source_file=file_path.name,
                    window_id=win_id,
                    total_frames_file=total_frames_file,
                )
                all_rows.append(row)
                kept_windows += 1

            print(f"[OK] {file_path.name} -> {total_frames_file} frames -> {kept_windows} windows")

        except Exception as e:
            msg = f"[ERROR] {file_path.name}: {e}"
            print(msg)
            error_logs.append(msg)

    if not all_rows:
        raise RuntimeError("Tidak ada sampel yang berhasil diproses.")

    features_df = pd.DataFrame(all_rows)

    metadata_cols = [
        "file_name", "label", "subject", "sample_id",
        "rep_id", "window_id", "group_id",
        "source_system", "hand_side", "feature_mode",
        "num_frames_total_file", "file_duration_s", "num_frames_window",
        "window_start_time_s", "window_end_time_s", "window_duration_s",
    ]

    ordered_cols = (
        metadata_cols
        + ANGLE_MEAN_COLUMNS
        + ANGLE_STD_COLUMNS
        + ANGLE_MIN_COLUMNS
        + ANGLE_MAX_COLUMNS
        + EXTRA_MEAN_COLUMNS
        + EXTRA_STD_COLUMNS
        + EXTRA_MIN_COLUMNS
        + EXTRA_MAX_COLUMNS
    )

    features_df = features_df[ordered_cols]

    output_file = output_dir / args.output_name
    features_df.to_csv(output_file, index=False)
    print(f"\nSelesai. Dataset fitur tersimpan di: {output_file}")

    if args.save_schema:
        schema_payload = {
            "feature_mode": "xyz_to_angle_jsonl_plus_discriminative",
            "num_windows": int(args.num_windows),
            "num_files_processed": int(len(jsonl_files)),
            "num_rows_output": int(len(features_df)),
            "metadata_columns": metadata_cols,
            "angle_columns": ANGLE_COLUMNS,
            "extra_feature_columns": EXTRA_FEATURE_COLUMNS,
            "ordered_output_columns": ordered_cols,
            "output_csv": output_file.name,
        }
        schema_file = output_dir / "feature_schema.json"
        with open(schema_file, "w", encoding="utf-8") as f:
            json.dump(schema_payload, f, indent=2, ensure_ascii=False)
        print(f"Schema fitur tersimpan di: {schema_file}")

    if error_logs:
        error_file = output_dir / "extract_errors.log"
        with open(error_file, "w", encoding="utf-8") as f:
            for line in error_logs:
                f.write(line + "\n")
        print(f"Log error tersimpan di: {error_file}")


if __name__ == "__main__":
    main()