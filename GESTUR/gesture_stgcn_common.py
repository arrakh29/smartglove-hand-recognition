
import json
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


LABELS = [
    "down",
    "left",
    "right",
    "rotate_clockwise",
    "rotate_counterclockwise",
    "up",
    "idle",
]

JOINT_ORDER = [
    "wrist",
    "thumb_prox", "thumb_med", "thumb_dist", "thumb_tip",
    "index_prox", "index_med", "index_dist", "index_tip",
    "middle_prox", "middle_med", "middle_dist", "middle_tip",
    "ring_prox", "ring_med", "ring_dist", "ring_tip",
    "pinky_prox", "pinky_med", "pinky_dist", "pinky_tip",
]

JOINT_INDEX = {name: i for i, name in enumerate(JOINT_ORDER)}

HAND_EDGES = [
    (0, 1),
    (1, 2), (2, 3), (3, 4),
    (0, 5),
    (5, 6), (6, 7), (7, 8),
    (0, 9),
    (9, 10), (10, 11), (11, 12),
    (0, 13),
    (13, 14), (14, 15), (15, 16),
    (0, 17),
    (17, 18), (18, 19), (19, 20),
]

LABEL_OPTIONS = LABELS
BONES = [(JOINT_ORDER[a], JOINT_ORDER[b]) for a, b in HAND_EDGES]

ANGLE_COLUMNS = [
    "Thumb_MCP", "Thumb_IP",
    "Index_MCP", "Index_PIP", "Index_DIP",
    "Middle_MCP", "Middle_PIP", "Middle_DIP",
    "Ring_MCP", "Ring_PIP", "Ring_DIP",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP",
]

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

FRAME_FEATURE_NAMES = ANGLE_COLUMNS + EXTRA_FEATURE_COLUMNS


def try_parse_json(packet: bytes):
    try:
        return json.loads(packet.decode("utf-8"))
    except Exception:
        return None


def vec3(node):
    p = node["position"]
    return np.array([p["x"], p["y"], p["z"]], dtype=np.float64)


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


def extract_right_hand_frame_angles(pts: Dict[str, np.ndarray]) -> Dict[str, float]:
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

    palm_center = np.mean([wrist, index_mcp, middle_mcp, ring_mcp, pinky_mcp], axis=0)

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


def extract_frame_feature_vector(pts: Dict[str, np.ndarray]) -> np.ndarray:
    angles = extract_right_hand_frame_angles(pts)
    extra = extract_discriminative_features_from_points(pts)
    values = [angles[c] for c in ANGLE_COLUMNS] + [extra[c] for c in EXTRA_FEATURE_COLUMNS]
    return np.array(values, dtype=np.float32)


def points_to_relative_joint_array(pts: Dict[str, np.ndarray]) -> np.ndarray:
    arr = np.stack([pts[name] for name in JOINT_ORDER], axis=0).astype(np.float32)
    wrist = arr[0].copy()
    arr = arr - wrist[None, :]
    scale = (
        np.linalg.norm(arr[JOINT_INDEX["index_prox"]]) +
        np.linalg.norm(arr[JOINT_INDEX["middle_prox"]]) +
        np.linalg.norm(arr[JOINT_INDEX["pinky_prox"]])
    ) / 3.0
    scale = max(float(scale), 1e-6)
    arr = arr / scale
    return arr


def build_adjacency(num_nodes: int = 21, edges: List[Tuple[int, int]] = None) -> np.ndarray:
    if edges is None:
        edges = HAND_EDGES
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    for i in range(num_nodes):
        A[i, i] = 1.0
    deg = np.sum(A, axis=1)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-8))
    return (deg_inv_sqrt @ A @ deg_inv_sqrt).astype(np.float32)


def parse_jsonl_filename(stem: str):
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid file name format: {stem}")
    rep_token = parts[-1]
    subject = parts[-2]
    label = "_".join(parts[:-2])
    if not rep_token.lower().startswith("rep"):
        raise ValueError(f"Invalid rep token in file name: {stem}")
    rep_id = rep_token[3:].zfill(2)
    return label, subject, rep_id


def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    T = len(seq)
    if T == 0:
        raise ValueError("Empty sequence")
    if T == target_len:
        return seq.copy()
    if T == 1:
        return np.repeat(seq, target_len, axis=0)
    idx = np.linspace(0, T - 1, target_len)
    out = []
    for i in idx:
        lo = int(np.floor(i))
        hi = min(lo + 1, T - 1)
        a = i - lo
        frame = (1.0 - a) * seq[lo] + a * seq[hi]
        out.append(frame)
    return np.stack(out, axis=0)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dropout=0.3, residual=True):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        xg = torch.einsum("nctv,vw->nctw", x, A)
        xg = self.gcn(xg)
        xt = self.tcn(xg)
        return self.relu(xt + self.residual(x))


class TemporalFeatureBranch(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_features, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.mean(dim=-1)
        return x


class GestureSTGCN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        joint_in_channels: int = 3,
        feature_in_dim: int = len(FRAME_FEATURE_NAMES),
        stgcn_dropout: float = 0.3,
        feat_dropout: float = 0.3,
        hidden_dim: int = 64,
        feat_hidden_dim: int = 128,
    ):
        super().__init__()
        num_joints = len(JOINT_ORDER)
        self.data_bn = nn.BatchNorm1d(joint_in_channels * num_joints)
        self.block1 = STGCNBlock(joint_in_channels, hidden_dim, residual=False, dropout=stgcn_dropout)
        self.block2 = STGCNBlock(hidden_dim, hidden_dim, dropout=stgcn_dropout)
        self.block3 = STGCNBlock(hidden_dim, hidden_dim * 2, stride=2, dropout=stgcn_dropout)
        self.block4 = STGCNBlock(hidden_dim * 2, hidden_dim * 2, dropout=stgcn_dropout)
        self.block5 = STGCNBlock(hidden_dim * 2, hidden_dim * 4, stride=2, dropout=stgcn_dropout)
        self.block6 = STGCNBlock(hidden_dim * 4, hidden_dim * 4, dropout=stgcn_dropout)
        self.feature_branch = TemporalFeatureBranch(in_features=feature_in_dim, hidden_dim=feat_hidden_dim, dropout=feat_dropout)
        fusion_dim = hidden_dim * 4 + feat_hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, joint_x, feat_x, A):
        n, c, t, v = joint_x.shape
        x = joint_x.permute(0, 3, 1, 2).contiguous()
        x = x.view(n, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, v, c, t).permute(0, 2, 3, 1).contiguous()
        x = self.block1(x, A)
        x = self.block2(x, A)
        x = self.block3(x, A)
        x = self.block4(x, A)
        x = self.block5(x, A)
        x = self.block6(x, A)
        x = x.mean(dim=-1).mean(dim=-1)
        f = self.feature_branch(feat_x)
        z = torch.cat([x, f], dim=1)
        logits = self.classifier(z)
        return logits
