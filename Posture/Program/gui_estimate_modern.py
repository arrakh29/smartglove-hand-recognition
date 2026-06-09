import sys
import json
import pickle
import socket
import threading
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph.opengl as gl


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

STAT_SUFFIXES = ["_mean", "_std", "_min", "_max"]

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

BONES = [
    ("wrist", "thumb_prox"),
    ("thumb_prox", "thumb_med"),
    ("thumb_med", "thumb_dist"),
    ("thumb_dist", "thumb_tip"),

    ("wrist", "index_prox"),
    ("index_prox", "index_med"),
    ("index_med", "index_dist"),
    ("index_dist", "index_tip"),

    ("wrist", "middle_prox"),
    ("middle_prox", "middle_med"),
    ("middle_med", "middle_dist"),
    ("middle_dist", "middle_tip"),

    ("wrist", "ring_prox"),
    ("ring_prox", "ring_med"),
    ("ring_med", "ring_dist"),
    ("ring_dist", "ring_tip"),

    ("wrist", "pinky_prox"),
    ("pinky_prox", "pinky_med"),
    ("pinky_med", "pinky_dist"),
    ("pinky_dist", "pinky_tip"),
]


# =========================================================
# MODEL
# =========================================================
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

        self.gcn1 = GCNLayer(angle_in_features, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn3 = GCNLayer(hidden_dim, hidden_dim)

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

        if self.has_extra:
            xm = self.extra_mlp(extra_x)
            x = torch.cat([xg, xm], dim=1)
        else:
            x = xg

        logits = self.classifier(x)
        return logits


def build_hand_adjacency() -> np.ndarray:
    n = len(ANGLE_COLUMNS)
    A = np.zeros((n, n), dtype=np.float32)

    edges = [
        (0, 1),
        (2, 3), (3, 4),
        (5, 6), (6, 7),
        (8, 9), (9, 10),
        (11, 12), (12, 13),
        (0, 2),
        (2, 5),
        (5, 8),
        (8, 11),
    ]

    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0

    for i in range(n):
        A[i, i] = 1.0

    deg = np.sum(A, axis=1)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-8))
    return (deg_inv_sqrt @ A @ deg_inv_sqrt).astype(np.float32)


def build_angle_node_features_from_row(row: np.ndarray) -> np.ndarray:
    return row.reshape(len(ANGLE_COLUMNS), len(STAT_SUFFIXES)).astype(np.float32)


# =========================================================
# GEOMETRY
# =========================================================
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


def extract_right_hand_points(packet):
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


def compute_palm_plane_normal(pts):
    wrist = pts["wrist"]
    idx = pts["index_prox"]
    mid = pts["middle_prox"]
    pky = pts["pinky_prox"]
    v1 = mid - wrist
    v2 = pky - idx
    return normalize_vec(np.cross(v1, v2))


def compute_mcp_plane_flexion(pts, joint_name: str) -> float:
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


def extract_right_hand_frame_angles(packet):
    pts = extract_right_hand_points(packet)
    angles = {}

    for joint_name in ANGLE_COLUMNS:
        if joint_name in MCP_JOINTS:
            angles[joint_name] = compute_mcp_plane_flexion(pts, joint_name)
        else:
            a_name, b_name, c_name = ANGLE_DEFINITIONS[joint_name]
            raw = angle_3pts(pts[a_name], pts[b_name], pts[c_name])
            angles[joint_name] = 180.0 - raw

    return pts, angles


def extract_discriminative_features_from_points(pts):
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


# =========================================================
# STABILIZATION + FEATURE BUILDING
# =========================================================
def ema_filter_angles(current_angles, prev_angles, alpha):
    if prev_angles is None:
        return dict(current_angles)
    out = {}
    for k in current_angles:
        out[k] = alpha * current_angles[k] + (1.0 - alpha) * prev_angles[k]
    return out


def ema_filter_extra(current_extra, prev_extra, alpha):
    if prev_extra is None:
        return dict(current_extra)
    out = {}
    for k in current_extra:
        out[k] = alpha * current_extra[k] + (1.0 - alpha) * prev_extra[k]
    return out


def compute_window_stability(angle_window) -> float:
    arr = np.array([[frame[k] for k in ANGLE_COLUMNS] for frame in angle_window], dtype=np.float32)
    return float(np.mean(np.std(arr, axis=0)))


def summarize_window(window_frames, base_names):
    feats = []
    for base in base_names:
        arr = np.array([frame[base] for frame in window_frames], dtype=np.float32)
        feats.extend([
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.min(arr)),
            float(np.max(arr)),
        ])
    return np.array(feats, dtype=np.float32)


def stable_prediction(pred_history):
    if not pred_history:
        return None
    return Counter(pred_history).most_common(1)[0][0]


def decide_pose_output(pred_label, angle_window, extra_window):
    """
    Keputusan akhir realtime:
    - Jika tangan tampak open-hand netral / ambigu, keluarkan idle.
    - Jika ambigu/open-hand, keluarkan idle.
    """
    if not angle_window or not extra_window:
        return pred_label, "predicted", None

    latest_angles = angle_window[-1]
    latest_extra = extra_window[-1]

    mean_pip_dip_flex = float(np.mean([
        latest_angles["Index_PIP"], latest_angles["Index_DIP"],
        latest_angles["Middle_PIP"], latest_angles["Middle_DIP"],
        latest_angles["Ring_PIP"], latest_angles["Ring_DIP"],
        latest_angles["Pinky_PIP"], latest_angles["Pinky_DIP"],
    ]))

    mean_mcp_flex = float(np.mean([
        latest_angles["Index_MCP"],
        latest_angles["Middle_MCP"],
        latest_angles["Ring_MCP"],
        latest_angles["Pinky_MCP"],
    ]))

    mean_extension = float(latest_extra.get("mean_four_finger_extension", 0.0))
    spread_index_pinky = float(latest_extra.get("angle_index_pinky", 0.0))
    thumb_index_dist = float(latest_extra.get("thumb_index_tip_dist", 0.0))

    open_hand_score = 0
    if mean_extension >= 2.55:
        open_hand_score += 1
    if mean_pip_dip_flex <= 28.0:
        open_hand_score += 1
    if mean_mcp_flex <= 35.0:
        open_hand_score += 1
    if spread_index_pinky >= 18.0:
        open_hand_score += 1
    if thumb_index_dist >= 0.55:
        open_hand_score += 1

    rake_score = 0
    if mean_extension <= 2.35:
        rake_score += 1
    if mean_pip_dip_flex >= 36.0:
        rake_score += 1
    if mean_mcp_flex >= 30.0:
        rake_score += 1

    likely_open_idle = open_hand_score >= 4
    likely_rake_shape = rake_score >= 2 and mean_pip_dip_flex >= 40.0

    if likely_open_idle and pred_label in {"rake", "palmar", "radialpalmar"}:
        return "idle", "idle_open_hand", (
            f"idle open-hand | raw={pred_label} | ext={mean_extension:.2f}, "
            f"pipdip={mean_pip_dip_flex:.1f}, mcp={mean_mcp_flex:.1f}, "
            f"spread={spread_index_pinky:.1f}, ti={thumb_index_dist:.2f}"
        )
    return pred_label, "predicted", None


# =========================================================
# 3D VIEWER
# =========================================================
class Hand3DView(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.setCameraPosition(distance=0.7, elevation=18, azimuth=35)

        grid = gl.GLGridItem()
        grid.scale(0.1, 0.1, 0.1)
        self.addItem(grid)

        self.scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=float),
            size=10,
            color=(1.0, 1.0, 1.0, 1.0),
            pxMode=True
        )
        self.addItem(self.scatter)

        self.lines = []
        for _ in BONES:
            line = gl.GLLinePlotItem(
                pos=np.zeros((2, 3), dtype=float),
                width=2,
                color=(0.2, 0.8, 1.0, 1.0),
                antialias=True,
                mode="lines"
            )
            self.lines.append(line)
            self.addItem(line)

    def _visualize_point(self, p: np.ndarray) -> np.ndarray:
        # Mirror sumbu X hanya untuk VISUALISASI,
        # agar right hand tidak tampak sebagai left hand.
        return np.array([-p[0], p[1], p[2]], dtype=float)

    def update_hand(self, pts: dict):
        if not pts:
            return

        keys = list(pts.keys())
        pos = np.array([self._visualize_point(pts[k]) for k in keys], dtype=float)
        self.scatter.setData(pos=pos, size=10)

        for line_item, (a, b) in zip(self.lines, BONES):
            segment = np.array([
                self._visualize_point(pts[a]),
                self._visualize_point(pts[b])
            ], dtype=float)
            line_item.setData(pos=segment)

        center = self._visualize_point(pts["wrist"])
        self.opts["center"] = QtGui.QVector3D(
            float(center[0]),
            float(center[1]),
            float(center[2])
        )


# =========================================================
# GUI APP
# =========================================================
class RokokoRealtimeEstimateApp(QtWidgets.QMainWindow):
    log_signal = QtCore.Signal(str)
    packet_signal = QtCore.Signal(int)
    status_signal = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rokoko Realtime Estimate - Hybrid GCN + MLP")
        self.resize(1280, 800)

        self.sock = None
        self.connected = False
        self.receiver_thread = None
        self.stop_receiver = threading.Event()

        self.model = None
        self.angle_scaler = None
        self.extra_scaler = None
        self.label_mapping = None
        self.angle_feature_columns = None
        self.extra_feature_columns = None
        self.extra_base_names = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adj = torch.tensor(build_hand_adjacency(), dtype=torch.float32, device=self.device)

        self.packet_count = 0
        self.latest_hand_points = None
        self.prev_filtered_angles = None
        self.prev_filtered_extra = None
        self.angle_window = deque(maxlen=10)
        self.extra_window = deque(maxlen=10)
        self.pred_history = deque(maxlen=2)

        self.current_label = "idle"
        self.current_raw_label = "-"
        self.current_conf = 0.0
        self.current_stability = 0.0
        self.current_state = "Idle"

        self._build_ui()

        self.log_signal.connect(self._append_log)
        self.packet_signal.connect(self._update_packet_label)
        self.status_signal.connect(self._refresh_prediction_status)

        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.timeout.connect(self._draw_preview)
        self.preview_timer.start(30)

        self._log("GUI realtime hybrid siap.")

    # -------------------------
    # UI BUILD
    # -------------------------
    def _build_ui(self):
        self.setStyleSheet("""
            QMainWindow { background: #07152b; }
            QWidget { color: #e8eefc; font-size: 13px; }
            QGroupBox {
                background: #172844;
                border: 1px solid #2d4469;
                border-radius: 18px;
                margin-top: 14px;
                padding-top: 12px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 2px 10px;
                color: #f3f7ff;
                background: #0b1833;
                border-radius: 8px;
            }
            QLabel { color: #d7e5ff; }
            QLineEdit, QPlainTextEdit {
                background: #f6f8fc;
                color: #10233f;
                border: 1px solid #c8d4e5;
                border-radius: 12px;
                padding: 10px 12px;
                selection-background-color: #3fb7ef;
            }
            QPushButton {
                background: #2a3d5d;
                border: 1px solid #3b567f;
                border-radius: 12px;
                padding: 10px 14px;
                color: white;
                font-weight: 600;
            }
            QPushButton:hover { background: #34527e; }
            QPushButton:pressed { background: #213756; }
            QPushButton#accentButton {
                background: #40b8ea;
                color: #081426;
                border: none;
            }
            QPushButton#accentButton:hover { background: #63c7f0; }
            QScrollArea { border: none; background: transparent; }
        """)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QHBoxLayout(central)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(18)

        left_panel = QtWidgets.QWidget()
        left_panel.setMinimumWidth(520)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(14)

        header_card = QtWidgets.QFrame()
        header_card.setStyleSheet("QFrame { background:#08152e; border:1px solid #22385d; border-radius:22px; }")
        header_layout = QtWidgets.QVBoxLayout(header_card)
        header_layout.setContentsMargins(20, 18, 20, 18)
        title = QtWidgets.QLabel("Realtime Gesture Estimate")
        title.setStyleSheet("font-size: 24px; font-weight: 800; color: #f4f8ff;")
        subtitle = QtWidgets.QLabel("Tema disamakan dengan model UI modern. Struktur isi estimator tetap asli dan tidak diubah.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #c3d6f5; font-size: 13px;")
        self.hero_badge = QtWidgets.QLabel("IDLE")
        self.hero_badge.setAlignment(QtCore.Qt.AlignCenter)
        self.hero_badge.setFixedWidth(86)
        self.hero_badge.setStyleSheet("background:#40b8ea; color:#07152b; border-radius:16px; padding:8px 10px; font-weight:800;")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        header_layout.addSpacing(4)
        header_layout.addWidget(self.hero_badge, 0, QtCore.Qt.AlignLeft)
        left_layout.addWidget(header_card)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(14)
        scroll.setWidget(scroll_content)
        left_layout.addWidget(scroll, 1)

        conn_group = QtWidgets.QGroupBox("Connection")
        conn_form = QtWidgets.QFormLayout(conn_group)
        conn_form.setLabelAlignment(QtCore.Qt.AlignLeft)
        conn_form.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        conn_form.setSpacing(10)
        self.host_edit = QtWidgets.QLineEdit("127.0.0.1")
        self.port_edit = QtWidgets.QLineEdit("14043")
        conn_form.addRow("Host", self.host_edit)
        conn_form.addRow("Port", self.port_edit)
        conn_btn_row = QtWidgets.QHBoxLayout()
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.setObjectName("accentButton")
        self.disconnect_btn = QtWidgets.QPushButton("Disconnect")
        self.connect_btn.clicked.connect(self.connect)
        self.disconnect_btn.clicked.connect(self.disconnect)
        conn_btn_row.addWidget(self.connect_btn)
        conn_btn_row.addWidget(self.disconnect_btn)
        conn_form.addRow(conn_btn_row)
        self.connection_status_label = QtWidgets.QLabel("Disconnected")
        self.connection_status_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #ffffff;")
        conn_form.addRow("Status", self.connection_status_label)
        scroll_layout.addWidget(conn_group)

        model_group = QtWidgets.QGroupBox("Model Files")
        model_layout = QtWidgets.QGridLayout(model_group)
        model_layout.setHorizontalSpacing(10)
        model_layout.setVerticalSpacing(10)
        self.model_path_edit = QtWidgets.QLineEdit()
        self.angle_scaler_path_edit = QtWidgets.QLineEdit()
        self.extra_scaler_path_edit = QtWidgets.QLineEdit()
        self.label_map_path_edit = QtWidgets.QLineEdit()
        self.angle_feat_cols_path_edit = QtWidgets.QLineEdit()
        self.extra_feat_cols_path_edit = QtWidgets.QLineEdit()
        self._add_browse_row(model_layout, 0, "Model .pth", self.model_path_edit, self.browse_model)
        self._add_browse_row(model_layout, 1, "Angle scaler .pkl", self.angle_scaler_path_edit, self.browse_angle_scaler)
        self._add_browse_row(model_layout, 2, "Extra scaler .pkl", self.extra_scaler_path_edit, self.browse_extra_scaler)
        self._add_browse_row(model_layout, 3, "Label map .json", self.label_map_path_edit, self.browse_label_map)
        self._add_browse_row(model_layout, 4, "Angle feat .json", self.angle_feat_cols_path_edit, self.browse_angle_feat_cols)
        self._add_browse_row(model_layout, 5, "Extra feat .json", self.extra_feat_cols_path_edit, self.browse_extra_feat_cols)
        self.hidden_dim_edit = QtWidgets.QLineEdit("64")
        self.dropout_edit = QtWidgets.QLineEdit("0.5")
        model_layout.addWidget(QtWidgets.QLabel("Hidden Dim"), 6, 0)
        model_layout.addWidget(self.hidden_dim_edit, 6, 1)
        model_layout.addWidget(QtWidgets.QLabel("Dropout"), 6, 2)
        model_layout.addWidget(self.dropout_edit, 6, 3)
        self.load_model_btn = QtWidgets.QPushButton("Load Model")
        self.load_model_btn.setObjectName("accentButton")
        self.load_model_btn.clicked.connect(self.load_model_files)
        model_layout.addWidget(self.load_model_btn, 7, 0, 1, 4)
        self.model_status_label = QtWidgets.QLabel("Model not loaded")
        self.model_status_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #ffffff;")
        model_layout.addWidget(self.model_status_label, 8, 0, 1, 4)
        scroll_layout.addWidget(model_group)

        param_group = QtWidgets.QGroupBox("Inference Parameters")
        param_form = QtWidgets.QFormLayout(param_group)
        param_form.setSpacing(10)
        self.window_size_edit = QtWidgets.QLineEdit("10")
        self.smooth_preds_edit = QtWidgets.QLineEdit("2")
        self.conf_threshold_edit = QtWidgets.QLineEdit("0.75")
        self.stability_threshold_edit = QtWidgets.QLineEdit("2.0")
        self.ema_alpha_edit = QtWidgets.QLineEdit("0.40")
        self.temperature_edit = QtWidgets.QLineEdit("2.0")
        param_form.addRow("Window Size", self.window_size_edit)
        param_form.addRow("Smooth Preds", self.smooth_preds_edit)
        param_form.addRow("Conf Threshold", self.conf_threshold_edit)
        param_form.addRow("Stab Threshold", self.stability_threshold_edit)
        param_form.addRow("EMA Alpha", self.ema_alpha_edit)
        param_form.addRow("Temperature", self.temperature_edit)
        scroll_layout.addWidget(param_group)

        status_group = QtWidgets.QGroupBox("Prediction Status")
        status_form = QtWidgets.QFormLayout(status_group)
        status_form.setSpacing(8)
        self.packet_status_label = QtWidgets.QLabel("Packets: 0")
        self.state_status_label = QtWidgets.QLabel("State: Idle")
        self.label_status_label = QtWidgets.QLabel("Label: -")
        self.raw_label_status_label = QtWidgets.QLabel("Raw: -")
        self.conf_status_label = QtWidgets.QLabel("Confidence: 0.000")
        self.stability_status_label = QtWidgets.QLabel("Stability: 0.000")
        status_form.addRow(self.packet_status_label)
        status_form.addRow(self.state_status_label)
        status_form.addRow(self.label_status_label)
        status_form.addRow(self.raw_label_status_label)
        status_form.addRow(self.conf_status_label)
        status_form.addRow(self.stability_status_label)
        scroll_layout.addWidget(status_group)

        log_group = QtWidgets.QGroupBox("Log")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(180)
        log_layout.addWidget(self.log_text)
        scroll_layout.addWidget(log_group, 1)
        scroll_layout.addStretch(1)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(14)

        self.info_cards = {}
        cards_layout = QtWidgets.QHBoxLayout()
        cards_layout.setSpacing(12)
        for key, title in [("connection", "Connection"), ("model", "Model"), ("state", "State"), ("label", "Label")]:
            card, value = self._create_info_card(title, "-")
            self.info_cards[key] = value
            cards_layout.addWidget(card)
        right_layout.addLayout(cards_layout)

        preview_group = QtWidgets.QGroupBox("Realtime Skeleton Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        preview_layout.setSpacing(10)
        self.preview_label = QtWidgets.QLabel("-")
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setStyleSheet("font-size: 28px; font-weight: 800; color: #ffffff; background:#08152e; border-radius: 12px; padding: 10px;")
        self.conf_progress = QtWidgets.QProgressBar()
        self.conf_progress.setRange(0, 100)
        self.conf_progress.setValue(0)
        self.conf_progress.setTextVisible(True)
        self.conf_progress.setStyleSheet("QProgressBar { background:#08152e; border:1px solid #29456e; border-radius: 10px; text-align:center; } QProgressBar::chunk { background:#40b8ea; border-radius: 10px; }")
        self.viewer = Hand3DView()
        self.viewer.setMinimumHeight(640)
        preview_layout.addWidget(self.preview_label)
        preview_layout.addWidget(self.conf_progress)
        preview_layout.addWidget(self.viewer, 1)
        right_layout.addWidget(preview_group, 1)

        outer.addWidget(left_panel, 0)
        outer.addWidget(right_panel, 1)

    def _create_info_card(self, title: str, value: str):
        card = QtWidgets.QFrame()
        card.setStyleSheet("QFrame { background:#172844; border:1px solid #2d4469; border-radius: 18px; }")
        card.setMinimumHeight(92)
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("color:#b7ccee; font-size:12px;")
        value_label = QtWidgets.QLabel(value)
        value_label.setStyleSheet("font-size: 20px; font-weight: 800; color:#ffffff;")
        value_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        return card, value_label

    def _add_browse_row(self, layout, row, label, line_edit, callback):
        layout.addWidget(QtWidgets.QLabel(label), row, 0)
        layout.addWidget(line_edit, row, 1, 1, 2)
        btn = QtWidgets.QPushButton("Browse")
        btn.clicked.connect(callback)
        layout.addWidget(btn, row, 3)

    # -------------------------
    # LOG + STATUS
    # -------------------------
    def _log(self, msg):
        self.log_signal.emit(msg)

    def _append_log(self, msg):
        from time import strftime
        ts = strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{ts}] {msg}")

    def _update_packet_label(self, count):
        self.packet_status_label.setText(f"Packets: {count}")

    def _push_status(self):
        self.status_signal.emit()

    def _refresh_prediction_status(self):
        self.state_status_label.setText(f"State: {self.current_state}")
        self.label_status_label.setText(f"Label: {self.current_label}")
        self.raw_label_status_label.setText(f"Raw: {self.current_raw_label}")
        self.conf_status_label.setText(f"Confidence: {self.current_conf:.3f}")
        self.stability_status_label.setText(f"Stability: {self.current_stability:.3f}")
        self.hero_badge.setText(str(self.current_label).upper())
        self.preview_label.setText(self.current_label if self.current_label else "-")
        self.conf_progress.setValue(max(0, min(100, int(round(self.current_conf * 100)))))
        self.info_cards["connection"].setText(self.connection_status_label.text())
        self.info_cards["model"].setText(self.model_status_label.text())
        self.info_cards["state"].setText(self.current_state)
        self.info_cards["label"].setText(self.current_label)

        self.setWindowTitle(
            f"Realtime Gesture ST-GCN - Modern UI | {self.current_label} | "
            f"conf={self.current_conf:.3f} | {self.current_state}"
        )

    # -------------------------
    # FILE BROWSER
    # -------------------------
    def browse_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Model", "", "PyTorch Model (*.pth);;All Files (*)"
        )
        if path:
            self.model_path_edit.setText(path)

    def browse_angle_scaler(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Angle Scaler", "", "Pickle (*.pkl);;All Files (*)"
        )
        if path:
            self.angle_scaler_path_edit.setText(path)

    def browse_extra_scaler(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Extra Scaler", "", "Pickle (*.pkl);;All Files (*)"
        )
        if path:
            self.extra_scaler_path_edit.setText(path)

    def browse_label_map(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Label Map", "", "JSON (*.json);;All Files (*)"
        )
        if path:
            self.label_map_path_edit.setText(path)

    def browse_angle_feat_cols(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Angle Feature Columns", "", "JSON (*.json);;All Files (*)"
        )
        if path:
            self.angle_feat_cols_path_edit.setText(path)

    def browse_extra_feat_cols(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Extra Feature Columns", "", "JSON (*.json);;All Files (*)"
        )
        if path:
            self.extra_feat_cols_path_edit.setText(path)

    # -------------------------
    # MODEL LOAD
    # -------------------------
    def load_model_files(self):
        try:
            model_path = Path(self.model_path_edit.text().strip())
            angle_scaler_path = Path(self.angle_scaler_path_edit.text().strip())
            extra_scaler_path = Path(self.extra_scaler_path_edit.text().strip())
            label_map_path = Path(self.label_map_path_edit.text().strip())
            angle_feat_cols_path = Path(self.angle_feat_cols_path_edit.text().strip())
            extra_feat_cols_path = Path(self.extra_feat_cols_path_edit.text().strip())

            for p, name in [
                (model_path, "Model"),
                (angle_scaler_path, "Angle scaler"),
                (extra_scaler_path, "Extra scaler"),
                (label_map_path, "Label map"),
                (angle_feat_cols_path, "Angle feature columns"),
                (extra_feat_cols_path, "Extra feature columns"),
            ]:
                if not p.exists():
                    raise FileNotFoundError(f"{name} tidak ditemukan: {p}")

            with open(angle_scaler_path, "rb") as f:
                self.angle_scaler = pickle.load(f)

            with open(extra_scaler_path, "rb") as f:
                self.extra_scaler = pickle.load(f)

            with open(label_map_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.label_mapping = {int(k): v for k, v in raw.items()}

            with open(angle_feat_cols_path, "r", encoding="utf-8") as f:
                self.angle_feature_columns = json.load(f)

            with open(extra_feat_cols_path, "r", encoding="utf-8") as f:
                self.extra_feature_columns = json.load(f)

            self.extra_base_names = []
            seen = set()
            for c in self.extra_feature_columns:
                for s in STAT_SUFFIXES:
                    if c.endswith(s):
                        base = c[:-len(s)]
                        if base not in seen:
                            seen.add(base)
                            self.extra_base_names.append(base)
                        break

            hidden_dim = int(self.hidden_dim_edit.text().strip())
            dropout = float(self.dropout_edit.text().strip())

            self.model = HybridGCNMLPClassifier(
                angle_in_features=len(STAT_SUFFIXES),
                extra_in_features=len(self.extra_feature_columns),
                hidden_dim=hidden_dim,
                num_classes=len(self.label_mapping),
                dropout=dropout,
            ).to(self.device)

            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

            self.model_status_label.setText("Model loaded")
            self.info_cards["model"].setText(self.model_status_label.text())
            self._log("Model hybrid berhasil diload.")
            self._log(f"Angle feature cols: {len(self.angle_feature_columns)}")
            self._log(f"Extra feature cols: {len(self.extra_feature_columns)}")

        except Exception as e:
            self.model_status_label.setText("Load failed")
            self.info_cards["model"].setText(self.model_status_label.text())
            self._log(f"Gagal load model: {e}")
            QtWidgets.QMessageBox.critical(self, "Load Model Error", str(e))

    # -------------------------
    # CONNECTION
    # -------------------------
    def connect(self):
        if self.connected:
            self._log("Sudah terhubung.")
            return

        try:
            host = self.host_edit.text().strip()
            port = int(self.port_edit.text().strip())

            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((host, port))
            self.sock.settimeout(0.5)

            self.stop_receiver.clear()
            self.receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
            self.receiver_thread.start()

            self.connected = True
            self.connection_status_label.setText(f"Connected ({host}:{port})")
            self.info_cards["connection"].setText(self.connection_status_label.text())
            self._log(f"Connected ke UDP {host}:{port}")

        except Exception as e:
            self._log(f"Gagal connect: {e}")
            QtWidgets.QMessageBox.critical(self, "Connect Error", str(e))

    def disconnect(self):
        if not self.connected:
            return

        self.stop_receiver.set()
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass

        self.sock = None
        self.connected = False
        self.connection_status_label.setText("Disconnected")
        self.info_cards["connection"].setText(self.connection_status_label.text())
        self._log("Disconnected.")

    # -------------------------
    # RECEIVER LOOP
    # -------------------------
    def _receiver_loop(self):
        while not self.stop_receiver.is_set():
            try:
                packet, _addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as e:
                self._log(f"Receiver error: {e}")
                continue

            data = try_parse_json(packet)
            if data is None:
                continue

            self.packet_count += 1
            self.packet_signal.emit(self.packet_count)

            try:
                pts, raw_angles = extract_right_hand_frame_angles(data)
                raw_extra = extract_discriminative_features_from_points(pts)
                self.latest_hand_points = pts
            except Exception:
                continue

            try:
                alpha = float(self.ema_alpha_edit.text().strip())

                filtered_angles = ema_filter_angles(raw_angles, self.prev_filtered_angles, alpha)
                filtered_extra = ema_filter_extra(raw_extra, self.prev_filtered_extra, alpha)

                self.prev_filtered_angles = filtered_angles
                self.prev_filtered_extra = filtered_extra

                window_size = int(self.window_size_edit.text().strip())
                smooth_preds = int(self.smooth_preds_edit.text().strip())
                conf_threshold = float(self.conf_threshold_edit.text().strip())
                stab_threshold = float(self.stability_threshold_edit.text().strip())
                temperature = float(self.temperature_edit.text().strip())

                if self.angle_window.maxlen != window_size:
                    old = list(self.angle_window)
                    self.angle_window = deque(old[-window_size:], maxlen=window_size)

                if self.extra_window.maxlen != window_size:
                    old = list(self.extra_window)
                    self.extra_window = deque(old[-window_size:], maxlen=window_size)

                if self.pred_history.maxlen != smooth_preds:
                    old = list(self.pred_history)
                    self.pred_history = deque(old[-smooth_preds:], maxlen=smooth_preds)

                self.angle_window.append(filtered_angles)
                self.extra_window.append(filtered_extra)

                if len(self.angle_window) < window_size:
                    self.current_label = "idle"
                    self.current_state = f"buffering {len(self.angle_window)}/{window_size}"
                    self._push_status()
                    continue

                stability_value = compute_window_stability(self.angle_window)
                self.current_stability = stability_value

                if stability_value > stab_threshold:
                    self.pred_history.clear()
                    self.current_label = "idle"
                    self.current_state = f"unstable ({stability_value:.3f})"
                    self._push_status()
                    continue

                if (
                    self.model is None
                    or self.angle_scaler is None
                    or self.extra_scaler is None
                    or self.label_mapping is None
                    or self.angle_feature_columns is None
                    or self.extra_feature_columns is None
                ):
                    self.current_state = "model not loaded"
                    self._push_status()
                    continue

                angle56 = summarize_window(self.angle_window, ANGLE_COLUMNS)
                extra_vec = summarize_window(self.extra_window, self.extra_base_names)

                x_angle = self.angle_scaler.transform(angle56.reshape(1, -1))
                x_extra = self.extra_scaler.transform(extra_vec.reshape(1, -1))

                angle_node_features = build_angle_node_features_from_row(x_angle[0])

                angle_tensor = torch.tensor(angle_node_features, dtype=torch.float32).unsqueeze(0).to(self.device)
                extra_tensor = torch.tensor(x_extra, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    logits = self.model(angle_tensor, extra_tensor, self.adj)
                    probs = torch.softmax(logits / temperature, dim=1).cpu().numpy()[0]
                    pred_idx = int(np.argmax(probs))
                    pred_label = self.label_mapping[pred_idx]
                    conf = float(np.max(probs))

                decided_label, decided_state, decision_msg = decide_pose_output(
                    pred_label=pred_label,
                    angle_window=self.angle_window,
                    extra_window=self.extra_window,
                )

                self.current_raw_label = pred_label
                self.current_conf = conf

                if conf < conf_threshold:
                    self.pred_history.clear()
                    self.current_label = "idle"
                    self.current_state = f"low_conf ({conf:.3f})"
                    self._push_status()
                    continue

                if decision_msg is not None:
                    self._log(decision_msg)

                if decided_label == "idle":
                    self.pred_history.clear()
                    self.current_label = "idle"
                    self.current_state = decided_state
                    self._push_status()
                    continue

                self.pred_history.append(decided_label)
                stable_label = stable_prediction(self.pred_history)
                self.current_label = stable_label if stable_label is not None else decided_label
                self.current_state = decided_state
                self._push_status()

            except Exception as e:
                self._log(f"Inference error: {e}")

    # -------------------------
    # DRAW PREVIEW
    # -------------------------
    def _draw_preview(self):
        if self.latest_hand_points is None:
            return
        self.viewer.update_hand(self.latest_hand_points)

    # -------------------------
    # CLOSE EVENT
    # -------------------------
    def closeEvent(self, event):
        try:
            self.disconnect()
        finally:
            event.accept()


# =========================================================
# MAIN
# =========================================================
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = RokokoRealtimeEstimateApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()