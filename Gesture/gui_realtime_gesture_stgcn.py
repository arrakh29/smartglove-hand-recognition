import json
import socket
import threading
from collections import Counter, deque
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import torch

from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph.opengl as gl

from gesture_stgcn_common import (
    GestureSTGCN,
    build_adjacency,
    extract_right_hand_points,
    extract_frame_feature_vector,
    points_to_relative_joint_array,
    try_parse_json,
    JOINT_ORDER,
    HAND_EDGES,
)


def stable_prediction(pred_history):
    if not pred_history:
        return None
    return Counter(pred_history).most_common(1)[0][0]


def debounced_prediction(pred_history, min_count=2):
    if not pred_history:
        return None
    label, count = Counter(pred_history).most_common(1)[0]
    return label if count >= min_count else None


class Hand3DView(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.setBackgroundColor("#081120")
        self.setCameraPosition(distance=0.65, elevation=18, azimuth=35)

        grid = gl.GLGridItem()
        grid.scale(0.1, 0.1, 0.1)
        grid.setColor((0.30, 0.38, 0.50, 0.65))
        self.addItem(grid)

        axis = gl.GLAxisItem()
        axis.setSize(0.08, 0.08, 0.08)
        self.addItem(axis)

        self.scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=float),
            size=11,
            color=(0.22, 0.74, 0.97, 1.0),
            pxMode=True,
        )
        self.addItem(self.scatter)

        self.lines = []
        for _ in HAND_EDGES:
            line = gl.GLLinePlotItem(
                pos=np.zeros((2, 3), dtype=float),
                width=3,
                color=(0.38, 0.65, 0.98, 1.0),
                antialias=True,
                mode="lines",
            )
            self.lines.append(line)
            self.addItem(line)

    def _visualize_point(self, p: np.ndarray) -> np.ndarray:
        return np.array([-p[0], p[1], p[2]], dtype=float)

    def update_hand(self, pts: dict):
        if not pts:
            return

        pos = np.array([self._visualize_point(pts[k]) for k in JOINT_ORDER], dtype=float)
        self.scatter.setData(pos=pos, size=11)

        for idx, (line_item, (a, b)) in enumerate(zip(self.lines, HAND_EDGES)):
            pa = self._visualize_point(pts[JOINT_ORDER[a]])
            pb = self._visualize_point(pts[JOINT_ORDER[b]])
            color = (0.13, 0.77, 0.37, 1.0) if idx < 4 else (0.38, 0.65, 0.98, 1.0)
            line_item.setData(pos=np.array([pa, pb], dtype=float), color=color)

        center = self._visualize_point(pts["wrist"])
        self.opts["center"] = QtGui.QVector3D(float(center[0]), float(center[1]), float(center[2]))

        spread = np.ptp(pos, axis=0)
        radius = max(0.12, float(np.max(spread)) * 1.8)
        self.setCameraPosition(distance=radius * 3.4, elevation=self.opts["elevation"], azimuth=self.opts["azimuth"])


class RealtimeGestureSTGCNApp(QtWidgets.QMainWindow):
    log_signal = QtCore.Signal(str)
    status_signal = QtCore.Signal()
    packet_signal = QtCore.Signal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Realtime Gesture ST-GCN - Modern UI")
        self.resize(1560, 930)
        self.setMinimumSize(1280, 780)

        self.colors = {
            "bg": "#0f172a",
            "panel": "#111827",
            "card": "#1e293b",
            "card2": "#243244",
            "text": "#e5eefc",
            "muted": "#a9b8d4",
            "accent": "#38bdf8",
            "accent2": "#22c55e",
            "warning": "#f59e0b",
            "danger": "#ef4444",
        }

        self.sock = None
        self.connected = False
        self.receiver_thread = None
        self.stop_receiver = threading.Event()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.A = torch.tensor(build_adjacency(), dtype=torch.float32, device=self.device)

        self.model = None
        self.scaler = None
        self.label_names = None

        self.packet_count = 0
        self.latest_hand_points = None
        self.prev_joint_frame = None
        self.prev_wrist_point = None

        self.joint_window = deque(maxlen=30)
        self.feat_window = deque(maxlen=30)
        self.pred_history = deque(maxlen=3)
        self.debounce_buffer = deque(maxlen=3)
        self.predicted_history = deque(maxlen=40)

        self.current_label = "-"
        self.current_raw_label = "-"
        self.current_conf = 0.0
        self.current_state = "Idle"
        self.current_motion = 0.0
        self.current_wrist_motion = 0.0
        self.cooldown_counter = 0
        self.last_logged_prediction = None

        self._apply_styles()
        self._build_ui()

        self.log_signal.connect(self._append_log)
        self.status_signal.connect(self._refresh_status)
        self.packet_signal.connect(self._update_packet_label)

        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.timeout.connect(self._draw_preview)
        self.preview_timer.start(30)

        self.badge_timer = QtCore.QTimer(self)
        self.badge_timer.timeout.connect(self._refresh_badge)
        self.badge_timer.start(250)

        self._log("ST-GCN realtime modern UI siap.")

    def _apply_styles(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {self.colors['bg']};
                color: {self.colors['text']};
                font-family: Segoe UI, Arial, sans-serif;
                font-size: 10pt;
            }}
            QFrame#panel {{
                background: {self.colors['panel']};
                border-radius: 16px;
            }}
            QFrame#card {{
                background: {self.colors['card']};
                border: 1px solid #334155;
                border-radius: 14px;
            }}
            QLabel#heroTitle {{
                font-size: 24pt;
                font-weight: 700;
                color: {self.colors['text']};
            }}
            QLabel#heroSub {{
                font-size: 10pt;
                color: {self.colors['muted']};
            }}
            QLabel#sectionTitle {{
                font-size: 11pt;
                font-weight: 700;
                color: {self.colors['text']};
            }}
            QLabel#muted {{
                color: {self.colors['muted']};
            }}
            QLabel#value {{
                color: {self.colors['text']};
                font-size: 12pt;
                font-weight: 700;
            }}
            QLabel#bigValue {{
                color: {self.colors['text']};
                font-size: 18pt;
                font-weight: 700;
            }}
            QLabel#historyItem {{
                color: #dbeafe;
                background: {self.colors['card2']};
                border: 1px solid #3b4d63;
                border-radius: 8px;
                padding: 6px 8px;
            }}
            QLineEdit {{
                background: #f8fafc;
                color: #111827;
                border: 1px solid #cbd5e1;
                border-radius: 10px;
                padding: 8px 10px;
                min-height: 20px;
            }}
            QPushButton {{
                background: {self.colors['card2']};
                color: {self.colors['text']};
                border: 1px solid #3b4d63;
                border-radius: 10px;
                padding: 8px 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: #30445d;
            }}
            QPushButton#accent {{
                background: {self.colors['accent']};
                color: #08111f;
                border: none;
            }}
            QPushButton#accent:hover {{
                background: #67d3fb;
            }}
            QPlainTextEdit, QListWidget {{
                background: #0b1220;
                color: #dbeafe;
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 8px;
            }}
            QListWidget::item {{
                padding: 4px;
                border-bottom: 1px solid #1e293b;
            }}
            QProgressBar {{
                background: #0b1220;
                border: 1px solid #334155;
                border-radius: 10px;
                text-align: center;
                height: 18px;
            }}
            QProgressBar::chunk {{
                background: {self.colors['accent2']};
                border-radius: 9px;
            }}
            QScrollArea {{
                border: none;
                background: transparent;
            }}
            QScrollBar:vertical {{
                background: {self.colors['panel']};
                width: 12px;
                margin: 4px 0 4px 0;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: #3b4d63;
                min-height: 24px;
                border-radius: 6px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QHBoxLayout(central)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        left_panel = QtWidgets.QFrame()
        left_panel.setObjectName("panel")
        left_panel.setFixedWidth(500)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        hero = QtWidgets.QFrame()
        hero.setStyleSheet(f"background:{self.colors['panel']}; border-top-left-radius:16px; border-top-right-radius:16px;")
        hero_layout = QtWidgets.QVBoxLayout(hero)
        hero_layout.setContentsMargins(18, 18, 18, 12)
        title = QtWidgets.QLabel("Realtime Gesture ST-GCN")
        title.setObjectName("heroTitle")
        subtitle = QtWidgets.QLabel("Tema disamakan dengan dataset recorder, plus histori prediksi realtime.")
        subtitle.setObjectName("heroSub")
        self.badge = QtWidgets.QLabel("IDLE")
        self.badge.setAlignment(QtCore.Qt.AlignCenter)
        self.badge.setStyleSheet(
            f"background:{self.colors['accent']}; color:#08111f; border-radius:12px; padding:8px 12px; font-weight:700;"
        )
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        hero_layout.addSpacing(8)
        hero_layout.addWidget(self.badge, 0, QtCore.Qt.AlignLeft)
        left_layout.addWidget(hero)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        scroll_content = QtWidgets.QWidget()
        self.left_content_layout = QtWidgets.QVBoxLayout(scroll_content)
        self.left_content_layout.setContentsMargins(10, 10, 10, 10)
        self.left_content_layout.setSpacing(10)
        scroll.setWidget(scroll_content)
        left_layout.addWidget(scroll)

        right_panel = QtWidgets.QFrame()
        right_panel.setObjectName("panel")
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(10)

        outer.addWidget(left_panel)
        outer.addWidget(right_panel, 1)

        self._build_left_cards()
        self._build_right_panel(right_layout)

    def _make_card(self, title_text):
        frame = QtWidgets.QFrame()
        frame.setObjectName("card")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QtWidgets.QLabel(title_text)
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        return frame, layout

    def _form_row(self, label_text, widget):
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)
        label = QtWidgets.QLabel(label_text)
        label.setObjectName("muted")
        label.setMinimumWidth(120)
        row.addWidget(label)
        row.addWidget(widget, 1)
        return row

    def _add_browse_row(self, layout, row, label, line_edit, callback):
        layout.addWidget(QtWidgets.QLabel(label), row, 0)
        layout.addWidget(line_edit, row, 1, 1, 2)
        btn = QtWidgets.QPushButton("Browse")
        btn.clicked.connect(callback)
        layout.addWidget(btn, row, 3)

    def _build_left_cards(self):
        self.host_edit = QtWidgets.QLineEdit("127.0.0.1")
        self.port_edit = QtWidgets.QLineEdit("14043")
        self.model_path_edit = QtWidgets.QLineEdit()
        self.scaler_path_edit = QtWidgets.QLineEdit()
        self.label_path_edit = QtWidgets.QLineEdit()

        self.connection_status_label = QtWidgets.QLabel("Disconnected")
        self.connection_status_label.setObjectName("value")
        self.model_status_label = QtWidgets.QLabel("Model not loaded")
        self.model_status_label.setObjectName("value")
        self.packet_status_label = QtWidgets.QLabel("Packets: 0")
        self.packet_status_label.setObjectName("value")
        self.state_status_label = QtWidgets.QLabel("State: Idle")
        self.state_status_label.setObjectName("value")
        self.label_status_label = QtWidgets.QLabel("Label: -")
        self.label_status_label.setObjectName("value")
        self.raw_label_status_label = QtWidgets.QLabel("Raw: -")
        self.conf_status_label = QtWidgets.QLabel("Confidence: 0.000")
        self.motion_status_label = QtWidgets.QLabel("Motion: 0.000")
        self.wrist_motion_status_label = QtWidgets.QLabel("Wrist Motion: 0.000")
        self.cooldown_status_label = QtWidgets.QLabel("Cooldown: 0")

        card, layout = self._make_card("Connection")
        layout.addLayout(self._form_row("Host", self.host_edit))
        layout.addLayout(self._form_row("Port", self.port_edit))
        btn_row = QtWidgets.QHBoxLayout()
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.setObjectName("accent")
        self.disconnect_btn = QtWidgets.QPushButton("Disconnect")
        btn_row.addWidget(self.connect_btn)
        btn_row.addWidget(self.disconnect_btn)
        layout.addLayout(btn_row)
        layout.addLayout(self._form_row("Status", self.connection_status_label))
        self.left_content_layout.addWidget(card)

        card, layout = self._make_card("Model Files")
        grid = QtWidgets.QGridLayout()
        self._add_browse_row(grid, 0, "Model bundle .pth", self.model_path_edit, self.browse_model)
        self._add_browse_row(grid, 1, "Feature scaler .pkl", self.scaler_path_edit, self.browse_scaler)
        self._add_browse_row(grid, 2, "Label names .json", self.label_path_edit, self.browse_label_json)
        layout.addLayout(grid)
        self.load_model_btn = QtWidgets.QPushButton("Load Model")
        self.load_model_btn.setObjectName("accent")
        layout.addWidget(self.load_model_btn)
        layout.addLayout(self._form_row("Model", self.model_status_label))
        self.left_content_layout.addWidget(card)

        card, layout = self._make_card("Inference Parameters")
        self.window_size_edit = QtWidgets.QLineEdit("24")
        self.smooth_preds_edit = QtWidgets.QLineEdit("2")
        self.conf_threshold_edit = QtWidgets.QLineEdit("0.60")
        self.raw_motion_threshold_edit = QtWidgets.QLineEdit("0.030")
        self.wrist_motion_threshold_edit = QtWidgets.QLineEdit("0.015")
        self.debounce_frames_edit = QtWidgets.QLineEdit("2")
        self.cooldown_frames_edit = QtWidgets.QLineEdit("3")
        for text, widget in [
            ("Window Size", self.window_size_edit),
            ("Smooth Preds", self.smooth_preds_edit),
            ("Conf Threshold", self.conf_threshold_edit),
            ("Raw Motion Th", self.raw_motion_threshold_edit),
            ("Wrist Motion Th", self.wrist_motion_threshold_edit),
            ("Debounce Frames", self.debounce_frames_edit),
            ("Cooldown Frames", self.cooldown_frames_edit),
        ]:
            layout.addLayout(self._form_row(text, widget))
        self.left_content_layout.addWidget(card)

        card, layout = self._make_card("Prediction Status")
        for text, widget in [
            ("Packets", self.packet_status_label),
            ("State", self.state_status_label),
            ("Label", self.label_status_label),
            ("Raw", self.raw_label_status_label),
            ("Confidence", self.conf_status_label),
            ("Motion", self.motion_status_label),
            ("Wrist Motion", self.wrist_motion_status_label),
            ("Cooldown", self.cooldown_status_label),
        ]:
            layout.addLayout(self._form_row(text, widget))
        self.left_content_layout.addWidget(card)

        card, layout = self._make_card("Predicted History")
        self.history_list = QtWidgets.QListWidget()
        self.history_list.setMinimumHeight(180)
        layout.addWidget(self.history_list)
        self.left_content_layout.addWidget(card)

        card, layout = self._make_card("Log")
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        self.left_content_layout.addWidget(card, 1)
        self.left_content_layout.addStretch(1)

        self.connect_btn.clicked.connect(self.connect_socket)
        self.disconnect_btn.clicked.connect(self.disconnect_socket)
        self.load_model_btn.clicked.connect(self.load_model_files)

    def _build_right_panel(self, parent_layout):
        summary = QtWidgets.QHBoxLayout()
        summary.setSpacing(10)
        self.summary_value_labels = {}

        for title in ["Connection", "Model", "State", "Label"]:
            card = QtWidgets.QFrame()
            card.setObjectName("card")
            card_layout = QtWidgets.QVBoxLayout(card)
            card_layout.setContentsMargins(14, 12, 14, 12)
            label = QtWidgets.QLabel(title)
            label.setObjectName("muted")
            value = QtWidgets.QLabel("-")
            value.setObjectName("bigValue")
            card_layout.addWidget(label)
            card_layout.addWidget(value)
            summary.addWidget(card, 1)
            self.summary_value_labels[title] = value

        parent_layout.addLayout(summary)

        topbar = QtWidgets.QFrame()
        topbar.setStyleSheet(f"background:{self.colors['card2']}; border:1px solid #334155; border-radius:12px;")
        topbar_layout = QtWidgets.QHBoxLayout(topbar)
        topbar_layout.setContentsMargins(14, 10, 14, 10)

        self.preview_title = QtWidgets.QLabel("Realtime Skeleton Preview")
        self.preview_title.setStyleSheet("font-size:20px; font-weight:700; color:#e5eefc;")
        self.preview_info = QtWidgets.QLabel("Rotate: drag | Zoom: wheel | Pan: middle-drag")
        self.preview_info.setStyleSheet("color:#a9b8d4; font-size:11px;")
        topbar_layout.addWidget(self.preview_title)
        topbar_layout.addStretch(1)
        topbar_layout.addWidget(self.preview_info)
        parent_layout.addWidget(topbar)

        live_card = QtWidgets.QFrame()
        live_card.setObjectName("card")
        live_layout = QtWidgets.QVBoxLayout(live_card)
        live_layout.setContentsMargins(18, 18, 18, 18)

        self.big_label = QtWidgets.QLabel("-")
        self.big_label.setAlignment(QtCore.Qt.AlignCenter)
        self.big_label.setStyleSheet("font-size:28pt; font-weight:800; color:#e5eefc;")
        live_layout.addWidget(self.big_label)

        self.conf_bar = QtWidgets.QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setValue(0)
        live_layout.addWidget(self.conf_bar)

        self.viewer = Hand3DView()
        live_layout.addWidget(self.viewer, 1)
        parent_layout.addWidget(live_card, 1)

    def _log(self, msg):
        self.log_signal.emit(msg)

    def _append_log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{ts}] {msg}")

    def _push_status(self):
        self.status_signal.emit()

    def _refresh_badge(self):
        status = self.current_state.lower()
        if "predicted" in status:
            bg, fg = self.colors["accent2"], "#04110a"
        elif "cooldown" in status or "debouncing" in status or "smoothing" in status:
            bg, fg = self.colors["warning"], "#1a1104"
        elif "low_conf" in status or "idle" in status:
            bg, fg = self.colors["accent"], "#08111f"
        else:
            bg, fg = self.colors["danger"], "#190606"
        self.badge.setText(self.current_state.upper())
        self.badge.setStyleSheet(f"background:{bg}; color:{fg}; border-radius:12px; padding:8px 12px; font-weight:700;")

    def _update_packet_label(self, count):
        self.packet_status_label.setText(f"Packets: {count}")

    def _refresh_status(self):
        self.state_status_label.setText(self.current_state)
        self.label_status_label.setText(self.current_label)
        self.raw_label_status_label.setText(self.current_raw_label)
        self.conf_status_label.setText(f"{self.current_conf:.3f}")
        self.motion_status_label.setText(f"{self.current_motion:.3f}")
        self.wrist_motion_status_label.setText(f"{self.current_wrist_motion:.3f}")
        self.cooldown_status_label.setText(str(self.cooldown_counter))
        self.big_label.setText(self.current_label)
        self.conf_bar.setValue(int(max(0.0, min(1.0, self.current_conf)) * 100))

        self.summary_value_labels["Connection"].setText(self.connection_status_label.text())
        self.summary_value_labels["Model"].setText(self.model_status_label.text())
        self.summary_value_labels["State"].setText(self.current_state)
        self.summary_value_labels["Label"].setText(self.current_label)

        self.setWindowTitle(
            f"Realtime Gesture ST-GCN - {self.current_label} | conf={self.current_conf:.3f} | {self.current_state}"
        )

    def _append_prediction_history(self, label, conf):
        ts = time.strftime("%H:%M:%S")
        item_text = f"[{ts}] {label} ({conf:.3f})"
        if self.last_logged_prediction == item_text:
            return
        self.last_logged_prediction = item_text
        self.predicted_history.appendleft(item_text)
        self.history_list.clear()
        self.history_list.addItems(list(self.predicted_history))

    def browse_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Model", "", "PyTorch (*.pth);;All Files (*)")
        if path:
            self.model_path_edit.setText(path)

    def browse_scaler(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Scaler", "", "Pickle (*.pkl);;All Files (*)")
        if path:
            self.scaler_path_edit.setText(path)

    def browse_label_json(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Label Names", "", "JSON (*.json);;All Files (*)")
        if path:
            self.label_path_edit.setText(path)

    def load_model_files(self):
        try:
            model_path = Path(self.model_path_edit.text().strip())
            scaler_path = Path(self.scaler_path_edit.text().strip())
            label_path = Path(self.label_path_edit.text().strip())

            for p, name in [(model_path, "Model"), (scaler_path, "Scaler"), (label_path, "Label names")]:
                if not p.exists():
                    raise FileNotFoundError(f"{name} not found: {p}")

            bundle = torch.load(model_path, map_location="cpu")
            self.model = GestureSTGCN(
                num_classes=bundle["num_classes"],
                joint_in_channels=bundle["joint_in_channels"],
                feature_in_dim=bundle["feature_in_dim"],
                hidden_dim=bundle.get("hidden_dim", 64),
                feat_hidden_dim=bundle.get("feat_hidden_dim", 128),
                stgcn_dropout=bundle.get("stgcn_dropout", 0.3),
                feat_dropout=bundle.get("feat_dropout", 0.3),
            ).to(self.device)
            self.model.load_state_dict(bundle["model_state_dict"])
            self.model.eval()

            self.scaler = joblib.load(scaler_path)

            with open(label_path, "r", encoding="utf-8") as f:
                self.label_names = json.load(f)

            self.model_status_label.setText("Model loaded")
            self._log("Model ST-GCN berhasil diload.")
            self._push_status()
        except Exception as e:
            self.model_status_label.setText("Load failed")
            self._log(f"Load error: {e}")
            self._push_status()
            QtWidgets.QMessageBox.critical(self, "Load Model Error", str(e))

    def connect_socket(self):
        if self.connected:
            self._log("Already connected.")
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
            self._push_status()
            self._log(f"Connected ke UDP {host}:{port}")
        except Exception as e:
            self._log(f"Connect failed: {e}")
            QtWidgets.QMessageBox.critical(self, "Connect Error", str(e))

    def disconnect_socket(self):
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
        self._push_status()
        self._log("Disconnected.")

    def _compute_motion(self, joint_frame, pts):
        raw_motion = 0.0
        wrist_motion = 0.0

        if self.prev_joint_frame is not None:
            raw_motion = float(np.mean(np.linalg.norm(joint_frame - self.prev_joint_frame, axis=0)))

        wrist = pts["wrist"]
        if self.prev_wrist_point is not None:
            wrist_motion = float(np.linalg.norm(wrist - self.prev_wrist_point))

        self.prev_joint_frame = joint_frame.copy()
        self.prev_wrist_point = wrist.copy()
        return raw_motion, wrist_motion

    def _reset_to_idle(self, state, clear_buffers=False):
        if clear_buffers:
            self.pred_history.clear()
            self.debounce_buffer.clear()
        self.current_label = "-"
        self.current_state = state
        self._push_status()

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
                pts = extract_right_hand_points(data)
                self.latest_hand_points = pts

                joint_frame_21x3 = points_to_relative_joint_array(pts)
                feat_frame = extract_frame_feature_vector(pts)
                joint_frame = np.transpose(joint_frame_21x3, (1, 0)).astype(np.float32)

                raw_motion, wrist_motion = self._compute_motion(joint_frame, pts)
                self.current_motion = raw_motion
                self.current_wrist_motion = wrist_motion

                self.joint_window.append(joint_frame)
                self.feat_window.append(feat_frame)
            except Exception:
                continue

            try:
                window_size = int(self.window_size_edit.text().strip())
                smooth_preds = int(self.smooth_preds_edit.text().strip())
                conf_threshold = float(self.conf_threshold_edit.text().strip())
                raw_motion_threshold = float(self.raw_motion_threshold_edit.text().strip())
                wrist_motion_threshold = float(self.wrist_motion_threshold_edit.text().strip())
                debounce_frames = int(self.debounce_frames_edit.text().strip())
                cooldown_frames = int(self.cooldown_frames_edit.text().strip())

                if self.joint_window.maxlen != window_size:
                    old = list(self.joint_window)
                    self.joint_window = deque(old[-window_size:], maxlen=window_size)

                if self.feat_window.maxlen != window_size:
                    old = list(self.feat_window)
                    self.feat_window = deque(old[-window_size:], maxlen=window_size)

                if self.pred_history.maxlen != max(1, smooth_preds):
                    old = list(self.pred_history)
                    self.pred_history = deque(old[-smooth_preds:], maxlen=max(1, smooth_preds))

                if self.debounce_buffer.maxlen != max(1, debounce_frames):
                    old = list(self.debounce_buffer)
                    self.debounce_buffer = deque(old[-debounce_frames:], maxlen=max(1, debounce_frames))

                if len(self.joint_window) < window_size:
                    self.current_state = f"buffering {len(self.joint_window)}/{window_size}"
                    self.current_label = "-"
                    self.current_raw_label = "-"
                    self._push_status()
                    continue

                if self.model is None or self.scaler is None or self.label_names is None:
                    self.current_state = "model not loaded"
                    self.current_label = "-"
                    self.current_raw_label = "-"
                    self._push_status()
                    continue

                if raw_motion < raw_motion_threshold and wrist_motion < wrist_motion_threshold:
                    self.debounce_buffer.clear()
                    self.pred_history.clear()
                    self._reset_to_idle("idle_motion", clear_buffers=False)
                    continue

                joint_seq = np.stack(self.joint_window, axis=1)
                feat_seq = np.stack(self.feat_window, axis=0)

                feat_seq = self.scaler.transform(feat_seq)
                feat_seq = np.transpose(feat_seq, (1, 0)).astype(np.float32)

                joint_tensor = torch.tensor(joint_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
                feat_tensor = torch.tensor(feat_seq, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.model(joint_tensor, feat_tensor, self.A)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    pred_idx = int(np.argmax(probs))
                    pred_label = self.label_names[pred_idx]
                    conf = float(np.max(probs))

                self.current_raw_label = pred_label
                self.current_conf = conf

                self._log(
                    f"raw={pred_label} conf={conf:.3f} "
                    f"motion={raw_motion:.4f} wrist={wrist_motion:.4f} "
                    f"debounce={list(self.debounce_buffer)} cooldown={self.cooldown_counter}"
                )

                if conf < conf_threshold:
                    self.current_label = "-"
                    self.current_state = f"low_conf ({conf:.3f})"
                    self._push_status()
                    continue

                self.debounce_buffer.append(pred_label)

                min_debounce_count = max(2, debounce_frames)
                stable_debounce_label = debounced_prediction(
                    self.debounce_buffer,
                    min_count=min_debounce_count,
                )

                if len(self.debounce_buffer) < debounce_frames or stable_debounce_label is None:
                    self.current_label = "-"
                    self.current_state = f"debouncing {len(self.debounce_buffer)}/{debounce_frames}"
                    self._push_status()
                    continue

                self.pred_history.append(stable_debounce_label)
                stable_label = stable_prediction(self.pred_history)

                if stable_label is None:
                    self.current_label = "-"
                    self.current_state = "smoothing"
                    self._push_status()
                    continue

                if self.cooldown_counter > 0:
                    if stable_label == self.current_label:
                        self.current_state = f"cooldown ({self.cooldown_counter})"
                        self.cooldown_counter -= 1
                        self._push_status()
                        continue
                    self.cooldown_counter -= 1

                self.current_label = stable_label
                self.current_state = "predicted"
                self.cooldown_counter = cooldown_frames
                self._append_prediction_history(self.current_label, self.current_conf)
                self._push_status()

            except Exception as e:
                self._log(f"Inference error: {e}")

    def _draw_preview(self):
        if self.latest_hand_points is not None:
            self.viewer.update_hand(self.latest_hand_points)

    def closeEvent(self, event):
        try:
            self.disconnect_socket()
        finally:
            event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = RealtimeGestureSTGCNApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()