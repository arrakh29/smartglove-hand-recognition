
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl

from gesture_stgcn_common import BONES, LABEL_OPTIONS, extract_right_hand_points, try_parse_json


GESTURE_HELP = {
    "up": "move the cube upward in a straight vertical direction",
    "down": "move the cube downward in a straight vertical direction",
    "left": "move the cube left in a straight horizontal direction",
    "right": "move the cube right in a straight horizontal direction",
    "rotate_clockwise": "rotate the cube 90 degrees clockwise",
    "rotate_counterclockwise": "rotate the cube 90 degrees counterclockwise",
    "idle": "neutral / no-gesture state; hand is resting, waiting, or making small non-command movements",
}


class Hand3DView(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.setBackgroundColor("#081120")
        self.setCameraPosition(distance=0.6, elevation=18, azimuth=35)

        grid = gl.GLGridItem()
        grid.scale(0.1, 0.1, 0.1)
        grid.setColor((0.30, 0.38, 0.50, 0.65))
        self.addItem(grid)

        axis = gl.GLAxisItem()
        axis.setSize(0.08, 0.08, 0.08)
        self.addItem(axis)

        self.scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=float),
            size=12,
            color=(0.22, 0.74, 0.97, 1.0),
            pxMode=True,
        )
        self.addItem(self.scatter)

        self.lines = []
        for _ in BONES:
            line = gl.GLLinePlotItem(
                pos=np.zeros((2, 3), dtype=float),
                width=3,
                color=(0.38, 0.65, 0.98, 1.0),
                antialias=True,
                mode="lines",
            )
            self.lines.append(line)
            self.addItem(line)

    def _visualize_point(self, p):
        return np.array([-p[0], p[1], p[2]], dtype=float)

    def update_hand(self, pts):
        if not pts:
            return

        keys = list(pts.keys())
        pos = np.array([self._visualize_point(pts[k]) for k in keys], dtype=float)
        self.scatter.setData(pos=pos, size=12)

        for line_item, (a, b) in zip(self.lines, BONES):
            pa = self._visualize_point(pts[a])
            pb = self._visualize_point(pts[b])
            line_item.setData(pos=np.array([pa, pb], dtype=float))

        center = self._visualize_point(pts["wrist"])
        self.opts["center"] = QtGui.QVector3D(float(center[0]), float(center[1]), float(center[2]))

        spread = np.ptp(pos, axis=0)
        radius = max(0.12, float(np.max(spread)) * 1.8)
        self.setCameraPosition(distance=radius * 3.4, elevation=self.opts["elevation"], azimuth=self.opts["azimuth"])


class RecorderWindow(QtWidgets.QMainWindow):
    log_signal = QtCore.Signal(str)
    packet_signal = QtCore.Signal()
    status_signal = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rokoko Recorder - Full PySide6")
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
            "plotbg": "#081120",
        }

        self.sock = None
        self.connected = False
        self.receiver_thread = None
        self.stop_receiver = threading.Event()

        self.record_mode = "idle"
        self.is_recording = False
        self.record_start_time = None

        self.packet_count = 0
        self.recorded_packet_count = 0
        self.current_packets = []
        self.current_output_file = None
        self.latest_hand_points = None

        self._build_ui()
        self._connect_signals()

        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.timeout.connect(self._update_preview)
        self.preview_timer.start(30)

        self.status_timer = QtCore.QTimer(self)
        self.status_timer.timeout.connect(self._refresh_status_badge)
        self.status_timer.start(250)

        self.log("Full PySide6 recorder is ready.")

    def _connect_signals(self):
        self.log_signal.connect(self._append_log)
        self.packet_signal.connect(self._refresh_packet_status)
        self.status_signal.connect(self._refresh_summary_cards)

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
            QLineEdit, QComboBox {{
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
            QPlainTextEdit {{
                background: #0b1220;
                color: #dbeafe;
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 8px;
                font-family: Consolas, monospace;
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
        self._apply_styles()

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
        title = QtWidgets.QLabel("Gesture Dataset Recorder")
        title.setObjectName("heroTitle")
        subtitle = QtWidgets.QLabel("Full PySide6 one-app layout with native Qt scrolling and pyqtgraph 3D preview.")
        subtitle.setObjectName("heroSub")
        self.badge = QtWidgets.QLabel("IDLE")
        self.badge.setAlignment(QtCore.Qt.AlignCenter)
        self.badge.setStyleSheet(f"background:{self.colors['accent']}; color:#08111f; border-radius:12px; padding:8px 12px; font-weight:700;")
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
        label.setMinimumWidth(110)
        row.addWidget(label)
        row.addWidget(widget, 1)
        return row

    def _build_left_cards(self):
        # State vars as widgets
        self.host_edit = QtWidgets.QLineEdit("127.0.0.1")
        self.port_edit = QtWidgets.QLineEdit("14043")
        self.gesture_combo = QtWidgets.QComboBox()
        self.gesture_combo.addItems(LABEL_OPTIONS)
        self.subject_edit = QtWidgets.QLineEdit("person1")
        self.rep_edit = QtWidgets.QLineEdit("01")
        self.duration_edit = QtWidgets.QLineEdit("3.0")
        self.save_dir_edit = QtWidgets.QLineEdit(str(Path.cwd() / "recordings_gesture"))

        self.connection_status_label = QtWidgets.QLabel("Disconnected")
        self.connection_status_label.setObjectName("value")
        self.record_status_label = QtWidgets.QLabel("Idle")
        self.record_status_label.setObjectName("value")
        self.packet_status_label = QtWidgets.QLabel("Packets: 0")
        self.packet_status_label.setObjectName("value")
        self.timer_status_label = QtWidgets.QLabel("Timer: 0.0 s")
        self.timer_status_label.setObjectName("value")
        self.file_status_label = QtWidgets.QLabel("File: -")
        self.file_status_label.setWordWrap(True)

        self.gesture_help_label = QtWidgets.QLabel(GESTURE_HELP[self.gesture_combo.currentText()])
        self.gesture_help_label.setWordWrap(True)
        self.gesture_help_label.setStyleSheet(f"background:{self.colors['card2']}; border:1px solid #3b4d63; border-radius:10px; padding:10px;")

        # Connection card
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

        # Metadata
        card, layout = self._make_card("Gesture Metadata")
        layout.addLayout(self._form_row("Gesture", self.gesture_combo))
        layout.addWidget(self.gesture_help_label)
        layout.addLayout(self._form_row("Subject", self.subject_edit))
        layout.addLayout(self._form_row("Repetition", self.rep_edit))
        layout.addLayout(self._form_row("Duration (sec)", self.duration_edit))

        save_row = QtWidgets.QHBoxLayout()
        save_label = QtWidgets.QLabel("Save Folder")
        save_label.setObjectName("muted")
        save_label.setMinimumWidth(110)
        self.browse_btn = QtWidgets.QPushButton("Browse")
        save_row.addWidget(save_label)
        save_row.addWidget(self.save_dir_edit, 1)
        save_row.addWidget(self.browse_btn)
        layout.addLayout(save_row)
        self.left_content_layout.addWidget(card)

        # Reference
        card, layout = self._make_card("Gesture Reference")
        for key in LABEL_OPTIONS:
            item = QtWidgets.QFrame()
            item.setStyleSheet(f"background:{self.colors['card2']}; border:1px solid #334155; border-radius:10px;")
            item_layout = QtWidgets.QHBoxLayout(item)
            item_layout.setContentsMargins(8, 8, 8, 8)
            badge = QtWidgets.QLabel(key.upper())
            badge.setStyleSheet(f"background:{self.colors['accent']}; color:#07111d; border-radius:8px; padding:4px 8px; font-weight:700;")
            desc = QtWidgets.QLabel(GESTURE_HELP[key])
            desc.setWordWrap(True)
            item_layout.addWidget(badge, 0, QtCore.Qt.AlignTop)
            item_layout.addWidget(desc, 1)
            layout.addWidget(item)
        self.left_content_layout.addWidget(card)

        # Controls
        card, layout = self._make_card("Controls")
        self.auto_btn = QtWidgets.QPushButton("Record + Countdown")
        self.auto_btn.setObjectName("accent")
        self.manual_btn = QtWidgets.QPushButton("Start Manual [A]")
        self.stop_btn = QtWidgets.QPushButton("Stop [S]")
        self.reset_btn = QtWidgets.QPushButton("Reset Wrong Take [W]")
        self.next_btn = QtWidgets.QPushButton("Next Repetition [D]")
        self.manual_btn.setToolTip("Shortcut: A")
        self.stop_btn.setToolTip("Shortcut: S")
        self.reset_btn.setToolTip("Shortcut: W")
        self.next_btn.setToolTip("Shortcut: D")
        self.auto_btn.setToolTip("Start auto recording with countdown")
        for btn in [self.auto_btn, self.manual_btn, self.stop_btn, self.reset_btn, self.next_btn]:
            layout.addWidget(btn)
        self.left_content_layout.addWidget(card)

        # Status
        card, layout = self._make_card("Status")
        layout.addLayout(self._form_row("Recording", self.record_status_label))
        layout.addLayout(self._form_row("Packets", self.packet_status_label))
        layout.addLayout(self._form_row("Timer", self.timer_status_label))
        layout.addLayout(self._form_row("File", self.file_status_label))
        self.left_content_layout.addWidget(card)

        # Log
        card, layout = self._make_card("Log")
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        self.left_content_layout.addWidget(card, 1)
        self.left_content_layout.addStretch(1)

        # Connect UI events
        self.connect_btn.clicked.connect(self.connect_socket)
        self.disconnect_btn.clicked.connect(self.disconnect_socket)
        self.browse_btn.clicked.connect(self.browse_folder)
        self.auto_btn.clicked.connect(self.start_auto_record)
        self.manual_btn.clicked.connect(self.start_manual_record)
        self.stop_btn.clicked.connect(self.stop_recording)
        self.reset_btn.clicked.connect(self.reset_wrong_take)
        self.next_btn.clicked.connect(self.next_repetition)
        self.gesture_combo.currentTextChanged.connect(self._on_gesture_changed)

        self.shortcut_manual = QtGui.QShortcut(QtGui.QKeySequence("A"), self)
        self.shortcut_manual.activated.connect(self.start_manual_record)
        self.shortcut_stop = QtGui.QShortcut(QtGui.QKeySequence("S"), self)
        self.shortcut_stop.activated.connect(self.stop_recording)
        self.shortcut_reset = QtGui.QShortcut(QtGui.QKeySequence("W"), self)
        self.shortcut_reset.activated.connect(self.reset_wrong_take)
        self.shortcut_next = QtGui.QShortcut(QtGui.QKeySequence("D"), self)
        self.shortcut_next.activated.connect(self.next_repetition)

    def _build_right_panel(self, parent_layout):
        summary = QtWidgets.QHBoxLayout()
        summary.setSpacing(10)
        self.summary_value_labels = {}

        for title in ["Connection", "Recording", "Packets", "Timer"]:
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

        self.preview_title = QtWidgets.QLabel("Preview - -")
        self.preview_title.setStyleSheet("font-size:20px; font-weight:700; color:#e5eefc;")
        self.preview_info = QtWidgets.QLabel("Rotate: drag | Zoom: wheel | Pan: middle-drag")
        self.preview_info.setStyleSheet("color:#a9b8d4; font-size:11px;")
        topbar_layout.addWidget(self.preview_title)
        topbar_layout.addStretch(1)
        topbar_layout.addWidget(self.preview_info)
        parent_layout.addWidget(topbar)

        self.subject_rep_label = QtWidgets.QLabel("Subject: -   Rep: -")
        self.subject_rep_label.setStyleSheet("color:#93c5fd; font-size:13px; font-weight:600; padding-left:4px;")
        parent_layout.addWidget(self.subject_rep_label)

        preview_card = QtWidgets.QFrame()
        preview_card.setObjectName("card")
        preview_layout = QtWidgets.QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(8, 8, 8, 8)

        self.viewer = Hand3DView()
        preview_layout.addWidget(self.viewer)
        parent_layout.addWidget(preview_card, 1)

    def log(self, msg):
        self.log_signal.emit(msg)

    def _append_log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{ts}] {msg}")

    def _refresh_summary_cards(self):
        self.summary_value_labels["Connection"].setText(self.connection_status_label.text())
        self.summary_value_labels["Recording"].setText(self.record_status_label.text())
        self.summary_value_labels["Packets"].setText(self.packet_status_label.text())
        self.summary_value_labels["Timer"].setText(self.timer_status_label.text())

    def _refresh_status_badge(self):
        status = self.record_status_label.text().lower()
        if "recording" in status:
            bg, fg = self.colors["accent2"], "#04110a"
        elif "countdown" in status:
            bg, fg = self.colors["warning"], "#1a1104"
        elif "error" in status:
            bg, fg = self.colors["danger"], "#190606"
        else:
            bg, fg = self.colors["accent"], "#08111f"
        self.badge.setText(self.record_status_label.text().upper())
        self.badge.setStyleSheet(f"background:{bg}; color:{fg}; border-radius:12px; padding:8px 12px; font-weight:700;")

    def _on_gesture_changed(self):
        self.gesture_help_label.setText(GESTURE_HELP.get(self.gesture_combo.currentText(), ""))
        self.preview_title.setText(f"Preview - {self.gesture_combo.currentText()}")

    def browse_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Save Folder", self.save_dir_edit.text())
        if folder:
            self.save_dir_edit.setText(folder)

    def build_output_path(self):
        save_dir = Path(self.save_dir_edit.text().strip())
        save_dir.mkdir(parents=True, exist_ok=True)
        gesture = self.gesture_combo.currentText().strip()
        subject = self.subject_edit.text().strip()
        rep = self.rep_edit.text().strip().zfill(2)
        filename = f"{gesture}_{subject}_rep{rep}.jsonl"
        return save_dir / filename

    def connect_socket(self):
        if self.connected:
            self.log("Already connected.")
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
            self.status_signal.emit()
            self.log(f"Connected to UDP {host}:{port}")
        except Exception as e:
            self.log(f"Connect failed: {e}")
            QtWidgets.QMessageBox.critical(self, "Connect Error", str(e))

    def disconnect_socket(self):
        if not self.connected:
            return
        self.stop_recording()
        self.stop_receiver.set()
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass
        self.sock = None
        self.connected = False
        self.connection_status_label.setText("Disconnected")
        self.status_signal.emit()
        self.log("Disconnected.")

    def _receiver_loop(self):
        while not self.stop_receiver.is_set():
            try:
                packet, _addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as e:
                QtCore.QMetaObject.invokeMethod(self, lambda e=e: self.log(f"Receiver error: {e}"), QtCore.Qt.QueuedConnection)
                continue

            data = try_parse_json(packet)
            if data is None:
                continue

            self.packet_count += 1
            self.packet_signal.emit()

            try:
                self.latest_hand_points = extract_right_hand_points(data)
            except Exception:
                pass

            if self.is_recording:
                self.current_packets.append(data)
                self.recorded_packet_count += 1
                self.packet_signal.emit()

    def _refresh_packet_status(self):
        if self.is_recording:
            self.packet_status_label.setText(f"Packets: {self.packet_count} | Recorded: {self.recorded_packet_count}")
        else:
            self.packet_status_label.setText(f"Packets: {self.packet_count}")
        self.status_signal.emit()

    def start_auto_record(self):
        if not self.connected:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please connect first.")
            return
        if self.is_recording or self.record_mode == "countdown":
            self.log("Recording or countdown is already running.")
            return
        self.current_output_file = self.build_output_path()
        self.file_status_label.setText(self.current_output_file.name)
        self.record_mode = "countdown"
        self.record_status_label.setText("Countdown")
        self.status_signal.emit()
        self.log(f"Ready to record {self.gesture_combo.currentText()} into {self.current_output_file.name}")
        self._countdown_step(3)

    def _countdown_step(self, remaining):
        if self.record_mode != "countdown":
            return
        if remaining > 0:
            self.record_status_label.setText(f"Countdown: {remaining}")
            self.timer_status_label.setText(f"Timer: {remaining}")
            self.status_signal.emit()
            self.log(f"Countdown {remaining}...")
            QtCore.QTimer.singleShot(1000, lambda: self._countdown_step(remaining - 1))
        else:
            self._start_record_common(mode="auto")
            duration_ms = int(float(self.duration_edit.text().strip()) * 1000)
            QtCore.QTimer.singleShot(duration_ms, self._auto_finish_if_needed)

    def start_manual_record(self):
        if not self.connected:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please connect first.")
            return
        if self.is_recording or self.record_mode == "countdown":
            self.log("Recording or countdown is already running.")
            return
        self.current_output_file = self.build_output_path()
        self.file_status_label.setText(self.current_output_file.name)
        self._start_record_common(mode="manual")

    def _start_record_common(self, mode):
        try:
            float(self.duration_edit.text().strip())
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Invalid Duration", "Duration must be a valid number.")
            self.record_mode = "idle"
            return

        self.current_packets = []
        self.recorded_packet_count = 0
        self.record_start_time = time.time()
        self.is_recording = True
        self.record_mode = mode
        self._refresh_packet_status()

        if mode == "auto":
            self.record_status_label.setText("Recording Auto")
            self.log("Auto recording started.")
        else:
            self.record_status_label.setText("Recording Manual")
            self.log("Manual recording started.")

        self.status_signal.emit()

    def _auto_finish_if_needed(self):
        if self.is_recording and self.record_mode == "auto":
            self.stop_recording(auto_finished=True)

    def stop_recording(self, auto_finished=False):
        if not self.is_recording and self.record_mode != "countdown":
            return

        if self.record_mode == "countdown":
            self.record_mode = "idle"
            self.record_status_label.setText("Idle")
            self.timer_status_label.setText("Timer: 0.0 s")
            self.status_signal.emit()
            self.log("Countdown canceled.")
            return

        self.is_recording = False
        old_mode = self.record_mode
        self.record_mode = "idle"

        elapsed = time.time() - self.record_start_time if self.record_start_time else 0.0
        self.timer_status_label.setText(f"Timer: {elapsed:.1f} s")

        if not self.current_output_file:
            self.record_status_label.setText("Idle")
            self.status_signal.emit()
            return

        try:
            self._save_jsonl(self.current_output_file, self.current_packets)
            self.record_status_label.setText("Saved")
            self.status_signal.emit()
            self.log(
                f"Recording finished ({old_mode}). Duration={elapsed:.2f}s, "
                f"packets={len(self.current_packets)}, saved={self.current_output_file}"
            )
            if auto_finished:
                self.next_repetition()
        except Exception as e:
            self.record_status_label.setText("Save Error")
            self.status_signal.emit()
            self.log(f"Failed to save file: {e}")
            QtWidgets.QMessageBox.critical(self, "Save Error", str(e))

    def _save_jsonl(self, path: Path, packets):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for item in packets:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


    def reset_wrong_take(self):
        if self.record_mode == "countdown":
            self.record_mode = "idle"
            self.is_recording = False
            self.current_packets = []
            self.recorded_packet_count = 0
            self.record_start_time = None
            self.record_status_label.setText("Reset")
            self.timer_status_label.setText("Timer: 0.0 s")
            self.file_status_label.setText("File: -")
            self._refresh_packet_status()
            self.status_signal.emit()
            self.log("Wrong take reset during countdown.")
            return

        if self.is_recording:
            self.is_recording = False
            self.record_mode = "idle"
            discarded = len(self.current_packets)
            self.current_packets = []
            self.recorded_packet_count = 0
            self.record_start_time = None
            self.record_status_label.setText("Reset")
            self.timer_status_label.setText("Timer: 0.0 s")
            self.file_status_label.setText("File: -")
            self._refresh_packet_status()
            self.status_signal.emit()
            self.log(f"Wrong take discarded. Removed {discarded} captured packets.")
            return

        self.log("Reset ignored because no recording is active.")

    def next_repetition(self):
        current = self.rep_edit.text().strip()
        try:
            num = int(current)
        except ValueError:
            num = 1
        num += 1
        self.rep_edit.setText(str(num).zfill(2))
        self.log(f"Repetition advanced to {self.rep_edit.text()}")

    def _update_preview(self):
        if self.is_recording and self.record_start_time is not None:
            elapsed = time.time() - self.record_start_time
            self.timer_status_label.setText(f"Timer: {elapsed:.1f} s")
            self.status_signal.emit()

        self.preview_title.setText(f"Preview - {self.gesture_combo.currentText()}")
        self.subject_rep_label.setText(f"Subject: {self.subject_edit.text()}   Rep: {self.rep_edit.text()}")
        if self.latest_hand_points is not None:
            self.viewer.update_hand(self.latest_hand_points)

    def closeEvent(self, event):
        try:
            self.disconnect_socket()
        finally:
            event.accept()


def main():
    app = QtWidgets.QApplication([])
    window = RecorderWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
