import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtGui, QtWidgets


POSE_OPTIONS = [
    "inferiorpincer",
    "palmar",
    "pincer",
    "radialdigital",
    "radialpalmar",
    "rake",
]


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


KEY_HINTS = [
    "A = Record 3 Seconds",
    "S = Start Manual Recording",
    "D = Stop / Cancel Countdown",
    "W = Next Repetition",
]


ACCENT = "#40b8ea"
BG = "#07152b"
CARD = "#172844"
CARD_DARK = "#08152e"
TEXT = "#f4f8ff"
MUTED = "#d7e5ff"
WARN = "#ffb84d"


def try_parse_json(packet: bytes):
    try:
        return json.loads(packet.decode("utf-8"))
    except Exception:
        return None


def vec3(node):
    p = node["position"]
    return np.array([p["x"], p["y"], p["z"]], dtype=np.float64)


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


class HandPreviewWidget(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setCameraPosition(distance=0.8, elevation=18, azimuth=35)
        self.setBackgroundColor(pg.mkColor(CARD_DARK))

        grid = gl.GLGridItem()
        grid.scale(0.05, 0.05, 0.05)
        self.addItem(grid)

        self.scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            size=10,
            color=(0.25, 0.72, 0.92, 1.0),
            pxMode=True,
        )
        self.addItem(self.scatter)

        self.lines = []
        for _ in BONES:
            item = gl.GLLinePlotItem(
                pos=np.zeros((2, 3)),
                color=(0.90, 0.96, 1.0, 1.0),
                width=2,
                antialias=True,
                mode="lines",
            )
            self.addItem(item)
            self.lines.append(item)

    def clear_preview(self):
        self.scatter.setData(pos=np.zeros((1, 3)), size=0)
        for item in self.lines:
            item.setData(pos=np.zeros((2, 3)), width=2, mode="lines")

    def update_points(self, points):
        if points is None:
            self.clear_preview()
            return

        vis_pts = {name: np.array([-p[0], p[1], p[2]], dtype=np.float32) for name, p in points.items()}
        coords = np.array([vis_pts[name] for name in vis_pts], dtype=np.float32)
        self.scatter.setData(pos=coords, size=10, color=(0.25, 0.72, 0.92, 1.0), pxMode=True)

        for item, (a, b) in zip(self.lines, BONES):
            seg = np.array([vis_pts[a], vis_pts[b]], dtype=np.float32)
            item.setData(pos=seg, color=(0.90, 0.96, 1.0, 1.0), width=2, antialias=True, mode="lines")

        center = vis_pts["wrist"]
        self.opts["center"] = pg.Vector(float(center[0]), float(center[1]), float(center[2]))


class RokokoRecorderPreviewApp(QtWidgets.QMainWindow):
    log_signal = QtCore.Signal(str)
    packet_signal = QtCore.Signal()
    receiver_error_signal = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rokoko Recorder + Skeleton Preview")
        self.resize(1380, 820)
        self.setMinimumSize(1180, 720)

        self.sock = None
        self.connected = False
        self.receiver_thread = None
        self.stop_receiver = threading.Event()

        self.is_recording = False
        self.record_mode = "idle"  # idle / countdown / auto / manual
        self.record_start_time = None
        self.record_duration_sec = 3.0
        self.countdown_remaining = 0

        self.current_packets = []
        self.packet_count = 0
        self.recorded_packet_count = 0
        self.current_output_file = None
        self.latest_hand_points = None

        self._build_ui()
        self._apply_styles()
        self._connect_signals()
        self._bind_shortcuts()

        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.timeout.connect(self._update_preview_loop)
        self.preview_timer.start(80)

        self.ui_timer = QtCore.QTimer(self)
        self.ui_timer.timeout.connect(self._update_timer_loop)
        self.ui_timer.start(100)

        self.countdown_timer = QtCore.QTimer(self)
        self.countdown_timer.setInterval(1000)
        self.countdown_timer.timeout.connect(self._countdown_tick)

        self.auto_stop_timer = QtCore.QTimer(self)
        self.auto_stop_timer.setSingleShot(True)
        self.auto_stop_timer.timeout.connect(self._auto_finish_if_needed)

        self._refresh_summary_cards()
        self._refresh_progress(0.0)
        self._log("GUI recorder + preview siap.")
        self._log("Shortcut aktif: A, S, D, W")

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QHBoxLayout(central)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(12)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll_area.setMinimumWidth(460)
        self.scroll_area.setMaximumWidth(520)
        outer.addWidget(self.scroll_area, 0)

        left_container = QtWidgets.QWidget()
        self.scroll_area.setWidget(left_container)
        self.left_layout = QtWidgets.QVBoxLayout(left_container)
        self.left_layout.setSpacing(12)
        self.left_layout.setContentsMargins(0, 0, 6, 0)

        right_container = QtWidgets.QWidget()
        outer.addWidget(right_container, 1)
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_layout.setSpacing(12)
        right_layout.setContentsMargins(0, 0, 0, 0)

        hero = self._card_widget(padding=18)
        hero_layout = hero.layout()
        title = QtWidgets.QLabel("Realtime Dataset Recorder")
        title.setObjectName("heroTitle")
        hero_layout.addWidget(title)
        subtitle = QtWidgets.QLabel("Preview 3D menggunakan PyQtGraph dan kini menyatu dalam satu GUI. Logika recorder tetap dipertahankan.")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("heroSubtitle")
        hero_layout.addWidget(subtitle)
        self.hero_badge = QtWidgets.QLabel("IDLE")
        self.hero_badge.setObjectName("badge")
        hero_layout.addWidget(self.hero_badge, 0, QtCore.Qt.AlignLeft)
        self.left_layout.addWidget(hero)

        summary_row = QtWidgets.QHBoxLayout()
        summary_row.setSpacing(12)
        right_layout.addLayout(summary_row)
        self.summary_labels = {}
        for key, title_text in [("connection", "Connection"), ("recording", "Recording"), ("packets", "Packets"), ("file", "File")]:
            card = self._group_box(title_text)
            card_layout = QtWidgets.QVBoxLayout(card)
            value = QtWidgets.QLabel("-")
            value.setWordWrap(True)
            value.setObjectName("summaryValue")
            card_layout.addWidget(value)
            summary_row.addWidget(card, 1)
            self.summary_labels[key] = value

        conn_box = self._group_box("Connection")
        conn_form = QtWidgets.QGridLayout(conn_box)
        conn_form.setHorizontalSpacing(8)
        conn_form.setVerticalSpacing(10)
        self.host_input = QtWidgets.QLineEdit("127.0.0.1")
        self.port_input = QtWidgets.QLineEdit("14043")
        self.connection_status_label = QtWidgets.QLabel("Disconnected")
        self.connection_status_label.setObjectName("value")
        conn_form.addWidget(QtWidgets.QLabel("Host"), 0, 0)
        conn_form.addWidget(self.host_input, 0, 1)
        conn_form.addWidget(QtWidgets.QLabel("Port"), 1, 0)
        conn_form.addWidget(self.port_input, 1, 1)
        self.connect_button = self._button("Connect", primary=True)
        self.disconnect_button = self._button("Disconnect")
        conn_form.addWidget(self.connect_button, 2, 0)
        conn_form.addWidget(self.disconnect_button, 2, 1)
        conn_form.addWidget(QtWidgets.QLabel("Status"), 3, 0)
        conn_form.addWidget(self.connection_status_label, 3, 1)
        self.left_layout.addWidget(conn_box)

        meta_box = self._group_box("Recording Metadata")
        meta_form = QtWidgets.QGridLayout(meta_box)
        meta_form.setHorizontalSpacing(8)
        meta_form.setVerticalSpacing(10)
        self.pose_combo = QtWidgets.QComboBox()
        self.pose_combo.addItems(POSE_OPTIONS)
        self.subject_input = QtWidgets.QLineEdit("person1")
        self.rep_input = QtWidgets.QLineEdit("01")
        self.save_dir_input = QtWidgets.QLineEdit(str(Path.cwd() / "recordings"))
        self.browse_button = self._button("Browse")
        meta_form.addWidget(QtWidgets.QLabel("Pose"), 0, 0)
        meta_form.addWidget(self.pose_combo, 0, 1)
        meta_form.addWidget(QtWidgets.QLabel("Subject"), 1, 0)
        meta_form.addWidget(self.subject_input, 1, 1)
        meta_form.addWidget(QtWidgets.QLabel("Repetition"), 2, 0)
        meta_form.addWidget(self.rep_input, 2, 1)
        meta_form.addWidget(QtWidgets.QLabel("Save Folder"), 3, 0)
        meta_form.addWidget(self.save_dir_input, 3, 1)
        meta_form.addWidget(self.browse_button, 4, 1, 1, 1, QtCore.Qt.AlignRight)
        self.left_layout.addWidget(meta_box)

        ctrl_box = self._group_box("Controls")
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_box)
        self.auto_button = self._button("Record 3 Seconds (A)", primary=True)
        self.manual_button = self._button("Start Manual Recording (S)")
        self.stop_button = self._button("Stop (D)")
        self.next_rep_button = self._button("Next Repetition (W)")
        self.reset_button = self._button("Reset", warn=True)
        for btn in [self.auto_button, self.manual_button, self.stop_button, self.next_rep_button, self.reset_button]:
            ctrl_layout.addWidget(btn)
        self.left_layout.addWidget(ctrl_box)

        shortcut_box = self._group_box("Keyboard Shortcuts")
        shortcut_layout = QtWidgets.QVBoxLayout(shortcut_box)
        for hint in KEY_HINTS:
            shortcut_layout.addWidget(QtWidgets.QLabel(hint))
        self.left_layout.addWidget(shortcut_box)

        status_box = self._group_box("Status")
        status_form = QtWidgets.QGridLayout(status_box)
        status_form.setHorizontalSpacing(8)
        status_form.setVerticalSpacing(10)
        self.record_status_label = QtWidgets.QLabel("Idle")
        self.packet_status_label = QtWidgets.QLabel("Packets: 0")
        self.timer_status_label = QtWidgets.QLabel("Timer: 0.0 s")
        self.file_status_label = QtWidgets.QLabel("File: -")
        for label in [self.record_status_label, self.packet_status_label, self.timer_status_label, self.file_status_label]:
            label.setObjectName("value")
        status_form.addWidget(QtWidgets.QLabel("Recording"), 0, 0)
        status_form.addWidget(self.record_status_label, 0, 1)
        status_form.addWidget(QtWidgets.QLabel("Packets"), 1, 0)
        status_form.addWidget(self.packet_status_label, 1, 1)
        status_form.addWidget(QtWidgets.QLabel("Timer"), 2, 0)
        status_form.addWidget(self.timer_status_label, 2, 1)
        status_form.addWidget(QtWidgets.QLabel("File"), 3, 0)
        status_form.addWidget(self.file_status_label, 3, 1)
        self.left_layout.addWidget(status_box)

        log_box = self._group_box("Log")
        log_layout = QtWidgets.QVBoxLayout(log_box)
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(240)
        log_layout.addWidget(self.log_text)
        self.left_layout.addWidget(log_box, 1)
        self.left_layout.addStretch(1)

        preview_box = self._group_box("Realtime Skeleton Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_box)
        self.preview_title_label = QtWidgets.QLabel("PyQtGraph 3D Preview")
        self.preview_title_label.setObjectName("previewTitle")
        self.preview_title_label.setAlignment(QtCore.Qt.AlignCenter)
        preview_layout.addWidget(self.preview_title_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        preview_layout.addWidget(self.progress_bar)

        info_label = QtWidgets.QLabel(
            "Preview 3D PyQtGraph kini menyatu di dalam GUI utama. Reset membersihkan state rekaman sementara tanpa memutus koneksi UDP."
        )
        info_label.setWordWrap(True)
        info_label.setObjectName("heroSubtitle")
        preview_layout.addWidget(info_label)

        self.preview_widget = HandPreviewWidget()
        preview_layout.addWidget(self.preview_widget, 1)
        right_layout.addWidget(preview_box, 1)

    def _apply_styles(self):
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{
                background: {BG};
                color: {MUTED};
                font-family: 'Segoe UI';
                font-size: 10pt;
            }}
            QScrollArea {{ border: none; background: {BG}; }}
            QGroupBox {{
                background: {CARD};
                border: 1px solid #203455;
                border-radius: 14px;
                margin-top: 14px;
                padding: 14px;
                font-weight: 700;
                color: {TEXT};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                top: 6px;
                padding: 0 4px;
                color: {TEXT};
            }}
            QLabel#heroTitle {{ font-size: 24pt; font-weight: 800; color: {TEXT}; background: {CARD}; }}
            QLabel#heroSubtitle, QLabel {{ color: {MUTED}; background: transparent; }}
            QLabel#summaryValue, QLabel#value {{ color: white; font-weight: 700; }}
            QLabel#previewTitle {{ font-size: 17pt; font-weight: 800; color: {TEXT}; }}
            QLabel#badge {{
                background: {ACCENT};
                color: {BG};
                border-radius: 14px;
                padding: 7px 16px;
                font-weight: 800;
            }}
            QLineEdit, QComboBox, QPlainTextEdit {{
                background: {CARD_DARK};
                border: 1px solid #284268;
                border-radius: 10px;
                padding: 8px;
                color: white;
            }}
            QComboBox QAbstractItemView {{ background: {CARD_DARK}; color: white; selection-background-color: {ACCENT}; }}
            QPushButton {{
                background: #2a3d5d;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 12px;
                font-weight: 700;
            }}
            QPushButton:hover {{ background: #34527e; }}
            QPushButton[role='primary'] {{ background: {ACCENT}; color: {BG}; }}
            QPushButton[role='primary']:hover {{ background: #63c7f0; }}
            QPushButton[role='warn'] {{ background: {WARN}; color: {BG}; }}
            QPushButton[role='warn']:hover {{ background: #ffc66d; }}
            QProgressBar {{
                background: {CARD};
                border: 1px solid #284268;
                border-radius: 8px;
                text-align: center;
                color: white;
                min-height: 22px;
            }}
            QProgressBar::chunk {{ background: {ACCENT}; border-radius: 7px; }}
            QScrollBar:vertical {{ background: {BG}; width: 12px; margin: 0; }}
            QScrollBar::handle:vertical {{ background: #35547c; min-height: 24px; border-radius: 6px; }}
            """
        )

    def _connect_signals(self):
        self.connect_button.clicked.connect(self.connect_socket)
        self.disconnect_button.clicked.connect(self.disconnect_socket)
        self.browse_button.clicked.connect(self.browse_folder)
        self.auto_button.clicked.connect(self.start_auto_record)
        self.manual_button.clicked.connect(self.start_manual_record)
        self.stop_button.clicked.connect(self.stop_recording)
        self.next_rep_button.clicked.connect(self.next_repetition)
        self.reset_button.clicked.connect(self.reset_state)
        self.log_signal.connect(self._append_log)
        self.packet_signal.connect(self._refresh_packet_status)
        self.receiver_error_signal.connect(self._show_receiver_error)

    def _bind_shortcuts(self):
        self._make_shortcut("A", self.start_auto_record)
        self._make_shortcut("S", self.start_manual_record)
        self._make_shortcut("D", self.stop_recording)
        self._make_shortcut("W", self.next_repetition)

    def _make_shortcut(self, key, handler):
        shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
        shortcut.setContext(QtCore.Qt.ApplicationShortcut)
        shortcut.activated.connect(lambda h=handler: self._handle_shortcut(h))

    def _handle_shortcut(self, handler):
        widget = QtWidgets.QApplication.focusWidget()
        if isinstance(widget, (QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit, QtWidgets.QComboBox)):
            return
        handler()

    def _card_widget(self, padding=14):
        frame = QtWidgets.QFrame()
        frame.setStyleSheet(f"QFrame {{ background: {CARD}; border-radius: 16px; }}")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(padding, padding, padding, padding)
        return frame

    def _group_box(self, title):
        return QtWidgets.QGroupBox(title)

    def _button(self, text, primary=False, warn=False):
        button = QtWidgets.QPushButton(text)
        if primary:
            button.setProperty("role", "primary")
        elif warn:
            button.setProperty("role", "warn")
        return button

    def _refresh_summary_cards(self):
        self.summary_labels["connection"].setText(self.connection_status_label.text())
        self.summary_labels["recording"].setText(self.record_status_label.text())
        self.summary_labels["packets"].setText(self.packet_status_label.text())
        self.summary_labels["file"].setText(self.file_status_label.text().replace("File: ", ""))
        badge = self.record_status_label.text().upper() if self.record_status_label.text() else "IDLE"
        self.hero_badge.setText(badge)
        self.preview_title_label.setText(self.record_status_label.text() or "PyQtGraph 3D Preview")

    def _refresh_progress(self, fraction):
        self.progress_bar.setValue(int(round(max(0.0, min(1.0, fraction)) * 100)))

    def _log(self, msg):
        self.log_signal.emit(msg)

    @QtCore.Slot(str)
    def _append_log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{ts}] {msg}")
        bar = self.log_text.verticalScrollBar()
        bar.setValue(bar.maximum())

    def browse_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Save Folder", self.save_dir_input.text() or str(Path.cwd()))
        if folder:
            self.save_dir_input.setText(folder)

    def connect_socket(self):
        if self.connected:
            self._log("Sudah terhubung.")
            return
        try:
            host = self.host_input.text().strip()
            port = int(self.port_input.text().strip())
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((host, port))
            self.sock.settimeout(0.5)
            self.stop_receiver.clear()
            self.receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
            self.receiver_thread.start()
            self.connected = True
            self.connection_status_label.setText(f"Connected ({host}:{port})")
            self._refresh_summary_cards()
            self._log(f"Connected ke UDP {host}:{port}")
        except Exception as e:
            self._log(f"Gagal connect: {e}")
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
        self._refresh_summary_cards()
        self._log("Disconnected.")

    def _receiver_loop(self):
        while not self.stop_receiver.is_set():
            try:
                packet, _addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as e:
                self.receiver_error_signal.emit(f"Receiver error: {e}")
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

    @QtCore.Slot(str)
    def _show_receiver_error(self, msg):
        self._log(msg)

    @QtCore.Slot()
    def _refresh_packet_status(self):
        if self.is_recording:
            self.packet_status_label.setText(f"Packets: {self.packet_count} | Recorded: {self.recorded_packet_count}")
            self._refresh_progress(min(1.0, self.recorded_packet_count / 120.0))
        else:
            self.packet_status_label.setText(f"Packets: {self.packet_count}")
        self._refresh_summary_cards()

    def build_output_path(self):
        save_dir = Path(self.save_dir_input.text().strip())
        save_dir.mkdir(parents=True, exist_ok=True)
        pose = self.pose_combo.currentText().strip()
        subject = self.subject_input.text().strip()
        rep = self.rep_input.text().strip().zfill(2)
        return save_dir / f"{pose}_{subject}_rep{rep}.jsonl"

    def start_auto_record(self):
        if not self.connected:
            QtWidgets.QMessageBox.warning(self, "Warning", "Connect dulu.")
            return
        if self.is_recording or self.record_mode == "countdown":
            self._log("Sedang recording/countdown.")
            return
        self.current_output_file = self.build_output_path()
        self.file_status_label.setText(f"File: {self.current_output_file.name}")
        self.record_mode = "countdown"
        self.countdown_remaining = 0
        self.record_status_label.setText("Countdown")
        self._refresh_summary_cards()
        self._log(f"Siap auto-record ke {self.current_output_file.name}")
        self._countdown_step_display()
        self.countdown_timer.start()

    def _countdown_step_display(self):
        if self.record_mode != "countdown":
            return
        if self.countdown_remaining > 0:
            self.record_status_label.setText(f"Countdown: {self.countdown_remaining}")
            self.timer_status_label.setText(f"Timer: {self.countdown_remaining}")
            self._refresh_progress((3 - self.countdown_remaining) / 3 if self.countdown_remaining <= 3 else 0.0)
            self._refresh_summary_cards()
            self._log(f"Countdown {self.countdown_remaining}...")
        else:
            self._start_record_common(mode="auto")
            self.auto_stop_timer.start(int(self.record_duration_sec * 1000))

    def _countdown_tick(self):
        if self.record_mode != "countdown":
            self.countdown_timer.stop()
            return
        self.countdown_remaining -= 1
        if self.countdown_remaining <= 0:
            self.countdown_timer.stop()
        self._countdown_step_display()

    def start_manual_record(self):
        if not self.connected:
            QtWidgets.QMessageBox.warning(self, "Warning", "Connect dulu.")
            return
        if self.is_recording or self.record_mode == "countdown":
            self._log("Sedang recording/countdown.")
            return
        self.current_output_file = self.build_output_path()
        self.file_status_label.setText(f"File: {self.current_output_file.name}")
        self._refresh_summary_cards()
        self._start_record_common(mode="manual")

    def _start_record_common(self, mode):
        self.current_packets = []
        self.recorded_packet_count = 0
        self.record_start_time = time.time()
        self.is_recording = True
        self.record_mode = mode
        self._refresh_packet_status()
        if mode == "auto":
            self.record_status_label.setText("Recording Auto")
            self._log("Auto recording dimulai.")
        else:
            self.record_status_label.setText("Recording Manual")
            self._log("Manual recording dimulai.")
        self._refresh_summary_cards()

    def _update_timer_loop(self):
        if not self.is_recording:
            return
        elapsed = time.time() - self.record_start_time
        self.timer_status_label.setText(f"Timer: {elapsed:.1f} s")
        progress = min(1.0, elapsed / self.record_duration_sec) if self.record_mode == "auto" else min(1.0, self.recorded_packet_count / 180.0)
        self._refresh_progress(progress)
        self._refresh_summary_cards()

    def _auto_finish_if_needed(self):
        if self.is_recording and self.record_mode == "auto":
            self.stop_recording(auto_finished=True)

    def stop_recording(self, auto_finished=False):
        if not self.is_recording and self.record_mode != "countdown":
            return
        if self.record_mode == "countdown":
            self.countdown_timer.stop()
            self.record_mode = "idle"
            self.record_status_label.setText("Idle")
            self.timer_status_label.setText("Timer: 0.0 s")
            self._refresh_summary_cards()
            self._refresh_progress(0.0)
            self._log("Countdown dibatalkan.")
            return
        self.auto_stop_timer.stop()
        self.is_recording = False
        old_mode = self.record_mode
        self.record_mode = "idle"
        elapsed = time.time() - self.record_start_time if self.record_start_time else 0.0
        self.timer_status_label.setText(f"Timer: {elapsed:.1f} s")
        if not self.current_output_file:
            self.record_status_label.setText("Idle")
            self._refresh_summary_cards()
            self._refresh_progress(0.0)
            return
        try:
            self._save_jsonl(self.current_output_file, self.current_packets)
            self.record_status_label.setText("Saved")
            self._refresh_summary_cards()
            self._refresh_progress(1.0)
            self._log(f"Recording selesai ({old_mode}). Durasi={elapsed:.2f}s, packets={len(self.current_packets)}, saved={self.current_output_file}")
            if auto_finished:
                self.next_repetition()
        except Exception as e:
            self.record_status_label.setText("Save Error")
            self._refresh_summary_cards()
            self._log(f"Gagal simpan file: {e}")
            QtWidgets.QMessageBox.critical(self, "Save Error", str(e))

    def reset_state(self):
        was_recording = self.is_recording or self.record_mode == "countdown"
        self.countdown_timer.stop()
        self.auto_stop_timer.stop()
        self.is_recording = False
        self.record_mode = "idle"
        self.record_start_time = None
        self.current_packets = []
        self.recorded_packet_count = 0
        self.current_output_file = None
        self.latest_hand_points = None
        self.record_status_label.setText("Idle")
        self.timer_status_label.setText("Timer: 0.0 s")
        self.file_status_label.setText("File: -")
        self.packet_status_label.setText(f"Packets: {self.packet_count}")
        self._refresh_summary_cards()
        self._refresh_progress(0.0)
        self.preview_widget.clear_preview()
        self._log("State di-reset. Data sementara dibersihkan.")
        if was_recording:
            self._log("Recording aktif dibatalkan oleh Reset.")

    def _save_jsonl(self, path: Path, packets):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for item in packets:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def next_repetition(self):
        current = self.rep_input.text().strip()
        try:
            num = int(current)
        except ValueError:
            num = 1
        num += 1
        self.rep_input.setText(str(num).zfill(2))
        self._refresh_summary_cards()
        self._log(f"Repetition pindah ke {self.rep_input.text()}")

    def _update_preview_loop(self):
        if self.latest_hand_points is not None:
            self.preview_widget.update_points(self.latest_hand_points)

    def closeEvent(self, event):
        try:
            self.disconnect_socket()
        finally:
            event.accept()


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = RokokoRecorderPreviewApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
