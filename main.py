import sys
import os
import cv2
import numpy as np
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QScrollArea,
    QFileDialog, QStatusBar, QSizePolicy,
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QSplitter, QButtonGroup,
    QDialog, QGroupBox, QFormLayout, QSlider, QComboBox,
    QDialogButtonBox, QFrame, QSpinBox, QDoubleSpinBox, QGridLayout
)
from PyQt6.QtGui import (
    QPixmap, QIcon, QAction, QKeySequence, QPainter, QColor,
    QImage, QPen, QBrush, QFont
)
from PyQt6.QtCore import Qt, QSize, QRect, pyqtSignal

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")


def load_settings() -> "SegmentationSettings":
    try:
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
        s = SegmentationSettings()
        for k, v in data.items():
            if hasattr(s, k):
                setattr(s, k, type(getattr(s, k))(v))
        return s
    except (FileNotFoundError, json.JSONDecodeError, Exception):
        return SegmentationSettings()


def save_settings(s: "SegmentationSettings"):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(asdict(s), f, indent=2)
    except Exception:
        pass


# ── Palette ────────────────────────────────────────────────────────────────
C_BG          = "#f9f6f1"
C_SIDEBAR     = "#f0ece4"
C_SIDEBAR_HDR = "#e8e2d8"
C_BORDER      = "#ddd8cf"
C_TEXT        = "#2d2a25"
C_TEXT_MUTED  = "#8a8078"
C_SELECTED    = "#c8956c"
C_SELECTED_BG = "#f0e4d8"
C_HOVER       = "#e8e2d8"
C_BTN_ADD     = "#c8956c"
C_BTN_ADD_H   = "#b8855c"
C_BTN_REM     = "#b5705a"
C_BTN_REM_H   = "#a0604a"
C_FILTER_BAR  = "#edeae3"
C_FILTER_ACT  = "#c8956c"
C_FILTER_ACT_BG = "#fdf0e6"

THUMB_SIZE    = 64
SIDEBAR_WIDTH = 220
LAYERS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "layers")
ICON_SIZE     = 26


# ── OpenCV filter functions ────────────────────────────────────────────────

def cv_to_pixmap(img):
    if img is None:
        return QPixmap()
    if len(img.shape) == 2:
        h, w = img.shape
        qimg = QImage(img.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
    else:
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def filter_original(img):
    return img


def filter_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def filter_otsu(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def filter_adaptive(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    result = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def filter_clahe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    result = clahe.apply(gray)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def filter_denoise(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.medianBlur(gray, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    result = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def filter_deskew(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    result = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


_active_char_rects   = []    # set by MainWindow before calling filters
_active_seg_settings = None  # set by MainWindow before calling filters


def filter_isolate_chars(img):
    if not _active_char_rects:
        return img
    result = np.full_like(img, 255)
    for (x, y, w, h) in _active_char_rects:
        y2, x2 = min(y + h, img.shape[0]), min(x + w, img.shape[1])
        result[y:y2, x:x2] = img[y:y2, x:x2]
    return result


def filter_remove_chars(img):
    if not _active_char_rects:
        return img
    result = img.copy()
    for (x, y, w, h) in _active_char_rects:
        y2, x2 = min(y + h, img.shape[0]), min(x + w, img.shape[1])
        result[y:y2, x:x2] = 255
    return result


def filter_binarize(img):
    """Binarize using the method and threshold from Detection Settings."""
    s = _active_seg_settings
    if s is None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        if s.threshold_method == "manual":
            _, result = cv2.threshold(gray, s.manual_threshold, 255, cv2.THRESH_BINARY)
        elif s.threshold_method == "adaptive":
            result = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        else:
            _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


# ── Segmentation settings ─────────────────────────────────────────────────

@dataclass
class SegmentationSettings:
    threshold_method: str = "otsu"   # "otsu", "adaptive", "manual"
    manual_threshold: int = 128
    line_kernel_width: int = 40
    line_threshold_pct: int = 5      # percent 1-30
    min_area: int = 20
    max_area: int = 5000
    min_width: int = 2
    min_height: int = 2
    max_width: int = 25
    max_height: int = 25
    connectivity: int = 8


# ── Segmentation functions ────────────────────────────────────────────────

def _to_binary(img, s: SegmentationSettings):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    if s.threshold_method == "manual":
        _, binary = cv2.threshold(gray, s.manual_threshold, 255, cv2.THRESH_BINARY_INV)
    elif s.threshold_method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def detect_lines(img, s: SegmentationSettings):
    binary = _to_binary(img, s)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (s.line_kernel_width, 1))
    dilated = cv2.dilate(binary, kernel)
    h_proj = np.sum(dilated, axis=1)
    threshold = np.max(h_proj) * (s.line_threshold_pct / 100)

    rects, start = [], None
    for i, val in enumerate(h_proj):
        if val > threshold and start is None:
            start = i
        elif val <= threshold and start is not None:
            if i - start > 5:
                rects.append((0, start, img.shape[1], i - start))
            start = None
    if start is not None:
        rects.append((0, start, img.shape[1], img.shape[0] - start))
    return rects


def detect_characters(img, line_rects, s: SegmentationSettings):
    binary = _to_binary(img, s)
    rects = []
    for (lx, ly, lw, lh) in line_rects:
        region = binary[ly:ly + lh, lx:lx + lw]
        n, _, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=s.connectivity)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if (s.min_area <= area <= s.max_area and
                    s.min_width <= w <= s.max_width and
                    s.min_height <= h <= s.max_height):
                rects.append((lx + x, ly + y, w, h))
    return rects


def draw_overlays(img, char_rects, show_chars):
    out = img.copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if show_chars:
        for (x, y, w, h) in char_rects:
            cv2.rectangle(out, (x, y), (x + w, y + h), (30, 100, 220), 1)
    return out


# ── Icon drawing ───────────────────────────────────────────────────────────

def _make_icon(draw_fn):
    px = QPixmap(ICON_SIZE, ICON_SIZE)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    draw_fn(p, ICON_SIZE)
    p.end()
    return px


def icon_original(p, s):
    # Colored RGB squares
    for i, color in enumerate(["#e05c5c", "#5cb85c", "#5c8fe0"]):
        p.fillRect(QRect(2 + i * 8, 4, 7, 18), QColor(color))


def icon_grayscale(p, s):
    # Gradient bar
    for i in range(s - 4):
        v = int(20 + (i / (s - 4)) * 215)
        p.fillRect(QRect(2 + i, 4, 1, s - 8), QColor(v, v, v))


def icon_otsu(p, s):
    # Left half black, right half white, thin border
    p.fillRect(QRect(2, 4, s // 2 - 2, s - 8), QColor("#222"))
    p.fillRect(QRect(s // 2, 4, s // 2 - 2, s - 8), QColor("#eee"))
    p.setPen(QPen(QColor(C_BORDER), 1))
    p.drawRect(QRect(2, 4, s - 4, s - 8))


def icon_adaptive(p, s):
    # 2x2 checker
    colors = ["#222", "#eee", "#eee", "#222"]
    hw = (s - 4) // 2
    positions = [(2, 4), (2 + hw, 4), (2, 4 + hw), (2 + hw, 4 + hw)]
    for (x, y), c in zip(positions, colors):
        p.fillRect(QRect(x, y, hw, hw), QColor(c))


def icon_clahe(p, s):
    # Rising histogram bars
    heights = [6, 10, 16, 12, 8]
    bar_w = (s - 4) // len(heights)
    for i, h in enumerate(heights):
        v = 60 + i * 40
        p.fillRect(QRect(2 + i * bar_w, s - 4 - h, bar_w - 1, h), QColor(v, v, v))


def icon_denoise(p, s):
    # Dots — some faded
    p.setBrush(QBrush(QColor("#555")))
    p.setPen(Qt.PenStyle.NoPen)
    for x, y, alpha in [(5, 7, 255), (13, 5, 120), (19, 10, 255),
                         (8, 16, 80),  (15, 18, 255), (21, 20, 140)]:
        color = QColor(80, 80, 80, alpha)
        p.setBrush(QBrush(color))
        p.drawEllipse(x, y, 4, 4)


def icon_deskew(p, s):
    pen = QPen(QColor("#c44"), 2)
    pen.setStyle(Qt.PenStyle.DashLine)
    p.setPen(pen)
    p.drawLine(3, s - 6, s - 3, 5)
    p.setPen(QPen(QColor("#333"), 2))
    p.drawLine(3, s // 2, s - 3, s // 2)


def icon_binarize(p, s):
    # Left half dark gray gradient, right half pure black/white split
    for i in range(s // 2 - 1):
        v = int(80 + i * (160 / (s // 2)))
        p.fillRect(QRect(2 + i, 2, 1, s - 4), QColor(v, v, v))
    p.fillRect(QRect(s // 2, 2, s // 2 - 2, (s - 4) // 2), QColor("#000"))
    p.fillRect(QRect(s // 2, 2 + (s - 4) // 2, s // 2 - 2, (s - 4) // 2), QColor("#fff"))
    p.setPen(QPen(QColor(C_BORDER), 1))
    p.drawLine(s // 2, 2, s // 2, s - 2)


def icon_isolate(p, s):
    p.fillRect(QRect(2, 2, s - 4, s - 4), QColor("#fff"))
    p.setPen(QPen(QColor(C_BORDER), 1))
    p.drawRect(QRect(2, 2, s - 4, s - 4))
    p.setPen(Qt.PenStyle.NoPen)
    for rx, ry, rw, rh in [(5, 6, 5, 10), (13, 8, 4, 8), (19, 5, 4, 12)]:
        p.fillRect(QRect(rx, ry, rw, rh), QColor("#333"))


def icon_remove_chars(p, s):
    # Dark background with white cutouts where letters were
    p.fillRect(QRect(2, 2, s - 4, s - 4), QColor("#ccc"))
    p.setPen(QPen(QColor(C_BORDER), 1))
    p.drawRect(QRect(2, 2, s - 4, s - 4))
    p.setPen(Qt.PenStyle.NoPen)
    for rx, ry, rw, rh in [(5, 6, 5, 10), (13, 8, 4, 8), (19, 5, 4, 12)]:
        p.fillRect(QRect(rx, ry, rw, rh), QColor("#fff"))


def icon_characters(p, s):
    # Small rectangles representing individual characters
    p.setPen(QPen(QColor("#2264e0"), 1))
    p.setBrush(Qt.BrushStyle.NoBrush)
    for x, y, w, h in [(2, 4, 7, 18), (11, 4, 5, 18), (18, 4, 6, 18)]:
        p.drawRect(QRect(x, y, w, h))


# ── Settings dialog ────────────────────────────────────────────────────────

class SettingsDialog(QDialog):
    def __init__(self, settings: SegmentationSettings, on_changed, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.on_changed = on_changed
        self.setWindowTitle("Detection Settings")
        self.setFixedWidth(420)
        self.setStyleSheet(f"background: {C_BG}; color: {C_TEXT};")

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        layout.addWidget(self._group_binarization())
        layout.addWidget(self._group_lines())
        layout.addWidget(self._group_chars())

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.setStyleSheet(f"""
            QPushButton {{
                background: {C_BTN_ADD}; color: white; border: none;
                border-radius: 4px; padding: 6px 20px;
            }}
            QPushButton:hover {{ background: {C_BTN_ADD_H}; }}
        """)
        btn_box.rejected.connect(self.close)
        layout.addWidget(btn_box)

    def _group_binarization(self):
        group = self._make_group("Binarization")
        form = QFormLayout(group)

        method_box = QComboBox()
        method_box.addItems(["Otsu (auto)", "Adaptive", "Manual"])
        method_box.setCurrentIndex(["otsu", "adaptive", "manual"]
                                    .index(self.settings.threshold_method))
        method_box.setStyleSheet(self._combo_style())

        manual_spin = QSpinBox()
        manual_spin.setRange(0, 255)
        manual_spin.setValue(self.settings.manual_threshold)
        manual_spin.setEnabled(self.settings.threshold_method == "manual")
        manual_spin.setStyleSheet(self._spin_style())
        manual_spin.setToolTip("Active only when method is Manual")

        def on_method(idx):
            self.settings.threshold_method = ["otsu", "adaptive", "manual"][idx]
            manual_spin.setEnabled(idx == 2)
            self.on_changed()

        method_box.currentIndexChanged.connect(on_method)
        manual_spin.valueChanged.connect(
            lambda v: self._set_and_notify("manual_threshold", v))

        form.addRow("Method:", method_box)
        form.addRow("Manual threshold:", manual_spin)
        return group

    def _group_lines(self):
        group = self._make_group("Line Detection")
        form = QFormLayout(group)

        form.addRow("Merge kernel width:",
                    self._slider(10, 120, self.settings.line_kernel_width, 1,
                                 "line_kernel_width", suffix="px"))
        form.addRow("Line sensitivity:",
                    self._slider(1, 30, self.settings.line_threshold_pct, 1,
                                 "line_threshold_pct", suffix="%"))
        return group

    def _group_chars(self):
        group = self._make_group("Character Detection")
        form = QFormLayout(group)

        connectivity_box = QComboBox()
        connectivity_box.addItems(["4 — horizontal/vertical only",
                                   "8 — include diagonals"])
        connectivity_box.setCurrentIndex(0 if self.settings.connectivity == 4 else 1)
        connectivity_box.setStyleSheet(self._combo_style())
        connectivity_box.currentIndexChanged.connect(
            lambda i: self._set_and_notify("connectivity", 4 if i == 0 else 8))

        form.addRow("Min area (px²):",
                    self._slider(5, 500, self.settings.min_area, 5,
                                 "min_area", suffix="px²"))
        form.addRow("Max area (px²):",
                    self._slider(100, 10000, self.settings.max_area, 100,
                                 "max_area", suffix="px²"))
        form.addRow("Min width:",
                    self._slider(1, 100, self.settings.min_width, 1,
                                 "min_width", suffix="px"))
        form.addRow("Max width:",
                    self._slider(10, 1000, self.settings.max_width, 10,
                                 "max_width", suffix="px"))
        form.addRow("Min height:",
                    self._slider(1, 100, self.settings.min_height, 1,
                                 "min_height", suffix="px"))
        form.addRow("Max height:",
                    self._slider(10, 1000, self.settings.max_height, 10,
                                 "max_height", suffix="px"))
        form.addRow("Connectivity:", connectivity_box)
        return group

    def _slider(self, min_val, max_val, current, step, attr, suffix=""):
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(current)
        slider.setSingleStep(step)
        slider.setStyleSheet(self._slider_style())

        val_label = QLabel(f"{current}{suffix}")
        val_label.setFixedWidth(52)
        val_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        val_label.setStyleSheet(f"color: {C_SELECTED}; font-weight: bold; font-size: 12px;")

        def on_change(v):
            val_label.setText(f"{v}{suffix}")
            self._set_and_notify(attr, v)

        slider.valueChanged.connect(on_change)
        row.addWidget(slider)
        row.addWidget(val_label)
        return container

    def _set_and_notify(self, attr, value):
        setattr(self.settings, attr, value)
        self.on_changed()

    def _make_group(self, title):
        group = QGroupBox(title)
        group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold; font-size: 12px;
                color: {C_TEXT}; border: 1px solid {C_BORDER};
                border-radius: 6px; margin-top: 8px; padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; left: 10px; padding: 0 4px;
                color: {C_TEXT_MUTED};
            }}
        """)
        return group

    def _combo_style(self):
        return (f"QComboBox {{ background: {C_BG}; border: 1px solid {C_BORDER}; "
                f"border-radius: 4px; padding: 3px 6px; color: {C_TEXT}; }}")

    def _spin_style(self):
        return (f"QSpinBox {{ background: {C_BG}; border: 1px solid {C_BORDER}; "
                f"border-radius: 4px; padding: 3px 6px; color: {C_TEXT}; }}")

    def _slider_style(self):
        return f"""
            QSlider::groove:horizontal {{
                height: 4px; background: {C_BORDER}; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {C_SELECTED}; width: 14px; height: 14px;
                margin: -5px 0; border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{
                background: {C_SELECTED}; border-radius: 2px;
            }}
        """


# ── Filter bar ─────────────────────────────────────────────────────────────

FILTERS = [
    ("Original",   icon_original,  filter_original,  "Show original image"),
    ("Grayscale",  icon_grayscale, filter_grayscale, "Convert to grayscale"),
    ("Otsu",       icon_otsu,      filter_otsu,      "Otsu global binarization"),
    ("Adaptive",   icon_adaptive,  filter_adaptive,  "Adaptive threshold binarization"),
    ("CLAHE",      icon_clahe,     filter_clahe,     "Contrast enhancement (CLAHE)"),
    ("Denoise",    icon_denoise,   filter_denoise,   "Noise removal (median + morphology)"),
    ("Deskew",    icon_deskew,     filter_deskew,        "Auto deskew correction"),
    ("Binarize",  icon_binarize,   filter_binarize,      "Binarize using method and threshold from Detection Settings"),
    ("Isolate",  icon_isolate,       filter_isolate_chars, "Show only detected characters on white background"),
    ("Erase",    icon_remove_chars,  filter_remove_chars,  "Remove detected characters, keep background"),
]


class FilterBar(QWidget):
    def __init__(self, on_filter_changed, on_chars_toggled, parent=None):
        super().__init__(parent)
        self.on_filter_changed = on_filter_changed
        self.setFixedHeight(52)
        self.setStyleSheet(f"background: {C_FILTER_BAR}; border-bottom: 1px solid {C_BORDER};")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(6)

        # ── Filter buttons (independent toggles) ──
        label = QLabel("Filters:")
        label.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 11px; font-weight: bold;")
        layout.addWidget(label)

        self._btns = []
        for i, (name, icon_fn, fn, tip) in enumerate(FILTERS):
            btn = QPushButton()
            btn.setCheckable(True)
            btn.setChecked(i == 0)   # Original on by default
            btn.setFixedSize(38, 38)
            btn.setIcon(QIcon(_make_icon(icon_fn)))
            btn.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
            btn.setToolTip(f"<b>{name}</b><br>{tip}")
            btn.setStyleSheet(self._btn_style())
            btn.toggled.connect(lambda checked, idx=i: self._on_toggled(idx, checked))
            self._btns.append(btn)
            layout.addWidget(btn)

        # ── Separator ──
        sep = QWidget()
        sep.setFixedSize(1, 34)
        sep.setStyleSheet(f"background: {C_BORDER};")
        layout.addSpacing(6)
        layout.addWidget(sep)
        layout.addSpacing(6)

        # ── Segmentation buttons (independent toggles) ──
        seg_label = QLabel("Detect:")
        seg_label.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 11px; font-weight: bold;")
        layout.addWidget(seg_label)

        self._btn_chars = self._make_seg_btn(
            icon_characters, "<b>Find Characters</b><br>Detect individual character boundaries", on_chars_toggled
        )
        layout.addWidget(self._btn_chars)
        layout.addStretch()

    def _make_seg_btn(self, icon_fn, tip, callback):
        btn = QPushButton()
        btn.setCheckable(True)
        btn.setFixedSize(38, 38)
        btn.setIcon(QIcon(_make_icon(icon_fn)))
        btn.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        btn.setToolTip(tip)
        btn.setStyleSheet(self._btn_style())
        btn.toggled.connect(callback)
        return btn

    def _btn_style(self):
        return f"""
            QPushButton {{
                border: 1px solid {C_BORDER};
                border-radius: 6px;
                background: {C_BG};
            }}
            QPushButton:hover {{
                background: {C_HOVER};
                border-color: {C_SELECTED};
            }}
            QPushButton:checked {{
                background: {C_FILTER_ACT_BG};
                border: 2px solid {C_FILTER_ACT};
            }}
        """

    def _on_toggled(self, btn_id, checked):
        # "Original" (idx 0) unchecks all others; checking any other unchecks Original
        if btn_id == 0 and checked:
            for i, btn in enumerate(self._btns):
                if i != 0:
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(False)
        elif btn_id != 0 and checked:
            self._btns[0].blockSignals(True)
            self._btns[0].setChecked(False)
            self._btns[0].blockSignals(False)
        self.on_filter_changed()

    def active_filters(self):
        """Return list of active filter functions in order, skipping Original."""
        return [FILTERS[i][2] for i, btn in enumerate(self._btns)
                if i != 0 and btn.isChecked()]


# ── Hebrew alphabet ────────────────────────────────────────────────────────

HEBREW_ALPHABET = [
    ('alef',        'א', 'Alef'),
    ('bet',         'ב', 'Bet'),
    ('gimel',       'ג', 'Gimel'),
    ('dalet',       'ד', 'Dalet'),
    ('he',          'ה', 'He'),
    ('vav',         'ו', 'Vav'),
    ('zayin',       'ז', 'Zayin'),
    ('het',         'ח', 'Het'),
    ('tet',         'ט', 'Tet'),
    ('yod',         'י', 'Yod'),
    ('kaf',         'כ', 'Kaf'),
    ('kaf_final',   'ך', 'Kaf (final)'),
    ('lamed',       'ל', 'Lamed'),
    ('mem',         'מ', 'Mem'),
    ('mem_final',   'ם', 'Mem (final)'),
    ('nun',         'נ', 'Nun'),
    ('nun_final',   'ן', 'Nun (final)'),
    ('samekh',      'ס', 'Samekh'),
    ('ayin',        'ע', 'Ayin'),
    ('pe',          'פ', 'Pe'),
    ('pe_final',    'ף', 'Pe (final)'),
    ('tsadi',       'צ', 'Tsadi'),
    ('tsadi_final', 'ץ', 'Tsadi (final)'),
    ('qof',         'ק', 'Qof'),
    ('resh',        'ר', 'Resh'),
    ('shin',        'ש', 'Shin'),
    ('tav',         'ת', 'Tav'),
]


class LetterTile(QFrame):
    clicked = pyqtSignal(str)

    def __init__(self, name, letter, label, parent=None):
        super().__init__(parent)
        self.name = name
        self._has_images = False
        self.setFixedSize(70, 76)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"{label}  ({letter})")
        self._apply_style(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(2)

        self._letter_lbl = QLabel(letter)
        self._letter_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._letter_lbl.setStyleSheet(
            "font-size: 26px; font-weight: bold; background: transparent;"
        )
        layout.addWidget(self._letter_lbl)

        self._dot = QLabel("●")
        self._dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._dot.setFixedHeight(10)
        self._dot.setStyleSheet("font-size: 7px; color: #ccc; background: transparent;")
        layout.addWidget(self._dot)

    def set_has_images(self, value: bool):
        self._has_images = value
        self._dot.setStyleSheet(
            f"font-size: 7px; color: {'#4caf50' if value else '#ccc'}; background: transparent;"
        )

    def _apply_style(self, hovered):
        bg = C_HOVER if hovered else C_BG
        self.setStyleSheet(f"""
            QFrame {{
                background: {bg};
                border: 1px solid {C_BORDER};
                border-radius: 6px;
            }}
        """)

    def enterEvent(self, e):
        self._apply_style(True)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._apply_style(False)
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.name)
        super().mousePressEvent(e)


class HebrewAlphabetBar(QWidget):
    def __init__(self, on_letter_clicked, parent=None):
        super().__init__(parent)
        self.setFixedHeight(104)
        self.setStyleSheet(
            f"background: {C_SIDEBAR_HDR}; border-top: 1px solid {C_BORDER};"
        )

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Scrollable tiles
        scroll = QScrollArea()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"border: none; background: {C_SIDEBAR_HDR};")

        tiles_widget = QWidget()
        tiles_widget.setStyleSheet(f"background: {C_SIDEBAR_HDR};")
        tiles_layout = QHBoxLayout(tiles_widget)
        tiles_layout.setContentsMargins(10, 8, 10, 8)
        tiles_layout.setSpacing(6)

        self._tiles = {}
        for name, letter, label in HEBREW_ALPHABET:
            tile = LetterTile(name, letter, label)
            tile.clicked.connect(on_letter_clicked)
            self._tiles[name] = tile
            tiles_layout.addWidget(tile)

        tiles_layout.addStretch()
        scroll.setWidget(tiles_widget)
        outer.addWidget(scroll)

        # Return button (disabled until letter-preview mode is implemented)
        sep = QFrame()
        sep.setFixedWidth(1)
        sep.setStyleSheet(f"background: {C_BORDER};")
        outer.addWidget(sep)

        self._return_btn = QPushButton("← Back")
        self._return_btn.setFixedSize(72, 104)
        self._return_btn.setEnabled(False)
        self._return_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C_SIDEBAR_HDR}; color: {C_TEXT_MUTED};
                border: none; font-size: 12px;
            }}
            QPushButton:enabled:hover {{ background: {C_HOVER}; color: {C_TEXT}; }}
        """)
        outer.addWidget(self._return_btn)

    def tile(self, name) -> LetterTile:
        return self._tiles.get(name)


# ── Letter preview widgets ─────────────────────────────────────────────────

class ImagePreviewCard(QFrame):
    removed = pyqtSignal(int)

    def __init__(self, img_bgr, index, parent=None):
        super().__init__(parent)
        self._index = index
        self.setFixedSize(162, 162)
        self._apply_style(False)

        img_lbl = QLabel(self)
        img_lbl.setGeometry(6, 6, 150, 150)
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_lbl.setStyleSheet("border: none; background: transparent;")
        px = cv_to_pixmap(img_bgr)
        img_lbl.setPixmap(px.scaled(
            150, 150,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

        self._del = QPushButton("✕", self)
        self._del.setGeometry(136, 4, 22, 22)
        self._del.setStyleSheet("""
            QPushButton {
                background: #e05050; color: white; border: none;
                border-radius: 11px; font-size: 10px; font-weight: bold;
            }
            QPushButton:hover { background: #c03030; }
        """)
        self._del.hide()
        self._del.clicked.connect(lambda: self.removed.emit(self._index))

    def _apply_style(self, hovered):
        self.setStyleSheet(f"""
            QFrame {{
                background: {C_BG};
                border: {'2px solid ' + C_SELECTED if hovered else '1px solid ' + C_BORDER};
                border-radius: 6px;
            }}
        """)

    def enterEvent(self, e):
        self._apply_style(True)
        self._del.show()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._apply_style(False)
        self._del.hide()
        super().leaveEvent(e)


class LetterPreviewWidget(QWidget):
    image_removed = pyqtSignal(int)
    right_clicked = pyqtSignal()

    COLS = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {C_BG};")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._header = QLabel()
        self._header.setFixedHeight(40)
        self._header.setStyleSheet(
            f"color: {C_TEXT}; font-size: 14px; font-weight: bold; "
            f"padding-left: 16px; background: {C_SIDEBAR_HDR}; "
            f"border-bottom: 1px solid {C_BORDER};"
        )
        outer.addWidget(self._header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"border: none; background: {C_BG};")

        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet(f"background: {C_BG};")
        self._grid = QGridLayout(self._grid_widget)
        self._grid.setContentsMargins(16, 16, 16, 16)
        self._grid.setSpacing(12)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        scroll.setWidget(self._grid_widget)
        outer.addWidget(scroll)

    def populate(self, letter_char, label, images):
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        n = len(images)
        self._header.setText(
            f"  {letter_char}  —  {label}  —  {n} sample{'s' if n != 1 else ''}"
        )

        if n == 0:
            empty = QLabel("No images assigned to this letter yet")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 14px;")
            self._grid.addWidget(empty, 0, 0)
            return

        for i, img in enumerate(images):
            card = ImagePreviewCard(img, i)
            card.removed.connect(self.image_removed.emit)
            self._grid.addWidget(card, i // self.COLS, i % self.COLS)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit()
        super().mousePressEvent(e)


# ── Image list sidebar ─────────────────────────────────────────────────────

class ImageListItem(QWidget):
    def __init__(self, path, pixmap):
        super().__init__()
        self.path = path
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(10)

        thumb = QLabel()
        thumb.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb.setPixmap(pixmap.scaled(
            THUMB_SIZE, THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
        thumb.setStyleSheet(f"border: 1px solid {C_BORDER}; background: {C_BG};")

        name = QLabel(os.path.basename(path))
        name.setWordWrap(True)
        name.setStyleSheet(f"font-size: 12px; color: {C_TEXT};")
        name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        layout.addWidget(thumb)
        layout.addWidget(name)


def _list_style():
    return f"""
        QListWidget {{ background: {C_SIDEBAR}; border: none; }}
        QListWidget::item {{ border-bottom: 1px solid {C_BORDER}; color: {C_TEXT}; }}
        QListWidget::item:selected {{
            background: {C_SELECTED_BG}; border-left: 3px solid {C_SELECTED};
        }}
        QListWidget::item:hover:!selected {{ background: {C_HOVER}; }}
    """


def _btn_style(bg, hover):
    return (f"QPushButton {{ background: {bg}; color: #fff; border: none; "
            f"border-radius: 4px; padding: 4px 8px; font-size: 12px; }}"
            f"QPushButton:hover {{ background: {hover}; }}")


class _ListPanel(QWidget):
    """Reusable panel: header + list + button bar."""
    def __init__(self, title, on_select, buttons, parent=None):
        super().__init__(parent)
        self.on_select = on_select
        self.setStyleSheet(f"background: {C_SIDEBAR};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QLabel(f"  {title}")
        header.setFixedHeight(32)
        header.setStyleSheet(
            f"background: {C_SIDEBAR_HDR}; color: {C_TEXT_MUTED}; "
            f"font-size: 11px; font-weight: bold; border-bottom: 1px solid {C_BORDER};"
        )
        layout.addWidget(header)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(_list_style())
        self.list_widget.setSpacing(2)
        self.list_widget.currentItemChanged.connect(self._on_changed)
        layout.addWidget(self.list_widget)

        if buttons:
            bar = QWidget()
            bar.setFixedHeight(40)
            bar.setStyleSheet(f"background: {C_SIDEBAR_HDR}; border-top: 1px solid {C_BORDER};")
            bl = QHBoxLayout(bar)
            bl.setContentsMargins(6, 5, 6, 5)
            bl.setSpacing(6)
            for label, bg, bg_h, slot in buttons:
                btn = QPushButton(label)
                btn.setStyleSheet(_btn_style(bg, bg_h))
                btn.clicked.connect(slot)
                bl.addWidget(btn)
            layout.addWidget(bar)

    def add_item(self, path, pixmap=None):
        if pixmap is None:
            pixmap = QPixmap(path)
        if pixmap.isNull():
            return
        for i in range(self.list_widget.count()):
            w = self.list_widget.itemWidget(self.list_widget.item(i))
            if w and w.path == path:
                return
        item = QListWidgetItem(self.list_widget)
        widget = ImageListItem(path, pixmap)
        item.setSizeHint(widget.sizeHint())
        item.setData(Qt.ItemDataRole.UserRole, path)
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, widget)
        if self.list_widget.count() == 1:
            self.list_widget.setCurrentItem(item)

    def remove_selected(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            self.list_widget.takeItem(row)
            count = self.list_widget.count()
            if count > 0:
                self.list_widget.setCurrentRow(min(row, count - 1))
            else:
                self.on_select(None)

    def _on_changed(self, current, previous):
        path = current.data(Qt.ItemDataRole.UserRole) if current else None
        self.on_select(path)


class Sidebar(QWidget):
    def __init__(self, on_select, on_merge_layers=None, parent=None):
        super().__init__(parent)
        self.on_select = on_select
        self.setFixedWidth(SIDEBAR_WIDTH)
        self.setStyleSheet(f"background: {C_SIDEBAR};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {C_BORDER}; }}"
        )

        self._images_panel = _ListPanel(
            "Images", self._on_images_select,
            buttons=[
                ("+ Add",    C_BTN_ADD, C_BTN_ADD_H, self._add_images),
                ("− Remove", C_BTN_REM, C_BTN_REM_H, self._remove_image),
            ]
        )

        merge_btn = [("⊕ OR", C_BTN_ADD, C_BTN_ADD_H, on_merge_layers)] \
                    if on_merge_layers else []
        self._layers_panel = _ListPanel(
            "Layers", self._on_layers_select,
            buttons=[("− Remove", C_BTN_REM, C_BTN_REM_H, self._remove_layer)] + merge_btn
        )

        splitter.addWidget(self._images_panel)
        splitter.addWidget(self._layers_panel)
        splitter.setSizes([400, 200])

        layout.addWidget(splitter)

    def _on_images_select(self, path):
        if path:
            self._layers_panel.list_widget.blockSignals(True)
            self._layers_panel.list_widget.clearSelection()
            self._layers_panel.list_widget.setCurrentRow(-1)
            self._layers_panel.list_widget.blockSignals(False)
        self.on_select(path)

    def _on_layers_select(self, path):
        if path:
            self._images_panel.list_widget.blockSignals(True)
            self._images_panel.list_widget.clearSelection()
            self._images_panel.list_widget.setCurrentRow(-1)
            self._images_panel.list_widget.blockSignals(False)
        self.on_select(path)

    def _add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Image Files", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
        )
        for path in paths:
            self._images_panel.add_item(path)

    def _remove_image(self):
        self._images_panel.remove_selected()

    def _remove_layer(self):
        self._layers_panel.remove_selected()

    def add_images(self):
        self._add_images()

    def add_layer(self, path):
        self._layers_panel.add_item(path)


# ── Image viewer ───────────────────────────────────────────────────────────

class ImageViewer(QLabel):
    pixel_clicked = pyqtSignal(float, float)
    right_clicked = pyqtSignal()
    drag_started  = pyqtSignal(float, float)
    drag_moved    = pyqtSignal(float, float)
    pixel_moved   = pyqtSignal(float, float)   # absolute image coords on mouse-hold move

    ZOOM_MIN = 0.05
    ZOOM_MAX = 10.0
    ZOOM_STEP = 1.25

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap      = None   # base (unscaled) pixmap
        self._origin      = (0, 0)
        self._drag_last   = None
        self._zoom        = 1.0
        self._zoom_enabled = True
        self._sa          = None   # QScrollArea reference
        self._show_placeholder()

    # ── public API ──────────────────────────────────────────────────────────

    def set_scroll_area(self, sa):
        self._sa = sa

    def set_zoom_enabled(self, enabled):
        self._zoom_enabled = enabled
        if not enabled:
            self._reset_zoom()

    def reset_zoom(self):
        self._reset_zoom()

    def set_display_origin(self, ox, oy):
        self._origin = (ox, oy)

    def set_pixmap(self, pixmap):
        if pixmap is None or pixmap.isNull():
            self._pixmap = None
            self._reset_zoom(render=False)
            self._show_placeholder()
            return
        self._pixmap = pixmap
        self.setStyleSheet(f"background: {C_BG};")
        self._render()

    # ── coordinate mapping ───────────────────────────────────────────────────

    def widget_to_image(self, wx, wy):
        if not self._pixmap or self._pixmap.isNull():
            return -1.0, -1.0
        pw, ph = self._pixmap.width(), self._pixmap.height()
        vw, vh = self.width(), self.height()
        scale  = min(vw / pw, vh / ph)
        offx   = (vw - pw * scale) / 2
        offy   = (vh - ph * scale) / 2
        return (wx - offx) / scale + self._origin[0], \
               (wy - offy) / scale + self._origin[1]

    # ── internal rendering ───────────────────────────────────────────────────

    def _show_placeholder(self):
        self.setText("Add images using the sidebar\n\nCtrl+O  or  click  + Add")
        self.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 16px; background: {C_BG};")

    def _reset_zoom(self, render=True):
        self._zoom = 1.0
        if self._sa:
            self._sa.setWidgetResizable(True)
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)
        if render:
            self._render()

    def _render(self):
        if not self._pixmap:
            return
        if self._zoom == 1.0:
            if self._sa:
                self._sa.setWidgetResizable(True)
            self.setMinimumSize(0, 0)
            self.setMaximumSize(16777215, 16777215)
            self._fit()
        else:
            if self._sa:
                self._sa.setWidgetResizable(False)
            w = int(self._pixmap.width()  * self._zoom)
            h = int(self._pixmap.height() * self._zoom)
            self.setFixedSize(w, h)
            self.setPixmap(self._pixmap.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

    def _fit(self):
        if self._pixmap:
            self.setPixmap(self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

    def resizeEvent(self, event):
        if self._zoom == 1.0:
            self._fit()
        super().resizeEvent(event)

    # ── zoom ─────────────────────────────────────────────────────────────────

    def wheelEvent(self, e):
        if not self._zoom_enabled or not self._pixmap:
            super().wheelEvent(e)
            return

        factor   = self.ZOOM_STEP if e.angleDelta().y() > 0 else 1 / self.ZOOM_STEP
        old_zoom = self._zoom
        new_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, old_zoom * factor))
        if new_zoom == old_zoom:
            return

        mx, my = e.position().x(), e.position().y()

        if old_zoom == 1.0:
            # Currently fit-to-view — compute effective render scale
            pw, ph = self._pixmap.width(), self._pixmap.height()
            eff    = min(self.width() / pw, self.height() / ph)
            offx   = (self.width()  - pw * eff) / 2
            offy   = (self.height() - ph * eff) / 2
            img_x  = (mx - offx) / eff
            img_y  = (my - offy) / eff
            self._zoom = new_zoom
            self._render()
            if self._sa:
                self._sa.horizontalScrollBar().setValue(int(img_x * new_zoom - mx))
                self._sa.verticalScrollBar().setValue(int(img_y * new_zoom - my))
        else:
            if self._sa:
                sx = self._sa.horizontalScrollBar().value()
                sy = self._sa.verticalScrollBar().value()
                img_x = (sx + mx) / old_zoom
                img_y = (sy + my) / old_zoom
            self._zoom = new_zoom
            self._render()
            if self._sa:
                self._sa.horizontalScrollBar().setValue(int(img_x * new_zoom - mx))
                self._sa.verticalScrollBar().setValue(int(img_y * new_zoom - my))

        e.accept()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit()
        elif e.button() == Qt.MouseButton.LeftButton:
            ix, iy = self.widget_to_image(e.position().x(), e.position().y())
            self._drag_last = (e.position().x(), e.position().y())
            self.pixel_clicked.emit(ix, iy)
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._drag_last and e.buttons() & Qt.MouseButton.LeftButton:
            ix, iy = self.widget_to_image(e.position().x(), e.position().y())
            self.pixel_moved.emit(ix, iy)
            dx_w = e.position().x() - self._drag_last[0]
            dy_w = e.position().y() - self._drag_last[1]
            if not self._pixmap or self._pixmap.isNull():
                return
            pw, ph = self._pixmap.width(), self._pixmap.height()
            scale  = min(self.width() / pw, self.height() / ph)
            self.drag_moved.emit(dx_w / scale, dy_w / scale)
            self._drag_last = (e.position().x(), e.position().y())
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._drag_last = None
        super().mouseReleaseEvent(e)


# ── Assignment panel ────────────────────────────────────────────────────────

class AssignmentPanel(QWidget):
    width_changed  = pyqtSignal(int)
    height_changed = pyqtSignal(int)
    angle_changed  = pyqtSignal(int)
    move_requested = pyqtSignal(int, int)
    eraser_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(200)
        self.setStyleSheet(
            f"background: {C_SIDEBAR}; border-left: 1px solid {C_BORDER};"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(14)

        title = QLabel("Selection")
        title.setStyleSheet(f"font-weight: bold; font-size: 13px; color: {C_TEXT};")
        layout.addWidget(title)

        # Preview
        self._preview = QLabel()
        self._preview.setFixedSize(172, 100)
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setStyleSheet(
            f"border: 1px solid {C_BORDER}; background: {C_BG}; border-radius: 4px;"
        )
        layout.addWidget(self._preview)

        # Width slider
        layout.addWidget(self._section("Width"))
        self._w_slider, self._w_label = self._make_slider()
        self._w_slider.valueChanged.connect(self._on_w)
        layout.addWidget(self._w_slider)
        layout.addWidget(self._w_label)

        # Height slider
        layout.addWidget(self._section("Height"))
        self._h_slider, self._h_label = self._make_slider()
        self._h_slider.valueChanged.connect(self._on_h)
        layout.addWidget(self._h_slider)
        layout.addWidget(self._h_label)

        # Angle slider
        layout.addWidget(self._section("Angle"))
        self._a_slider, self._a_label = self._make_slider(min_v=-45, max_v=45)
        self._a_slider.valueChanged.connect(self._on_a)
        layout.addWidget(self._a_slider)
        layout.addWidget(self._a_label)

        # D-pad
        layout.addWidget(self._section("Position"))
        layout.addWidget(self._make_dpad())

        # Eraser tool
        layout.addWidget(self._section("Tools"))
        self._eraser_btn = QPushButton("⬜  Eraser  (2×2 px)")
        self._eraser_btn.setCheckable(True)
        self._eraser_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C_BG}; color: {C_TEXT};
                border: 1px solid {C_BORDER}; border-radius: 6px;
                padding: 6px; font-size: 12px;
            }}
            QPushButton:hover {{ background: {C_HOVER}; border-color: {C_SELECTED}; }}
            QPushButton:checked {{
                background: {C_FILTER_ACT_BG}; border: 2px solid {C_FILTER_ACT};
                color: {C_TEXT};
            }}
        """)
        self._eraser_btn.toggled.connect(self.eraser_toggled.emit)
        layout.addWidget(self._eraser_btn)

        hint = QLabel("Click a letter below\nto assign this crop")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet(
            f"color: {C_TEXT_MUTED}; font-size: 11px; "
            f"border: 1px dashed {C_BORDER}; border-radius: 4px; padding: 8px;"
        )
        layout.addWidget(hint)
        layout.addStretch()

    def reset_eraser(self):
        self._eraser_btn.blockSignals(True)
        self._eraser_btn.setChecked(False)
        self._eraser_btn.blockSignals(False)

    def _make_dpad(self):
        container = QWidget()
        container.setFixedSize(114, 114)
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)

        directions = [
            (0, 1, "▲",  0, -1),
            (1, 0, "◄", -1,  0),
            (1, 2, "►",  1,  0),
            (2, 1, "▼",  0,  1),
        ]
        for row, col, arrow, dx, dy in directions:
            btn = QPushButton(arrow)
            btn.setFixedSize(34, 34)
            btn.setAutoRepeat(True)
            btn.setAutoRepeatDelay(400)
            btn.setAutoRepeatInterval(80)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {C_BG}; color: {C_TEXT};
                    border: 1px solid {C_BORDER}; border-radius: 6px;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background: {C_HOVER}; border-color: {C_SELECTED};
                }}
                QPushButton:pressed {{
                    background: {C_SELECTED_BG}; border-color: {C_SELECTED};
                }}
            """)
            btn.clicked.connect(lambda _, x=dx, y=dy: self.move_requested.emit(x, y))
            grid.addWidget(btn, row, col)

        # Center dot
        center = QLabel("✛")
        center.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.setFixedSize(34, 34)
        center.setStyleSheet(f"color: {C_BORDER}; font-size: 14px;")
        grid.addWidget(center, 1, 1)

        return container

    def _section(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 11px; font-weight: bold;")
        return lbl

    def _make_slider(self, min_v=4, max_v=800):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 4px; background: {C_BORDER}; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {C_SELECTED}; width: 14px; height: 14px;
                margin: -5px 0; border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{
                background: {C_SELECTED}; border-radius: 2px;
            }}
        """)
        label = QLabel("—")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(f"color: {C_SELECTED}; font-weight: bold; font-size: 12px;")
        return slider, label

    def _on_w(self, v):
        self._w_label.setText(f"{v} px")
        self.width_changed.emit(v)

    def _on_h(self, v):
        self._h_label.setText(f"{v} px")
        self.height_changed.emit(v)

    def _on_a(self, v):
        self._a_label.setText(f"{v}°")
        self.angle_changed.emit(v)

    def set_rect(self, w, h):
        self._w_slider.blockSignals(True)
        self._h_slider.blockSignals(True)
        self._w_slider.setValue(int(w))
        self._h_slider.setValue(int(h))
        self._w_label.setText(f"{int(w)} px")
        self._h_label.setText(f"{int(h)} px")
        self._w_slider.blockSignals(False)
        self._h_slider.blockSignals(False)

    def set_angle(self, angle):
        self._a_slider.blockSignals(True)
        self._a_slider.setValue(int(angle))
        self._a_label.setText(f"{int(angle)}°")
        self._a_slider.blockSignals(False)

    def set_preview(self, img_crop):
        if img_crop is None or img_crop.size == 0:
            self._preview.clear()
            return
        px = cv_to_pixmap(img_crop)
        self._preview.setPixmap(px.scaled(
            self._preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))


# ── Main window ────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Font Generator")
        self.resize(1100, 720)
        self.setStyleSheet(f"background: {C_BG};")

        self._current_path = None
        self._cv_cache     = {}
        self._line_rects   = []
        self._char_rects   = []
        self._show_chars   = False
        self._seg_settings = load_settings()
        self._settings_dialog = None
        # assignment / preview mode
        self._mode           = "normal"
        self._selected_rect  = None     # [x, y, w, h] mutable
        self._selected_angle = 0
        self._eraser_active  = False
        self._erase_mask     = None   # np.ndarray bool, applied after filters
        self._zoom_origin    = (0, 0)
        self._assigned       = {}       # letter_name → list of np.ndarray
        self._preview_letter = None

        self._build_menu()
        self._build_central()
        self._build_statusbar()

    def _build_menu(self):
        menu = self.menuBar()
        menu.setStyleSheet(f"background: {C_SIDEBAR_HDR}; color: {C_TEXT};")
        file_menu = menu.addMenu("&File")

        open_act = QAction("&Add Images...", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(lambda: self._sidebar.add_images())
        file_menu.addAction(open_act)

        save_act = QAction("&Save Layer", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self._save_layer)
        file_menu.addAction(save_act)

        file_menu.addSeparator()

        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        font_menu = menu.addMenu("&Font")
        self._create_font_act = QAction("&Create Font...", self)
        self._create_font_act.setShortcut("Ctrl+F")
        self._create_font_act.setEnabled(False)
        self._create_font_act.triggered.connect(self._create_font)
        font_menu.addAction(self._create_font_act)

        view_menu = menu.addMenu("&View")
        reset_zoom_act = QAction("Reset Zoom", self)
        reset_zoom_act.setShortcut("Ctrl+0")
        reset_zoom_act.triggered.connect(lambda: self._viewer.reset_zoom())
        view_menu.addAction(reset_zoom_act)

        detect_menu = menu.addMenu("&Detection")
        settings_act = QAction("&Settings...", self)
        settings_act.setShortcut("Ctrl+,")
        settings_act.triggered.connect(self._open_settings)
        detect_menu.addAction(settings_act)

    def _build_central(self):
        self._viewer = ImageViewer()
        self._viewer.pixel_clicked.connect(self._on_viewer_click)
        self._viewer.right_clicked.connect(self._on_back_clicked)
        self._viewer.drag_moved.connect(self._on_viewer_drag)
        self._viewer.pixel_moved.connect(self._on_viewer_pixel_moved)

        self._scroll = QScrollArea()
        self._scroll.setWidget(self._viewer)
        self._scroll.setWidgetResizable(True)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setStyleSheet(f"border: none; background: {C_BG};")
        scroll = self._scroll
        self._viewer.set_scroll_area(self._scroll)

        self._assign_panel = AssignmentPanel()
        self._assign_panel.width_changed.connect(self._on_assign_width)
        self._assign_panel.height_changed.connect(self._on_assign_height)
        self._assign_panel.angle_changed.connect(self._on_assign_angle)
        self._assign_panel.move_requested.connect(self._on_assign_move)
        self._assign_panel.eraser_toggled.connect(self._on_eraser_toggled)
        self._assign_panel.hide()

        self._letter_preview = LetterPreviewWidget()
        self._letter_preview.image_removed.connect(self._on_preview_remove)
        self._letter_preview.right_clicked.connect(self._exit_preview)
        self._letter_preview.hide()

        view_row = QWidget()
        view_row_layout = QHBoxLayout(view_row)
        view_row_layout.setContentsMargins(0, 0, 0, 0)
        view_row_layout.setSpacing(0)
        view_row_layout.addWidget(scroll)
        view_row_layout.addWidget(self._letter_preview)
        view_row_layout.addWidget(self._assign_panel)

        self._filter_bar = FilterBar(
            on_filter_changed=self._on_filter_changed,
            on_chars_toggled=self._on_chars_toggled,
        )
        self._sidebar = Sidebar(on_select=self._on_image_selected,
                                on_merge_layers=self._merge_layers_or)
        self._alphabet_bar = HebrewAlphabetBar(on_letter_clicked=self._on_letter_clicked)
        self._alphabet_bar._return_btn.setEnabled(False)
        self._alphabet_bar._return_btn.clicked.connect(self._on_back_clicked)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self._filter_bar)
        right_layout.addWidget(view_row)
        right_layout.addWidget(self._alphabet_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._sidebar)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {C_BORDER}; }}")

        self.setCentralWidget(splitter)

    def _build_statusbar(self):
        self._status = QStatusBar()
        self._status.setStyleSheet(
            f"background: {C_SIDEBAR_HDR}; color: {C_TEXT_MUTED}; "
            f"font-size: 12px; border-top: 1px solid {C_BORDER};"
        )
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

    def _load_cv(self, path):
        if path not in self._cv_cache:
            img = cv2.imread(path)
            if img is not None:
                self._cv_cache[path] = img
        return self._cv_cache.get(path)

    def _apply_filters(self, img):
        global _active_char_rects, _active_seg_settings
        _active_char_rects   = self._char_rects
        _active_seg_settings = self._seg_settings
        fns = self._filter_bar.active_filters()
        result = img.copy()
        for fn in fns:
            result = fn(result)
        return result

    def _refresh_view(self):
        if not self._current_path:
            return
        img = self._load_cv(self._current_path)
        if img is None:
            return
        filtered = self._apply_filters(img)
        if self._show_chars:
            self._line_rects = detect_lines(filtered, self._seg_settings)
            self._char_rects = detect_characters(filtered, self._line_rects, self._seg_settings)
        result = draw_overlays(filtered, self._char_rects, self._show_chars)
        self._viewer.set_pixmap(cv_to_pixmap(result))
        h, w = img.shape[:2]
        active = self._filter_bar.active_filters()
        filter_names = [f[0] for f in FILTERS if f[2] in active] or ["Original"]
        parts = [os.path.basename(self._current_path), f"{w}×{h}px",
                 "Filters: " + " + ".join(filter_names)]
        if self._show_chars:
            parts.append(f"Chars: {len(self._char_rects)}")
        self._status.showMessage("  —  ".join(parts))

    def _on_image_selected(self, path):
        self._current_path = path
        self._line_rects = []
        self._char_rects = []
        if path is None:
            self._viewer.set_pixmap(None)
            self._status.showMessage("Ready")
            self.setWindowTitle("Font Generator")
            return
        self._refresh_view()
        self.setWindowTitle(f"Font Generator — {os.path.basename(path)}")

    def _on_filter_changed(self):
        self._refresh_view()

    def _on_chars_toggled(self, checked):
        self._show_chars = checked
        if not checked:
            self._line_rects = []
            self._char_rects = []
        self._refresh_view()

    # ── Assignment mode ────────────────────────────────────────────────────

    def _on_viewer_click(self, ix, iy):
        if self._mode == "assignment":
            if self._eraser_active:
                self._erase_at(ix, iy)
            return
        if not self._char_rects:
            return
        # Find which rect was clicked
        for i, (x, y, w, h) in enumerate(self._char_rects):
            if x <= ix <= x + w and y <= iy <= y + h:
                self._enter_assignment(i)
                return

    def _enter_assignment(self, rect_idx):
        self._mode = "assignment"
        self._selected_rect = list(self._char_rects[rect_idx])
        self._selected_angle = 0
        self._erase_mask = None
        self._assign_panel.set_rect(self._selected_rect[2], self._selected_rect[3])
        self._assign_panel.set_angle(0)
        self._assign_panel.show()
        self._viewer.set_zoom_enabled(False)
        self._alphabet_bar._return_btn.setEnabled(True)
        self._viewer.setCursor(Qt.CursorShape.SizeAllCursor)
        self._render_assignment()

    def _on_back_clicked(self):
        if self._mode == "assignment":
            self._exit_assignment()
        elif self._mode == "preview":
            self._exit_preview()

    def _exit_assignment(self):
        if self._mode != "assignment":
            return
        self._mode = "normal"
        self._selected_rect = None
        self._zoom_origin = (0, 0)
        self._assign_panel.hide()
        self._alphabet_bar._return_btn.setEnabled(False)
        self._viewer.set_display_origin(0, 0)
        self._viewer.setCursor(Qt.CursorShape.ArrowCursor)
        self._viewer.set_zoom_enabled(True)
        self._eraser_active = False
        self._erase_mask = None
        self._assign_panel.reset_eraser()
        self._refresh_view()

    def _render_assignment(self):
        if not self._current_path or self._selected_rect is None:
            return
        img = self._load_cv(self._current_path)
        filtered = self._apply_filters(img)

        # Apply erase mask — force exact white after all filters
        if self._erase_mask is not None:
            ih2, iw2 = filtered.shape[:2]
            mask = self._erase_mask[:ih2, :iw2]
            if len(filtered.shape) == 3:
                filtered[mask] = (255, 255, 255)
            else:
                filtered[mask] = 255

        x, y, w, h = [int(v) for v in self._selected_rect]
        ih, iw = filtered.shape[:2]

        PAD = max(80, w, h)
        x1 = max(0, x - PAD)
        y1 = max(0, y - PAD)
        x2 = min(iw, x + w + PAD)
        y2 = min(ih, y + h + PAD)
        self._zoom_origin = (x1, y1)
        self._viewer.set_display_origin(x1, y1)

        crop = filtered[y1:y2, x1:x2].copy()
        if len(crop.shape) == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

        # Rotated rect in crop coordinates
        rx, ry = x - x1, y - y1
        cx_c, cy_c = rx + w / 2, ry + h / 2
        angle = self._selected_angle
        box = cv2.boxPoints(((cx_c, cy_c), (w, h), angle)).astype(np.int32)

        # Dim area outside the rotated selection
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [box], 255)
        overlay = (crop * 0.4).astype(np.uint8)
        overlay[mask > 0] = crop[mask > 0]
        cv2.polylines(overlay, [box], True, (30, 120, 220), 2)

        self._viewer.set_pixmap(cv_to_pixmap(overlay))

        # Preview in panel — extract rotated crop
        char_crop = self._extract_rotated(filtered, x, y, w, h, angle)
        self._assign_panel.set_preview(char_crop)
        self._status.showMessage(
            f"Assignment mode  —  rect {w}×{h}px  —  drag to reposition  |  "
            f"right-click or ← Back to exit"
        )

    def _on_viewer_drag(self, dx, dy):
        if self._mode != "assignment" or self._selected_rect is None:
            return
        if self._eraser_active:
            return
        img = self._load_cv(self._current_path)
        ih, iw = img.shape[:2]
        x, y, w, h = self._selected_rect
        x = max(0, min(iw - w, x + dx))
        y = max(0, min(ih - h, y + dy))
        self._selected_rect[0] = x
        self._selected_rect[1] = y
        self._render_assignment()

    def _on_assign_width(self, val):
        if self._selected_rect:
            self._selected_rect[2] = val
            self._render_assignment()

    def _on_assign_height(self, val):
        if self._selected_rect:
            self._selected_rect[3] = val
            self._render_assignment()

    def _on_assign_angle(self, val):
        self._selected_angle = val
        self._render_assignment()

    def _on_assign_move(self, dx, dy):
        if not self._selected_rect or not self._current_path:
            return
        img = self._load_cv(self._current_path)
        ih, iw = img.shape[:2]
        x, y, w, h = self._selected_rect
        self._selected_rect[0] = max(0, min(iw - w, x + dx))
        self._selected_rect[1] = max(0, min(ih - h, y + dy))
        self._render_assignment()

    def _on_eraser_toggled(self, active):
        self._eraser_active = active
        if self._mode == "assignment":
            cursor = Qt.CursorShape.CrossCursor if active else Qt.CursorShape.SizeAllCursor
            self._viewer.setCursor(cursor)

    def _erase_at(self, ix, iy):
        if not self._current_path:
            return
        img = self._load_cv(self._current_path)
        if img is None:
            return
        ih, iw = img.shape[:2]
        if self._erase_mask is None or self._erase_mask.shape != (ih, iw):
            self._erase_mask = np.zeros((ih, iw), dtype=bool)
        x = max(0, min(iw - 1, int(round(ix))))
        y = max(0, min(ih - 1, int(round(iy))))
        self._erase_mask[y:min(ih, y + 2), x:min(iw, x + 2)] = True
        self._render_assignment()

    def _on_viewer_pixel_moved(self, ix, iy):
        if self._mode == "assignment" and self._eraser_active:
            self._erase_at(ix, iy)

    def _extract_rotated(self, img, x, y, w, h, angle):
        ih, iw = img.shape[:2]
        cx, cy = x + w / 2, y + h / 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        fill = (255, 255, 255) if len(img.shape) == 3 else 255
        rotated = cv2.warpAffine(img, M, (iw, ih),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=fill)
        x1 = max(0, int(cx - w / 2))
        y1 = max(0, int(cy - h / 2))
        x2 = min(iw, int(cx + w / 2))
        y2 = min(ih, int(cy + h / 2))
        return rotated[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None

    def _on_letter_clicked(self, name):
        if self._mode == "assignment":
            # Assign current crop to letter
            img = self._load_cv(self._current_path)
            filtered = self._apply_filters(img)
            x, y, w, h = [int(v) for v in self._selected_rect]
            crop = self._extract_rotated(filtered, x, y, w, h, self._selected_angle)
            if crop is None:
                return
            if name not in self._assigned:
                self._assigned[name] = []
            self._assigned[name].append(crop)
            tile = self._alphabet_bar.tile(name)
            if tile:
                tile.set_has_images(True)
            self._update_create_font_action()
            entry = next((e for e in HEBREW_ALPHABET if e[0] == name), None)
            lbl = entry[2] if entry else name
            filled = sum(1 for n, _, _ in HEBREW_ALPHABET if self._assigned.get(n))
            self._status.showMessage(
                f"✓ Assigned to {lbl} — {len(self._assigned[name])} sample(s)  "
                f"({filled}/{len(HEBREW_ALPHABET)} letters filled)"
            )

        elif self._mode == "normal":
            self._enter_preview(name)

    def _enter_preview(self, name):
        entry = next((e for e in HEBREW_ALPHABET if e[0] == name), None)
        if not entry:
            return
        self._mode = "preview"
        self._preview_letter = name
        images = self._assigned.get(name, [])
        self._letter_preview.populate(entry[1], entry[2], images)
        # swap: hide viewer scroll, show preview
        self._letter_preview.show()
        self._scroll.hide()
        self._alphabet_bar._return_btn.setEnabled(True)
        self._status.showMessage(
            f"Preview: {entry[2]} ({entry[1]}) — {len(images)} sample(s)  |  "
            f"right-click or ← Back to exit"
        )

    def _exit_preview(self):
        if self._mode != "preview":
            return
        self._mode = "preview_exit"
        self._letter_preview.hide()
        self._scroll.show()
        self._alphabet_bar._return_btn.setEnabled(False)
        self._preview_letter = None
        self._mode = "normal"
        self._refresh_view()

    def _on_preview_remove(self, index):
        name = self._preview_letter
        if name not in self._assigned:
            return
        del self._assigned[name][index]
        # update tile indicator
        tile = self._alphabet_bar.tile(name)
        if tile:
            tile.set_has_images(len(self._assigned[name]) > 0)
        self._update_create_font_action()
        entry = next((e for e in HEBREW_ALPHABET if e[0] == name), None)
        if entry:
            self._letter_preview.populate(entry[1], entry[2], self._assigned[name])
        self._status.showMessage(
            f"Removed sample — {len(self._assigned[name])} remaining"
        )

    def _update_create_font_action(self):
        all_filled = all(
            bool(self._assigned.get(name))
            for name, _, _ in HEBREW_ALPHABET
        )
        self._create_font_act.setEnabled(all_filled)

    def _create_font(self):
        missing = [label for name, _, label in HEBREW_ALPHABET
                   if not self._assigned.get(name)]
        if missing:
            self._status.showMessage(f"Missing letters: {', '.join(missing)}")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Font As", "hebrew_handwritten.ttf",
            "TrueType Font (*.ttf)"
        )
        if not out_path:
            return

        self._status.showMessage("Building font…")
        QApplication.processEvents()

        try:
            self._build_ttf(out_path)
            self._status.showMessage(f"Font saved: {out_path}")
        except Exception as ex:
            self._status.showMessage(f"Font creation failed: {ex}")

    def _build_ttf(self, out_path):
        from fontTools.fontBuilder import FontBuilder
        from fontTools.pens.ttGlyphPen import TTGlyphPen

        UPM        = 1000
        ASCENDER   = 800
        DESCENDER  = -200
        CAP_HEIGHT = 600
        BASELINE   = 100

        cmap_map   = {0x0020: "space"}
        glyph_order = [".notdef", "space"]
        glyphs     = {}
        metrics    = {}

        # .notdef — empty box
        pen = TTGlyphPen(None)
        pen.moveTo((50, 0));  pen.lineTo((50, 700))
        pen.lineTo((450, 700)); pen.lineTo((450, 0)); pen.closePath()
        pen.moveTo((400, 50)); pen.lineTo((400, 650))
        pen.lineTo((100, 650)); pen.lineTo((100, 50)); pen.closePath()
        glyphs[".notdef"] = pen.glyph(); metrics[".notdef"] = (500, 50)

        glyphs["space"] = TTGlyphPen(None).glyph(); metrics["space"] = (250, 0)

        for name, char, _ in HEBREW_ALPHABET:
            glyph_order.append(name)
            cmap_map[ord(char)] = name
            imgs = self._assigned.get(name, [])
            img  = imgs[0] if imgs else None

            if img is None or img.size == 0:
                glyphs[name] = TTGlyphPen(None).glyph()
                metrics[name] = (500, 0)
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) \
                   if len(img.shape) == 3 else img
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            ih, iw   = binary.shape
            scale    = CAP_HEIGHT / ih
            contours, hierarchy = cv2.findContours(
                binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            pen = TTGlyphPen(None)
            if hierarchy is not None and len(contours) > 0:
                hierarchy = hierarchy[0]
                for i, contour in enumerate(contours):
                    eps   = max(1.0, 0.008 * cv2.arcLength(contour, True))
                    approx = cv2.approxPolyDP(contour, eps, True).squeeze()
                    if approx.ndim == 1:
                        approx = approx.reshape(1, 2)
                    if len(approx) < 3:
                        continue
                    pts = [(int(p[0] * scale),
                            int((ih - p[1]) * scale + BASELINE))
                           for p in approx]
                    is_hole = hierarchy[i][3] != -1
                    if is_hole:
                        pts = pts[::-1]
                    pen.moveTo(pts[0])
                    for p in pts[1:]:
                        pen.lineTo(p)
                    pen.closePath()

            glyphs[name]  = pen.glyph()
            adv = int(iw * scale + 60)
            metrics[name] = (adv, 20)

        fb = FontBuilder(UPM, isTTF=True)
        fb.setupGlyphOrder(glyph_order)
        fb.setupCharacterMap(cmap_map)
        fb.setupGlyf(glyphs)
        fb.setupHorizontalMetrics(metrics)
        fb.setupHorizontalHeader(ascent=ASCENDER, descent=DESCENDER)
        fb.setupNameTable({
            "familyName":           "HebrewHandwritten",
            "styleName":            "Regular",
            "fullName":             "HebrewHandwritten",
            "psName":               "HebrewHandwritten-Regular",
            "version":              "Version 1.0",
            "uniqueFontIdentifier": "HebrewHandwritten",
        })
        fb.setupOS2(sTypoAscender=ASCENDER, sTypoDescender=DESCENDER,
                    sCapHeight=CAP_HEIGHT, fsType=0)
        fb.setupPost()
        fb.setupHead(unitsPerEm=UPM)
        fb.font.save(out_path)

    def _merge_layers_or(self):
        panel = self._sidebar._layers_panel
        paths = []
        for i in range(panel.list_widget.count()):
            item = panel.list_widget.item(i)
            w = panel.list_widget.itemWidget(item)
            if w:
                paths.append(w.path)

        if len(paths) < 2:
            self._status.showMessage("Need at least 2 layers to merge.")
            return

        imgs = []
        for path in paths:
            img = cv2.imread(path)
            if img is not None:
                imgs.append(img)

        if not imgs:
            return

        # Resize all to the first layer's dimensions
        h, w = imgs[0].shape[:2]
        result = imgs[0].copy()
        for img in imgs[1:]:
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            # Pixel-wise minimum = OR in ink space (darkest pixel wins)
            result = np.minimum(result, img)

        os.makedirs(LAYERS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"layer_merged_{timestamp}.png"
        out_path = os.path.join(LAYERS_DIR, filename)
        cv2.imwrite(out_path, result)

        self._sidebar.add_layer(out_path)
        self._status.showMessage(
            f"Merged {len(imgs)} layers with OR → {filename}")

    def _save_layer(self):
        if not self._current_path:
            self._status.showMessage("No image loaded to save.")
            return
        img = self._load_cv(self._current_path)
        if img is None:
            return
        filtered = self._apply_filters(img)
        result = filtered

        os.makedirs(LAYERS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"layer_{timestamp}.png"
        path = os.path.join(LAYERS_DIR, filename)
        cv2.imwrite(path, result)

        self._sidebar.add_layer(path)
        self._status.showMessage(f"Layer saved: {filename}")

    def _open_settings(self):
        if self._settings_dialog is None or not self._settings_dialog.isVisible():
            self._settings_dialog = SettingsDialog(
                self._seg_settings, self._on_settings_changed, self)
        self._settings_dialog.show()
        self._settings_dialog.raise_()

    def _on_settings_changed(self):
        save_settings(self._seg_settings)
        self._refresh_view()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Font Generator")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
