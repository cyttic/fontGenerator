import sys
import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QScrollArea,
    QFileDialog, QStatusBar, QSizePolicy,
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QSplitter, QButtonGroup,
    QDialog, QGroupBox, QFormLayout, QSlider, QComboBox,
    QDialogButtonBox, QFrame, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtGui import (
    QPixmap, QIcon, QAction, QKeySequence, QPainter, QColor,
    QImage, QPen, QBrush, QFont
)
from PyQt6.QtCore import Qt, QSize, QRect, pyqtSignal

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


# ── Segmentation settings ─────────────────────────────────────────────────

@dataclass
class SegmentationSettings:
    threshold_method: str = "otsu"   # "otsu", "adaptive", "manual"
    manual_threshold: int = 128
    line_kernel_width: int = 40
    line_threshold_pct: int = 5      # percent 1-30
    min_area: int = 20
    min_width: int = 2
    min_height: int = 2
    max_width: int = 500
    max_height: int = 500
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
            if (area >= s.min_area and
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
    ("Deskew",     icon_deskew,    filter_deskew,    "Auto deskew correction"),
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


class Sidebar(QWidget):
    def __init__(self, on_select, parent=None):
        super().__init__(parent)
        self.on_select = on_select
        self.setFixedWidth(SIDEBAR_WIDTH)
        self.setStyleSheet(f"background: {C_SIDEBAR};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QLabel("  Images")
        header.setFixedHeight(36)
        header.setStyleSheet(
            f"background: {C_SIDEBAR_HDR}; color: {C_TEXT_MUTED}; "
            f"font-size: 12px; font-weight: bold; border-bottom: 1px solid {C_BORDER};"
        )
        layout.addWidget(header)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(f"""
            QListWidget {{ background: {C_SIDEBAR}; border: none; }}
            QListWidget::item {{ border-bottom: 1px solid {C_BORDER}; color: {C_TEXT}; }}
            QListWidget::item:selected {{
                background: {C_SELECTED_BG}; border-left: 3px solid {C_SELECTED};
            }}
            QListWidget::item:hover:!selected {{ background: {C_HOVER}; }}
        """)
        self.list_widget.setSpacing(2)
        self.list_widget.currentItemChanged.connect(self._on_item_changed)
        layout.addWidget(self.list_widget)

        btn_bar = QWidget()
        btn_bar.setFixedHeight(44)
        btn_bar.setStyleSheet(f"background: {C_SIDEBAR_HDR}; border-top: 1px solid {C_BORDER};")
        btn_layout = QHBoxLayout(btn_bar)
        btn_layout.setContentsMargins(8, 6, 8, 6)
        btn_layout.setSpacing(8)

        btn_add = QPushButton("+ Add")
        btn_add.setStyleSheet(self._btn_style(C_BTN_ADD, C_BTN_ADD_H))
        btn_add.clicked.connect(self.add_images)

        btn_remove = QPushButton("− Remove")
        btn_remove.setStyleSheet(self._btn_style(C_BTN_REM, C_BTN_REM_H))
        btn_remove.clicked.connect(self.remove_selected)

        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_remove)
        layout.addWidget(btn_bar)

    def _btn_style(self, bg, hover):
        return (f"QPushButton {{ background: {bg}; color: #fff; border: none; "
                f"border-radius: 4px; padding: 4px 8px; font-size: 12px; }}"
                f"QPushButton:hover {{ background: {hover}; }}")

    def add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Image Files", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
        )
        for path in paths:
            self._add_item(path)

    def _add_item(self, path):
        for i in range(self.list_widget.count()):
            w = self.list_widget.itemWidget(self.list_widget.item(i))
            if w and w.path == path:
                return
        pixmap = QPixmap(path)
        if pixmap.isNull():
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

    def _on_item_changed(self, current, previous):
        path = current.data(Qt.ItemDataRole.UserRole) if current else None
        self.on_select(path)


# ── Image viewer ───────────────────────────────────────────────────────────

class ImageViewer(QLabel):
    pixel_clicked       = pyqtSignal(float, float)   # image x, y  (left click)
    right_clicked       = pyqtSignal()
    drag_started        = pyqtSignal(float, float)   # image x, y
    drag_moved          = pyqtSignal(float, float)   # delta image dx, dy

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap       = None
        self._origin       = (0, 0)   # top-left of displayed region in image coords
        self._drag_last    = None
        self._show_placeholder()

    def set_display_origin(self, ox, oy):
        self._origin = (ox, oy)

    def widget_to_image(self, wx, wy):
        if not self._pixmap or self._pixmap.isNull():
            return -1.0, -1.0
        pw, ph = self._pixmap.width(), self._pixmap.height()
        vw, vh = self.width(), self.height()
        scale  = min(vw / pw, vh / ph)
        offx   = (vw - pw * scale) / 2
        offy   = (vh - ph * scale) / 2
        ix = (wx - offx) / scale + self._origin[0]
        iy = (wy - offy) / scale + self._origin[1]
        return ix, iy

    def _show_placeholder(self):
        self.setText("Add images using the sidebar\n\nCtrl+O  or  click  + Add")
        self.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 16px; background: {C_BG};")

    def set_pixmap(self, pixmap):
        if pixmap is None or pixmap.isNull():
            self._pixmap = None
            self._show_placeholder()
            return
        self._pixmap = pixmap
        self.setStyleSheet(f"background: {C_BG};")
        self._fit()

    def _fit(self):
        if self._pixmap:
            self.setPixmap(self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

    def resizeEvent(self, event):
        self._fit()
        super().resizeEvent(event)

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

        hint = QLabel("Click a letter below\nto assign this crop")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet(
            f"color: {C_TEXT_MUTED}; font-size: 11px; "
            f"border: 1px dashed {C_BORDER}; border-radius: 4px; padding: 8px;"
        )
        layout.addWidget(hint)
        layout.addStretch()

    def _section(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 11px; font-weight: bold;")
        return lbl

    def _make_slider(self):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(4, 800)
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

    def set_rect(self, w, h):
        self._w_slider.blockSignals(True)
        self._h_slider.blockSignals(True)
        self._w_slider.setValue(int(w))
        self._h_slider.setValue(int(h))
        self._w_label.setText(f"{int(w)} px")
        self._h_label.setText(f"{int(h)} px")
        self._w_slider.blockSignals(False)
        self._h_slider.blockSignals(False)

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
        self._seg_settings = SegmentationSettings()
        self._settings_dialog = None
        # assignment mode
        self._mode          = "normal"
        self._selected_rect = None     # [x, y, w, h] mutable
        self._zoom_origin   = (0, 0)   # crop top-left in image coords
        self._assigned      = {}       # letter_name → list of np.ndarray

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
        file_menu.addSeparator()

        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        detect_menu = menu.addMenu("&Detection")
        settings_act = QAction("&Settings...", self)
        settings_act.setShortcut("Ctrl+,")
        settings_act.triggered.connect(self._open_settings)
        detect_menu.addAction(settings_act)

    def _build_central(self):
        self._viewer = ImageViewer()
        self._viewer.pixel_clicked.connect(self._on_viewer_click)
        self._viewer.right_clicked.connect(self._exit_assignment)
        self._viewer.drag_moved.connect(self._on_viewer_drag)

        scroll = QScrollArea()
        scroll.setWidget(self._viewer)
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setStyleSheet(f"border: none; background: {C_BG};")

        self._assign_panel = AssignmentPanel()
        self._assign_panel.width_changed.connect(self._on_assign_width)
        self._assign_panel.height_changed.connect(self._on_assign_height)
        self._assign_panel.hide()

        view_row = QWidget()
        view_row_layout = QHBoxLayout(view_row)
        view_row_layout.setContentsMargins(0, 0, 0, 0)
        view_row_layout.setSpacing(0)
        view_row_layout.addWidget(scroll)
        view_row_layout.addWidget(self._assign_panel)

        self._filter_bar = FilterBar(
            on_filter_changed=self._on_filter_changed,
            on_chars_toggled=self._on_chars_toggled,
        )
        self._sidebar = Sidebar(on_select=self._on_image_selected)
        self._alphabet_bar = HebrewAlphabetBar(on_letter_clicked=self._on_letter_clicked)
        self._alphabet_bar._return_btn.setEnabled(False)
        self._alphabet_bar._return_btn.clicked.connect(self._exit_assignment)

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
        self._assign_panel.set_rect(self._selected_rect[2], self._selected_rect[3])
        self._assign_panel.show()
        self._alphabet_bar._return_btn.setEnabled(True)
        self._viewer.setCursor(Qt.CursorShape.SizeAllCursor)
        self._render_assignment()

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
        self._refresh_view()

    def _render_assignment(self):
        if not self._current_path or self._selected_rect is None:
            return
        img = self._load_cv(self._current_path)
        filtered = self._apply_filters(img)
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
        rx, ry = x - x1, y - y1
        # Dim everything outside the selection
        overlay = crop.copy()
        overlay[:, :] = (overlay * 0.4).astype(np.uint8)
        overlay[ry:ry+h, rx:rx+w] = crop[ry:ry+h, rx:rx+w]
        cv2.rectangle(overlay, (rx, ry), (rx+w, ry+h), (30, 120, 220), 2)

        self._viewer.set_pixmap(cv_to_pixmap(overlay))

        # Preview in panel
        char_crop = filtered[y:y+h, x:x+w] if h > 0 and w > 0 else None
        self._assign_panel.set_preview(char_crop)
        self._status.showMessage(
            f"Assignment mode  —  rect {w}×{h}px  —  drag to reposition  |  "
            f"right-click or ← Back to exit"
        )

    def _on_viewer_drag(self, dx, dy):
        if self._mode != "assignment" or self._selected_rect is None:
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

    def _on_letter_clicked(self, name):
        if self._mode != "assignment":
            entry = next((e for e in HEBREW_ALPHABET if e[0] == name), None)
            if entry:
                self._status.showMessage(
                    f"Letter: {entry[2]} ({entry[1]}) — enable character detection first")
            return

        # Assign current crop to letter
        img = self._load_cv(self._current_path)
        filtered = self._apply_filters(img)
        x, y, w, h = [int(v) for v in self._selected_rect]
        ih, iw = filtered.shape[:2]
        x = max(0, min(iw - 1, x));  y = max(0, min(ih - 1, y))
        x2 = min(iw, x + w);         y2 = min(ih, y + h)
        crop = filtered[y:y2, x:x2].copy()

        if name not in self._assigned:
            self._assigned[name] = []
        self._assigned[name].append(crop)

        tile = self._alphabet_bar.tile(name)
        if tile:
            tile.set_has_images(True)

        entry = next((e for e in HEBREW_ALPHABET if e[0] == name), None)
        label = entry[2] if entry else name
        self._status.showMessage(
            f"✓ Assigned to {label} — total: {len(self._assigned[name])} sample(s)"
        )

    def _open_settings(self):
        if self._settings_dialog is None:
            self._settings_dialog = SettingsDialog(
                self._seg_settings, self._on_settings_changed, self)
        self._settings_dialog.show()
        self._settings_dialog.raise_()

    def _on_settings_changed(self):
        self._refresh_view()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Font Generator")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
