import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QScrollArea,
    QFileDialog, QStatusBar, QSizePolicy,
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QSplitter, QButtonGroup
)
from PyQt6.QtGui import (
    QPixmap, QIcon, QAction, QKeySequence, QPainter, QColor,
    QImage, QPen, QBrush, QFont
)
from PyQt6.QtCore import Qt, QSize, QRect

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


# ── Segmentation functions ────────────────────────────────────────────────

def _to_binary(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def detect_lines(img):
    binary = _to_binary(img)
    # Dilate horizontally to merge nearby ink into solid line bands
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    dilated = cv2.dilate(binary, kernel)

    # Horizontal projection profile
    h_proj = np.sum(dilated, axis=1)
    threshold = np.max(h_proj) * 0.05

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


def detect_characters(img, line_rects):
    binary = _to_binary(img)
    min_area = max(20, int(img.shape[0] * img.shape[1] * 0.00005))
    rects = []
    for (lx, ly, lw, lh) in line_rects:
        region = binary[ly:ly + lh, lx:lx + lw]
        n, _, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area >= min_area and w > 2 and h > 2:
                rects.append((lx + x, ly + y, w, h))
    return rects


def draw_overlays(img, line_rects, char_rects, show_lines, show_chars):
    out = img.copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if show_lines:
        for (x, y, w, h) in line_rects:
            cv2.rectangle(out, (x, y), (x + w, y + h), (34, 160, 34), 2)
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


def icon_lines(p, s):
    # Three horizontal rectangles representing text lines
    p.setPen(Qt.PenStyle.NoPen)
    for i, y in enumerate([4, 11, 18]):
        color = QColor("#22a022") if i == 1 else QColor("#aaddaa")
        p.setBrush(QBrush(color))
        p.drawRect(QRect(2, y, s - 4, 5))


def icon_characters(p, s):
    # Small rectangles representing individual characters
    p.setPen(QPen(QColor("#2264e0"), 1))
    p.setBrush(Qt.BrushStyle.NoBrush)
    for x, y, w, h in [(2, 4, 7, 18), (11, 4, 5, 18), (18, 4, 6, 18)]:
        p.drawRect(QRect(x, y, w, h))


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
    def __init__(self, on_filter_changed, on_lines_toggled, on_chars_toggled, parent=None):
        super().__init__(parent)
        self.on_filter_changed = on_filter_changed
        self.setFixedHeight(52)
        self.setStyleSheet(f"background: {C_FILTER_BAR}; border-bottom: 1px solid {C_BORDER};")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(6)

        # ── Filter buttons (exclusive) ──
        label = QLabel("Filters:")
        label.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 11px; font-weight: bold;")
        layout.addWidget(label)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        for i, (name, icon_fn, fn, tip) in enumerate(FILTERS):
            btn = QPushButton()
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            btn.setFixedSize(38, 38)
            btn.setIcon(QIcon(_make_icon(icon_fn)))
            btn.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
            btn.setToolTip(f"<b>{name}</b><br>{tip}")
            btn.setStyleSheet(self._btn_style())
            self._group.addButton(btn, i)
            layout.addWidget(btn)

        self._group.idToggled.connect(self._on_toggled)

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

        self._btn_lines = self._make_seg_btn(
            icon_lines, "<b>Find Lines</b><br>Detect text line boundaries", on_lines_toggled
        )
        self._btn_chars = self._make_seg_btn(
            icon_characters, "<b>Find Characters</b><br>Detect individual character boundaries", on_chars_toggled
        )
        layout.addWidget(self._btn_lines)
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
        if checked:
            self.on_filter_changed(FILTERS[btn_id][2])

    def current_filter(self):
        idx = self._group.checkedId()
        return FILTERS[idx][2] if idx >= 0 else filter_original


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
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap = None
        self._show_placeholder()

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


# ── Main window ────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Font Generator")
        self.resize(1100, 720)
        self.setStyleSheet(f"background: {C_BG};")

        self._current_path = None
        self._cv_cache = {}            # path → original cv2 image
        self._active_filter = filter_original
        self._line_rects = []
        self._char_rects = []
        self._show_lines = False
        self._show_chars = False

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

    def _build_central(self):
        self._viewer = ImageViewer()

        scroll = QScrollArea()
        scroll.setWidget(self._viewer)
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setStyleSheet(f"border: none; background: {C_BG};")

        self._filter_bar = FilterBar(
            on_filter_changed=self._on_filter_changed,
            on_lines_toggled=self._on_lines_toggled,
            on_chars_toggled=self._on_chars_toggled,
        )
        self._sidebar = Sidebar(on_select=self._on_image_selected)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self._filter_bar)
        right_layout.addWidget(scroll)

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

    def _refresh_view(self):
        if not self._current_path:
            return
        img = self._load_cv(self._current_path)
        if img is None:
            return
        result = self._active_filter(img)
        result = draw_overlays(result, self._line_rects, self._char_rects,
                               self._show_lines, self._show_chars)
        self._viewer.set_pixmap(cv_to_pixmap(result))
        h, w = img.shape[:2]
        filter_name = next(f[0] for f in FILTERS if f[2] == self._active_filter)
        parts = [os.path.basename(self._current_path), f"{w}×{h}px",
                 f"Filter: {filter_name}"]
        if self._show_lines:
            parts.append(f"Lines: {len(self._line_rects)}")
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
        if self._show_lines:
            self._line_rects = detect_lines(self._load_cv(path))
        if self._show_chars:
            self._char_rects = detect_characters(self._load_cv(path), self._line_rects)
        self._refresh_view()
        self.setWindowTitle(f"Font Generator — {os.path.basename(path)}")

    def _on_filter_changed(self, filter_fn):
        self._active_filter = filter_fn
        self._refresh_view()

    def _on_lines_toggled(self, checked):
        self._show_lines = checked
        if self._current_path and checked:
            self._line_rects = detect_lines(self._load_cv(self._current_path))
            if self._show_chars:
                self._char_rects = detect_characters(
                    self._load_cv(self._current_path), self._line_rects)
        elif not checked:
            self._line_rects = []
            self._char_rects = []
        self._refresh_view()

    def _on_chars_toggled(self, checked):
        self._show_chars = checked
        if self._current_path and checked:
            if not self._line_rects:
                self._line_rects = detect_lines(self._load_cv(self._current_path))
                self._show_lines = True
            self._char_rects = detect_characters(
                self._load_cv(self._current_path), self._line_rects)
        elif not checked:
            self._char_rects = []
        self._refresh_view()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Font Generator")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
