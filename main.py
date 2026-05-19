import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QScrollArea,
    QFileDialog, QToolBar, QStatusBar, QSizePolicy,
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QSplitter, QFrame
)
from PyQt6.QtGui import QPixmap, QAction, QKeySequence
from PyQt6.QtCore import Qt, QSize


THUMB_SIZE = 64
SIDEBAR_WIDTH = 220

# Color palette — warm neutral (Claude-inspired)
C_BG          = "#f9f6f1"   # main background
C_SIDEBAR     = "#f0ece4"   # sidebar background
C_SIDEBAR_HDR = "#e8e2d8"   # sidebar header
C_BORDER      = "#ddd8cf"   # borders / dividers
C_TEXT        = "#2d2a25"   # primary text
C_TEXT_MUTED  = "#8a8078"   # secondary text
C_SELECTED    = "#c8956c"   # selection accent (warm orange)
C_SELECTED_BG = "#f0e4d8"   # selected item background
C_HOVER       = "#e8e2d8"   # hover background
C_BTN_ADD     = "#c8956c"   # add button
C_BTN_ADD_H   = "#b8855c"   # add button hover
C_BTN_REM     = "#b5705a"   # remove button
C_BTN_REM_H   = "#a0604a"   # remove button hover


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
        thumb.setPixmap(
            pixmap.scaled(THUMB_SIZE, THUMB_SIZE,
                          Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
        )
        thumb.setStyleSheet(f"border: 1px solid {C_BORDER}; background: {C_BG};")

        name = QLabel(os.path.basename(path))
        name.setWordWrap(True)
        name.setStyleSheet("font-size: 12px;")
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

        # Header
        header = QLabel("  Images")
        header.setFixedHeight(36)
        header.setStyleSheet(f"background: {C_SIDEBAR_HDR}; color: {C_TEXT_MUTED}; font-size: 12px; font-weight: bold; border-bottom: 1px solid {C_BORDER};")
        layout.addWidget(header)

        # List
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(f"""
            QListWidget {{
                background: {C_SIDEBAR};
                border: none;
            }}
            QListWidget::item {{
                border-bottom: 1px solid {C_BORDER};
                color: {C_TEXT};
            }}
            QListWidget::item:selected {{
                background: {C_SELECTED_BG};
                border-left: 3px solid {C_SELECTED};
            }}
            QListWidget::item:hover:!selected {{
                background: {C_HOVER};
            }}
        """)
        self.list_widget.setSpacing(2)
        self.list_widget.currentItemChanged.connect(self._on_item_changed)
        layout.addWidget(self.list_widget)

        # Buttons
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
        return f"""
            QPushButton {{
                background: {bg}; color: #eee; border: none;
                border-radius: 4px; padding: 4px 8px; font-size: 12px;
            }}
            QPushButton:hover {{ background: {hover}; }}
            QPushButton:pressed {{ background: {bg}; }}
        """

    def add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Image Files", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
        )
        for path in paths:
            self._add_item(path)

    def _add_item(self, path):
        # Skip duplicates
        for i in range(self.list_widget.count()):
            widget = self.list_widget.itemWidget(self.list_widget.item(i))
            if widget and widget.path == path:
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

        # Auto-select first added image
        if self.list_widget.count() == 1:
            self.list_widget.setCurrentItem(item)

    def remove_selected(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            self.list_widget.takeItem(row)
            # Select the next available item
            count = self.list_widget.count()
            if count > 0:
                self.list_widget.setCurrentRow(min(row, count - 1))
            else:
                self.on_select(None)

    def _on_item_changed(self, current, previous):
        if current is None:
            self.on_select(None)
            return
        path = current.data(Qt.ItemDataRole.UserRole)
        self.on_select(path)


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

    def load(self, path):
        if path is None:
            self._pixmap = None
            self._show_placeholder()
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return
        self._pixmap = pixmap
        self.setStyleSheet(f"background: {C_BG};")
        self._fit()

    def _fit(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled)

    def resizeEvent(self, event):
        self._fit()
        super().resizeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Font Generator")
        self.resize(1100, 700)
        self.setStyleSheet(f"background: {C_BG};")

        self._build_menu()
        self._build_central()
        self._build_statusbar()

    def _build_menu(self):
        menu = self.menuBar()
        menu.setStyleSheet(f"background: {C_SIDEBAR_HDR}; color: {C_TEXT};")

        file_menu = menu.addMenu("&File")

        open_action = QAction("&Add Images...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(lambda: self._sidebar.add_images())
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _build_central(self):
        self._viewer = ImageViewer()

        scroll = QScrollArea()
        scroll.setWidget(self._viewer)
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setStyleSheet(f"border: none; background: {C_BG};")

        self._sidebar = Sidebar(on_select=self._on_image_selected)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._sidebar)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {C_BORDER}; }}")

        self.setCentralWidget(splitter)

    def _build_statusbar(self):
        self._status = QStatusBar()
        self._status.setStyleSheet(f"background: {C_SIDEBAR_HDR}; color: {C_TEXT_MUTED}; font-size: 12px; border-top: 1px solid {C_BORDER};")
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

    def _on_image_selected(self, path):
        self._viewer.load(path)
        if path:
            pixmap = QPixmap(path)
            w, h = pixmap.width(), pixmap.height()
            self._status.showMessage(f"{path}  —  {w}×{h}px")
            self.setWindowTitle(f"Font Generator — {os.path.basename(path)}")
        else:
            self._status.showMessage("Ready")
            self.setWindowTitle("Font Generator")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Font Generator")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
