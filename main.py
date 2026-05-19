import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QScrollArea,
    QFileDialog, QToolBar, QStatusBar, QSizePolicy, QWidget, QHBoxLayout
)
from PyQt6.QtGui import QPixmap, QAction, QIcon, QKeySequence
from PyQt6.QtCore import Qt, QSize


class ImageViewer(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap = None
        self._show_placeholder()

    def _show_placeholder(self):
        self.setText("Open an image file to get started\n\nFile → Open Image  (Ctrl+O)")
        self.setStyleSheet("color: #888; font-size: 16px;")

    def load_pixmap(self, pixmap):
        self._pixmap = pixmap
        self.setStyleSheet("")
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
        self.resize(900, 650)

        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()

    def _build_menu(self):
        menu = self.menuBar()

        file_menu = menu.addMenu("&File")

        open_action = QAction("&Open Image...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.setStatusTip("Open one or more image files")
        open_action.triggered.connect(self.open_images)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _build_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("Open Image", self)
        open_action.setStatusTip("Open one or more image files")
        open_action.triggered.connect(self.open_images)
        toolbar.addAction(open_action)

    def _build_central(self):
        self._viewer = ImageViewer()

        scroll = QScrollArea()
        scroll.setWidget(self._viewer)
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.setCentralWidget(scroll)

    def _build_statusbar(self):
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

    def open_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Image Files",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
        )
        if not paths:
            return

        # Load and display the first selected image
        pixmap = QPixmap(paths[0])
        if pixmap.isNull():
            self._status.showMessage(f"Failed to load: {paths[0]}")
            return

        self._viewer.load_pixmap(pixmap)

        filename = paths[0].split("/")[-1]
        w, h = pixmap.width(), pixmap.height()
        count = len(paths)
        msg = f"{filename}  —  {w}×{h}px"
        if count > 1:
            msg += f"  (+{count - 1} more selected)"
        self._status.showMessage(msg)
        self.setWindowTitle(f"Font Generator — {filename}")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Font Generator")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
