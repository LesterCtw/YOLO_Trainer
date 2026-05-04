from __future__ import annotations

import argparse
import sys

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow


APP_NAME = "YOLO Trainer"
MINIMUM_WINDOW_SIZE = QSize(960, 640)


def build_main_window() -> QMainWindow:
    window = QMainWindow()
    window.setWindowTitle(APP_NAME)
    window.setMinimumSize(MINIMUM_WINDOW_SIZE)

    status = QLabel("YOLO Trainer scaffold is ready.")
    status.setObjectName("appStatusLabel")
    status.setMargin(24)
    window.setCentralWidget(status)

    return window


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yolo-trainer")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Start the Qt app in smoke-test mode and exit immediately.",
    )
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = build_main_window()
    window.show()
    app.processEvents()

    if args.smoke:
        print("YOLO Trainer app smoke OK")
        window.close()
        app.processEvents()
        return 0

    return app.exec()


def main() -> int:
    return run()
