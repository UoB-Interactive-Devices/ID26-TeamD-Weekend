import os
import signal
import sys

os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.window=false")

from app.main_window import DemoMainWindow
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    window = DemoMainWindow()
    window.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    window.show()
    window.raise_()
    window.activateWindow()

    def release_startup_topmost():
        window.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)
        window.show()

    QTimer.singleShot(700, release_startup_topmost)

    heartbeat = QTimer()
    heartbeat.timeout.connect(lambda: None)
    heartbeat.start(200)

    def handle_sigint(_signal_number, _frame):
        window.shutdown()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        return app.exec()
    except KeyboardInterrupt:
        window.shutdown()
        return 0


if __name__ == "__main__":
    sys.exit(main())
