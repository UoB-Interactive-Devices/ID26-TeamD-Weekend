import os
import signal
import sys
from pathlib import Path

os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.window=false")
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication

SOFTWARE_DIR = Path(__file__).resolve().parent.parent
if str(SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(SOFTWARE_DIR))


def main():
    from app.main_window import DemoMainWindow

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
