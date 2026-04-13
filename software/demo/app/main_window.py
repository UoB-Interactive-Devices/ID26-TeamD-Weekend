from __future__ import annotations

import copy
import time

import pyautogui
from PyQt6.QtCore import QMetaObject, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QStatusBar

from .actions import (
    AVAILABLE_MACROS,
    DEFAULT_GESTURE_MAPPING,
    PHOTO_SORT_CATEGORY_MACROS,
    PHOTO_SORT_UNDO_MACRO,
    ActionController,
)
from .config import (
    FINE_TUNE_CLASSES,
    PHOTO_SORT_GESTURE_COOLDOWN_SECONDS,
    SAMPLES_PER_CLASS,
)
from .pages import CalibrationPage, DashboardPage, FineTuningPage, MappingPage
from .serial_worker import SerialInferenceWorker
from .speech_worker import SpeechMappingWorker
from .styles import APP_STYLESHEET
from .windows_notification import show_windows_toast


class DemoMainWindow(QMainWindow):
    request_calibration = pyqtSignal()
    request_collection = pyqtSignal(list, int)
    request_collect_class = pyqtSignal()
    request_fine_tune = pyqtSignal()
    request_inference = pyqtSignal()
    request_hand_tap = pyqtSignal()
    request_notification_tap = pyqtSignal()
    request_category_tap = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Teensy Gesture Visualiser")
        self.setMinimumSize(1180, 760)

        self.setStyleSheet(APP_STYLESHEET)
        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0

        self._shutting_down = False
        self._listening = False
        self._speech_thread: QThread | None = None
        self._speech_worker: SpeechMappingWorker | None = None
        self._fine_tune_started = False
        self._fine_tune_collecting = False
        self._fine_tune_class_index = 0
        self._fine_tune_classes = list(FINE_TUNE_CLASSES)
        self._last_dashboard_gesture = "none"
        self._photo_sort_action_armed = True
        self._last_photo_sort_action_time = 0.0
        self._pending_gesture_sort_category: str | None = None

        self.custom_mapping = copy.deepcopy(DEFAULT_GESTURE_MAPPING)
        self.action_controller = ActionController()

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.calibration_page = CalibrationPage()
        self.fine_tuning_page = FineTuningPage()
        self.mapping_page = MappingPage()
        self.dashboard_page = DashboardPage()

        self.stack.addWidget(self.calibration_page)
        self.stack.addWidget(self.fine_tuning_page)
        self.stack.addWidget(self.mapping_page)
        self.stack.addWidget(self.dashboard_page)

        self._setup_shortcuts()

        self.dashboard_page.photo_sort_game.score_changed.connect(
            self.dashboard_page.set_score
        )
        self.dashboard_page.photo_sort_game.remaining_changed.connect(
            self.dashboard_page.set_remaining
        )
        self.dashboard_page.photo_sort_game.photo_sorted.connect(self._on_photo_sorted)
        self.dashboard_page.photo_sort_game.stack_completed.connect(
            self._on_photo_stack_completed
        )

        self.dashboard_page.set_score(self.dashboard_page.photo_sort_game.sorted_count)
        self.dashboard_page.set_remaining(
            self.dashboard_page.photo_sort_game.remaining_count
        )

        self._sync_mapping_views()

        self.worker_thread = QThread(self)
        self.worker = SerialInferenceWorker()
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.start)
        self.request_calibration.connect(self.worker.begin_calibration)
        self.request_collection.connect(self.worker.begin_collection)
        self.request_collect_class.connect(self.worker.collect_current_class)
        self.request_fine_tune.connect(self.worker.fine_tune_model)
        self.request_inference.connect(self.worker.begin_inference)
        self.request_hand_tap.connect(self.worker.tap_hand)
        self.request_notification_tap.connect(self.worker.notification_finger_tap)
        self.request_category_tap.connect(self.worker.category_group_tap)

        self.worker.connected.connect(self._on_worker_connected)
        self.worker.status_message.connect(self._on_status_message)
        self.worker.connection_error.connect(self._on_connection_error)

        self.worker.calibration_progress.connect(self.calibration_page.set_progress)
        self.worker.calibration_complete.connect(self._on_calibration_complete)

        self.worker.collection_instruction.connect(
            self.fine_tuning_page.set_instruction
        )
        self.worker.collection_progress.connect(
            self.fine_tuning_page.update_collection_progress
        )
        self.worker.collection_class_complete.connect(
            self._on_collection_class_complete
        )
        self.worker.collection_complete.connect(self._on_collection_complete)

        self.worker.fine_tuning_started.connect(
            self.fine_tuning_page.set_training_started
        )
        self.worker.fine_tuning_epoch.connect(
            self.fine_tuning_page.update_training_epoch
        )
        self.worker.fine_tuning_complete.connect(self._on_fine_tuning_complete)

        self.worker.inference_frame.connect(self._on_inference_frame)
        self.worker.stable_gesture.connect(self._on_stable_gesture)

        self.worker_thread.start()

        self._status_bar.showMessage("Initialising serial connection...")
        self.stack.setCurrentWidget(self.calibration_page)

    def _sync_mapping_views(self):
        self.mapping_page.set_mapping(self.custom_mapping)
        self.dashboard_page.set_mapping(self.custom_mapping)

    def _on_worker_connected(self, message: str):
        self._status_bar.showMessage(message)
        self.request_calibration.emit()

    def _on_status_message(self, message: str):
        self._status_bar.showMessage(message)
        self.mapping_page.set_status(message)

    def _on_connection_error(self, message: str):
        self._status_bar.showMessage(message)

    def _on_calibration_complete(self):
        self.stack.setCurrentWidget(self.fine_tuning_page)
        self._fine_tune_started = False
        self._fine_tune_collecting = False
        self._fine_tune_class_index = 0
        self.fine_tuning_page.set_step_hint("STEP 1 OF 7")
        first = self._fine_tune_classes[0]
        self.fine_tuning_page.prepare_next_gesture(
            first,
            class_index=1,
            class_total=len(self._fine_tune_classes),
            per_class_total=SAMPLES_PER_CLASS,
        )
        self.fine_tuning_page.set_instruction(
            "Calibration complete. Place your hand naturally. Press ENTER to close the hand and begin guided capture."
        )

    def _start_fine_tune_session(self):
        self._fine_tune_started = True
        self._fine_tune_collecting = False
        self._fine_tune_class_index = 0
        self.fine_tuning_page.set_step_hint("STEP 2 OF 7")
        self.request_collection.emit(list(self._fine_tune_classes), SAMPLES_PER_CLASS)
        first = self._fine_tune_classes[0].upper()
        self.fine_tuning_page.prepare_next_gesture(
            self._fine_tune_classes[0],
            class_index=1,
            class_total=len(self._fine_tune_classes),
            per_class_total=SAMPLES_PER_CLASS,
        )
        self.fine_tuning_page.set_instruction(
            f"Hand closing now. Prepare to lean {first}. Press ENTER to start collection."
        )

    def _start_current_class_collection(self):
        if self._fine_tune_class_index >= len(self._fine_tune_classes):
            return

        gesture = self._fine_tune_classes[self._fine_tune_class_index].upper()
        self._fine_tune_collecting = True
        self.fine_tuning_page.set_step_hint(
            f"STEP {self._fine_tune_class_index + 3} OF 7"
        )
        self.fine_tuning_page.set_instruction(
            f"Collecting {gesture}. Keep steady until all {SAMPLES_PER_CLASS} samples are complete."
        )
        self.request_collect_class.emit()

    def _on_collection_class_complete(self, gesture: str, completed: int, total: int):
        self._fine_tune_collecting = False
        self._fine_tune_class_index = completed

        if completed >= total:
            self.fine_tuning_page.set_instruction(
                "All gestures captured. Starting transfer learning now."
            )
            return

        next_gesture_raw = self._fine_tune_classes[self._fine_tune_class_index]
        next_gesture = next_gesture_raw.upper()
        self.fine_tuning_page.prepare_next_gesture(
            next_gesture_raw,
            class_index=self._fine_tune_class_index + 1,
            class_total=total,
            per_class_total=SAMPLES_PER_CLASS,
        )
        self.fine_tuning_page.set_instruction(
            f"Prepare {next_gesture}. Press ENTER to start collection."
        )

    def _on_collection_complete(self, total_samples: int):
        self.fine_tuning_page.set_training_message(
            f"Collected {total_samples} samples. Starting transfer learning."
        )
        self.fine_tuning_page.set_step_hint("STEP 7 OF 7")
        self.request_fine_tune.emit()

    def _on_fine_tuning_complete(self, success: bool, message: str):
        self.fine_tuning_page.set_training_message(message)
        self.request_inference.emit()

        self.stack.setCurrentWidget(self.mapping_page)
        if success:
            self.mapping_page.set_status(
                "Fine-tuning complete. Hold a gesture to remap, then press Enter to continue."
            )
        else:
            self.mapping_page.set_status(
                "Fine-tuning failed; continuing with available model. Hold a gesture to remap."
            )

    def _on_inference_frame(self, front, back, gesture: str, confidence: float):
        self.dashboard_page.heatmaps.update_matrices(front, back)
        self.dashboard_page.set_classification(gesture, confidence)

        self.mapping_page.set_status(
            f"Detected gesture: {gesture} ({confidence * 100:.1f}%). Press Enter for visualiser."
        )

        if self.stack.currentWidget() is not self.dashboard_page:
            self.action_controller.release_mouse()
            self.dashboard_page.set_action("none")
            self._last_dashboard_gesture = "none"
            self._photo_sort_action_armed = True
            return

        if gesture != self._last_dashboard_gesture:
            self._last_dashboard_gesture = gesture
            self._photo_sort_action_armed = True

        if gesture == "none":
            self.action_controller.release_mouse()
            self.dashboard_page.set_action("none")
            return

        macro = self.custom_mapping.get(gesture, "none")

        if macro in PHOTO_SORT_CATEGORY_MACROS:
            now = time.time()
            if (
                not self._photo_sort_action_armed
                or now - self._last_photo_sort_action_time
                < PHOTO_SORT_GESTURE_COOLDOWN_SECONDS
            ):
                self.dashboard_page.set_action("categorise_cooldown")
                return

            category = PHOTO_SORT_CATEGORY_MACROS[macro]
            sorted_ok = self.dashboard_page.photo_sort_game.sort_current_photo(
                category,
                animate=True,
            )
            self._photo_sort_action_armed = False
            self._last_photo_sort_action_time = now
            self._pending_gesture_sort_category = category if sorted_ok else None
            self.dashboard_page.set_action(macro if sorted_ok else "none")
            return

        if macro == PHOTO_SORT_UNDO_MACRO:
            now = time.time()
            if (
                not self._photo_sort_action_armed
                or now - self._last_photo_sort_action_time
                < PHOTO_SORT_GESTURE_COOLDOWN_SECONDS
            ):
                self.dashboard_page.set_action("categorise_cooldown")
                return

            undo_ok = self.dashboard_page.photo_sort_game.undo_last_sort()
            self._photo_sort_action_armed = False
            self._last_photo_sort_action_time = now
            self._pending_gesture_sort_category = None
            self.dashboard_page.set_action(macro if undo_ok else "none")
            return

        try:
            action_result = self.action_controller.apply_macro(macro)
        except pyautogui.FailSafeException:
            self._status_bar.showMessage(
                "PyAutoGUI fail-safe triggered by top-left corner. Shutting down."
            )
            self.shutdown()
            return

        if action_result.action_taken and macro == "click":
            self.request_hand_tap.emit()

        self.dashboard_page.set_action(macro)

    def _on_photo_sorted(self, _photo_path: str, category: str, remaining: int):
        if self._pending_gesture_sort_category == category:
            group_to_finger = {
                "group_1": 0,
                "group_2": 1,
                "group_3": 2,
                "group_4": 3,
            }
            finger_index = group_to_finger.get(category)
            if finger_index is not None:
                self.request_category_tap.emit(finger_index)
            self._pending_gesture_sort_category = None

        self.dashboard_page.set_action(f"categorise_{category}")
        group_label = category.replace("_", " ").title()
        self._status_bar.showMessage(
            f"Photo sorted into {group_label}. {remaining} remaining."
        )

    def _on_photo_stack_completed(self):
        self._status_bar.showMessage("All photos sorted.")

    def _on_stable_gesture(self, gesture: str):
        if self.stack.currentWidget() is not self.mapping_page:
            return

        if gesture not in self.custom_mapping:
            return

        if self._listening:
            return

        self.mapping_page.highlight_gesture(gesture)
        self._start_speech_mapping(gesture)

    def _start_speech_mapping(self, gesture: str):
        self._listening = True
        self.mapping_page.set_listening(True, gesture)
        self.mapping_page.set_status(
            f"Stable gesture {gesture.upper()} detected. Listening for spoken command..."
        )

        self._speech_thread = QThread(self)
        self._speech_worker = SpeechMappingWorker(gesture, AVAILABLE_MACROS)
        self._speech_worker.moveToThread(self._speech_thread)

        self._speech_thread.started.connect(self._speech_worker.run)
        self._speech_worker.listening_started.connect(self._on_listening_started)
        self._speech_worker.transcript_ready.connect(self._on_transcript_ready)
        self._speech_worker.mapping_ready.connect(self._on_mapping_ready)
        self._speech_worker.failed.connect(self._on_mapping_failed)
        self._speech_worker.finished.connect(self._on_mapping_finished)
        self._speech_worker.finished.connect(self._speech_thread.quit)
        self._speech_worker.finished.connect(self._speech_worker.deleteLater)
        self._speech_thread.finished.connect(self._speech_thread.deleteLater)

        self._speech_thread.start()

    def _on_listening_started(self, gesture: str):
        self.mapping_page.set_status(f"Listening for {gesture.upper()} command...")

    def _on_transcript_ready(self, transcript: str):
        self.mapping_page.show_processing_banner()
        self.mapping_page.set_status(f"Heard: '{transcript}'. Mapping intent...")

    def _on_mapping_ready(self, gesture: str, macro: str):
        self.custom_mapping[gesture] = macro
        self._sync_mapping_views()
        self.mapping_page.show_success_banner(f"{gesture.upper()} -> {macro}")
        self.mapping_page.set_status(
            f"Updated mapping: {gesture.upper()} -> {macro}. Press Enter when ready."
        )

    def _on_mapping_failed(self, reason: str):
        self.mapping_page.hide_activity_banner()
        self.mapping_page.set_status(f"Remapping failed: {reason}")
        self._status_bar.showMessage(f"Remapping failed: {reason}")

    def _on_mapping_finished(self):
        self._listening = False
        self.mapping_page.clear_highlight()
        self._speech_thread = None
        self._speech_worker = None

    def _setup_shortcuts(self):
        quit_shortcut = QShortcut(QKeySequence("Q"), self)
        quit_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        quit_shortcut.activated.connect(self.shutdown)
        self._quit_shortcut = quit_shortcut

        fine_tune_enter = QShortcut(QKeySequence(Qt.Key.Key_Return), self)
        fine_tune_enter.setContext(Qt.ShortcutContext.ApplicationShortcut)
        fine_tune_enter.activated.connect(self._handle_enter_shortcut)
        self._fine_tune_enter_shortcut = fine_tune_enter

        fine_tune_enter_alt = QShortcut(QKeySequence(Qt.Key.Key_Enter), self)
        fine_tune_enter_alt.setContext(Qt.ShortcutContext.ApplicationShortcut)
        fine_tune_enter_alt.activated.connect(self._handle_enter_shortcut)
        self._fine_tune_enter_alt_shortcut = fine_tune_enter_alt

        space_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        space_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        space_shortcut.activated.connect(self._handle_space_shortcut)
        self._space_shortcut = space_shortcut

        notification_shortcut = QShortcut(QKeySequence("N"), self)
        notification_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        notification_shortcut.activated.connect(self._handle_notification_shortcut)
        self._notification_shortcut = notification_shortcut

    def _handle_enter_shortcut(self):
        if self.stack.currentWidget() is self.fine_tuning_page:
            if not self._fine_tune_started:
                self._start_fine_tune_session()
                return
            if not self._fine_tune_collecting and self._fine_tune_class_index < len(
                self._fine_tune_classes
            ):
                self._start_current_class_collection()
                return

        if self.stack.currentWidget() is self.mapping_page:
            self._show_dashboard()

    def _handle_space_shortcut(self):
        if self.stack.currentWidget() is self.dashboard_page:
            self._show_mapping()

    def _handle_notification_shortcut(self):
        ok, message = show_windows_toast(
            "Demo Notification",
            "Test notification from the demo.",
        )

        self._status_bar.showMessage(message)
        if ok:
            self.request_notification_tap.emit()

    def _show_dashboard(self):
        self._last_dashboard_gesture = "none"
        self._photo_sort_action_armed = True
        self._pending_gesture_sort_category = None
        self.stack.setCurrentWidget(self.dashboard_page)
        self._status_bar.showMessage(
            "Dashboard active. Press Space to return to gesture mapping."
        )

    def _show_mapping(self):
        self.stack.setCurrentWidget(self.mapping_page)
        self._last_dashboard_gesture = "none"
        self._photo_sort_action_armed = True
        self._pending_gesture_sort_category = None
        self.action_controller.release_mouse()
        self._status_bar.showMessage(
            "Gesture mapping active. Press Enter for dashboard."
        )

    def keyPressEvent(self, a0):
        if a0 is None:
            return

        key = a0.key()

        if key == Qt.Key.Key_Q:
            self.shutdown()
            return

        if self.stack.currentWidget() is self.fine_tuning_page and key in (
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
        ):
            if not self._fine_tune_started:
                self._start_fine_tune_session()
                return
            if not self._fine_tune_collecting and self._fine_tune_class_index < len(
                self._fine_tune_classes
            ):
                self._start_current_class_collection()
                return

        if self.stack.currentWidget() is self.mapping_page and key in (
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
        ):
            self._show_dashboard()
            return

        if (
            self.stack.currentWidget() is self.dashboard_page
            and key == Qt.Key.Key_Space
        ):
            self._show_mapping()
            return

        super().keyPressEvent(a0)

    def closeEvent(self, a0: QCloseEvent | None):
        self.shutdown()
        if a0 is not None:
            a0.accept()

    def shutdown(self):
        if self._shutting_down:
            return

        self._shutting_down = True

        try:
            self.action_controller.release_mouse()
        except Exception:
            pass

        if self._speech_thread is not None and self._speech_thread.isRunning():
            self._speech_thread.quit()
            self._speech_thread.wait(1500)

        if self.worker_thread.isRunning():
            try:
                QMetaObject.invokeMethod(
                    self.worker,
                    "stop",
                    Qt.ConnectionType.BlockingQueuedConnection,
                )
            except RuntimeError:
                pass
            self.worker_thread.quit()
            self.worker_thread.wait(2500)

        app = QApplication.instance()
        if app is not None:
            app.quit()
