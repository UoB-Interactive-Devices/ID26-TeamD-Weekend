from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .actions import AVAILABLE_MACROS, DEFAULT_GESTURE_MAPPING
from .game_view import PhotoSortGameView
from .heatmap_canvas import HeatmapCanvas


class CalibrationPage(QWidget):
    def __init__(self):
        super().__init__()
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(80, 40, 80, 40)

        card = QFrame()
        card.setObjectName("Card")

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(40, 35, 40, 35)
        card_layout.setSpacing(20)
        card_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        title = QLabel("Calibrating...")
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        instruction = QLabel(
            "Please do not touch the device. Keep the hand still and empty."
        )
        instruction.setObjectName("Subtitle")
        instruction.setWordWrap(True)
        instruction.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.counter_label = QLabel("0 / 30 samples")
        self.counter_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        card_layout.addWidget(title)
        card_layout.addWidget(instruction)
        card_layout.addWidget(self.progress)
        card_layout.addWidget(self.counter_label)

        outer_layout.addStretch()
        outer_layout.addWidget(card)
        outer_layout.addStretch()

    def set_progress(self, current: int, total: int):
        self.progress.setRange(0, total)
        self.progress.setValue(current)
        self.counter_label.setText(f"{current} / {total} samples")


class FineTuningPage(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 28, 40, 28)
        root.setSpacing(16)

        title = QLabel("Fine-Tuning")
        title.setObjectName("Title")

        subtitle = QLabel("Press Enter to advance each step.")
        subtitle.setObjectName("Subtitle")
        subtitle.setWordWrap(True)

        self.step_hint = QLabel("STEP 1")
        self.step_hint.setStyleSheet(
            "font-size: 14px; font-weight: 700; color: #1c5e8b; letter-spacing: 1px;"
        )

        self.instruction = QLabel(
            "Press ENTER to begin fine-tuning. The hand will close to fist posture."
        )
        self.instruction.setStyleSheet(
            "font-size: 20px; font-weight: 700; color: #124568; padding: 10px;"
            "border: 2px solid #9bc4e4; border-radius: 10px; background: #f4fbff;"
        )
        self.instruction.setWordWrap(True)

        self.gesture_label = QLabel("Gesture: -")
        self.gesture_label.setStyleSheet(
            "font-size: 26px; font-weight: 700; color: #235a84;"
        )

        self.class_progress = QProgressBar()
        self.class_progress.setRange(0, 100)

        self.class_counter = QLabel("0 / 0")
        self.class_counter.setStyleSheet("font-size: 14px; font-weight: 600;")

        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 1)
        self.training_progress.setValue(0)

        self.training_label = QLabel("Fine-tuning not started.")
        self.training_label.setStyleSheet("font-size: 15px; font-weight: 600;")
        self.training_label.setWordWrap(True)

        root.addWidget(title)
        root.addWidget(subtitle)
        root.addWidget(self.step_hint)
        root.addWidget(self.instruction)
        root.addWidget(self.gesture_label)
        root.addWidget(self.class_progress)
        root.addWidget(self.class_counter)
        root.addWidget(self.training_label)
        root.addWidget(self.training_progress)
        root.addStretch()

    def set_instruction(self, text: str):
        self.instruction.setText(text)

    def set_step_hint(self, text: str):
        self.step_hint.setText(text)

    def update_collection_progress(
        self,
        gesture: str,
        count: int,
        per_class_total: int,
        class_index: int,
        class_total: int,
    ):
        self.gesture_label.setText(f"Gesture: {gesture.upper()}")
        self.class_progress.setRange(0, per_class_total)
        self.class_progress.setValue(count)
        self.class_counter.setText(
            f"{count} / {per_class_total} samples  |  Class {class_index} of {class_total}"
        )

    def prepare_next_gesture(
        self,
        gesture: str,
        class_index: int,
        class_total: int,
        per_class_total: int,
    ):
        self.gesture_label.setText(f"Gesture: {gesture.upper()}")
        self.class_progress.setRange(0, per_class_total)
        self.class_progress.setValue(0)
        self.class_counter.setText(
            f"0 / {per_class_total} samples  |  Class {class_index} of {class_total}"
        )

    def set_training_started(self, total_epochs: int):
        self.training_progress.setRange(0, total_epochs)
        self.training_progress.setValue(0)
        self.training_label.setText("Fine-tuning in progress...")

    def update_training_epoch(self, epoch: int, loss: float, accuracy: float):
        self.training_progress.setValue(epoch)
        self.training_label.setText(
            f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy * 100:.1f}%"
        )

    def set_training_message(self, message: str):
        self.training_label.setText(message)


class MappingPage(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(10)

        title = QLabel("Gesture Mapping")
        title.setObjectName("Title")

        instruction = QLabel(
            "HOW TO USE: Hold one gesture steadily for around one second to remap it by voice. "
            "When ready, press ENTER to open the visualiser dashboard."
        )
        instruction.setObjectName("Subtitle")
        instruction.setStyleSheet(
            "font-size: 16px; font-weight: 700; color: #154f76; "
            "padding: 8px; border: 1px solid #a4c8e5; border-radius: 8px; background: #f4fbff;"
        )
        instruction.setWordWrap(True)

        split = QGridLayout()
        split.setHorizontalSpacing(14)

        self.mapping_table = QTableWidget(5, 2)
        self.mapping_table.setHorizontalHeaderLabels(["Gesture", "Current Action"])
        self.mapping_table.verticalHeader().setVisible(False)
        self.mapping_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.mapping_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.mapping_table.horizontalHeader().setStretchLastSection(True)

        self.macros_list = QListWidget()
        for macro, description in AVAILABLE_MACROS.items():
            item = QListWidgetItem(f"{macro} - {description}")
            self.macros_list.addItem(item)

        split.addWidget(self.mapping_table, 0, 0)
        split.addWidget(self.macros_list, 0, 1)

        self.listening_label = QLabel("LISTENING NOW")
        self.listening_label.setVisible(False)
        self._set_listening_banner_style()

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(
            "font-size: 15px; font-weight: 600; color: #194f74;"
        )
        self.status_label.setWordWrap(True)

        root.addWidget(title)
        root.addWidget(instruction)
        root.addLayout(split)
        root.addWidget(self.listening_label)
        root.addWidget(self.status_label)

        self.set_mapping(DEFAULT_GESTURE_MAPPING)

    def set_mapping(self, mapping: dict[str, str]):
        gestures = list(mapping.keys())
        self.mapping_table.setRowCount(len(gestures))

        for row, gesture in enumerate(gestures):
            gesture_item = QTableWidgetItem(gesture)
            macro_item = QTableWidgetItem(mapping[gesture])
            self.mapping_table.setItem(row, 0, gesture_item)
            self.mapping_table.setItem(row, 1, macro_item)

        self.clear_highlight()

    def highlight_gesture(self, gesture: str):
        for row in range(self.mapping_table.rowCount()):
            item = self.mapping_table.item(row, 0)
            action_item = self.mapping_table.item(row, 1)
            if item is None or action_item is None:
                continue

            if item.text() == gesture:
                item.setBackground(QColor("#b4e6d4"))
                action_item.setBackground(QColor("#b4e6d4"))
            else:
                item.setBackground(QColor("#ffffff"))
                action_item.setBackground(QColor("#ffffff"))

    def clear_highlight(self):
        for row in range(self.mapping_table.rowCount()):
            item = self.mapping_table.item(row, 0)
            action_item = self.mapping_table.item(row, 1)
            if item is not None:
                item.setBackground(QColor("#ffffff"))
            if action_item is not None:
                action_item.setBackground(QColor("#ffffff"))

    def set_listening(self, active: bool, gesture: str = ""):
        if active:
            self.show_listening_banner(gesture)
            return
        self.hide_activity_banner()

    def _set_listening_banner_style(self):
        self.listening_label.setStyleSheet(
            "font-size: 24px; color: #8a1322; font-weight: 800;"
            "padding: 10px; border: 2px solid #d9485f; border-radius: 10px;"
            "background: #ffeef1;"
        )

    def _set_processing_banner_style(self):
        self.listening_label.setStyleSheet(
            "font-size: 24px; color: #124f86; font-weight: 800;"
            "padding: 10px; border: 2px solid #3d8ad6; border-radius: 10px;"
            "background: #ebf5ff;"
        )

    def _set_success_banner_style(self):
        self.listening_label.setStyleSheet(
            "font-size: 22px; color: #165d2a; font-weight: 800;"
            "padding: 10px; border: 2px solid #3bb36a; border-radius: 10px;"
            "background: #ecfff3;"
        )

    def show_processing_banner(self):
        self._set_processing_banner_style()
        self.listening_label.setText("PROCESSING REMAP...")
        self.listening_label.setVisible(True)

    def show_listening_banner(self, gesture: str = ""):
        self._set_listening_banner_style()
        gesture_text = gesture.upper() if gesture else "CURRENT"
        self.listening_label.setText(f"LISTENING: SAY WHAT {gesture_text} SHOULD DO")
        self.listening_label.setVisible(True)

    def show_success_banner(self, remap_text: str):
        self._set_success_banner_style()
        self.listening_label.setText(f"REMAPPED: {remap_text}")
        self.listening_label.setVisible(True)

    def hide_activity_banner(self):
        self.listening_label.setVisible(False)

    def set_status(self, text: str):
        self.status_label.setText(text)


class DashboardPage(QWidget):
    def __init__(self):
        super().__init__()
        root = QHBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        side_panel = QFrame()
        side_panel.setObjectName("Card")
        side_panel.setMinimumWidth(260)

        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(14, 14, 14, 14)
        side_layout.setSpacing(8)

        side_title = QLabel("Live Status")
        side_title.setStyleSheet("font-size: 20px; font-weight: 700; color: #1d486b;")

        self.classification_label = QLabel("Classification: -")
        self.confidence_label = QLabel("Confidence: -")
        self.action_label = QLabel("Action: -")
        self.score_label = QLabel("Categorised: 0")
        self.remaining_label = QLabel("Photos Remaining: 0")

        mapping_title = QLabel("Gesture Mappings")
        mapping_title.setStyleSheet("font-weight: 700; color: #2e5a7e;")
        self.mapping_summary = QLabel("")
        self.mapping_summary.setWordWrap(True)

        side_layout.addWidget(side_title)
        side_layout.addWidget(self.classification_label)
        side_layout.addWidget(self.confidence_label)
        side_layout.addWidget(self.action_label)
        side_layout.addWidget(self.score_label)
        side_layout.addWidget(self.remaining_label)
        side_layout.addSpacing(8)
        side_layout.addWidget(mapping_title)
        side_layout.addWidget(self.mapping_summary)
        side_layout.addStretch()

        content_layout = QVBoxLayout()
        content_layout.setSpacing(10)

        self.photo_sort_game = PhotoSortGameView()
        self.heatmaps = HeatmapCanvas()

        self.dashboard_hint = QLabel(
            "Photo sorting game: click the top photo to pick it up and it will follow your cursor, then click Group 1-4 to place it. "
            "You can also categorise by gesture. Press SPACE to return to gesture mapping."
        )
        self.dashboard_hint.setStyleSheet(
            "font-size: 14px; font-weight: 700; color: #1a567d;"
        )
        self.dashboard_hint.setWordWrap(True)

        content_layout.addWidget(self.photo_sort_game, 2)
        content_layout.addWidget(self.dashboard_hint)
        content_layout.addWidget(self.heatmaps, 1)

        root.addWidget(side_panel)
        root.addLayout(content_layout, 1)

    def set_mapping(self, mapping: dict[str, str]):
        lines = [f"{gesture} -> {macro}" for gesture, macro in mapping.items()]
        self.mapping_summary.setText("\n".join(lines))

    def set_classification(self, gesture: str, confidence: float):
        self.classification_label.setText(f"Classification: {gesture}")
        self.confidence_label.setText(f"Confidence: {confidence * 100:.1f}%")

    def set_action(self, action_name: str):
        self.action_label.setText(f"Action: {action_name}")

    def set_score(self, score: int):
        self.score_label.setText(f"Categorised: {score}")

    def set_remaining(self, remaining: int):
        self.remaining_label.setText(f"Photos Remaining: {remaining}")
