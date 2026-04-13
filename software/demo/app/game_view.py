import random
from pathlib import Path

from PyQt6.QtCore import QEasingCurve, QEvent, QPoint, QRect, QPropertyAnimation, Qt, pyqtSignal
from PyQt6.QtGui import QCursor, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .config import ASSETS_DIR


class DraggablePhotoLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._photo_path: str | None = None
        self._source_pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "border: 2px solid #4b7aa3; border-radius: 12px; background: #ffffff;"
        )

    @property
    def photo_path(self) -> str | None:
        return self._photo_path

    def set_photo(self, photo_path: str, pixmap: QPixmap):
        self._photo_path = photo_path
        self._source_pixmap = pixmap
        self._refresh_pixmap()

    def clear_photo(self, message: str):
        self._photo_path = None
        self._source_pixmap = None
        self.setPixmap(QPixmap())
        self.setText(message)

    def _refresh_pixmap(self):
        if self._source_pixmap is None or self._source_pixmap.isNull():
            self.setPixmap(QPixmap())
            self.setText("Unable to load image")
            return

        scaled = self._source_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def mousePressEvent(self, event):
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class DropCategoryFrame(QFrame):
    def __init__(self, owner: "PhotoSortGameView", category_key: str, title: str):
        super().__init__()
        self._owner = owner
        self.category_key = category_key
        self._last_thumb_keys: list[str] = []
        self.setStyleSheet(
            "QFrame {"
            "border: 2px dashed #6d9cc2; border-radius: 12px; background: #f3f8fc;"
            "}"
            "QFrame[pickupReady='true'] {"
            "background: #dff0ff; border: 2px solid #3e7ca8;"
            "}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self._title = QLabel(title)
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet("font-size: 16px; font-weight: 700; color: #1e4f75;")

        self._thumbs_widget = QWidget()
        self._thumbs_grid = QGridLayout(self._thumbs_widget)
        self._thumbs_grid.setContentsMargins(0, 0, 0, 0)
        self._thumbs_grid.setHorizontalSpacing(4)
        self._thumbs_grid.setVerticalSpacing(4)

        layout.addWidget(self._title)
        layout.addWidget(self._thumbs_widget, 1)
        layout.addStretch()

    def set_thumbnails(self, photo_paths: list[str]):
        preview_paths = photo_paths[-24:]
        if preview_paths == self._last_thumb_keys:
            return
        self._last_thumb_keys = list(preview_paths)

        while self._thumbs_grid.count() > 0:
            item = self._thumbs_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        columns = 6
        for index, photo_path in enumerate(preview_paths):
            thumb = QLabel()
            thumb.setFixedSize(38, 38)
            thumb.setStyleSheet(
                "border: 1px solid #8eaec8; border-radius: 4px; background: #ffffff;"
            )
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)

            pixmap = self._owner.get_thumbnail_pixmap(photo_path, 34)
            if not pixmap.isNull():
                thumb.setPixmap(pixmap)

            row = index // columns
            col = index % columns
            self._thumbs_grid.addWidget(thumb, row, col)

    def set_pickup_ready(self, ready: bool):
        self.setProperty("pickupReady", ready)
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event):
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            self._owner.drop_picked_photo(self.category_key)
        super().mousePressEvent(event)


class PhotoSortGameView(QWidget):
    score_changed = pyqtSignal(int)
    remaining_changed = pyqtSignal(int)
    photo_sorted = pyqtSignal(str, str, int)
    stack_completed = pyqtSignal()

    CATEGORY_TITLES = {
        "group_1": "Group 1",
        "group_2": "Group 2",
        "group_3": "Group 3",
        "group_4": "Group 4",
    }
    MAX_DECK_IMAGES = 20

    def __init__(self):
        super().__init__()
        self.setMinimumSize(820, 460)
        self.setObjectName("PhotoSortRoot")
        self.setStyleSheet(
            "QWidget#PhotoSortRoot {"
            "border: 2px solid #7aa5d2;"
            "border-radius: 10px;"
            "background: #f5fbff;"
            "}"
        )

        self._photo_queue: list[str] = []
        self._history: list[tuple[str, str]] = []
        self._grouped_photos = {key: [] for key in self.CATEGORY_TITLES.keys()}
        self._all_photo_count = 0
        self._photo_picked_up = False
        self._photo_cache: dict[str, QPixmap] = {}
        self._thumb_cache: dict[tuple[str, int], QPixmap] = {}
        self._cursor_active = False
        self._sort_animation: QPropertyAnimation | None = None
        self._animation_overlay: QLabel | None = None
        self._pending_animation_sort: tuple[str, str] | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        self._instruction = QLabel(
            "Click once on the top photo to pick it up, then click a group to place it."
        )
        self._instruction.setWordWrap(True)
        self._instruction.setStyleSheet(
            "font-size: 14px; font-weight: 700; color: #1f527a; background: transparent; border: 0;"
        )

        board = QHBoxLayout()
        board.setSpacing(16)

        stack_panel = QFrame()
        stack_panel.setObjectName("PhotoStackPanel")
        stack_panel.setStyleSheet(
            "QFrame#PhotoStackPanel {"
            "border: 0; background: transparent;"
            "}"
        )
        stack_layout = QVBoxLayout(stack_panel)
        stack_layout.setContentsMargins(0, 0, 0, 0)

        self._stack_container = QFrame()
        self._stack_container.setMinimumSize(380, 290)
        self._stack_container.setStyleSheet("QFrame { border: 0; background: transparent; }")
        self._stack_container.installEventFilter(self)

        self._back_card_far = QFrame(self._stack_container)
        self._back_card_far.setStyleSheet(
            "QFrame { border: 2px solid #adc5d9; border-radius: 12px; background: #e6f1fa; }"
        )

        self._back_card_near = QFrame(self._stack_container)
        self._back_card_near.setStyleSheet(
            "QFrame { border: 2px solid #93b2cb; border-radius: 12px; background: #edf5fc; }"
        )

        self._top_photo = DraggablePhotoLabel(self._stack_container)
        self._top_photo.clicked.connect(self._toggle_pickup)

        self._remaining_label = QLabel("Photos remaining: 0")
        self._remaining_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._remaining_label.setStyleSheet(
            "font-size: 15px; font-weight: 700; color: #1a4e75; background: transparent; border: 0;"
        )

        stack_layout.addWidget(self._stack_container, 1)
        stack_layout.addWidget(self._remaining_label)

        categories_panel = QFrame()
        categories_panel.setStyleSheet("QFrame { border: 0; background: transparent; }")
        categories_layout = QGridLayout(categories_panel)
        categories_layout.setContentsMargins(0, 0, 0, 0)
        categories_layout.setHorizontalSpacing(10)
        categories_layout.setVerticalSpacing(10)

        self._drop_zones: dict[str, DropCategoryFrame] = {}
        order = ["group_1", "group_2", "group_3", "group_4"]
        for index, category in enumerate(order):
            zone = DropCategoryFrame(self, category, self.CATEGORY_TITLES[category])
            categories_layout.addWidget(zone, index // 2, index % 2)
            self._drop_zones[category] = zone

        board.addWidget(stack_panel, 3)
        board.addWidget(categories_panel, 4)

        root.addWidget(self._instruction)
        root.addLayout(board, 1)

        self.reset()

    @property
    def remaining_count(self) -> int:
        return len(self._photo_queue)

    @property
    def sorted_count(self) -> int:
        return len(self._history)

    def has_current_photo(self, path: str) -> bool:
        current = self._current_photo_path()
        if current is None:
            return False
        return str(Path(path)) == current

    def has_picked_photo(self) -> bool:
        return self._photo_picked_up and self._current_photo_path() is not None

    def drop_picked_photo(self, category: str) -> bool:
        if not self.has_picked_photo():
            return False
        return self.sort_current_photo(category)

    def _toggle_pickup(self):
        if self._sort_animation is not None:
            return

        if self._current_photo_path() is None:
            self._set_photo_picked_up(False)
            self._refresh_ui()
            return

        self._set_photo_picked_up(not self._photo_picked_up)
        self._refresh_ui()

    def sort_current_photo(self, category: str, animate: bool = False) -> bool:
        if category not in self._grouped_photos:
            return False

        current = self._current_photo_path()
        if current is None:
            return False

        if animate:
            return self._animate_sort(category, current)

        self._commit_sort(category, current)
        return True

    def _commit_sort(self, category: str, current: str):
        head = self._current_photo_path()
        if head is None:
            return
        if head != current:
            current = head

        self._photo_queue.pop(0)
        self._history.append((current, category))
        self._grouped_photos[category].append(current)
        self._set_photo_picked_up(False)

        self._refresh_ui()
        self.photo_sorted.emit(current, category, self.remaining_count)

        if self.remaining_count == 0:
            self.stack_completed.emit()

    def _animate_sort(self, category: str, current: str) -> bool:
        if self._sort_animation is not None:
            return False

        zone = self._drop_zones.get(category)
        if zone is None:
            return False

        source_pos = self._top_photo.mapTo(self, QPoint(0, 0))
        source_rect = QRect(source_pos, self._top_photo.size())

        zone_center = zone.mapTo(self, zone.rect().center())
        target_size = 44
        target_rect = QRect(
            zone_center.x() - target_size // 2,
            zone_center.y() - target_size // 2,
            target_size,
            target_size,
        )

        source_pixmap = self.get_photo_pixmap(current)
        if source_pixmap.isNull():
            return False

        overlay = QLabel(self)
        overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        overlay.setGeometry(source_rect)
        overlay.setStyleSheet(
            "border: 2px solid #4b7aa3; border-radius: 10px; background: #ffffff;"
        )
        overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        overlay.setPixmap(
            source_pixmap.scaled(
                source_rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        overlay.show()

        self._top_photo.setVisible(False)
        self._animation_overlay = overlay
        self._pending_animation_sort = (current, category)
        self._set_photo_picked_up(False)

        animation = QPropertyAnimation(overlay, b"geometry", self)
        animation.setDuration(220)
        animation.setStartValue(source_rect)
        animation.setEndValue(target_rect)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.finished.connect(self._finish_sort_animation)
        self._sort_animation = animation
        animation.start()
        return True

    def _finish_sort_animation(self):
        pending = self._pending_animation_sort
        self._pending_animation_sort = None

        if self._animation_overlay is not None:
            self._animation_overlay.deleteLater()
            self._animation_overlay = None

        self._top_photo.setVisible(True)

        if pending is not None:
            current, category = pending
            self._commit_sort(category, current)

        self._sort_animation = None

    def undo_last_sort(self) -> bool:
        if not self._history:
            return False

        photo_path, category = self._history.pop()
        self._photo_queue.insert(0, photo_path)

        grouped = self._grouped_photos.get(category, [])
        for index in range(len(grouped) - 1, -1, -1):
            if grouped[index] == photo_path:
                grouped.pop(index)
                break

        self._set_photo_picked_up(False)
        self._refresh_ui()
        return True

    def reset(self):
        if self._sort_animation is not None:
            self._sort_animation.stop()
            self._sort_animation = None
        if self._animation_overlay is not None:
            self._animation_overlay.deleteLater()
            self._animation_overlay = None
        self._pending_animation_sort = None

        self._history.clear()
        self._grouped_photos = {key: [] for key in self.CATEGORY_TITLES.keys()}
        self._photo_queue = self._build_photo_deck()
        self._all_photo_count = len(self._photo_queue)
        self._set_photo_picked_up(False)
        self._refresh_ui()

    def eventFilter(self, watched, event):
        if watched is self._stack_container and event is not None:
            if event.type() == QEvent.Type.Resize:
                self._position_stack_cards()
                self._refresh_top_photo_preview()
        return super().eventFilter(watched, event)

    def _position_stack_cards(self):
        width = self._stack_container.width()
        height = self._stack_container.height()
        card_width = max(180, min(320, width - 80))
        card_height = max(130, min(240, height - 70))
        x = max(12, (width - card_width) // 2)
        y = max(10, (height - card_height) // 2 - 8)

        self._back_card_far.setGeometry(x + 22, y + 22, card_width, card_height)
        self._back_card_near.setGeometry(x + 11, y + 11, card_width, card_height)
        self._top_photo.setGeometry(x, y, card_width, card_height)

    def _build_photo_deck(self) -> list[str]:
        photo_paths: list[str] = []
        if ASSETS_DIR.exists():
            for candidate in sorted(ASSETS_DIR.iterdir()):
                if not candidate.is_file():
                    continue
                if candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    photo_paths.append(str(candidate))

        if len(photo_paths) > self.MAX_DECK_IMAGES:
            photo_paths = random.sample(photo_paths, self.MAX_DECK_IMAGES)

        random.shuffle(photo_paths)
        return photo_paths

    def _current_photo_path(self) -> str | None:
        if not self._photo_queue:
            return None
        return self._photo_queue[0]

    def get_photo_pixmap(self, photo_path: str) -> QPixmap:
        cached = self._photo_cache.get(photo_path)
        if cached is not None:
            return cached

        pixmap = QPixmap(photo_path)
        self._photo_cache[photo_path] = pixmap
        return pixmap

    def get_thumbnail_pixmap(self, photo_path: str, size: int) -> QPixmap:
        key = (photo_path, size)
        cached = self._thumb_cache.get(key)
        if cached is not None:
            return cached

        source = self.get_photo_pixmap(photo_path)
        if source.isNull():
            self._thumb_cache[key] = QPixmap()
            return self._thumb_cache[key]

        thumb = source.scaled(
            size,
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumb_cache[key] = thumb
        return thumb

    def _set_photo_picked_up(self, picked: bool):
        self._photo_picked_up = picked

        if not picked:
            if self._cursor_active:
                QApplication.restoreOverrideCursor()
                self._cursor_active = False
            return

        current = self._current_photo_path()
        if current is None:
            return

        source = self.get_photo_pixmap(current)
        if source.isNull():
            return

        cursor_pixmap = source.scaled(
            56,
            56,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        if self._cursor_active:
            QApplication.restoreOverrideCursor()

        QApplication.setOverrideCursor(
            QCursor(cursor_pixmap, cursor_pixmap.width() // 2, cursor_pixmap.height() // 2)
        )
        self._cursor_active = True

    def _refresh_top_photo_preview(self):
        current = self._current_photo_path()
        if current is None:
            self._set_photo_picked_up(False)
            self._top_photo.clear_photo("All photos sorted")
            return

        pixmap = self.get_photo_pixmap(current)
        self._top_photo.set_photo(current, pixmap)
        if self._photo_picked_up:
            self._top_photo.setStyleSheet(
                "border: 3px solid #2e7cb1; border-radius: 12px; background: #f0f8ff;"
            )
        else:
            self._top_photo.setStyleSheet(
                "border: 2px solid #4b7aa3; border-radius: 12px; background: #ffffff;"
            )

    def _refresh_ui(self):
        self._refresh_top_photo_preview()

        for category, zone in self._drop_zones.items():
            zone.set_thumbnails(self._grouped_photos[category])
            zone.set_pickup_ready(self.has_picked_photo())

        self._remaining_label.setText(
            f"Photos remaining: {self.remaining_count} / {self._all_photo_count}"
        )

        self.score_changed.emit(self.sorted_count)
        self.remaining_changed.emit(self.remaining_count)
