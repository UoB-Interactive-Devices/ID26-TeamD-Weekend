import random

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
)


class TargetFieldView(QGraphicsView):
    score_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(0, 0, 780, 420, self)
        self.setScene(self._scene)
        self.setSceneRect(self._scene.sceneRect())
        self.setMinimumSize(820, 460)

        self._score = 0
        self._targets: list[QGraphicsEllipseItem] = []

        self.setStyleSheet(
            "QGraphicsView {"
            "border: 2px solid #7aa5d2;"
            "border-radius: 10px;"
            "background: #f5fbff;"
            "}"
        )

        self._instruction_item = QGraphicsTextItem(
            "Use your gestures to move the mouse and squeeze to click the targets."
        )
        self._instruction_item.setDefaultTextColor(QColor("#294867"))
        self._instruction_item.setPos(18, 10)
        self._scene.addItem(self._instruction_item)

        self.reset()

    def reset(self):
        for target in self._targets:
            self._scene.removeItem(target)
        self._targets.clear()

        self._score = 0
        self.score_changed.emit(self._score)

        for _ in range(3):
            self._spawn_target()

    def _spawn_target(self):
        rect = self._scene.sceneRect()
        radius = random.randint(22, 36)
        x = random.uniform(rect.left() + 20, rect.right() - (2 * radius) - 20)
        y = random.uniform(rect.top() + 60, rect.bottom() - (2 * radius) - 20)

        target = QGraphicsEllipseItem(0, 0, 2 * radius, 2 * radius)
        target.setPos(x, y)
        target.setPen(QPen(QColor("#2c5f8a"), 2))
        target.setBrush(QBrush(QColor("#57c4ad")))
        target.setData(0, "target")
        self._scene.addItem(target)
        self._targets.append(target)

    def process_global_click(self, global_pos: tuple[int, int]):
        view_pos = self.mapFromGlobal(QPoint(global_pos[0], global_pos[1]))
        if not self.viewport().rect().contains(view_pos):
            return

        scene_pos = self.mapToScene(view_pos)
        self._process_scene_click(scene_pos)

    def mousePressEvent(self, event):
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._process_scene_click(scene_pos)

        super().mousePressEvent(event)

    def _process_scene_click(self, scene_pos: QPointF):
        hit_item = self._find_target(scene_pos)
        if hit_item is None:
            return

        self._scene.removeItem(hit_item)
        self._targets.remove(hit_item)
        self._score += 1
        self.score_changed.emit(self._score)
        self._spawn_target()

    def _find_target(self, scene_pos: QPointF):
        for item in self._scene.items(scene_pos):
            if isinstance(item, QGraphicsEllipseItem) and item.data(0) == "target":
                return item
        return None

    @property
    def score(self):
        return self._score

    @property
    def scene_bounds(self):
        return QRectF(self._scene.sceneRect())
