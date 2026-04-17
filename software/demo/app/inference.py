import csv
import time
from collections import deque
from pathlib import Path

import numpy as np
import serial
import torch
from ml.train import format_for_cnn
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from .config import (
    BAUD_RATE,
    ENCODER_PATH,
    FINE_TUNE_CLASSES,
    FINE_TUNE_EPOCHS,
    FINE_TUNING_CSV,
    LEARNING_RATE,
    MAX_EXPECTED_VALUE,
    MODEL_PATH,
    NUM_BASELINE_SAMPLES,
    PORT,
    SAMPLES_PER_CLASS,
    SMOOTHING_FRAMES,
    STABLE_GESTURE_COOLDOWN_SECONDS,
    STABLE_GESTURE_FRAMES,
    TOTAL_SENSORS,
)
from .fine_tuning import run_fine_tuning
from .model import load_label_encoder, load_model


class Inference:
    def __init__(self, callbacks: dict[str, callable]):
        self._cb = callbacks

        self._serial: serial.Serial | None = None
        self._running = False

        self._mode = "idle"
        self._baseline: np.ndarray | None = None
        self._history_buffer: deque[np.ndarray] = deque(maxlen=SMOOTHING_FRAMES)

        self._baseline_samples: list[np.ndarray] = []

        self._collection_classes = list(FINE_TUNE_CLASSES)
        self._samples_per_class = SAMPLES_PER_CLASS
        self._collection_class_index = 0
        self._collection_count = 0
        self._collection_session_id = ""
        self._collection_features: list[np.ndarray] = []
        self._collection_labels: list[str] = []
        self._collection_file = None
        self._collection_writer = None

        self._model = None
        self._label_encoder = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._prediction_window: deque[str] = deque(maxlen=STABLE_GESTURE_FRAMES)
        self._last_stable_emit = 0.0

        self._next_connect_attempt = 0.0

    def start(self):
        self._running = True
        self._connect_serial()

    def stop(self):
        self._running = False
        self._mode = "idle"
        self._close_collection_file()
        self._cleanup_serial()

    def begin_calibration(self):
        if self._serial is None:
            self._cb["status_message"](
                "Waiting for Teensy connection before calibration."
            )
            return

        self._mode = "calibrating"
        self._baseline_samples = []
        self._history_buffer.clear()

        try:
            self._serial.write(b"O")
            self._serial.reset_input_buffer()
        except serial.SerialException as exc:
            self._cb["connection_error"](f"Calibration failed to start: {exc}")
            return

        self._cb["calibration_progress"](0, NUM_BASELINE_SAMPLES)
        self._cb["status_message"]("Calibrating sensors.")

    def begin_collection(self, classes, samples_per_class):
        if self._serial is None:
            self._cb["status_message"](
                "Waiting for Teensy connection before collection."
            )
            return

        if self._baseline is None:
            self._cb["status_message"]("Calibration must complete before collection.")
            return

        self._collection_classes = classes or list(FINE_TUNE_CLASSES)
        self._samples_per_class = samples_per_class or SAMPLES_PER_CLASS
        self._collection_class_index = 0
        self._collection_count = 0
        self._collection_features = []
        self._collection_labels = []
        self._collection_session_id = str(int(time.time()))
        self._history_buffer.clear()

        self._prepare_collection_csv(FINE_TUNING_CSV)
        self._mode = "idle"

        try:
            self._serial.write(b"F")
            self._serial.reset_input_buffer()
        except serial.SerialException as exc:
            self._cb["connection_error"](f"Collection failed to start: {exc}")
            return

        first_gesture = self._collection_classes[0]
        self._cb["collection_instruction"](
            f"Hand closed. Prepare gesture {first_gesture.upper()}, then press Enter to start capture."
        )
        self._cb["status_message"]("Fine-tuning session initialised.")

    def collect_current_class(self):
        if self._serial is None:
            self._cb["status_message"]("Cannot collect: Teensy is not connected.")
            return

        if self._baseline is None:
            self._cb["status_message"]("Cannot collect: calibration is missing.")
            return

        if self._collection_class_index >= len(self._collection_classes):
            self._cb["status_message"]("All classes already collected.")
            return

        self._collection_count = 0
        self._history_buffer.clear()
        self._mode = "collecting"

        gesture = self._collection_classes[self._collection_class_index]
        self._cb["collection_instruction"](
            f"Collecting {gesture.upper()} now. Keep still until the progress bar finishes."
        )
        self._cb["status_message"](
            f"Collecting class {self._collection_class_index + 1}."
        )

    def fine_tune_model(self):
        self._mode = "training"

        try:
            self._load_model_assets()
            assert self._model is not None
            assert self._label_encoder is not None
            success, message = run_fine_tuning(
                model=self._model,
                label_encoder=self._label_encoder,
                collection_features=self._collection_features,
                collection_labels=self._collection_labels,
                device=self._device,
                epochs=FINE_TUNE_EPOCHS,
                learning_rate=LEARNING_RATE,
                on_started=self._cb["fine_tuning_started"],
                on_epoch=self._cb["fine_tuning_epoch"],
            )
            self._cb["fine_tuning_complete"](success, message)
        except Exception as exc:  # pylint: disable=broad-except
            self._cb["fine_tuning_complete"](False, f"Fine-tuning failed: {exc}")
        finally:
            self._mode = "idle"

    def begin_inference(self):
        try:
            self._load_model_assets()
            assert self._model is not None
            assert self._label_encoder is not None
        except Exception as exc:  # pylint: disable=broad-except
            self._cb["connection_error"](f"Unable to start inference: {exc}")
            return

        if self._baseline is None:
            self._cb["status_message"]("Calibration baseline is missing.")
            return

        self._mode = "inference"
        self._history_buffer.clear()
        self._prediction_window.clear()
        self._last_stable_emit = 0.0
        self._cb["status_message"](
            f"Live inference running on {self._device.type.upper()}."
        )

    def tap_hand(self):
        if self._serial is None or not self._serial.is_open:
            return

        try:
            self._serial.write(b"O")
            self._serial.flush()
            time.sleep(0.17)
            self._serial.write(b"F")
            self._serial.flush()
        except serial.SerialException:
            self._cb["connection_error"]("Failed to trigger hand tap on click.")

    def notification_finger_tap(self):
        if self._serial is None or not self._serial.is_open:
            return

        try:
            self._serial.write(b"F")
            self._serial.flush()
            time.sleep(0.15)

            finger_count = 5
            for finger_index in range(finger_count):
                positions = [0] * finger_count
                positions[finger_index] = 180
                self._write_servo_positions(positions)
                time.sleep(0.15)
                self._write_servo_positions([0] * finger_count)
                time.sleep(0.15)
        except serial.SerialException:
            self._cb["connection_error"]("Failed to trigger notification finger taps.")

    def category_group_tap(self, finger_index: int):
        if self._serial is None or not self._serial.is_open:
            return

        if finger_index < 0 or finger_index > 4:
            return

        try:
            self._run_finger_tap_sequence([finger_index], up_angle=180, press_time=0.17)
        except serial.SerialException:
            self._cb["connection_error"]("Failed to trigger category finger tap.")

    def _run_finger_tap_sequence(
        self,
        finger_indices: list[int],
        up_angle: int,
        press_time: float,
    ):
        if self._serial is None or not self._serial.is_open:
            return

        self._serial.write(b"F")
        self._serial.flush()
        time.sleep(0.02)

        finger_count = 5
        for finger_index in finger_indices:
            positions = [0] * finger_count
            positions[finger_index] = up_angle
            self._write_servo_positions(positions)
            time.sleep(press_time)
            self._write_servo_positions([0] * finger_count)
            time.sleep(press_time)

    def _write_servo_positions(self, positions: list[int]):
        if self._serial is None or not self._serial.is_open:
            return

        constrained = [max(0, min(180, int(angle))) for angle in positions]
        packet = "S " + " ".join(str(value) for value in constrained) + "\n"
        self._serial.write(packet.encode("ascii"))
        self._serial.flush()

    def tick(self):
        if not self._running:
            return

        if self._serial is None:
            if time.time() >= self._next_connect_attempt:
                self._connect_serial()
                self._next_connect_attempt = time.time() + 1.0
            return

        if self._mode not in {"calibrating", "collecting", "inference"}:
            if self._serial.in_waiting > 4096:
                self._serial.read_all()
            return

        sample = self._read_latest_sample()
        if sample is None:
            return

        if self._mode == "calibrating":
            self._process_calibration_sample(sample)
        elif self._mode == "collecting":
            self._process_collection_sample(sample)
        elif self._mode == "inference":
            self._process_inference_sample(sample)

    def _connect_serial(self):
        try:
            self._serial = serial.Serial(PORT, BAUD_RATE, timeout=0.1)
            self._cb["connected"](f"Connected to {PORT} at {BAUD_RATE} baud")
            self._cb["status_message"]("Teensy connected.")
        except (serial.SerialException, FileNotFoundError) as exc:
            self._serial = None
            self._cb["connection_error"](f"Waiting for Teensy: {exc}")

    def _cleanup_serial(self):
        if self._serial and self._serial.is_open:
            try:
                self._serial.write(b"O")
                self._serial.flush()
                time.sleep(0.1)
            except serial.SerialException:
                pass
            try:
                self._serial.close()
            except serial.SerialException:
                pass
        self._serial = None

    def _read_latest_sample(self):
        if self._serial is None or self._serial.in_waiting <= 0:
            return None

        try:
            raw_bytes = self._serial.read_all()
        except serial.SerialException as exc:
            self._cb["connection_error"](f"Serial read failed: {exc}")
            self._cleanup_serial()
            self._next_connect_attempt = time.time() + 1.0
            return None

        if raw_bytes is None:
            return None

        lines = raw_bytes.decode("utf-8", errors="ignore").splitlines()
        for line in reversed(lines):
            try:
                values = [int(value) for value in line.split(",")]
            except ValueError:
                continue

            if len(values) == TOTAL_SENSORS:
                return np.asarray(values, dtype=np.float32)

        return None

    def _process_calibration_sample(self, sample: np.ndarray):
        self._baseline_samples.append(sample)
        self._cb["calibration_progress"](
            len(self._baseline_samples), NUM_BASELINE_SAMPLES
        )

        if len(self._baseline_samples) >= NUM_BASELINE_SAMPLES:
            self._baseline = np.mean(self._baseline_samples, axis=0)
            self._mode = "idle"
            self._cb["calibration_complete"]()
            self._cb["status_message"]("Calibration complete.")

    def _prepare_collection_csv(self, csv_path: Path):
        self._close_collection_file()

        csv_path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = csv_path.exists()
        self._collection_file = open(csv_path, mode="a", newline="", encoding="utf-8")
        self._collection_writer = csv.writer(self._collection_file)

        if not file_exists:
            header = [f"sensor_{index}" for index in range(TOTAL_SENSORS)] + [
                "label",
                "session_id",
            ]
            self._collection_writer.writerow(header)
            self._collection_file.flush()

    def _close_collection_file(self):
        if self._collection_file is not None:
            try:
                self._collection_file.close()
            except OSError:
                pass
        self._collection_file = None
        self._collection_writer = None

    def _process_collection_sample(self, sample: np.ndarray):
        if self._collection_class_index >= len(self._collection_classes):
            return

        gesture = self._collection_classes[self._collection_class_index]

        if self._collection_writer is not None:
            row = sample.astype(int).tolist() + [gesture, self._collection_session_id]
            self._collection_writer.writerow(row)

        zeroed = sample - self._baseline
        self._history_buffer.append(zeroed)
        smoothed = np.mean(self._history_buffer, axis=0)
        normalised = np.clip(smoothed / MAX_EXPECTED_VALUE, 0.0, 1.0).astype(np.float32)

        self._collection_features.append(normalised)
        self._collection_labels.append(gesture)

        self._collection_count += 1
        self._cb["collection_progress"](
            gesture,
            self._collection_count,
            self._samples_per_class,
            self._collection_class_index + 1,
            len(self._collection_classes),
        )

        if self._collection_count < self._samples_per_class:
            return

        self._collection_class_index += 1
        self._collection_count = 0
        self._history_buffer.clear()

        if self._collection_class_index < len(self._collection_classes):
            next_gesture = self._collection_classes[self._collection_class_index]
            self._cb["collection_class_complete"](
                gesture,
                self._collection_class_index,
                len(self._collection_classes),
            )
            self._cb["collection_instruction"](
                f"Completed {gesture.upper()}. Prepare {next_gesture.upper()}, then press Enter."
            )
            self._mode = "idle"
            return

        if self._collection_file is not None:
            self._collection_file.flush()

        total_samples = len(self._collection_features)
        self._mode = "idle"
        self._close_collection_file()
        self._cb["collection_class_complete"](
            gesture,
            len(self._collection_classes),
            len(self._collection_classes),
        )
        self._cb["collection_complete"](total_samples)
        self._cb["status_message"]("Fine-tuning collection complete.")

    def _process_inference_sample(self, sample: np.ndarray):
        assert self._model is not None
        assert self._label_encoder is not None

        zeroed = sample - self._baseline
        self._history_buffer.append(zeroed)
        smoothed = np.mean(self._history_buffer, axis=0)
        normalised = np.clip(smoothed / MAX_EXPECTED_VALUE, 0.0, 1.0).astype(np.float32)

        front = normalised[:49].reshape((7, 7))
        back = normalised[49:].reshape((6, 7))

        features = (
            torch.from_numpy(format_for_cnn(normalised.reshape(1, -1)))
            .float()
            .to(self._device)
        )
        self._model.eval()
        with torch.no_grad():
            logits = self._model(features)
            probabilities = (
                torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
            )

        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])
        label = self._label_encoder.inverse_transform([predicted_index])[0]

        self._cb["inference_frame"](front, back, label, confidence)
        self._update_stable_detection(label)

    def _update_stable_detection(self, label: str):
        self._prediction_window.append(label)

        if label == "none" or len(self._prediction_window) < STABLE_GESTURE_FRAMES:
            return

        now = time.time()
        if now - self._last_stable_emit < STABLE_GESTURE_COOLDOWN_SECONDS:
            return

        label_count = self._prediction_window.count(label)
        required_votes = int(np.ceil(STABLE_GESTURE_FRAMES * 0.7))

        if label_count >= required_votes:
            self._cb["stable_gesture"](label)
            self._last_stable_emit = now
            self._prediction_window.clear()

    def _load_model_assets(self):
        if self._label_encoder is None:
            self._label_encoder = load_label_encoder(ENCODER_PATH)

        if self._model is None:
            assert self._label_encoder is not None
            self._model = load_model(
                MODEL_PATH,
                len(self._label_encoder.classes_),
                self._device,
            )


class InferenceWorker(QObject):
    connected = pyqtSignal(str)
    status_message = pyqtSignal(str)
    connection_error = pyqtSignal(str)

    calibration_progress = pyqtSignal(int, int)
    calibration_complete = pyqtSignal()

    collection_instruction = pyqtSignal(str)
    collection_progress = pyqtSignal(str, int, int, int, int)
    collection_class_complete = pyqtSignal(str, int, int)
    collection_complete = pyqtSignal(int)

    fine_tuning_started = pyqtSignal(int)
    fine_tuning_epoch = pyqtSignal(int, float, float)
    fine_tuning_complete = pyqtSignal(bool, str)

    inference_frame = pyqtSignal(object, object, str, float)
    stable_gesture = pyqtSignal(str)

    stopped = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._timer: QTimer | None = None
        self._core = Inference(
            {
                "connected": self.connected.emit,
                "status_message": self.status_message.emit,
                "connection_error": self.connection_error.emit,
                "calibration_progress": self.calibration_progress.emit,
                "calibration_complete": self.calibration_complete.emit,
                "collection_instruction": self.collection_instruction.emit,
                "collection_progress": self.collection_progress.emit,
                "collection_class_complete": self.collection_class_complete.emit,
                "collection_complete": self.collection_complete.emit,
                "fine_tuning_started": self.fine_tuning_started.emit,
                "fine_tuning_epoch": self.fine_tuning_epoch.emit,
                "fine_tuning_complete": self.fine_tuning_complete.emit,
                "inference_frame": self.inference_frame.emit,
                "stable_gesture": self.stable_gesture.emit,
            }
        )

    @pyqtSlot()
    def start(self):
        self._core.start()

        timer = QTimer(self)
        timer.setInterval(0)
        timer.timeout.connect(self._core.tick)
        timer.start()
        self._timer = timer

    @pyqtSlot()
    def stop(self):
        if self._timer is not None:
            self._timer.stop()
        self._core.stop()
        self.stopped.emit()

    @pyqtSlot()
    def begin_calibration(self):
        self._core.begin_calibration()

    @pyqtSlot(list, int)
    def begin_collection(self, classes, samples_per_class):
        self._core.begin_collection(classes, samples_per_class)

    @pyqtSlot()
    def collect_current_class(self):
        self._core.collect_current_class()

    @pyqtSlot()
    def fine_tune_model(self):
        self._core.fine_tune_model()

    @pyqtSlot()
    def begin_inference(self):
        self._core.begin_inference()

    @pyqtSlot()
    def tap_hand(self):
        self._core.tap_hand()

    @pyqtSlot()
    def notification_finger_tap(self):
        self._core.notification_finger_tap()

    @pyqtSlot(int)
    def category_group_tap(self, finger_index: int):
        self._core.category_group_tap(finger_index)
