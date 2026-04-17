import copy
import csv
import pickle
import time
from collections import deque
from pathlib import Path

import numpy as np
import serial
import torch
import torch.nn as nn
import torch.optim as optim
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot
from torch.utils.data import DataLoader, TensorDataset

from hand.train import GestureCNN, format_for_cnn

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
class SerialInferenceWorker(QObject):
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
        self._serial: serial.Serial | None = None
        self._timer: QTimer | None = None
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

    @pyqtSlot()
    def start(self):
        self._running = True
        self._connect_serial()

        timer = QTimer(self)
        timer.setInterval(0)
        timer.timeout.connect(self._tick)
        timer.start()
        self._timer = timer

    @pyqtSlot()
    def stop(self):
        self._running = False
        self._mode = "idle"

        if self._timer is not None:
            self._timer.stop()

        self._close_collection_file()
        self._cleanup_serial()
        self.stopped.emit()

    @pyqtSlot()
    def begin_calibration(self):
        if self._serial is None:
            self.status_message.emit(
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
            self.connection_error.emit(f"Calibration failed to start: {exc}")
            return

        self.calibration_progress.emit(0, NUM_BASELINE_SAMPLES)
        self.status_message.emit("Calibrating sensors.")

    @pyqtSlot(list, int)
    def begin_collection(self, classes, samples_per_class):
        if self._serial is None:
            self.status_message.emit("Waiting for Teensy connection before collection.")
            return

        if self._baseline is None:
            self.status_message.emit("Calibration must complete before collection.")
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
            self.connection_error.emit(f"Collection failed to start: {exc}")
            return

        first_gesture = self._collection_classes[0]
        self.collection_instruction.emit(
            f"Hand closed. Prepare gesture {first_gesture.upper()}, then press Enter to start capture."
        )
        self.status_message.emit("Fine-tuning session initialised.")

    @pyqtSlot()
    def collect_current_class(self):
        if self._serial is None:
            self.status_message.emit("Cannot collect: Teensy is not connected.")
            return

        if self._baseline is None:
            self.status_message.emit("Cannot collect: calibration is missing.")
            return

        if self._collection_class_index >= len(self._collection_classes):
            self.status_message.emit("All classes already collected.")
            return

        self._collection_count = 0
        self._history_buffer.clear()
        self._mode = "collecting"

        gesture = self._collection_classes[self._collection_class_index]
        self.collection_instruction.emit(
            f"Collecting {gesture.upper()} now. Keep still until the progress bar finishes."
        )
        self.status_message.emit(
            f"Collecting class {self._collection_class_index + 1}."
        )

    @pyqtSlot()
    def fine_tune_model(self):
        if not self._collection_features or not self._collection_labels:
            self.fine_tuning_complete.emit(
                False, "No collection data available for fine-tuning."
            )
            return

        self._mode = "training"

        try:
            self._load_model_assets()
            assert self._model is not None
            assert self._label_encoder is not None
            model = self._model

            x_all = format_for_cnn(
                np.asarray(self._collection_features, dtype=np.float32)
            )
            y_encoded = np.asarray(
                self._label_encoder.transform(self._collection_labels), dtype=np.int64
            )

            total_samples = len(y_encoded)
            if total_samples < 10:
                self.fine_tuning_complete.emit(
                    False,
                    "Fine-tuning requires at least 10 samples for safe validation.",
                )
                self._mode = "idle"
                return

            rng = np.random.default_rng(seed=42)
            indices = np.arange(total_samples)
            rng.shuffle(indices)
            val_size = max(1, int(total_samples * 0.2))
            if val_size >= total_samples:
                val_size = total_samples - 1

            val_indices = indices[:val_size]
            train_indices = indices[val_size:]

            x_train = x_all[train_indices]
            y_train = y_encoded[train_indices]
            x_val = x_all[val_indices]
            y_val = y_encoded[val_indices]

            x_tensor = torch.from_numpy(x_train).float().to(self._device)
            y_tensor = torch.from_numpy(y_train).to(self._device)
            train_loader = DataLoader(
                TensorDataset(x_tensor, y_tensor),
                batch_size=32,
                shuffle=True,
            )

            x_val_tensor = torch.from_numpy(x_val).float().to(self._device)
            y_val_tensor = torch.from_numpy(y_val).to(self._device)

            def evaluate_on_validation():
                model.eval()
                with torch.no_grad():
                    logits = model(x_val_tensor)
                    loss = criterion(logits, y_val_tensor)
                    predictions = logits.argmax(dim=1)
                    accuracy = (predictions == y_val_tensor).float().mean().item()
                return float(loss.item()), float(accuracy)

            for parameter in model.conv1.parameters():
                parameter.requires_grad = False
            for parameter in model.conv2.parameters():
                parameter.requires_grad = False
            for parameter in model.fc1.parameters():
                parameter.requires_grad = True
            for parameter in model.fc2.parameters():
                parameter.requires_grad = True

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                [
                    {"params": model.fc2.parameters()},
                    {"params": model.fc1.parameters(), "weight_decay": 0.01},
                ],
                lr=LEARNING_RATE,
            )

            baseline_state = copy.deepcopy(model.state_dict())
            baseline_loss, baseline_accuracy = evaluate_on_validation()

            self.fine_tuning_started.emit(FINE_TUNE_EPOCHS)

            model.train()
            for epoch in range(FINE_TUNE_EPOCHS):
                running_loss = 0.0
                total = 0
                correct = 0

                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    running_loss += float(loss.item()) * inputs.size(0)
                    predictions = outputs.argmax(dim=1)
                    total += targets.size(0)
                    correct += int((predictions == targets).sum().item())

                epoch_loss = running_loss / max(total, 1)
                epoch_accuracy = correct / max(total, 1)
                self.fine_tuning_epoch.emit(epoch + 1, epoch_loss, epoch_accuracy)

            tuned_loss, tuned_accuracy = evaluate_on_validation()

            if tuned_accuracy + 1e-6 < baseline_accuracy:
                model.load_state_dict(baseline_state)
                model.eval()
                self.fine_tuning_complete.emit(
                    True,
                    (
                        "Fine-tuning rejected: validation accuracy dropped "
                        f"from {baseline_accuracy:.3f} to {tuned_accuracy:.3f}. "
                        "Kept previous model weights."
                    ),
                )
                return

            model.eval()
            torch.save(model.state_dict(), MODEL_PATH)
            self.fine_tuning_complete.emit(
                True,
                (
                    f"Fine-tuning accepted on {self._device.type.upper()}. "
                    f"Validation accuracy {baseline_accuracy:.3f} -> {tuned_accuracy:.3f}"
                ),
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.fine_tuning_complete.emit(False, f"Fine-tuning failed: {exc}")
        finally:
            self._mode = "idle"

    @pyqtSlot()
    def begin_inference(self):
        try:
            self._load_model_assets()
            assert self._model is not None
            assert self._label_encoder is not None
        except Exception as exc:  # pylint: disable=broad-except
            self.connection_error.emit(f"Unable to start inference: {exc}")
            return

        if self._baseline is None:
            self.status_message.emit("Calibration baseline is missing.")
            return

        self._mode = "inference"
        self._history_buffer.clear()
        self._prediction_window.clear()
        self._last_stable_emit = 0.0
        self.status_message.emit(
            f"Live inference running on {self._device.type.upper()}."
        )

    @pyqtSlot()
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
            self.connection_error.emit("Failed to trigger hand tap on click.")

    @pyqtSlot()
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
            self.connection_error.emit("Failed to trigger notification finger taps.")

    @pyqtSlot(int)
    def category_group_tap(self, finger_index: int):
        if self._serial is None or not self._serial.is_open:
            return

        if finger_index < 0 or finger_index > 4:
            return

        try:
            self._run_finger_tap_sequence([finger_index], up_angle=180, press_time=0.17)
        except serial.SerialException:
            self.connection_error.emit("Failed to trigger category finger tap.")

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

    def _tick(self):
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
            self.connected.emit(f"Connected to {PORT} at {BAUD_RATE} baud")
            self.status_message.emit("Teensy connected.")
        except (serial.SerialException, FileNotFoundError) as exc:
            self._serial = None
            self.connection_error.emit(f"Waiting for Teensy: {exc}")

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
            self.connection_error.emit(f"Serial read failed: {exc}")
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
        self.calibration_progress.emit(
            len(self._baseline_samples), NUM_BASELINE_SAMPLES
        )

        if len(self._baseline_samples) >= NUM_BASELINE_SAMPLES:
            self._baseline = np.mean(self._baseline_samples, axis=0)
            self._mode = "idle"
            self.calibration_complete.emit()
            self.status_message.emit("Calibration complete.")

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
        self.collection_progress.emit(
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
            self.collection_class_complete.emit(
                gesture,
                self._collection_class_index,
                len(self._collection_classes),
            )
            self.collection_instruction.emit(
                f"Completed {gesture.upper()}. Prepare {next_gesture.upper()}, then press Enter."
            )
            self._mode = "idle"
            return

        if self._collection_file is not None:
            self._collection_file.flush()

        total_samples = len(self._collection_features)
        self._mode = "idle"
        self._close_collection_file()
        self.collection_class_complete.emit(
            gesture,
            len(self._collection_classes),
            len(self._collection_classes),
        )
        self.collection_complete.emit(total_samples)
        self.status_message.emit("Fine-tuning collection complete.")

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

        self.inference_frame.emit(front, back, label, confidence)
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
            self.stable_gesture.emit(label)
            self._last_stable_emit = now
            self._prediction_window.clear()

    def _load_model_assets(self):
        if self._label_encoder is None:
            if not ENCODER_PATH.exists():
                raise FileNotFoundError(f"Label encoder not found at {ENCODER_PATH}")
            with open(ENCODER_PATH, "rb") as file:
                self._label_encoder = pickle.load(file)

        if self._model is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
            assert self._label_encoder is not None
            self._model = GestureCNN(len(self._label_encoder.classes_)).to(self._device)

            state_dict = torch.load(
                MODEL_PATH,
                map_location=self._device,
                weights_only=False,
            )
            self._model.load_state_dict(state_dict)
            self._model.eval()
