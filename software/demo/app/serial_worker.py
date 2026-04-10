import csv
import logging
import os
import pickle
import time
import warnings
from collections import deque
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import serial
import tensorflow as tf
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

warnings.filterwarnings(
    "ignore",
    message=(
        "TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11.*"
    ),
)
tf.get_logger().setLevel(logging.ERROR)


class _FineTuneCallback(tf.keras.callbacks.Callback):
    def __init__(self, epoch_signal: pyqtSignal):
        super().__init__()
        self._epoch_signal = epoch_signal

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = float(logs.get("loss", 0.0))
        accuracy = float(logs.get("accuracy", 0.0))
        self._epoch_signal.emit(epoch + 1, loss, accuracy)


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

        self._last_seen_gesture = "none"
        self._stable_frame_count = 0
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
            x_train = self._format_for_cnn(
                np.asarray(self._collection_features, dtype=np.float32)
            )
            y_encoded = self._label_encoder.transform(self._collection_labels)

            pool_layers = (
                tf.keras.layers.MaxPooling1D,
                tf.keras.layers.MaxPooling2D,
                tf.keras.layers.MaxPooling3D,
                tf.keras.layers.AveragePooling1D,
                tf.keras.layers.AveragePooling2D,
                tf.keras.layers.AveragePooling3D,
                tf.keras.layers.GlobalAveragePooling1D,
                tf.keras.layers.GlobalAveragePooling2D,
                tf.keras.layers.GlobalAveragePooling3D,
            )
            conv_layers = (
                tf.keras.layers.Conv1D,
                tf.keras.layers.Conv2D,
                tf.keras.layers.Conv3D,
                tf.keras.layers.SeparableConv1D,
                tf.keras.layers.SeparableConv2D,
                tf.keras.layers.DepthwiseConv2D,
            )

            for layer in self._model.layers:
                if isinstance(layer, conv_layers + pool_layers):
                    layer.trainable = False
                elif isinstance(layer, tf.keras.layers.Dense):
                    layer.trainable = True

            self._model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            self.fine_tuning_started.emit(FINE_TUNE_EPOCHS)
            callback = _FineTuneCallback(self.fine_tuning_epoch)

            self._model.fit(
                x_train,
                y_encoded,
                epochs=FINE_TUNE_EPOCHS,
                batch_size=32,
                verbose=0,
                callbacks=[callback],
                shuffle=True,
            )

            self.fine_tuning_complete.emit(True, "Fine-tuning complete.")
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
        self._last_seen_gesture = "none"
        self._stable_frame_count = 0
        self._last_stable_emit = 0.0
        self.status_message.emit("Live inference running.")

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

        features = self._format_for_cnn(normalised.reshape(1, -1))
        probabilities = self._model(features, training=False).numpy()[0]
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])
        label = self._label_encoder.inverse_transform([predicted_index])[0]

        self.inference_frame.emit(front, back, label, confidence)
        self._update_stable_detection(label)

    def _update_stable_detection(self, label: str):
        if label == "none":
            self._last_seen_gesture = "none"
            self._stable_frame_count = 0
            return

        if label == self._last_seen_gesture:
            self._stable_frame_count += 1
        else:
            self._last_seen_gesture = label
            self._stable_frame_count = 1

        now = time.time()
        if (
            self._stable_frame_count >= STABLE_GESTURE_FRAMES
            and now - self._last_stable_emit >= STABLE_GESTURE_COOLDOWN_SECONDS
        ):
            self.stable_gesture.emit(label)
            self._last_stable_emit = now
            self._stable_frame_count = 0

    def _load_model_assets(self):
        if self._label_encoder is None:
            if not ENCODER_PATH.exists():
                raise FileNotFoundError(f"Label encoder not found at {ENCODER_PATH}")
            with open(ENCODER_PATH, "rb") as file:
                self._label_encoder = pickle.load(file)

        if self._model is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
            self._model = tf.keras.models.load_model(MODEL_PATH)

    @staticmethod
    def _format_for_cnn(flat_samples: np.ndarray):
        front = flat_samples[:, :49].reshape(-1, 7, 7)
        back = flat_samples[:, 49:].reshape(-1, 6, 7)
        back_padded = np.pad(back, ((0, 0), (0, 1), (0, 0)), mode="constant")
        return np.stack([front, back_padded], axis=-1)
