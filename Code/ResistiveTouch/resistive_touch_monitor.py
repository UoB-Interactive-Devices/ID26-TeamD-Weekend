import time
from collections import deque

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import serial


def connect_to_teensy(port="COM5", baudrate=2000000):
    while True:
        try:
            ser = serial.Serial(port, baudrate, timeout=0.1)
            print(f"Connected to {port} at {baudrate} baud!")
            return ser
        except (serial.SerialException, FileNotFoundError):
            print("Waiting for Teensy...")
            time.sleep(1)


ser = connect_to_teensy()

# Preprocessing Parameters
NUM_BASELINE_SAMPLES = 30
SMOOTHING_FRAMES = 3
MAX_EXPECTED_VALUE = 4000.0
TOTAL_SENSORS = 91  # 49 (Front) + 42 (Back)

# State Variables
baseline = None
baseline_samples = []
history_buffer = deque(maxlen=SMOOTHING_FRAMES)

# Initialize the Matplotlib figure with TWO subplots side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.canvas.manager.set_window_title("Teensy Dual Sensor Matrices - ML Preprocessing")

# Initial empty matrices
initial_front = np.zeros((7, 7))
initial_back = np.zeros((6, 7))

# Setup Front Visualizer (7x7)
im1 = ax1.imshow(
    initial_front, cmap="magma", interpolation="nearest", vmin=0.0, vmax=1.0
)
ax1.set_title("Front Matrix (7x7)")
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Normalized Intensity")

# Setup Back Visualizer (6x7)
im2 = ax2.imshow(
    initial_back, cmap="magma", interpolation="nearest", vmin=0.0, vmax=1.0
)
ax2.set_title("Back Matrix (6x7)")
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Normalized Intensity")

# Main title for overall status
fig.suptitle("Calibrating Sensor Baselines...", fontsize=14)


def update(frame):
    global baseline, baseline_samples, history_buffer, ser

    try:
        last_valid_array = None

        # High-Speed Buffer Draining
        if ser.in_waiting > 0:
            raw_bytes = ser.read_all()
            lines = raw_bytes.decode("utf-8", errors="ignore").split("\r\n")

            for line in reversed(lines):
                if line:
                    try:
                        data = [int(x) for x in line.split(",")]
                        if len(data) == TOTAL_SENSORS:
                            last_valid_array = np.array(data)
                            break
                    except ValueError:
                        continue

        if last_valid_array is not None:
            # Calibration Phase
            if baseline is None:
                baseline_samples.append(last_valid_array)
                fig.suptitle(
                    f"Calibrating... {len(baseline_samples)}/{NUM_BASELINE_SAMPLES}",
                    fontsize=14,
                )

                if len(baseline_samples) >= NUM_BASELINE_SAMPLES:
                    baseline = np.mean(baseline_samples, axis=0)
                    fig.suptitle("Sensor Matrices (Live & High Speed)", fontsize=14)
                    print("\nCalibration complete! Matrices are live.")
                return [im1, im2]

            # Math Operations
            zeroed_array = last_valid_array - baseline
            history_buffer.append(zeroed_array)
            smoothed_array = np.mean(history_buffer, axis=0)
            normalized_array = np.clip(smoothed_array / MAX_EXPECTED_VALUE, 0.0, 1.0)

            # Slice into Front and Back
            front_matrix = normalized_array[:49].reshape((7, 7))
            back_matrix = normalized_array[49:].reshape((6, 7))

            # Update UI
            im1.set_array(front_matrix)
            im2.set_array(back_matrix)

    except serial.SerialException:
        print("\nConnection lost. Reconnecting...")
        fig.suptitle("Connection lost. Reconnecting...", color="red", fontsize=14)
        ser.close()
        ser = connect_to_teensy()

    return [im1, im2]


ani = animation.FuncAnimation(
    fig, update, interval=15, blit=False, cache_frame_data=False
)

plt.tight_layout()
plt.show()
