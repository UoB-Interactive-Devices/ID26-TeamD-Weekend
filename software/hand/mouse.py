import pickle
import sys
import time
from collections import deque

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pyautogui
import serial
from tensorflow.keras.models import load_model

# Configuration Parameters
PORT = "COM5"
BAUD_RATE = 2000000
TOTAL_SENSORS = 91

NUM_BASELINE_SAMPLES = 30
SMOOTHING_FRAMES = 3
MAX_EXPECTED_VALUE = 4000.0

MODEL_PATH = "gesture_cnn_model.keras"
ENCODER_PATH = "label_encoder.pkl"

MOUSE_MOVE_DISTANCE = 30
MOUSE_MOVE_DELAY = 0.05
CLICK_DELAY = 0.5

# Global State Variables
baseline = None
stored_baseline = None
use_calibration = True
history_buffer = deque(maxlen=SMOOTHING_FRAMES)
last_mouse_move_time = 0
last_click_time = 0
ser = None
model = None
label_encoder = None
fist_closed = False

# New Global for Visual Wiggles
overlay_timer = 0


def connect_to_teensy(port=PORT, baudrate=BAUD_RATE):
    while True:
        try:
            connection = serial.Serial(port, baudrate, timeout=0.1)
            print(f"Connected to {port} at {baudrate} baud!")
            return connection
        except (serial.SerialException, FileNotFoundError):
            print("Waiting for Teensy...")
            time.sleep(1)


def calibrate_sensors(ser):
    print("\nCalibrating sensor baselines. Please keep the hand still and empty...")

    ser.write(b"O")
    time.sleep(0.5)
    ser.reset_input_buffer()

    samples = []
    while len(samples) < NUM_BASELINE_SAMPLES:
        if ser.in_waiting > 0:
            raw_bytes = ser.read_all()
            lines = raw_bytes.decode("utf-8", errors="ignore").split("\r\n")

            for line in reversed(lines):
                if line:
                    try:
                        data = [int(x) for x in line.split(",")]
                        if len(data) == TOTAL_SENSORS:
                            samples.append(np.array(data))
                            print(
                                f"Calibrating... {len(samples)}/{NUM_BASELINE_SAMPLES}",
                                end="\r",
                            )
                            break
                    except ValueError:
                        continue
        time.sleep(0.01)

    print("\nCalibration complete! Matrices are initialised.")
    return np.mean(samples, axis=0)


def execute_action(gesture):
    if gesture == "left":
        pyautogui.move(-MOUSE_MOVE_DISTANCE, 0)
    elif gesture == "right":
        pyautogui.move(MOUSE_MOVE_DISTANCE, 0)
    elif gesture == "up":
        pyautogui.move(0, -MOUSE_MOVE_DISTANCE)
    elif gesture == "down":
        pyautogui.move(0, MOUSE_MOVE_DISTANCE)
    elif gesture == "squeeze":
        pyautogui.click()


def on_key(event):
    global baseline, stored_baseline, use_calibration, fig

    if event.key == "c":
        if stored_baseline is not None:
            use_calibration = not use_calibration
            if use_calibration:
                baseline = stored_baseline
                fig.suptitle("Calibration: ON", fontsize=14)
                print("\nCalibration toggled ON ", end="\r")
            else:
                baseline = np.zeros_like(stored_baseline)
                fig.suptitle("Calibration: OFF", fontsize=14)
                print("\nCalibration toggled OFF", end="\r")
    elif event.key == "q":
        print("\n'q' pressed. Initiating shutdown...")
        plt.close(fig)


def update(frame):
    global \
        baseline, \
        history_buffer, \
        ser, \
        last_mouse_move_time, \
        last_click_time, \
        fist_closed, \
        overlay_timer

    try:
        last_valid_array = None

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
            zeroed_array = last_valid_array - baseline
            history_buffer.append(zeroed_array)
            smoothed_array = np.mean(history_buffer, axis=0)
            normalized_array = np.clip(smoothed_array / MAX_EXPECTED_VALUE, 0.0, 1.0)

            front_matrix = normalized_array[:49].reshape((7, 7))
            back_matrix = normalized_array[49:].reshape((6, 7))

            im1.set_array(front_matrix)
            im2.set_array(back_matrix)

            front_reshape = front_matrix.reshape(1, 7, 7)
            back_reshape = back_matrix.reshape(1, 6, 7)
            back_padded = np.pad(
                back_reshape, ((0, 0), (0, 1), (0, 0)), mode="constant"
            )
            cnn_features = np.stack([front_reshape, back_padded], axis=-1)

            predicted_probs = model.predict(cnn_features, verbose=0)
            predicted_idx = np.argmax(predicted_probs, axis=1)[0]
            gesture = label_encoder.inverse_transform([predicted_idx])[0]

            current_time = time.time()

            if gesture != "none" and not fist_closed:
                print("\nFirst movement detected! Closing fist.")
                ser.write(b"F")
                fist_closed = True

            # --- Execute Actions & Visual Wiggles ---
            if gesture == "none":
                fig.suptitle(
                    f"Action: NONE (Press 'c' to toggle calibration, 'q' to quit)",
                    fontsize=14,
                )
                print(f"Action: NONE    ", end="\r")

                # Smoothly fade out the central text overlay
                if overlay_timer > 0:
                    overlay_timer -= 1
                    action_overlay.set_alpha(overlay_timer / 15.0)
                else:
                    action_overlay.set_visible(False)

            elif gesture == "squeeze":
                if current_time - last_click_time >= CLICK_DELAY:
                    execute_action(gesture)

                    # Visual pop for squeeze
                    action_overlay.set_text("💥 SQUEEZE")
                    action_overlay.set_color("yellow")
                    action_overlay.set_alpha(1.0)
                    action_overlay.set_visible(True)
                    overlay_timer = 15  # Reset the fade-out timer

                    ser.write(b"O")
                    time.sleep(0.2)
                    ser.write(b"F")
                    ser.reset_input_buffer()

                    last_click_time = time.time()
                    fig.suptitle(f"Action: CLICK (Squeeze)", fontsize=14)
                    print(f"Action: CLICK   ", end="\r")

            else:
                if current_time - last_mouse_move_time >= MOUSE_MOVE_DELAY:
                    execute_action(gesture)
                    last_mouse_move_time = current_time

                    # Visual pop for directional movement
                    symbols = {
                        "left": "⬅️ LEFT",
                        "right": "RIGHT ➡️",
                        "up": "⬆️ UP",
                        "down": "⬇️ DOWN",
                    }
                    action_overlay.set_text(symbols.get(gesture, gesture.upper()))
                    action_overlay.set_color("cyan")
                    action_overlay.set_alpha(1.0)
                    action_overlay.set_visible(True)
                    overlay_timer = 15  # Reset the fade-out timer

                    fig.suptitle(f"Action: {gesture.upper()}", fontsize=14)
                    print(f"Action: {gesture.upper():8s}", end="\r")

    except pyautogui.FailSafeException:
        print(
            "\n\nFail-safe triggered! You moved the mouse into the corner of the screen."
        )
        plt.close(fig)
    except serial.SerialException:
        print("\nConnection lost. Reconnecting...")
        fig.suptitle("Connection lost. Reconnecting...", color="red", fontsize=14)
        ser.close()
        ser = connect_to_teensy()

    return [im1, im2, action_overlay]


if __name__ == "__main__":
    # Re-enabled PyAutoGUI safeguards
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0

    print("Loading models...")
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    model = load_model(MODEL_PATH)
    print(f"Model loaded. Classes: {label_encoder.classes_}")

    ser = connect_to_teensy()

    stored_baseline = calibrate_sensors(ser)
    baseline = stored_baseline

    # Setup UI
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.canvas.manager.set_window_title("Gesture Control & Live Visualiser")
    fig.canvas.mpl_connect("key_press_event", on_key)

    initial_front = np.zeros((7, 7))
    initial_back = np.zeros((6, 7))

    im1 = ax1.imshow(
        initial_front, cmap="magma", interpolation="nearest", vmin=0.0, vmax=1.0
    )
    ax1.set_title("Front Matrix (7x7)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Normalised Intensity")

    im2 = ax2.imshow(
        initial_back, cmap="magma", interpolation="nearest", vmin=0.0, vmax=1.0
    )
    ax2.set_title("Back Matrix (6x7)")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Normalised Intensity")

    # Central Action Overlay
    action_overlay = fig.text(
        0.5,
        0.5,
        "",
        fontsize=50,
        ha="center",
        va="center",
        weight="bold",
        zorder=10,
        bbox=dict(
            facecolor="black", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.5"
        ),
    )
    action_overlay.set_visible(False)

    print(
        "\nStarting mouse control & visualiser! Move mouse to top-left corner or press 'q' to stop."
    )

    ani = animation.FuncAnimation(
        fig, update, interval=15, blit=False, cache_frame_data=False
    )

    plt.tight_layout()

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nProgramme interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        print("\nCleaning up and releasing hand...")
        if ser and ser.is_open:
            ser.write(b"O")
            time.sleep(0.5)
            ser.close()
        print("Programme exited cleanly.")
        sys.exit(0)
