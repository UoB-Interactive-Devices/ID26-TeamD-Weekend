# import pickle
# import time
# from collections import deque

# import matplotlib.animation as animation
# import matplotlib.pyplot as plt
# import numpy as np
# import pyautogui
# import serial

# # Configuration Parameters
# PORT = "COM5"
# BAUD_RATE = 2000000
# TOTAL_SENSORS = 91

# NUM_BASELINE_SAMPLES = 30
# SMOOTHING_FRAMES = 3
# MAX_EXPECTED_VALUE = 4000.0

# MODEL_PATH = "gesture_rf_model.pkl"
# ENCODER_PATH = "label_encoder.pkl"

# MOUSE_MOVE_DISTANCE = 30
# MOUSE_MOVE_DELAY = 0.05
# CLICK_DELAY = 0.5  # Prevents rapid-fire clicking when squeezing

# # Global State Variables
# baseline = None
# stored_baseline = None  # To keep the calculated baseline safe when toggled off
# use_calibration = True  # Toggle state
# baseline_samples = []
# history_buffer = deque(maxlen=SMOOTHING_FRAMES)
# last_mouse_move_time = 0
# last_click_time = 0
# ser = None
# model = None
# label_encoder = None


# def connect_to_teensy(port=PORT, baudrate=BAUD_RATE):
#     while True:
#         try:
#             connection = serial.Serial(port, baudrate, timeout=0.1)
#             print(f"Connected to {port} at {baudrate} baud!")
#             return connection
#         except (serial.SerialException, FileNotFoundError):
#             print("Waiting for Teensy...")
#             time.sleep(1)


# def execute_action(gesture):
#     if gesture == "left":
#         pyautogui.move(-MOUSE_MOVE_DISTANCE, 0)
#     elif gesture == "right":
#         pyautogui.move(MOUSE_MOVE_DISTANCE, 0)
#     elif gesture == "up":
#         pyautogui.move(0, -MOUSE_MOVE_DISTANCE)
#     elif gesture == "down":
#         pyautogui.move(0, MOUSE_MOVE_DISTANCE)
#     elif gesture == "squeeze":
#         pyautogui.click()


# def on_key(event):
#     """Handles keyboard presses whilst the Matplotlib window is in focus."""
#     global baseline, stored_baseline, use_calibration, fig

#     if event.key == "c":
#         if stored_baseline is not None:
#             use_calibration = not use_calibration
#             if use_calibration:
#                 baseline = stored_baseline
#                 fig.suptitle("Calibration: ON", fontsize=14)
#                 print("\nCalibration toggled ON ", end="\r")
#             else:
#                 baseline = np.zeros_like(stored_baseline)
#                 fig.suptitle("Calibration: OFF", fontsize=14)
#                 print("\nCalibration toggled OFF", end="\r")


# def update(frame):
#     global \
#         baseline, \
#         stored_baseline, \
#         baseline_samples, \
#         history_buffer, \
#         ser, \
#         last_mouse_move_time, \
#         last_click_time

#     try:
#         last_valid_array = None

#         # High-Speed Buffer Draining
#         if ser.in_waiting > 0:
#             raw_bytes = ser.read_all()
#             lines = raw_bytes.decode("utf-8", errors="ignore").split("\r\n")

#             for line in reversed(lines):
#                 if line:
#                     try:
#                         data = [int(x) for x in line.split(",")]
#                         if len(data) == TOTAL_SENSORS:
#                             last_valid_array = np.array(data)
#                             break
#                     except ValueError:
#                         continue

#         if last_valid_array is not None:
#             # Calibration Phase
#             if stored_baseline is None:
#                 baseline_samples.append(last_valid_array)
#                 fig.suptitle(
#                     f"Calibrating... {len(baseline_samples)}/{NUM_BASELINE_SAMPLES}",
#                     fontsize=14,
#                 )

#                 if len(baseline_samples) >= NUM_BASELINE_SAMPLES:
#                     stored_baseline = np.mean(baseline_samples, axis=0)
#                     baseline = stored_baseline
#                     fig.suptitle(
#                         "Sensor Matrices Live (Press 'c' to toggle calibration)",
#                         fontsize=14,
#                     )
#                     print(
#                         "\nCalibration complete! Matrices and mouse control are live."
#                     )
#                 return [im1, im2]

#             # Preprocessing
#             zeroed_array = last_valid_array - baseline
#             history_buffer.append(zeroed_array)
#             smoothed_array = np.mean(history_buffer, axis=0)
#             normalized_array = np.clip(smoothed_array / MAX_EXPECTED_VALUE, 0.0, 1.0)

#             # Matrix Slicing for Visualiser
#             front_matrix = normalized_array[:49].reshape((7, 7))
#             back_matrix = normalized_array[49:].reshape((6, 7))

#             im1.set_array(front_matrix)
#             im2.set_array(back_matrix)

#             # Machine Learning Prediction
#             features = normalized_array.reshape(1, -1)
#             predicted_idx = model.predict(features)[0]
#             gesture = label_encoder.inverse_transform([predicted_idx])[0]

#             current_time = time.time()

#             # Execute mouse movement, click logic, and UI updates
#             if gesture == "none":
#                 # Explicitly show 'none' so the user knows the model is resting
#                 fig.suptitle(
#                     f"Action: NONE (Press 'c' to toggle calibration)", fontsize=14
#                 )
#                 print(f"Action: NONE    ", end="\r")

#             elif gesture == "squeeze":
#                 if current_time - last_click_time >= CLICK_DELAY:
#                     execute_action(gesture)
#                     last_click_time = current_time
#                     fig.suptitle(
#                         f"Action: CLICK (Squeeze) (Press 'c' to toggle calibration)",
#                         fontsize=14,
#                     )
#                     print(f"Action: CLICK   ", end="\r")

#             else:
#                 if current_time - last_mouse_move_time >= MOUSE_MOVE_DELAY:
#                     execute_action(gesture)
#                     last_mouse_move_time = current_time
#                     fig.suptitle(
#                         f"Action: {gesture.upper()} (Press 'c' to toggle calibration)",
#                         fontsize=14,
#                     )
#                     print(f"Action: {gesture.upper():8s}", end="\r")

#     except pyautogui.FailSafeException:
#         print(
#             "\n\nFail-safe triggered! You moved the mouse into the corner of the screen."
#         )
#         plt.close(fig)  # Close the Matplotlib window to exit the programme cleanly
#     except serial.SerialException:
#         print("\nConnection lost. Reconnecting...")
#         fig.suptitle("Connection lost. Reconnecting...", color="red", fontsize=14)
#         ser.close()
#         ser = connect_to_teensy()

#     return [im1, im2]


# if __name__ == "__main__":
#     # Setup PyAutoGUI safeguards
#     pyautogui.FAILSAFE = True
#     pyautogui.PAUSE = 0  # Handled manually via delays

#     # Load machine learning models
#     print("Loading models...")
#     with open(ENCODER_PATH, "rb") as f:
#         label_encoder = pickle.load(f)
#     with open(MODEL_PATH, "rb") as f:
#         model = pickle.load(f)

#     model.n_jobs = 1
#     print(f"Model loaded. Classes: {label_encoder.classes_}")

#     # Establish Serial Connection
#     ser = connect_to_teensy()

#     # Initialise the Matplotlib figure
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#     fig.canvas.manager.set_window_title("Gesture Control & Live Visualiser")

#     # Bind the key press event to our figure
#     fig.canvas.mpl_connect("key_press_event", on_key)

#     initial_front = np.zeros((7, 7))
#     initial_back = np.zeros((6, 7))

#     im1 = ax1.imshow(
#         initial_front, cmap="magma", interpolation="nearest", vmin=0.0, vmax=1.0
#     )
#     ax1.set_title("Front Matrix (7x7)")
#     fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Normalised Intensity")

#     im2 = ax2.imshow(
#         initial_back, cmap="magma", interpolation="nearest", vmin=0.0, vmax=1.0
#     )
#     ax2.set_title("Back Matrix (6x7)")
#     fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Normalised Intensity")

#     fig.suptitle("Calibrating Sensor Baselines...", fontsize=14)

#     print(
#         "\nStarting mouse control & visualiser! Move mouse to top-left corner to stop."
#     )

#     # Start the animation loop
#     ani = animation.FuncAnimation(
#         fig, update, interval=15, blit=False, cache_frame_data=False
#     )

#     plt.tight_layout()
#     plt.show()  # This will block until the window is closed or failsafe is triggered

#     # Clean up
#     if ser and ser.is_open:
#         ser.close()
#     print("\nProgramme exited cleanly.")


import pickle
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

MODEL_PATH = "gesture_cnn_model.keras"  # Updated to Keras model
ENCODER_PATH = "label_encoder.pkl"

MOUSE_MOVE_DISTANCE = 30
MOUSE_MOVE_DELAY = 0.05
CLICK_DELAY = 0.5  # Prevents rapid-fire clicking when squeezing

# Global State Variables
baseline = None
stored_baseline = None  # To keep the calculated baseline safe when toggled off
use_calibration = True  # Toggle state
baseline_samples = []
history_buffer = deque(maxlen=SMOOTHING_FRAMES)
last_mouse_move_time = 0
last_click_time = 0
ser = None
model = None
label_encoder = None


def connect_to_teensy(port=PORT, baudrate=BAUD_RATE):
    while True:
        try:
            connection = serial.Serial(port, baudrate, timeout=0.1)
            print(f"Connected to {port} at {baudrate} baud!")
            return connection
        except (serial.SerialException, FileNotFoundError):
            print("Waiting for Teensy...")
            time.sleep(1)


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
    """Handles keyboard presses whilst the Matplotlib window is in focus."""
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


def update(frame):
    global \
        baseline, \
        stored_baseline, \
        baseline_samples, \
        history_buffer, \
        ser, \
        last_mouse_move_time, \
        last_click_time

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
            if stored_baseline is None:
                baseline_samples.append(last_valid_array)
                fig.suptitle(
                    f"Calibrating... {len(baseline_samples)}/{NUM_BASELINE_SAMPLES}",
                    fontsize=14,
                )

                if len(baseline_samples) >= NUM_BASELINE_SAMPLES:
                    stored_baseline = np.mean(baseline_samples, axis=0)
                    baseline = stored_baseline
                    fig.suptitle(
                        "Sensor Matrices Live (Press 'c' to toggle calibration)",
                        fontsize=14,
                    )
                    print(
                        "\nCalibration complete! Matrices and mouse control are live."
                    )
                return [im1, im2]

            # Preprocessing
            zeroed_array = last_valid_array - baseline
            history_buffer.append(zeroed_array)
            smoothed_array = np.mean(history_buffer, axis=0)
            normalized_array = np.clip(smoothed_array / MAX_EXPECTED_VALUE, 0.0, 1.0)

            # Matrix Slicing for Visualiser
            front_matrix = normalized_array[:49].reshape((7, 7))
            back_matrix = normalized_array[49:].reshape((6, 7))

            im1.set_array(front_matrix)
            im2.set_array(back_matrix)

            # Machine Learning Prediction (CNN Format)
            front_reshape = front_matrix.reshape(1, 7, 7)
            back_reshape = back_matrix.reshape(1, 6, 7)
            # Pad the back matrix to 7x7
            back_padded = np.pad(
                back_reshape, ((0, 0), (0, 1), (0, 0)), mode="constant"
            )

            # Stack into (1, 7, 7, 2)
            cnn_features = np.stack([front_reshape, back_padded], axis=-1)

            # Predict probabilities and get the highest index
            predicted_probs = model.predict(cnn_features, verbose=0)
            predicted_idx = np.argmax(predicted_probs, axis=1)[0]
            gesture = label_encoder.inverse_transform([predicted_idx])[0]

            current_time = time.time()

            # Execute mouse movement, click logic, and UI updates
            if gesture == "none":
                # Explicitly show 'none' so the user knows the model is resting
                fig.suptitle(
                    f"Action: NONE (Press 'c' to toggle calibration)", fontsize=14
                )
                print(f"Action: NONE    ", end="\r")

            elif gesture == "squeeze":
                if current_time - last_click_time >= CLICK_DELAY:
                    execute_action(gesture)
                    last_click_time = current_time
                    fig.suptitle(
                        f"Action: CLICK (Squeeze) (Press 'c' to toggle calibration)",
                        fontsize=14,
                    )
                    print(f"Action: CLICK   ", end="\r")

            else:
                if current_time - last_mouse_move_time >= MOUSE_MOVE_DELAY:
                    execute_action(gesture)
                    last_mouse_move_time = current_time
                    fig.suptitle(
                        f"Action: {gesture.upper()} (Press 'c' to toggle calibration)",
                        fontsize=14,
                    )
                    print(f"Action: {gesture.upper():8s}", end="\r")

    except pyautogui.FailSafeException:
        print(
            "\n\nFail-safe triggered! You moved the mouse into the corner of the screen."
        )
        plt.close(fig)  # Close the Matplotlib window to exit the programme cleanly
    except serial.SerialException:
        print("\nConnection lost. Reconnecting...")
        fig.suptitle("Connection lost. Reconnecting...", color="red", fontsize=14)
        ser.close()
        ser = connect_to_teensy()

    return [im1, im2]


if __name__ == "__main__":
    # Setup PyAutoGUI safeguards
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0  # Handled manually via delays

    # Load machine learning models
    print("Loading models...")
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    model = load_model(MODEL_PATH)
    print(f"Model loaded. Classes: {label_encoder.classes_}")

    # Establish Serial Connection
    ser = connect_to_teensy()

    # Initialise the Matplotlib figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.canvas.manager.set_window_title("Gesture Control & Live Visualiser")

    # Bind the key press event to our figure
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

    fig.suptitle("Calibrating Sensor Baselines...", fontsize=14)

    print(
        "\nStarting mouse control & visualiser! Move mouse to top-left corner to stop."
    )

    # Start the animation loop
    ani = animation.FuncAnimation(
        fig, update, interval=15, blit=False, cache_frame_data=False
    )

    plt.tight_layout()
    plt.show()  # This will block until the window is closed or failsafe is triggered

    # Clean up
    if ser and ser.is_open:
        ser.close()
    print("\nProgramme exited cleanly.")
