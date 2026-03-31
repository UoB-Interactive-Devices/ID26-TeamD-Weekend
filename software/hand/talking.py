import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import os
import pickle
import sys
import threading
import time
from collections import deque

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pyautogui
import serial
import speech_recognition as sr
from dotenv import load_dotenv
from openai import OpenAI
from tensorflow.keras.models import load_model

# Load environment variables
load_dotenv()

# Configuration Parameters
PORT = "COM5"
BAUD_RATE = 2000000
TOTAL_SENSORS = 91

NUM_BASELINE_SAMPLES = 30
SMOOTHING_FRAMES = 3
MAX_EXPECTED_VALUE = 4000.0

MODEL_PATH = "gesture_cnn_model.keras"
ENCODER_PATH = "label_encoder.pkl"

# Mouse Acceleration Parameters
BASE_MOVE_SPEED = 5.0  # Starting speed in pixels (for fine targeting)
MAX_MOVE_SPEED = 45.0  # Maximum speed in pixels
ACCELERATION_RATE = 1.5  # How much speed increases per frame whilst held
MOUSE_MOVE_DELAY = 0.05
CLICK_DELAY = 0.5

# Initialise OpenAI Client
try:
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception as e:
    print(f"Failed to initialise OpenAI client. Check your .env file: {e}")
    sys.exit(1)

# Premade Macros
AVAILABLE_MACROS = {
    "move_left": "Move the mouse cursor to the left",
    "move_right": "Move the mouse cursor to the right",
    "move_up": "Move the mouse cursor up",
    "move_down": "Move the mouse cursor down",
    "click": "Perform a standard quick left mouse click",
    "click_and_hold": "Hold the left mouse button down (drag) whilst the gesture is active",
    "scroll_up": "Scroll the page up",
    "scroll_down": "Scroll the page down",
    "volume_up": "Increase the system volume",
    "volume_down": "Decrease the system volume",
    "mute": "Mute or unmute the system volume",
    "play_pause": "Play or pause the current media",
    "copy": "Copy the currently selected text or item",
    "paste": "Paste the copied text or item",
    "undo": "Undo the last action",
    "switch_window": "Switch to the next open window or application",
    "zoom_in": "Zoom in on the screen or document",
    "zoom_out": "Zoom out on the screen or document",
}

DISCRETE_MACROS = [
    "click",
    "play_pause",
    "copy",
    "paste",
    "undo",
    "switch_window",
    "mute",
]

# Global State Variables
baseline = None
stored_baseline = None
use_calibration = True
history_buffer = deque(maxlen=SMOOTHING_FRAMES)

last_mouse_move_time = 0
last_click_time = 0
current_move_speed = BASE_MOVE_SPEED
is_mouse_held = False

ser = None
model = None
label_encoder = None
fist_closed = False
overlay_timer = 0

# Updated mapping to use the new click_and_hold macro for dragging
custom_gesture_mapping = {
    "left": "move_left",
    "right": "move_right",
    "up": "move_up",
    "down": "move_down",
    "squeeze": "click_and_hold",
}


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


def listen_and_transcribe():
    recogniser = sr.Recognizer()
    recogniser.pause_threshold = 1.0

    with sr.Microphone() as source:
        print("\n🎤 Listening... Tell me what you want this gesture to do.")
        recogniser.adjust_for_ambient_noise(source, duration=0.5)

        try:
            audio = recogniser.listen(source, timeout=7, phrase_time_limit=15)
            print("Processing speech...")
            text = recogniser.recognize_google(audio)
            print(f"🗣️ You said: '{text}'")
            return text
        except sr.WaitTimeoutError:
            print("Timed out listening for speech.")
            return None
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition service error: {e}")
            return None


def map_intent_to_macro(spoken_text):
    macro_descriptions = "\n".join([f"- {k}: {v}" for k, v in AVAILABLE_MACROS.items()])
    system_prompt = (
        "You are an intelligent assistant mapping a user's spoken command to a strict set of computer macros. "
        "Review the user's intent and select the single best matching macro from the list below.\n\n"
        f"Available Macros:\n{macro_descriptions}\n\n"
        "Reply ONLY with the exact macro key (e.g., 'move_left', 'click_and_hold', 'copy'). "
        "If the user's command does not match any macro, reply strictly with 'none'."
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": spoken_text},
            ],
            max_tokens=20,
            temperature=0,
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"LLM mapping error: {e}")
        return "none"


def run_onboarding():
    global custom_gesture_mapping, fist_closed
    print("\n" + "=" * 50)
    print("🚀 GESTURE ONBOARDING PROGRAMME")
    print("=" * 50)
    print("Default mappings are pre-loaded. You can skip this step if you wish.")
    print("1. Perform and hold a gesture for a few seconds to map a new command.")
    print("2. When prompted, speak your command aloud.")
    print("3. Press ENTER at any time to finish and launch the visualiser.")
    print("=" * 50 + "\n")

    onboarding_done = [False]

    def wait_for_enter():
        input()
        onboarding_done[0] = True

    threading.Thread(target=wait_for_enter, daemon=True).start()

    consecutive_frames = 0
    target_frames = 15
    last_seen = "none"

    print(
        "👉 Awaiting your first gesture... (or press ENTER to keep defaults and finish)"
    )

    while not onboarding_done[0]:
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
                normalised_array = np.clip(zeroed_array / MAX_EXPECTED_VALUE, 0.0, 1.0)

                front_matrix = normalised_array[:49].reshape((1, 7, 7))
                back_matrix = normalised_array[49:].reshape((1, 6, 7))
                back_padded = np.pad(
                    back_matrix, ((0, 0), (0, 1), (0, 0)), mode="constant"
                )
                cnn_features = np.stack([front_matrix, back_padded], axis=-1)

                predicted_probs = model(cnn_features, training=False).numpy()
                predicted_idx = np.argmax(predicted_probs, axis=1)[0]
                raw_gesture = label_encoder.inverse_transform([predicted_idx])[0]

                if raw_gesture != "none":
                    if raw_gesture == last_seen:
                        consecutive_frames += 1
                    else:
                        last_seen = raw_gesture
                        consecutive_frames = 1

                    if consecutive_frames >= target_frames:
                        print(f"\n✅ Stable gesture [{raw_gesture.upper()}] detected!")

                        if not fist_closed:
                            print("First movement detected! Sending grip command.")
                            ser.write(b"F")
                            fist_closed = True

                        ser.reset_input_buffer()
                        spoken_command = listen_and_transcribe()

                        if spoken_command and not onboarding_done[0]:
                            mapped_macro = map_intent_to_macro(spoken_command)
                            custom_gesture_mapping[raw_gesture] = mapped_macro

                            print(
                                f"\n🎯 Successfully mapped [{raw_gesture.upper()}] to: {mapped_macro}"
                            )
                            print(
                                "   (Want to change it? Just hold the gesture again.)"
                            )

                        consecutive_frames = 0
                        last_seen = "none"
                        print(
                            "\n👉 Awaiting next gesture... (or press ENTER to finish)"
                        )
                else:
                    consecutive_frames = 0
                    last_seen = "none"

            time.sleep(0.05)

        except Exception as e:
            print(f"Error reading gesture: {e}")
            time.sleep(1)

    print("\n" + "=" * 50)
    print("✅ Active Mappings:")
    for g, m in custom_gesture_mapping.items():
        print(f" - {g.upper()} -> {m}")
    if not custom_gesture_mapping:
        print(" - No custom mappings set. Defaulting to 'none'.")
    print("=" * 50 + "\n")
    time.sleep(1)


def execute_action(macro, speed=BASE_MOVE_SPEED):
    if macro == "move_left":
        pyautogui.move(-speed, 0)
    elif macro == "move_right":
        pyautogui.move(speed, 0)
    elif macro == "move_up":
        pyautogui.move(0, -speed)
    elif macro == "move_down":
        pyautogui.move(0, speed)
    elif macro == "click":
        pyautogui.click()
    elif macro == "scroll_up":
        pyautogui.scroll(200)
    elif macro == "scroll_down":
        pyautogui.scroll(-200)
    elif macro == "volume_up":
        pyautogui.press("volumeup")
    elif macro == "volume_down":
        pyautogui.press("volumedown")
    elif macro == "mute":
        pyautogui.press("volumemute")
    elif macro == "play_pause":
        pyautogui.press("playpause")
    elif macro == "copy":
        pyautogui.hotkey("ctrl", "c")
    elif macro == "paste":
        pyautogui.hotkey("ctrl", "v")
    elif macro == "undo":
        pyautogui.hotkey("ctrl", "z")
    elif macro == "switch_window":
        pyautogui.hotkey("alt", "tab")
    elif macro == "zoom_in":
        pyautogui.hotkey("ctrl", "+")
    elif macro == "zoom_out":
        pyautogui.hotkey("ctrl", "-")


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
        overlay_timer, \
        current_move_speed, \
        is_mouse_held

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

            predicted_probs = model(cnn_features, training=False).numpy()
            predicted_idx = np.argmax(predicted_probs, axis=1)[0]
            raw_gesture = label_encoder.inverse_transform([predicted_idx])[0]

            active_macro = custom_gesture_mapping.get(raw_gesture, "none")
            current_time = time.time()

            # Maintain robotic hand state
            if raw_gesture != "none" and not fist_closed:
                print(f"\nGesture [{raw_gesture}] detected! Sending grip command.")
                ser.write(b"F")
                fist_closed = True

            # --- Acceleration Logic ---
            if active_macro in ["move_left", "move_right", "move_up", "move_down"]:
                current_move_speed = min(
                    current_move_speed + ACCELERATION_RATE, MAX_MOVE_SPEED
                )
            else:
                current_move_speed = BASE_MOVE_SPEED

            # --- Mouse Down/Up Logic (Drag) ---
            if active_macro == "click_and_hold":
                if not is_mouse_held:
                    pyautogui.mouseDown()
                    is_mouse_held = True
            else:
                if is_mouse_held:
                    pyautogui.mouseUp()
                    is_mouse_held = False

            # --- Execute Actions & Visual Wiggles ---
            if active_macro == "none" or raw_gesture == "none":
                print(f"Action: NONE    ", end="\r")

                if overlay_timer > 0:
                    overlay_timer -= 1
                    action_overlay.set_alpha(overlay_timer / 15.0)
                else:
                    action_overlay.set_visible(False)

            # Route for discrete, one-off macros
            elif active_macro in DISCRETE_MACROS:
                if current_time - last_click_time >= CLICK_DELAY:
                    execute_action(active_macro)

                    action_overlay.set_text(
                        f"💥 {active_macro.replace('_', ' ').upper()}"
                    )
                    action_overlay.set_color("yellow")
                    action_overlay.set_alpha(1.0)
                    action_overlay.set_visible(True)
                    overlay_timer = 15

                    if active_macro == "click":
                        ser.write(b"O")
                        time.sleep(0.2)
                        ser.write(b"F")
                        ser.reset_input_buffer()

                    last_click_time = time.time()
                    print(f"Action: {active_macro.upper():12s}", end="\r")

            # Route for continuous macros (moving, dragging, scrolling)
            else:
                if active_macro == "click_and_hold":
                    action_overlay.set_text("HOLDING")
                    action_overlay.set_color("yellow")
                    action_overlay.set_alpha(1.0)
                    action_overlay.set_visible(True)
                    overlay_timer = 15
                    print(f"Action: HOLDING         ", end="\r")

                elif current_time - last_mouse_move_time >= MOUSE_MOVE_DELAY:
                    execute_action(active_macro, speed=current_move_speed)
                    last_mouse_move_time = current_time

                    action_overlay.set_text(active_macro.replace("_", " ").upper())
                    action_overlay.set_color("cyan")
                    action_overlay.set_alpha(1.0)
                    action_overlay.set_visible(True)
                    overlay_timer = 15

                    print(f"Action: {active_macro.upper():12s}", end="\r")

    except pyautogui.FailSafeException:
        print(
            "\n\nFail-safe triggered! You moved the mouse into the corner of the screen."
        )
        # Ensure mouse isn't stuck down if they fail-safe whilst dragging
        if is_mouse_held:
            pyautogui.mouseUp()
            is_mouse_held = False
        plt.close(fig)
    except serial.SerialException:
        print("\nConnection lost. Reconnecting...")
        if is_mouse_held:
            pyautogui.mouseUp()
            is_mouse_held = False
        fig.suptitle("Connection lost. Reconnecting...", color="red", fontsize=14)
        ser.close()
        ser = connect_to_teensy()

    return [im1, im2, action_overlay]


if __name__ == "__main__":
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

    run_onboarding()

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
        "\nStarting live control & visualiser! Move mouse to top-left corner or press 'q' to stop."
    )

    ani = animation.FuncAnimation(
        fig, update, interval=40, blit=False, cache_frame_data=False
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
        # Release the mouse properly if exiting cleanly
        if is_mouse_held:
            pyautogui.mouseUp()
        if ser and ser.is_open:
            ser.write(b"O")
            time.sleep(0.5)
            ser.close()
        print("Programme exited cleanly.")
        sys.exit(0)
