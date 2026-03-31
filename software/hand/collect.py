import csv
import os
import random
import time
from collections import deque

import numpy as np
import serial
from tqdm import tqdm

PORT = "COM5"
BAUD_RATE = 2000000
TOTAL_SENSORS = 91

# Collection Parameters
PASSES = 6
SAMPLES_PER_PASS = 500
CLASSES = ["left", "right", "up", "down", "none"]

# Preprocessing Parameters
NUM_BASELINE_SAMPLES = 30
SMOOTHING_FRAMES = 3
MAX_EXPECTED_VALUE = 4000.0


def connect_to_teensy():
    while True:
        try:
            ser = serial.Serial(PORT, BAUD_RATE, timeout=0.1)
            print(f"Connected to {PORT} at {BAUD_RATE} baud!")
            return ser
        except (serial.SerialException, FileNotFoundError):
            print("Waiting for Teensy...")
            time.sleep(1)


def calibrate_sensors(ser):
    print("\nCalibrating sensor baselines. Please do not touch the sensors...")
    baseline_samples = []

    with tqdm(total=NUM_BASELINE_SAMPLES, desc="Calibrating") as pbar:
        while len(baseline_samples) < NUM_BASELINE_SAMPLES:
            if ser.in_waiting > 0:
                raw_bytes = ser.read_all()
                lines = raw_bytes.decode("utf-8", errors="ignore").split("\r\n")

                for line in reversed(lines):
                    if line:
                        try:
                            data = [int(x) for x in line.split(",")]
                            if len(data) == TOTAL_SENSORS:
                                baseline_samples.append(np.array(data))
                                pbar.update(1)
                                break
                        except ValueError:
                            continue

    return np.mean(baseline_samples, axis=0)


def collect_batch(class_name, participant_id, current_pass, output_file, ser, baseline):
    print(f"\n{'=' * 60}")
    print(
        f"Participant: {participant_id} | Class: {class_name.upper()} | Pass: {current_pass}/{PASSES}"
    )
    print(f"{'=' * 60}")
    input(
        f"Get ready for the '{class_name.upper()}' gesture. Press Enter to start collecting {SAMPLES_PER_PASS} samples..."
    )

    session_id = str(int(time.time()))
    history_buffer = deque(maxlen=SMOOTHING_FRAMES)

    ser.reset_input_buffer()

    with open(output_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        count = 0

        with tqdm(total=SAMPLES_PER_PASS, desc=f"Collecting {class_name}") as pbar:
            while count < SAMPLES_PER_PASS:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    try:
                        data = [int(x) for x in line.split(",")]
                        if len(data) == TOTAL_SENSORS:
                            raw_array = np.array(data)

                            # Preprocessing Math
                            zeroed_array = raw_array - baseline
                            history_buffer.append(zeroed_array)
                            smoothed_array = np.mean(history_buffer, axis=0)
                            normalized_array = np.clip(
                                smoothed_array / MAX_EXPECTED_VALUE, 0.0, 1.0
                            )

                            # Save processed flat array + metadata
                            row = normalized_array.tolist() + [
                                class_name,
                                participant_id,
                                session_id,
                            ]
                            writer.writerow(row)

                            count += 1
                            pbar.update(1)
                    except ValueError:
                        continue

    print(f"✓ '{class_name.upper()}' complete for Pass {current_pass}!")


def main():
    output_file = "processed_gesture_data.csv"

    if not os.path.exists(output_file):
        with open(output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            header = [f"sensor_{i}" for i in range(TOTAL_SENSORS)] + [
                "label",
                "participant",
                "session_id",
            ]
            writer.writerow(header)

    ser = connect_to_teensy()
    
    # Ensure hand is open before starting calibration
    ser.write(b'O')
    time.sleep(1)
    
    baseline = calibrate_sensors(ser)
    print("Calibration complete!\n")

    participant_id = (
        input("Enter participant ID (e.g., person_1): ").strip() or "unknown"
    )

    print("\nAvailable classes:", ", ".join(CLASSES))
    selection = (
        input("Enter classes to collect (space-separated, or 'all'): ").strip().lower()
    )

    if selection == "all":
        selected_classes = CLASSES
    else:
        selected_classes = [c for c in selection.split() if c in CLASSES]

    # Loop passes first, and randomise the order of classes within each pass
    for current_pass in range(1, PASSES + 1):
        
        # --- NEW: Open the hand between passes ---
        ser.write(b'O')
        
        print(f"\n\n{'#' * 60}")
        print(f"--- STARTING PASS {current_pass}/{PASSES} ---")
        print(
            "CRITICAL: Take your hand off the sensor entirely, then place it back down in a slightly new, natural position."
        )
        print(
            "Do NOT lift your hand between gestures during this pass. Keep it anchored."
        )
        print(f"{'#' * 60}")
        input("Press Enter when your hand is placed and you are ready to begin...")

        # --- NEW: Close the hand to a fist for the collection pass ---
        print("Closing to FIST position...")
        ser.write(b'F')
        time.sleep(1) # Give the servos physical time to move before reading data

        # Create a fresh copy of the selected classes and shuffle them
        pass_classes = list(selected_classes)
        random.shuffle(pass_classes)

        for class_name in pass_classes:
            collect_batch(
                class_name, participant_id, current_pass, output_file, ser, baseline
            )
            time.sleep(0.5)

    print("\nData collection entirely complete. Thank you!")
    # Open the hand at the very end to release
    ser.write(b'O')
    ser.close()


if __name__ == "__main__":
    main()
