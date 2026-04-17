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

PASSES = 6
SAMPLES_PER_PASS = 500
CLASSES = ["left", "right", "up", "down", "none", "squeeze"]

NUM_BASELINE_SAMPLES = 30
SMOOTHING_FRAMES = 3
MAX_EXPECTED_VALUE = 4000.0


OUTPUT_FILE = "data/processed_gesture_data.csv"


def connect_to_teensy():
    print(f"Connecting to {PORT}...")
    return serial.Serial(PORT, BAUD_RATE, timeout=0.1)


def calibrate_sensors(ser):
    print("Calibrating baselines...")
    baseline_samples = []

    while len(baseline_samples) < NUM_BASELINE_SAMPLES:
        if ser.in_waiting > 0:
            lines = ser.read_all().decode().split("\r\n")
            for line in reversed(lines):
                if line:
                    data = [int(x) for x in line.split(",")]
                    if len(data) == TOTAL_SENSORS:
                        baseline_samples.append(np.array(data))
                        break

    return np.mean(baseline_samples, axis=0)


def collect_batch(class_name, participant, current_pass, output_file, ser, baseline):
    input(f"Press Enter for '{class_name}' (Pass {current_pass}/{PASSES})...")

    session_id = str(int(time.time()))
    history = deque(maxlen=SMOOTHING_FRAMES)
    ser.reset_input_buffer()

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)

        for _ in tqdm(range(SAMPLES_PER_PASS), leave=False):
            data = []
            while len(data) != TOTAL_SENSORS:
                line = ser.readline().decode().strip()
                if line:
                    data = [int(x) for x in line.split(",")]

            raw = np.array(data)
            history.append(raw - baseline)
            smoothed = np.mean(history, axis=0)
            normalised = np.clip(smoothed / MAX_EXPECTED_VALUE, 0.0, 1.0)

            writer.writerow(normalised.tolist() + [class_name, participant, session_id])


def main():
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", newline="") as f:
            headers = [f"sensor_{i}" for i in range(TOTAL_SENSORS)] + [
                "label",
                "participant",
                "session_id",
            ]
            csv.writer(f).writerow(headers)

    ser = connect_to_teensy()
    ser.write(b"O")
    time.sleep(1)

    baseline = calibrate_sensors(ser)

    participant = input("Participant ID: ").strip() or "unknown"
    selection = input(f"Classes ({', '.join(CLASSES)}) or 'all': ").strip().lower()

    if selection == "all":
        selected_classes = CLASSES
    else:
        selected_classes = [c for c in selection.split() if c in CLASSES]

    for current_pass in range(1, PASSES + 1):
        ser.write(b"O")
        print(
            "\nTake your hand off the sensor, then replace it in a new, natural position."
        )
        input(f"Ready for Pass {current_pass}? Press Enter...")

        ser.write(b"F")
        time.sleep(1)

        pass_classes = list(selected_classes)
        random.shuffle(pass_classes)

        for class_name in pass_classes:
            collect_batch(
                class_name, participant, current_pass, OUTPUT_FILE, ser, baseline
            )
            time.sleep(0.5)

    ser.write(b"O")
    ser.close()
    print("\nCollection finished.")


if __name__ == "__main__":
    main()
