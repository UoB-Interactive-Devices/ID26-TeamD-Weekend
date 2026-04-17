import csv
from pathlib import Path

import numpy as np

TOTAL_SENSORS = 91
MAX_EXPECTED_VALUE = 4000.0

REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_FILE = REPO_ROOT / "data" / "fine_tuning_gesture_data.csv"
OUTPUT_FILE = REPO_ROOT / "data" / "processed_gesture_data2.csv"


def process_data():
    with (
        INPUT_FILE.open("r", newline="", encoding="utf-8") as infile,
        OUTPUT_FILE.open("w", newline="", encoding="utf-8") as outfile,
    ):
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader, None)
        if header:
            new_header = [f"sensor_{i}" for i in range(TOTAL_SENSORS)] + [
                "label",
                "participant",
                "session_id",
            ]
            writer.writerow(new_header)

        for row in reader:
            raw_sensors = np.array([float(val) for val in row[:TOTAL_SENSORS]])
            normalised_sensors = np.clip(raw_sensors / MAX_EXPECTED_VALUE, 0.0, 1.0)

            label = row[TOTAL_SENSORS]
            session_id = row[-1]

            new_row = [f"{val:.6f}" for val in normalised_sensors] + [
                label,
                "fine_tune_user",
                session_id,
            ]

            writer.writerow(new_row)

    print(f"Saved to {OUTPUT_FILE}.")


if __name__ == "__main__":
    process_data()
