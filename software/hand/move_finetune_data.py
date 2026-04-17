import csv

import numpy as np

INPUT_FILE = "data/fine_tuning_gesture_data.csv"
OUTPUT_FILE = "data/processed_gesture_data2.csv"
TOTAL_SENSORS = 91
MAX_EXPECTED_VALUE = 4000.0


def process_data():
    with (
        open(INPUT_FILE, "r", newline="") as infile,
        open(OUTPUT_FILE, "w", newline="") as outfile,
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

        processed_count = 0
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
            processed_count += 1

    print(f"Saved to {OUTPUT_FILE}.")


if __name__ == "__main__":
    process_data()
