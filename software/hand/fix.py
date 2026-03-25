import csv

INPUT_FILE = "processed_gesture_data.csv"
OUTPUT_FILE = "repaired_gesture_data.csv"
EXPECTED_COLUMNS = 94  # 91 sensors + label + participant + session_id


def repair_csv():
    print(f"Scanning {INPUT_FILE} for corrupted rows...")
    valid_rows = 0
    bad_rows = 0

    with (
        open(INPUT_FILE, "r", encoding="utf-8") as infile,
        open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile,
    ):
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row_num, row in enumerate(reader, 1):
            if len(row) == EXPECTED_COLUMNS:
                writer.writerow(row)
                valid_rows += 1
            else:
                print(
                    f"Removed line {row_num}: Found {len(row)} columns instead of {EXPECTED_COLUMNS}."
                )
                bad_rows += 1

    print(f"\nRepair complete!")
    print(f"✓ Kept {valid_rows} healthy rows.")
    print(f"✗ Discarded {bad_rows} corrupted rows.")
    print(f"Clean data saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    repair_csv()
