import serial
import csv
import time

PORT = "/dev/ttyACM0"
BAUD_RATE = 115200
OUTPUT_FILE = "matrix_data.csv"
SAMPLES_TO_COLLECT = 100


def collect_data():
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"Connected to {PORT}. Collecting data...")

        with open(OUTPUT_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)

            count = 0
            while count < SAMPLES_TO_COLLECT:
                line = ser.readline().decode("utf-8").strip()

                if line:
                    data_point = [int(val) for val in line.split(",")]

                    if len(data_point) == 20:
                        writer.writerow(data_point)
                        count += 1
                        if count % 10 == 0:
                            print(f"Captured {count}/{SAMPLES_TO_COLLECT} samples...")

        print(f"Done! Data saved to {OUTPUT_FILE}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if "ser" in locals():
            ser.close()

if __name__ == "__main__":
    collect_data()