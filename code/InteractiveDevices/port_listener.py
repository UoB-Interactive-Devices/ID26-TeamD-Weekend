import serial
import time



def connect_to_teensy(port="/dev/ttyACM0"):
    while True:
        try:
            ser = serial.Serial(port, 115200, timeout=1)
            print(f"Connected to {port}!")
            return ser
        except (serial.SerialException, FileNotFoundError):
            print("Waiting for Teensy...")
            time.sleep(1)

ser = connect_to_teensy()

def read_from_port():
    if ser.in_waiting > 0:  # Only read if there is actually data waiting
        data = ser.readline().decode('utf-8').strip()
        return data
    return None


while True:
    try:
        if ser.in_waiting > 0:
            line = ser.read().decode('utf-8').strip()
            if line:
                print(f"Teensy says: {line}")
    except serial.SerialException:
        print("Connection lost. Reconnecting...")
        ser = connect_to_teensy()

ser.close()