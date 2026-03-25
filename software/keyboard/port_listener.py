import serial
import time
import json
import pyautogui
from numpy._core.defchararray import capitalize

mode = "alphabetic" # "alphabetic"/"numeric" to refer to json  

with open("chords.json", "r") as jsonfile:
    chords_dict = json.load(jsonfile) # load the chords dictionary from the JSON file


def connect_to_teensy(port="COM7"):
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
    data = ser.readline().decode('utf-8').strip()
    return data


def check_for_capital(s): #checks for a v in the string, which is a stand in character for SHIFT
    if 'v' in s:
        return True
    return False


def process_string(s):
    capitalise = False
    space = False
    processed_string = ''
    # Key strings are in alphabetical order to save space. Sort the string output from the MC before addressing the dictionary
    sorted_array = sorted(s)

    # remove junk modifier characters from the string before addressing the dictionary
    if 'v' in sorted_array:
        sorted_array.remove('v')  # V is stand in character for SHIFT
    if 'b' in sorted_array:
        sorted_array.remove('b')  # B is stand in character for SPACE
        space = True

    # create key
    sorted_string = ''.join(sorted_array)

    # check for key
    if sorted_string in chords_dict[mode]:
        processed_string = chords_dict[mode][sorted_string]

    # always add a space if there is a space character in the input, even if the key is not recognised (e.g. for partial chords)
    if space:  # B is stand in character for SPACE
        processed_string = processed_string + " "

    return processed_string

while True:
    try:
        if ser.in_waiting > 0:   # Only read if there is actually data waiting
            string = read_from_port()
            if string:
                line = process_string(string)
                if line == "backspace":
                    pyautogui.press('backspace')
                    print(f"Pressed: {string} -> {line}")
                elif line == "switch":
                    if mode == "alphabetic":
                        mode = "numeric"
                    else:
                        mode = "alphabetic"
                    print("Switched to mode",mode)
                else:
                    if check_for_capital(string):
                        line = line.capitalize()

                    print(f"Pressed: {string} -> {line}")

                    for i in range(len(line)): # write with pyautogui one character at a time to allow for capitalisation and spaces (and eventually partials)
                        pyautogui.press(line[i])

    except serial.SerialException:
        print("Connection lost. Reconnecting...")
        ser = connect_to_teensy()
