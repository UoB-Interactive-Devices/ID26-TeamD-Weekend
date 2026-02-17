import serial
import time

ser = serial.Serial("/dev/ttyUSB0") # RENAME TO PORT THAT MC IS ON
ser.baudrate = 9600 # set buadrate to match the baudrate of the microcontroller
ser.timeout = 0.1

def read_from_port():
    data = ser.readline() # this is blocking, it will wait until it receives a newline character
    return data

