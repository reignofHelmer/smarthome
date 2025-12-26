import serial
import time
import random

try:
    arduino = serial.Serial('COM3', 9600, timeout=1)
    time.sleep(2)
    print("Serial connection established.")
except serial.SerialException as e:
    print(f"Failed to connect to Arduino: {e}")
    exit()

commands = ["OPENDOOR", "CLOSEDOOR", "ONLIGHT"]

try:
    while True:
        cmd = random.choice(commands)
        print(f"Sending: {cmd}")
        arduino.write((cmd + '\n').encode())

        # Read Arduino response
        if arduino.in_waiting:
            response = arduino.readline().decode().strip()
            print(f"Arduino says: {response}")

        time.sleep(3)
except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    arduino.close()
    print("Serial connection closed.")