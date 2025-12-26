import sounddevice as sd
import soundfile as sf
import os
import glob
import time

folder = r"C:\Users\MICHEAL\Desktop\CD3\MY CODING FILES\Jarvis\Jarvis\audiocomms\onlight"
os.makedirs(folder, exist_ok=True)

def get_next_filename(base_name="turn_on_light", extension="wav"):
    pattern = os.path.join(folder, f"{base_name}_*.{extension}")
    existing_files = glob.glob(pattern)
    count = len(existing_files) + 1
    filename = f"{base_name}_{count:02d}.{extension}"
    return os.path.join(folder, filename)

def record_voice(filename, duration=3, fs=16000):
    print(f"Recording: {filename}")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(filename, recording, fs)
    print("Recording complete.\n")

# Infinite loop with 3-second delay between recordings
try:
    while True:
        next_file = get_next_filename("turn_on_light")
        record_voice(next_file)
        print("Waiting 3 seconds before next recording...\n")
        time.sleep(3)

except KeyboardInterrupt:
    print("Recording loop stopped by Helmer.")