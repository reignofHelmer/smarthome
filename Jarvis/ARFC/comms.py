# Python example using pyaudio
import sounddevice as sd
import soundfile as sf
import os
import glob

# Set your target folder path (use raw string to handle backslashes)
folder = r"C:\Users\MICHEAL\Desktop\CD3\MY CODING FILES\Jarvis\Jarvis\audiocomms\onlight"
os.makedirs(folder, exist_ok=True)

def get_next_filename(base_name="on_light", extension="wav"):
    # Get all matching files in the folder
    pattern = os.path.join(folder, f"{base_name}_*.{extension}")
    existing_files = glob.glob(pattern)
    
    # Extract the next number
    count = len(existing_files) + 1
    filename = f"{base_name}_{count:02d}.{extension}"
    return os.path.join(folder, filename)

def record_voice(filename, duration=3, fs=16000):
    print(f"Recording: {filename}")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(filename, recording, fs)
    print("Recording complete.\n")

# Generate a new filename and record
next_file = get_next_filename("turn_on_light")
record_voice(next_file)