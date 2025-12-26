import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import tensorflow as tf
import joblib
import uuid
import os
import time
import serial
import pyttsx3

# --- SETTINGS ---
SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 40
MAX_LEN = 40
COMMANDS = ['offlight', 'onlight', 'opendoor', 'lockdoor']

# --- TTS INITIALIZATION ---
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# --- LOAD MODEL AND SCALER ---
model = tf.keras.models.load_model("cnn_voice_model_best.keras")
scaler = joblib.load("scaler.pkl")
print("✅ Model and scaler loaded.")

# --- SETUP SERIAL ---
def connect_serial(port='COM3', baudrate=9600):
    try:
        arduino = serial.Serial(port, baudrate)
        time.sleep(2)
        print("✅ Serial connection established.")
        return arduino
    except Exception as e:
        print(f"❌ Serial connection failed: {e}")
        return None

arduino = connect_serial()

# --- FEATURE EXTRACTION ---
def extract_mfcc_sequence(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_LEN]
    return mfcc

# --- RECORD + PREDICT ---
def listen_and_predict():
    temp_filename = f"temp_{uuid.uuid4().hex}.wav"
    print(f"\n🎤 Speak your command (recording {DURATION} seconds)...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    sf.write(temp_filename, recording, SAMPLE_RATE)

    features = extract_mfcc_sequence(temp_filename)
    features = features.reshape(1, -1)
    features = scaler.transform(features)
    features = features.reshape(1, N_MFCC, MAX_LEN, 1)

    pred = model.predict(features, verbose=0)
    confidence = np.max(pred) * 100
    pred_label = COMMANDS[np.argmax(pred)]

    os.remove(temp_filename)
    return pred_label, confidence

# --- MAIN LOOP ---
try:
    while True:
        label, conf = listen_and_predict()
        print(f"✅ Predicted Command: {label.upper()} ({conf:.2f}%)")

        # Log prediction
        with open("command_log.txt", "a") as log:
            log.write(f"{time.ctime()}: {label.upper()} ({conf:.2f}%)\n")

        if conf < 60:
            print("⚠️ Confidence too low. Retrying...")
            speak("Sorry, I didn't catch that.")
            time.sleep(2)
            continue

        if arduino is None or not arduino.is_open:
            print("❌ Arduino not connected. Attempting to reconnect...")
            arduino = connect_serial()
            if arduino is None:
                speak("Connection to hardware failed.")
                continue

        # --- Send to Arduino ---
        arduino.write((label.upper() + '\n').encode())
        print(f"📤 Sent to Arduino: {label.upper()}")
        speak(f"{label.replace('onlight', 'Turning on the light').replace('offlight', 'Turning off the light').replace('opendoor', 'Opening the door').replace('lockdoor', 'Locking the door')}")

        try:
            time.sleep(0.5)
            while arduino.in_waiting > 0:
                response = arduino.readline().decode().strip()
                print(f"🔁 Arduino: {response}")
        except Exception as e:
            print(f"⚠️ Error reading from Arduino: {e}")
            speak("There was an error reading from Arduino.")

        print("⏳ Waiting 5 seconds before next command...\n")
        time.sleep(5)

except KeyboardInterrupt:
    print("\n👋 Exiting on keyboard interrupt.")
    speak("Goodbye.")

finally:
    if arduino:
        arduino.close()
    print("🛑 Program terminated and serial connection closed.")