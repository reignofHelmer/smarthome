import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import tensorflow as tf
import joblib
import uuid
import os
import time

# --- SETTINGS ---
SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 40
MAX_LEN = 40
COMMANDS = ['offlight', 'onlight', 'opendoor', 'lockdoor']

# --- LOAD MODEL AND SCALER ---
model = tf.keras.models.load_model("cnn_voice_model_best.keras")
scaler = joblib.load("scaler.pkl")
print("✅ Model and scaler loaded.")

# --- RECORD AUDIO ---
temp_filename = f"temp_{uuid.uuid4().hex}.wav"
print(f"\n🎤 Speak your command (recording {DURATION} seconds)...")
recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
sd.wait()
sf.write(temp_filename, recording, SAMPLE_RATE)
print(f"✅ Recording saved to {temp_filename}\n")

# --- PROCESS AUDIO ---
def extract_mfcc_sequence(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_LEN]
    return mfcc

# --- START TIMING ---
start_time = time.time()

# Extract features and make prediction
features = extract_mfcc_sequence(temp_filename)
features = features.reshape(1, -1)  # Flatten for scaler
features = scaler.transform(features)
features = features.reshape(1, N_MFCC, MAX_LEN, 1)

pred = model.predict(features, verbose=0)
end_time = time.time()

# --- OUTPUT ---
recognition_time = end_time - start_time
confidence = np.max(pred) * 100
pred_label = COMMANDS[np.argmax(pred)]

print(f"✅ Predicted Command: {pred_label.upper()} ({confidence:.2f}% confidence)")
print(f"🕒 Recognition Time: {recognition_time:.3f} seconds")

if confidence < 60:
    print("⚠️  Low confidence. You might want to try again.")

# --- CLEANUP ---
os.remove(temp_filename)