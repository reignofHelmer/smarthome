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
COMMANDS = ['offlight', 'onlight', 'opendoor', 'plockdoor']

# --- LOAD MODEL AND SCALER ---
model = tf.keras.models.load_model("lstm_voice_model_best.keras")
scaler = joblib.load("scaler_lstm.pkl")
print("✅ LSTM model and scaler loaded.")

# --- RECORD AUDIO ---
temp_filename = f"temp_{uuid.uuid4().hex}.wav"
print(f"\n🎤 Speak your command (recording {DURATION} seconds)...")
recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
sd.wait()
sf.write(temp_filename, recording, SAMPLE_RATE)
print(f"✅ Recording saved to {temp_filename}\n")

# --- FEATURE EXTRACTION (MFCC + delta + delta-delta) ---
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # Compute delta and delta-delta
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Stack into (120, frames)
    stacked = np.vstack([mfcc, delta1, delta2])

    # Pad or truncate to MAX_LEN
    if stacked.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - stacked.shape[1]
        stacked = np.pad(stacked, ((0, 0), (0, pad_width)), mode='constant')
    else:
        stacked = stacked[:, :MAX_LEN]

    return stacked.T  # shape (MAX_LEN, 120)

# --- START TIMING ---
start_time = time.time()

# Extract features
features = extract_features(temp_filename)  # (40, 120)

# Apply scaler row-wise (each frame separately, as during training)
features_scaled = scaler.transform(features)  # (40, 120)

# Add batch dimension
features_scaled = np.expand_dims(features_scaled, axis=0)  # (1, 40, 120)

# Predict
pred = model.predict(features_scaled, verbose=0)
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