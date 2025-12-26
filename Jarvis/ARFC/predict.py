import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import joblib
import os
import uuid

# --- Constants ---
DURATION = 3  # seconds
SAMPLE_RATE = 16000
TEMP_FILENAME = f"temp_{uuid.uuid4().hex[:8]}.wav"

# --- Load trained model ---
if not os.path.exists("voice_command_model.pkl"):
    print("Model file not found. Please train the model first.")
    exit()
model = joblib.load("voice_command_model.pkl")
print("✅ Model loaded.")

# --- Record new voice command ---
print("\n🎤 Speak your command (recording 3 seconds)...")
recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
sd.wait()
sf.write(TEMP_FILENAME, recording, SAMPLE_RATE)
print(f"✅ Recording saved to {TEMP_FILENAME}")

# --- Extract MFCC features from the new file ---
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

features = extract_mfcc(TEMP_FILENAME)

# --- Predict command ---
# predicted_label = model.predict([features])[0]
probs = model.predict_proba([features])[0]
for label, prob in zip(model.classes_, probs):
    print(f"{label}: {prob:.2f}")
predicted_label = model.classes_[np.argmax(probs)]

print(f"\n🤖 Predicted Command: **{predicted_label.upper()}**")

# --- Clean up temp file ---
os.remove(TEMP_FILENAME)
