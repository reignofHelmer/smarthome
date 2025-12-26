import os
import librosa
import numpy as np
import joblib

# --- SETTINGS ---
DATA_DIR = r"C:\Users\MICHEAL\Desktop\CD3\MY CODING FILES\Jarvis\Jarvis\audiocomms"
COMMANDS = ['offlight', 'onlight', 'opendoor', 'lockdoor']
N_MFCC = 13
SAMPLE_RATE = 16000

# --- FUNCTION TO EXTRACT MFCC ---
def extract_mfcc(file_path, n_mfcc=N_MFCC):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

# --- LOAD ALL FILES ---
X = []
y = []

for label in COMMANDS:
    folder_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    print(f"Processing: {label}")
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            full_path = os.path.join(folder_path, file)
            try:
                features = extract_mfcc(full_path)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"Error processing {full_path}: {e}")

print(f"\n✅ Finished. Total samples: {len(X)}")

# --- OPTIONAL: Save to disk for later reuse ---
joblib.dump((X, y), "voice_dataset.pkl")
print("✅ Features and labels saved to voice_dataset.pkl")