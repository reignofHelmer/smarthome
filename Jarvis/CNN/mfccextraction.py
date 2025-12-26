import os
import librosa
import numpy as np
import joblib

try:
    from tqdm import tqdm  # optional for progress bar
except ImportError:
    tqdm = lambda x: x

# --- SETTINGS ---
DATA_DIR = r"C:\Users\MICHEAL\Desktop\CD3\MY CODING FILES\Jarvis\Jarvis\audiocomms"
COMMANDS = ['offlight','onlight','opendoor','lockdoor']
N_MFCC = 40
MAX_LEN = 40
SAMPLE_RATE = 16000

# --- FUNCTION TO EXTRACT MFCC SEQUENCE ---
def extract_mfcc_sequence(file_path, n_mfcc=N_MFCC, max_len=MAX_LEN, sr=SAMPLE_RATE):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    y = y / np.max(np.abs(y))  # normalize audio to -1 to 1
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = librosa.util.fix_length(mfcc, size=max_len, axis=1)  # pad or trim to max_len
    return mfcc

# --- LOAD ALL FILES ---
X = []
y = []

for label in COMMANDS:
    folder_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        continue

    print(f"Processing: {label}")
    for file in tqdm(os.listdir(folder_path)):
        if file.endswith(".wav"):
            full_path = os.path.join(folder_path, file)
            try:
                features = extract_mfcc_sequence(full_path)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"⚠️ Error processing {full_path}: {e}")

print(f"\n✅ Finished. Total samples: {len(X)}")

# --- SAVE TO DISK ---
joblib.dump((X, y), "voice_dataset.pkl")
print("✅ Features and labels saved to voice_dataset.pkl")