# train_improved_lstm.py
"""
Improved training pipeline for voice-command recognition:
- Stacks MFCC + delta + delta-delta features (feature dimension increases by 3x)
- Lightweight on-feature augmentations: gaussian noise and time-shift (roll)
- Conv1D -> Bidirectional LSTM hybrid model
- EarlyStopping + ReduceLROnPlateau callbacks
- Saves best model & scaler
- Produces confusion matrix & classification report

Assumptions:
- DATA_FILE contains (X_raw, y_raw) as loaded by joblib.load(DATA_FILE)
- Each X_raw[i] is a 2D numpy array: (N_MFCC, MAX_LEN)
- y_raw are labels as strings matching COMMANDS (possibly augmented labels ending with '_a')
"""
# train_lstm.py (optimized with float32 + safe augmentation)

import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from collections import Counter

# --- SETTINGS ---
DATA_FILE = "voice_dataset.pkl"
N_MFCC = 40
MAX_LEN = 40
SAMPLE_RATE = 16000
COMMANDS = ['offlight', 'onlight', 'opendoor', 'plockdoor']
NUM_CLASSES = len(COMMANDS)
AUGMENT_FACTOR = 1  # keep light to prevent memory blow-up

# --- MAP AUGMENTED LABELS ---
def map_label(label):
    return label.replace('_a', '')

# --- LOAD DATA ---
print("Loading dataset...")
X_raw, y_raw = joblib.load(DATA_FILE)
print(f"Loaded {len(X_raw)} samples from {DATA_FILE}")

# --- STACK DELTA FEATURES ---
def add_deltas(mfcc):
    """Stack MFCC + delta + delta-delta"""
    import librosa
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta1, delta2])

print("Stacking delta features...")
X_stacked = [add_deltas(x.reshape(N_MFCC, MAX_LEN)) for x in X_raw]
X_stacked = np.array(X_stacked, dtype=np.float32)  # force float32
print("Stacked feature shape:", X_stacked.shape)

# Reshape for LSTM input (samples, timesteps, features)
X = np.transpose(X_stacked, (0, 2, 1))  # -> (samples, 40, 120)
print("Re-shaped X ->", X.shape)

# --- OPTIONAL DATA AUGMENTATION ---
print(f"Applying on-feature augmentation (factor={AUGMENT_FACTOR}) ...")
aug_X, aug_y = [], []
for i in range(len(X)):
    aug_X.append(X[i])
    aug_y.append(y_raw[i])
    for _ in range(AUGMENT_FACTOR):
        noise = np.random.normal(0, 0.01, X[i].shape).astype(np.float32)
        aug_X.append(X[i] + noise)
        aug_y.append(y_raw[i])

X = np.array(aug_X, dtype=np.float32)  # ✅ float32 to save memory
y = np.array([COMMANDS.index(map_label(label)) for label in aug_y])
print("Final dataset shape:", X.shape, "Labels:", y.shape)

# --- NORMALIZE FEATURES ---
scaler = StandardScaler()
X_flat = X.reshape(-1, X.shape[-1])  # flatten across time
X_flat = scaler.fit_transform(X_flat)
X = X_flat.reshape(X.shape).astype(np.float32)
joblib.dump(scaler, "scaler_lstm.pkl")
print("✅ Features normalized and scaler saved.")

# --- TRAIN-TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- BUILD MODEL ---
def build_lstm(input_shape, num_classes, alpha=1e-4, dropout=0.3, lr=0.001):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(alpha)), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Bidirectional(LSTM(64, kernel_regularizer=l2(alpha))))
    model.add(BatchNormalization())
    model.add(Dropout(dropout + 0.1))

    model.add(Dense(128, activation='relu', kernel_regularizer=l2(alpha)))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

print("Building LSTM model...")
model = build_lstm(X_train.shape[1:], NUM_CLASSES)

# --- TRAINING ---
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# --- EVALUATION ---
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n🏆 LSTM Test Accuracy: {test_acc * 100:.2f}%")

# --- SAVE MODEL ---
model.save("lstm_voice_model_best.keras")
print("✅ Best LSTM model saved to lstm_voice_model_best.keras")

# --- PLOT RESULTS ---
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title('Accuracy')
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title('Loss')
plt.show()

# --- CONFUSION MATRIX & REPORT ---
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred, labels=range(NUM_CLASSES))
disp = ConfusionMatrixDisplay(cm, display_labels=COMMANDS)
disp.plot()
plt.savefig("confusion_matrix_lstm.png")
plt.show()

report = classification_report(y_test, y_pred, target_names=COMMANDS)
print("Classification Report:\n", report)
print("Label distribution:", Counter(map_label(label) for label in y_raw))
