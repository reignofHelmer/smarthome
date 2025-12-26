import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
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
COMMANDS = ['offlight', 'onlight', 'opendoor', 'lockdoor']
NUM_CLASSES = len(COMMANDS)

# --- MAP AUGMENTED LABELS ---
def map_label(label):
    return label.replace('_a', '')

# --- LOAD DATA ---
X_raw, y_raw = joblib.load(DATA_FILE)
print(f"Loaded {len(X_raw)} samples.")

# --- PREPARE DATA ---
X = np.array(X_raw)
y = np.array([COMMANDS.index(map_label(label)) for label in y_raw])

# --- NORMALIZE FEATURES ---
scaler = StandardScaler()
X = X.reshape(-1, N_MFCC * MAX_LEN)
X = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")  # ✅ Save the scaler for later use in prediction
X = X.reshape(-1, N_MFCC, MAX_LEN, 1)

# --- TRAIN-TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- HYPERPARAMETER GRID ---
learning_rates = [0.001]
dropout_rates = [0.3]
batch_sizes = [32]
lr_factors = [0.5]
early_patiences = [6]
ALPHA = 1e-4

best_val_acc = 0
best_model = None
best_params = {}

for lr in learning_rates:
    for dr in dropout_rates:
        for bs in batch_sizes:
            for factor in lr_factors:
                for patience in early_patiences:
                    print(f"\n🔍 Trying lr={lr}, dropout={dr}, batch={bs}, factor={factor}, patience={patience}")

                    # --- BUILD MODEL ---
                    def build_tuned_cnn(input_shape, num_classes, dropout, alpha):
                        model = Sequential()
                        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape, kernel_regularizer=l2(alpha)))
                        model.add(BatchNormalization())
                        model.add(MaxPooling2D((2, 2)))
                        model.add(Dropout(dropout))

                        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(alpha)))
                        model.add(BatchNormalization())
                        model.add(MaxPooling2D((2, 2)))
                        model.add(Dropout(dropout + 0.1))

                        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(alpha)))
                        model.add(BatchNormalization())
                        model.add(MaxPooling2D((2, 2)))
                        model.add(Dropout(dropout + 0.2))

                        model.add(Flatten())
                        model.add(Dense(256, activation='relu', kernel_regularizer=l2(alpha)))
                        model.add(Dropout(0.5))
                        model.add(Dense(num_classes, activation='softmax'))

                        model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                        return model

                    model = build_tuned_cnn(X_train.shape[1:], NUM_CLASSES, dropout=dr, alpha=ALPHA)

                    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
                    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=4, min_lr=1e-6, verbose=1)

                    history = model.fit(
                        X_train, y_train,
                        validation_split=0.1,
                        epochs=50,
                        batch_size=bs,
                        callbacks=[early_stop, reduce_lr],
                        verbose=1
                    )

                    val_acc = max(history.history['val_accuracy'])
                    print(f"✅ Validation Accuracy: {val_acc:.4f}")

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_model = model
                        best_params = {
                            'learning_rate': lr,
                            'dropout_rate': dr,
                            'batch_size': bs,
                            'factor': factor,
                            'patience': patience
                        }

# --- FINAL EVALUATION ---
test_loss, test_acc = best_model.evaluate(X_test, y_test)
print(f"\n🏆 Best Model Test Accuracy: {test_acc * 100:.2f}%")
print(f"🔧 Best Hyperparameters: {best_params}")

# --- SAVE MODEL ---
best_model.save("cnn_voice_model_best.keras")
print("✅ Best model saved to cnn_voice_model_best.keras")

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

# --- CONFUSION MATRIX AND CLASSIFICATION REPORT ---
y_pred = np.argmax(best_model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred, labels=range(NUM_CLASSES))
disp = ConfusionMatrixDisplay(cm, display_labels=COMMANDS)
disp.plot()
plt.savefig("confusion_matrix.png")
plt.show()

print("Label distribution:", Counter(map_label(label) for label in y_raw))
report = classification_report(y_test, y_pred, target_names=COMMANDS)
print("Classification Report:")
print(report)