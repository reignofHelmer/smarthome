import os
import librosa
import soundfile as sf
import numpy as np
import random

# Paths
INPUT_DIR = r"C:\Users\MICHEAL\Desktop\C#D3\MY CODING FILES\Jarvis\Jarvis\audiocomms"
OUTPUT_DIR = r"C:\Users\MICHEAL\Desktop\C#D3\MY CODING FILES\Jarvis\Jarvis\audiocomms\audiocomms_augmented"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
SAMPLE_RATE = 16000
TARGET_SAMPLES_PER_COMMAND = 200  # Balance target (adjust as needed)

def add_white_noise(y, noise_level=0.003):
    noise = np.random.randn(len(y))
    return y + noise_level * noise

def change_pitch(y, sr, n_steps):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

def change_speed(y, rate):
    return librosa.effects.time_stretch(y=y, rate=rate)

def normalize_audio(y):
    return y / np.max(np.abs(y))

def mix_with_background(y, bg, snr_db=10):
    """Mix signal y with background bg at specified SNR (signal-to-noise ratio in dB)."""
    if len(bg) < len(y):
        bg = np.tile(bg, int(np.ceil(len(y)/len(bg))))
    bg = bg[:len(y)]
    y_power = np.mean(y ** 2)
    bg_power = np.mean(bg ** 2)
    desired_bg_power = y_power / (10 ** (snr_db / 10))
    bg = bg * np.sqrt(desired_bg_power / bg_power)
    return normalize_audio(y + bg)

# Optional: Load some background noise files if you have them
background_noises = []
BG_NOISE_DIR = None  # Set to folder path if you have real background .wav files
if BG_NOISE_DIR:
    for bg_file in os.listdir(BG_NOISE_DIR):
        if bg_file.endswith(".wav"):
            bg_y, _ = librosa.load(os.path.join(BG_NOISE_DIR, bg_file), sr=SAMPLE_RATE)
            background_noises.append(bg_y)

# Process each command folder
for command in os.listdir(INPUT_DIR):
    input_folder = os.path.join(INPUT_DIR, command)
    output_folder = os.path.join(OUTPUT_DIR, command)
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.isdir(input_folder):
        continue

    files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]
    n_existing = len(files)
    n_to_generate = max(0, TARGET_SAMPLES_PER_COMMAND - n_existing)
    augment_per_file = max(1, n_to_generate // n_existing) if n_existing else 0

    print(f"\nProcessing command: {command} (existing: {n_existing}, augment x{augment_per_file})")

    for file in files:
        file_path = os.path.join(input_folder, file)
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        y = normalize_audio(y)

        base_name = os.path.splitext(file)[0]
        sf.write(os.path.join(output_folder, f"{base_name}_orig.wav"), y, sr)

        for i in range(augment_per_file):
            # Randomly pick augmentation
            aug_choice = random.choice(['pitch', 'speed', 'noise', 'background'])

            if aug_choice == 'pitch':
                n_steps = random.uniform(1, 3) * random.choice([-1, 1])
                y_aug = change_pitch(y, sr, n_steps)
            elif aug_choice == 'speed':
                rate = random.uniform(0.9, 1.1)
                y_aug = change_speed(y, rate)
            elif aug_choice == 'noise':
                y_aug = add_white_noise(y, noise_level=random.uniform(0.003, 0.01))
            elif aug_choice == 'background' and background_noises:
                bg = random.choice(background_noises)
                y_aug = mix_with_background(y, bg)
            else:
                continue

            y_aug = normalize_audio(y_aug)
            aug_filename = f"{base_name}_aug{i+1}.wav"
            sf.write(os.path.join(output_folder, aug_filename), y_aug, sr)
            print(f"✅ Saved {aug_filename}")

print("\n🎉 Enhanced augmentation complete! Check the 'audiocomms_augmented' folder.")