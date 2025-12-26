import os

# --- SETTINGS ---
DATA_DIR = r"C:\Users\MICHEAL\Desktop\CD3\MY CODING FILES\Jarvis\Jarvis\audiocomms"

def count_wav_files(directory):
    # Count the number of .wav files in the given directory and subdirectories
    return len([f for f in os.listdir(directory) if f.endswith(".wav")])

def count_files_in_subdirectories(root_dir):
    total_files = 0
    for root, dirs, files in os.walk(root_dir):
        # Only count .wav files in the subfolders
        for file in files:
            if file.endswith(".wav"):
                total_files += 1
    
    return total_files

def main():
    total_files = count_files_in_subdirectories(DATA_DIR)
    print(f"Total .wav files in all subfolders: {total_files}")

if __name__ == "__main__":
    main()