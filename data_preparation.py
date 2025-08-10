# Step 1: Import Libraries
import os
import shutil
import pandas as pd
from tqdm import tqdm
from pydub import AudioSegment

# Step 2: Define paths
MAPPING_FILE = "LibriSpeech/SPEAKERS.TXT"  # Replace with actual path
SOURCE_DIR = "LibriSpeech/dev-clean"  # Folder containing speaker-id subfolders
OUTPUT_DIR = "LibriSpeech"  # Will contain 'male/' and 'female/'

os.makedirs(os.path.join(OUTPUT_DIR, "voice_dataset"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "voice_dataset/male"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "voice_dataset/female"), exist_ok=True)

# Step 3: Parse mapping file to get speaker-gender dictionary
speaker_gender = {}
with open(MAPPING_FILE, "r") as f:
    for line in f:
        if line.strip().startswith(";") or line.strip() == "":
            continue  # skip comments and empty lines
        parts = [p.strip() for p in line.strip().split("|")]
        if len(parts) >= 2:
            speaker_id = parts[0]
            gender = parts[1].lower()  # 'f' or 'm'
            if gender in ("f", "m"):
                speaker_gender[speaker_id] = "female" if gender == "f" else "male"

print(speaker_gender)

# Step 4: Prepare to collect 250 samples per gender from as many speakers as possible
samples_per_gender = 1000
max_duration_ms = 60000  # 1 minute in milliseconds
selected_counts = {"male": 0, "female": 0}
speakers_seen = {"male": set(), "female": set()}

# Step 5: Walk and extract clipped WAV samples
for speaker_id in tqdm(sorted(os.listdir(SOURCE_DIR)), desc="Processing speakers"):
    if speaker_id not in speaker_gender:
        continue
    gender_label = speaker_gender[speaker_id]
    if selected_counts[gender_label] >= samples_per_gender:
        continue
    speaker_path = os.path.join(SOURCE_DIR, speaker_id)
    for root, _, files in os.walk(speaker_path):
        for fname in sorted(files):
            if (
                fname.endswith(".flac")
                and selected_counts[gender_label] < samples_per_gender
            ):
                src_path = os.path.join(root, fname)
                try:
                    # Load and clip
                    audio = AudioSegment.from_file(src_path, format="flac")
                    clipped = audio[:max_duration_ms]  # Clip to 1 min
                    # Save as WAV
                    dst_folder = os.path.join(OUTPUT_DIR, "voice_dataset", gender_label)
                    dst_filename = f"{speaker_id}_{fname.replace('.flac', '.wav')}"
                    dst_path = os.path.join(dst_folder, dst_filename)
                    clipped.export(dst_path, format="wav")
                    selected_counts[gender_label] += 1
                    speakers_seen[gender_label].add(speaker_id)
                    if selected_counts[gender_label] >= samples_per_gender:
                        break
                except Exception as e:
                    print(f"Error processing {src_path}: {e}")
        if selected_counts[gender_label] >= samples_per_gender:
            break

print("\nDataset creation completed:")
print(
    f" - Male samples: {selected_counts['male']} from {len(speakers_seen['male'])} speakers"
)
print(
    f" - Female samples: {selected_counts['female']} from {len(speakers_seen['female'])} speakers"
)
