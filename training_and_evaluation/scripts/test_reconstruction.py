import os
import torch
import soundfile as sf

from src.dataset import LJSpeechDataset
from src.preprocessing import spectrogram_to_audio
from src.config import Config


# ============================================================
# PATHS
# ============================================================

METADATA_PATH = os.path.join(
    Config.RAW_DATA_DIR,
    "metadata.csv"
)

PROCESSED_DIR = Config.PROCESSED_DATA_DIR

OUTPUT_DIR = os.path.join(
    Config.BASE_DIR,
    "generated_audio"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_WAV = os.path.join(
    OUTPUT_DIR,
    "real_sample.wav"
)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("REAL SPECTROGRAM RECONSTRUCTION TEST")
print("=" * 60)

print(f"Metadata: {METADATA_PATH}")
print(f"Processed data: {PROCESSED_DIR}")


# ============================================================
# CHECK PATHS
# ============================================================

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(
        f"Metadata file not found:\n{METADATA_PATH}"
    )

if not os.path.exists(PROCESSED_DIR):
    raise FileNotFoundError(
        f"Processed directory not found:\n{PROCESSED_DIR}"
    )


# ============================================================
# LOAD DATASET
# ============================================================

dataset = LJSpeechDataset(
    METADATA_PATH,
    PROCESSED_DIR
)

print(f"Dataset size: {len(dataset)}")


# ============================================================
# GET FIRST SAMPLE
# ============================================================

sequence, spectrogram = dataset[0]

print(f"Text sequence shape: {sequence.shape}")
print(f"Spectrogram shape: {spectrogram.shape}")


# ============================================================
# CONVERT TO NUMPY
# ============================================================

if torch.is_tensor(spectrogram):
    spectrogram = spectrogram.cpu().numpy()


# ============================================================
# SPECTROGRAM → AUDIO
# ============================================================

print("Converting real spectrogram to audio...")

audio = spectrogram_to_audio(
    spectrogram
)


# ============================================================
# NORMALIZE
# ============================================================

max_amplitude = abs(audio).max()

if max_amplitude > 0:
    audio = audio / max_amplitude


# ============================================================
# SAVE WAV
# ============================================================

sf.write(
    OUTPUT_WAV,
    audio,
    Config.sample_rate
)


# ============================================================
# RESULT
# ============================================================

duration = len(audio) / Config.sample_rate

print("=" * 60)
print("RECONSTRUCTION COMPLETE")
print("=" * 60)

print(f"Output WAV: {OUTPUT_WAV}")
print(f"Sample rate: {Config.sample_rate} Hz")
print(f"Audio samples: {len(audio)}")
print(f"Duration: {duration:.2f} seconds")
print("=" * 60)