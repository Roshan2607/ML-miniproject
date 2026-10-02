import os
import numpy as np
import soundfile as sf

from src.preprocessing import spectrogram_to_audio
from src.config import Config


spec_path = os.path.join(
    Config.PROCESSED_DATA_DIR,
    "LJ001-0005.npy"
)

output_dir = os.path.join(
    Config.BASE_DIR,
    "generated_audio"
)

os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir,
    "reconstruction_LJ001-0005.wav"
)


print("=" * 60)
print("REAL SPECTROGRAM RECONSTRUCTION")
print("=" * 60)

print("Spectrogram:", spec_path)

spectrogram = np.load(spec_path)

print("Spectrogram shape:", spectrogram.shape)

audio = spectrogram_to_audio(
    spectrogram,
    n_fft=Config.n_fft,
    hop_length=Config.hop_length,
    win_length=Config.win_length,
    preemph=Config.preemphasis_k
)

sf.write(
    output_path,
    audio,
    Config.sample_rate
)

print("Output:", output_path)
print("Sample rate:", Config.sample_rate)
print("Audio samples:", len(audio))
print("Duration:", len(audio) / Config.sample_rate, "seconds")

print("=" * 60)
print("RECONSTRUCTION COMPLETE")
print("=" * 60)