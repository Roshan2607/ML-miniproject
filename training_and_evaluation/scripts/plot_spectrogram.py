import os
import numpy as np
import matplotlib.pyplot as plt

from src.config import Config


print("=" * 60)
print("REAL SPECTROGRAM VISUALIZATION")
print("=" * 60)


# ---------------------------------------------------------
# Select one preprocessed sample
# ---------------------------------------------------------

sample_name = "LJ001-0005.npy"

spectrogram_path = os.path.join(
    Config.PROCESSED_DATA_DIR,
    sample_name
)


# ---------------------------------------------------------
# Check file
# ---------------------------------------------------------

if not os.path.exists(spectrogram_path):

    raise FileNotFoundError(
        f"\nSpectrogram not found:\n{spectrogram_path}"
    )


# ---------------------------------------------------------
# Load spectrogram
# ---------------------------------------------------------

spectrogram = np.load(spectrogram_path)


print("\nSpectrogram:")
print(spectrogram_path)

print("\nShape:")
print(spectrogram.shape)

print("\nMinimum value:")
print(float(spectrogram.min()))

print("\nMaximum value:")
print(float(spectrogram.max()))


# ---------------------------------------------------------
# Create output directory
# ---------------------------------------------------------

output_dir = os.path.join(
    Config.BASE_DIR,
    "generated_audio"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# ---------------------------------------------------------
# Plot spectrogram
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.imshow(
    spectrogram.T,
    aspect="auto",
    origin="lower"
)

plt.xlabel("Time Frames")

plt.ylabel("Frequency Bins")

plt.title(
    "Preprocessed Speech Spectrogram - LJ001-0005"
)

plt.colorbar(
    label="Log Magnitude"
)

plt.tight_layout()


# ---------------------------------------------------------
# Save image
# ---------------------------------------------------------

output_path = os.path.join(
    output_dir,
    "real_spectrogram.png"
)

plt.savefig(
    output_path,
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# Done
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SPECTROGRAM PLOT CREATED")
print("=" * 60)

print(
    "Output:",
    output_path
)

print("=" * 60)