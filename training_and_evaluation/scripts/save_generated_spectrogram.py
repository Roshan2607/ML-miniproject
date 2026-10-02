import os
import numpy as np
import matplotlib.pyplot as plt

from src.config import Config


print("=" * 60)
print("GENERATED SPECTROGRAM VISUALIZATION")
print("=" * 60)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

output_dir = os.path.join(
    Config.BASE_DIR,
    "generated_audio"
)

spectrogram_path = os.path.join(
    output_dir,
    "generated_spectrogram.npy"
)


# ---------------------------------------------------------
# Check whether generated spectrogram exists
# ---------------------------------------------------------

if not os.path.exists(spectrogram_path):

    print("\nGenerated spectrogram not found:")
    print(spectrogram_path)

    print("\nYour inference.py must save the predicted")
    print("spectrogram as generated_spectrogram.npy")

    raise SystemExit


# ---------------------------------------------------------
# Load generated spectrogram
# ---------------------------------------------------------

spectrogram = np.load(
    spectrogram_path
)

print("\nGenerated spectrogram shape:")
print(spectrogram.shape)

print("\nMinimum value:")
print(float(spectrogram.min()))

print("\nMaximum value:")
print(float(spectrogram.max()))


# ---------------------------------------------------------
# Plot
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
    "Generated Speech Spectrogram"
)

plt.colorbar(
    label="Predicted Magnitude"
)

plt.tight_layout()


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output_path = os.path.join(
    output_dir,
    "generated_spectrogram.png"
)

plt.savefig(
    output_path,
    dpi=300
)

plt.close()


print("\n" + "=" * 60)
print("GENERATED SPECTROGRAM SAVED")
print("=" * 60)

print("Output:")
print(output_path)

print("=" * 60)