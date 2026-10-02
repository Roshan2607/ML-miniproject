import os
import json
import numpy as np

from src.config import Config


print("=" * 60)
print("TTS PROJECT EVALUATION")
print("=" * 60)


# ---------------------------------------------------------
# 1. Check processed dataset
# ---------------------------------------------------------

processed_dir = Config.PROCESSED_DATA_DIR

if os.path.exists(processed_dir):

    npy_files = [
        f for f in os.listdir(processed_dir)
        if f.endswith(".npy")
    ]

    print("\nProcessed dataset")
    print("-" * 60)
    print("Directory:", processed_dir)
    print("Number of spectrograms:", len(npy_files))

    if len(npy_files) > 0:

        sample_file = os.path.join(processed_dir, npy_files[0])

        sample = np.load(sample_file)

        print("Sample file:", npy_files[0])
        print("Sample spectrogram shape:", sample.shape)
        print("Minimum value:", float(sample.min()))
        print("Maximum value:", float(sample.max()))

else:

    print("\nProcessed dataset directory NOT FOUND.")


# ---------------------------------------------------------
# 2. Check checkpoints
# ---------------------------------------------------------

checkpoint_dir = os.path.join(
    Config.BASE_DIR,
    "checkpoints"
)

print("\n\nModel checkpoints")
print("-" * 60)

best_model = os.path.join(
    checkpoint_dir,
    "seq2seq_best.pt"
)

final_model = os.path.join(
    checkpoint_dir,
    "seq2seq_final.pt"
)

loss_file = os.path.join(
    checkpoint_dir,
    "loss_history.json"
)

print(
    "Best model:",
    "FOUND" if os.path.exists(best_model) else "NOT FOUND"
)

print(
    "Final model:",
    "FOUND" if os.path.exists(final_model) else "NOT FOUND"
)

print(
    "Loss history:",
    "FOUND" if os.path.exists(loss_file) else "NOT FOUND"
)


# ---------------------------------------------------------
# 3. Display training history
# ---------------------------------------------------------

if os.path.exists(loss_file):

    print("\n\nTraining history")
    print("-" * 60)

    try:

        with open(loss_file, "r") as f:
            history = json.load(f)

        print(json.dumps(history, indent=2))

    except Exception as e:

        print("Could not read loss history.")
        print("Error:", e)


# ---------------------------------------------------------
# 4. Check generated audio
# ---------------------------------------------------------

audio_dir = os.path.join(
    Config.BASE_DIR,
    "generated_audio"
)

print("\n\nGenerated outputs")
print("-" * 60)

if os.path.exists(audio_dir):

    wav_files = [
        f for f in os.listdir(audio_dir)
        if f.lower().endswith(".wav")
    ]

    print("Generated WAV files:")

    if wav_files:

        for f in wav_files:
            print("  -", f)

    else:

        print("  No WAV files found.")

else:

    print("Generated audio directory does not exist.")


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)