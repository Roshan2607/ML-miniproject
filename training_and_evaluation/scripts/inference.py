import os
import numpy as np
import torch

from src.config import Config
from src.model import Seq2SeqTTS
from src.preprocessing import (
    text_to_sequence,
    spectrogram_to_audio
)
from src.preprocessing import get_vocab_size


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = os.path.join(
    Config.BASE_DIR,
    "checkpoints",
      "seq2seq_augmented_best.pt"
)

OUTPUT_DIR = os.path.join(
    Config.BASE_DIR,
    "generated_audio"
)

OUTPUT_WAV = os.path.join(
    OUTPUT_DIR,
    "test_output.wav"
)

OUTPUT_SPECTROGRAM = os.path.join(
    OUTPUT_DIR,
    "generated_spectrogram.npy"
)

# Text used for TTS inference
INPUT_TEXT = "She had your dark suit in greasy wash water all year."


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 60)
print("TTS INFERENCE")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Input text: {INPUT_TEXT}")
print(f"Checkpoint: {CHECKPOINT_PATH}")


# ============================================================
# CHECK CHECKPOINT
# ============================================================

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"\nCheckpoint not found:\n{CHECKPOINT_PATH}\n\n"
        "Please train the model first using:\n"
        "python -m scripts.train"
    )


# ============================================================
# CREATE MODEL
# ============================================================

vocab_size = get_vocab_size()

model = Seq2SeqTTS(
    vocab_size=vocab_size,
    embed_dim=Config.embed_dim,
    hidden_dim=Config.hidden_dim,
    output_dim=Config.spec_bins
)

model = model.to(DEVICE)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

# Handle both normal state_dict checkpoints
# and checkpoints stored inside a dictionary.
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully.")


# ============================================================
# TEXT PREPROCESSING
# ============================================================

sequence = text_to_sequence(INPUT_TEXT)

text_tensor = torch.LongTensor(sequence).unsqueeze(0).to(DEVICE)

seq_lengths = torch.LongTensor(
    [len(sequence)]
).to(DEVICE)

print(f"Input sequence length: {len(sequence)}")


# ============================================================
# ENCODER
# ============================================================

with torch.no_grad():

    encoder_outputs, hidden, cell = model.encoder(
        text_tensor,
        seq_lengths
    )

print(
    f"Encoder output shape: "
    f"{tuple(encoder_outputs.shape)}"
)

print(
    f"Initial hidden shape: "
    f"{tuple(hidden.shape)}"
)

print(
    f"Initial cell shape: "
    f"{tuple(cell.shape)}"
)


# ============================================================
# GENERATE SPECTROGRAM
# ============================================================

# Number of spectrogram time frames.
#
# 300 is used here to match your previous inference setup.
TARGET_LENGTH = 300

with torch.no_grad():

    batch_size = 1

    # Mask for padded text tokens
    mask = text_tensor != 0

    # Initial decoder input
    input_step = torch.zeros(
        batch_size,
        1,
        Config.spec_bins,
        device=DEVICE
    )

    predictions = []

    for t in range(TARGET_LENGTH):

        prediction, hidden, cell, attention = model.decoder(
            input_step,
            hidden,
            cell,
            encoder_outputs,
            mask
        )

        predictions.append(
            prediction.unsqueeze(1)
        )

        # Autoregressive decoding:
        # feed the current prediction into the next step.
        input_step = prediction.unsqueeze(1)

    generated_spec = torch.cat(
        predictions,
        dim=1
    )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

generated_spec = (
    generated_spec
    .squeeze(0)
    .cpu()
    .numpy()
)


print(
    f"Generated spectrogram shape: "
    f"{generated_spec.shape}"
)


# ============================================================
# SAVE GENERATED SPECTROGRAM
# ============================================================

np.save(
    OUTPUT_SPECTROGRAM,
    generated_spec
)

print(
    f"Generated spectrogram saved → "
    f"{OUTPUT_SPECTROGRAM}"
)


# ============================================================
# SPECTROGRAM STATISTICS
# ============================================================

print(
    f"Spectrogram minimum: "
    f"{generated_spec.min():.6f}"
)

print(
    f"Spectrogram maximum: "
    f"{generated_spec.max():.6f}"
)

print(
    f"Spectrogram mean: "
    f"{generated_spec.mean():.6f}"
)


# ============================================================
# CONVERT SPECTROGRAM TO AUDIO
# ============================================================

print("Converting spectrogram to audio...")

audio = spectrogram_to_audio(
    generated_spec,
    n_fft=Config.n_fft,
    hop_length=Config.hop_length,
    win_length=Config.win_length,
    preemph=Config.preemphasis_k
)


# ============================================================
# NORMALIZE AUDIO
# ============================================================

# Prevent clipping when saving the WAV.
max_abs = np.max(np.abs(audio))

if max_abs > 0:
    audio = audio / max_abs


# ============================================================
# SAVE WAV
# ============================================================

import soundfile as sf

sf.write(
    OUTPUT_WAV,
    audio,
    Config.sample_rate
)


# ============================================================
# FINAL INFORMATION
# ============================================================

duration = len(audio) / Config.sample_rate

print("=" * 60)
print("INFERENCE COMPLETE")
print("=" * 60)

print(f"Output WAV: {OUTPUT_WAV}")

print(
    f"Output spectrogram: "
    f"{OUTPUT_SPECTROGRAM}"
)

print(
    f"Sample rate: "
    f"{Config.sample_rate} Hz"
)

print(
    f"Audio samples: "
    f"{len(audio)}"
)

print(
    f"Duration: "
    f"{duration:.2f} seconds"
)

print("=" * 60)