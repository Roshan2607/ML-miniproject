import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.config import Config
from src.dataset import LJSpeechDataset, collate_fn
from src.model import Seq2SeqTTS
from src.preprocessing import get_vocab_size


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

# ------------------------------------------------------------
# CPU EXPERIMENT SETTINGS
# ------------------------------------------------------------

MAX_SAMPLES = 100

BATCH_SIZE = 2

NUM_EPOCHS = 3

LEARNING_RATE = 1e-4

TRAIN_RATIO = 0.8

TEACHER_FORCING_RATIO = 0.5


# ------------------------------------------------------------
# AUGMENTATION SETTINGS
# ------------------------------------------------------------

USE_AUGMENTATION = True

# Small random Gaussian noise
NOISE_STD = 0.015

# Maximum percentage of time frames that can be masked
TIME_MASK_RATIO = 0.08

# Maximum percentage of frequency bins that can be masked
FREQ_MASK_RATIO = 0.04

# Random gain variation
GAIN_MIN = 0.90
GAIN_MAX = 1.10


# ============================================================
# CHECKPOINT DIRECTORY
# ============================================================

CHECKPOINT_DIR = os.path.join(
    Config.BASE_DIR,
    "checkpoints"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("=" * 60)
print("TTS TRAINING - AUGMENTED")
print("=" * 60)

print(f"Device: {device}")
print(f"Maximum samples: {MAX_SAMPLES}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")

print("\nAugmentation:")
print(f"  Enabled: {USE_AUGMENTATION}")
print(f"  Noise std: {NOISE_STD}")
print(f"  Time mask ratio: {TIME_MASK_RATIO}")
print(f"  Frequency mask ratio: {FREQ_MASK_RATIO}")
print(f"  Gain range: {GAIN_MIN} - {GAIN_MAX}")

print("=" * 60)


# ============================================================
# SPECTROGRAM AUGMENTATION
# ============================================================

def augment_spectrogram(
    spectrogram,
    noise_std=NOISE_STD,
    time_mask_ratio=TIME_MASK_RATIO,
    freq_mask_ratio=FREQ_MASK_RATIO,
    gain_min=GAIN_MIN,
    gain_max=GAIN_MAX
):
    """
    Apply lightweight spectrogram-domain augmentation.

    Input:
        spectrogram:
            Tensor of shape [batch, time, frequency]

    Output:
        Augmented spectrogram with the same shape.

    Augmentations:
        1. Gaussian noise
        2. Random time masking
        3. Random frequency masking
        4. Random gain scaling
    """

    augmented = spectrogram.clone()

    batch_size, time_steps, freq_bins = augmented.shape

    # --------------------------------------------------------
    # 1. RANDOM GAIN
    # --------------------------------------------------------

    gain = torch.empty(
        batch_size,
        1,
        1,
        device=augmented.device
    ).uniform_(
        gain_min,
        gain_max
    )

    augmented = augmented * gain


    # --------------------------------------------------------
    # 2. GAUSSIAN NOISE
    # --------------------------------------------------------

    if noise_std > 0:

        noise = torch.randn_like(augmented) * noise_std

        augmented = augmented + noise


    # --------------------------------------------------------
    # 3. TIME MASKING
    # --------------------------------------------------------

    max_time_mask = max(
        1,
        int(time_steps * time_mask_ratio)
    )

    for b in range(batch_size):

        if time_steps > 1:

            mask_length = random.randint(
                0,
                max_time_mask
            )

            if mask_length > 0:

                start = random.randint(
                    0,
                    max(
                        0,
                        time_steps - mask_length
                    )
                )

                augmented[
                    b,
                    start:start + mask_length,
                    :
                ] = 0.0


    # --------------------------------------------------------
    # 4. FREQUENCY MASKING
    # --------------------------------------------------------

    max_freq_mask = max(
        1,
        int(freq_bins * freq_mask_ratio)
    )

    for b in range(batch_size):

        if freq_bins > 1:

            mask_length = random.randint(
                0,
                max_freq_mask
            )

            if mask_length > 0:

                start = random.randint(
                    0,
                    max(
                        0,
                        freq_bins - mask_length
                    )
                )

                augmented[
                    b,
                    :,
                    start:start + mask_length
                ] = 0.0


    return augmented


# ============================================================
# DATASET
# ============================================================

metadata_path = os.path.join(
    Config.RAW_DATA_DIR,
    "metadata.csv"
)

processed_dir = Config.PROCESSED_DATA_DIR


dataset = LJSpeechDataset(
    metadata_path=metadata_path,
    processed_dir=processed_dir
)

print(f"Full dataset size: {len(dataset)}")


# ------------------------------------------------------------
# Use only MAX_SAMPLES
# ------------------------------------------------------------

if len(dataset) > MAX_SAMPLES:

    indices = list(range(MAX_SAMPLES))

    dataset = torch.utils.data.Subset(
        dataset,
        indices
    )


print(f"Using dataset size: {len(dataset)}")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_size = int(
    TRAIN_RATIO * len(dataset)
)

val_size = len(dataset) - train_size


train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)


print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

vocab_size = get_vocab_size()


model = Seq2SeqTTS(
    vocab_size=vocab_size,
    embed_dim=Config.embed_dim,
    hidden_dim=Config.hidden_dim,
    output_dim=Config.spec_bins
).to(device)


print("\nModel created.")

print(f"Vocabulary size: {vocab_size}")
print(f"Embedding dimension: {Config.embed_dim}")
print(f"Hidden dimension: {Config.hidden_dim}")
print(f"Spectrogram bins: {Config.spec_bins}")


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# MASKED MSE LOSS
# ============================================================

def masked_mse_loss(
    prediction,
    target,
    lengths
):
    """
    Calculate MSE only over real spectrogram frames.

    prediction:
        [batch, time, spectrogram_bins]

    target:
        [batch, time, spectrogram_bins]

    lengths:
        [batch]
    """

    batch_size, max_time, _ = target.shape

    time_indices = torch.arange(
        max_time,
        device=target.device
    ).unsqueeze(0)

    mask = (
        time_indices
        < lengths.unsqueeze(1)
    )

    # --------------------------------------------------------
    # MSE for every time frame
    # --------------------------------------------------------

    frame_loss = torch.mean(
        (prediction - target) ** 2,
        dim=2
    )

    # --------------------------------------------------------
    # Keep only real frames
    # --------------------------------------------------------

    frame_loss = (
        frame_loss
        * mask.float()
    )

    # --------------------------------------------------------
    # Avoid division by zero
    # --------------------------------------------------------

    denominator = (
        mask.float()
        .sum()
        .clamp(min=1.0)
    )

    return (
        frame_loss.sum()
        / denominator
    )


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer
):

    model.train()

    total_loss = 0.0

    for batch_idx, batch in enumerate(loader):

        # ----------------------------------------------------
        # Load batch
        # ----------------------------------------------------

        (
            text,
            seq_lengths,
            target_spec,
            spec_lengths
        ) = batch


        # ----------------------------------------------------
        # Move to device
        # ----------------------------------------------------

        text = text.to(device)

        seq_lengths = seq_lengths.to(device)

        target_spec = target_spec.to(device)

        spec_lengths = spec_lengths.to(device)


        # ----------------------------------------------------
        # AUGMENTATION
        #
        # Only training data is augmented.
        # Validation remains completely unchanged.
        # ----------------------------------------------------

        if USE_AUGMENTATION:

            target_spec = augment_spectrogram(
                target_spec
            )


        # ----------------------------------------------------
        # Clear previous gradients
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        prediction = model(
            text=text,
            seq_lengths=seq_lengths,
            target_spectrogram=target_spec,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO
        )


        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = masked_mse_loss(
            prediction,
            target_spec,
            spec_lengths
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Prevent exploding gradients
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        # ----------------------------------------------------
        # Update weights
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # Store loss
        # ----------------------------------------------------

        total_loss += loss.item()


        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        print(
            f"    Batch {batch_idx + 1}/{len(loader)} "
            f"| Loss: {loss.item():.6f}"
        )


    return total_loss / len(loader)


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate(
    model,
    loader
):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for batch in loader:

            (
                text,
                seq_lengths,
                target_spec,
                spec_lengths
            ) = batch


            # ------------------------------------------------
            # Move to device
            # ------------------------------------------------

            text = text.to(device)

            seq_lengths = seq_lengths.to(device)

            target_spec = target_spec.to(device)

            spec_lengths = spec_lengths.to(device)


            # ------------------------------------------------
            # IMPORTANT:
            # NO augmentation during validation.
            # ------------------------------------------------

            prediction = model(
                text=text,
                seq_lengths=seq_lengths,
                target_spectrogram=target_spec,
                teacher_forcing_ratio=0.0
            )


            # ------------------------------------------------
            # Validation loss
            # ------------------------------------------------

            loss = masked_mse_loss(
                prediction,
                target_spec,
                spec_lengths
            )


            total_loss += loss.item()


    return total_loss / len(loader)


# ============================================================
# TRAINING LOOP
# ============================================================

train_losses = []

val_losses = []

best_val_loss = float("inf")


print("\nStarting training...\n")


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    print("=" * 60)

    print(
        f"EPOCH {epoch}/{NUM_EPOCHS}"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_loss = validate(
        model,
        val_loader
    )


    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"\nEpoch {epoch} results:"
    )

    print(
        f"Training Loss   : {train_loss:.6f}"
    )

    print(
        f"Validation Loss : {val_loss:.6f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_path = os.path.join(
            CHECKPOINT_DIR,
            "seq2seq_augmented_best.pt"
        )


        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss,

                "vocab_size":
                    vocab_size,

                "augmentation":
                    True
            },

            best_path
        )


        print(
            f"Saved best augmented model → {best_path}"
        )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_path = os.path.join(
    CHECKPOINT_DIR,
    "seq2seq_augmented_final.pt"
)


torch.save(
    {
        "epoch": NUM_EPOCHS,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "train_loss":
            train_losses[-1],

        "val_loss":
            val_losses[-1],

        "vocab_size":
            vocab_size,

        "augmentation":
            True
    },

    final_path
)


# ============================================================
# SAVE LOSS HISTORY
# ============================================================

loss_path = os.path.join(
    CHECKPOINT_DIR,
    "augmented_loss_history.json"
)


with open(
    loss_path,
    "w"
) as f:

    json.dump(
        {
            "train_loss":
                train_losses,

            "validation_loss":
                val_losses,

            "augmentation":
                True,

            "max_samples":
                MAX_SAMPLES,

            "batch_size":
                BATCH_SIZE,

            "epochs":
                NUM_EPOCHS,

            "learning_rate":
                LEARNING_RATE,

            "noise_std":
                NOISE_STD,

            "time_mask_ratio":
                TIME_MASK_RATIO,

            "freq_mask_ratio":
                FREQ_MASK_RATIO,

            "gain_min":
                GAIN_MIN,

            "gain_max":
                GAIN_MAX
        },

        f,

        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)

print("AUGMENTED TRAINING COMPLETE")

print("=" * 60)

print(
    f"Best model : {best_path}"
)

print(
    f"Final model: {final_path}"
)

print(
    f"Loss file  : {loss_path}"
)

print("\nAugmentation was applied ONLY during training.")
print("Validation data remained unchanged.")