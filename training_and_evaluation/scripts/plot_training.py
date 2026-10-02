import os
import json
import matplotlib.pyplot as plt

from src.config import Config


print("=" * 60)
print("TRAINING LOSS PLOT")
print("=" * 60)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

checkpoint_dir = os.path.join(
    Config.BASE_DIR,
    "checkpoints"
)

loss_file = os.path.join(
    checkpoint_dir,
    "loss_history.json"
)

output_dir = os.path.join(
    Config.BASE_DIR,
    "generated_audio"
)

os.makedirs(output_dir, exist_ok=True)


# ---------------------------------------------------------
# Check loss file
# ---------------------------------------------------------

if not os.path.exists(loss_file):

    raise FileNotFoundError(
        f"Loss history not found:\n{loss_file}"
    )


# ---------------------------------------------------------
# Load loss history
# ---------------------------------------------------------

with open(loss_file, "r") as f:
    history = json.load(f)


print("\nLoss history loaded.")
print("Data type:", type(history).__name__)


# ---------------------------------------------------------
# Extract losses
# ---------------------------------------------------------

train_losses = []
val_losses = []


if isinstance(history, dict):

    print("Dictionary keys:", list(history.keys()))

    # Possible key names used by different train.py versions

    possible_train_keys = [
        "train_loss",
        "training_loss",
        "train_losses",
        "training_losses"
    ]

    possible_val_keys = [
        "val_loss",
        "validation_loss",
        "val_losses",
        "validation_losses"
    ]

    for key in possible_train_keys:

        if key in history:

            train_losses = history[key]
            break

    for key in possible_val_keys:

        if key in history:

            val_losses = history[key]
            break


elif isinstance(history, list):

    for item in history:

        if isinstance(item, dict):

            train_value = None
            val_value = None

            for key in [
                "train_loss",
                "training_loss"
            ]:

                if key in item:
                    train_value = item[key]
                    break

            for key in [
                "val_loss",
                "validation_loss"
            ]:

                if key in item:
                    val_value = item[key]
                    break

            if train_value is not None:
                train_losses.append(train_value)

            if val_value is not None:
                val_losses.append(val_value)


# ---------------------------------------------------------
# Convert to lists
# ---------------------------------------------------------

if train_losses is None:
    train_losses = []

if val_losses is None:
    val_losses = []


train_losses = list(train_losses)
val_losses = list(val_losses)


print("\nTraining losses:", train_losses)
print("Validation losses:", val_losses)


# ---------------------------------------------------------
# Make sure both have data
# ---------------------------------------------------------

if len(train_losses) == 0:

    raise ValueError(
        "\nNo training-loss values were found in "
        "loss_history.json.\n\n"
        "Open checkpoints/loss_history.json and check "
        "its contents."
    )

if len(val_losses) == 0:

    print(
        "\nWARNING: No validation-loss values were found."
    )

    print(
        "A training-only graph will be generated."
    )


# ---------------------------------------------------------
# Determine number of epochs
# ---------------------------------------------------------

if len(val_losses) > 0:

    num_epochs = min(
        len(train_losses),
        len(val_losses)
    )

else:

    num_epochs = len(train_losses)


train_losses = train_losses[:num_epochs]

if len(val_losses) > 0:
    val_losses = val_losses[:num_epochs]


epochs = list(
    range(1, num_epochs + 1)
)


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))


plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Training Loss"
)


if len(val_losses) > 0:

    plt.plot(
        epochs,
        val_losses,
        marker="o",
        label="Validation Loss"
    )


plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss"
)

plt.xticks(epochs)

plt.legend()

plt.grid(True)

plt.tight_layout()


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output_path = os.path.join(
    output_dir,
    "training_validation_loss.png"
)


plt.savefig(
    output_path,
    dpi=300
)


plt.close()


print("\n" + "=" * 60)
print("PLOT CREATED SUCCESSFULLY")
print("=" * 60)

print(
    "Epochs plotted:",
    num_epochs
)

print(
    "Output:",
    output_path
)

print("=" * 60)