# ============================================================
# 11_train_image_model.py
#
# PURPOSE:
#   Train an image classification model to predict animal
#   diseases from photos.
#
#   We use MobileNetV2 - a lightweight pretrained CNN.
#   "Pretrained" means it already knows how to detect shapes,
#   edges, textures from millions of images. We just teach it
#   the final step: which disease each image represents.
#
# APPROACH:
#   Phase 1 - Train only the top classification layer
#             (backbone is frozen / not updated)
#   Phase 2 - Unfreeze the top layers of the backbone and
#             fine-tune at a very low learning rate
#
# OUTPUT:
#   models/animal_disease_model.keras  <- saved best model
#   results/training_history.png
#   results/metrics.json
#
# HOW TO RUN:
#   cd "D:\Animal disease Prediction"
#   python anaconda_projects/11_train_image_model.py
# ============================================================


import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# SECTION 1: CONFIGURATION
# All important settings are here. Change these to experiment.
# ============================================================

# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Where the prepared images live (organized in train/val/test folders)
PREPARED_DIR = BASE_DIR / "prepared_dataset"

# CSVs with image paths + numeric disease labels
TRAIN_CSV = BASE_DIR / "dataset_split" / "labeled" / "train_labeled.csv"
VAL_CSV   = BASE_DIR / "dataset_split" / "labeled" / "validation_labeled.csv"

# Class mapping (so we know index 15 = Mange, etc.)
CLASS_MAPPING_FILE = BASE_DIR / "class_mapping" / "class_mapping.json"

# Where to save the trained model
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "animal_disease_model.keras"

# Where to save charts and results
RESULTS_DIR = BASE_DIR / "results"

# ---- Image settings ----
IMAGE_SIZE  = (224, 224)  # MobileNetV2 default input size
BATCH_SIZE  = 32          # Lower to 16 if you run out of memory

# ---- Training settings ----
PHASE1_EPOCHS   = 20     # Max epochs for Phase 1 (head training)
PHASE2_EPOCHS   = 20     # Max epochs for Phase 2 (fine-tuning)
PHASE1_LR       = 1e-3   # Learning rate for Phase 1
PHASE2_LR       = 1e-5   # Very small LR for fine-tuning
UNFREEZE_LAYERS = 30     # How many top backbone layers to unfreeze in Phase 2

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# SECTION 2: CREATE OUTPUT DIRECTORIES
# ============================================================

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ANIMAL DISEASE IMAGE MODEL TRAINING")
print("=" * 70)
print(f"\nBase directory   : {BASE_DIR}")
print(f"Prepared dataset : {PREPARED_DIR}")
print(f"Models output    : {MODELS_DIR}")
print(f"Results output   : {RESULTS_DIR}")


# ============================================================
# SECTION 3: LOAD CLASS MAPPING
# ============================================================

print("\n" + "=" * 70)
print("LOADING CLASS MAPPING")
print("=" * 70)

if not CLASS_MAPPING_FILE.exists():
    print(f"ERROR: class_mapping.json not found at:")
    print(CLASS_MAPPING_FILE)
    print("Please run: python anaconda_projects/08_create_class_mapping.py")
    raise SystemExit(1)

with open(CLASS_MAPPING_FILE, "r", encoding="utf-8") as f:
    class_mapping = json.load(f)

disease_classes  = class_mapping["disease_classes"]
index_to_disease = class_mapping["index_to_disease"]

NUM_CLASSES = len(disease_classes)

print(f"Total disease classes : {NUM_CLASSES}")
print(f"Animal classes        : {class_mapping['animal_classes']}")
print("\nDisease classes:")
for i, disease in enumerate(disease_classes):
    print(f"  {i:2d}: {disease}")


# ============================================================
# SECTION 4: LOAD CSV FILES
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASET CSV FILES")
print("=" * 70)

if not TRAIN_CSV.exists():
    print(f"ERROR: Training CSV not found: {TRAIN_CSV}")
    raise SystemExit(1)

if not VAL_CSV.exists():
    print(f"ERROR: Validation CSV not found: {VAL_CSV}")
    raise SystemExit(1)

train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

print(f"Training images   : {len(train_df)}")
print(f"Validation images : {len(val_df)}")


# ============================================================
# SECTION 5: BUILD FULL IMAGE PATHS
#
# Images in prepared_dataset are named like:
#   prepared_dataset/train/Cat/Dental Disease/IMG_00001.jpg
#
# Disease folders use spaces (e.g., "Dental Disease")
# CSV uses underscores (e.g., "Dental_Disease")
# We try both variants when resolving paths.
# ============================================================

print("\n" + "=" * 70)
print("RESOLVING IMAGE PATHS")
print("=" * 70)


def format_to_extension(fmt):
    """Convert image format string to file extension."""
    fmt = str(fmt).upper().strip()
    mapping = {
        "JPEG" : ".jpg",
        "JPG"  : ".jpg",
        "PNG"  : ".png",
        "WEBP" : ".webp",
        "GIF"  : ".gif",
        "BMP"  : ".bmp",
    }
    return mapping.get(fmt, ".jpg")


def resolve_image_path(row, split_name):
    """
    Build the full path to an image in the prepared_dataset folder.
    Tries both space and underscore variants of the disease folder name.
    """
    image_id = str(row["image_id"])
    animal   = str(row["animal"])
    disease  = str(row["disease"])
    fmt      = str(row["format"])

    ext = format_to_extension(fmt)

    # Try with spaces (original folder name) and underscores (CSV name)
    for disease_variant in [disease.replace("_", " "), disease]:
        path = (
            PREPARED_DIR
            / split_name
            / animal
            / disease_variant
            / f"{image_id}{ext}"
        )
        if path.exists():
            return str(path)

    return None


def build_path_column(df, split_name):
    """Add a full_path column to the dataframe."""
    df = df.copy()
    df["full_path"]  = df.apply(
        lambda row: resolve_image_path(row, split_name), axis=1
    )
    df["path_found"] = df["full_path"].notna()
    return df


print("Resolving training image paths...")
train_df = build_path_column(train_df, "train")

print("Resolving validation image paths...")
val_df   = build_path_column(val_df, "validation")

train_missing = (~train_df["path_found"]).sum()
val_missing   = (~val_df["path_found"]).sum()

print(f"\nTraining   - found: {train_df['path_found'].sum()} / missing: {train_missing}")
print(f"Validation - found: {val_df['path_found'].sum()} / missing: {val_missing}")

if train_missing > 0:
    print(f"\nWARNING: {train_missing} training images not found. First few:")
    print(train_df[~train_df["path_found"]][["image_id","animal","disease","format"]].head(5))

# Keep only found images
train_df = train_df[train_df["path_found"]].reset_index(drop=True)
val_df   = val_df[val_df["path_found"]].reset_index(drop=True)

print(f"\nFinal training images   : {len(train_df)}")
print(f"Final validation images : {len(val_df)}")

if len(train_df) < 100:
    print("ERROR: Too few training images found. Check prepared_dataset/ folder.")
    raise SystemExit(1)


# ============================================================
# SECTION 6: CLASS DISTRIBUTION CHECK
# ============================================================

print("\n" + "=" * 70)
print("DISEASE CLASS DISTRIBUTION (Training set)")
print("=" * 70)

disease_counts = train_df["disease"].value_counts()
print(disease_counts.to_string())

print(f"\nSmallest class : {disease_counts.min()} images ({disease_counts.idxmin()})")
print(f"Largest class  : {disease_counts.max()} images ({disease_counts.idxmax()})")
print(f"Imbalance ratio: {disease_counts.max() / disease_counts.min():.1f}x")


# ============================================================
# SECTION 7: COMPUTE CLASS WEIGHTS
# Higher weight = model pays more attention to that class.
# This compensates for class imbalance.
# ============================================================

labels_array   = train_df["disease_label"].values
unique_classes = np.unique(labels_array)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=unique_classes,
    y=labels_array
)

class_weight_dict = dict(zip(unique_classes, class_weights_array))

print("\n" + "=" * 70)
print("CLASS WEIGHTS (higher weight = more attention during training)")
print("=" * 70)
for class_idx, weight in sorted(class_weight_dict.items()):
    disease_name = index_to_disease.get(str(class_idx), f"Class_{class_idx}")
    print(f"  {class_idx:2d}: {disease_name:<40} weight={weight:.3f}")


# ============================================================
# SECTION 8: BUILD DATA AUGMENTATION
#
# Augmentation randomly transforms training images so the model
# sees different versions of each photo.
# This reduces overfitting (memorizing training images).
# We do NOT augment validation images.
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.10, 0.10),
    layers.RandomBrightness(0.10),
    layers.RandomContrast(0.10),
], name="data_augmentation")


# ============================================================
# SECTION 9: BUILD tf.data DATASET PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("BUILDING DATA PIPELINE")
print("=" * 70)


def load_and_preprocess_image(image_path, label):
    """
    Load image from disk, resize to 224x224, and apply
    MobileNetV2 preprocessing (scales pixels to [-1, 1]).
    """
    raw   = tf.io.read_file(image_path)
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = preprocess_input(image)   # IMPORTANT: required for MobileNetV2
    return image, label


def build_dataset(df, split, apply_augmentation=False):
    """Build a tf.data.Dataset from a dataframe."""
    paths  = df["full_path"].values
    labels = df["disease_label"].values.astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if apply_augmentation:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if apply_augmentation:
        dataset = dataset.map(
            lambda img, lbl: (data_augmentation(img, training=True), lbl),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    batches = len(df) // BATCH_SIZE + 1
    print(f"  {split}: {len(df)} images, {batches} batches/epoch")

    return dataset


print("Building training dataset (with augmentation)...")
train_dataset = build_dataset(train_df, "train", apply_augmentation=True)

print("Building validation dataset (no augmentation)...")
val_dataset   = build_dataset(val_df, "validation", apply_augmentation=False)


# ============================================================
# SECTION 10: BUILD THE MODEL
#
# Transfer Learning Architecture:
#
#   Input (224x224x3)
#     -> MobileNetV2 backbone (frozen in Phase 1)
#     -> GlobalAveragePooling2D
#     -> Dense(256, ReLU) + BatchNorm + Dropout(0.5)
#     -> Dense(128, ReLU) + Dropout(0.3)
#     -> Dense(28, Softmax)  <- 28 disease probabilities
# ============================================================

print("\n" + "=" * 70)
print("BUILDING MODEL")
print("=" * 70)


def build_model():
    """Build and return the full disease classification model."""

    # Load MobileNetV2 with pretrained ImageNet weights
    # include_top=False removes the original 1000-class output layer
    backbone = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = False  # Freeze for Phase 1

    print(f"  MobileNetV2 layers     : {len(backbone.layers)}")
    print(f"  Backbone frozen        : Yes (Phase 1)")

    # Build the full model
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="input_image")

    # training=False keeps BatchNorm inside MobileNetV2 in inference mode
    x = backbone(inputs, training=False)

    # Compress (7,7,1280) feature maps to a (1280,) vector
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)

    # Our custom classification layers
    x = layers.Dense(256, activation="relu",  name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(0.5, name="dropout_1")(x)

    x = layers.Dense(128, activation="relu",  name="dense_2")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)

    # Output: probability for each of the 28 disease classes
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="disease_predictions"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="AnimalDiseaseModel"
    )

    return model, backbone


model, backbone = build_model()

print(f"\n  Output classes         : {NUM_CLASSES}")

# Count trainable parameters
trainable_params = sum(
    tf.size(v).numpy() for v in model.trainable_variables
)
total_params = sum(
    tf.size(v).numpy() for v in model.variables
)
print(f"  Trainable parameters   : {trainable_params:,}")
print(f"  Total parameters       : {total_params:,}")


# ============================================================
# SECTION 11: PHASE 1 — Train Classification Head Only
# ============================================================

print("\n" + "=" * 70)
print("PHASE 1: TRAINING CLASSIFICATION HEAD (backbone frozen)")
print("=" * 70)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop_p1 = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint_p1 = ModelCheckpoint(
    filepath=str(MODEL_FILE),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

reduce_lr_p1 = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

print(f"\nLearning rate : {PHASE1_LR}")
print(f"Max epochs    : {PHASE1_EPOCHS}")
print(f"Early stop    : patience=5 epochs")
print("\nStarting Phase 1...\n")

history_p1 = model.fit(
    train_dataset,
    epochs=PHASE1_EPOCHS,
    validation_data=val_dataset,
    class_weight=class_weight_dict,
    callbacks=[early_stop_p1, checkpoint_p1, reduce_lr_p1],
    verbose=1
)

p1_best_val_acc  = max(history_p1.history["val_accuracy"])
p1_best_val_loss = min(history_p1.history["val_loss"])

print(f"\nPhase 1 Results:")
print(f"  Best val accuracy : {p1_best_val_acc*100:.2f}%")
print(f"  Best val loss     : {p1_best_val_loss:.4f}")


# ============================================================
# SECTION 12: PHASE 2 — Fine-tune Top Backbone Layers
#
# We unfreeze the last UNFREEZE_LAYERS layers of MobileNetV2
# and train them at a very small learning rate.
# This adapts the backbone to recognize disease-specific features.
# ============================================================

print("\n" + "=" * 70)
print(f"PHASE 2: FINE-TUNING (unfreezing last {UNFREEZE_LAYERS} backbone layers)")
print("=" * 70)

# Unfreeze the backbone
backbone.trainable = True

# Freeze all layers except the last UNFREEZE_LAYERS
total_layers = len(backbone.layers)
freeze_until = total_layers - UNFREEZE_LAYERS

for i, layer in enumerate(backbone.layers):
    layer.trainable = (i >= freeze_until)

trainable_count = sum(1 for l in backbone.layers if l.trainable)
print(f"  Total backbone layers     : {total_layers}")
print(f"  Frozen layers             : {freeze_until}")
print(f"  Unfrozen backbone layers  : {trainable_count}")

# Recompile with a much smaller learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE2_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop_p2 = EarlyStopping(
    monitor="val_accuracy",
    patience=7,
    restore_best_weights=True,
    verbose=1
)

checkpoint_p2 = ModelCheckpoint(
    filepath=str(MODEL_FILE),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

reduce_lr_p2 = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-8,
    verbose=1
)

print(f"\nLearning rate : {PHASE2_LR}")
print(f"Max epochs    : {PHASE2_EPOCHS}")
print(f"Early stop    : patience=7 epochs")
print("\nStarting Phase 2...\n")

history_p2 = model.fit(
    train_dataset,
    epochs=PHASE2_EPOCHS,
    validation_data=val_dataset,
    class_weight=class_weight_dict,
    callbacks=[early_stop_p2, checkpoint_p2, reduce_lr_p2],
    verbose=1
)

p2_best_val_acc  = max(history_p2.history["val_accuracy"])
p2_best_val_loss = min(history_p2.history["val_loss"])

print(f"\nPhase 2 Results:")
print(f"  Best val accuracy : {p2_best_val_acc*100:.2f}%")
print(f"  Best val loss     : {p2_best_val_loss:.4f}")


# ============================================================
# SECTION 13: SAVE TRAINING HISTORY PLOTS
# ============================================================

print("\n" + "=" * 70)
print("SAVING TRAINING PLOTS")
print("=" * 70)


def combine_histories(h1, h2, metric):
    return h1.history[metric] + h2.history[metric]


train_acc  = combine_histories(history_p1, history_p2, "accuracy")
val_acc    = combine_histories(history_p1, history_p2, "val_accuracy")
train_loss = combine_histories(history_p1, history_p2, "loss")
val_loss   = combine_histories(history_p1, history_p2, "val_loss")
epochs     = range(1, len(train_acc) + 1)
phase1_end = len(history_p1.history["accuracy"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy subplot
axes[0].plot(epochs, train_acc, "b-o", label="Train Accuracy",      markersize=3)
axes[0].plot(epochs, val_acc,   "r-o", label="Validation Accuracy",  markersize=3)
axes[0].axvline(x=phase1_end, color="gray", linestyle="--",
                label=f"Phase boundary (epoch {phase1_end})")
axes[0].set_title("Model Accuracy", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss subplot
axes[1].plot(epochs, train_loss, "b-o", label="Train Loss",      markersize=3)
axes[1].plot(epochs, val_loss,   "r-o", label="Validation Loss", markersize=3)
axes[1].axvline(x=phase1_end, color="gray", linestyle="--",
                label=f"Phase boundary (epoch {phase1_end})")
axes[1].set_title("Model Loss", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Animal Disease Model — Training History", fontsize=14, fontweight="bold")
plt.tight_layout()

history_plot = RESULTS_DIR / "training_history.png"
plt.savefig(history_plot, dpi=150, bbox_inches="tight")
plt.close()

print(f"  Saved: {history_plot}")


# ============================================================
# SECTION 14: SAVE METRICS JSON
# ============================================================

overall_best = max(p1_best_val_acc, p2_best_val_acc)

metrics = {
    "phase_1": {
        "epochs_run"           : len(history_p1.history["accuracy"]),
        "best_val_accuracy"    : round(float(p1_best_val_acc), 4),
        "best_val_loss"        : round(float(p1_best_val_loss), 4),
        "final_train_accuracy" : round(float(history_p1.history["accuracy"][-1]), 4),
    },
    "phase_2": {
        "epochs_run"           : len(history_p2.history["accuracy"]),
        "best_val_accuracy"    : round(float(p2_best_val_acc), 4),
        "best_val_loss"        : round(float(p2_best_val_loss), 4),
        "final_train_accuracy" : round(float(history_p2.history["accuracy"][-1]), 4),
    },
    "overall": {
        "best_val_accuracy" : round(float(overall_best), 4),
        "num_classes"       : NUM_CLASSES,
        "train_images"      : len(train_df),
        "val_images"        : len(val_df),
        "image_size"        : list(IMAGE_SIZE),
        "batch_size"        : BATCH_SIZE,
        "backbone"          : "MobileNetV2",
        "model_file"        : str(MODEL_FILE),
    }
}

metrics_file = RESULTS_DIR / "metrics.json"
with open(metrics_file, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print(f"  Saved: {metrics_file}")


# ============================================================
# SECTION 15: FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE — FINAL SUMMARY")
print("=" * 70)

print(f"\n  Model saved to      : {MODEL_FILE}")
print(f"  Phase 1 best val acc: {p1_best_val_acc*100:.2f}%")
print(f"  Phase 2 best val acc: {p2_best_val_acc*100:.2f}%")
print(f"  OVERALL BEST        : {overall_best*100:.2f}%")
print(f"\n  Results saved to    : {RESULTS_DIR}")

print()
if overall_best >= 0.70:
    print(f"  STATUS: GOOD — {overall_best*100:.1f}% accuracy is acceptable for a prototype.")
elif overall_best >= 0.50:
    print(f"  STATUS: FAIR — {overall_best*100:.1f}% accuracy — model learned but can improve.")
    print("  Consider: more epochs, different augmentation, or more data.")
else:
    print(f"  STATUS: LOW — {overall_best*100:.1f}% accuracy — check for issues.")
    print("  Check: image paths, class imbalance, or try fewer classes first.")

print(f"\n  NEXT STEP: Run python anaconda_projects/12_evaluate_image_model.py")
print("=" * 70)
