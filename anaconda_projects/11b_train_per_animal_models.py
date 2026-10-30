# ============================================================
# 11b_train_per_animal_models.py  (v2 — with augmented data)
#
# PURPOSE:
#   Train one separate image classification model per animal.
#
#   v2 Change: Training images are now loaded by SCANNING
#   the prepared_dataset/train/ folders directly.
#   This picks up BOTH original AND augmented (_aug_) images.
#
#   Validation images still come from the CSV (no augmentation
#   applied to validation — this is intentional and correct).
#
# HOW TO RUN:
#   cd "D:\Animal disease Prediction"
#   python anaconda_projects/11b_train_per_animal_models.py
#
# TIME: Roughly 20-60 minutes per animal on CPU
# ============================================================

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR      = Path(__file__).resolve().parent.parent
PREPARED_DIR  = BASE_DIR / "prepared_dataset"
LABELED_DIR   = BASE_DIR / "dataset_split" / "labeled"
MODELS_DIR    = BASE_DIR / "models"
RESULTS_DIR   = BASE_DIR / "results"

VAL_CSV = LABELED_DIR / "validation_labeled.csv"

IMAGE_SIZE      = (224, 224)
BATCH_SIZE      = 32     # Lower to 16 if you get a memory error

PHASE1_EPOCHS   = 25    # More epochs since we have more data now
PHASE2_EPOCHS   = 20
PHASE1_LR       = 1e-3
PHASE2_LR       = 1e-5
UNFREEZE_LAYERS = 30

# Valid image file extensions
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FUNCTION: Scan a folder to build a dataframe of images
#
# This scans prepared_dataset/train/[animal]/[disease]/
# and returns a DataFrame with columns:
#   full_path, animal, disease, local_label
#
# It picks up EVERY image file — original + augmented.
# ============================================================

def scan_train_folder(animal_name, disease_list):
    """
    Scan prepared_dataset/train/[animal]/ and collect all image files.
    Only includes diseases in disease_list (so we can filter out
    leftover folders like 'mastitis normal').

    Returns a DataFrame with columns:
      full_path, animal, disease, local_label
    """
    train_animal_dir = PREPARED_DIR / "train" / animal_name
    rows = []

    for disease_name in disease_list:
        # Try both the exact name and with underscores replaced by spaces
        for dv in [disease_name, disease_name.replace("_", " ")]:
            disease_dir = train_animal_dir / dv
            if disease_dir.exists():
                for img_file in disease_dir.iterdir():
                    if img_file.suffix.lower() in IMG_EXTENSIONS:
                        rows.append({
                            "full_path": str(img_file),
                            "animal":    animal_name,
                            "disease":   disease_name,
                        })
                break  # found the folder, no need to try alternate name

    df = pd.DataFrame(rows)
    return df


# ============================================================
# FUNCTION: Resolve validation image paths from CSV
#
# Validation uses original images only (no augmentation).
# ============================================================

def format_to_ext(fmt):
    return {"JPEG": ".jpg", "JPG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(
        str(fmt).upper().strip(), ".jpg"
    )


def resolve_val_path(row):
    """Find the validation image in prepared_dataset/validation/"""
    image_id = str(row["image_id"])
    animal   = str(row["animal"])
    disease  = str(row["disease"])
    ext      = format_to_ext(row["format"])
    for dv in [disease.replace("_", " "), disease]:
        p = PREPARED_DIR / "validation" / animal / dv / f"{image_id}{ext}"
        if p.exists():
            return str(p)
    return None


# ============================================================
# DATA AUGMENTATION (applied online during training)
# Gentler than before — we already have more data now.
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),       # ±8 degrees (gentle)
    layers.RandomZoom(0.08),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomBrightness(0.08),
], name="data_augmentation")


def load_image(path, label):
    """Load one image from disk, resize, and normalize for MobileNetV2."""
    raw   = tf.io.read_file(path)
    img   = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img   = tf.image.resize(img, IMAGE_SIZE)
    img   = preprocess_input(img)   # Scales pixels to [-1, 1]
    return img, label


def make_dataset(df, apply_augmentation=False):
    """Build a tf.data pipeline from a DataFrame."""
    paths  = df["full_path"].values
    labels = df["local_label"].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if apply_augmentation:
        ds = ds.shuffle(len(df), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if apply_augmentation:
        ds = ds.map(
            lambda img, lbl: (data_augmentation(img, training=True), lbl),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(num_classes):
    """MobileNetV2 backbone + custom classification head."""
    backbone = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = False  # Freeze for Phase 1

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model, backbone


# ============================================================
# TRAIN ONE ANIMAL MODEL
# ============================================================

def train_animal_model(animal_name, val_all):
    """
    Train a disease classifier for one animal.
    Training data: scanned from prepared_dataset/train/ (includes augmented)
    Validation data: from validation CSV (original only)
    """
    print("\n" + "=" * 70)
    print(f"  TRAINING MODEL FOR: {animal_name.upper()}")
    print("=" * 70)

    # --- Get validation data from CSV (original images only) ---
    val_animal = val_all[val_all["animal"] == animal_name].copy()
    val_animal["full_path"]  = val_animal.apply(resolve_val_path, axis=1)
    val_animal["path_found"] = val_animal["full_path"].notna()
    val_df = val_animal[val_animal["path_found"]].reset_index(drop=True)

    # --- Determine the disease list from the validation data ---
    # This gives us the authoritative list of valid diseases for this animal
    diseases = sorted(val_df["disease"].unique())
    num_classes = len(diseases)

    if num_classes == 0:
        print(f"  SKIP: No validation data found for {animal_name}.")
        return None, None

    disease_to_local = {d: i for i, d in enumerate(diseases)}
    local_to_disease = {i: d for d, i in disease_to_local.items()}

    # --- Scan training folder for ALL images (original + augmented) ---
    train_df = scan_train_folder(animal_name, diseases)

    if len(train_df) < 20:
        print(f"  SKIP: Too few training images for {animal_name}.")
        return None, None

    # Add local numeric label
    train_df["local_label"] = train_df["disease"].map(disease_to_local)
    val_df["local_label"]   = val_df["disease"].map(disease_to_local)
    val_df = val_df[val_df["local_label"].notna()].copy()
    val_df["local_label"] = val_df["local_label"].astype(int)

    print(f"\n  Training images   : {len(train_df)}  (includes augmented copies)")
    print(f"  Validation images : {len(val_df)}  (original only, no augmentation)")
    print(f"\n  Diseases ({num_classes}):")
    for i, d in enumerate(diseases):
        cnt = (train_df["disease"] == d).sum()
        print(f"    {i:2d}: {d} ({cnt} images)")

    # --- Class weights (compensate for imbalance) ---
    labels_arr  = train_df["local_label"].values
    unique_cls  = np.unique(labels_arr)
    weights_arr = compute_class_weight("balanced", classes=unique_cls, y=labels_arr)
    class_weights = dict(zip(unique_cls, weights_arr))

    # --- Build tf.data datasets ---
    train_ds = make_dataset(train_df, apply_augmentation=True)
    val_ds   = make_dataset(val_df,   apply_augmentation=False)

    # --- Build model ---
    model, backbone = build_model(num_classes)
    model_file = MODELS_DIR / f"{animal_name}_model.keras"

    # ---- PHASE 1: Train head only (backbone frozen) ----
    print(f"\n  PHASE 1 — Training head only (backbone frozen)")
    print(f"  Learning rate: {PHASE1_LR} | Max epochs: {PHASE1_EPOCHS}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(PHASE1_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    cb_p1 = [
        EarlyStopping(monitor="val_accuracy", patience=6,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(model_file), monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
    ]

    history_p1 = model.fit(
        train_ds,
        epochs=PHASE1_EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=cb_p1,
        verbose=1
    )

    p1_best_acc = max(history_p1.history["val_accuracy"])
    print(f"\n  Phase 1 best val accuracy: {p1_best_acc*100:.2f}%")

    # ---- PHASE 2: Fine-tune top backbone layers ----
    print(f"\n  PHASE 2 — Fine-tuning top {UNFREEZE_LAYERS} backbone layers")
    print(f"  Learning rate: {PHASE2_LR} | Max epochs: {PHASE2_EPOCHS}")

    backbone.trainable = True
    total_layers  = len(backbone.layers)
    freeze_until  = total_layers - UNFREEZE_LAYERS
    for i, layer in enumerate(backbone.layers):
        layer.trainable = (i >= freeze_until)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(PHASE2_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    cb_p2 = [
        EarlyStopping(monitor="val_accuracy", patience=8,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(model_file), monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-8, verbose=1),
    ]

    history_p2 = model.fit(
        train_ds,
        epochs=PHASE2_EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=cb_p2,
        verbose=1
    )

    p2_best_acc = max(history_p2.history["val_accuracy"])
    best_acc    = max(p1_best_acc, p2_best_acc)
    print(f"\n  Phase 2 best val accuracy: {p2_best_acc*100:.2f}%")
    print(f"  OVERALL BEST for {animal_name}: {best_acc*100:.2f}%")

    # ---- Save training plot ----
    train_acc  = history_p1.history["accuracy"]     + history_p2.history["accuracy"]
    val_acc    = history_p1.history["val_accuracy"]  + history_p2.history["val_accuracy"]
    train_loss = history_p1.history["loss"]          + history_p2.history["loss"]
    val_loss   = history_p1.history["val_loss"]      + history_p2.history["val_loss"]
    p1_end     = len(history_p1.history["accuracy"])
    epochs_x   = range(1, len(train_acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs_x, train_acc,  "b-o", markersize=3, label="Train")
    axes[0].plot(epochs_x, val_acc,    "r-o", markersize=3, label="Validation")
    axes[0].axvline(x=p1_end, color="gray", linestyle="--",
                    label=f"Phase boundary (epoch {p1_end})")
    axes[0].set_title(f"{animal_name} — Accuracy", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_x, train_loss, "b-o", markersize=3, label="Train")
    axes[1].plot(epochs_x, val_loss,   "r-o", markersize=3, label="Validation")
    axes[1].axvline(x=p1_end, color="gray", linestyle="--")
    axes[1].set_title(f"{animal_name} — Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Animal Disease Model — {animal_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = RESULTS_DIR / f"{animal_name}_training_history.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training plot saved: {plot_path}")

    # ---- Save per-animal class mapping ----
    animal_mapping = {
        "animal":           animal_name,
        "num_classes":      num_classes,
        "disease_classes":  diseases,
        "disease_to_index": disease_to_local,
        "index_to_disease": {str(k): v for k, v in local_to_disease.items()},
        "best_val_accuracy": round(float(best_acc), 4),
        "model_file":       str(model_file),
    }
    mapping_file = MODELS_DIR / f"{animal_name}_class_mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(animal_mapping, f, indent=4)
    print(f"  Class mapping saved: {mapping_file}")

    return best_acc, animal_mapping


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("PER-ANIMAL DISEASE MODEL TRAINING  (v2 — with augmented data)")
print("=" * 70)
print(f"\nBase directory : {BASE_DIR}")
print(f"Train images   : {PREPARED_DIR / 'train'} (original + augmented)")
print(f"Models output  : {MODELS_DIR}")

# Load validation CSV once
print("\nLoading validation CSV...")
val_all = pd.read_csv(VAL_CSV)
print(f"  Validation total: {len(val_all)}")

# Train in order: most data first
ANIMALS = ["Cow", "Dog", "Cat", "Fish"]

all_results = {}

for animal in ANIMALS:
    best_acc, mapping = train_animal_model(animal, val_all)
    if best_acc is not None:
        all_results[animal] = {
            "best_val_accuracy": round(float(best_acc), 4),
            "num_classes":       mapping["num_classes"],
        }

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ALL MODELS TRAINED — FINAL SUMMARY")
print("=" * 70)

for animal, res in all_results.items():
    acc    = res["best_val_accuracy"] * 100
    nc     = res["num_classes"]
    status = "GOOD ✓" if acc >= 60 else "FAIR" if acc >= 40 else "LOW"
    print(f"\n  {animal:<6} | {nc} classes | Best val accuracy: {acc:.1f}% | {status}")

print(f"\n  Models saved in : {MODELS_DIR}/")
print(f"  Plots saved in  : {RESULTS_DIR}/")
print()
print("  NEXT STEP: python anaconda_projects/12_evaluate_image_model.py")
print("=" * 70)
