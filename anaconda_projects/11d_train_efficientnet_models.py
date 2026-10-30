# ============================================================
# 11d_train_efficientnet_models.py
#
# PURPOSE:
#   Train per-animal disease classification models using
#   EfficientNetB0 backbone (higher accuracy and feature representation
#   than MobileNetV2).
#
# KEY IMPROVEMENTS FOR IMAGE ACCURACY:
#   1. EfficientNetB0 backbone (ImageNet top-1: 77.1% vs 71.8%)
#   2. Deep fine-tuning (unfreezing top 60 layers in Phase 2)
#   3. Automatic backup of existing models before replacing
#   4. Command-line flag --animal [Cow|Dog|Cat|Fish|all]
#
# HOW TO RUN:
#   Train one animal (faster, test improvement first):
#     python anaconda_projects/11d_train_efficientnet_models.py --animal Cow
#
#   Train all animals:
#     python anaconda_projects/11d_train_efficientnet_models.py --animal all
# ============================================================

import sys
import json
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
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
BATCH_SIZE      = 32

PHASE1_EPOCHS   = 25
PHASE2_EPOCHS   = 20
PHASE1_LR       = 1e-3
PHASE2_LR       = 1e-5
UNFREEZE_LAYERS = 60    # Deeper fine-tuning for dermatological textures

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA SCANNING & RESOLUTION
# ============================================================

def scan_train_folder(animal_name, disease_list):
    """Scan prepared_dataset/train/[animal]/ for all image files."""
    train_animal_dir = PREPARED_DIR / "train" / animal_name
    rows = []

    for disease_name in disease_list:
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
                break

    return pd.DataFrame(rows)


def format_to_ext(fmt):
    return {"JPEG": ".jpg", "JPG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(
        str(fmt).upper().strip(), ".jpg"
    )


def resolve_val_path(row):
    """Find validation image in prepared_dataset/validation/"""
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
# DATA AUGMENTATION & PIPELINE
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.12),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomContrast(0.15),
    layers.RandomBrightness(0.15),
], name="data_augmentation")


def load_image(path, label):
    """Load image from disk and resize."""
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = preprocess_input(img)
    return img, label


def make_dataset(df, num_classes, apply_augmentation=False):
    """Build tf.data pipeline with one-hot targets for Focal Loss and label smoothing."""
    paths  = df["full_path"].values
    labels = df["local_label"].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if apply_augmentation:
        ds = ds.shuffle(len(df), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if apply_augmentation:
        ds = ds.map(
            lambda img, lbl: (data_augmentation(img, training=True), tf.one_hot(lbl, num_classes)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        ds = ds.map(
            lambda img, lbl: (img, tf.one_hot(lbl, num_classes)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# BUILD EFFICIENTNET-B0 MODEL
# ============================================================

def build_efficientnet_model(num_classes):
    """EfficientNetB0 backbone + enhanced dense classification head."""
    backbone = EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(384, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model, backbone


# ============================================================
# TRAIN ONE ANIMAL
# ============================================================

def train_animal_model(animal_name, val_all):
    print("\n" + "=" * 70)
    print(f"  TRAINING EFFICIENTNET-B0 MODEL FOR: {animal_name.upper()}")
    print("=" * 70)

    val_animal = val_all[val_all["animal"] == animal_name].copy()
    val_animal["full_path"]  = val_animal.apply(resolve_val_path, axis=1)
    val_animal["path_found"] = val_animal["full_path"].notna()
    val_df = val_animal[val_animal["path_found"]].reset_index(drop=True)

    diseases = sorted(val_df["disease"].unique())
    num_classes = len(diseases)

    if num_classes == 0:
        print(f"  SKIP: No validation data found for {animal_name}.")
        return None, None

    disease_to_local = {d: i for i, d in enumerate(diseases)}
    local_to_disease = {i: d for d, i in disease_to_local.items()}

    train_df = scan_train_folder(animal_name, diseases)
    if len(train_df) < 20:
        print(f"  SKIP: Too few training images for {animal_name}.")
        return None, None

    train_df["local_label"] = train_df["disease"].map(disease_to_local)
    val_df["local_label"]   = val_df["disease"].map(disease_to_local)
    val_df = val_df[val_df["local_label"].notna()].copy()
    val_df["local_label"]   = val_df["local_label"].astype(int)

    print(f"\n  Training images   : {len(train_df)}")
    print(f"  Validation images : {len(val_df)}")
    print(f"\n  Diseases ({num_classes}):")
    for i, d in enumerate(diseases):
        cnt = (train_df["disease"] == d).sum()
        print(f"    {i:2d}: {d} ({cnt} images)")

    # Class weights for imbalance
    labels_arr  = train_df["local_label"].values
    unique_cls  = np.unique(labels_arr)
    weights_arr = compute_class_weight("balanced", classes=unique_cls, y=labels_arr)
    class_weights = dict(zip(unique_cls, weights_arr))

    train_ds = make_dataset(train_df, num_classes=num_classes, apply_augmentation=True)
    val_ds   = make_dataset(val_df,   num_classes=num_classes, apply_augmentation=False)

    model, backbone = build_efficientnet_model(num_classes)
    model_file   = MODELS_DIR / f"{animal_name}_model.keras"
    backup_file  = MODELS_DIR / f"{animal_name}_model_mobilenet_backup.keras"

    # Backup existing model if present
    if model_file.exists() and not backup_file.exists():
        shutil.copy2(model_file, backup_file)
        print(f"  Backup of current model saved to: {backup_file.name}")

    focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(
        alpha=0.25,
        gamma=2.0,
        label_smoothing=0.1
    )

    # ---- PHASE 1: Head Only ----
    print(f"\n  PHASE 1 — Training head only (backbone frozen)")
    print(f"  Learning rate: {PHASE1_LR} | Max epochs: {PHASE1_EPOCHS} | Focal Loss (gamma=2.0, label_smoothing=0.1)")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(PHASE1_LR),
        loss=focal_loss,
        metrics=["accuracy"]
    )

    cb_p1 = [
        EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(model_file), monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
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

    # ---- PHASE 2: Fine-Tuning Top 60 Layers ----
    print(f"\n  PHASE 2 — Fine-tuning top {UNFREEZE_LAYERS} backbone layers")
    print(f"  Learning rate: {PHASE2_LR} | Max epochs: {PHASE2_EPOCHS}")

    backbone.trainable = True
    total_layers = len(backbone.layers)
    freeze_until = max(0, total_layers - UNFREEZE_LAYERS)
    for i, layer in enumerate(backbone.layers):
        layer.trainable = (i >= freeze_until)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(PHASE2_LR),
        loss=focal_loss,
        metrics=["accuracy"]
    )

    cb_p2 = [
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(model_file), monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-8, verbose=1),
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

    # Plot history
    train_acc  = history_p1.history["accuracy"]    + history_p2.history["accuracy"]
    val_acc    = history_p1.history["val_accuracy"] + history_p2.history["val_accuracy"]
    train_loss = history_p1.history["loss"]         + history_p2.history["loss"]
    val_loss   = history_p1.history["val_loss"]     + history_p2.history["val_loss"]
    p1_end     = len(history_p1.history["accuracy"])
    epochs_x   = range(1, len(train_acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs_x, train_acc, "b-o", markersize=3, label="Train")
    axes[0].plot(epochs_x, val_acc,   "r-o", markersize=3, label="Validation")
    axes[0].axvline(x=p1_end, color="gray", linestyle="--", label=f"Phase boundary ({p1_end})")
    axes[0].set_title(f"{animal_name} — Accuracy (EfficientNetB0)", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_x, train_loss, "b-o", markersize=3, label="Train")
    axes[1].plot(epochs_x, val_loss,   "r-o", markersize=3, label="Validation")
    axes[1].axvline(x=p1_end, color="gray", linestyle="--")
    axes[1].set_title(f"{animal_name} — Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Animal Disease Model (EfficientNetB0) — {animal_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = RESULTS_DIR / f"{animal_name}_training_history.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Save class mapping
    animal_mapping = {
        "animal":            animal_name,
        "backbone":          "EfficientNetB0",
        "num_classes":       num_classes,
        "disease_classes":   diseases,
        "disease_to_index":  disease_to_local,
        "index_to_disease":  {str(k): v for k, v in local_to_disease.items()},
        "best_val_accuracy": round(float(best_acc), 4),
        "model_file":        str(model_file),
    }
    mapping_file = MODELS_DIR / f"{animal_name}_class_mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(animal_mapping, f, indent=4)
    print(f"  Class mapping saved: {mapping_file}")

    return best_acc, animal_mapping


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 animal disease models")
    parser.add_argument("--animal", default="all", choices=["Cow", "Dog", "Cat", "Fish", "all"],
                        help="Specify single animal or 'all'")
    args = parser.parse_args()

    print("=" * 70)
    print("PER-ANIMAL DISEASE MODEL TRAINING  (EfficientNetB0 Backbone)")
    print("=" * 70)

    val_all = pd.read_csv(VAL_CSV)
    targets = ["Cow", "Dog", "Cat", "Fish"] if args.animal == "all" else [args.animal]

    all_results = {}
    for animal in targets:
        best_acc, mapping = train_animal_model(animal, val_all)
        if best_acc is not None:
            all_results[animal] = {
                "best_val_accuracy": round(float(best_acc), 4),
                "num_classes":       mapping["num_classes"],
            }

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    for animal, res in all_results.items():
        acc = res["best_val_accuracy"] * 100
        print(f"  {animal:<6} | {res['num_classes']} classes | Best val accuracy: {acc:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
