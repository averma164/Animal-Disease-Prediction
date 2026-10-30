# ============================================================
# 12_evaluate_image_model.py
#
# PURPOSE:
#   Evaluate the trained per-animal models on the TEST set.
#   (Test set = images never seen during training or validation)
#
#   Generates:
#   - Accuracy, Precision, Recall, F1-score per class
#   - Confusion matrix (visual chart)
#   - Per-animal summary
#   - results/evaluation_report.json
#   - results/[Animal]_confusion_matrix.png
#
# HOW TO RUN:
#   cd "D:\Animal disease Prediction"
#   python anaconda_projects/12_evaluate_image_model.py
# ============================================================

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for scripts)
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

BASE_DIR     = Path(__file__).resolve().parent.parent
PREPARED_DIR = BASE_DIR / "prepared_dataset"
LABELED_DIR  = BASE_DIR / "dataset_split" / "labeled"
MODELS_DIR   = BASE_DIR / "models"
RESULTS_DIR  = BASE_DIR / "results"

TEST_CSV = LABELED_DIR / "test_labeled.csv"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("EVALUATING PER-ANIMAL DISEASE MODELS ON TEST SET")
print("=" * 70)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_to_ext(fmt):
    return {"JPEG": ".jpg", "JPG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(
        str(fmt).upper().strip(), ".jpg"
    )


def resolve_test_path(row):
    """Find image in prepared_dataset/test/"""
    image_id = str(row["image_id"])
    animal   = str(row["animal"])
    disease  = str(row["disease"])
    ext      = format_to_ext(row["format"])
    for dv in [disease.replace("_", " "), disease]:
        p = PREPARED_DIR / "test" / animal / dv / f"{image_id}{ext}"
        if p.exists():
            return str(p)
    return None


def load_and_preprocess(image_path, backbone_name="MobileNetV2", target_size=IMAGE_SIZE):
    """Load a single image and preprocess according to model backbone."""
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, target_size)
    if backbone_name == "MobileNetV2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        img = preprocess_input(img)
    # EfficientNet expects unscaled [0, 255] float input
    return img


import argparse

parser = argparse.ArgumentParser(description="Evaluate animal disease models.")
parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation (flips + zoom crops).")
parser.add_argument("--animal", choices=["all", "Cow", "Dog", "Cat", "Fish"], default="all", help="Animal to evaluate.")
args, _ = parser.parse_known_args()

def predict_batch(model, paths, backbone_name="MobileNetV2", use_tta=False):
    """Run model inference on a list of image paths. Supports dynamic size & Test-Time Augmentation."""
    target_size = model.input_shape[1:3] if model.input_shape and model.input_shape[1] else IMAGE_SIZE
    if not use_tta:
        images = []
        for p in paths:
            try:
                img = load_and_preprocess(p, backbone_name=backbone_name, target_size=target_size)
                images.append(img.numpy())
            except Exception:
                images.append(np.zeros((*target_size, 3), dtype=np.float32))
        batch = np.array(images)
        return model.predict(batch, verbose=0)
    else:
        # Test-Time Augmentation: average predictions over original, flip, and center crop
        orig_imgs, flip_imgs, crop_imgs = [], [], []
        for p in paths:
            try:
                raw = tf.io.read_file(p)
                img = tf.image.decode_image(raw, channels=3, expand_animations=False)
                img1 = tf.image.resize(img, target_size)
                img2 = tf.image.flip_left_right(img1)
                img3 = tf.image.resize(tf.image.central_crop(img1, 0.9), target_size)
                if backbone_name == "MobileNetV2":
                    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
                    img1 = preprocess_input(img1)
                    img2 = preprocess_input(img2)
                    img3 = preprocess_input(img3)
                orig_imgs.append(img1.numpy())
                flip_imgs.append(img2.numpy())
                crop_imgs.append(img3.numpy())
            except Exception:
                zeros = np.zeros((*target_size, 3), dtype=np.float32)
                orig_imgs.append(zeros)
                flip_imgs.append(zeros)
                crop_imgs.append(zeros)
        p1 = model.predict(np.array(orig_imgs), verbose=0)
        p2 = model.predict(np.array(flip_imgs), verbose=0)
        p3 = model.predict(np.array(crop_imgs), verbose=0)
        return (p1 + p2 + p3) / 3.0


def plot_confusion_matrix(cm, class_names, animal_name, save_path):
    """Plot and save a confusion matrix as a PNG file."""
    n = len(class_names)
    fig_size = max(8, n * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    # Write counts in cells
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"{animal_name} — Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# LOAD TEST CSV
# ============================================================

print(f"\nLoading test CSV: {TEST_CSV}")
if not TEST_CSV.exists():
    print("ERROR: test_labeled.csv not found.")
    raise SystemExit(1)

test_all = pd.read_csv(TEST_CSV)
print(f"  Total test records: {len(test_all)}")

# ============================================================
# EVALUATE EACH ANIMAL MODEL
# ============================================================

ANIMALS = ["Cow", "Dog", "Cat", "Fish"] if args.animal == "all" else [args.animal]
all_results = {}

for animal in ANIMALS:
    print("\n" + "=" * 70)
    print(f"  EVALUATING: {animal.upper()} (TTA={'ON' if args.tta else 'OFF'})")
    print("=" * 70)

    model_file   = MODELS_DIR / f"{animal}_model.keras"
    mapping_file = MODELS_DIR / f"{animal}_class_mapping.json"

    if not model_file.exists():
        print(f"  SKIP: {model_file.name} not found.")
        continue
    if not mapping_file.exists():
        print(f"  SKIP: {mapping_file.name} not found.")
        continue

    # Load class mapping
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    diseases        = mapping["disease_classes"]
    disease_to_idx  = mapping["disease_to_index"]
    idx_to_disease  = {int(k): v for k, v in mapping["index_to_disease"].items()}
    num_classes     = mapping["num_classes"]

    # Load model
    print(f"\n  Loading model: {model_file.name}")
    model = tf.keras.models.load_model(str(model_file))
    print(f"  Model loaded. Classes: {num_classes}")

    # Filter test data for this animal
    test_animal = test_all[test_all["animal"] == animal].copy()
    test_animal["full_path"]  = test_animal.apply(resolve_test_path, axis=1)
    test_animal["path_found"] = test_animal["full_path"].notna()
    test_df = test_animal[test_animal["path_found"]].reset_index(drop=True)

    # Only keep diseases in the model's class mapping
    test_df = test_df[test_df["disease"].isin(diseases)].reset_index(drop=True)
    test_df["local_label"] = test_df["disease"].map(disease_to_idx).astype(int)

    print(f"  Test images found: {len(test_df)}")
    if len(test_df) == 0:
        print("  SKIP: No test images found.")
        continue

    # Print test distribution
    print(f"\n  Test set distribution:")
    for d, cnt in test_df["disease"].value_counts().items():
        print(f"    {d}: {cnt}")

    # ---- Run predictions in batches ----
    print(f"\n  Running predictions (TTA={'ON' if args.tta else 'OFF'})...")
    all_preds  = []
    all_labels = test_df["local_label"].values

    backbone_name = mapping.get("backbone", "MobileNetV2")
    paths_list = test_df["full_path"].tolist()
    for start in range(0, len(paths_list), BATCH_SIZE):
        batch_paths = paths_list[start : start + BATCH_SIZE]
        probs = predict_batch(model, batch_paths, backbone_name=backbone_name, use_tta=args.tta)
        preds = np.argmax(probs, axis=1)
        all_preds.extend(preds.tolist())

    all_preds = np.array(all_preds)

    # ---- Metrics ----
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"\n  Test Accuracy : {acc*100:.2f}%")
    print(f"  Weighted F1   : {f1*100:.2f}%")

    # Per-class report
    print(f"\n  Per-class report:")
    report = classification_report(
        all_labels, all_preds,
        target_names=diseases,
        zero_division=0
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    cm_path = RESULTS_DIR / f"{animal}_confusion_matrix.png"
    plot_confusion_matrix(cm, diseases, animal, cm_path)

    # Store results
    all_results[animal] = {
        "test_accuracy": round(float(acc), 4),
        "weighted_f1":   round(float(f1), 4),
        "num_test_images": len(test_df),
        "num_classes": num_classes,
        "diseases": diseases,
    }

    # Clean up model from memory
    del model
    tf.keras.backend.clear_session()

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE — FINAL SUMMARY")
print("=" * 70)
print(f"\n{'Animal':<8} {'Test Images':>12} {'Accuracy':>10} {'F1 Score':>10}")
print("-" * 45)

for animal, res in all_results.items():
    acc = res["test_accuracy"] * 100
    f1  = res["weighted_f1"] * 100
    n   = res["num_test_images"]
    print(f"  {animal:<6} {n:>12}  {acc:>9.1f}%  {f1:>9.1f}%")

print("-" * 45)
print(f"\n  Confusion matrices saved in: {RESULTS_DIR}/")

# Save evaluation report (merge if evaluating single animal)
report_file = RESULTS_DIR / "evaluation_report.json"
existing_results = {}
if report_file.exists():
    try:
        with open(report_file, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
    except Exception:
        existing_results = {}
existing_results.update(all_results)
with open(report_file, "w", encoding="utf-8") as f:
    json.dump(existing_results, f, indent=4)
print(f"  Evaluation report saved: {report_file}")

print()
print("  NEXT STEP: python anaconda_projects/13_test_image_prediction.py")
print("             --image <path_to_image> --animal <Cat|Dog|Cow|Fish>")
print("=" * 70)
