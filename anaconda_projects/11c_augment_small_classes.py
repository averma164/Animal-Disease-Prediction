# ============================================================
# 11c_augment_small_classes.py
#
# PURPOSE:
#   Some disease classes have very few images (as low as 41).
#   This script physically generates augmented copies of
#   those images and saves them to prepared_dataset/train/
#
#   WHY PHYSICAL COPIES instead of online augmentation?
#   - With only 47 images, even online augmentation shows
#     the same images too many times per epoch.
#   - Physical copies increase the actual dataset size,
#     so the model sees more variety per epoch.
#
#   TARGET: Bring every class to at least 200 training images.
#   Classes that already have >= 200 images are left alone.
#
# HOW TO RUN:
#   cd "D:\Animal disease Prediction"
#   python anaconda_projects/11c_augment_small_classes.py
#
# TIME: ~5-10 minutes
# ============================================================

import os
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

BASE_DIR      = Path(__file__).resolve().parent.parent
TRAIN_DIR     = BASE_DIR / "prepared_dataset" / "train"

# ---- TARGET: minimum images per class after augmentation ----
TARGET_COUNT = 200

# ---- Random seed for reproducibility ----
random.seed(42)
np.random.seed(42)

# ---- Valid image extensions ----
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ============================================================
# AUGMENTATION FUNCTIONS
# Each function takes a PIL Image and returns a new PIL Image.
# We use GENTLE augmentation — nothing that would distort
# the disease appearance unrealistically.
# ============================================================

def augment_image(img):
    """
    Apply a random combination of gentle augmentations to one image.
    Returns a new augmented PIL Image.
    """
    # Always convert to RGB (handles RGBA, grayscale, etc.)
    img = img.convert("RGB")

    # 1. Horizontal flip (50% chance)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 2. Gentle rotation (-15 to +15 degrees)
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, fillcolor=(128, 128, 128))

    # 3. Brightness adjustment (0.8 to 1.2)
    brightness = random.uniform(0.8, 1.2)
    img = ImageEnhance.Brightness(img).enhance(brightness)

    # 4. Contrast adjustment (0.8 to 1.2)
    contrast = random.uniform(0.8, 1.2)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    # 5. Slight zoom/crop (80% to 100% of image, then resize back)
    zoom_factor = random.uniform(0.85, 1.0)
    w, h = img.size
    new_w = int(w * zoom_factor)
    new_h = int(h * zoom_factor)
    left  = random.randint(0, w - new_w)
    top   = random.randint(0, h - new_h)
    img   = img.crop((left, top, left + new_w, top + new_h))
    img   = img.resize((w, h), Image.LANCZOS)

    return img


# ============================================================
# MAIN LOGIC
# ============================================================

print("=" * 65)
print("AUGMENTING SMALL CLASSES")
print(f"Target minimum images per class: {TARGET_COUNT}")
print("=" * 65)

total_generated = 0
summary = []

# Loop through all animal/disease folders in train/
for animal_dir in sorted(TRAIN_DIR.iterdir()):
    if not animal_dir.is_dir():
        continue
    animal = animal_dir.name

    for disease_dir in sorted(animal_dir.iterdir()):
        if not disease_dir.is_dir():
            continue
        disease = disease_dir.name

        # Get list of current original images
        orig_images = [
            f for f in disease_dir.iterdir()
            if f.suffix.lower() in IMG_EXTENSIONS
            and "_aug" not in f.name   # skip already-augmented files
        ]

        current_count = len(orig_images)

        if current_count >= TARGET_COUNT:
            print(f"  SKIP {animal}/{disease}: {current_count} images (>= {TARGET_COUNT})")
            summary.append((animal, disease, current_count, 0, current_count))
            continue

        # How many augmented images do we need to generate?
        needed = TARGET_COUNT - current_count
        print(f"\n  {animal}/{disease}: {current_count} → generating {needed} augmented images")

        generated = 0
        attempt   = 0

        while generated < needed:
            # Pick a random original image to augment
            src_path = random.choice(orig_images)

            try:
                img = Image.open(src_path)
                aug_img = augment_image(img)

                # Save with _aug_NNNN suffix
                aug_filename = f"{src_path.stem}_aug_{generated:04d}.jpg"
                aug_path = disease_dir / aug_filename
                aug_img.save(aug_path, "JPEG", quality=92)
                generated += 1

            except Exception as e:
                print(f"    WARNING: Could not augment {src_path.name}: {e}")

            attempt += 1
            if attempt > needed * 3:
                print(f"    WARNING: Too many failed attempts, stopping at {generated}")
                break

        total_generated += generated
        final_count = current_count + generated
        summary.append((animal, disease, current_count, generated, final_count))
        print(f"    Done: {current_count} → {final_count} images (+{generated})")


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("AUGMENTATION COMPLETE — SUMMARY")
print("=" * 65)
print(f"\n{'Animal':<8} {'Disease':<40} {'Before':>7} {'Added':>7} {'After':>7}")
print("-" * 65)

for animal, disease, before, added, after in summary:
    marker = "  ← augmented" if added > 0 else ""
    print(f"{animal:<8} {disease:<40} {before:>7} {added:>7} {after:>7}{marker}")

print("-" * 65)
print(f"\nTotal augmented images generated: {total_generated}")
print(f"\nNEXT STEP: python anaconda_projects/11b_train_per_animal_models.py")
print("(Retrain all models with the larger dataset)")
print("=" * 65)
