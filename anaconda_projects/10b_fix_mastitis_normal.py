# ============================================================
# 10b_fix_mastitis_normal.py
#
# PURPOSE:
#   The 'mastitis normal' class has only 7 training images.
#   This is far too few for a model to learn anything useful.
#
#   'mastitis normal' represents a healthy/normal udder
#   (not diseased), so it makes sense to merge it into
#   the existing 'Healthy' class.
#
#   This script:
#   1. Updates train_labeled.csv, validation_labeled.csv,
#      test_labeled.csv to remap disease='mastitis normal'
#      -> disease='Healthy'
#   2. Updates class_mapping.json to remove 'mastitis normal'
#      as a separate class (goes from 28 to 27 classes)
#
# HOW TO RUN:
#   cd "D:\Animal disease Prediction"
#   python anaconda_projects/10b_fix_mastitis_normal.py
# ============================================================

import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LABELED_DIR = BASE_DIR / "dataset_split" / "labeled"
CLASS_MAPPING_FILE = BASE_DIR / "class_mapping" / "class_mapping.json"

print("=" * 60)
print("FIX: Merge 'mastitis normal' -> 'Healthy'")
print("=" * 60)


# ============================================================
# STEP 1: Load and update the labeled CSV files
# ============================================================

for split in ["train", "validation", "test"]:
    csv_path = LABELED_DIR / f"{split}_labeled.csv"

    if not csv_path.exists():
        print(f"WARNING: {csv_path.name} not found, skipping.")
        continue

    df = pd.read_csv(csv_path)

    # Count before
    before = (df["disease"] == "mastitis normal").sum()

    if before == 0:
        print(f"\n{split}: No 'mastitis normal' rows found — already clean.")
        continue

    print(f"\n{split}: Found {before} 'mastitis normal' rows.")

    # Change disease name to 'Healthy'
    df.loc[df["disease"] == "mastitis normal", "disease"] = "Healthy"

    # We will fix the numeric disease_label below after updating class mapping
    # For now, save the disease name fix; we'll recalculate labels after
    df.to_csv(csv_path, index=False)
    print(f"  -> Disease name updated in {csv_path.name}")

print()


# ============================================================
# STEP 2: Update class_mapping.json
#
# Remove 'mastitis normal' (was index 27) and rebuild all
# indices so they are clean and consecutive.
# ============================================================

print("=" * 60)
print("Updating class_mapping.json")
print("=" * 60)

with open(CLASS_MAPPING_FILE, "r", encoding="utf-8") as f:
    class_mapping = json.load(f)

disease_classes = class_mapping["disease_classes"]

print(f"\nBefore: {len(disease_classes)} disease classes")
print(f"  'mastitis normal' was at index: "
      f"{class_mapping['disease_to_index'].get('mastitis normal', 'NOT FOUND')}")

# Remove 'mastitis normal' from the list
if "mastitis normal" in disease_classes:
    disease_classes.remove("mastitis normal")
    print("  Removed 'mastitis normal' from disease_classes")

# Rebuild all disease index mappings from scratch
# (clean consecutive integers: 0, 1, 2, ...)
disease_to_index = {disease: idx for idx, disease in enumerate(disease_classes)}
index_to_disease = {str(idx): disease for idx, disease in enumerate(disease_classes)}

# Update the class mapping dict
class_mapping["disease_classes"]  = disease_classes
class_mapping["disease_to_index"] = disease_to_index
class_mapping["index_to_disease"] = index_to_disease

print(f"\nAfter: {len(disease_classes)} disease classes")
print(f"  'Healthy' is now at index: {disease_to_index.get('Healthy', 'NOT FOUND')}")

# Save updated class mapping
with open(CLASS_MAPPING_FILE, "w", encoding="utf-8") as f:
    json.dump(class_mapping, f, indent=4)

print(f"\n  Saved updated class_mapping.json")


# ============================================================
# STEP 3: Recalculate numeric disease_label in all CSVs
#
# Because indices may have shifted (anything after index 27
# is unaffected in this case, but we do it properly anyway).
# ============================================================

print("\n" + "=" * 60)
print("Recalculating numeric disease_label in all split CSVs")
print("=" * 60)

for split in ["train", "validation", "test"]:
    csv_path = LABELED_DIR / f"{split}_labeled.csv"

    if not csv_path.exists():
        continue

    df = pd.read_csv(csv_path)

    # Map disease name -> numeric label using updated mapping
    df["disease_label"] = df["disease"].map(disease_to_index)

    # Check for any unmapped entries
    unmapped = df[df["disease_label"].isna()]["disease"].unique()
    if len(unmapped) > 0:
        print(f"\nWARNING in {split}: These disease names have no mapping:")
        for d in unmapped:
            print(f"  '{d}'")
    else:
        print(f"\n{split}: All disease_label values updated correctly.")

    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path.name}")


# ============================================================
# STEP 4: Quick summary
# ============================================================

print("\n" + "=" * 60)
print("FINAL CLASS MAPPING SUMMARY")
print("=" * 60)

with open(CLASS_MAPPING_FILE, "r", encoding="utf-8") as f:
    final_mapping = json.load(f)

print(f"\nAnimal classes ({len(final_mapping['animal_classes'])}):")
for a in final_mapping["animal_classes"]:
    print(f"  {a}")

print(f"\nDisease classes ({len(final_mapping['disease_classes'])}):")
for idx, disease in enumerate(final_mapping["disease_classes"]):
    print(f"  {idx:2d}: {disease}")

print("\n" + "=" * 60)
print("DONE. class_mapping.json now has 27 disease classes.")
print("NEXT STEP: python anaconda_projects/11_train_image_model.py")
print("=" * 60)
