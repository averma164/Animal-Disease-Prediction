# ============================================================
# 16_evaluate_multimodal.py
#
# PURPOSE:
#   Evaluate the COMBINED diagnostic accuracy (Image Model + Symptom Engine)
#   on the test dataset when a user provides both an image and 2 observed symptoms.
# ============================================================

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report
import sys

BASE_DIR    = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE_DIR / "models"
LABELED_DIR = BASE_DIR / "dataset_split" / "labeled"
PREPARED_DIR= BASE_DIR / "prepared_dataset"
SYMPTOM_FILE= BASE_DIR / "symptom_data" / "disease_symptoms.json"
TEST_CSV    = LABELED_DIR / "test_labeled.csv"

sys.path.insert(0, str(BASE_DIR / "anaconda_projects"))
from importlib import import_module
combined_module = import_module("15_combined_prediction")

# Load symptoms database
with open(SYMPTOM_FILE, "r", encoding="utf-8") as f:
    symptom_db = json.load(f)

def format_to_ext(fmt):
    return {"JPEG": ".jpg", "JPG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(
        str(fmt).upper().strip(), ".jpg"
    )

def resolve_test_path(row):
    image_id = str(row["image_id"])
    animal   = str(row["animal"])
    disease  = str(row["disease"])
    ext      = format_to_ext(row["format"])
    for dv in [disease.replace("_", " "), disease]:
        p = PREPARED_DIR / "test" / animal / dv / f"{image_id}{ext}"
        if p.exists():
            return str(p)
    return None

test_all = pd.read_csv(TEST_CSV)
random.seed(42)

print("=" * 70)
print("EVALUATING COMBINED MULTIMODAL DIAGNOSTIC ACCURACY (IMAGE + SYMPTOMS)")
print("=" * 70)

for animal in ["Cow", "Dog", "Cat", "Fish"]:
    mapping_file = MODELS_DIR / f"{animal}_class_mapping.json"
    if not mapping_file.exists():
        continue
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    diseases = mapping["disease_classes"]
    animal_symps = symptom_db.get(animal, {})

    test_animal = test_all[test_all["animal"] == animal].copy()
    test_animal["full_path"] = test_animal.apply(resolve_test_path, axis=1)
    df = test_animal[test_animal["full_path"].notna() & test_animal["disease"].isin(diseases)].reset_index(drop=True)

    true_labels = []
    pred_labels = []

    for _, row in df.iterrows():
        d_true = row["disease"]
        img_path = row["full_path"]

        # Simulate 2 observed clinical symptoms for this disease
        avail_symps = animal_symps.get(d_true, [])
        if len(avail_symps) >= 2:
            sampled_symps = random.sample(avail_symps, 2)
        elif len(avail_symps) == 1:
            sampled_symps = avail_symps
        else:
            sampled_symps = []

        try:
            res = combined_module.predict_combined(
                animal_name=animal,
                selected_symptoms=sampled_symps,
                image_path=img_path
            )
            top_pred = res["top_predictions"][0]["disease"]
            true_labels.append(d_true)
            pred_labels.append(top_pred)
        except Exception:
            continue

    acc = accuracy_score(true_labels, pred_labels)
    f1  = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    print(f"\n{animal.upper()} — Combined Multimodal Results:")
    print(f"  Test Images Evaluated : {len(true_labels)}")
    print(f"  Combined Accuracy     : {acc*100:.2f}%")
    print(f"  Combined Weighted F1  : {f1*100:.2f}%")
print("=" * 70)
