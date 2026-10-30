# ============================================================
# 15_combined_prediction.py
#
# PURPOSE:
#   Combine image model prediction + symptom engine score
#   into one final ranked prediction.
#
# FORMULA (as agreed):
#   combined_score = 0.75 × image_score + 0.25 × symptom_score
#
#   Image contributes 75% (the main signal)
#   Symptoms contribute 25% (supporting signal)
#
# HOW TO RUN (standalone test):
#   cd "D:\Animal disease Prediction"
#   python anaconda_projects/15_combined_prediction.py
#
# USED BY:
#   The web application (app.py) imports and calls predict_combined()
# ============================================================

import json
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Add the project root to path so we can import 14_symptom_engine
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib import import_module
symptom_engine = import_module("14_symptom_engine")

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
IMAGE_SIZE = (224, 224)

# Weights for combining image and symptom scores (balanced 50/50 for higher diagnostic accuracy)
IMAGE_WEIGHT   = 0.50
SYMPTOM_WEIGHT = 0.50

# Confidence threshold — below this label as "Uncertain"
UNCERTAINTY_THRESHOLD = 0.20

# Cache loaded models to avoid reloading on every request
_model_cache   = {}
_mapping_cache = {}


# ============================================================
# MODEL LOADING
# ============================================================

def load_animal_model(animal_name):
    """Load and cache the model + mapping for the given animal."""
    if animal_name not in _model_cache:
        model_file   = MODELS_DIR / f"{animal_name}_model.keras"
        mapping_file = MODELS_DIR / f"{animal_name}_class_mapping.json"

        if not model_file.exists():
            raise FileNotFoundError(
                f"Model not found: {model_file}\n"
                "Please run 11b_train_per_animal_models.py first."
            )

        _model_cache[animal_name] = tf.keras.models.load_model(str(model_file))

        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        _mapping_cache[animal_name] = {
            "disease_classes":  mapping["disease_classes"],
            "index_to_disease": {int(k): v for k, v in mapping["index_to_disease"].items()},
            "backbone":         mapping.get("backbone", "MobileNetV2"),
        }

    return _model_cache[animal_name], _mapping_cache[animal_name]


# Disease type category mapping for dynamic weighting
# Internal/systemic diseases rely much more heavily on clinical symptoms
# Cutaneous/lesion diseases rely primarily on visual image evidence
SYSTEMIC_DISEASES = {
    "Distemper", "Parvovirus", "Worm Infection", "Kennel Cough",
    "Urinary Tract Infection"
}

CUTANEOUS_DISEASES = {
    "Ringworm", "Mange", "Lumpy_Skin", "Warts", "Foot_and_Mouth",
    "Tick Infestation", "Ear Mites", "Bacterial gill disease", "Healthy Fish", "Healthy",
    "Hot Spots", "Fungal Infection", "Skin Allergy"
}

def get_disease_weights(disease_name):
    """Return (image_weight, symptom_weight) tailored to disease nature."""
    if disease_name in SYSTEMIC_DISEASES:
        return 0.15, 0.85  # Symptoms carry 85% of diagnostic signal for internal diseases
    elif disease_name in CUTANEOUS_DISEASES:
        return 0.70, 0.30  # Image carries 70% for visual skin lesions
    else:
        return 0.45, 0.55  # Balanced for eye infections, dental disease, etc.


# ============================================================
# IMAGE PREDICTION WITH TEST-TIME AUGMENTATION (TTA)
# ============================================================

def predict_from_image(animal_name, image_path=None, image_bytes=None, use_tta=True):
    """
    Run the image model with Test-Time Augmentation (TTA) and return a dict of {disease: confidence_score}.

    Parameters:
        animal_name  : "Cat", "Dog", "Cow", or "Fish"
        image_path   : path to image file (str or Path)
        image_bytes  : raw image bytes (from web upload)
        use_tta      : bool, whether to use 3-view test-time augmentation (averaging original,
                       horizontal flip, and central crop) for robust predictions.

    Returns:
        dict: {disease_name: score_float, ...}  — scores sum to ~1.0
    """
    model, mapping = load_animal_model(animal_name)

    if image_path is not None:
        raw = tf.io.read_file(str(image_path))
    elif image_bytes is not None:
        raw = tf.constant(image_bytes)
    else:
        raise ValueError("Provide either image_path or image_bytes.")

    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    target_size = model.input_shape[1:3] if model.input_shape and model.input_shape[1] else IMAGE_SIZE
    img1 = tf.image.resize(img, target_size)
    backbone_name = mapping.get("backbone", "MobileNetV2")

    if use_tta:
        # 3-view TTA: 1) Original, 2) Horizontal Flip, 3) Subtle Zoom (90% center crop)
        img2 = tf.image.flip_left_right(img1)
        img3 = tf.image.resize(tf.image.central_crop(img1, 0.90), target_size)

        if backbone_name == "MobileNetV2":
            img1 = preprocess_input(img1)
            img2 = preprocess_input(img2)
            img3 = preprocess_input(img3)

        batch = tf.stack([img1, img2, img3])
        probs_batch = model.predict(batch, verbose=0)
        probs = np.mean(probs_batch, axis=0)
    else:
        if backbone_name == "MobileNetV2":
            img1 = preprocess_input(img1)
        batch = tf.expand_dims(img1, axis=0)
        probs = model.predict(batch, verbose=0)[0]

    result = {
        mapping["index_to_disease"][i]: float(probs[i])
        for i in range(len(probs))
    }
    return result


# ============================================================
# SYMPTOM SCORE NORMALIZATION
# ============================================================

def get_symptom_scores_normalized(animal_name, selected_symptoms):
    """
    Return a dict of {disease: normalized_score_0_to_1}.
    If no symptoms selected, returns 0.0 for all diseases.
    """
    if not selected_symptoms:
        # No symptoms = no signal from symptoms side
        return {}

    raw_results = symptom_engine.calculate_symptom_scores(animal_name, selected_symptoms)
    return {r["disease"]: r["score"] for r in raw_results}


# ============================================================
# COMBINED PREDICTION
# ============================================================

def predict_combined(animal_name, selected_symptoms, image_path=None, image_bytes=None):
    """
    Main prediction function. Combines image model + symptom engine.

    Parameters:
        animal_name       : "Cat", "Dog", "Cow", or "Fish"
        selected_symptoms : list of symptom strings (may be empty)
        image_path        : path to image file
        image_bytes       : raw image bytes (from web upload)

    Returns:
        dict with keys:
          "top_predictions" : list of top predictions (sorted by combined score)
          "image_scores"    : raw image model scores for all diseases
          "symptom_scores"  : symptom compatibility scores for all diseases
          "mode"            : "image_only" or "image_and_symptoms"
    """
    # --- Get image scores ---
    image_scores = predict_from_image(animal_name, image_path, image_bytes)
    all_diseases = list(image_scores.keys())

    # --- Get symptom scores ---
    use_symptoms   = bool(selected_symptoms)
    symptom_scores = get_symptom_scores_normalized(animal_name, selected_symptoms)

    # --- Combine scores with dynamic disease-aware weighting ---
    raw_combined = []
    for disease in all_diseases:
        img_score  = image_scores.get(disease, 0.0)
        symp_score = symptom_scores.get(disease, 0.0)

        if use_symptoms:
            w_img, w_symp = get_disease_weights(disease)
            # Synergistic multimodal fusion:
            # If symptoms support this disease (symp_score > 0), image and symptoms reinforce each other.
            # If symptoms do not match, the disease hypothesis is moderated by (1 - w_symp).
            if symp_score > 0:
                score = (w_img * img_score) + (w_symp * symp_score)
            else:
                score = (1.0 - w_symp * 0.75) * img_score
        else:
            score = img_score   # pure vision mode when no symptoms provided

        raw_combined.append({
            "disease":       disease,
            "image_score":   round(img_score,  4),
            "symptom_score": round(symp_score, 4),
            "raw_score":     max(0.0, float(score)),
        })

    # Calibrate combined scores into a valid normalized probability distribution summing to 1.0
    total_raw = sum(x["raw_score"] for x in raw_combined)
    combined = []
    for item in raw_combined:
        if total_raw > 0:
            final_score = item["raw_score"] / total_raw
        else:
            final_score = item["raw_score"]

        combined.append({
            "disease":       item["disease"],
            "image_score":   item["image_score"],
            "symptom_score": item["symptom_score"],
            "combined_score": round(final_score, 4),
        })

    # Sort by combined score descending
    combined.sort(key=lambda x: x["combined_score"], reverse=True)

    # Add rank, percentage, and confidence label
    top_confidence = combined[0]["combined_score"] if combined else 0.0

    for i, item in enumerate(combined):
        item["rank"] = i + 1
        item["percentage"] = round(item["combined_score"] * 100, 1)

        # Confidence label based on top prediction
        score = item["combined_score"]
        if score >= 0.70:
            item["confidence_label"] = "Very High"
        elif score >= 0.50:
            item["confidence_label"] = "High"
        elif score >= 0.35:
            item["confidence_label"] = "Medium"
        elif score >= 0.20:
            item["confidence_label"] = "Low"
        else:
            item["confidence_label"] = "Very Low"

    # Determine overall certainty
    if top_confidence < UNCERTAINTY_THRESHOLD:
        status = "uncertain"
    elif combined[0]["confidence_label"] in ("Very High", "High"):
        status = "confident"
    else:
        status = "moderate"

    return {
        "animal":         animal_name,
        "mode":           "image_and_symptoms" if use_symptoms else "image_only",
        "status":         status,
        "top_disease":    combined[0]["disease"] if combined else None,
        "top_confidence": round(top_confidence * 100, 1),
        "top_predictions": combined[:5],    # Top 5
        "all_predictions": combined,        # All diseases
        "image_scores":    image_scores,
        "symptom_scores":  symptom_scores,
    }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image",    required=True,  help="Path to image")
    parser.add_argument("--animal",   required=True,  choices=["Cat", "Dog", "Cow", "Fish"])
    parser.add_argument("--symptoms", nargs="*", default=[], help="List of symptoms")
    args = parser.parse_args()

    print(f"\nRunning combined prediction...")
    print(f"  Animal   : {args.animal}")
    print(f"  Image    : {args.image}")
    print(f"  Symptoms : {args.symptoms or 'none selected'}")

    result = predict_combined(
        animal_name       = args.animal,
        selected_symptoms = args.symptoms,
        image_path        = args.image,
    )

    print(f"\n{'='*60}")
    print(f"COMBINED PREDICTION RESULT")
    print(f"{'='*60}")
    print(f"  Mode    : {result['mode']}")
    print(f"  Status  : {result['status'].upper()}")
    print(f"\n  Top 5 Predictions:")
    print(f"  {'Rank':<5} {'Disease':<32} {'Image':>7} {'Symptom':>9} {'Combined':>10}")
    print(f"  {'-'*60}")
    for p in result["top_predictions"]:
        print(f"  #{p['rank']:<4} {p['disease']:<32} "
              f"{p['image_score']*100:>6.1f}% "
              f"{p['symptom_score']*100:>8.1f}% "
              f"{p['percentage']:>9.1f}%  [{p['confidence_label']}]")

    print(f"\n  Top Diagnosis: {result['top_disease']}")
    print(f"  Confidence   : {result['top_confidence']}%")
    print(f"{'='*60}")
