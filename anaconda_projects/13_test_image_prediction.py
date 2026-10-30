# ============================================================
# 13_test_image_prediction.py
#
# PURPOSE:
#   Test the trained models on a single image.
#   Shows top-3 disease predictions with confidence scores.
#
# HOW TO RUN:
#   cd "D:\Animal disease Prediction"
#   python anaconda_projects/13_test_image_prediction.py --image "path\to\image.jpg" --animal Cat
#
# EXAMPLE:
#   python anaconda_projects/13_test_image_prediction.py --image "D:\Animal disease Prediction\rawdata\Cat\Ringworm\Image_1.jpg" --animal Cat
# ============================================================

import argparse
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
IMAGE_SIZE = (224, 224)

# Confidence threshold — below this we say "uncertain"
UNCERTAINTY_THRESHOLD = 0.25


def load_model_and_mapping(animal_name):
    """Load the trained model and class mapping for the given animal."""
    model_file   = MODELS_DIR / f"{animal_name}_model.keras"
    mapping_file = MODELS_DIR / f"{animal_name}_class_mapping.json"

    if not model_file.exists():
        print(f"ERROR: Model not found: {model_file}")
        print(f"  Please run: python anaconda_projects/11b_train_per_animal_models.py")
        raise SystemExit(1)

    if not mapping_file.exists():
        print(f"ERROR: Class mapping not found: {mapping_file}")
        raise SystemExit(1)

    model = tf.keras.models.load_model(str(model_file))

    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    return model, mapping


def preprocess_image(image_path):
    """Load and preprocess one image for MobileNetV2."""
    path = Path(image_path)
    if not path.exists():
        print(f"ERROR: Image not found: {image_path}")
        raise SystemExit(1)

    raw = tf.io.read_file(str(path))
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = preprocess_input(img)
    img = tf.expand_dims(img, axis=0)   # Add batch dimension: (1, 224, 224, 3)
    return img


def predict(image_path, animal_name):
    """Run prediction and return top-3 results."""
    print(f"\nLoading model for: {animal_name}")
    model, mapping = load_model_and_mapping(animal_name)

    index_to_disease = {int(k): v for k, v in mapping["index_to_disease"].items()}

    print(f"Preprocessing image: {image_path}")
    img_tensor = preprocess_image(image_path)

    print("Running prediction...")
    probs = model.predict(img_tensor, verbose=0)[0]   # Shape: (num_classes,)

    # Get top-3 predictions
    top3_indices = np.argsort(probs)[::-1][:3]
    top3 = [
        {
            "rank":       i + 1,
            "disease":    index_to_disease[idx],
            "confidence": float(probs[idx]),
        }
        for i, idx in enumerate(top3_indices)
    ]

    return top3, float(probs[top3_indices[0]])


def print_results(image_path, animal_name, top3, top_confidence):
    """Print results in a clear, readable format."""
    print("\n" + "=" * 55)
    print("  ANIMAL DISEASE PREDICTION RESULT")
    print("=" * 55)
    print(f"  Animal : {animal_name}")
    print(f"  Image  : {Path(image_path).name}")
    print()

    if top_confidence < UNCERTAINTY_THRESHOLD:
        print("  ⚠ STATUS: UNCERTAIN PREDICTION")
        print(f"  Top confidence is only {top_confidence*100:.1f}%.")
        print("  The model is not confident about this image.")
        print("  This may be due to:")
        print("    - Image quality or angle")
        print("    - Disease not clearly visible")
        print("    - Disease not well represented in training data")
        print()

    print("  Top Predictions:")
    print(f"  {'Rank':<5} {'Disease':<35} {'Confidence':>12}")
    print("  " + "-" * 52)

    for pred in top3:
        bar_len = int(pred["confidence"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  #{pred['rank']:<4} {pred['disease']:<35} {pred['confidence']*100:>8.1f}%")
        print(f"       [{bar}]")

    print()
    print("  " + "=" * 52)
    print("  ⚠ DISCLAIMER:")
    print("  This is an experimental machine-learning prototype.")
    print("  These are model confidence scores, NOT medical")
    print("  probabilities or veterinary diagnoses.")
    print("  Always consult a qualified veterinarian.")
    print("=" * 55)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict animal disease from a single image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help='Path to the image file. Example: "D:\\path\\to\\image.jpg"'
    )
    parser.add_argument(
        "--animal",
        required=True,
        choices=["Cat", "Dog", "Cow", "Fish"],
        help="Animal type: Cat, Dog, Cow, or Fish"
    )

    args = parser.parse_args()

    top3, top_conf = predict(args.image, args.animal)
    print_results(args.image, args.animal, top3, top_conf)
