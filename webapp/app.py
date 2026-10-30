"""
app.py — Animal Disease Prediction Web Application
====================================================
Flask web server that serves the prediction UI.

HOW TO RUN:
    cd "D:\Animal disease Prediction\webapp"
    python app.py

Then open your browser to: http://localhost:5000
"""

import sys
import os
import json
import base64
import traceback
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# ---- Add anaconda_projects to Python path so we can import our scripts ----
BASE_DIR     = Path(__file__).resolve().parent.parent
SCRIPTS_DIR  = BASE_DIR / "anaconda_projects"
SYMPTOM_FILE = BASE_DIR / "symptom_data" / "disease_symptoms.json"
sys.path.insert(0, str(SCRIPTS_DIR))

# ---- Import our prediction engine ----
from importlib import import_module
combined_module = import_module("15_combined_prediction")
symptom_module  = import_module("14_symptom_engine")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

ANIMALS = ["Cat", "Dog", "Cow", "Fish"]

# Animal emoji mapping
ANIMAL_EMOJI = {
    "Cat":  "🐱",
    "Dog":  "🐶",
    "Cow":  "🐄",
    "Fish": "🐟",
}


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    """Main page."""
    return render_template("index.html", animals=ANIMALS, animal_emoji=ANIMAL_EMOJI)


@app.route("/get_symptoms/<animal>")
def get_symptoms(animal):
    """
    Returns the list of all symptoms for a given animal.
    Called by the frontend when the user selects an animal.
    """
    if animal not in ANIMALS:
        return jsonify({"error": f"Unknown animal: {animal}"}), 400

    try:
        symptoms = symptom_module.get_all_symptoms_for_animal(animal)
        diseases = list(symptom_module.get_animal_symptoms(animal).keys())
        return jsonify({
            "animal":   animal,
            "symptoms": symptoms,
            "diseases": diseases,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.
    Accepts: multipart/form-data with:
        - animal        : "Cat" / "Dog" / "Cow" / "Fish"
        - image         : uploaded image file
        - symptoms[]    : list of selected symptom strings
    """
    try:
        animal = request.form.get("animal", "").strip()
        if animal not in ANIMALS:
            return jsonify({"error": f"Invalid animal: {animal}"}), 400

        # Get uploaded image
        if "image" not in request.files or request.files["image"].filename == "":
            return jsonify({"error": "No image uploaded. Please select an image."}), 400

        image_file = request.files["image"]
        image_bytes = image_file.read()

        if len(image_bytes) == 0:
            return jsonify({"error": "Uploaded file is empty."}), 400

        # Get selected symptoms
        selected_symptoms = request.form.getlist("symptoms[]")

        # Run combined prediction
        result = combined_module.predict_combined(
            animal_name       = animal,
            selected_symptoms = selected_symptoms,
            image_bytes       = image_bytes,
        )

        # Encode image as base64 for display in result page
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        ext = image_file.filename.rsplit(".", 1)[-1].lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

        result["image_data"]  = f"data:{mime};base64,{image_b64}"
        result["emoji"]       = ANIMAL_EMOJI.get(animal, "🐾")
        result["symptoms_used"] = selected_symptoms

        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({"error": f"Model not found: {e}"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  ANIMAL DISEASE PREDICTION WEB APP")
    print("=" * 60)
    print(f"  Base directory : {BASE_DIR}")
    print(f"  Models directory: {BASE_DIR / 'models'}")
    print()
    print("  Starting server...")
    print("  Open your browser at: http://localhost:5000")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
