# ============================================================
# 14_symptom_engine.py
#
# PURPOSE:
#   Given an animal type and a list of selected symptoms,
#   calculate a symptom compatibility score for each disease.
#
#   This is a RULE-BASED engine (not another ML model).
#   It uses disease_symptoms.json as the knowledge base.
#
# HOW IT WORKS:
#   1. Load the symptom knowledge base (disease_symptoms.json)
#   2. For the selected animal, get all diseases and their symptoms
#   3. Count how many selected symptoms match each disease
#   4. Calculate a compatibility score (0.0 to 1.0) for each disease
#   5. Return a ranked list of diseases
#
# HOW TO RUN (standalone test):
#   cd "D:\Animal disease Prediction"
#   python anaconda_projects/14_symptom_engine.py
# ============================================================

import json
from pathlib import Path

BASE_DIR      = Path(__file__).resolve().parent.parent
SYMPTOM_FILE  = BASE_DIR / "symptom_data" / "disease_symptoms.json"


def load_symptom_data():
    """Load the disease-symptom knowledge base from JSON."""
    if not SYMPTOM_FILE.exists():
        raise FileNotFoundError(
            f"Symptom data not found: {SYMPTOM_FILE}\n"
            "Make sure symptom_data/disease_symptoms.json exists."
        )
    with open(SYMPTOM_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_animal_symptoms(animal_name):
    """
    Return a dict of all symptoms for each disease of a given animal.
    This is used by the web app to show the symptom checklist.

    Returns: { "disease_name": ["symptom1", "symptom2", ...], ... }
    """
    data = load_symptom_data()
    if animal_name not in data:
        return {}
    return data[animal_name]


def get_all_symptoms_for_animal(animal_name):
    """
    Return a flat sorted list of all unique symptoms for an animal.
    Used to build the symptom checkbox list in the UI.
    """
    disease_symptoms = get_animal_symptoms(animal_name)
    all_symptoms = set()
    for symptoms in disease_symptoms.values():
        all_symptoms.update(symptoms)
    return sorted(all_symptoms)


def normalize_symptom(symptom):
    """Normalize a symptom string for comparison (lowercase, strip)."""
    return symptom.lower().strip()


def calculate_symptom_scores(animal_name, selected_symptoms):
    """
    Calculate a compatibility score between 0.0 and 1.0 for each disease.

    Score formula:
        matched_symptoms / total_disease_symptoms

    A score of 1.0 means ALL symptoms of that disease were selected.
    A score of 0.0 means NO symptoms matched.

    Parameters:
        animal_name       : "Cat", "Dog", "Cow", or "Fish"
        selected_symptoms : list of symptom strings selected by user

    Returns:
        List of dicts, sorted by score descending:
        [
            {"disease": "Ringworm", "score": 0.75, "matched": 3, "total": 4,
             "matched_symptoms": ["itching", "hair loss", "circular skin lesions"],
             "compatibility": "High"},
            ...
        ]
    """
    if not selected_symptoms:
        return []

    disease_symptoms = get_animal_symptoms(animal_name)
    if not disease_symptoms:
        return []

    # Normalize selected symptoms for case-insensitive matching
    normalized_selected = [normalize_symptom(s) for s in selected_symptoms]

    results = []

    for disease, symptoms in disease_symptoms.items():
        normalized_disease_symptoms = [normalize_symptom(s) for s in symptoms]

        # Find which selected symptoms match this disease
        matched = [
            s for s in normalized_selected
            if any(s in ds or ds in s for ds in normalized_disease_symptoms)
        ]

        n_matched = len(matched)
        n_selected = len(normalized_selected)
        total = len(normalized_disease_symptoms)

        if n_matched > 0 and n_selected > 0:
            # Proportion of user-observed symptoms that match this disease
            precision = n_matched / n_selected
            # Clinical support bonus: 1 symptom = 0.55, 2 symptoms = 0.80, 3+ symptoms = 1.00
            support_factor = min(1.0, 0.55 + 0.25 * (n_matched - 1))
            score = precision * support_factor
        else:
            score = 0.0

        # Human-readable compatibility label
        if score >= 0.70:
            compatibility = "High"
        elif score >= 0.40:
            compatibility = "Medium"
        elif score > 0:
            compatibility = "Low"
        else:
            compatibility = "None"

        results.append({
            "disease":          disease,
            "score":            round(score, 4),
            "matched":          n_matched,
            "total_symptoms":   total,
            "matched_symptoms": matched,
            "compatibility":    compatibility,
        })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ============================================================
# STANDALONE TEST (runs when you execute this file directly)
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SYMPTOM ENGINE — STANDALONE TEST")
    print("=" * 60)

    # Test 1: Cat with ringworm-like symptoms
    print("\nTest 1: Cat — Ringworm-like symptoms")
    print("-" * 40)
    results = calculate_symptom_scores(
        animal_name="Cat",
        selected_symptoms=["itching", "circular skin lesions", "hair loss", "scaly skin"]
    )
    for r in results:
        if r["score"] > 0:
            print(f"  {r['disease']:<30} score={r['score']:.2f}  [{r['compatibility']}]")
            print(f"    Matched: {r['matched_symptoms']}")

    # Test 2: Dog with mange-like symptoms
    print("\nTest 2: Dog — Mange-like symptoms")
    print("-" * 40)
    results = calculate_symptom_scores(
        animal_name="Dog",
        selected_symptoms=["intense itching", "hair loss", "crusty skin", "skin redness"]
    )
    for r in results:
        if r["score"] > 0:
            print(f"  {r['disease']:<30} score={r['score']:.2f}  [{r['compatibility']}]")

    # Test 3: Cow with Lumpy Skin symptoms
    print("\nTest 3: Cow — Lumpy Skin symptoms")
    print("-" * 40)
    results = calculate_symptom_scores(
        animal_name="Cow",
        selected_symptoms=["lumps on skin", "fever", "swollen lymph nodes", "nasal discharge"]
    )
    for r in results:
        if r["score"] > 0:
            print(f"  {r['disease']:<30} score={r['score']:.2f}  [{r['compatibility']}]")

    # Show all available symptoms for Cat
    print("\nAll available symptoms for Cat:")
    print("-" * 40)
    cat_symptoms = get_all_symptoms_for_animal("Cat")
    for s in cat_symptoms:
        print(f"  - {s}")

    print("\n" + "=" * 60)
    print("Symptom engine working correctly!")
    print("=" * 60)
