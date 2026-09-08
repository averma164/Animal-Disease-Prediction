import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "image_master_cleaned.csv"
REPORT_FILE = BASE_DIR / "cleaned_dataset_verification.csv"

EXPECTED_TOTAL_IMAGES = 9595
EXPECTED_ANIMALS = {"Cow", "Dog", "Cat", "Fish"}


# ============================================================
# LOAD CLEANED DATA
# ============================================================

print("=" * 70)
print("LOADING CLEANED IMAGE MASTER DATA")
print("=" * 70)

if not INPUT_FILE.exists():
    print(f"ERROR: File not found:")
    print(INPUT_FILE)
    raise SystemExit(1)

df = pd.read_csv(INPUT_FILE)

print(f"File: {INPUT_FILE}")
print(f"Total records: {len(df)}")


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(f"Rows              : {len(df)}")
print(f"Columns           : {len(df.columns)}")
print(f"Animal classes    : {df['animal'].nunique()}")
print(f"Disease classes   : {df['disease'].nunique()}")


# ============================================================
# REQUIRED COLUMNS CHECK
# ============================================================

print("\n" + "=" * 70)
print("REQUIRED COLUMN CHECK")
print("=" * 70)

required_columns = [
    "image_id",
    "image_path",
    "animal",
    "disease",
    "width",
    "height",
    "format",
    "file_size_kb",
    "status"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if len(missing_columns) == 0:
    print("✓ All required columns are present")
else:
    print("✗ Missing columns:")
    for column in missing_columns:
        print(f"  - {column}")


# ============================================================
# TOTAL IMAGE COUNT
# ============================================================

print("\n" + "=" * 70)
print("IMAGE COUNT CHECK")
print("=" * 70)

if len(df) == EXPECTED_TOTAL_IMAGES:
    print(f"✓ Image count is correct: {len(df)}")
else:
    print(
        f"✗ Image count mismatch: "
        f"expected {EXPECTED_TOTAL_IMAGES}, found {len(df)}"
    )


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

missing_values = df[required_columns].isnull().sum()

print(missing_values)

if missing_values.sum() == 0:
    print("\n✓ No missing values found")
else:
    print("\n✗ Missing values detected")


# ============================================================
# DUPLICATE IMAGE ID CHECK
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE IMAGE ID CHECK")
print("=" * 70)

duplicate_ids = df["image_id"].duplicated().sum()

print(f"Duplicate image IDs: {duplicate_ids}")

if duplicate_ids == 0:
    print("✓ No duplicate image IDs")
else:
    print("✗ Duplicate image IDs found")


# ============================================================
# DUPLICATE IMAGE PATH CHECK
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE IMAGE PATH CHECK")
print("=" * 70)

duplicate_paths = df["image_path"].duplicated().sum()

print(f"Duplicate image paths: {duplicate_paths}")

if duplicate_paths == 0:
    print("✓ No duplicate image paths")
else:
    print("✗ Duplicate image paths found")


# ============================================================
# ANIMAL LABEL CHECK
# ============================================================

print("\n" + "=" * 70)
print("ANIMAL LABEL CHECK")
print("=" * 70)

animal_labels = set(df["animal"].dropna().unique())

print("Animal labels found:")

for animal in sorted(animal_labels):
    print(f" - {animal}")

unknown_animals = animal_labels - EXPECTED_ANIMALS

if len(unknown_animals) == 0:
    print("\n✓ All animal labels are valid")
else:
    print("\n✗ Unknown animal labels found:")
    for animal in unknown_animals:
        print(f" - {animal}")


# ============================================================
# ANIMAL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ANIMAL DISTRIBUTION")
print("=" * 70)

animal_distribution = df["animal"].value_counts()

print(animal_distribution)


# ============================================================
# DISEASE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLEANED DISEASE DISTRIBUTION")
print("=" * 70)

disease_distribution = df["disease"].value_counts()

print(disease_distribution)


# ============================================================
# RINGWORM LABEL CHECK
# ============================================================

print("\n" + "=" * 70)
print("RINGWORM LABEL CHECK")
print("=" * 70)

lowercase_ringworm = (
    df["disease"].astype(str) == "ringworm"
).sum()

uppercase_ringworm = (
    df["disease"].astype(str) == "Ringworm"
).sum()

print(f"Lowercase 'ringworm': {lowercase_ringworm}")
print(f"'Ringworm': {uppercase_ringworm}")

if lowercase_ringworm == 0:
    print("✓ Lowercase 'ringworm' has been removed")
else:
    print("✗ Lowercase 'ringworm' still exists")

if uppercase_ringworm == 1039:
    print("✓ Ringworm count is correct: 1039")
else:
    print(
        f"⚠ Ringworm count is {uppercase_ringworm}, "
        f"expected 1039"
    )


# ============================================================
# SMALL CLASS CHECK
# ============================================================

print("\n" + "=" * 70)
print("SMALL DISEASE CLASS CHECK")
print("=" * 70)

small_classes = disease_distribution[
    disease_distribution < 50
]

if len(small_classes) == 0:
    print("✓ No disease classes have fewer than 50 images")
else:
    print("Disease classes with fewer than 50 images:")
    print(small_classes)


# ============================================================
# MASTITIS NORMAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("MASTITIS NORMAL CHECK")
print("=" * 70)

mastitis_normal = (
    df["disease"] == "mastitis normal"
).sum()

mastitis = (
    df["disease"] == "Mastitis"
).sum()

print(f"Mastitis          : {mastitis}")
print(f"mastitis normal   : {mastitis_normal}")

if mastitis_normal == 10:
    print("\n⚠ MANUAL INSPECTION REQUIRED")
    print("10 images are labeled 'mastitis normal'.")
    print("These images should be checked manually.")
    print("Do NOT merge them with Mastitis automatically.")
else:
    print("No 'mastitis normal' label found.")


# ============================================================
# ANIMAL × DISEASE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ANIMAL × DISEASE DISTRIBUTION")
print("=" * 70)

animal_disease = pd.crosstab(
    df["animal"],
    df["disease"]
)

print(animal_disease)


# ============================================================
# CHECK INVALID ANIMAL × DISEASE COMBINATIONS
# ============================================================

print("\n" + "=" * 70)
print("ANIMAL × DISEASE VALIDATION")
print("=" * 70)

invalid_combinations = []

for _, row in df.iterrows():

    animal = row["animal"]
    disease = row["disease"]

    # Fish-specific diseases
    fish_diseases = {
        "Bacterial Red disease",
        "Bacterial diseases - Aeromoniasis",
        "Bacterial gill disease",
        "Fungal diseases Saprolegniasis",
        "Healthy Fish",
        "Parasitic diseases",
        "Viral diseases White tail disease"
    }

    if disease in fish_diseases and animal != "Fish":
        invalid_combinations.append(
            (animal, disease)
        )

    # Cow-specific diseases
    cow_diseases = {
        "Foot_and_Mouth",
        "Lumpy_Skin",
        "Mastitis",
        "Warts",
        "mastitis normal"
    }

    if disease in cow_diseases and animal != "Cow":
        invalid_combinations.append(
            (animal, disease)
        )

if len(invalid_combinations) == 0:
    print("✓ No obvious invalid animal × disease combinations")
else:
    print("✗ Potentially invalid combinations found:")

    unique_invalid = sorted(
        set(invalid_combinations)
    )

    for animal, disease in unique_invalid:
        print(f" - {animal} × {disease}")


# ============================================================
# IMAGE STATUS CHECK
# ============================================================

print("\n" + "=" * 70)
print("IMAGE STATUS CHECK")
print("=" * 70)

if "status" in df.columns:

    status_distribution = df["status"].value_counts()

    print(status_distribution)

    bad_statuses = df[
        ~df["status"].astype(str).str.lower().isin(
            ["valid", "ok", "healthy"]
        )
    ]

    print(f"\nNon-standard status records: {len(bad_statuses)}")

else:
    print("⚠ Status column not found")


# ============================================================
# CREATE VERIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CREATING VERIFICATION REPORT")
print("=" * 70)

verification_data = {
    "check": [
        "Total images",
        "Animal classes",
        "Disease classes",
        "Missing values",
        "Duplicate image IDs",
        "Duplicate image paths",
        "Lowercase ringworm",
        "Ringworm count",
        "Mastitis count",
        "mastitis normal count"
    ],
    "result": [
        len(df),
        df["animal"].nunique(),
        df["disease"].nunique(),
        int(missing_values.sum()),
        int(duplicate_ids),
        int(duplicate_paths),
        int(lowercase_ringworm),
        int(uppercase_ringworm),
        int(mastitis),
        int(mastitis_normal)
    ]
}

verification_report = pd.DataFrame(
    verification_data
)

verification_report.to_csv(
    REPORT_FILE,
    index=False
)

print(f"Saved: {REPORT_FILE}")


# ============================================================
# FINAL VERDICT
# ============================================================

print("\n" + "=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

checks_passed = True

if len(df) != EXPECTED_TOTAL_IMAGES:
    checks_passed = False

if missing_values.sum() != 0:
    checks_passed = False

if duplicate_ids != 0:
    checks_passed = False

if duplicate_paths != 0:
    checks_passed = False

if lowercase_ringworm != 0:
    checks_passed = False

if unknown_animals:
    checks_passed = False

if checks_passed:
    print("✓ DATASET VERIFICATION PASSED")
    print("✓ Dataset is structurally ready for the next stage.")
else:
    print("✗ DATASET VERIFICATION FAILED")
    print("Please fix the reported issues before proceeding.")

print("\n⚠ IMPORTANT:")
print("The 10 'mastitis normal' images still require manual inspection.")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)