import os
import json
import pandas as pd
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\Animal disease Prediction"

TRAIN_CSV = os.path.join(
    BASE_DIR, "dataset_split", "train.csv"
)

VAL_CSV = os.path.join(
    BASE_DIR, "dataset_split", "validation.csv"
)

TEST_CSV = os.path.join(
    BASE_DIR, "dataset_split", "test.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "class_mapping"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CLASS MAPPING CREATION")
print("=" * 70)


# ============================================================
# LOAD DATASETS
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASET SPLITS")
print("=" * 70)

try:
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

except FileNotFoundError as e:
    print("\nERROR: Dataset split file not found.")
    print(e)
    print("\nMake sure you have already run:")
    print("python 06_split_dataset.py")
    raise SystemExit(1)


print(f"Train images      : {len(train_df)}")
print(f"Validation images : {len(val_df)}")
print(f"Test images       : {len(test_df)}")

total_images = len(train_df) + len(val_df) + len(test_df)

print(f"Total images      : {total_images}")


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

print("\n" + "=" * 70)
print("CHECKING REQUIRED COLUMNS")
print("=" * 70)

required_columns = [
    "image_id",
    "image_path",
    "animal",
    "disease"
]

for name, df in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        print(f"✗ {name} is missing columns:")
        print(missing_columns)
        raise SystemExit(1)

    print(f"✓ {name}: all required columns present")


# ============================================================
# COMBINE DATASETS FOR GLOBAL CLASS MAPPING
# ============================================================

print("\n" + "=" * 70)
print("CREATING GLOBAL CLASS LISTS")
print("=" * 70)

all_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)


# ============================================================
# ANIMAL CLASSES
# ============================================================

animal_classes = sorted(
    all_df["animal"].dropna().unique().tolist()
)

print("\nAnimal classes:")
for i, animal in enumerate(animal_classes):
    print(f"  {i}: {animal}")


# ============================================================
# DISEASE CLASSES
# ============================================================

disease_classes = sorted(
    all_df["disease"].dropna().unique().tolist()
)

print("\nDisease classes:")

for i, disease in enumerate(disease_classes):
    print(f"  {i}: {disease}")


print(f"\nTotal animal classes  : {len(animal_classes)}")
print(f"Total disease classes : {len(disease_classes)}")


# ============================================================
# CREATE MAPPINGS
# ============================================================

animal_to_index = {
    animal: i
    for i, animal in enumerate(animal_classes)
}

index_to_animal = {
    str(i): animal
    for animal, i in animal_to_index.items()
}


disease_to_index = {
    disease: i
    for i, disease in enumerate(disease_classes)
}

index_to_disease = {
    str(i): disease
    for disease, i in disease_to_index.items()
}


# ============================================================
# SAVE JSON MAPPING
# ============================================================

mapping = {
    "animal_classes": animal_classes,
    "disease_classes": disease_classes,

    "animal_to_index": animal_to_index,
    "index_to_animal": index_to_animal,

    "disease_to_index": disease_to_index,
    "index_to_disease": index_to_disease
}


mapping_json = os.path.join(
    OUTPUT_DIR,
    "class_mapping.json"
)

with open(mapping_json, "w", encoding="utf-8") as f:
    json.dump(
        mapping,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\n✓ Saved:")
print(mapping_json)


# ============================================================
# SAVE ANIMAL CLASS CSV
# ============================================================

animal_rows = []

for index, animal in enumerate(animal_classes):

    count = int(
        (all_df["animal"] == animal).sum()
    )

    train_count = int(
        (train_df["animal"] == animal).sum()
    )

    val_count = int(
        (val_df["animal"] == animal).sum()
    )

    test_count = int(
        (test_df["animal"] == animal).sum()
    )

    animal_rows.append({
        "class_index": index,
        "animal": animal,
        "total_images": count,
        "train_images": train_count,
        "validation_images": val_count,
        "test_images": test_count
    })


animal_df = pd.DataFrame(animal_rows)

animal_csv = os.path.join(
    OUTPUT_DIR,
    "animal_classes.csv"
)

animal_df.to_csv(
    animal_csv,
    index=False
)

print("✓ Saved:")
print(animal_csv)


# ============================================================
# SAVE DISEASE CLASS CSV
# ============================================================

disease_rows = []

for index, disease in enumerate(disease_classes):

    count = int(
        (all_df["disease"] == disease).sum()
    )

    train_count = int(
        (train_df["disease"] == disease).sum()
    )

    val_count = int(
        (val_df["disease"] == disease).sum()
    )

    test_count = int(
        (test_df["disease"] == disease).sum()
    )

    disease_rows.append({
        "class_index": index,
        "disease": disease,
        "total_images": count,
        "train_images": train_count,
        "validation_images": val_count,
        "test_images": test_count
    })


disease_df = pd.DataFrame(disease_rows)

disease_csv = os.path.join(
    OUTPUT_DIR,
    "disease_classes.csv"
)

disease_df.to_csv(
    disease_csv,
    index=False
)

print("✓ Saved:")
print(disease_csv)


# ============================================================
# ADD NUMERICAL LABELS TO DATASETS
# ============================================================

print("\n" + "=" * 70)
print("CREATING NUMERICAL LABELS")
print("=" * 70)


def add_labels(df):

    df = df.copy()

    df["animal_label"] = df["animal"].map(
        animal_to_index
    )

    df["disease_label"] = df["disease"].map(
        disease_to_index
    )

    return df


train_labeled = add_labels(train_df)
val_labeled = add_labels(val_df)
test_labeled = add_labels(test_df)


# ============================================================
# CHECK FOR UNMAPPED LABELS
# ============================================================

print("\nChecking for unmapped labels...")

for name, df in [
    ("Train", train_labeled),
    ("Validation", val_labeled),
    ("Test", test_labeled)
]:

    missing_animal = df["animal_label"].isna().sum()
    missing_disease = df["disease_label"].isna().sum()

    print(f"\n{name}:")
    print(f"  Missing animal labels  : {missing_animal}")
    print(f"  Missing disease labels : {missing_disease}")

    if missing_animal > 0 or missing_disease > 0:
        print(f"✗ {name} contains unmapped labels")
        raise SystemExit(1)

    print("  ✓ All labels mapped successfully")


# ============================================================
# SAVE LABELED DATASETS
# ============================================================

labeled_dir = os.path.join(
    BASE_DIR,
    "dataset_split",
    "labeled"
)

os.makedirs(labeled_dir, exist_ok=True)


train_labeled_path = os.path.join(
    labeled_dir,
    "train_labeled.csv"
)

val_labeled_path = os.path.join(
    labeled_dir,
    "validation_labeled.csv"
)

test_labeled_path = os.path.join(
    labeled_dir,
    "test_labeled.csv"
)


train_labeled.to_csv(
    train_labeled_path,
    index=False
)

val_labeled.to_csv(
    val_labeled_path,
    index=False
)

test_labeled.to_csv(
    test_labeled_path,
    index=False
)


print("\n✓ Saved labeled datasets:")
print(train_labeled_path)
print(val_labeled_path)
print(test_labeled_path)


# ============================================================
# CHECK CLASS PRESENCE IN EACH SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CHECKING CLASS PRESENCE ACROSS SPLITS")
print("=" * 70)


train_animals = set(train_df["animal"].unique())
val_animals = set(val_df["animal"].unique())
test_animals = set(test_df["animal"].unique())

train_diseases = set(train_df["disease"].unique())
val_diseases = set(val_df["disease"].unique())
test_diseases = set(test_df["disease"].unique())


# -----------------------------
# Animal check
# -----------------------------

print("\nAnimal classes:")

for animal in animal_classes:

    train_ok = animal in train_animals
    val_ok = animal in val_animals
    test_ok = animal in test_animals

    status = "✓" if (
        train_ok and val_ok and test_ok
    ) else "✗"

    print(
        f"{status} {animal:<10} "
        f"Train={train_ok} "
        f"Validation={val_ok} "
        f"Test={test_ok}"
    )


# -----------------------------
# Disease check
# -----------------------------

print("\nDisease classes:")

for disease in disease_classes:

    train_ok = disease in train_diseases
    val_ok = disease in val_diseases
    test_ok = disease in test_diseases

    status = "✓" if (
        train_ok and val_ok and test_ok
    ) else "✗"

    print(
        f"{status} {disease:<40} "
        f"Train={train_ok} "
        f"Validation={val_ok} "
        f"Test={test_ok}"
    )


# ============================================================
# DATASET COUNT CHECK
# ============================================================

print("\n" + "=" * 70)
print("DATASET COUNT VERIFICATION")
print("=" * 70)

print(f"Original total : {total_images}")
print(f"Train          : {len(train_labeled)}")
print(f"Validation     : {len(val_labeled)}")
print(f"Test           : {len(test_labeled)}")

calculated_total = (
    len(train_labeled)
    + len(val_labeled)
    + len(test_labeled)
)

print(f"Split total    : {calculated_total}")

if calculated_total == total_images:
    print("✓ Dataset count preserved")
else:
    print("✗ Dataset count mismatch")
    raise SystemExit(1)


# ============================================================
# LABEL RANGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("LABEL RANGE CHECK")
print("=" * 70)

animal_min = all_df["animal"].map(
    animal_to_index
).min()

animal_max = all_df["animal"].map(
    animal_to_index
).max()

disease_min = all_df["disease"].map(
    disease_to_index
).min()

disease_max = all_df["disease"].map(
    disease_to_index
).max()


print(
    f"Animal label range  : {animal_min} - {animal_max}"
)

print(
    f"Disease label range : {disease_min} - {disease_max}"
)


if animal_min == 0 and animal_max == len(animal_classes) - 1:
    print("✓ Animal labels are valid")

else:
    print("✗ Animal label range is invalid")
    raise SystemExit(1)


if disease_min == 0 and disease_max == len(disease_classes) - 1:
    print("✓ Disease labels are valid")

else:
    print("✗ Disease label range is invalid")
    raise SystemExit(1)


# ============================================================
# PRINT DISEASE MAPPING TABLE
# ============================================================

print("\n" + "=" * 70)
print("DISEASE CLASS MAPPING")
print("=" * 70)

print(
    f"{'ID':<5} {'Disease':<45} {'Total':<8}"
)

print("-" * 70)

for _, row in disease_df.iterrows():

    print(
        f"{int(row['class_index']):<5} "
        f"{row['disease']:<45} "
        f"{int(row['total_images']):<8}"
    )


# ============================================================
# PRINT ANIMAL MAPPING TABLE
# ============================================================

print("\n" + "=" * 70)
print("ANIMAL CLASS MAPPING")
print("=" * 70)

print(
    f"{'ID':<5} {'Animal':<15} {'Total':<8}"
)

print("-" * 40)

for _, row in animal_df.iterrows():

    print(
        f"{int(row['class_index']):<5} "
        f"{row['animal']:<15} "
        f"{int(row['total_images']):<8}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLASS MAPPING SUMMARY")
print("=" * 70)

print(f"Total images          : {total_images}")
print(f"Animal classes        : {len(animal_classes)}")
print(f"Disease classes       : {len(disease_classes)}")

print("\nAnimal classes:")
for i, animal in enumerate(animal_classes):
    print(f"  {i}: {animal}")

print("\nDisease classes:")
for i, disease in enumerate(disease_classes):
    print(f"  {i}: {disease}")


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

all_good = True


if len(train_labeled) != 6716:
    print("✗ Train count mismatch")
    all_good = False

if len(val_labeled) != 1439:
    print("✗ Validation count mismatch")
    all_good = False

if len(test_labeled) != 1440:
    print("✗ Test count mismatch")
    all_good = False

if len(animal_classes) != 4:
    print("✗ Expected 4 animal classes")
    all_good = False

if len(disease_classes) != 28:
    print("✗ Expected 28 disease classes")
    all_good = False


if all_good:

    print("✓ Dataset counts verified")
    print("✓ Animal classes verified")
    print("✓ Disease classes verified")
    print("✓ Numerical labels created")
    print("✓ Mapping files created")
    print("✓ Train/validation/test mappings consistent")

    print("\n" + "=" * 70)
    print("CLASS MAPPING CREATION COMPLETE")
    print("=" * 70)

else:

    print("\n✗ CLASS MAPPING VERIFICATION FAILED")
    raise SystemExit(1)