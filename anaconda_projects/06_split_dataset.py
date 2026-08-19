import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"D:\Animal disease Prediction")

INPUT_FILE = BASE_DIR / "image_master_cleaned.csv"

OUTPUT_DIR = BASE_DIR / "dataset_split"

TRAIN_FILE = OUTPUT_DIR / "train.csv"
VAL_FILE = OUTPUT_DIR / "validation.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING CLEANED DATASET")
print("=" * 70)

if not INPUT_FILE.exists():
    print("ERROR: Cleaned dataset not found:")
    print(INPUT_FILE)
    raise SystemExit(1)

df = pd.read_csv(INPUT_FILE)

print(f"Total images: {len(df)}")


# ============================================================
# BASIC CHECK
# ============================================================

required_columns = [
    "image_id",
    "image_path",
    "animal",
    "disease"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("ERROR: Missing columns:")
    for col in missing_columns:
        print(f" - {col}")
    raise SystemExit(1)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

duplicate_ids = df["image_id"].duplicated().sum()
duplicate_paths = df["image_path"].duplicated().sum()

print(f"Duplicate image IDs   : {duplicate_ids}")
print(f"Duplicate image paths : {duplicate_paths}")

if duplicate_ids > 0 or duplicate_paths > 0:
    print("\nERROR: Duplicate images detected.")
    print("Fix duplicates before splitting.")
    raise SystemExit(1)

print("✓ No duplicates found")


# ============================================================
# CREATE STRATIFICATION LABEL
# ============================================================

print("\n" + "=" * 70)
print("CREATING STRATIFICATION LABEL")
print("=" * 70)

df["stratify_label"] = (
    df["animal"].astype(str)
    + "_"
    + df["disease"].astype(str)
)

combination_counts = df["stratify_label"].value_counts()

print(
    f"Animal × disease combinations: "
    f"{len(combination_counts)}"
)

print("\nSmallest combinations:")

print(
    combination_counts
    .sort_values()
    .head(15)
)


# ============================================================
# HANDLE VERY SMALL COMBINATIONS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING SMALL STRATIFICATION GROUPS")
print("=" * 70)

small_groups = combination_counts[
    combination_counts < 3
]

if len(small_groups) > 0:

    print(
        f"Groups with fewer than 3 images: "
        f"{len(small_groups)}"
    )

    print("\nThese groups cannot be safely stratified.")

    for label, count in small_groups.items():
        print(f" - {label}: {count}")

else:
    print("✓ All groups have at least 3 images")


# ============================================================
# CREATE SAFE STRATIFICATION LABEL
# ============================================================

# Groups with fewer than 3 images cannot be present
# in train/validation/test using normal stratification.
#
# We keep these very small groups together initially
# and distribute them after the main stratified split.

small_labels = set(small_groups.index)

df["safe_stratify"] = df["stratify_label"]

df.loc[
    df["stratify_label"].isin(small_labels),
    "safe_stratify"
] = "SMALL_GROUP"


# ============================================================
# FIRST SPLIT
# 70% TRAIN
# 30% TEMPORARY
# ============================================================

print("\n" + "=" * 70)
print("CREATING TRAIN / TEMPORARY SPLIT")
print("=" * 70)

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=df["safe_stratify"]
)

print(f"Training images : {len(train_df)}")
print(f"Temporary images: {len(temp_df)}")


# ============================================================
# SECOND SPLIT
# 15% VALIDATION
# 15% TEST
# ============================================================

print("\n" + "=" * 70)
print("CREATING VALIDATION / TEST SPLIT")
print("=" * 70)

# The temporary set is 30% of the total.
# Split it equally:
#
# 50% of 30% = 15% validation
# 50% of 30% = 15% test

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["safe_stratify"]
)

print(f"Validation images: {len(validation_df)}")
print(f"Test images      : {len(test_df)}")


# ============================================================
# REMOVE TEMPORARY COLUMNS
# ============================================================

columns_to_remove = [
    "stratify_label",
    "safe_stratify"
]

train_df = train_df.drop(
    columns=columns_to_remove,
    errors="ignore"
)

validation_df = validation_df.drop(
    columns=columns_to_remove,
    errors="ignore"
)

test_df = test_df.drop(
    columns=columns_to_remove,
    errors="ignore"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE SPLITS
# ============================================================

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

validation_df.to_csv(
    VAL_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(f"Train      : {TRAIN_FILE}")
print(f"Validation : {VAL_FILE}")
print(f"Test       : {TEST_FILE}")


# ============================================================
# VERIFY TOTAL COUNT
# ============================================================

print("\n" + "=" * 70)
print("SPLIT COUNT VERIFICATION")
print("=" * 70)

total_split_images = (
    len(train_df)
    + len(validation_df)
    + len(test_df)
)

print(f"Original images : {len(df)}")
print(f"Train images    : {len(train_df)}")
print(f"Validation     : {len(validation_df)}")
print(f"Test            : {len(test_df)}")
print(f"Total split     : {total_split_images}")

if total_split_images == len(df):
    print("\n✓ All images are accounted for")
else:
    print("\n✗ ERROR: Image count mismatch")


# ============================================================
# CHECK IMAGE ID OVERLAP
# ============================================================

print("\n" + "=" * 70)
print("CHECKING DATA LEAKAGE")
print("=" * 70)

train_ids = set(train_df["image_id"])
val_ids = set(validation_df["image_id"])
test_ids = set(test_df["image_id"])

train_val_overlap = train_ids.intersection(val_ids)
train_test_overlap = train_ids.intersection(test_ids)
val_test_overlap = val_ids.intersection(test_ids)

print(f"Train × Validation overlap: {len(train_val_overlap)}")
print(f"Train × Test overlap      : {len(train_test_overlap)}")
print(f"Validation × Test overlap : {len(val_test_overlap)}")

if (
    len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
):
    print("\n✓ NO DATA LEAKAGE DETECTED")
else:
    print("\n✗ DATA LEAKAGE DETECTED")


# ============================================================
# CHECK IMAGE PATH OVERLAP
# ============================================================

print("\n" + "=" * 70)
print("CHECKING IMAGE PATH OVERLAP")
print("=" * 70)

train_paths = set(train_df["image_path"])
val_paths = set(validation_df["image_path"])
test_paths = set(test_df["image_path"])

path_overlap = (
    len(train_paths.intersection(val_paths))
    + len(train_paths.intersection(test_paths))
    + len(val_paths.intersection(test_paths))
)

print(f"Overlapping image paths: {path_overlap}")

if path_overlap == 0:
    print("✓ No image path overlap")
else:
    print("✗ Image path overlap detected")


# ============================================================
# ANIMAL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ANIMAL DISTRIBUTION BY SPLIT")
print("=" * 70)

animal_split = pd.DataFrame({
    "Train": train_df["animal"].value_counts(),
    "Validation": validation_df["animal"].value_counts(),
    "Test": test_df["animal"].value_counts()
}).fillna(0).astype(int)

print(animal_split)


# ============================================================
# DISEASE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("DISEASE DISTRIBUTION BY SPLIT")
print("=" * 70)

disease_split = pd.DataFrame({
    "Train": train_df["disease"].value_counts(),
    "Validation": validation_df["disease"].value_counts(),
    "Test": test_df["disease"].value_counts()
}).fillna(0).astype(int)

print(disease_split.to_string())


# ============================================================
# ANIMAL × DISEASE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ANIMAL × DISEASE DISTRIBUTION BY SPLIT")
print("=" * 70)

for split_name, split_df in [
    ("TRAIN", train_df),
    ("VALIDATION", validation_df),
    ("TEST", test_df)
]:

    print(f"\n--- {split_name} ---")

    distribution = pd.crosstab(
        split_df["animal"],
        split_df["disease"]
    )

    print(distribution)


# ============================================================
# PERCENTAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL SPLIT PERCENTAGES")
print("=" * 70)

train_percentage = (
    len(train_df) / len(df) * 100
)

validation_percentage = (
    len(validation_df) / len(df) * 100
)

test_percentage = (
    len(test_df) / len(df) * 100
)

print(f"Train      : {train_percentage:.2f}%")
print(f"Validation : {validation_percentage:.2f}%")
print(f"Test       : {test_percentage:.2f}%")


# ============================================================
# FINAL VERDICT
# ============================================================

print("\n" + "=" * 70)
print("FINAL SPLIT VERIFICATION")
print("=" * 70)

all_checks_passed = True

if total_split_images != len(df):
    all_checks_passed = False

if train_val_overlap:
    all_checks_passed = False

if train_test_overlap:
    all_checks_passed = False

if val_test_overlap:
    all_checks_passed = False

if path_overlap != 0:
    all_checks_passed = False


if all_checks_passed:

    print("✓ DATASET SPLIT COMPLETED SUCCESSFULLY")
    print("✓ No image loss")
    print("✓ No duplicate image IDs across splits")
    print("✓ No image path overlap")
    print("✓ Train / validation / test datasets are ready")

else:

    print("✗ DATASET SPLIT VERIFICATION FAILED")
    print("Please investigate the errors above.")


print("\n" + "=" * 70)
print("DATASET SPLITTING COMPLETE")
print("=" * 70)