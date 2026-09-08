import pandas as pd
from pathlib import Path

# ============================================================
# 03 LABEL ANALYSIS
# Animal Disease Prediction Project
# ============================================================

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_FILE = BASE_DIR / "image_master.csv"

# -----------------------------
# LOAD DATA
# -----------------------------
print("=" * 70)
print("LOADING IMAGE MASTER DATA")
print("=" * 70)

df = pd.read_csv(CSV_FILE)

print(f"Total records: {len(df)}")
print()

# ============================================================
# 1. DISEASE DISTRIBUTION
# ============================================================

print("=" * 70)
print("DISEASE DISTRIBUTION")
print("=" * 70)

disease_counts = df["disease"].value_counts()

print(disease_counts)

# Save disease distribution
disease_counts_df = disease_counts.reset_index()
disease_counts_df.columns = ["disease", "image_count"]

output_file = BASE_DIR / "disease_distribution.csv"
disease_counts_df.to_csv(output_file, index=False)

print()
print(f"Saved: {output_file}")

# ============================================================
# 2. ANIMAL DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("ANIMAL DISTRIBUTION")
print("=" * 70)

animal_counts = df["animal"].value_counts()

print(animal_counts)

# ============================================================
# 3. ANIMAL × DISEASE DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("ANIMAL × DISEASE DISTRIBUTION")
print("=" * 70)

animal_disease = pd.crosstab(
    df["animal"],
    df["disease"]
)

print(animal_disease)

# Save cross-tabulation
animal_disease.to_csv(
    BASE_DIR / "animal_disease_distribution.csv"
)

print()
print(f"Saved: {BASE_DIR / 'animal_disease_distribution.csv'}")

# ============================================================
# 4. SMALL DISEASE CLASSES
# ============================================================

print()
print("=" * 70)
print("DISEASE CLASSES WITH FEWER THAN 50 IMAGES")
print("=" * 70)

small_classes = disease_counts[disease_counts < 50]

if len(small_classes) == 0:
    print("None")
else:
    print(small_classes)

# ============================================================
# 5. VERY SMALL CLASSES
# ============================================================

print()
print("=" * 70)
print("DISEASE CLASSES WITH FEWER THAN 20 IMAGES")
print("=" * 70)

very_small_classes = disease_counts[disease_counts < 20]

if len(very_small_classes) == 0:
    print("None")
else:
    print(very_small_classes)

# ============================================================
# 6. ANIMAL × DISEASE COMBINATIONS WITH FEW IMAGES
# ============================================================

print()
print("=" * 70)
print("ANIMAL × DISEASE COMBINATIONS WITH FEWER THAN 20 IMAGES")
print("=" * 70)

combination_counts = (
    df.groupby(["animal", "disease"])
    .size()
    .reset_index(name="image_count")
)

rare_combinations = combination_counts[
    combination_counts["image_count"] < 20
]

if len(rare_combinations) == 0:
    print("None")
else:
    print(rare_combinations.to_string(index=False))

# ============================================================
# 7. CHECK FOR SUSPICIOUS LABELS
# ============================================================

print()
print("=" * 70)
print("SUSPICIOUS / POTENTIALLY INCONSISTENT LABELS")
print("=" * 70)

# Print all unique labels
print("Unique disease labels:")
for disease in sorted(df["disease"].unique()):
    print(" -", disease)

# ============================================================
# 8. CHECK DISEASES PER ANIMAL
# ============================================================

print()
print("=" * 70)
print("DISEASES FOUND FOR EACH ANIMAL")
print("=" * 70)

for animal in sorted(df["animal"].unique()):

    diseases = sorted(
        df.loc[df["animal"] == animal, "disease"].unique()
    )

    print()
    print(f"{animal}:")
    
    for disease in diseases:
        count = len(
            df[
                (df["animal"] == animal) &
                (df["disease"] == disease)
            ]
        )

        print(f"   {disease}: {count}")

# ============================================================
# 9. CHECK FOR EMPTY / UNKNOWN LABELS
# ============================================================

print()
print("=" * 70)
print("EMPTY OR UNKNOWN LABEL CHECK")
print("=" * 70)

missing_animal = df["animal"].isna().sum()
missing_disease = df["disease"].isna().sum()

unknown_animal = (
    df["animal"]
    .astype(str)
    .str.lower()
    .isin(["unknown", "none", "nan", ""])
    .sum()
)

unknown_disease = (
    df["disease"]
    .astype(str)
    .str.lower()
    .isin(["unknown", "none", "nan", ""])
    .sum()
)

print(f"Missing animal labels: {missing_animal}")
print(f"Missing disease labels: {missing_disease}")
print(f"Unknown animal labels: {unknown_animal}")
print(f"Unknown disease labels: {unknown_disease}")

# ============================================================
# 10. DATASET SUMMARY
# ============================================================

print()
print("=" * 70)
print("LABEL ANALYSIS SUMMARY")
print("=" * 70)

print(f"Total images       : {len(df)}")
print(f"Animal classes     : {df['animal'].nunique()}")
print(f"Disease classes    : {df['disease'].nunique()}")
print(f"Small disease cls  : {len(small_classes)}")
print(f"Very small classes : {len(very_small_classes)}")

print()
print("Animal classes:")
for animal, count in animal_counts.items():
    print(f"  {animal}: {count}")

print()
print("Small disease classes:")

if len(small_classes) == 0:
    print("  None")
else:
    for disease, count in small_classes.items():
        print(f"  {disease}: {count}")

print()
print("=" * 70)
print("LABEL ANALYSIS COMPLETE")
print("=" * 70)