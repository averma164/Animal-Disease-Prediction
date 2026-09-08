import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "image_master.csv"
OUTPUT_FILE = BASE_DIR / "image_master_cleaned.csv"
REPORT_FILE = BASE_DIR / "label_cleaning_report.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING IMAGE MASTER DATA")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Total records: {len(df)}")


# ============================================================
# ORIGINAL LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL DISEASE LABELS")
print("=" * 70)

original_counts = df["disease"].value_counts()

print(original_counts)


# ============================================================
# LABEL STANDARDIZATION
# ============================================================

print("\n" + "=" * 70)
print("CLEANING DISEASE LABELS")
print("=" * 70)

# Keep original disease label for tracking
df["original_disease"] = df["disease"]


# ------------------------------------------------------------
# 1. Fix Ringworm capitalization inconsistency
# ------------------------------------------------------------

df["disease"] = df["disease"].replace({
    "ringworm": "Ringworm"
})

print("✓ ringworm → Ringworm")


# ------------------------------------------------------------
# 2. Standardize common formatting
# ------------------------------------------------------------

# Remove unnecessary spaces
df["disease"] = df["disease"].astype(str).str.strip()

# Replace multiple spaces with a single space
df["disease"] = df["disease"].str.replace(
    r"\s+",
    " ",
    regex=True
)


# ============================================================
# MASTITIS NORMAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("SPECIAL LABEL CHECK")
print("=" * 70)

mastitis_normal_count = (
    df["disease"] == "mastitis normal"
).sum()

print(f"'mastitis normal' images: {mastitis_normal_count}")

if mastitis_normal_count > 0:
    print("\n⚠ WARNING:")
    print("'mastitis normal' has been KEPT unchanged.")
    print("These images should be manually inspected before")
    print("deciding whether they are Mastitis or Healthy.")


# ============================================================
# CREATE CLEANING REPORT
# ============================================================

print("\n" + "=" * 70)
print("LABEL CLEANING REPORT")
print("=" * 70)

report = (
    df.groupby(["original_disease", "disease"])
    .size()
    .reset_index(name="image_count")
)

print(report.to_string(index=False))


# ============================================================
# REMOVE TEMPORARY COLUMN
# ============================================================

df = df.drop(columns=["original_disease"])


# ============================================================
# SAVE CLEANED DATASET
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)

report.to_csv(REPORT_FILE, index=False)

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(f"Cleaned dataset : {OUTPUT_FILE}")
print(f"Cleaning report : {REPORT_FILE}")


# ============================================================
# FINAL DISEASE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLEANED DISEASE DISTRIBUTION")
print("=" * 70)

cleaned_counts = df["disease"].value_counts()

print(cleaned_counts)


# ============================================================
# CHECK FOR DUPLICATE LABELS
# ============================================================

print("\n" + "=" * 70)
print("FINAL LABEL CHECK")
print("=" * 70)

print(f"Total disease classes: {df['disease'].nunique()}")

print("\nDisease classes:")

for disease in sorted(df["disease"].unique()):
    print(f" - {disease}")


# ============================================================
# DATA INTEGRITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("DATA INTEGRITY CHECK")
print("=" * 70)

print(f"Original records : {len(original_counts)} labels")
print(f"Final records    : {len(df)} images")

if len(df) == 9595:
    print("✓ Image count preserved: 9595")
else:
    print("⚠ WARNING: Image count changed!")

if df["animal"].isna().sum() == 0:
    print("✓ No missing animal labels")
else:
    print("⚠ Missing animal labels found")

if df["disease"].isna().sum() == 0:
    print("✓ No missing disease labels")
else:
    print("⚠ Missing disease labels found")


print("\n" + "=" * 70)
print("LABEL CLEANING COMPLETE")
print("=" * 70)