import os
import pandas as pd
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\Animal disease Prediction"

CLEANED_CSV = os.path.join(
    BASE_DIR,
    "image_master_cleaned.csv"
)

INSPECTION_DIR = os.path.join(
    BASE_DIR,
    "mastitis_inspection"
)

INSPECTION_CSV = os.path.join(
    BASE_DIR,
    "mastitis_inspection.csv"
)

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("MASTITIS NORMAL IMAGE INSPECTION")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING CLEANED DATASET")
print("=" * 70)

df = pd.read_csv(CLEANED_CSV)

mastitis_df = df[
    df["disease"].astype(str).str.strip() == "mastitis normal"
].copy()

print(f"Total dataset images : {len(df)}")
print(f"'mastitis normal'    : {len(mastitis_df)}")

if len(mastitis_df) == 0:
    print("\n✓ No 'mastitis normal' images found.")
    print("Nothing to inspect.")
    exit()

# ============================================================
# CREATE INSPECTION DIRECTORY
# ============================================================

print("\n" + "=" * 70)
print("CREATING INSPECTION DIRECTORY")
print("=" * 70)

os.makedirs(INSPECTION_DIR, exist_ok=True)

print(f"Inspection directory:")
print(INSPECTION_DIR)

# ============================================================
# CREATE INSPECTION CSV
# ============================================================

inspection_rows = []

for _, row in mastitis_df.iterrows():

    inspection_rows.append({
        "image_id": row["image_id"],
        "image_path": row["image_path"],
        "animal": row["animal"],
        "original_disease": row["disease"],
        "manual_label": "",
        "notes": ""
    })

inspection_df = pd.DataFrame(inspection_rows)

inspection_df.to_csv(
    INSPECTION_CSV,
    index=False
)

print(f"\nInspection CSV saved:")
print(INSPECTION_CSV)

# ============================================================
# COPY IMAGES
# ============================================================

print("\n" + "=" * 70)
print("COPYING IMAGES FOR MANUAL INSPECTION")
print("=" * 70)

copied = 0
missing = 0

for _, row in mastitis_df.iterrows():

    relative_path = str(row["image_path"]).replace("/", os.sep)

    source_path = os.path.join(
        BASE_DIR,
        "rawdata",
        relative_path
    )

    image_id = str(row["image_id"])

    filename = os.path.basename(source_path)

    destination_filename = f"{image_id}_{filename}"

    destination_path = os.path.join(
        INSPECTION_DIR,
        destination_filename
    )

    if os.path.exists(source_path):

        shutil.copy2(
            source_path,
            destination_path
        )

        copied += 1

        print(f"✓ {destination_filename}")

    else:

        missing += 1

        print(f"✗ Missing: {source_path}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("INSPECTION SUMMARY")
print("=" * 70)

print(f"Images requiring inspection : {len(mastitis_df)}")
print(f"Images copied               : {copied}")
print(f"Missing images              : {missing}")

print("\n" + "=" * 70)
print("WHAT TO DO NEXT")
print("=" * 70)

print("""
Open the folder:

D:\\Animal disease Prediction\\mastitis_inspection

Look at each of the 10 images.

For every image, edit:

mastitis_inspection.csv

Set 'manual_label' to exactly one of:

Mastitis
Healthy

If you are unsure about an image, use:

Uncertain

Do NOT guess.
""")

print("\n" + "=" * 70)
print("MASTITIS INSPECTION FILE CREATED")
print("=" * 70)
