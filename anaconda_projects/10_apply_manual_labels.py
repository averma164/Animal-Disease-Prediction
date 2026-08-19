import os
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\Animal disease Prediction"

INPUT_CSV = os.path.join(
    BASE_DIR,
    "image_master_cleaned.csv"
)

INSPECTION_CSV = os.path.join(
    BASE_DIR,
    "mastitis_inspection.csv"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "image_master_final.csv"
)

REPORT_CSV = os.path.join(
    BASE_DIR,
    "manual_label_report.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("APPLYING MANUAL MASTITIS LABELS")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(INPUT_CSV)
inspection = pd.read_csv(INSPECTION_CSV)

print(f"Original dataset : {len(df)}")
print(f"Inspection rows  : {len(inspection)}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "image_id",
    "image_path",
    "animal",
    "original_disease",
    "manual_label"
]

missing_columns = [
    col for col in required_columns
    if col not in inspection.columns
]

if missing_columns:
    print("\nERROR: Missing columns:")
    for col in missing_columns:
        print("-", col)
    raise SystemExit


# ============================================================
# CLEAN MANUAL LABELS
# ============================================================

inspection["manual_label"] = (
    inspection["manual_label"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# CHECK FOR EMPTY LABELS
# ============================================================

empty_labels = inspection[
    inspection["manual_label"] == ""
]

if len(empty_labels) > 0:

    print("\n" + "=" * 70)
    print("ERROR: EMPTY MANUAL LABELS")
    print("=" * 70)

    print(
        f"{len(empty_labels)} images do not have a manual label."
    )

    print("\nImages requiring labels:")

    for _, row in empty_labels.iterrows():
        print(
            f"{row['image_id']} -> "
            f"{row['image_path']}"
        )

    print("\nPlease edit:")
    print(INSPECTION_CSV)

    raise SystemExit


# ============================================================
# VALID LABEL CHECK
# ============================================================

allowed_labels = {
    "Mastitis",
    "Healthy",
    "Uncertain"
}

invalid_labels = inspection[
    ~inspection["manual_label"].isin(allowed_labels)
]

if len(invalid_labels) > 0:

    print("\n" + "=" * 70)
    print("ERROR: INVALID MANUAL LABELS")
    print("=" * 70)

    for _, row in invalid_labels.iterrows():

        print(
            f"{row['image_id']} -> "
            f"{row['manual_label']}"
        )

    print("\nAllowed labels are:")
    print("Mastitis")
    print("Healthy")
    print("Uncertain")

    raise SystemExit


print("\n✓ All manual labels are valid")


# ============================================================
# SHOW MANUAL LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("MANUAL LABEL DISTRIBUTION")
print("=" * 70)

label_counts = inspection["manual_label"].value_counts()

print(label_counts)


# ============================================================
# CREATE LOOKUP
# ============================================================

label_lookup = dict(
    zip(
        inspection["image_id"],
        inspection["manual_label"]
    )
)


# ============================================================
# APPLY LABELS
# ============================================================

print("\n" + "=" * 70)
print("APPLYING LABELS")
print("=" * 70)

changed = 0

for index, row in df.iterrows():

    image_id = row["image_id"]

    if image_id in label_lookup:

        new_label = label_lookup[image_id]

        old_label = row["disease"]

        if old_label != new_label:

            df.at[index, "disease"] = new_label

            changed += 1

            print(
                f"{image_id}: "
                f"{old_label} -> {new_label}"
            )


print(f"\nLabels changed: {changed}")


# ============================================================
# HANDLE UNCERTAIN IMAGES
# ============================================================

print("\n" + "=" * 70)
print("HANDLING UNCERTAIN IMAGES")
print("=" * 70)

uncertain_ids = set(
    inspection.loc[
        inspection["manual_label"] == "Uncertain",
        "image_id"
    ]
)

print(
    f"Uncertain images: {len(uncertain_ids)}"
)

if len(uncertain_ids) > 0:

    print("\nThese images will be REMOVED from the final dataset:")

    for image_id in uncertain_ids:
        print("-", image_id)

    df = df[
        ~df["image_id"].isin(uncertain_ids)
    ].copy()

else:

    print("✓ No uncertain images")


# ============================================================
# CHECK REMAINING MASTITIS NORMAL
# ============================================================

remaining = df[
    df["disease"].astype(str).str.strip()
    == "mastitis normal"
]

print("\nRemaining 'mastitis normal' images:", len(remaining))

if len(remaining) > 0:

    print("\nERROR:")
    print("Some 'mastitis normal' labels remain.")

    raise SystemExit


# ============================================================
# SAVE FINAL DATASET
# ============================================================

print("\n" + "=" * 70)
print("SAVING FINAL DATASET")
print("=" * 70)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("Saved:")
print(OUTPUT_CSV)


# ============================================================
# CREATE REPORT
# ============================================================

report_rows = []

for label, count in label_counts.items():

    report_rows.append({
        "manual_label": label,
        "image_count": count
    })

report_df = pd.DataFrame(report_rows)

report_df.to_csv(
    REPORT_CSV,
    index=False
)

print("\nReport saved:")
print(REPORT_CSV)


# ============================================================
# FINAL DISEASE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("FINAL DISEASE DISTRIBUTION")
print("=" * 70)

disease_counts = (
    df["disease"]
    .value_counts()
)

print(disease_counts)


# ============================================================
# FINAL ANIMAL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("FINAL ANIMAL DISTRIBUTION")
print("=" * 70)

animal_counts = (
    df["animal"]
    .value_counts()
)

print(animal_counts)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET CHECK")
print("=" * 70)

print("Original images :", len(pd.read_csv(INPUT_CSV)))
print("Final images    :", len(df))

print(
    "Disease classes :",
    df["disease"].nunique()
)

print(
    "Animal classes  :",
    df["animal"].nunique()
)

print(
    "\nRemaining 'mastitis normal':",
    (
        df["disease"] == "mastitis normal"
    ).sum()
)


# ============================================================
# SUCCESS
# ============================================================

if (
    len(df) > 0
    and
    (
        df["disease"] == "mastitis normal"
    ).sum() == 0
):

    print("\n" + "=" * 70)
    print("✓ MANUAL LABEL PROCESSING COMPLETE")
    print("=" * 70)

    print("""
✓ mastitis normal has been removed
✓ Manual labels have been applied
✓ Final dataset created
✓ No uncertain images remain
✓ Dataset is ready for the next stage
""")

else:

    print("\n" + "=" * 70)
    print("✗ FINAL DATASET CHECK FAILED")
    print("=" * 70)