import pandas as pd
from pathlib import Path
from PIL import Image
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"D:\Animal disease Prediction")

# ACTUAL IMAGE LOCATION
RAW_DATA_DIR = BASE_DIR / "rawdata"

# CSV SPLITS
SPLIT_DIR = BASE_DIR / "dataset_split"

TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV = SPLIT_DIR / "validation.csv"
TEST_CSV = SPLIT_DIR / "test.csv"

# OUTPUT
OUTPUT_DIR = BASE_DIR / "prepared_dataset"

TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "validation"
TEST_DIR = OUTPUT_DIR / "test"

REPORT_FILE = BASE_DIR / "dataset_preparation_report.csv"

# Copy actual images
COPY_IMAGES = True


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("DATASET IMAGE PREPARATION")
print("=" * 70)


# ============================================================
# CHECK RAW DATA DIRECTORY
# ============================================================

print("\n" + "=" * 70)
print("CHECKING RAW DATA DIRECTORY")
print("=" * 70)

print(f"Raw data directory:")
print(RAW_DATA_DIR)

if not RAW_DATA_DIR.exists():

    print("\nERROR: Raw data directory does not exist!")
    print(RAW_DATA_DIR)
    raise SystemExit(1)

print("✓ Raw data directory found")


# ============================================================
# LOAD CSV FILES
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASET SPLITS")
print("=" * 70)


for file in [TRAIN_CSV, VAL_CSV, TEST_CSV]:

    if not file.exists():

        print("\nERROR: CSV file not found:")
        print(file)

        raise SystemExit(1)


train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)


print(f"Train images      : {len(train_df)}")
print(f"Validation images : {len(val_df)}")
print(f"Test images       : {len(test_df)}")


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

print("\n" + "=" * 70)
print("CREATING PREPARED DATASET DIRECTORIES")
print("=" * 70)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print(f"Output directory:")
print(OUTPUT_DIR)


# ============================================================
# IMAGE PATH RESOLVER
# ============================================================

def resolve_image_path(image_path):

    """
    CSV contains paths like:

    Cat/Dental Disease/Image_10.jpg

    Actual image:

    D:/Animal disease Prediction/rawdata/
        Cat/Dental Disease/Image_10.jpg
    """

    relative_path = Path(str(image_path))

    full_path = RAW_DATA_DIR / relative_path

    if full_path.exists():

        return full_path

    return None


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(image_path):

    try:

        # First verify the file
        with Image.open(image_path) as img:

            img.verify()

        # Open again to get information
        with Image.open(image_path) as img:

            width, height = img.size
            image_format = img.format

        return (
            True,
            width,
            height,
            image_format,
            ""
        )

    except Exception as error:

        return (
            False,
            None,
            None,
            None,
            str(error)
        )


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    dataframe,
    split_name,
    output_directory
):

    print("\n" + "=" * 70)
    print(f"PROCESSING {split_name.upper()} DATASET")
    print("=" * 70)

    valid_count = 0
    missing_count = 0
    corrupted_count = 0
    copied_count = 0

    records = []

    total = len(dataframe)

    for index, row in dataframe.iterrows():

        image_id = str(row["image_id"])
        image_path = str(row["image_path"])
        animal = str(row["animal"])
        disease = str(row["disease"])

        # ----------------------------------------------------
        # FIND IMAGE
        # ----------------------------------------------------

        source_path = resolve_image_path(image_path)

        if source_path is None:

            missing_count += 1

            records.append({

                "image_id": image_id,

                "image_path": image_path,

                "resolved_path": "",

                "animal": animal,

                "disease": disease,

                "split": split_name,

                "status": "missing",

                "width": "",

                "height": "",

                "format": "",

                "error": "Image file not found"

            })

            continue

        # ----------------------------------------------------
        # VALIDATE IMAGE
        # ----------------------------------------------------

        (
            valid,
            width,
            height,
            image_format,
            error
        ) = validate_image(source_path)

        if not valid:

            corrupted_count += 1

            records.append({

                "image_id": image_id,

                "image_path": image_path,

                "resolved_path": str(source_path),

                "animal": animal,

                "disease": disease,

                "split": split_name,

                "status": "corrupted",

                "width": "",

                "height": "",

                "format": "",

                "error": error

            })

            continue

        valid_count += 1

        # ----------------------------------------------------
        # CREATE FOLDER
        # ----------------------------------------------------

        safe_animal = animal.replace(
            "/",
            "_"
        ).replace(
            "\\",
            "_"
        )

        safe_disease = disease.replace(
            "/",
            "_"
        ).replace(
            "\\",
            "_"
        )

        disease_directory = (
            output_directory
            / safe_animal
            / safe_disease
        )

        disease_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # DESTINATION
        # ----------------------------------------------------

        extension = source_path.suffix.lower()

        destination_path = (
            disease_directory
            / f"{image_id}{extension}"
        )

        # ----------------------------------------------------
        # COPY IMAGE
        # ----------------------------------------------------

        if COPY_IMAGES:

            try:

                shutil.copy2(
                    source_path,
                    destination_path
                )

                copied_count += 1

            except Exception as error:

                print(
                    f"WARNING: Could not copy "
                    f"{image_id}: {error}"
                )

        # ----------------------------------------------------
        # SAVE RECORD
        # ----------------------------------------------------

        records.append({

            "image_id": image_id,

            "image_path": image_path,

            "resolved_path": str(source_path),

            "animal": animal,

            "disease": disease,

            "split": split_name,

            "status": "valid",

            "width": width,

            "height": height,

            "format": image_format,

            "error": ""

        })

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (index + 1) % 500 == 0:

            print(
                f"Processed: "
                f"{index + 1}/{total}"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "-" * 70)
    print(f"{split_name.upper()} SUMMARY")
    print("-" * 70)

    print(f"Total images     : {total}")
    print(f"Valid images     : {valid_count}")
    print(f"Missing images   : {missing_count}")
    print(f"Corrupted images : {corrupted_count}")

    if COPY_IMAGES:

        print(f"Images copied    : {copied_count}")

    return records


# ============================================================
# PROCESS TRAIN
# ============================================================

train_records = process_split(
    train_df,
    "train",
    TRAIN_DIR
)


# ============================================================
# PROCESS VALIDATION
# ============================================================

val_records = process_split(
    val_df,
    "validation",
    VAL_DIR
)


# ============================================================
# PROCESS TEST
# ============================================================

test_records = process_split(
    test_df,
    "test",
    TEST_DIR
)


# ============================================================
# COMBINE REPORTS
# ============================================================

all_records = (
    train_records
    + val_records
    + test_records
)


report_df = pd.DataFrame(
    all_records
)


# ============================================================
# SAVE REPORT
# ============================================================

print("\n" + "=" * 70)
print("CREATING PREPARATION REPORT")
print("=" * 70)


report_df.to_csv(
    REPORT_FILE,
    index=False
)


print("Report saved:")
print(REPORT_FILE)


# ============================================================
# OVERALL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("OVERALL DATASET SUMMARY")
print("=" * 70)


total_images = len(report_df)

valid_images = (
    report_df["status"] == "valid"
).sum()

missing_images = (
    report_df["status"] == "missing"
).sum()

corrupted_images = (
    report_df["status"] == "corrupted"
).sum()


print(f"Total images     : {total_images}")
print(f"Valid images     : {valid_images}")
print(f"Missing images   : {missing_images}")
print(f"Corrupted images : {corrupted_images}")


# ============================================================
# IMAGE VALIDATION CHECK
# ============================================================

print("\n" + "=" * 70)
print("IMAGE VALIDATION")
print("=" * 70)


if missing_images == 0:

    print("✓ No missing images")

else:

    print(
        f"✗ {missing_images} missing images found"
    )


if corrupted_images == 0:

    print("✓ No corrupted images")

else:

    print(
        f"✗ {corrupted_images} corrupted images found"
    )


# ============================================================
# TOTAL IMAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("TOTAL IMAGE CHECK")
print("=" * 70)


expected_total = (
    len(train_df)
    + len(val_df)
    + len(test_df)
)


print(f"Expected images : {expected_total}")
print(f"Processed images: {total_images}")


if total_images == expected_total:

    print(
        f"✓ Total image count is correct: "
        f"{total_images}"
    )

else:

    print("✗ Image count mismatch")


# ============================================================
# SPLIT VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("SPLIT VALIDATION")
print("=" * 70)


train_valid = sum(
    1
    for r in train_records
    if r["status"] == "valid"
)

val_valid = sum(
    1
    for r in val_records
    if r["status"] == "valid"
)

test_valid = sum(
    1
    for r in test_records
    if r["status"] == "valid"
)


print(f"Train valid      : {train_valid}")
print(f"Validation valid : {val_valid}")
print(f"Test valid       : {test_valid}")

print(
    f"Total valid      : "
    f"{train_valid + val_valid + test_valid}"
)


# ============================================================
# PREPARED DIRECTORY CHECK
# ============================================================

print("\n" + "=" * 70)
print("PREPARED DIRECTORY CHECK")
print("=" * 70)


print("Train directory:")
print(TRAIN_DIR)

print("\nValidation directory:")
print(VAL_DIR)

print("\nTest directory:")
print(TEST_DIR)


# ============================================================
# COPIED IMAGE CHECK
# ============================================================

if COPY_IMAGES:

    print("\n" + "=" * 70)
    print("COPIED IMAGE CHECK")
    print("=" * 70)

    print(f"Train copied      : {train_valid}")
    print(f"Validation copied : {val_valid}")
    print(f"Test copied       : {test_valid}")

    total_copied = (
        train_valid
        + val_valid
        + test_valid
    )

    print(f"Total copied      : {total_copied}")


# ============================================================
# DISEASE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREPARED DISEASE DISTRIBUTION")
print("=" * 70)


print(
    report_df[
        report_df["status"] == "valid"
    ]["disease"].value_counts()
)


# ============================================================
# ANIMAL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREPARED ANIMAL DISTRIBUTION")
print("=" * 70)


print(
    report_df[
        report_df["status"] == "valid"
    ]["animal"].value_counts()
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL PREPARATION VERIFICATION")
print("=" * 70)


if (
    missing_images == 0
    and corrupted_images == 0
    and total_images == expected_total
    and train_valid == len(train_df)
    and val_valid == len(val_df)
    and test_valid == len(test_df)
):

    print("✓ DATASET PREPARATION PASSED")
    print("✓ All images were found")
    print("✓ All images are readable")
    print("✓ No corrupted images detected")
    print("✓ All train images prepared")
    print("✓ All validation images prepared")
    print("✓ All test images prepared")
    print("✓ Dataset is ready for the next stage")

else:

    print("✗ DATASET PREPARATION FAILED")
    print("Please inspect the preparation report.")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("DATASET PREPARATION COMPLETE")
print("=" * 70)