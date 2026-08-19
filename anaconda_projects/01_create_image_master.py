import os
import pandas as pd
from PIL import Image

# ============================================
# PATHS
# ============================================

BASE_DIR = r"D:\Animal disease Prediction"

RAW_DATA = os.path.join(BASE_DIR, "rawdata")

OUTPUT_FILE = os.path.join(BASE_DIR, "image_master.csv")


# ============================================
# SUPPORTED IMAGE FORMATS
# ============================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif"
}


# ============================================
# STORE IMAGE INFORMATION
# ============================================

records = []

image_id = 1


# ============================================
# SCAN DATASET
# ============================================

for animal in os.listdir(RAW_DATA):

    animal_path = os.path.join(RAW_DATA, animal)

    # Ignore files
    if not os.path.isdir(animal_path):
        continue

    # Disease folders
    for disease in os.listdir(animal_path):

        disease_path = os.path.join(animal_path, disease)

        if not os.path.isdir(disease_path):
            continue

        # Images inside disease folder
        for filename in os.listdir(disease_path):

            file_path = os.path.join(disease_path, filename)

            # Ignore non-image files
            extension = os.path.splitext(filename)[1].lower()

            if extension not in IMAGE_EXTENSIONS:
                continue

            # --------------------------------
            # Default values
            # --------------------------------

            width = None
            height = None
            status = "valid"

            # --------------------------------
            # Read image information
            # --------------------------------

            try:

                with Image.open(file_path) as img:

                    width, height = img.size

                    image_format = img.format

            except Exception:

                image_format = extension.replace(".", "").upper()

                status = "corrupt/unreadable"


            # --------------------------------
            # File size
            # --------------------------------

            file_size_kb = round(
                os.path.getsize(file_path) / 1024,
                2
            )


            # --------------------------------
            # Relative path
            # --------------------------------

            relative_path = os.path.relpath(
                file_path,
                RAW_DATA
            )

            # Convert Windows \ to /
            relative_path = relative_path.replace("\\", "/")


            # --------------------------------
            # Create record
            # --------------------------------

            records.append({

                "image_id":
                    f"IMG_{image_id:05d}",

                "image_path":
                    relative_path,

                "animal":
                    animal,

                "disease":
                    disease,

                "width":
                    width,

                "height":
                    height,

                "format":
                    image_format,

                "file_size_kb":
                    file_size_kb,

                "status":
                    status

            })


            image_id += 1


# ============================================
# CREATE DATAFRAME
# ============================================

df = pd.DataFrame(records)


# ============================================
# SAVE CSV
# ============================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================
# PRINT SUMMARY
# ============================================

print("=" * 50)
print("IMAGE MASTER CREATED")
print("=" * 50)

print()

print("Total images:", len(df))

print()

print("Images by animal:")
print(df["animal"].value_counts())

print()

print("Images by disease:")
print(df["disease"].value_counts())

print()

print("Image status:")
print(df["status"].value_counts())

print()

print("Output:")
print(OUTPUT_FILE)

print()

print("CSV CREATED SUCCESSFULLY")