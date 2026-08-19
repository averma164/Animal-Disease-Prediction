import os
import pandas as pd
from PIL import Image

# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"D:\Animal disease Prediction"

IMAGE_MASTER = os.path.join(
    BASE_DIR,
    "image_master.csv"
)

RAW_DATA_DIR = os.path.join(
    BASE_DIR,
    "rawdata"
)


# ============================================================
# 2. LOAD IMAGE MASTER CSV
# ============================================================

print("=" * 60)
print("DATASET AUDIT")
print("=" * 60)

if not os.path.exists(IMAGE_MASTER):

    print("\nERROR: image_master.csv was not found.")
    print("Expected location:")
    print(IMAGE_MASTER)

    exit()


df = pd.read_csv(IMAGE_MASTER)


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("\nTotal images:")
print(len(df))


print("\n----------------------------------------")
print("COLUMNS")
print("----------------------------------------")

print(df.columns.tolist())


# ============================================================
# 4. IMAGES BY ANIMAL
# ============================================================

print("\n----------------------------------------")
print("IMAGES BY ANIMAL")
print("----------------------------------------")

print(
    df["animal"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 5. IMAGES BY DISEASE
# ============================================================

print("\n----------------------------------------")
print("IMAGES BY DISEASE")
print("----------------------------------------")

print(
    df["disease"]
    .value_counts()
)


# ============================================================
# 6. ANIMAL × DISEASE
# ============================================================

print("\n----------------------------------------")
print("ANIMAL × DISEASE")
print("----------------------------------------")

animal_disease = pd.crosstab(
    df["animal"],
    df["disease"]
)

print(animal_disease)


# ============================================================
# 7. DUPLICATE IMAGE PATHS
# ============================================================

print("\n----------------------------------------")
print("DUPLICATE IMAGE PATHS")
print("----------------------------------------")

duplicate_paths = df[
    df["image_path"].duplicated(keep=False)
]

print(
    "Duplicate path rows:",
    len(duplicate_paths)
)

if len(duplicate_paths) > 0:

    print("\nDuplicate paths:")

    print(
        duplicate_paths[
            ["image_path", "animal", "disease"]
        ]
    )


# ============================================================
# 8. DUPLICATE IMAGE IDs
# ============================================================

if "image_id" in df.columns:

    print("\n----------------------------------------")
    print("DUPLICATE IMAGE IDs")
    print("----------------------------------------")

    duplicate_ids = df[
        df["image_id"].duplicated(keep=False)
    ]

    print(
        "Duplicate image ID rows:",
        len(duplicate_ids)
    )


# ============================================================
# 9. MISSING VALUES
# ============================================================

print("\n----------------------------------------")
print("MISSING VALUES")
print("----------------------------------------")

print(
    df.isnull().sum()
)


# ============================================================
# 10. IMAGE STATUS
# ============================================================

print("\n----------------------------------------")
print("IMAGE STATUS")
print("----------------------------------------")

print(
    df["status"]
    .value_counts()
)


# ============================================================
# 11. IMAGE FORMATS
# ============================================================

print("\n----------------------------------------")
print("IMAGE FORMATS")
print("----------------------------------------")

print(
    df["format"]
    .value_counts()
)


# ============================================================
# 12. IMAGE DIMENSIONS
# ============================================================

print("\n----------------------------------------")
print("IMAGE DIMENSIONS")
print("----------------------------------------")

print("\nWidth:")

print(
    df["width"].describe()
)

print("\nHeight:")

print(
    df["height"].describe()
)


# ============================================================
# 13. VERY SMALL DISEASE CLASSES
# ============================================================

print("\n----------------------------------------")
print("CLASSES WITH FEWER THAN 50 IMAGES")
print("----------------------------------------")

disease_counts = df["disease"].value_counts()

small_classes = disease_counts[
    disease_counts < 50
]

if len(small_classes) == 0:

    print("No disease classes have fewer than 50 images.")

else:

    print(small_classes)


# ============================================================
# 14. ANIMALS WITH FEWER THAN 100 IMAGES
# ============================================================

print("\n----------------------------------------")
print("ANIMAL CLASSES")
print("----------------------------------------")

animal_counts = df["animal"].value_counts()

print(animal_counts)

print("\nAnimals with fewer than 100 images:")

small_animals = animal_counts[
    animal_counts < 100
]

if len(small_animals) == 0:

    print("None")

else:

    print(small_animals)


# ============================================================
# 15. CHECK ACTUAL IMAGE FILES
# ============================================================

print("\n----------------------------------------")
print("CHECKING IMAGE FILES")
print("----------------------------------------")

missing_files = []
unreadable_files = []

for index, row in df.iterrows():

    image_path = row["image_path"]

    full_path = os.path.join(
        RAW_DATA_DIR,
        image_path
    )

    # ----------------------------------------
    # Check if file exists
    # ----------------------------------------

    if not os.path.exists(full_path):

        missing_files.append(
            image_path
        )

        continue


    # ----------------------------------------
    # Try opening image
    # ----------------------------------------

    try:

        with Image.open(full_path) as img:

            img.verify()

    except Exception:

        unreadable_files.append(
            image_path
        )


print(
    "Missing image files:",
    len(missing_files)
)

print(
    "Unreadable image files:",
    len(unreadable_files)
)


# ============================================================
# 16. DISPLAY MISSING FILES
# ============================================================

if len(missing_files) > 0:

    print("\nMissing files:")

    for path in missing_files[:20]:

        print(path)

    if len(missing_files) > 20:

        print(
            "... and",
            len(missing_files) - 20,
            "more"
        )


# ============================================================
# 17. DISPLAY UNREADABLE FILES
# ============================================================

if len(unreadable_files) > 0:

    print("\nUnreadable files:")

    for path in unreadable_files[:20]:

        print(path)

    if len(unreadable_files) > 20:

        print(
            "... and",
            len(unreadable_files) - 20,
            "more"
        )


# ============================================================
# 18. CHECK ZERO / INVALID DIMENSIONS
# ============================================================

print("\n----------------------------------------")
print("INVALID IMAGE DIMENSIONS")
print("----------------------------------------")

invalid_dimensions = df[
    (df["width"] <= 0) |
    (df["height"] <= 0)
]

print(
    "Images with invalid dimensions:",
    len(invalid_dimensions)
)


# ============================================================
# 19. SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("AUDIT SUMMARY")
print("=" * 60)

print(
    "Total images:",
    len(df)
)

print(
    "Animals:",
    df["animal"].nunique()
)

print(
    "Disease classes:",
    df["disease"].nunique()
)

print(
    "Duplicate paths:",
    df["image_path"].duplicated().sum()
)

print(
    "Missing CSV values:",
    df.isnull().sum().sum()
)

print(
    "Missing image files:",
    len(missing_files)
)

print(
    "Unreadable images:",
    len(unreadable_files)
)

print(
    "Invalid dimensions:",
    len(invalid_dimensions)
)


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)