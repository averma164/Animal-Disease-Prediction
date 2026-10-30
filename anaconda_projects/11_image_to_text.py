import pandas as pd
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# CSV file
csv_file = BASE_DIR / "image_master_final.csv"

# Folder where text files will be created
output_folder = BASE_DIR / "image_text_labels"
output_folder.mkdir(exist_ok=True)

# Read CSV
df = pd.read_csv(csv_file)

count = 0

for _, row in df.iterrows():

    image_id = row["image_id"]
    image_path = row["image_path"]
    animal = row["animal"]
    disease = row["disease"]

    # Create txt filename using image ID
    txt_file = output_folder / f"{image_id}.txt"

    # Write information about the image
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"Image ID: {image_id}\n")
        f.write(f"Image Path: {image_path}\n")
        f.write(f"Animal: {animal}\n")
        f.write(f"Disease: {disease}\n")

    count += 1

print(f"Created {count} text files.")
print(f"Text files are saved in: {output_folder}")