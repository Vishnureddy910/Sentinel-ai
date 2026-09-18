import os
import shutil
from pathlib import Path
import pandas as pd

data_dir = Path(__file__).parent / "data"
client_1_dir = data_dir / "client_1"
client_2_dir = data_dir / "client_2"

client_1_dir.mkdir(parents=True, exist_ok=True)
client_2_dir.mkdir(parents=True, exist_ok=True)

# 1. Locate Master CSV
csv_candidates = list(data_dir.rglob("Data_Entry*.csv"))
if not csv_candidates:
    raise FileNotFoundError("Could not find Data_Entry_2017.csv inside the data directory.")
master_csv_path = csv_candidates[0]
print(f"Using master CSV: {master_csv_path}")
df = pd.read_csv(master_csv_path)

# 2. Find all images
print("Scanning for extracted images...")
all_images = [p for p in data_dir.rglob("*.png") if "client_1" not in str(p) and "client_2" not in str(p)]
print(f"Found {len(all_images)} images to partition.")

# 3. Move images 50/50
client_1_names = set()
client_2_names = set()

for i, img_path in enumerate(all_images):
    try:
        if i % 2 == 0:
            shutil.move(str(img_path), str(client_1_dir / img_path.name))
            client_1_names.add(img_path.name)
        else:
            shutil.move(str(img_path), str(client_2_dir / img_path.name))
            client_2_names.add(img_path.name)
    except Exception:
        pass

print(f"Images moved: {len(client_1_names)} to client_1, {len(client_2_names)} to client_2.")

# 4. Generate partitioned CSVs
image_col = "Image Index" if "Image Index" in df.columns else df.columns[0]

df_client_1 = df[df[image_col].isin(client_1_names)]
df_client_2 = df[df[image_col].isin(client_2_names)]

# Save as Data_Entry_2017.csv and metadata.csv in case your code expects either name
for folder, client_df in [(client_1_dir, df_client_1), (client_2_dir, df_client_2)]:
    client_df.to_csv(folder / "Data_Entry_2017.csv", index=False)
    client_df.to_csv(folder / "metadata.csv", index=False)

print("Partitioned CSVs created successfully for both clients!")