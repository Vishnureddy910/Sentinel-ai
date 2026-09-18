import pandas as pd
from pathlib import Path

base_dir = Path(__file__).parent
client_1_dir = base_dir / "data" / "client_1"
client_2_dir = base_dir / "data" / "client_2"

# Explicitly point to the real master CSV from your screenshot
master_csv = base_dir / "data" / "Data_Entry_2017.csv"

print(f"Reading from REAL master CSV: {master_csv}")
df = pd.read_csv(master_csv)
image_col = "Image Index" if "Image Index" in df.columns else df.columns[0]

for client_num, client_dir in [(1, client_1_dir), (2, client_2_dir)]:
    # Get EXACT filenames currently sitting in the folder
    actual_images = [f.name for f in client_dir.glob("*.png")]
    
    # Create a tiny custom dataframe that perfectly matches the folder
    client_df = df[df[image_col].isin(actual_images)]
    
    # Save it
    client_df.to_csv(client_dir / "metadata.csv", index=False)
    print(f"Hospital {client_num} CSV created with {len(client_df)} valid labels.")