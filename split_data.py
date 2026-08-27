import os
import pandas as pd
import numpy as np
import cv2

# 14 disease labels from the NIH dataset
DISEASES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
            'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

def create_dummy_client_data(client_dir, num_images, start_idx):
    os.makedirs(client_dir, exist_ok=True)
    data = []

    for i in range(num_images):
        img_name = f"dummy_{start_idx + i:04d}.png"
        img_path = os.path.join(client_dir, img_name)

        # Create a blank 224x224 dummy image using OpenCV
        img = np.zeros((224, 224), dtype=np.uint8)
        cv2.imwrite(img_path, img)

        # Randomly assign diseases to simulate real patient data
        row = {'Image Index': img_name}
        labels = (np.random.rand(len(DISEASES)) > 0.85).astype(int)
        active_diseases = [DISEASES[j] for j, val in enumerate(labels) if val == 1]
        row['Finding Labels'] = "|".join(active_diseases) if active_diseases else "No Finding"
        
        data.append(row)

    # Save the metadata CSV
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(client_dir, 'metadata.csv'), index=False)

if __name__ == "__main__":
    print("Generating 150 images for Client 1...")
    create_dummy_client_data(r"data\client_1", 150, 1)
    
    print("Generating 50 images for Client 2...")
    create_dummy_client_data(r"data\client_2", 50, 151)
    
    print("Sandbox split complete!")