import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
from PIL import Image

# The 14 diseases tracked in the NIH dataset
DISEASES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
            'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Create a one-hot encoding logic for the multi-label problem
        self.labels = []
        for finding in self.df['Finding Labels']:
            # Example: finding might be "Atelectasis|Effusion"
            row_labels = [1.0 if d in str(finding) else 0.0 for d in DISEASES]
            self.labels.append(row_labels)
            
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['Image Index']
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image (RGB format expected by pretrained ResNet)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label

# Verification Step: Do not move on until this prints the correct shapes
if __name__ == "__main__":
    # ImageNet normalization standards for pretrained ResNet backbones
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Test loading Client 1's sandbox data
    test_dataset = ChestXrayDataset(csv_file=r"data\client_1\metadata.csv", 
                                    img_dir=r"data\client_1", 
                                    transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    
    images, labels = next(iter(test_loader))
    print("--- Phase 1 Verification ---")
    print(f"Image batch shape: {images.shape} (Expected: 4, 3, 224, 224)")
    print(f"Labels batch shape: {labels.shape} (Expected: 4, 14)")