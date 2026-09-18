import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# The 14 diseases tracked in the NIH dataset
DISEASES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
            'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Images are stored on disk at 256x256 by prepare_data.py; the network sees 224x224.
STORE_SIZE = 256
INPUT_SIZE = 224


def build_transforms(train: bool):
    """Augment during training; deterministic center crop for validation."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def inference_transform():
    """Preprocessing for a raw, arbitrary-sized upload at serving time.

    Must mirror the validation path (short side to 256, then centre 224) or the
    model sees a different framing at serving time than it was scored on.
    """
    return transforms.Compose([
        transforms.Resize(STORE_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Multi-label one-hot. Split on '|' and match exactly -- a substring test
        # would silently mislabel any disease name contained in another.
        disease_index = {d: i for i, d in enumerate(DISEASES)}
        labels = torch.zeros((len(self.df), len(DISEASES)), dtype=torch.float32)
        for row, finding in enumerate(self.df['Finding Labels'].astype(str)):
            for part in finding.split('|'):
                idx = disease_index.get(part.strip())
                if idx is not None:
                    labels[row, idx] = 1.0
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['Image Index']
        img_path = os.path.join(self.img_dir, img_name)

        # Stored grayscale; replicate to 3 channels for the ImageNet-pretrained stem.
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]

    def pos_weight(self, cap: float = 50.0):
        """Per-class BCE positive weight = #negatives / #positives.

        NIH prevalence ranges from Infiltration (17.7%) to Hernia (0.19%), so a
        single flat weight badly miscalibrates the rare classes. Capped because
        Hernia's raw ratio (~500) makes training unstable.
        """
        pos = self.labels.sum(dim=0)
        neg = len(self.labels) - pos
        weight = neg / pos.clamp(min=1.0)
        return weight.clamp(max=cap)


if __name__ == "__main__":
    ds = ChestXrayDataset(csv_file=os.path.join("data", "client_1_prep", "train.csv"),
                          img_dir=os.path.join("data", "client_1_prep"),
                          transform=build_transforms(train=True))
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    images, labels = next(iter(loader))
    print(f"Image batch shape: {images.shape} (Expected: 4, 3, 224, 224)")
    print(f"Labels batch shape: {labels.shape} (Expected: 4, 14)")
    print("pos_weight:", {d: round(w.item(), 1) for d, w in zip(DISEASES, ds.pos_weight())})
