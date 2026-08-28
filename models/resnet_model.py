import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np

# Import the dataset we built in Phase 1
from models.dataset import ChestXrayDataset

def get_model():
    # Load a pretrained ResNet-18 backbone (knows edges, shapes, textures from ImageNet)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Replace the final fully connected (fc) layer
    # ResNet18 outputs 512 features. We map those to our 14 disease classes.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 14)
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # 1. Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 2. Backward pass & optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Save predictions and labels for AUC-ROC calculation
        all_labels.append(labels.cpu().detach().numpy())
        # Outputs are raw logits. BCEWithLogitsLoss applies Sigmoid internally for loss,
        # but we must apply it manually here to get probabilities for our AUC metric.
        probs = torch.sigmoid(outputs).cpu().detach().numpy()
        all_preds.append(probs)

    epoch_loss = running_loss / len(dataloader)
    
    # Flatten arrays for sklearn metric
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    
    # Calculate AUC-ROC
    try:
        auc = roc_auc_score(all_labels, all_preds, average='macro')
    except ValueError:
        # Dummy data might randomly have a batch with 0 positive cases for a disease
        auc = float('nan') 

    return epoch_loss, auc

if __name__ == "__main__":
    print("--- Phase 2: Centralized Baseline Training ---")
    
    # Automatically use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ChestXrayDataset(csv_file=r"data\client_1\metadata.csv", 
                               img_dir=r"data\client_1", 
                               transform=transform)
    # Using a small batch size for sandbox testing
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 2. Initialize Model, Loss (BCEWithLogitsLoss), and Optimizer (Adam)
    model = get_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Run training loop for 3 epochs
    num_epochs = 3
    for epoch in range(num_epochs):
        loss, auc = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f} | AUC-ROC: {auc:.4f}")
        
    print("Phase 2 Complete! Baseline model is learning.")