import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
from collections import OrderedDict

# Import from our existing Phase 1 & 2 code
from models.resnet_model import get_model, train_one_epoch
from models.dataset import ChestXrayDataset

class SentinelClient(fl.client.NumPyClient):
    def __init__(self, dataloader, device):
        self.model = get_model().to(device)
        self.dataloader = dataloader
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()

    def get_parameters(self, config):
        # Extract weights as NumPy arrays for network transmission
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        # Inject aggregated weights received from the server back into the PyTorch model
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Train locally for 1 epoch
        loss, auc = train_one_epoch(self.model, self.dataloader, self.criterion, optimizer, self.device)
        
        return self.get_parameters(config={}), len(self.dataloader.dataset), {"loss": loss, "auc": auc}

    def evaluate(self, parameters, config):
        # In a real setup, this would test on a local validation set
        return float(0.0), len(self.dataloader.dataset), {"accuracy": float(0.0)}

if __name__ == "__main__":
    client_id = sys.argv[1] if len(sys.argv) > 1 else "1"
    print(f"--- Starting Hospital Node (Client {client_id}) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the specific dataset for this client (Non-IID split)
    data_path = rf"data\client_{client_id}"
    csv_path = rf"data\client_{client_id}\metadata.csv"
    
    dataset = ChestXrayDataset(csv_file=csv_path, img_dir=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Start the Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=SentinelClient(dataloader, device),
    )