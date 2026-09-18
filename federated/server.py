import flwr as fl
import torch
from collections import OrderedDict
from models.resnet_model import get_model

def fit_config(server_round: int):
    return {"server_round": server_round}

# Create a custom strategy that inherits Krum but adds saving capability
class SaveModelKrum(fl.server.strategy.Krum):
    def aggregate_fit(self, server_round, results, failures):
        # 1. Run the standard Krum aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # 2. Save the resulting global model
        if aggregated_parameters is not None:
            print(f"Saving PyTorch model for Round {server_round}...")
            
            # Convert network parameters to numpy
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            
            # Load the PyTorch blueprint
            model = get_model()
            
            # Match the arrays to the PyTorch dictionary keys
            params_dict = zip(model.state_dict().keys(), ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
            
            # Save to the bridged hard drive folder!
            save_path = f"/app/data/global_model_round_{server_round}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"SUCCESS: Model saved to {save_path}")
            
        return aggregated_parameters, aggregated_metrics

if __name__ == "__main__":
    print("--- Starting Sentinel-AI Central Server (Krum Defense Active) ---")
    
    # Use our custom saving strategy
    strategy = SaveModelKrum(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        num_malicious_clients=0,
        on_fit_config_fn=fit_config,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )