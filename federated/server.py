import flwr as fl

def fit_config(server_round: int):
    return {"server_round": server_round}

if __name__ == "__main__":
    print("--- Starting Sentinel-AI Central Server ---")
    
    # Phase 4 uses basic Federated Averaging (FedAvg)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,         # Require 100% of available clients to train
        min_fit_clients=2,        # We need at least 2 hospitals to start
        min_available_clients=2,
        on_fit_config_fn=fit_config,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )