import flwr as fl

def fit_config(server_round: int):
    return {"server_round": server_round}

if __name__ == "__main__":
    print("--- Starting Sentinel-AI Central Server (Krum Defense Active) ---")
    
    # Phase 5: Byzantine-robust Krum strategy
    strategy = fl.server.strategy.Krum(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,         # We need at least 2 hospitals to start
        min_evaluate_clients=2,
        min_available_clients=2,
        num_malicious_clients=0,   # Expecting 0 malicious clients for this test baseline
        on_fit_config_fn=fit_config,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )