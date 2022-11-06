import flwr as fl


if __name__ == '__main__':
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=2,  # Never sample less than 10 clients for training
        min_evaluate_clients=2,  # Never sample less than 5 clients for evaluation
        min_available_clients=3,  # Wait until all 10 clients are available
    )

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:5002",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )