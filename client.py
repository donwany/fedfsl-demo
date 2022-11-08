from collections import OrderedDict
from typing import List

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18

from model import PrototypicalNetworks
from train import train, test, test_fedfsl, train_fedfsl
from utils import load_data

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")


def get_parameters(net) -> List[np.ndarray]:
    return [val.numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

    path = 'protonet.pth'
    torch.save(state_dict, path)
    net.load_state_dict(torch.load(path), strict=True)

    # net.load_state_dict(state_dict, strict=True)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, train_loader, test_loader):
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        # print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        # print(f"[Client {self.cid}] fit, config: {config}")
        # set parameters on the local device
        set_parameters(self.net, parameters)
        # self.net.set_weights(parameters)
        # train model on local device with one epoch
        # train(self.net, self.train_loader, self.test_loader, epochs=1)
        train_fedfsl(self.net, self.train_loader)
        # return parameters back to the central server for aggregation
        return get_parameters(self.net), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Test local model"""
        # print(f"[Client {self.cid}] evaluate, config: {config}")
        # Update local model with global parameters
        set_parameters(self.net, parameters)
        # self.net.set_weights(parameters)
        # Evaluate global model parameters on the local val data
        # loss, accuracy = test(self.net, self.test_loader)
        total_predictions, correct_predictions = test_fedfsl(self.net, self.test_loader)
        print(f"Total predictions: {total_predictions} and accuracy: {correct_predictions}")
        # print(f"Evaluation loss: {loss} and accuracy: {accuracy}")
        # return float(loss), len(self.test_loader), {"accuracy": float(accuracy)}

        return len(self.test_loader), {"total predictions": total_predictions,
                                       "correct predictions": correct_predictions}


def main() -> None:
    # load model
    convolutional_network = resnet18(pretrained=True)
    convolutional_network.fc = nn.Flatten()
    net = PrototypicalNetworks(convolutional_network).to(DEVICE)

    # load data
    train_loader, test_loader = load_data()

    # flower client
    client = FlowerClient(net=net, train_loader=train_loader, test_loader=test_loader)

    # start flower client
    fl.client.start_numpy_client(server_address="0.0.0.0:5002", client=client)


if __name__ == '__main__':
    main()
