# file: fl_skeleton.py
import flwr as fl, torch, torch.nn as nn
class Net(nn.Module):
    def __init__(self): super().__init__(); self.fc = nn.Linear(100,2)
    def forward(self,x): return self.fc(x)
def client_fn(cid): return fl.client.NumPyClient(... )  # local hospital data
strategy = fl.server.strategy.FedAvg(min_available_clients=2)
fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy, config=fl.server.ServerConfig(num_rounds=5))