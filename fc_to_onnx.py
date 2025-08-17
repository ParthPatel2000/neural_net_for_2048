import torch
import onnx
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import os


# === 1. Define model ===
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 4)  # 4 possible moves

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)  # raw scores for each move
    
model = Net()  # your model class
model.load_state_dict(torch.load("trained_models/cnn_model_2048.pt"))
model.eval()

dummy_input = torch.randn(1, 20)  # batch size 1, 20 input features
torch.onnx.export(
    model,
    dummy_input,
    "Onnx_models/tempFC.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamo=True,  # <-- add this flag to enable new exporter
)

onnx_model = onnx.load("Onnx_models/tempFC.onnx")
onnx.save(onnx_model, "Onnx_models/FC_model_2048_for_js.onnx", save_as_external_data=False)

os.remove("Onnx_models/tempFC.onnx")
os.remove("Onnx_models/tempFC.onnx.data")