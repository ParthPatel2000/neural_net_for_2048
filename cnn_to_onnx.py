import torch
import onnx
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import os

import torch.nn as nn
import torch.nn.functional as F


class CNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2)  # 1 channel -> 16 filters, 2x2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2)  # 16 -> 32 filters, 2x2

        # Fully connected layers
        self.fc1 = nn.Linear(192, 64)  # flatten after conv2
        self.fc2 = nn.Linear(64, 4)  # 4 moves

    def forward(self, x):
        x = F.relu(self.conv1(x))  # activation after conv
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten for fc
        # print(x.shape)  # Debug: print shape after flattening
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # logits for CrossEntropyLoss
        return x


model = CNNNet()  # your model class
model.load_state_dict(torch.load("trained_models/cnn_model_2048.pt"))
model.eval()

dummy_input = torch.randn(1, 1, 5, 4)  # batch size 1, 1 channel, 5x4 input 
torch.onnx.export(
    model,
    dummy_input,
    "Onnx_models/temp.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamo=True,  # <-- add this flag to enable new exporter
)

onnx_model = onnx.load("Onnx_models/temp.onnx")
onnx.save(onnx_model, "Onnx_models/model_2048_for_js.onnx", save_as_external_data=False)

os.remove("Onnx_models/temp.onnx")
os.remove("Onnx_models/temp.onnx.data")
