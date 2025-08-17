import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import onnx

direction_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

# === 1. Load CSV ===
df = pd.read_csv("./processed_data/training_data.csv", header=None)

# Assuming first 16 columns are board, last column is direction
boards = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values

# === 2. Normalize ===
boards = boards / 12.0  # Keeps all values in range [0, 1]

# === 3. Convert to tensors ===
boards_tensor = torch.tensor(boards, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)  # For classification

# === 4. Dataset and split ===
dataset = TensorDataset(boards_tensor, labels_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)


# === 5. Define model ===
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 4)  # 4 possible moves

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return self.fc6(x)  # raw scores for each move


model = Net()

# === 6. Training setup ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === 7. Train loop ===
for epoch in range(10):
    model.train()
    total_loss = 0
    for boards_batch, labels_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(boards_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for boards_batch, labels_batch in val_loader:
            outputs = model(boards_batch)
            _, predicted = torch.max(outputs, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

    print(
        f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, "
        f"Val Acc: {100*correct/total:.2f}%"
    )

# === 8. Save model ===
torch.save(model.state_dict(), "trained_models/FC_model_2048.pt")
print("Model saved as FC_model_2048.pt")


# === 9. Convert model to ONNX ===
model.eval()
dummy_input = torch.randn(1, 20)  # batch size 1, 20 features (your board input size)
torch.onnx.export(
    model,
    dummy_input,
    "Onnx_models/temp.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamo=True  # <-- add this flag to enable new exporter
)

onnx_model = onnx.load("Onnx_models/temp.onnx")
onnx.save(onnx_model, "Onnx_models/model_2048_for_js.onnx", save_as_external_data=False)

os.remove('Onnx_models/temp.onnx')
os.remove('Onnx_models/temp.onnx.data')
