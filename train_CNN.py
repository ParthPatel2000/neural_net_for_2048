import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import onnx
import os

### Step 1: Load CSV and Preprocess into Tensors
# Load CSV
df = pd.read_csv("./processed_data/training_data_log2scale.csv", header=None)
boards = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values

#normalize
boards = boards / 12.0

# Convert to tensors
boards_tensor = torch.tensor(boards, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

# Reshape for CNN: N x 1 x 5 x 4
boards_tensor = boards_tensor.view(-1, 1, 5, 4)

# Dataset and DataLoader
dataset = TensorDataset(boards_tensor, labels_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)


### Step 2: 
import torch.nn as nn
import torch.nn.functional as F

class CNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2)  # 1 channel -> 16 filters, 2x2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2) # 16 -> 32 filters, 2x2
        
        # Fully connected layers
        self.fc1 = nn.Linear(192, 64)  # flatten after conv2
        self.fc2 = nn.Linear(64, 4)       # 4 moves

    def forward(self, x):
        x = F.relu(self.conv1(x))         # activation after conv
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)         # flatten for fc
        # print(x.shape)  # Debug: print shape after flattening
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                   # logits for CrossEntropyLoss
        return x

### Step 3: Define Loss and Optimizer
import torch.optim as optim

model = CNNNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


### Step 4: Training Loop
# Training loop
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
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {100*correct/total:.2f}%")

torch.save(model.state_dict(), "trained_models/cnn_model_2048.pt")
print("CNN model saved!")

model.eval()
dummy_input = torch.randn(1, 1, 5, 4)  # batch size 1, 1 channel, 5x4 input
torch.onnx.export(
    model,
    dummy_input,
    "Onnx_models/tempCNN.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamo=True  # <-- add this flag to enable new exporter
)

onnx_model = onnx.load("Onnx_models/tempCNN.onnx")
onnx.save(onnx_model, "Onnx_models/CNN_model_2048_for_js.onnx", save_as_external_data=False)

os.remove("Onnx_models/tempCNN.onnx")
os.remove("Onnx_models/tempCNN.onnx.data")

