import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Load data
data = fetch_olivetti_faces()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Reshape to (N, 1, 64, 64)
X_train = X_train.reshape(-1, 1, 64, 64)
X_test = X_test.reshape(-1, 1, 64, 64)

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# Simple CNN
class FaceCNN(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512), nn.ReLU(), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = FaceCNN()
model.to('cpu')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(50):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/50 completed")

# Save original (full precision) model
torch.save(model, 'model_original.pth')
original_size = os.path.getsize('model_original.pth') / 1024  # KB

# Dynamic quantization (quantizes Linear layers to int8)
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Save quantized model (whole model for easy loading)
torch.save(quantized_model, 'savedmodel.pth')
quant_size = os.path.getsize('savedmodel.pth') / 1024

print(f"Original model size: {original_size:.2f} KB")
print(f"Quantized model size: {quant_size:.2f} KB")
print(f"Size reduction: {100*(1 - quant_size/original_size):.2f}%")