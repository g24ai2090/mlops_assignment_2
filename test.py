import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

data = fetch_olivetti_faces()
X = data.data.reshape(-1, 1, 64, 64)
y = data.target

_, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).long()
test_ds = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_ds, batch_size=32)

model = torch.load('savedmodel.pth', map_location='cpu')
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for xb, yb in test_loader:
        pred = model(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")