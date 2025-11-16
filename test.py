from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = fetch_olivetti_faces()
X = data.data
y = data.target

# Split (same as train.py)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# LOAD MODEL WITH JOBLIB (NOT TORCH)
model = joblib.load('savedmodel.pth')

# Predict and show accuracy
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"Test Accuracy: {acc*100:.2f}%")