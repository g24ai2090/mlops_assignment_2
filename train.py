# train.py
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load data
data = fetch_olivetti_faces()
X = data.data
y = data.target

# Split 70-30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Save as .pth using joblib
joblib.dump(model, 'savedmodel.pth')
print("Model saved as savedmodel.pth using joblib")