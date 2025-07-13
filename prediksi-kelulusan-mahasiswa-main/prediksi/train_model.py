import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# === 1. SETUP PATHS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "dataset_kelulusan_realistic.csv")
features_path = os.path.join(BASE_DIR, "selected_features.pkl")
model_path = os.path.join(BASE_DIR, "logistic_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

# === 2. LOAD DATASET & SELECTED FEATURES ===
df = pd.read_csv(dataset_path)

# Pastikan 'target' adalah kolom target
X = df.drop(columns=["target"])
y = df["target"]

# === 3. LOAD SELECTED FEATURES ===
selected_features = joblib.load(features_path)
X_selected = X[selected_features]

# === 4. STANDARISASI FITUR ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# === 5. SPLIT DATA TRAIN-TEST ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# === 6. TRAINING MODEL LOGISTIC REGRESSION ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# === 7. SAVE MODEL, SCALER, FEATURES ===
with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

with open(features_path, "wb") as f:
    pickle.dump(selected_features, f)

# === 8. PRINT SUCCESS ===
print("✅ Model training selesai.")
print("📦 Model disimpan:", model_path)
print("📦 Scaler disimpan:", scaler_path)
print("📦 Fitur disimpan:", features_path)
