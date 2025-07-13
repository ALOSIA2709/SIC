import os
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
import joblib

# === 1. SETUP PATHS ===
# Pastikan path dataset dan output berada dalam folder project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(BASE_DIR, "dataset_kelulusan_realistic.csv")
output_path = os.path.join(BASE_DIR, "selected_features.pkl")

# === 2. LOAD DATASET ===
df = pd.read_csv(dataset_path)

# === 3. PISAHKAN FITUR DAN TARGET ===
X = df.drop("target", axis=1)
y = df["target"]

# === 4. NORMALISASI DATA (WAJIB UNTUK CHI2) ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === 5. SELEKSI FITUR MENGGUNAKAN CHI-SQUARE ===
selector = SelectKBest(score_func=chi2, k=8)  # k=8 fitur terbaik
selector.fit(X_scaled, y)

# === 6. AMBIL NAMA FITUR YANG DIPILIH ===
mask = selector.get_support()
selected_features = X.columns[mask].tolist()

# === 7. SIMPAN KE FILE .PKL ===
joblib.dump(selected_features, output_path)

# === 8. CETAK HASIL KE TERMINAL ===
print("✅ Feature selection selesai.")
print("🎯 Fitur terpilih:", selected_features)


