import pandas as pd
import numpy as np
import random

# Set seed agar hasil konsisten
random.seed(42)
np.random.seed(42)

# Jumlah data
n = 100

# Data simulasi
data = {
    "ipk": np.round(np.random.uniform(2.0, 4.0, n), 2),
    "sks": np.random.randint(110, 150, n),
    "kehadiran": np.random.randint(60, 100, n),
    "tidak_lulus": np.random.randint(0, 5, n),
    "organisasi": np.random.randint(0, 2, n),  # 0 = tidak aktif, 1 = aktif
    "semester": np.random.randint(6, 10, n),   # semester 6 sampai 9

    # One-hot encoding jurusan (random assign)
    "jurusan_Akuntansi": [0]*n,
    "jurusan_Manajemen": [0]*n,
    "jurusan_Sistem Informasi": [0]*n,
    "jurusan_Teknik Informatika": [0]*n,
}

# Assign jurusan secara acak
jurusan_list = ["jurusan_Akuntansi", "jurusan_Manajemen", 
                "jurusan_Sistem Informasi", "jurusan_Teknik Informatika"]
for i in range(n):
    jurusan_terpilih = random.choice(jurusan_list)
    data[jurusan_terpilih][i] = 1

# Target (0 = Tidak Lulus, 1 = Lulus)
# Skenario: IPK > 3.0, kehadiran > 80, tidak_lulus <= 1 → cenderung Lulus
target = []
for i in range(n):
    if data["ipk"][i] > 3.0 and data["kehadiran"][i] > 80 and data["tidak_lulus"][i] <= 1:
        target.append(1)
    else:
        target.append(0)

data["target"] = target

# Simpan ke CSV
df = pd.DataFrame(data)
df.to_csv("dataset_kelulusan_realistic.csv", index=False)

print("✅ Dataset berhasil dibuat dan disimpan sebagai dataset_kelulusan_realistic.csv")
