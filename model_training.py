import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn # Tambahkan ini
print(f"Versi scikit-learn: {sklearn.__version__}") # Tambahkan ini

# --- 1. Pengumpulan Data & Eksplorasi Awal ---
file_path = 'Jumlah_Penumpang_Angkutan_Umum_yang_Terlayani_Perhari.csv'
try:
    df = pd.read_csv(file_path, delimiter=';')
    print(f"Dataset '{file_path}' berhasil dimuat.")

    # CHANGE 1: Keep only KRL and Transjakarta
    modes_to_keep = ['krl', 'transjakarta']
    df = df[df['jenis_moda'].isin(modes_to_keep)].copy()
    print(f"\nData has been filtered to only include: {modes_to_keep}")

except FileNotFoundError:
    print(f"Error: File '{file_path}' tidak ditemukan. Pastikan file berada di direktori yang sama dengan script.")
    exit()
except Exception as e:
    print(f"Error saat membaca file: {e}")
    exit()


# --- 2. Pembersihan Data dan Rekayasa Fitur ---
kolom_jumlah_penumpang = 'jumlah_penumpang_per_hari'
kolom_tanggal = 'tanggal'
kolom_jenis_moda = 'jenis_moda'

df[kolom_jumlah_penumpang] = pd.to_numeric(
    df[kolom_jumlah_penumpang].astype(str).str.replace(',', '.'), errors='coerce'
)
df[kolom_tanggal] = pd.to_datetime(df[kolom_tanggal], format='%d/%m/%Y', errors='coerce')
df.dropna(subset=[kolom_jumlah_penumpang, kolom_tanggal, kolom_jenis_moda], inplace=True)
print(f"\nJumlah baris setelah menghapus NaN: {len(df)}")


# --- Rekayasa Fitur dari kolom Tanggal ---
df['Tahun'] = df[kolom_tanggal].dt.year
df['Hari_dalam_Bulan'] = df[kolom_tanggal].dt.day
df['Hari_sin'] = np.sin(2 * np.pi * df[kolom_tanggal].dt.dayofweek / 7)
df['Hari_cos'] = np.cos(2 * np.pi * df[kolom_tanggal].dt.dayofweek / 7)
df['Bulan_sin'] = np.sin(2 * np.pi * df[kolom_tanggal].dt.month / 12)
df['Bulan_cos'] = np.cos(2 * np.pi * df[kolom_tanggal].dt.month / 12)
df['lag_1_hari'] = df.groupby(kolom_jenis_moda)[kolom_jumlah_penumpang].shift(1)
df['lag_7_hari'] = df.groupby(kolom_jenis_moda)[kolom_jumlah_penumpang].shift(7)
df['is_weekend'] = (df[kolom_tanggal].dt.dayofweek >= 5).astype(int)

df = df.sort_values(by=[kolom_jenis_moda, kolom_tanggal]).reset_index(drop=True)

def calculate_weekly_avg(row, df_sorted, num_weeks=3):
    current_date = row[kolom_tanggal]
    current_dayofweek = row[kolom_tanggal].dayofweek
    current_moda = row[kolom_jenis_moda]
    relevant_data = df_sorted[
        (df_sorted[kolom_jenis_moda] == current_moda) &
        (df_sorted[kolom_tanggal].dt.dayofweek == current_dayofweek) &
        (df_sorted[kolom_tanggal] < current_date)
    ]
    start_lookback_date = current_date - pd.Timedelta(weeks=num_weeks + 1)
    end_lookback_date = current_date - pd.Timedelta(days=1)
    recent_relevant_data = relevant_data[
        (relevant_data[kolom_tanggal] >= start_lookback_date) &
        (relevant_data[kolom_tanggal] <= end_lookback_date)
    ]
    return recent_relevant_data[kolom_jumlah_penumpang].mean() if not recent_relevant_data.empty else np.nan

df['Rata_rata_penumpang_3_minggu_lalu'] = df.apply(
    lambda row: calculate_weekly_avg(row, df, num_weeks=3), axis=1
)
df.dropna(subset=['Rata_rata_penumpang_3_minggu_lalu', 'lag_1_hari', 'lag_7_hari'], inplace=True)


# --- 3. Definisi Target Klasifikasi ---
median_per_mode = df.groupby(kolom_jenis_moda)[kolom_jumlah_penumpang].median()
def set_target(row):
    moda = row[kolom_jenis_moda]
    threshold = median_per_mode.get(moda, 0)
    return 1 if row[kolom_jumlah_penumpang] > threshold else 0
df['target_penumpang'] = df.apply(set_target, axis=1)
print("Ini median per moda:", median_per_mode)

# --- 4. Pemilihan Fitur (X) dan One-Hot Encoding ---
numerical_features = [
    'Tahun', 'Hari_dalam_Bulan', 'Rata_rata_penumpang_3_minggu_lalu',
    'Hari_sin', 'Hari_cos', 'Bulan_sin', 'Bulan_cos',
    'lag_1_hari', 'lag_7_hari', 'is_weekend'
]
categorical_features = [kolom_jenis_moda]
all_features = numerical_features + categorical_features
X = df[all_features]
y = df['target_penumpang']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression(random_state=42))])


# --- 5. Pelatihan Model & Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model_pipeline.fit(X_train, y_train)


# --- 6. Evaluasi Model ---
y_pred = model_pipeline.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))


# --- 7. Penyimpanan Model ---
joblib.dump(model_pipeline, 'logistic_regression_penumpang_pipeline.pkl')
joblib.dump(all_features, 'model_features_with_moda.pkl')
joblib.dump(numerical_features, 'numerical_features.pkl')
joblib.dump(categorical_features, 'categorical_features.pkl')
print("\nPipeline Model, Daftar Fitur Numerik & Kategorikal berhasil disimpan.")