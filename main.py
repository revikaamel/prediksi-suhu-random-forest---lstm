import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =============================================================
# 1. LOAD DATASET TRAIN & TEST
# =============================================================
df_train = pd.read_csv("DailyDelhiClimateTrain.csv", parse_dates=["date"])
df_test  = pd.read_csv("DailyDelhiClimateTest.csv", parse_dates=["date"])

df_train.set_index("date", inplace=True)
df_test.set_index("date", inplace=True)

print("\n=== Dataset Train Loaded ===")
print(df_train.head())

print("\n=== Dataset Test Loaded ===")
print(df_test.head())

# =============================================================
# 2. CEK MISSING VALUE
# =============================================================
print("\n=== Missing Value Train ===")
print(df_train.isnull().sum())

print("\n=== Missing Value Test ===")
print(df_test.isnull().sum())

# =============================================================
# 3. TRANSFORMASI DATA HARIAN → MINGGUAN (MEAN PER WEEK)
# =============================================================
df_train_weekly = df_train.resample("W").mean()
df_test_weekly  = df_test.resample("W").mean()

print("\n=== Dataset Train Mingguan ===")
print(df_train_weekly.head())

print("\n=== Dataset Test Mingguan ===")
print(df_test_weekly.head())

# =============================================================
# 4. SCALING (MINMAXSCALER)
# =============================================================
feature_cols = ["humidity", "wind_speed", "meanpressure"]
target_col = "meantemp"

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(df_train_weekly[feature_cols])
y_train_scaled = scaler_y.fit_transform(df_train_weekly[[target_col]])

X_test_scaled  = scaler_X.transform(df_test_weekly[feature_cols])
y_test_scaled  = scaler_y.transform(df_test_weekly[[target_col]])

print("\n=== Scaling Selesai ===")

# =============================================================
# 5. RANDOM FOREST REGRESSION (MINGGUAN)
# =============================================================
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train_scaled.ravel())

y_pred_scaled = rf.predict(X_test_scaled)

# Kembalikan ke skala asli
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

# =============================================================
# 6. EVALUASI MODEL (MSE, RMSE, MAE, MAPE, R²)
# =============================================================
mse  = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test_actual, y_pred)
mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100
r2   = r2_score(y_test_actual, y_pred)

print("\n=== Evaluasi Random Forest Mingguan ===")
print("MSE  :", mse)
print("RMSE :", rmse)
print("MAE  :", mae)
print("MAPE :", mape, "%")
print("R²   :", r2)

# =============================================================
# 7. SIMPAN PLOT HASIL PREDIKSI
# =============================================================
if not os.path.exists("outputs"):
    os.makedirs("outputs")

plt.figure(figsize=(8, 5))
plt.scatter(y_test_actual, y_pred)
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Random Forest Weekly Prediction (Train-Test)")
plt.savefig("outputs/rf_weekly_scatter.png")
plt.close()

print("\n=== Plot disimpan di folder outputs/ ===")
print("\n=== SELESAI ===")