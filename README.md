🌡️ Prediksi Suhu Harian dan Mingguan Menggunakan Random Forest dan LSTM
📌 Deskripsi Project

Project ini merupakan implementasi Machine Learning dan Deep Learning untuk melakukan prediksi suhu berdasarkan data iklim menggunakan metode Random Forest Regression dan Long Short-Term Memory (LSTM).

Prediksi dilakukan dengan pendekatan time series forecasting menggunakan dataset iklim harian. Tujuan dari project ini adalah untuk membandingkan performa model Random Forest dan LSTM dalam memprediksi suhu pada skala harian dan mingguan.

Project ini dibuat sebagai tugas akhir praktikum Machine Learning di Universitas Airlangga.

📊 Dataset

Dataset yang digunakan adalah Daily Delhi Climate Dataset, yang berisi data iklim harian kota Delhi.

Dataset memiliki beberapa fitur utama:

date → tanggal pencatatan data

meantemp → suhu rata-rata harian (°C)

humidity → tingkat kelembaban (%)

wind_speed → kecepatan angin (km/h)

meanpressure → tekanan udara rata-rata (millibar)

Dataset terdiri dari:

DailyDelhiClimateTrain.csv (2013–2016)

DailyDelhiClimateTest.csv (2017)

🧠 Metode yang Digunakan

Project ini menggunakan empat pendekatan model:

1️⃣ Random Forest Harian

Model Random Forest digunakan untuk memprediksi suhu berdasarkan data harian.

2️⃣ Random Forest Mingguan

Data harian diubah menjadi data mingguan menggunakan proses resampling, kemudian dilakukan prediksi menggunakan Random Forest.

3️⃣ LSTM Harian

Model Long Short-Term Memory (LSTM) digunakan untuk menangkap pola sekuensial pada data time series harian.

4️⃣ LSTM Mingguan

Model LSTM juga diterapkan pada data mingguan untuk melihat performa model pada skala waktu yang berbeda.

⚙️ Tahapan Proses Machine Learning

1️⃣ Data preprocessing
2️⃣ Pengecekan missing value
3️⃣ Normalisasi data menggunakan MinMaxScaler
4️⃣ Resampling data (untuk model mingguan)
5️⃣ Training model Random Forest dan LSTM
6️⃣ Evaluasi model

📈 Evaluasi Model

Model dievaluasi menggunakan beberapa metrik regresi:

MSE (Mean Squared Error)

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

MAPE (Mean Absolute Percentage Error)

🏆 Hasil Eksperimen
Model	RMSE	MAE	MAPE
Random Forest Harian	3.48	2.65	13.92%
Random Forest Mingguan	3.73	2.88	15.27%
LSTM Harian	1.60	1.23	5.43%
LSTM Mingguan	6.79	4.02	14.81%

📌 Model terbaik: LSTM Harian dengan tingkat akurasi tertinggi.
