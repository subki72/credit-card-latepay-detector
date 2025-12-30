# FinanKu Credit Card Delay Prediction 

## Overview
Project ini bertujuan untuk memprediksi keterlambatan pembayaran kartu kredit (Late Payment) nasabah FinanKu menggunakan *Machine Learning*. Fokus utama dari perbaikan model ini adalah **meminimalkan risiko gagal bayar** dengan meningkatkan metrik **Recall**, sehingga Bank dapat mendeteksi lebih banyak nasabah berisiko tinggi.

## Key Achievements (Final Update)
Setelah melalui proses iterasi, perbaikan bug data leakage, dan penanganan *imbalanced data*, performa model mengalami peningkatan drastis:
* **Recall Score Melesat:** Dari **<40%** menjadi **81.65%**.
* **Business Impact:** Model kini mampu mendeteksi 8 dari 10 nasabah yang berpotensi macet (sebelumnya hanya 4 dari 10).

---

## Methodology & Improvements
Tantangan utama dataset ini adalah ketimpangan kelas (*imbalanced data*) di mana jumlah nasabah gagal bayar jauh lebih sedikit dibandingkan nasabah lancar. Berikut strategi yang diterapkan:

1.  **Feature Engineering (Horizon Waktu):**
    * **Eksperimen 1:** Menggunakan data historis 1 tahun terakhir.
    * **Eksperimen 2:** Menggunakan data historis 6 bulan terakhir (Terbukti lebih relevan).
    * *Bug Fix:* Memperbaiki variabel prediktor pada Eksperimen 2 yang sebelumnya mengalami kebocoran data (*data leakage*) dari Eksperimen 1.

2.  **Handling Imbalance:**
    * Menggunakan **SMOTE (Synthetic Minority Over-sampling Technique)** pada data *training* untuk menyeimbangkan kelas target.

3.  **Threshold Tuning:**
    * Mengubah *decision threshold* dari default `0.5` menjadi **`0.3`**.
    * Langkah ini dilakukan untuk membuat model lebih sensitif terhadap risiko (Prioritas: *Better Safe Than Sorry*).

---

## Model Performance Evaluation

Evaluasi dilakukan menggunakan **Logistic Regression**, **Random Forest**, dan **Gradient Boosting**. Model terbaik yang dipilih adalah **Logistic Regression (Eksperimen 2)**.

| Metric | Old Model (Initial) | **New Model (Final)** |
| :--- | :--- | :--- |
| **Recall (Validasi)** | < 40% | **81.65%** |
| **Accuracy** | ~90% (Bias) | 43% (Risk-Averse) |
| **Handling** | None | SMOTE + Threshold 0.3 |

**Analisa Bisnis:**
Meskipun Akurasi turun menjadi 43%, ini adalah *trade-off* yang disengaja. Dalam kasus deteksi Fraud/Risk, **False Negative** (gagal mendeteksi nasabah macet) jauh lebih merugikan daripada **False Positive** (mencurigai nasabah lancar). Model saat ini sangat agresif dalam memitigasi risiko.

---

## File Structure

| File | Description |
| :--- | :--- |
| `PredictionModel.ipynb` | Notebook utama (Data Cleaning, EDA, Feature Engineering, Modeling, Evaluation) |
| `FinanKu Data All.csv` | Dataset Utama |
| `FinanKu Data Validasi.csv`| Dataset Validasi (Unseen Data) |
| `README.md` | Dokumentasi Project |

---

## Getting Started

1.  **Clone Repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[username]/credit-card-latepay-detector.git
    cd credit-card-latepay-detector
    ```

2.  **Install Library:**
    Pastikan library berikut terinstall (terutama `imbalanced-learn`):
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Notebook:**
    Buka `PredictionModel.ipynb` menggunakan Jupyter Notebook atau VS Code dan jalankan semua cell (Run All).

---

## Recommendation for Future Work
* Melakukan eksplorasi algoritma lain seperti LightGBM atau CatBoost.
* Mencoba *Cost-Sensitive Learning* sebagai alternatif Threshold Tuning.
* Menambahkan fitur eksternal seperti riwayat BI Checking (SLIK) jika tersedia.

---

**Author:** Muhammad Syafii Assubki
**License:** For research/portfolio purpose only.
