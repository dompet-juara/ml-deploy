# API Prediksi Perilaku Keuangan 💸🧠

[![Versi Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![ML Library](https://img.shields.io/badge/ML%20Library-TensorFlow%20%26%20Keras-FF6F00.svg)](https://www.tensorflow.org/)
[![Deployment](https://img.shields.io/badge/Deployment-Docker-blue.svg)](https://www.docker.com/)
[![Lisensi: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/dompet-juara/ml-deploy/blob/main/LICENSE) <!-- Ganti dengan path LICENSE yang benar jika berbeda -->

Sebuah *REST API* berbasis `FastAPI` yang memprediksi perilaku keuangan menggunakan *machine learning*. *API* ini menganalisis berbagai metrik keuangan untuk mengklasifikasikan dan memprediksi pola pengeluaran serta perilaku keuangan.

## 📖 Daftar Isi

*   [✨ Fitur Utama](#-fitur-utama)
*   [📍 *API Endpoints*](#-api-endpoints)
    *   [Pemeriksaan Kesehatan (*Health Check*)](#pemeriksaan-kesehatan-health-check)
    *   [Prediksi](#prediksi)
*   [📥 Kolom Input](#-kolom-input)
*   [⚙️ Pengaturan dan Instalasi](#️-pengaturan-dan-instalasi)
    *   [Prasyarat](#prasyarat)
    *   [Pengaturan Lokal](#pengaturan-lokal)
    *   [Pengaturan *Docker*](#pengaturan-docker)
*   [🚀 Contoh Penggunaan](#-contoh-penggunaan)
    *   [Contoh Permintaan (*Request*)](#contoh-permintaan-request)
    *   [Contoh Respons](#contoh-respons)
    *   [Contoh Klien *Python*](#contoh-klien-python)
*   [📄 Persyaratan (*Requirements*)](#-persyaratan-requirements)
*   [🧠 Pelatihan Model](#-pelatihan-model)
*   [⚠️ Penanganan Kesalahan (*Error Handling*)](#️-penanganan-kesalahan-error-handling)
*   [🤝 Berkontribusi](#-berkontribusi)
*   [📜 Lisensi](#-lisensi)

## ✨ Fitur Utama

*   **Prediksi *Machine Learning***: Menggunakan *Neural Network* `TensorFlow`/`Keras` untuk klasifikasi perilaku keuangan.
*   ***RESTful API***: *Endpoint API* yang bersih dan terdokumentasi dengan baik.
*   **Validasi Data**: Validasi input yang tangguh menggunakan model `Pydantic`.
*   **Skalabilitas Tinggi**: Dibangun dengan `FastAPI` untuk performa tinggi.
*   ***Dockerized***: Siap untuk *deployment* menggunakan *container*.

## 📍 *API Endpoints*

### Pemeriksaan Kesehatan (*Health Check*)

*   `GET /` - *Root endpoint* dengan informasi dasar *API*.
*   `GET /health` - *Endpoint* pemeriksaan kesehatan untuk memverifikasi status *API* dan model.

### Prediksi

*   `POST /predict` - Memprediksi perilaku keuangan berdasarkan data input.

## 📥 Kolom Input

*API* menerima kolom data keuangan berikut:

| Nama Kolom                | Deskripsi                               | Tipe  | Contoh     |
| :------------------------ | :-------------------------------------- | :---- | :--------- |
| `Gaji`                    | Gaji bulanan                            | float | 5000000.0  |
| `Tabungan Lama`           | Tabungan lama                           | float | 10000000.0 |
| `Investasi`               | Nilai investasi                         | float | 2000000.0  |
| `Pemasukan Lainnya`       | Pemasukan lainnya                       | float | 1000000.0  |
| `Bahan Pokok`             | Pengeluaran kebutuhan pokok             | float | 1500000.0  |
| `Protein & Gizi Tambahan` | Pengeluaran protein dan nutrisi         | float | 500000.0   |
| `Tempat Tinggal`          | Biaya tempat tinggal                    | float | 2000000.0  |
| `Sandang`                 | Pengeluaran pakaian                     | float | 300000.0   |
| `Konsumsi Praktis`        | Konsumsi praktis                        | float | 800000.0   |
| `Barang & Jasa Sekunder`  | Barang dan jasa sekunder                | float | 400000.0   |
| `Pengeluaran Tidak Esensial`| Pengeluaran tidak esensial            | float | 200000.0   |
| `Pajak`                   | Pembayaran pajak                        | float | 250000.0   |
| `Asuransi`                | Premi asuransi                          | float | 300000.0   |
| `Sosial & Budaya`         | Pengeluaran sosial dan budaya           | float | 150000.0   |
| `Tabungan / Investasi`    | Tabungan/investasi baru                 | float | 500000.0   |

## ⚙️ Pengaturan dan Instalasi

### Prasyarat

*   `Python` 3.9+
*   `pip`
*   `Docker`

### Pengaturan Lokal

1.  **Gandakan repositori (*Clone the repository*)**
    ```bash
    git clone https://github.com/dompet-juara/ml-deploy.git
    cd financial-behavior-prediction-api
    ```

2.  **Buat lingkungan virtual (*Create virtual environment*)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Di Windows: venv\Scripts\activate
    ```

3.  **Instal dependensi (*Install dependencies*)**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Persiapkan file model (*Prepare model files*)**

    Pastikan Anda memiliki file-file berikut di direktori `model/`:
    *   `best_model.h5` - Model `TensorFlow`/`Keras` yang telah dilatih.
    *   `scaler.pkl` - *Preprocessor* `StandardScaler` atau sejenisnya yang telah di-*fit*.
    *   `label_encoder.pkl` - `LabelEncoder` yang telah di-*fit* untuk kelas target.

5.  **Jalankan aplikasi (*Run the application*)**
    ```bash
    python main.py
    ```

    Atau menggunakan `uvicorn` secara langsung:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 5000 --reload
    ```

### Pengaturan *Docker*

1.  **Bangun *image Docker* (*Build the Docker image*)**
    ```bash
    docker build -t financial-prediction-api .
    ```

2.  **Jalankan *container* (*Run the container*)**
    ```bash
    docker run -p 5000:5000 -e PORT=5000 financial-prediction-api
    ```

## 🚀 Contoh Penggunaan

### Contoh Permintaan (*Request*)

```bash
curl -X POST "http://localhost:5000/predict" \
-H "Content-Type: application/json" \
-d '{
  "Gaji": 5000000.0,
  "Tabungan Lama": 10000000.0,
  "Investasi": 2000000.0,
  "Pemasukan Lainnya": 1000000.0,
  "Bahan Pokok": 1500000.0,
  "Protein & Gizi Tambahan": 500000.0,
  "Tempat Tinggal": 2000000.0,
  "Sandang": 300000.0,
  "Konsumsi Praktis": 800000.0,
  "Barang & Jasa Sekunder": 400000.0,
  "Pengeluaran Tidak Esensial": 200000.0,
  "Pajak": 250000.0,
  "Asuransi": 300000.0,
  "Sosial & Budaya": 150000.0,
  "Tabungan / Investasi": 500000.0
}'
```

### Contoh Respons

```json
{
  "prediction": "Conservative_Saver",
  "probabilities": {
    "Conservative_Saver": 0.75,
    "Moderate_Spender": 0.20,
    "High_Risk_Investor": 0.05
  }
}
```
*(Catatan: Nama kelas seperti "Conservative_Saver" adalah output dari model dan tidak diterjemahkan)*

### Contoh Klien *Python*

```python
import requests

url = "http://localhost:5000/predict"
data = {
    "Gaji": 5000000.0,
    "Tabungan Lama": 10000000.0,
    "Investasi": 2000000.0,
    "Pemasukan Lainnya": 1000000.0,
    "Bahan Pokok": 1500000.0,
    "Protein & Gizi Tambahan": 500000.0,
    "Tempat Tinggal": 2000000.0,
    "Sandang": 300000.0,
    "Konsumsi Praktis": 800000.0,
    "Barang & Jasa Sekunder": 400000.0,
    "Pengeluaran Tidak Esensial": 200000.0,
    "Pajak": 250000.0,
    "Asuransi": 300000.0,
    "Sosial & Budaya": 150000.0,
    "Tabungan / Investasi": 500000.0
}

response = requests.post(url, json=data)
result = response.json()
print(f"Prediksi: {result['prediction']}")
print(f"Probabilitas: {result['probabilities']}")
```

## 📄 Persyaratan (*Requirements*)

Pustaka yang digunakan terdapat dalam file `requirements.txt` dengan dependensi berikut:

```
fastapi==0.115.12
numpy==1.24.4
pydantic==2.11.5
tensorflow-cpu==2.10.1
uvicorn==0.34.3
scikit-learn==1.5.0
```

## 🧠 Pelatihan Model

*API* ini mengharapkan model yang sudah dilatih sebelumnya. Pastikan *pipeline* pelatihan model Anda:

1.  Menggunakan urutan fitur yang sama seperti yang didefinisikan dalam `feature_names` (jika ada variabel global atau konfigurasi untuk ini).
2.  Menyimpan model dalam format `.h5`.
3.  Menyimpan *scaler* dan *label encoder* sebagai file `pickle`.
4.  Menggunakan versi `TensorFlow`/`Keras` yang kompatibel.

## ⚠️ Penanganan Kesalahan (*Error Handling*)

*API* menyertakan penanganan kesalahan yang komprehensif untuk:

*   Kolom input yang hilang atau tidak valid.
*   Kesalahan saat memuat model.
*   Kesalahan prediksi.
*   Kesalahan validasi data.

## 🤝 Berkontribusi

1.  *Fork* repositori ini.
2.  Buat *feature branch* (`git checkout -b feature/fitur-luar-biasa`).
3.  *Commit* perubahan Anda (`git commit -m 'Menambahkan fitur luar biasa'`).
4.  *Push* ke *branch* (`git push origin feature/fitur-luar-biasa`).
5.  Buka *Pull Request*.

## 📜 Lisensi

Proyek ini dilisensikan di bawah Lisensi MIT - lihat file [LICENSE](LICENSE) untuk detailnya. <!-- Pastikan path ke file LICENSE benar -->
