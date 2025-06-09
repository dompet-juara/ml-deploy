from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pickle

app = FastAPI(title="Financial Behavior Classifier")

model = tf.keras.models.load_model("best_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

feature_names = [
    "Gaji", "Tabungan_Lama", "Investasi", "Pemasukan_Lainnya", "Bahan_Pokok",
    "Protein_Gizi", "Tempat_Tinggal", "Sandang", "Konsumsi_Praktis", "Barang_Jasa_Sekunder",
    "Pengeluaran_Tidak_Esensial", "Pajak", "Asuransi", "Sosial_Budaya", "Tabungan_Investasi"
]

class FinanceInput(BaseModel):
    Gaji: float
    Tabungan_Lama: float
    Investasi: float
    Pemasukan_Lainnya: float
    Bahan_Pokok: float
    Protein_Gizi: float
    Tempat_Tinggal: float
    Sandang: float
    Konsumsi_Praktis: float
    Barang_Jasa_Sekunder: float
    Pengeluaran_Tidak_Esensial: float
    Pajak: float
    Asuransi: float
    Sosial_Budaya: float
    Tabungan_Investasi: float

@app.post("/predict")
def predict_financial_behavior(data: FinanceInput):
    try:
        input_data = np.array([[getattr(data, field) for field in feature_names]])

        input_scaled = scaler.transform(input_data)

        probs = model.predict(input_scaled)
        predicted_class_index = np.argmax(probs, axis=1)[0]

        try:
            label = label_encoder.inverse_transform([predicted_class_index])[0]
        except:
            label = f"Unknown_Class_Index_{predicted_class_index}"

        class_labels = label_encoder.classes_
        probabilities_dict = {
            str(class_labels[i]): float(probs[0][i])
            for i in range(len(class_labels))
        }

        return {
            "prediction": label,
            "probabilities": probabilities_dict
        }
    except Exception as e:
        return {"error": str(e)}
