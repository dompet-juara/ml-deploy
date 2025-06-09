from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
from pydantic import BaseModel, Field

app = Flask(__name__)

model = tf.keras.models.load_model("model/best_model.h5")

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

feature_names = [
    "Gaji", "Tabungan_Lama", "Investasi", "Pemasukan_Lainnya", "Bahan_Pokok",
    "Protein_Gizi", "Tempat_Tinggal", "Sandang", "Konsumsi_Praktis", "Barang_Jasa_Sekunder",
    "Pengeluaran_Tidak_Esensial", "Pajak", "Asuransi", "Sosial_Budaya", "Tabungan_Investasi"
]

class FinanceInput(BaseModel):
    Gaji: float = Field(..., alias="Gaji")
    Tabungan_Lama: float = Field(..., alias="Tabungan Lama")
    Investasi: float = Field(..., alias="Investasi")
    Pemasukan_Lainnya: float = Field(..., alias="Pemasukan Lainnya")
    Bahan_Pokok: float = Field(..., alias="Bahan Pokok")
    Protein_Gizi: float = Field(..., alias="Protein & Gizi Tambahan")
    Tempat_Tinggal: float = Field(..., alias="Tempat Tinggal")
    Sandang: float = Field(..., alias="Sandang")
    Konsumsi_Praktis: float = Field(..., alias="Konsumsi Praktis")
    Barang_Jasa_Sekunder: float = Field(..., alias="Barang & Jasa Sekunder")
    Pengeluaran_Tidak_Esensial: float = Field(..., alias="Pengeluaran Tidak Esensial")
    Pajak: float = Field(..., alias="Pajak")
    Asuransi: float = Field(..., alias="Asuransi")
    Sosial_Budaya: float = Field(..., alias="Sosial & Budaya")
    Tabungan_Investasi: float = Field(..., alias="Tabungan / Investasi")

    class Config:
        allow_population_by_field_name = True

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

json_key_map = {
    "Gaji": "Gaji",
    "Tabungan_Lama": "Tabungan Lama",
    "Investasi": "Investasi",
    "Pemasukan_Lainnya": "Pemasukan Lainnya",
    "Bahan_Pokok": "Bahan Pokok",
    "Protein_Gizi": "Protein & Gizi Tambahan",
    "Tempat_Tinggal": "Tempat Tinggal",
    "Sandang": "Sandang",
    "Konsumsi_Praktis": "Konsumsi Praktis",
    "Barang_Jasa_Sekunder": "Barang & Jasa Sekunder",
    "Pengeluaran_Tidak_Esensial": "Pengeluaran Tidak Esensial",
    "Pajak": "Pajak",
    "Asuransi": "Asuransi",
    "Sosial_Budaya": "Sosial & Budaya",
    "Tabungan_Investasi": "Tabungan / Investasi"
}

@app.route("/predict", methods=["POST"])
def predict_financial_behavior():
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "No JSON data provided"}), 400

        input_values = []
        missing_fields = []
        for feature_name_internal in feature_names:
            json_field_name = json_key_map.get(feature_name_internal)
            if json_field_name is None:
                return jsonify({"error": f"Internal configuration error for field: {feature_name_internal}"}), 500
            
            if json_field_name not in request_data:
                missing_fields.append(json_field_name)
            else:
                try:
                    input_values.append(float(request_data[json_field_name]))
                except ValueError:
                    return jsonify({"error": f"Field '{json_field_name}' must be a number."}), 400
        
        if missing_fields:
            return jsonify({"error": "Missing fields in input", "missing": missing_fields}), 400

        input_data_np = np.array([input_values])

        input_scaled = scaler.transform(input_data_np)
        probs = model.predict(input_scaled)
        predicted_class_index = np.argmax(probs, axis=1)[0]

        try:
            label = label_encoder.inverse_transform([predicted_class_index])[0]
        except IndexError:
            label = f"Unknown_Class_Index_{predicted_class_index}"
        except Exception as e_le:
            app.logger.error(f"Label encoder error: {e_le}")
            label = f"Error_Decoding_Class_Index_{predicted_class_index}"


        class_labels = label_encoder.classes_
        probabilities_dict = {
            str(class_labels[i]): float(probs[0][i])
            for i in range(len(class_labels))
        }

        return jsonify({
            "prediction": label,
            "probabilities": probabilities_dict
        })
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "An internal server error occurred", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)