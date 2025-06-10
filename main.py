from fastapi import FastAPI, HTTPException
import numpy as np
import tensorflow as tf
import pickle
from pydantic import BaseModel, Field
import logging
from typing import Dict
import uvicorn

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial Behavior Prediction API",
    description="API untuk prediksi perilaku keuangan menggunakan machine learning",
    version="1.0.0",
)

try:
    model = tf.keras.models.load_model("model/best_model.h5")

    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("model/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
except Exception as e:
    logger.error(f"Error loading model or preprocessors: {e}")
    raise

feature_names = [
    "Gaji",
    "Tabungan_Lama",
    "Investasi",
    "Pemasukan_Lainnya",
    "Bahan_Pokok",
    "Protein_Gizi",
    "Tempat_Tinggal",
    "Sandang",
    "Konsumsi_Praktis",
    "Barang_Jasa_Sekunder",
    "Pengeluaran_Tidak_Esensial",
    "Pajak",
    "Asuransi",
    "Sosial_Budaya",
    "Tabungan_Investasi",
]


class FinanceInput(BaseModel):
    """Model input untuk prediksi perilaku keuangan"""

    Gaji: float = Field(..., description="Gaji bulanan", example=5000000.0)
    Tabungan_Lama: float = Field(
        ..., alias="Tabungan Lama", description="Tabungan lama", example=10000000.0
    )
    Investasi: float = Field(..., description="Nilai investasi", example=2000000.0)
    Pemasukan_Lainnya: float = Field(
        ...,
        alias="Pemasukan Lainnya",
        description="Pemasukan dari sumber lain",
        example=1000000.0,
    )
    Bahan_Pokok: float = Field(
        ...,
        alias="Bahan Pokok",
        description="Pengeluaran bahan pokok",
        example=1500000.0,
    )
    Protein_Gizi: float = Field(
        ...,
        alias="Protein & Gizi Tambahan",
        description="Pengeluaran protein dan gizi",
        example=500000.0,
    )
    Tempat_Tinggal: float = Field(
        ...,
        alias="Tempat Tinggal",
        description="Biaya tempat tinggal",
        example=2000000.0,
    )
    Sandang: float = Field(..., description="Pengeluaran sandang", example=300000.0)
    Konsumsi_Praktis: float = Field(
        ..., alias="Konsumsi Praktis", description="Konsumsi praktis", example=800000.0
    )
    Barang_Jasa_Sekunder: float = Field(
        ...,
        alias="Barang & Jasa Sekunder",
        description="Pengeluaran barang dan jasa sekunder",
        example=400000.0,
    )
    Pengeluaran_Tidak_Esensial: float = Field(
        ...,
        alias="Pengeluaran Tidak Esensial",
        description="Pengeluaran tidak esensial",
        example=200000.0,
    )
    Pajak: float = Field(..., description="Pembayaran pajak", example=250000.0)
    Asuransi: float = Field(..., description="Premi asuransi", example=300000.0)
    Sosial_Budaya: float = Field(
        ...,
        alias="Sosial & Budaya",
        description="Pengeluaran sosial dan budaya",
        example=150000.0,
    )
    Tabungan_Investasi: float = Field(
        ...,
        alias="Tabungan / Investasi",
        description="Tabungan atau investasi baru",
        example=500000.0,
    )

    class Config:
        allow_population_by_field_name = True


class PredictionResponse(BaseModel):
    """Model response untuk hasil prediksi"""

    prediction: str = Field(..., description="Hasil prediksi perilaku keuangan")
    probabilities: Dict[str, float] = Field(
        ..., description="Probabilitas untuk setiap kelas"
    )


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
    "Tabungan_Investasi": "Tabungan / Investasi",
}


@app.get("/")
async def root():
    """Endpoint root untuk health check"""
    return {"message": "Financial Behavior Prediction API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict_financial_behavior(input_data: FinanceInput):
    """
    Prediksi perilaku keuangan berdasarkan input data keuangan

    - **input_data**: Data keuangan yang akan diprediksi
    - **return**: Hasil prediksi dan probabilitas untuk setiap kelas
    """
    try:
        request_data = input_data.dict(by_alias=True)

        input_values = []
        missing_fields = []

        for feature_name_internal in feature_names:
            json_field_name = json_key_map.get(feature_name_internal)
            if json_field_name is None:
                logger.error(
                    f"Internal configuration error for field: {feature_name_internal}"
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal configuration error for field: {feature_name_internal}",
                )

            if json_field_name not in request_data:
                missing_fields.append(json_field_name)
            else:
                try:
                    input_values.append(float(request_data[json_field_name]))
                except (ValueError, TypeError):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Field '{json_field_name}' must be a number.",
                    )

        if missing_fields:
            raise HTTPException(
                status_code=400, detail=f"Missing fields in input: {missing_fields}"
            )

        input_data_np = np.array([input_values])

        input_scaled = scaler.transform(input_data_np)

        probs = model.predict(input_scaled)
        predicted_class_index = np.argmax(probs, axis=1)[0]

        try:
            label = label_encoder.inverse_transform([predicted_class_index])[0]
        except IndexError:
            label = f"Unknown_Class_Index_{predicted_class_index}"
        except Exception as e_le:
            logger.error(f"Label encoder error: {e_le}")
            label = f"Error_Decoding_Class_Index_{predicted_class_index}"

        class_labels = label_encoder.classes_
        probabilities_dict = {
            str(class_labels[i]): float(probs[0][i]) for i in range(len(class_labels))
        }

        return PredictionResponse(prediction=label, probabilities=probabilities_dict)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"An internal server error occurred: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)
