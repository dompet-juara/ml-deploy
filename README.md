# Financial Behavior Prediction API

A FastAPI-based REST API that predicts financial behavior using machine learning. This API analyzes various financial metrics to classify and predict spending patterns and financial behaviors.

## Features

- **Machine Learning Prediction**: Uses TensorFlow/Keras neural network for financial behavior classification
- **RESTful API**: Clean and well-documented API endpoints
- **Data Validation**: Robust input validation using Pydantic models
- **Scalable**: Built with FastAPI for high performance
- **Dockerized**: Ready for containerized deployment

## API Endpoints

### Health Check
- `GET /` - Root endpoint with basic API information
- `GET /health` - Health check endpoint to verify API and model status

### Prediction
- `POST /predict` - Predict financial behavior based on input data

## Input Fields

The API accepts the following financial data fields:

| Field Name | Description | Type | Example |
|------------|-------------|------|---------|
| `Gaji` | Monthly salary | float | 5000000.0 |
| `Tabungan Lama` | Old savings | float | 10000000.0 |
| `Investasi` | Investment value | float | 2000000.0 |
| `Pemasukan Lainnya` | Other income | float | 1000000.0 |
| `Bahan Pokok` | Basic necessities expenses | float | 1500000.0 |
| `Protein & Gizi Tambahan` | Protein and nutrition expenses | float | 500000.0 |
| `Tempat Tinggal` | Housing costs | float | 2000000.0 |
| `Sandang` | Clothing expenses | float | 300000.0 |
| `Konsumsi Praktis` | Practical consumption | float | 800000.0 |
| `Barang & Jasa Sekunder` | Secondary goods and services | float | 400000.0 |
| `Pengeluaran Tidak Esensial` | Non-essential expenses | float | 200000.0 |
| `Pajak` | Tax payments | float | 250000.0 |
| `Asuransi` | Insurance premiums | float | 300000.0 |
| `Sosial & Budaya` | Social and cultural expenses | float | 150000.0 |
| `Tabungan / Investasi` | New savings/investments | float | 500000.0 |

## Installation

### Prerequisites

- Python 3.9+
- pip
- Docker (optional)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd financial-behavior-prediction-api
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare model files**
   
   Ensure you have the following files in the `model/` directory:
   - `best_model.h5` - Trained TensorFlow/Keras model
   - `scaler.pkl` - Fitted StandardScaler or similar preprocessor
   - `label_encoder.pkl` - Fitted LabelEncoder for target classes

5. **Run the application**
   ```bash
   python main.py
   ```

   Or using uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 5000 --reload
   ```

### Docker Setup

1. **Build the Docker image**
   ```bash
   docker build -t financial-prediction-api .
   ```

2. **Run the container**
   ```bash
   docker run -p 5000:5000 -e PORT=5000 financial-prediction-api
   ```

## Usage Examples

### Example Request

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

### Example Response

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

### Python Client Example

```python
import requests

url = "http://localhost:5000/predict"
data = {
    "Gaji": 5000000.0,
    "Tabungan Lama": 10000000.0,
    "Investasi": 2000000.0,
    # ... other fields
}

response = requests.post(url, json=data)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Probabilities: {result['probabilities']}")
```

## API Documentation

Once the application is running, you can access:

- **Interactive API Documentation**: http://localhost:5000/docs
- **ReDoc Documentation**: http://localhost:5000/redoc
- **OpenAPI Schema**: http://localhost:5000/openapi.json

## Requirements

Create a `requirements.txt` file with the following dependencies:

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
tensorflow==2.15.0
numpy==1.24.3
scikit-learn==1.3.2
pydantic==2.5.0
python-multipart==0.0.6
```

## Model Training

This API expects pre-trained models. Ensure your model training pipeline:

1. Uses the same feature order as defined in `feature_names`
2. Saves the model in `.h5` format
3. Saves the scaler and label encoder as pickle files
4. Uses compatible TensorFlow/Keras versions

## Error Handling

The API includes comprehensive error handling for:

- Missing or invalid input fields
- Model loading errors
- Prediction errors
- Data validation errors

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.