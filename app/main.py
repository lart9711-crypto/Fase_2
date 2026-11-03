from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
import os

app = FastAPI(
    title="Servicio de Clasificación ML",
    description="API basada en FastAPI para inferencia con modelos scikit-learn (.pkl)",
    version="1.0.0"
)

#clases del modelo Iris
CLASES = ["Iris-Setosa", "Iris-Versicolor", "Iris-Virginica"]

# Cargar el modelo pickle
MODEL_PATH = "app/models/model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")

class PredictRequest(BaseModel):
    features: list[float] 

@app.get("/healthcheck", tags=["Estado"])
def healthcheck():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

@app.post("/predict", tags=["Predicción"])
def predict(payload: PredictRequest):
    if not model:
        raise HTTPException(
            status_code=500, 
            detail="Modelo no disponible o no encontrado en el servidor."
        )
    
    if len(payload.features) != 4:
        raise HTTPException(
            status_code=400, 
            detail="Se esperan exactamente 4 características numéricas para este modelo."
        )

    try:
        data = np.array(payload.features).reshape(1, -1)
        
        # Predicción
        prediction_idx = int(model.predict(data)[0])
        probabilities = model.predict_proba(data)[0].tolist()

        return {
            "prediction_class_index": prediction_idx,
            "prediction_label": CLASES[prediction_idx] if prediction_idx < len(CLASES) else "Desconocida",
            "confidence": round(max(probabilities), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error durante la inferencia: {str(e)}")