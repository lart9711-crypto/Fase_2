from fastapi.testclient import TestClient
from app.main import app

# Inicializar el cliente de pruebas para FastAPI
client = TestClient(app)

def test_healthcheck():
    """Prueba que el endpoint de salud responda correctamente y confirme la carga del modelo."""
    response = client.get("/healthcheck")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    # Validamos que el modelo se haya cargado en memoria
    assert data["model_loaded"] is True

def test_predict_success():
    """Prueba un caso de predicción exitoso con un vector de 4 características válidas."""
    payload = {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    # Verificamos que la respuesta contenga las llaves esperadas
    assert "prediction_class_index" in data
    assert "prediction_label" in data
    assert "confidence" in data
    # Verificamos que la confianza sea un valor válido entre 0 y 1
    assert 0.0 <= data["confidence"] <= 1.0

def test_predict_invalid_dimensions():
    """Caso extremo: Envío de un vector con dimensiones incorrectas (menos de 4 valores)."""
    payload = {
        "features": [5.1, 3.5]  # Faltan características
    }
    response = client.post("/predict", json=payload)
    # Debe retornar un error de validación controlada (400 Bad Request)
    assert response.status_code == 400

def test_predict_invalid_data_types():
    """Caso extremo: Envío de tipos de datos incorrectos (cadenas de texto en lugar de números)."""
    payload = {
        "features": ["texto", 3.5, "1.4", 0.2]
    }
    response = client.post("/predict", json=payload)
    # Pydantic debe rechazarlo automáticamente con un error de validación (422 Unprocessable Entity)
    assert response.status_code == 422