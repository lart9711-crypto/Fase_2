# Servicio de Inferencia de Machine Learning con FastAPI y Docker

API REST desarrollada con FastAPI y contenedorizada con Docker, diseñada para servir un modelo de clasificación basado en `scikit-learn` serializado en formato `.pkl`.

---

# Estructura del Proyecto

```text
proyecto-ml-docker/
├── app/
│   ├── main.py              # Lógica de la API y endpoints (FastAPI)
│   └── models/
│       └── model.pkl        # Modelo de clasificación preentrenado (.pkl)
├── tests/
│   └── test_api.py          # Pruebas unitarias e integración
├── Dockerfile               # Configuración del contenedor de producción
├── requirements.txt         # Dependencias del proyecto
└── README.md                # Documentación y manual del sistema

Proceso de Despliegue

cd proyecto-ml-docker
pip install -r requirements.txt
python entrenar_modelo.py
docker build -t ml-classifier-api:latest .
docker run -d -p 8000:8000 --name ml_service_container ml-classifier-api:latest 
Si el build falla 
2. uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

