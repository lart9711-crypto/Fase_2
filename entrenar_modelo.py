import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    print("1. Cargando datos de entrenamiento...")
    data = load_iris()
    X, y = data.data, data.target

    print("2. Entrenando el modelo de clasificación (Random Forest)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Crear la carpeta de modelos si no existe
    os.makedirs("app/models", exist_ok=True)
    
    # Ruta donde se guardará el pickle
    model_path = "app/models/model.pkl"
    
    print(f"3. Guardando el modelo entrenado en: {model_path}")
    joblib.dump(model, model_path)
    
    print("¡Modelo entrenado y serializado con éxito!")
    print(f"Clases que predice este modelo: {list(data.target_names)}")

if __name__ == "__main__":
    main()