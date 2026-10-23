
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field



PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT / "machine_learning" / "models" / "best_model.joblib"
)

DATA_PATH = (
    PROJECT_ROOT / "machine_learning" / "data" / "processed" / "telco_features.csv"
)

app = FastAPI(
    title="Telco Churn API",
    description="API de prédiction et d'explicabilité du churn client",
    version="1.0.0"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset introuvable : {DATA_PATH}")

model = joblib.load(MODEL_PATH)

# Retrouver les colonnes attendues par le modèle
df = pd.read_csv(DATA_PATH)

FEATURE_COLUMNS = [
    col for col in df.columns
    if col not in ["Churn", "customerID"]
]

# Types des features utilisés lors du training
FEATURE_DTYPES = df[FEATURE_COLUMNS].dtypes


class PredictionRequest(BaseModel):
    features: dict[str, float | int | str] = Field(
        description="Valeurs des features du client"
    )


@app.get("/")
def home():
    return {
        "message": "Telco Churn API opérationnelle",
        "docs": "/docs"
    }


@app.get("/features")
def get_features():
    """Décrire les features attendues par le modèle."""
    return {
        "features": [
            {
                "name": col,
                "type": str(FEATURE_DTYPES[col])
            }
            for col in FEATURE_COLUMNS
        ]
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    # Vérifier que les données correspondent au modèle
    missing = [
        col for col in FEATURE_COLUMNS
        if col not in request.features
    ]

    extra = [
        col for col in request.features
        if col not in FEATURE_COLUMNS
    ]

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": missing,
                "unexpected_features": extra
            }
        )

    try:
        # Garder exactement l'ordre des colonnes du training
        X = pd.DataFrame(
            [request.features],
            columns=FEATURE_COLUMNS
        )

        # Reconstruire les types numériques
        for col in FEATURE_COLUMNS:
            if pd.api.types.is_numeric_dtype(FEATURE_DTYPES[col]):
                X[col] = pd.to_numeric(X[col], errors="raise")

        probability = float(model.predict_proba(X)[0, 1])
        prediction = int(model.predict(X)[0])

        return {
            "prediction": prediction,
            "label": "Churn" if prediction == 1 else "No churn",
            "churn_probability": round(probability, 4),
            "no_churn_probability": round(1 - probability, 4)
        }

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Valeurs invalides : {exc}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
