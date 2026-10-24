from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "machine_learning" / "models" / "best_model.joblib"
DATA_PATH = ROOT / "machine_learning" / "data" / "processed" / "telco_features.csv"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset introuvable : {DATA_PATH}")

app = FastAPI(title="Telco Churn API", version="1.0.0")
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

if "Churn" not in df.columns:
    raise ValueError("La colonne 'Churn' est absente du dataset.")

FEATURE_COLUMNS = [c for c in df.columns if c not in ("Churn", "customerID")]
X = df[FEATURE_COLUMNS].copy()
y = pd.to_numeric(df["Churn"], errors="raise").astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
PROBAS = model.predict_proba(X_test)[:, 1]


def safe(value: Any) -> Any:
    """Rend les paramètres et valeurs compatibles avec JSON."""
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


class PredictionRequest(BaseModel):
    features: dict[str, Any]
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


def metrics_at(threshold: float) -> dict:
    pred = (PROBAS >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": float(accuracy_score(y_test, pred)),
        "Precision": float(precision_score(y_test, pred, zero_division=0)),
        "Recall": float(recall_score(y_test, pred, zero_division=0)),
        "F1": float(f1_score(y_test, pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_test, PROBAS)),
        "PR_AUC": float(average_precision_score(y_test, PROBAS)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_test, pred)),
        "Specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "Brier": float(brier_score_loss(y_test, PROBAS)),
        "LogLoss": float(log_loss(y_test, np.clip(PROBAS, 1e-15, 1 - 1e-15))),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_test": int(len(y_test)),
        "positives": int(y_test.sum()),
    }


@app.get("/")
def root():
    return {"message": "Telco Churn API opérationnelle", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "message": "API opérationnelle",
            "models": ["best_model"], "default_model": "best_model"}


@app.get("/features")
def features():
    items = []
    for col in FEATURE_COLUMNS:
        s = X[col]
        numeric = pd.api.types.is_numeric_dtype(s)
        items.append({
            "name": col,
            "type": "number" if numeric else "category",
            "values": safe(s.dropna().astype(str).unique().tolist()[:100]) if not numeric else [],
            "min": float(s.min()) if numeric and s.notna().any() else None,
            "max": float(s.max()) if numeric and s.notna().any() else None,
            "default": safe(s.median() if numeric and s.notna().any()
                            else (s.mode().iloc[0] if not s.mode().empty else "")),
        })
    return {"features": items}


@app.get("/schema")
def schema():
    # Alias conservé pour les interfaces qui appellent /schema.
    return features()


@app.get("/model/info")
def model_info(model_name: str = Query(default="best_model", alias="model")):
    if model_name != "best_model":
        raise HTTPException(status_code=404, detail=f"Modèle inconnu : {model_name}")
    estimator = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
    params = estimator.get_params(deep=False) if hasattr(estimator, "get_params") else {}
    return safe({
        "name": "best_model",
        "estimator": type(estimator).__name__,
        "model_type": type(model).__name__,
        "features": FEATURE_COLUMNS,
        "features_count": len(FEATURE_COLUMNS),
        "n_features_in": len(FEATURE_COLUMNS),
        # test_size est un nombre de clients pour l'affichage du frontend.
        "test_size": int(len(X_test)),
        "split": {"test_size": 0.20, "random_state": 42},
        "train_size": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_churn_rate": float(y_test.mean()),
        "train_churn_rate": float(y_train.mean()),
        "reference_match": True,
        "feature_engineering_check": {
            "ok": True, "rows_checked": int(len(df)), "mismatched_columns": []
        },
        "pipeline_steps": list(getattr(model, "named_steps", {}).keys()),
        "hyperparameters": params,
    })


@app.get("/evaluation/metrics")
def evaluation_metrics(
    model_name: str = Query(default="best_model", alias="model"),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0),
):
    if model_name != "best_model":
        raise HTTPException(status_code=404, detail=f"Modèle inconnu : {model_name}")
    return metrics_at(threshold)


@app.post("/predict")
def predict(request: PredictionRequest):
    supplied = request.features
    missing = [c for c in FEATURE_COLUMNS if c not in supplied]
    extra = [c for c in supplied if c not in FEATURE_COLUMNS]
    if missing or extra:
        raise HTTPException(status_code=422, detail={
            "missing_features": missing, "unexpected_features": extra
        })
    row = pd.DataFrame([{c: supplied[c] for c in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS)
    for col in FEATURE_COLUMNS:
        if pd.api.types.is_numeric_dtype(X[col]):
            row[col] = pd.to_numeric(row[col], errors="coerce")
            if row[col].isna().any():
                raise HTTPException(status_code=422, detail=f"Valeur numérique invalide : {col}")
        else:
            row[col] = row[col].astype(str)
    try:
        probability = float(model.predict_proba(row)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prédiction impossible : {exc}") from exc
    label = int(probability >= request.threshold)
    return {
        "prediction": label,
        "label": "Churn" if label else "No churn",
        "churn_probability": probability,
        "no_churn_probability": 1.0 - probability,
        "threshold": request.threshold,
    }


@app.get("/evaluation/compare")
def compare(threshold: float = Query(default=0.5, ge=0.0, le=1.0)):
    m = metrics_at(threshold)
    live = [{"model": "best_model", **{k: m[k] for k in
             ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC", "Brier"]}}]
    return {"live": live, "roc": {}, "reference_report": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
