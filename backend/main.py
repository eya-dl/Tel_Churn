"""API Telco Churn : prédiction, évaluation de performance et explicabilité.

Lancement (depuis la racine du projet) :
    uvicorn backend.main:app --reload
ou :
    python -m backend.main
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from . import service
from .schemas import FORM_OPTIONS, BatchRequest, PredictRequest

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("churn.api")

registry: service.Registry | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global registry
    registry = service.Registry()
    # Pré-calcule les probabilités de test pour que la 1re requête soit rapide.
    for bundle in registry.models.values():
        _ = bundle.test_proba
    log.info("Modèles chargés : %s (défaut : %s)", list(registry.models), registry.default_model)
    if not registry.consistency["ok"]:
        log.warning("Feature engineering ≠ telco_features.csv : %s", registry.consistency)
    yield


app = FastAPI(
    title="Telco Churn API",
    description="Prédiction du churn, évaluation des performances et explicabilité (SHAP).",
    version="2.0.0",
    lifespan=lifespan,
)

if service.REPORTS_DIR.exists():
    app.mount("/static/reports", StaticFiles(directory=service.REPORTS_DIR), name="reports")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _registry() -> service.Registry:
    if registry is None:
        raise HTTPException(503, "Le service n'est pas encore initialisé.")
    return registry


def _bundle(name: str | None) -> service.ModelBundle:
    try:
        return _registry().get(name)
    except KeyError:
        raise HTTPException(
            404, f"Modèle inconnu : {name!r}. Disponibles : {list(_registry().models)}"
        )


def _clean(value):
    """Valeur JSON-safe pour l'affichage (NaN -> None)."""
    return service._f(value) if isinstance(value, float) else value


# --------------------------------------------------------------------------
# Général
# --------------------------------------------------------------------------
@app.get("/", tags=["général"])
def home():
    return {"message": "Telco Churn API opérationnelle", "docs": "/docs", "version": app.version}


@app.get("/health", tags=["général"])
def health():
    reg = _registry()
    return {
        "status": "ok",
        "models": list(reg.models),
        "default_model": reg.default_model,
        "feature_engineering_consistent": reg.consistency["ok"],
    }


@app.get("/models", tags=["modèles"])
def list_models():
    reg = _registry()
    return {"default": reg.default_model, "models": list(reg.models)}


@app.get("/model/info", tags=["modèles"])
def model_info(model: str | None = None):
    reg = _registry()
    info = _bundle(model).info()
    info["feature_engineering_check"] = reg.consistency
    info["high_monthly_charges_threshold"] = reg.high_charges_threshold
    return info


@app.get("/schema", tags=["modèles"])
def schema():
    """Options du formulaire client et bornes numériques (dérivées des données)."""
    reg = _registry()
    return {"options": FORM_OPTIONS, **reg.form_schema()}


# --------------------------------------------------------------------------
# Prédiction
# --------------------------------------------------------------------------
@app.post("/predict", tags=["prédiction"])
def predict(request: PredictRequest):
    reg = _registry()
    bundle = _bundle(request.model)
    raw_norm, eng = reg.features_from_customers([request.customer.model_dump()])
    p = float(bundle.predict_proba(eng)[0])
    pred = int(p >= request.threshold)

    explanation = None
    if request.explain:
        explanation = bundle.explain_one(raw_norm.iloc[0], eng, request.top_k)
        if explanation:
            for c in explanation["contributions"]:
                c["value"] = _clean(c["value"])

    return {
        "model": bundle.name,
        "threshold": request.threshold,
        "prediction": pred,
        "label": "Churn" if pred else "No churn",
        "churn_probability": round(p, 4),
        "no_churn_probability": round(1 - p, 4),
        "risk_percentile": round(bundle.percentile_of(p), 4),
        "normalized_input": {
            k: _clean(v) for k, v in raw_norm.iloc[0].to_dict().items()
        },
        "explanation": explanation,
    }


@app.post("/predict/batch", tags=["prédiction"])
def predict_batch(request: BatchRequest):
    reg = _registry()
    bundle = _bundle(request.model)
    customers = [c.model_dump() for c in request.customers]
    _, eng = reg.features_from_customers(customers)
    proba = bundle.predict_proba(eng)
    flagged = proba >= request.threshold
    return {
        "model": bundle.name,
        "threshold": request.threshold,
        "predictions": [
            {
                "customerID": c.get("customerID"),
                "churn_probability": round(float(p), 4),
                "prediction": int(f),
            }
            for c, p, f in zip(customers, proba, flagged)
        ],
        "summary": {
            "n": len(customers),
            "flagged": int(flagged.sum()),
            "flagged_rate": float(flagged.mean()),
            "mean_probability": float(proba.mean()),
        },
    }


# --------------------------------------------------------------------------
# Évaluation (toujours sur le jeu de test reproduit du split d'entraînement)
# --------------------------------------------------------------------------
@app.get("/evaluation/metrics", tags=["évaluation"])
def evaluation_metrics(
    model: str | None = None,
    threshold: float = Query(0.5, gt=0, lt=1),
):
    bundle = _bundle(model)
    y = _registry().y_test.to_numpy()
    return {
        "model": bundle.name,
        "n_test": int(len(y)),
        "positives": int(y.sum()),
        **service.metrics_at(y, bundle.test_proba, threshold),
    }


@app.get("/evaluation/curves", tags=["évaluation"])
def evaluation_curves(model: str | None = None):
    bundle = _bundle(model)
    return {"model": bundle.name, **service.curves(_registry().y_test.to_numpy(), bundle.test_proba)}


@app.get("/evaluation/threshold", tags=["évaluation"])
def evaluation_threshold(
    model: str | None = None,
    cost_fn: float = Query(100.0, ge=0, description="Coût d'un churner non détecté (FN)"),
    cost_fp: float = Query(20.0, ge=0, description="Coût d'une alerte inutile (FP)"),
):
    bundle = _bundle(model)
    return {
        "model": bundle.name,
        **service.threshold_sweep(_registry().y_test.to_numpy(), bundle.test_proba, cost_fn, cost_fp),
    }


@app.get("/evaluation/compare", tags=["évaluation"])
def evaluation_compare(threshold: float = Query(0.5, gt=0, lt=1)):
    """Métriques recalculées en direct pour tous les modèles + rapport de référence."""
    reg = _registry()
    y = reg.y_test.to_numpy()
    live, roc = [], {}
    for name, bundle in reg.models.items():
        m = service.metrics_at(y, bundle.test_proba, threshold)
        live.append({"model": name, "estimator": type(bundle.estimator).__name__,
                     **{k: m[k] for k in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC", "Brier"]}})
        c = service.curves(y, bundle.test_proba)["roc"]
        roc[name] = {"fpr": c["fpr"], "tpr": c["tpr"]}
    reference = (
        reg.reference.to_dict(orient="records") if reg.reference is not None else []
    )
    return {"threshold": threshold, "live": live, "roc": roc, "reference_report": reference}


# --------------------------------------------------------------------------
# Explicabilité
# --------------------------------------------------------------------------
@app.get("/explain/global", tags=["explicabilité"])
def explain_global(model: str | None = None, top: int = Query(20, ge=1, le=60)):
    bundle = _bundle(model)
    if bundle.explainer is None:
        raise HTTPException(501, f"SHAP n'est pas disponible pour {bundle.name}.")
    return {"model": bundle.name, **bundle.explain_global(top)}


@app.get("/explain/dependence", tags=["explicabilité"])
def explain_dependence(
    feature: str = Query("tenure", pattern="^(tenure|MonthlyCharges|TotalCharges)$"),
    model: str | None = None,
):
    bundle = _bundle(model)
    if bundle.explainer is None:
        raise HTTPException(501, f"SHAP n'est pas disponible pour {bundle.name}.")
    return {"model": bundle.name, **bundle.dependence(feature)}


# --------------------------------------------------------------------------
# Données & rapports
# --------------------------------------------------------------------------
@app.get("/data/overview", tags=["données"])
def data_overview():
    return _registry().data_overview()


@app.get("/reports", tags=["données"])
def reports():
    """Liste des artefacts produits par les notebooks (images, HTML, CSV)."""
    return {"base_url": "/static/reports", "files": _registry().reports_listing()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
