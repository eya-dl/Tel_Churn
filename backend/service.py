"""Couche métier du projet Telco Churn.

- charge tous les modèles ``*.joblib`` du dossier ``machine_learning/models``
- reproduit le jeu de test de l'entraînement (split stratifié 80/20, seed 42)
- reconstruit les features à partir des colonnes brutes d'un client
- calcule métriques, courbes, analyse de seuil et explications SHAP
"""
from __future__ import annotations

import logging
import math
from functools import cached_property
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

log = logging.getLogger("churn.service")

# --------------------------------------------------------------------------
# Chemins et constantes d'entraînement
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "machine_learning"
MODELS_DIR = ML_DIR / "models"
RAW_PATH = ML_DIR / "Churn.csv"
PROCESSED_PATH = ML_DIR / "data" / "processed" / "telco_features.csv"
REPORTS_DIR = ML_DIR / "reports"
REFERENCE_METRICS_PATH = REPORTS_DIR / "model_comparison.csv"

TARGET = "Churn"
ID_COL = "customerID"
# Paramètres du split retrouvés en recalculant les métriques de
# model_comparison.csv (correspondance exacte sur les 6 métriques).
TEST_SIZE = 0.20
RANDOM_STATE = 42

ONEHOT_LEVELS: dict[str, list[str]] = {
    "InternetService": ["DSL", "Fiber optic", "No"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
}
ONEHOT_GROUPS = [*ONEHOT_LEVELS, "TenureGroup"]
INTERNET_ADDONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
BINARY_YES_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
]
TENURE_BINS = [-1, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-12 mois", "13-24 mois", "25-48 mois", "49-72 mois"]
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]


# --------------------------------------------------------------------------
# Utilitaires
# --------------------------------------------------------------------------
def _f(x) -> float | None:
    """float JSON-safe (NaN / inf -> None)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _thin(*arrays: np.ndarray, n: int = 300) -> list[list]:
    """Sous-échantillonne des courbes pour alléger les réponses JSON."""
    size = len(arrays[0])
    idx = np.arange(size) if size <= n else np.unique(np.linspace(0, size - 1, n).astype(int))
    return [[_f(v) for v in np.asarray(a)[idx]] for a in arrays]


def group_of(feature: str) -> str:
    """'Contract_One year' -> 'Contract' ; 'tenure' -> 'tenure'."""
    for g in ONEHOT_GROUPS:
        if feature.startswith(g + "_"):
            return g
    return feature


def strip_prefix(name: str) -> str:
    return name.split("__", 1)[1] if "__" in name else name


# --------------------------------------------------------------------------
# Feature engineering (colonnes brutes -> features du modèle)
# --------------------------------------------------------------------------
def normalize_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Applique les règles de cohérence du dataset Telco.

    Sans Internet, les 6 services additionnels valent "No internet service".
    Sans téléphone, MultipleLines vaut "No phone service".
    """
    raw = raw.copy()
    no_internet = raw["InternetService"].eq("No")
    for col in INTERNET_ADDONS:
        raw.loc[no_internet, col] = "No internet service"
    raw.loc[raw["PhoneService"].eq("No"), "MultipleLines"] = "No phone service"
    return raw


def engineer_features(raw: pd.DataFrame, high_charges_threshold: float) -> pd.DataFrame:
    """Reproduit le preprocessing du notebook (voir telco_features.csv)."""
    raw = normalize_raw(raw)
    out = pd.DataFrame(index=raw.index)
    out["gender"] = raw["gender"].eq("Male").astype(int)
    out["SeniorCitizen"] = raw["SeniorCitizen"].astype(int)
    for col in BINARY_YES_COLS:  # Yes -> 1 ; No / No service -> 0
        out[col] = raw[col].eq("Yes").astype(int)
    out["tenure"] = raw["tenure"].astype(int)
    out["MonthlyCharges"] = raw["MonthlyCharges"].astype(float)
    out["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    for col, levels in ONEHOT_LEVELS.items():
        for level in levels:
            out[f"{col}_{level}"] = raw[col].eq(level).astype(int)
    out["TenureGroup"] = pd.cut(
        out["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS
    ).astype(str)
    out["HighMonthlyCharges"] = (out["MonthlyCharges"] > high_charges_threshold).astype(int)
    return out


# --------------------------------------------------------------------------
# Métriques
# --------------------------------------------------------------------------
def metrics_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    def div(a, b):
        return a / b if b else 0.0

    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    return {
        "threshold": float(threshold),
        "Accuracy": div(tp + tn, len(y)),
        "Precision": precision,
        "Recall": recall,
        "Specificity": div(tn, tn + fp),
        "F1": div(2 * precision * recall, precision + recall),
        "Balanced_Accuracy": float(balanced_accuracy_score(y, pred)),
        "MCC": _f(matthews_corrcoef(y, pred)),
        "ROC_AUC": float(roc_auc_score(y, p)),
        "PR_AUC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "LogLoss": float(log_loss(y, p)),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "flagged_rate": div(tp + fp, len(y)),
    }


def threshold_sweep(y: np.ndarray, p: np.ndarray, cost_fn: float, cost_fp: float) -> dict:
    thresholds = np.round(np.arange(0.02, 0.99, 0.01), 2)
    rows = []
    for t in thresholds:
        m = metrics_at(y, p, t)
        c = m["confusion"]
        rows.append({
            "threshold": float(t),
            "precision": m["Precision"],
            "recall": m["Recall"],
            "f1": m["F1"],
            "accuracy": m["Accuracy"],
            "specificity": m["Specificity"],
            "fn": c["fn"], "fp": c["fp"],
            "cost": c["fn"] * cost_fn + c["fp"] * cost_fp,
        })
    df = pd.DataFrame(rows)
    youden = df["recall"] + df["specificity"] - 1
    best = {
        "f1": float(df.loc[df["f1"].idxmax(), "threshold"]),
        "youden": float(df.loc[youden.idxmax(), "threshold"]),
        "min_cost": float(df.loc[df["cost"].idxmin(), "threshold"]),
    }
    return {"rows": rows, "best": best, "cost_fn": cost_fn, "cost_fp": cost_fp}


def curves(y: np.ndarray, p: np.ndarray) -> dict:
    fpr, tpr, roc_thr = roc_curve(y, p)
    prec, rec, pr_thr = precision_recall_curve(y, p)
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")

    order = np.argsort(-p)
    y_sorted = np.asarray(y)[order]
    cum_pos = np.cumsum(y_sorted) / y_sorted.sum()
    frac = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    marks = np.round(np.arange(0.05, 1.0001, 0.05), 2)
    gains = []
    for m in marks:
        k = max(int(round(m * len(y_sorted))), 1)
        captured = float(cum_pos[k - 1])
        gains.append({"fraction": float(m), "captured": captured, "lift": captured / m})

    bins = np.linspace(0, 1, 21)
    hist0, _ = np.histogram(p[np.asarray(y) == 0], bins=bins)
    hist1, _ = np.histogram(p[np.asarray(y) == 1], bins=bins)

    return {
        "roc": dict(zip(["fpr", "tpr", "thresholds"],
                        _thin(fpr, tpr, np.clip(roc_thr, 0, 1)))),
        "pr": {
            "precision": _thin(prec[:-1])[0], "recall": _thin(rec[:-1])[0],
            "thresholds": _thin(pr_thr)[0],
            "baseline": float(np.mean(y)),
        },
        "calibration": {
            "mean_predicted": [float(v) for v in mean_pred],
            "fraction_positive": [float(v) for v in frac_pos],
        },
        "gains": gains,
        "gains_curve": dict(zip(["fraction", "captured"], _thin(frac, cum_pos))),
        "score_distribution": {
            "bin_centers": [float(v) for v in (bins[:-1] + bins[1:]) / 2],
            "no_churn": [int(v) for v in hist0],
            "churn": [int(v) for v in hist1],
        },
    }


# --------------------------------------------------------------------------
# Un modèle chargé + ses caches (probabilités de test, SHAP)
# --------------------------------------------------------------------------
class ModelBundle:
    def __init__(self, name: str, path: Path, registry: "Registry"):
        self.name = name
        self.path = path
        self.registry = registry
        self.pipeline = joblib.load(path)
        self.feature_names: list[str] = list(
            getattr(self.pipeline, "feature_names_in_", registry.all_features)
        )

    # -- informations -------------------------------------------------------
    @property
    def estimator(self):
        return self.pipeline.steps[-1][1] if hasattr(self.pipeline, "steps") else self.pipeline

    def info(self) -> dict:
        est = self.estimator
        params = {}
        for k, v in est.get_params().items():
            if isinstance(v, float) and not math.isfinite(v):
                continue  # ex. XGBoost : missing=NaN, non sérialisable en JSON
            if isinstance(v, (int, float, str, bool)):
                params[k] = v
        reg = self.registry
        return {
            "name": self.name,
            "file": self.path.name,
            "estimator": type(est).__name__,
            "pipeline_steps": [s for s, _ in getattr(self.pipeline, "steps", [])],
            "n_features_in": len(self.feature_names),
            "hyperparameters": params,
            "explainable": self.explainer is not None,
            "train_size": int(len(reg.X_train)),
            "test_size": int(len(reg.X_test)),
            "train_churn_rate": float(reg.y_train.mean()),
            "test_churn_rate": float(reg.y_test.mean()),
            "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "stratified": True},
            "reference_match": self.reference_match(),
        }

    def reference_match(self) -> str | None:
        """Nom de la ligne de model_comparison.csv que ce modèle reproduit."""
        ref = self.registry.reference
        if ref is None:
            return None
        m = metrics_at(self.registry.y_test.to_numpy(), self.test_proba, 0.5)
        for _, row in ref.iterrows():
            if all(abs(m[k] - float(row[k])) < 1e-6 for k in METRIC_NAMES if k in row):
                return str(row["Model"])
        return None

    # -- prédiction -----------------------------------------------------------
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(features[self.feature_names])[:, 1]

    @cached_property
    def test_proba(self) -> np.ndarray:
        return self.predict_proba(self.registry.X_test)

    def percentile_of(self, p: float) -> float:
        """Part des clients du jeu de test dont le score est inférieur à p."""
        return float(np.mean(self.test_proba < p))

    # -- explicabilité --------------------------------------------------------
    @cached_property
    def explainer(self):
        try:
            return shap.TreeExplainer(self.estimator)
        except Exception as exc:  # modèle non arborescent
            log.warning("SHAP indisponible pour %s : %s", self.name, exc)
            return None

    @cached_property
    def _preprocessor(self):
        return self.pipeline[:-1]

    @cached_property
    def out_names(self) -> list[str]:
        return [strip_prefix(n) for n in self._preprocessor.get_feature_names_out()]

    def _transform(self, features: pd.DataFrame) -> np.ndarray:
        Xt = self._preprocessor.transform(features[self.feature_names])
        return Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)

    def _shap(self, features: pd.DataFrame) -> np.ndarray:
        values = self.explainer.shap_values(self._transform(features))
        if isinstance(values, list):  # anciennes versions de shap
            values = values[-1]
        return np.asarray(values)

    @cached_property
    def shap_test(self) -> np.ndarray:
        return self._shap(self.registry.X_test)

    def explain_global(self, top: int = 20) -> dict:
        sv = self.shap_test
        names = self.out_names
        per_feature = pd.Series(np.abs(sv).mean(axis=0), index=names)
        groups = pd.Series([group_of(n) for n in names])
        grouped_abs = (
            pd.DataFrame(sv, columns=names).T.groupby(groups.to_numpy()).sum().T.abs().mean()
        )
        return {
            "by_feature": [
                {"feature": k, "mean_abs_shap": float(v)}
                for k, v in per_feature.sort_values(ascending=False).head(top).items()
            ],
            "by_group": [
                {"feature": k, "mean_abs_shap": float(v)}
                for k, v in grouped_abs.sort_values(ascending=False).head(top).items()
            ],
            "unit": "log-odds (impact moyen absolu sur le score)",
            "n_samples": int(len(sv)),
        }

    def dependence(self, feature: str) -> dict:
        X = self.registry.X_test
        col = self.out_names.index(feature)
        x = X[feature].to_numpy(dtype=float)
        s = self.shap_test[:, col]
        mask = ~np.isnan(x)
        x, s = x[mask], s[mask]
        trend = (
            pd.DataFrame({"x": x, "shap": s})
            .assign(bin=lambda d: pd.qcut(d["x"], q=12, duplicates="drop"))
            .groupby("bin", observed=True)
            .agg(x=("x", "mean"), shap=("shap", "mean"))
        )
        rng = np.random.default_rng(0)
        idx = rng.choice(len(x), size=min(1000, len(x)), replace=False)
        return {
            "feature": feature,
            "points": {"x": [_f(v) for v in x[idx]], "shap": [_f(v) for v in s[idx]]},
            "trend": {"x": [_f(v) for v in trend["x"]], "shap": [_f(v) for v in trend["shap"]]},
        }

    def explain_one(self, raw_row: pd.Series, eng_row: pd.DataFrame, top_k: int) -> dict | None:
        """Contributions SHAP d'un client, regroupées par variable d'origine."""
        if self.explainer is None:
            return None
        sv = self._shap(eng_row)[0]
        base = float(np.ravel(self.explainer.expected_value)[0])
        frame = pd.DataFrame({"name": self.out_names, "shap": sv})
        frame["group"] = frame["name"].map(group_of)
        grouped = frame.groupby("group", sort=False)["shap"].sum()

        eng = eng_row.iloc[0]
        contributions = []
        for group, value in grouped.items():
            shown = raw_row[group] if group in raw_row.index else eng[group]
            contributions.append({
                "feature": group,
                "value": shown.item() if hasattr(shown, "item") else shown,
                "shap": float(value),
            })
        contributions.sort(key=lambda c: abs(c["shap"]), reverse=True)
        margin = base + float(sv.sum())
        return {
            "base_value": base,
            "base_probability": float(1 / (1 + np.exp(-base))),
            "output_value": margin,
            "unit": "log-odds : > 0 augmente le risque de churn, < 0 le diminue",
            "contributions": contributions[:top_k],
            "other_contribution": float(sum(c["shap"] for c in contributions[top_k:])),
        }


# --------------------------------------------------------------------------
# Registre : données + modèles
# --------------------------------------------------------------------------
class Registry:
    def __init__(self) -> None:
        for path in (RAW_PATH, PROCESSED_PATH):
            if not path.exists():
                raise FileNotFoundError(f"Fichier introuvable : {path}")
        if not MODELS_DIR.exists() or not list(MODELS_DIR.glob("*.joblib")):
            raise FileNotFoundError(f"Aucun modèle .joblib dans {MODELS_DIR}")

        self.raw = pd.read_csv(RAW_PATH)
        self.processed = pd.read_csv(PROCESSED_PATH)
        self.all_features = [c for c in self.processed.columns if c not in (ID_COL, TARGET)]
        # Seuil de HighMonthlyCharges = médiane des frais mensuels (vérifié
        # ci-dessous sur les 7 043 lignes).
        self.high_charges_threshold = float(self.processed["MonthlyCharges"].median())

        X, y = self.processed[self.all_features], self.processed[TARGET]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        self.reference = (
            pd.read_csv(REFERENCE_METRICS_PATH) if REFERENCE_METRICS_PATH.exists() else None
        )
        self.consistency = self._check_feature_consistency()

        self.models: dict[str, ModelBundle] = {}
        for path in sorted(MODELS_DIR.glob("*.joblib")):
            try:
                self.models[path.stem] = ModelBundle(path.stem, path, self)
            except Exception as exc:
                log.error("Modèle %s ignoré (chargement impossible) : %s", path.name, exc)
        if not self.models:
            raise RuntimeError(
                "Aucun modèle n'a pu être chargé. Vérifie que scikit-learn et xgboost "
                "ont la même version que celle utilisée pour l'entraînement."
            )
        self.default_model = "best_model" if "best_model" in self.models else next(iter(self.models))

    def get(self, name: str | None) -> ModelBundle:
        key = name or self.default_model
        if key not in self.models:
            raise KeyError(key)
        return self.models[key]

    def _check_feature_consistency(self) -> dict:
        """Vérifie que engineer_features() redonne exactement telco_features.csv."""
        eng = engineer_features(self.raw.set_index(ID_COL), self.high_charges_threshold)
        ref = self.processed.set_index(ID_COL).loc[eng.index]
        mismatched = []
        for col in self.all_features:
            a, b = eng[col], ref[col]
            if a.dtype.kind in "fi" and b.dtype.kind in "fi":
                same = np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)
            else:
                same = bool((a.astype(str) == b.astype(str)).all())
            if not same:
                mismatched.append(col)
        return {"ok": not mismatched, "mismatched_columns": mismatched, "rows_checked": int(len(eng))}

    # -- construction des features d'un ou plusieurs clients -----------------
    def features_from_customers(self, customers: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
        raw = pd.DataFrame(customers)
        raw_norm = normalize_raw(raw)
        return raw_norm, engineer_features(raw, self.high_charges_threshold)

    # -- jeux de données ------------------------------------------------------
    def data_overview(self) -> dict:
        raw = self.raw.copy()
        raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")
        raw["is_churn"] = raw[TARGET].eq("Yes").astype(int)
        raw["TenureGroup"] = raw[ID_COL].map(self.processed.set_index(ID_COL)["TenureGroup"])
        raw["SeniorCitizen"] = raw["SeniorCitizen"].map({0: "No", 1: "Yes"})

        def churn_by(col: str) -> list[dict]:
            g = raw.groupby(col, observed=True)["is_churn"].agg(["mean", "count"]).reset_index()
            return [
                {"category": str(r[col]), "customers": int(r["count"]), "churn_rate": float(r["mean"])}
                for _, r in g.iterrows()
            ]

        def hist(col: str, bins) -> dict:
            edges = np.asarray(bins, dtype=float)
            out = {"bin_centers": [float(v) for v in (edges[:-1] + edges[1:]) / 2]}
            for label, key in (("No", "no_churn"), ("Yes", "churn")):
                h, _ = np.histogram(raw.loc[raw[TARGET] == label, col].dropna(), bins=edges)
                out[key] = [int(v) for v in h]
            return out

        return {
            "n_customers": int(len(raw)),
            "churn_rate": float(raw["is_churn"].mean()),
            "avg_tenure": float(raw["tenure"].mean()),
            "avg_monthly_charges": float(raw["MonthlyCharges"].mean()),
            "missing_total_charges": int(raw["TotalCharges"].isna().sum()),
            "churn_by": {
                col: churn_by(col)
                for col in ["Contract", "InternetService", "PaymentMethod", "TenureGroup",
                            "SeniorCitizen", "PaperlessBilling", "TechSupport"]
            },
            "hist_tenure": hist("tenure", np.arange(0, 78, 6)),
            "hist_monthly_charges": hist("MonthlyCharges", np.linspace(15, 120, 22)),
        }

    def form_schema(self) -> dict:
        r = self.raw.copy()
        r["TotalCharges"] = pd.to_numeric(r["TotalCharges"], errors="coerce")
        return {
            "numeric": {
                "tenure": {"min": 0, "max": int(r["tenure"].max()), "default": int(r["tenure"].median())},
                "MonthlyCharges": {"min": 0.0, "max": float(r["MonthlyCharges"].max()) * 1.25,
                                   "default": float(r["MonthlyCharges"].median())},
                "TotalCharges": {"min": 0.0, "max": float(r["TotalCharges"].max()) * 1.25,
                                 "default": float(r["TotalCharges"].median())},
            },
            "high_monthly_charges_threshold": self.high_charges_threshold,
        }

    def reports_listing(self) -> list[dict]:
        if not REPORTS_DIR.exists():
            return []
        return [
            {"path": p.relative_to(REPORTS_DIR).as_posix(), "kind": p.suffix.lstrip(".").lower(),
             "size_kb": round(p.stat().st_size / 1024, 1)}
            for p in sorted(REPORTS_DIR.rglob("*"))
            if p.is_file() and p.suffix.lower() in {".png", ".html", ".csv"}
        ]
