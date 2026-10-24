
from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = os.getenv(
    "CHURN_API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

st.set_page_config(
    page_title="Telco Churn — Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Telco Customer Churn Prediction")
st.caption("Tableau de bord de prédiction du départ des clients.")


# =========================================================
# API
# =========================================================

class ApiError(Exception):
    pass


def api_get(path: str, **params) -> Any:
    try:
        response = requests.get(
            f"{API_URL}{path}",
            params=params,
            timeout=60,
        )
    except requests.RequestException as exc:
        raise ApiError(
            f"Impossible de joindre l'API : {exc}"
        ) from exc

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text

        raise ApiError(f"HTTP {response.status_code} — {detail}")

    return response.json()


def api_post(path: str, payload: dict) -> Any:
    try:
        response = requests.post(
            f"{API_URL}{path}",
            json=payload,
            timeout=120,
        )
    except requests.RequestException as exc:
        raise ApiError(
            f"Impossible de joindre l'API : {exc}"
        ) from exc

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text

        raise ApiError(f"HTTP {response.status_code} — {detail}")

    return response.json()


def pct(value: float) -> str:
    return f"{value * 100:.1f} %"


def is_binary_feature(item: dict) -> bool:
    """
    Détecte une variable binaire à partir des valeurs min/max
    communiquées par l'API.

    Sont considérées comme binaires les variables dont les
    valeurs observées sont uniquement 0, uniquement 1,
    ou un mélange de 0 et de 1.
    """
    if item.get("type") != "number":
        return False

    lo = item.get("min")
    hi = item.get("max")

    if lo is None or hi is None:
        return False

    try:
        lo = float(lo)
        hi = float(hi)
    except (TypeError, ValueError):
        return False

    return lo in (0.0, 1.0) and hi in (0.0, 1.0)


# =========================================================
# CHARGEMENT DES INFORMATIONS DE L'API
# =========================================================

try:
    health = api_get("/health")
    model_name = health.get("default_model", "best_model")

    info = api_get("/model/info", model=model_name)
    schema = api_get("/features")
    metric = api_get(
        "/evaluation/metrics",
        model=model_name,
        threshold=0.5,
    )

    items = schema["features"]

except (ApiError, KeyError, TypeError) as exc:
    st.error(f"Erreur lors du chargement des données : {exc}")
    st.info(
        "Depuis la racine du projet, lance :\n\n"
        "`python -m uvicorn backend.main:app --reload`"
    )
    st.stop()


# =========================================================
# ONGLETS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    ["Aperçu", "Prédire un client", "Prédiction CSV"]
)


# =========================================================
# ONGLET 1 : APERÇU
# =========================================================

with tab1:
    st.subheader("Performance sur le jeu de test")

    metric_columns = st.columns(6)

    metric_labels = [
        ("Accuracy", "Accuracy"),
        ("Precision", "Precision"),
        ("Recall", "Recall"),
        ("F1", "F1"),
        ("ROC AUC", "ROC_AUC"),
        ("PR AUC", "PR_AUC"),
    ]

    for col, (label, key) in zip(metric_columns, metric_labels):
        col.metric(label, f"{metric[key]:.3f}")

    confusion = metric["confusion"]

    fig = go.Figure(
        data=go.Heatmap(
            z=[
                [confusion["tn"], confusion["fp"]],
                [confusion["fn"], confusion["tp"]],
            ],
            x=["Prédit : No churn", "Prédit : Churn"],
            y=["Réel : No churn", "Réel : Churn"],
            text=[
                [confusion["tn"], confusion["fp"]],
                [confusion["fn"], confusion["tp"]],
            ],
            texttemplate="%{text}",
            showscale=False,
        )
    )

    fig.update_layout(
        title="Matrice de confusion",
        height=350,
    )

    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    left.write(
        f"**Modèle :** {info['name']} — {info['estimator']}"
    )
    left.write(f"**Variables :** {info['n_features_in']}")
    left.write(
        f"**Clients d'entraînement :** {info['train_size']:,}"
    )

    right.write(f"**Clients de test :** {info['test_size']:,}")
    right.write(
        f"**Taux de churn du test :** "
        f"{pct(info['test_churn_rate'])}"
    )

    split = info.get("split", {})
    test_size = float(split.get("test_size", 0.2))

    right.write(
        f"**Split :** {int((1 - test_size) * 100)}/"
        f"{int(test_size * 100)}"
    )

    with st.expander("Hyperparamètres"):
        st.json(info.get("hyperparameters", {}))


# =========================================================
# ONGLET 2 : PRÉDICTION D'UN CLIENT
# =========================================================

with tab2:
    st.subheader("Variables d'entrée du modèle")

    st.info(
        "Les variables binaires, notamment celles issues du "
        "One-Hot Encoding, sont proposées avec les choix 0 et 1. "
        "Les autres variables numériques conservent leur champ "
        "numérique."
    )

    inputs = {}

    with st.form("predict_form"):
        cols = st.columns(3)

        for i, item in enumerate(items):
            name = item["name"]

            with cols[i % 3]:
                lo = item.get("min")
                hi = item.get("max")
                default = item.get("default")

                # -----------------------------------------
                # VARIABLES BINAIRES : 0 OU 1
                # -----------------------------------------
                if is_binary_feature(item):
                    try:
                        default_value = (
                            0 if default is None else int(float(default))
                        )
                    except (TypeError, ValueError):
                        default_value = 0

                    default_value = (
                        default_value if default_value in (0, 1) else 0
                    )

                    inputs[name] = st.selectbox(
                        name,
                        options=[0, 1],
                        index=default_value,
                        format_func=lambda value: str(value),
                        key=f"input_{name}",
                    )

                # -----------------------------------------
                # AUTRES VARIABLES NUMÉRIQUES
                # -----------------------------------------
                elif item.get("type") == "number":
                    try:
                        lower = float(lo) if lo is not None else 0.0
                        upper = (
                            float(hi)
                            if hi is not None
                            else lower + 1.0
                        )
                        default_value = (
                            float(default)
                            if default is not None
                            else lower
                        )
                    except (TypeError, ValueError):
                        lower, upper, default_value = 0.0, 1.0, 0.0

                    if upper <= lower:
                        inputs[name] = st.number_input(
                            name,
                            value=default_value,
                            key=f"input_{name}",
                        )
                    else:
                        default_value = min(
                            max(default_value, lower),
                            upper,
                        )

                        inputs[name] = st.number_input(
                            name,
                            min_value=lower,
                            max_value=upper,
                            value=default_value,
                            key=f"input_{name}",
                        )

                # -----------------------------------------
                # VARIABLES CATÉGORIELLES
                # -----------------------------------------
                else:
                    values = item.get("values") or [""]
                    default_value = str(
                        default if default is not None else values[0]
                    )
                    values_as_strings = [str(value) for value in values]

                    index = (
                        values_as_strings.index(default_value)
                        if default_value in values_as_strings
                        else 0
                    )

                    inputs[name] = st.selectbox(
                        name,
                        options=values,
                        index=index,
                        key=f"input_{name}",
                    )

        threshold = st.slider(
            "Seuil de décision",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.01,
            key="single_threshold",
        )

        submitted = st.form_submit_button(
            "Calculer le risque",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            result = api_post(
                "/predict",
                {
                    "features": inputs,
                    "threshold": threshold,
                },
            )

            probability = float(result["churn_probability"])
            probability = min(max(probability, 0.0), 1.0)

            col_a, col_b = st.columns(2)

            col_a.metric(
                "Probabilité de churn",
                pct(probability),
            )
            col_b.metric("Décision", result["label"])

            if result["prediction"] == 1:
                st.error(
                    f"Client signalé comme à risque "
                    f"(seuil {threshold:.2f})."
                )
            else:
                st.success(
                    f"Probabilité sous le seuil de risque "
                    f"({threshold:.2f})."
                )

            st.progress(probability)

        except (ApiError, KeyError, TypeError, ValueError) as exc:
            st.error(f"Erreur lors de la prédiction : {exc}")


# =========================================================
# ONGLET 3 : PRÉDICTION CSV
# =========================================================

with tab3:
    st.subheader("Scorer un fichier CSV")

    st.write(
        "Le fichier CSV doit contenir toutes les colonnes "
        "de caractéristiques utilisées par le modèle."
    )

    uploaded = st.file_uploader(
        "Choisir un fichier CSV",
        type=["csv"],
    )

    batch_threshold = st.slider(
        "Seuil",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
        key="csv_threshold",
    )

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
            required = [item["name"] for item in items]

            missing = [
                column
                for column in required
                if column not in data.columns
            ]

            if missing:
                st.error(
                    "Colonnes manquantes : "
                    + ", ".join(missing)
                )

            elif data.empty:
                st.warning("Le fichier CSV ne contient aucune ligne.")

            elif len(data) > 10000:
                st.error(
                    "Le fichier est limité à 10 000 lignes."
                )

            elif st.button(
                "Lancer les prédictions",
                type="primary",
                key="run_csv_predictions",
            ):
                output = data.copy()
                probabilities = []
                labels = []

                progress = st.progress(0.0)
                status = st.empty()

                for i, (_, row) in enumerate(
                    data[required].iterrows(),
                    start=1,
                ):
                    result = api_post(
                        "/predict",
                        {
                            "features": row.to_dict(),
                            "threshold": batch_threshold,
                        },
                    )

                    probabilities.append(
                        result["churn_probability"]
                    )
                    labels.append(result["prediction"])

                    progress.progress(i / len(data))
                    status.write(
                        f"Traitement : {i}/{len(data)} clients"
                    )

                output["churn_probability"] = probabilities
                output["prediction"] = labels

                st.success("Prédictions terminées.")

                st.dataframe(
                    output,
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "Télécharger les résultats CSV",
                    data=output.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="telco_churn_predictions.csv",
                    mime="text/csv",
                )

        except (
            pd.errors.ParserError,
            UnicodeDecodeError,
            pd.errors.EmptyDataError,
        ) as exc:
            st.error(f"Erreur de lecture du CSV : {exc}")

        except (ApiError, KeyError, TypeError, ValueError) as exc:
            st.error(f"Erreur pendant le traitement : {exc}")
