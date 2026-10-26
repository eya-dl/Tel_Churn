"""Interface Streamlit : visualisation et évaluation du modèle Telco Churn.

Lancement (API démarrée dans un autre terminal) :
    streamlit run frontend/app.py
L'URL de l'API se règle avec la variable d'environnement CHURN_API_URL.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("CHURN_API_URL", "http://127.0.0.1:8000").rstrip("/")
C_CHURN, C_OK, C_NEUTRAL = "#E4572E", "#2E86AB", "#8D99AE"

st.set_page_config(page_title="Telco Churn — Évaluation du modèle", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: 12px 16px; border-radius: 10px;}
    .block-container {padding-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# Accès à l'API
# ==========================================================================
class ApiError(Exception):
    pass


def _detail(resp: requests.Response) -> str:
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text[:300]
    if isinstance(detail, list):  # erreurs de validation Pydantic
        return " ; ".join(f"{'.'.join(str(x) for x in d.get('loc', [])[1:])}: {d.get('msg')}" for d in detail[:5])
    return str(detail)


@st.cache_data(ttl=120, show_spinner=False)
def _get(path: str, params: tuple = ()) -> Any:
    resp = requests.get(f"{API_URL}{path}", params=dict(params), timeout=60)
    if not resp.ok:
        raise ApiError(f"{resp.status_code} — {_detail(resp)}")
    return resp.json()


def api_get(path: str, **params) -> Any:
    return _get(path, tuple(sorted((k, v) for k, v in params.items() if v is not None)))


def api_post(path: str, payload: dict) -> Any:
    resp = requests.post(f"{API_URL}{path}", json=payload, timeout=120)
    if not resp.ok:
        raise ApiError(f"{resp.status_code} — {_detail(resp)}")
    return resp.json()


@st.cache_data(ttl=600, show_spinner=False)
def api_file(url_path: str) -> bytes:
    resp = requests.get(f"{API_URL}{url_path}", timeout=60)
    resp.raise_for_status()
    return resp.content


# ==========================================================================
# Utilitaires d'affichage
# ==========================================================================
def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f} %"


def style(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.2), **layout)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch")


def vline(fig: go.Figure, x: float, label: str) -> None:
    fig.add_vline(x=x, line_dash="dash", line_color="gray", annotation_text=label,
                  annotation_position="top")


def confusion_fig(c: dict) -> go.Figure:
    z = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    rates = z / z.sum(axis=1, keepdims=True)
    text = [[f"<b>{z[i][j]:,}</b><br>{rates[i][j]:.1%} de la ligne" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=rates, x=["Prédit : No churn", "Prédit : Churn"], y=["Réel : No churn", "Réel : Churn"],
        text=text, texttemplate="%{text}", colorscale=[[0, "#eef3f7"], [1, C_OK]],
        showscale=False, zmin=0, zmax=1,
    ))
    fig.update_yaxes(autorange="reversed")
    return style(fig, 340, title="Matrice de confusion")


# ==========================================================================
# Pages
# ==========================================================================
def page_overview(ctx: dict) -> None:
    info = api_get("/model/info", model=ctx["model"])
    m = api_get("/evaluation/metrics", model=ctx["model"], threshold=ctx["threshold"])
    c = m["confusion"]

    st.title("📊 Telco Churn — Tableau de bord du modèle")
    st.caption(f"Modèle **{info['name']}** ({info['estimator']}) évalué sur le jeu de test "
               f"({info['test_size']:,} clients) au seuil de décision **{ctx['threshold']:.2f}**.")

    b1, b2, b3 = st.columns(3)
    if info["reference_match"]:
        b1.success(f"Métriques identiques à la ligne « {info['reference_match']} » de model_comparison.csv")
    else:
        b1.warning("Ne correspond à aucune ligne de model_comparison.csv")
    chk = info["feature_engineering_check"]
    if chk["ok"]:
        b2.success(f"Features de l'API = telco_features.csv ({chk['rows_checked']:,} lignes)")
    else:
        b2.error(f"Écart de features : {', '.join(chk['mismatched_columns'])}")
    b3.info(f"Split : {int((1 - info['split']['test_size']) * 100)} / "
            f"{int(info['split']['test_size'] * 100)} stratifié, seed {info['split']['random_state']}")

    cols = st.columns(6)
    for col, (label, key) in zip(cols, [("Accuracy", "Accuracy"), ("Precision", "Precision"),
                                         ("Recall", "Recall"), ("F1", "F1"),
                                         ("ROC AUC", "ROC_AUC"), ("PR AUC", "PR_AUC")]):
        col.metric(label, f"{m[key]:.3f}")

    left, right = st.columns([1.1, 1])
    with left:
        show(confusion_fig(c))
    with right:
        st.subheader("Lecture métier")
        baseline = 1 - info["test_churn_rate"]
        st.markdown(
            f"- Sur **{m['n_test']:,}** clients de test, **{m['positives']}** ont réellement quitté "
            f"({pct(info['test_churn_rate'])}).\n"
            f"- Le modèle signale **{c['tp'] + c['fp']}** clients : **{c['tp']}** sont de vrais churners "
            f"(précision {pct(m['Precision'])}), **{c['fp']}** sont de fausses alertes.\n"
            f"- Il **rate {c['fn']} churners** sur {m['positives']} : le rappel est de {pct(m['Recall'])}.\n"
            f"- L'accuracy ({pct(m['Accuracy'])}) est à comparer à celle d'un modèle qui répond toujours "
            f"« No churn » : **{pct(baseline)}**. Sur des classes déséquilibrées, privilégie "
            f"rappel, PR AUC et balanced accuracy ({m['Balanced_Accuracy']:.3f})."
        )
        if m["Recall"] < 0.6:
            st.warning("Au seuil actuel, une grande partie des churners échappe au modèle. "
                       "Explore la page **Seuil & coûts** pour ajuster ce compromis.")
        with st.expander("Fiche du modèle"):
            st.write(f"Étapes du pipeline : {' → '.join(info['pipeline_steps'])}")
            st.write(f"Variables en entrée : {info['n_features_in']} — "
                     f"entraînement : {info['train_size']:,} clients ({pct(info['train_churn_rate'])} de churn)")
            st.json(info["hyperparameters"], expanded=False)


def page_performance(ctx: dict) -> None:
    st.title("📈 Performance du modèle")
    cv = api_get("/evaluation/curves", model=ctx["model"])
    m = api_get("/evaluation/metrics", model=ctx["model"], threshold=ctx["threshold"])
    thr = ctx["threshold"]
    t_roc, t_pr, t_cal, t_gain, t_dist = st.tabs(
        ["Courbe ROC", "Précision-Rappel", "Calibration", "Gains & lift", "Distribution des scores"])

    with t_roc:
        fig = go.Figure()
        fig.add_scatter(x=cv["roc"]["fpr"], y=cv["roc"]["tpr"], name=f"ROC (AUC = {m['ROC_AUC']:.3f})",
                        line=dict(color=C_OK, width=3))
        fig.add_scatter(x=[0, 1], y=[0, 1], name="Hasard", line=dict(color=C_NEUTRAL, dash="dash"))
        fig.add_scatter(x=[1 - m["Specificity"]], y=[m["Recall"]], mode="markers", name=f"Seuil {thr:.2f}",
                        marker=dict(color=C_CHURN, size=13, line=dict(width=2, color="white")))
        show(style(fig, 440, xaxis_title="Taux de faux positifs", yaxis_title="Taux de vrais positifs (rappel)"))
        st.caption("Plus la courbe est proche du coin supérieur gauche, mieux le modèle classe les clients. "
                   "Le point rouge indique le seuil choisi dans la barre latérale.")

    with t_pr:
        fig = go.Figure()
        fig.add_scatter(x=cv["pr"]["recall"], y=cv["pr"]["precision"], name=f"PR (AP = {m['PR_AUC']:.3f})",
                        line=dict(color=C_OK, width=3))
        fig.add_hline(y=cv["pr"]["baseline"], line_dash="dash", line_color=C_NEUTRAL,
                      annotation_text=f"Hasard = taux de churn ({pct(cv['pr']['baseline'])})")
        fig.add_scatter(x=[m["Recall"]], y=[m["Precision"]], mode="markers", name=f"Seuil {thr:.2f}",
                        marker=dict(color=C_CHURN, size=13, line=dict(width=2, color="white")))
        show(style(fig, 440, xaxis_title="Rappel", yaxis_title="Précision"))
        st.caption("Plus informative que la ROC quand la classe positive est rare : elle montre combien "
                   "d'alertes sont justes pour chaque niveau de détection.")

    with t_cal:
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], name="Calibration parfaite", line=dict(color=C_NEUTRAL, dash="dash"))
        fig.add_scatter(x=cv["calibration"]["mean_predicted"], y=cv["calibration"]["fraction_positive"],
                        mode="lines+markers", name="Modèle", line=dict(color=C_OK, width=3))
        show(style(fig, 440, xaxis_title="Probabilité prédite (moyenne par tranche)",
                   yaxis_title="Fréquence réelle de churn"))
        st.caption(f"Brier = {m['Brier']:.3f}, log-loss = {m['LogLoss']:.3f}. Sur la diagonale, une probabilité "
                   "de 30 % correspond bien à ~30 % de churners : les scores sont exploitables comme probabilités.")

    with t_gain:
        g = cv["gains_curve"]
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], name="Sans modèle", line=dict(color=C_NEUTRAL, dash="dash"))
        fig.add_scatter(x=g["fraction"], y=g["captured"], name="Modèle", line=dict(color=C_OK, width=3),
                        fill="tonexty", fillcolor="rgba(46,134,171,.12)")
        show(style(fig, 420, xaxis_title="Part des clients contactés (scores décroissants)",
                   yaxis_title="Part des churners capturés"))
        table = pd.DataFrame([x for x in cv["gains"] if x["fraction"] in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)])
        table = table.rename(columns={"fraction": "Clients contactés", "captured": "Churners capturés", "lift": "Lift"})
        table["Clients contactés"] = table["Clients contactés"].map(pct)
        table["Churners capturés"] = table["Churners capturés"].map(pct)
        table["Lift"] = table["Lift"].map(lambda v: f"×{v:.2f}")
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("En contactant les 20 % de clients au score le plus élevé, on atteint la part de churners "
                   "indiquée ci-dessus. Le lift compare ce résultat à un ciblage aléatoire.")

    with t_dist:
        d = cv["score_distribution"]
        fig = go.Figure()
        fig.add_bar(x=d["bin_centers"], y=d["no_churn"], name="No churn", marker_color=C_OK, opacity=.75)
        fig.add_bar(x=d["bin_centers"], y=d["churn"], name="Churn", marker_color=C_CHURN, opacity=.75)
        fig.update_layout(barmode="overlay")
        vline(fig, thr, f"seuil {thr:.2f}")
        show(style(fig, 420, xaxis_title="Probabilité de churn prédite", yaxis_title="Nombre de clients"))
        st.caption("Plus les deux distributions sont séparées, plus le modèle discrimine bien. "
                   "Les churners qui tombent à gauche du seuil sont les faux négatifs.")


def page_threshold(ctx: dict) -> None:
    st.title("🎯 Seuil de décision & coûts métier")
    st.write("Le seuil 0,50 est une convention, pas un optimum. Choisis le compromis entre churners "
             "détectés et fausses alertes, ou laisse le coût métier décider.")
    c1, c2 = st.columns(2)
    cost_fn = c1.number_input("Coût d'un churner non détecté (FN)", min_value=0.0, value=100.0, step=10.0,
                              help="Valeur perdue quand un client part sans avoir été ciblé.")
    cost_fp = c2.number_input("Coût d'une alerte inutile (FP)", min_value=0.0, value=20.0, step=5.0,
                              help="Coût de l'offre de rétention envoyée à un client qui serait resté.")
    st.caption("⚠️ Ces valeurs sont des hypothèses illustratives : remplace-les par tes coûts réels, "
               "le seuil optimal en dépend directement.")

    sweep = api_get("/evaluation/threshold", model=ctx["model"], cost_fn=cost_fn, cost_fp=cost_fp)
    df = pd.DataFrame(sweep["rows"])
    thr = ctx["threshold"]

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for col, name, color in [("precision", "Précision", C_OK), ("recall", "Rappel", C_CHURN), ("f1", "F1", "#6A994E")]:
            fig.add_scatter(x=df["threshold"], y=df[col], name=name, line=dict(color=color, width=3))
        vline(fig, thr, f"seuil {thr:.2f}")
        show(style(fig, 400, title="Métriques selon le seuil", xaxis_title="Seuil", yaxis_title="Valeur"))
    with right:
        fig = go.Figure()
        fig.add_scatter(x=df["threshold"], y=df["cost"], name="Coût total", line=dict(color="#6C4AB6", width=3))
        best = sweep["best"]["min_cost"]
        fig.add_scatter(x=[best], y=[df.loc[df["threshold"] == best, "cost"].iloc[0]], mode="markers",
                        name=f"Minimum ({best:.2f})", marker=dict(color=C_CHURN, size=13))
        vline(fig, thr, f"seuil {thr:.2f}")
        show(style(fig, 400, title="Coût total sur le jeu de test", xaxis_title="Seuil", yaxis_title="Coût"))

    rows = []
    for label, key in [("F1 maximal", "f1"), ("Youden (rappel + spécificité)", "youden"), ("Coût minimal", "min_cost")]:
        t = sweep["best"][key]
        r = df.loc[df["threshold"] == t].iloc[0]
        rows.append({"Critère": label, "Seuil": t, "Précision": r["precision"], "Rappel": r["recall"],
                     "F1": r["f1"], "FN": int(r["fn"]), "FP": int(r["fp"]), "Coût": r["cost"]})
    for label, t in [("Seuil actuel", thr), ("Seuil 0,50 (défaut)", 0.5)]:
        r = df.iloc[(df["threshold"] - t).abs().argmin()]
        rows.append({"Critère": label, "Seuil": t, "Précision": r["precision"], "Rappel": r["recall"],
                     "F1": r["f1"], "FN": int(r["fn"]), "FP": int(r["fp"]), "Coût": r["cost"]})
    table = pd.DataFrame(rows)
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "Seuil": st.column_config.NumberColumn(format="%.2f"),
            "Précision": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
            "Rappel": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
            "F1": st.column_config.NumberColumn(format="%.3f"),
            "Coût": st.column_config.NumberColumn(format="%.0f"),
        },
    )

    def use_threshold(value: float) -> None:
        st.session_state["threshold"] = round(float(value), 2)

    b1, b2, b3 = st.columns(3)
    for col, label, key in [(b1, "Utiliser le seuil F1 maximal", "f1"),
                            (b2, "Utiliser le seuil Youden", "youden"),
                            (b3, "Utiliser le seuil à coût minimal", "min_cost")]:
        col.button(f"{label} ({sweep['best'][key]:.2f})", on_click=use_threshold,
                   args=(sweep["best"][key],), width="stretch")
    st.caption("Ces boutons déplacent le curseur de seuil de la barre latérale : toutes les pages s'y adaptent. "
               "Attention : choisir le seuil sur le jeu de test puis rapporter la performance sur ce même jeu "
               "l'embellit légèrement ; idéalement, fixe-le sur un jeu de validation.")


def page_explain(ctx: dict) -> None:
    st.title("🔍 Explicabilité (SHAP)")
    t_glob, t_dep, t_reports = st.tabs(["Importance globale", "Effet des variables numériques", "Rapports des notebooks"])

    with t_glob:
        g = api_get("/explain/global", model=ctx["model"], top=25)
        grouped = st.toggle("Regrouper les modalités par variable (Contract, InternetService…)", value=True)
        items = g["by_group" if grouped else "by_feature"][:15][::-1]
        fig = go.Figure(go.Bar(x=[i["mean_abs_shap"] for i in items], y=[i["feature"] for i in items],
                               orientation="h", marker_color=C_OK))
        show(style(fig, 520, xaxis_title="Impact moyen absolu sur le score (log-odds)",
                   title=f"Variables les plus influentes — {g['n_samples']:,} clients de test"))
        st.caption("Calculé en direct par SHAP sur le jeu de test. Une barre longue = la variable fait "
                   "beaucoup bouger la prédiction, dans un sens ou dans l'autre.")

    with t_dep:
        feature = st.selectbox("Variable", ["tenure", "MonthlyCharges", "TotalCharges"])
        d = api_get("/explain/dependence", model=ctx["model"], feature=feature)
        fig = go.Figure()
        fig.add_scatter(x=d["points"]["x"], y=d["points"]["shap"], mode="markers", name="Clients",
                        marker=dict(color=C_NEUTRAL, size=5, opacity=.45))
        fig.add_scatter(x=d["trend"]["x"], y=d["trend"]["shap"], mode="lines+markers", name="Tendance",
                        line=dict(color=C_CHURN, width=3))
        fig.add_hline(y=0, line_color="gray", line_dash="dot")
        show(style(fig, 440, xaxis_title=feature, yaxis_title="Contribution SHAP (log-odds)"))
        st.caption("Au-dessus de 0, la valeur augmente le risque de churn ; en dessous, elle le diminue.")

    with t_reports:
        listing = api_get("/reports")
        images = [f for f in listing["files"] if f["kind"] == "png"]
        htmls = [f for f in listing["files"] if f["kind"] == "html"]
        if not images and not htmls:
            st.info("Aucun artefact trouvé dans machine_learning/reports.")
        cols = st.columns(2)
        for i, f in enumerate(images):
            with cols[i % 2]:
                st.markdown(f"**{f['path']}**")
                st.image(api_file(f"{listing['base_url']}/{f['path']}"), width="stretch")
        for f in htmls:
            with st.expander(f"Rapport interactif : {f['path']}"):
                html = api_file(f"{listing['base_url']}/{f['path']}").decode("utf-8", errors="replace")
                st.iframe(html, height=520)


# -- Prédiction -------------------------------------------------------------
BATCH_REQUIRED = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
                  "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                  "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
                  "MonthlyCharges", "TotalCharges"]


def customer_form(schema: dict) -> dict:
    opt, num = schema["options"], schema["numeric"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### Profil")
        gender = st.selectbox("Genre", opt["gender"], key="f_gender")
        senior = st.selectbox("Senior (65 ans et +)", ["No", "Yes"], key="f_senior")
        partner = st.selectbox("En couple", opt["Partner"], index=1, key="f_partner")
        dependents = st.selectbox("Personnes à charge", opt["Dependents"], key="f_dependents")
    with c2:
        st.markdown("##### Contrat & facturation")
        tenure = st.number_input("Ancienneté (mois)", 0, 120, num["tenure"]["default"], key="f_tenure")
        contract = st.selectbox("Type de contrat", opt["Contract"], key="f_contract")
        paperless = st.selectbox("Facture dématérialisée", opt["PaperlessBilling"], index=1, key="f_paperless")
        payment = st.selectbox("Moyen de paiement", opt["PaymentMethod"], key="f_payment")
        monthly = st.number_input("Frais mensuels", 0.0, float(num["MonthlyCharges"]["max"]),
                                  round(num["MonthlyCharges"]["default"], 2), step=1.0, key="f_monthly")
        auto_total = st.checkbox("Estimer le total (ancienneté × frais mensuels)", value=True, key="f_auto_total")
        if auto_total:
            total = round(tenure * monthly, 2) if tenure > 0 else None
            st.caption("TotalCharges = " + ("vide (ancienneté 0)" if total is None else f"{total:,.2f}"))
        else:
            total = st.number_input("Frais totaux", 0.0, float(num["TotalCharges"]["max"]),
                                    round(num["TotalCharges"]["default"], 2), step=10.0, key="f_total")
            if tenure == 0:
                total = None
    with c3:
        st.markdown("##### Services")
        phone = st.selectbox("Service téléphonique", opt["PhoneService"], index=1, key="f_phone")
        if phone == "Yes":
            multiple = st.selectbox("Lignes multiples", ["No", "Yes"], key="f_multiple")
        else:
            st.selectbox("Lignes multiples", ["No phone service"], disabled=True, key="f_multiple_off")
            multiple = "No phone service"
        internet = st.selectbox("Service Internet", opt["InternetService"], index=1, key="f_internet")
        addons = {}
        for field, label in [("OnlineSecurity", "Sécurité en ligne"), ("OnlineBackup", "Sauvegarde en ligne"),
                             ("DeviceProtection", "Protection appareil"), ("TechSupport", "Support technique"),
                             ("StreamingTV", "Streaming TV"), ("StreamingMovies", "Streaming films")]:
            if internet == "No":
                st.selectbox(label, ["No internet service"], disabled=True, key=f"f_{field}_off")
                addons[field] = "No internet service"
            else:
                addons[field] = st.selectbox(label, ["No", "Yes"], key=f"f_{field}")
    return {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0, "Partner": partner,
        "Dependents": dependents, "tenure": int(tenure), "PhoneService": phone, "MultipleLines": multiple,
        "InternetService": internet, **addons, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": float(monthly), "TotalCharges": total,
    }


def render_prediction(res: dict, thr: float) -> None:
    p = res["churn_probability"]
    flagged = p >= thr
    left, right = st.columns([1, 1.6])
    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": " %", "valueformat": ".1f"},
            title={"text": "Probabilité de churn"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": C_CHURN if flagged else C_OK},
                   "steps": [{"range": [0, thr * 100], "color": "rgba(46,134,171,.15)"},
                             {"range": [thr * 100, 100], "color": "rgba(228,87,46,.15)"}],
                   "threshold": {"line": {"color": "gray", "width": 4}, "thickness": .8, "value": thr * 100}},
        ))
        show(style(fig, 300))
        if flagged:
            st.error(f"⚠️ Risque de churn (≥ seuil {thr:.2f})")
        else:
            st.success(f"✅ Client plutôt fidèle (< seuil {thr:.2f})")
        st.caption(f"Score plus élevé que {pct(res['risk_percentile'], 0)} des clients du jeu de test.")
    with right:
        exp = res.get("explanation")
        if not exp:
            st.info("Explication SHAP indisponible pour ce modèle.")
            return
        items = list(exp["contributions"])
        if abs(exp["other_contribution"]) > 1e-9:
            items.append({"feature": "Autres variables", "value": "", "shap": exp["other_contribution"]})
        items = sorted(items, key=lambda i: abs(i["shap"]))
        labels = [f"{i['feature']} = {i['value']}" if i["value"] not in ("", None) else i["feature"] for i in items]
        fig = go.Figure(go.Bar(
            x=[i["shap"] for i in items], y=labels, orientation="h",
            marker_color=[C_CHURN if i["shap"] > 0 else C_OK for i in items],
        ))
        fig.add_vline(x=0, line_color="gray")
        show(style(fig, 420, title="Pourquoi ce score ? (contributions SHAP)", xaxis_title="← baisse le risque | augmente le risque →"))
        st.caption(f"Point de départ = risque moyen du modèle ({pct(exp['base_probability'])}) ; chaque barre "
                   "déplace le score (échelle log-odds). Rouge = pousse vers le churn, bleu = vers la fidélité.")


def df_to_customers(df: pd.DataFrame) -> list[dict]:
    d = df.copy()
    for col in ("TotalCharges", "MonthlyCharges"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d[(["customerID"] if "customerID" in d.columns else []) + BATCH_REQUIRED]
    return json.loads(d.to_json(orient="records"))


def tab_batch(ctx: dict) -> None:
    st.write("Charge un CSV au format de `Churn.csv` (colonnes brutes : `Contract`, `InternetService`…).")
    file = st.file_uploader("Fichier CSV", type=["csv"])
    if file is None:
        return
    df = pd.read_csv(file)
    missing = [c for c in BATCH_REQUIRED if c not in df.columns]
    if missing:
        st.error(f"Colonnes manquantes : {', '.join(missing)}")
        return
    if len(df) > 10_000:
        st.error("Maximum 10 000 lignes par fichier.")
        return
    try:
        out = api_post("/predict/batch", {"customers": df_to_customers(df), "model": ctx["model"],
                                          "threshold": ctx["threshold"]})
    except ApiError as exc:
        st.error(f"Fichier refusé par l'API : {exc}")
        return

    scores = pd.DataFrame(out["predictions"]).rename(
        columns={"churn_probability": "Probabilité de churn", "prediction": "Alerte"})
    result = pd.concat([df.reset_index(drop=True), scores[["Probabilité de churn", "Alerte"]]], axis=1)
    s = out["summary"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Clients scorés", f"{s['n']:,}")
    k2.metric("Clients en alerte", f"{s['flagged']:,}", pct(s["flagged_rate"]), delta_color="off")
    k3.metric("Probabilité moyenne", pct(s["mean_probability"]))

    fig = go.Figure(go.Histogram(x=result["Probabilité de churn"], nbinsx=30, marker_color=C_OK))
    vline(fig, ctx["threshold"], f"seuil {ctx['threshold']:.2f}")
    show(style(fig, 280, xaxis_title="Probabilité de churn", yaxis_title="Clients"))

    if "Churn" in df.columns:
        truth = df["Churn"].astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).to_numpy()
        pred = result["Alerte"].to_numpy().astype(bool)
        tp, fp, fn = int((truth & pred).sum()), int((~truth & pred).sum()), int((truth & ~pred).sum())
        st.markdown("**Évaluation sur ce fichier** (colonne `Churn` détectée)")
        e1, e2, e3 = st.columns(3)
        e1.metric("Précision", f"{tp / (tp + fp):.3f}" if tp + fp else "—")
        e2.metric("Rappel", f"{tp / (tp + fn):.3f}" if tp + fn else "—")
        e3.metric("Accuracy", f"{(pred == truth).mean():.3f}")
        st.caption("⚠️ Si ce fichier contient des clients vus à l'entraînement, ces chiffres sont optimistes. "
                   "Les métriques de référence sont celles du jeu de test (pages Aperçu et Performance).")

    ranked = result.sort_values("Probabilité de churn", ascending=False)
    st.dataframe(ranked, hide_index=True, width="stretch",
                 column_config={"Probabilité de churn": st.column_config.ProgressColumn(
                     format="%.3f", min_value=0, max_value=1)})
    st.download_button("Télécharger les scores (CSV)", ranked.to_csv(index=False).encode("utf-8"),
                       file_name="churn_scores.csv", mime="text/csv")


def page_predict(ctx: dict) -> None:
    st.title("🧮 Prédire le risque de churn")
    t_one, t_batch = st.tabs(["Un client", "Fichier CSV"])
    with t_one:
        schema = api_get("/schema")
        customer = customer_form(schema)
        if st.button("Prédire le risque de churn", type="primary", width="stretch"):
            try:
                st.session_state["last_prediction"] = api_post(
                    "/predict", {"customer": customer, "model": ctx["model"],
                                 "threshold": ctx["threshold"], "explain": True, "top_k": 10})
            except ApiError as exc:
                st.session_state.pop("last_prediction", None)
                st.error(f"Prédiction impossible : {exc}")
        res = st.session_state.get("last_prediction")
        if res:
            st.divider()
            render_prediction(res, ctx["threshold"])
            with st.expander("Données envoyées au modèle (après règles de cohérence)"):
                st.json(res["normalized_input"], expanded=False)
    with t_batch:
        tab_batch(ctx)


def page_data(ctx: dict) -> None:
    st.title("🗂️ Exploration des données")
    ov = api_get("/data/overview")
    k = st.columns(4)
    k[0].metric("Clients", f"{ov['n_customers']:,}")
    k[1].metric("Taux de churn", pct(ov["churn_rate"]))
    k[2].metric("Ancienneté moyenne", f"{ov['avg_tenure']:.1f} mois")
    k[3].metric("Frais mensuels moyens", f"{ov['avg_monthly_charges']:.2f}")

    segment = st.selectbox("Taux de churn par…", list(ov["churn_by"]))
    seg = pd.DataFrame(ov["churn_by"][segment]).sort_values("churn_rate", ascending=False)
    fig = go.Figure(go.Bar(x=seg["category"], y=seg["churn_rate"], marker_color=C_CHURN,
                           text=[f"{r:.1%}<br>n={n:,}" for r, n in zip(seg["churn_rate"], seg["customers"])],
                           textposition="outside"))
    fig.add_hline(y=ov["churn_rate"], line_dash="dash", line_color="gray",
                  annotation_text=f"Moyenne {pct(ov['churn_rate'])}")
    fig.update_yaxes(range=[0, min(1, seg["churn_rate"].max() * 1.25)], tickformat=".0%")
    show(style(fig, 400, yaxis_title="Taux de churn"))

    c1, c2 = st.columns(2)
    for col, key, title in [(c1, "hist_tenure", "Ancienneté (mois)"), (c2, "hist_monthly_charges", "Frais mensuels")]:
        h = ov[key]
        fig = go.Figure()
        fig.add_bar(x=h["bin_centers"], y=h["no_churn"], name="No churn", marker_color=C_OK, opacity=.75)
        fig.add_bar(x=h["bin_centers"], y=h["churn"], name="Churn", marker_color=C_CHURN, opacity=.75)
        fig.update_layout(barmode="overlay")
        with col:
            show(style(fig, 340, title=title, xaxis_title=title, yaxis_title="Clients"))
    st.caption(f"Statistiques sur l'ensemble du fichier Churn.csv. {ov['missing_total_charges']} valeurs de "
               "TotalCharges sont manquantes (clients d'ancienneté 0) ; le pipeline les impute par la médiane.")


def page_compare(ctx: dict) -> None:
    st.title("⚖️ Comparaison des modèles")
    cmp_ = api_get("/evaluation/compare", threshold=ctx["threshold"])
    live = pd.DataFrame(cmp_["live"])
    st.subheader(f"Évalués en direct sur le jeu de test (seuil {ctx['threshold']:.2f})")
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
    st.dataframe(live, hide_index=True, width="stretch",
                 column_config={m: st.column_config.NumberColumn(format="%.3f") for m in metrics + ["Brier"]})
    if len(live) < 2:
        st.info("Un seul modèle est chargé. Pour en comparer d'autres, dépose simplement leurs fichiers "
                "`.joblib` dans `machine_learning/models/` (même features en entrée) et redémarre l'API.")

    fig = go.Figure()
    for i, row in live.iterrows():
        fig.add_bar(x=metrics, y=[row[m] for m in metrics], name=row["model"])
    show(style(fig, 380, barmode="group", yaxis_title="Valeur", yaxis_range=[0, 1]))

    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], name="Hasard", line=dict(color=C_NEUTRAL, dash="dash"))
    for name, r in cmp_["roc"].items():
        fig.add_scatter(x=r["fpr"], y=r["tpr"], name=name, line=dict(width=3))
    show(style(fig, 400, title="Courbes ROC", xaxis_title="Taux de faux positifs", yaxis_title="Rappel"))

    if cmp_["reference_report"]:
        st.subheader("Rapport enregistré par le notebook (model_comparison.csv)")
        st.dataframe(pd.DataFrame(cmp_["reference_report"]), hide_index=True, width="stretch")
        st.caption("Ce tableau est figé au moment de l'entraînement ; celui du dessus est recalculé à chaque requête.")


# ==========================================================================
# Navigation
# ==========================================================================
PAGES = {
    "📊 Aperçu": page_overview,
    "📈 Performance": page_performance,
    "🎯 Seuil & coûts": page_threshold,
    "🔍 Explicabilité": page_explain,
    "🧮 Prédire": page_predict,
    "🗂️ Données": page_data,
    "⚖️ Comparaison": page_compare,
}


def main() -> None:
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        health.raise_for_status()
        health = health.json()
    except requests.RequestException:
        st.error(f"Impossible de joindre l'API ({API_URL}). Démarre-la depuis la racine du projet avec :\n\n"
                 "`uvicorn backend.main:app --reload`")
        st.stop()

    st.session_state.setdefault("threshold", 0.50)
    with st.sidebar:
        st.header("Telco Churn")
        page = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
        st.divider()
        models = health["models"]
        model = st.selectbox("Modèle", models, index=models.index(health["default_model"]))
        st.slider("Seuil de décision", 0.05, 0.95, step=0.01, key="threshold",
                  help="Un client est signalé si sa probabilité de churn est ≥ au seuil.")
        st.divider()
        st.caption(f"API : {API_URL}")
        if st.button("Rafraîchir les données", width="stretch"):
            st.cache_data.clear()
            st.rerun()

    ctx = {"model": model, "threshold": float(st.session_state["threshold"])}
    try:
        PAGES[page](ctx)
    except ApiError as exc:
        st.error(f"Erreur de l'API : {exc}")
    except requests.RequestException as exc:
        st.error(f"Connexion à l'API perdue : {exc}")


main()
