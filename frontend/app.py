
import requests
import streamlit as st

# =========================================================
# 1. CONFIGURATION
# =========================================================

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide",
)

# =========================================================
# 2. MAPPINGS
# =========================================================

# Vérifier que cet encodage correspond à celui du dataset.
GENDER_MAP = {
    "Female": 0,
    "Male": 1,
}

BINARY_MAP = {
    "No": 0,
    "Yes": 1,
}

# =========================================================
# 3. TITRE
# =========================================================

st.title("📊 Telco Customer Churn Prediction")
st.write(
    "Évaluez le risque de départ d'un client "
    "à partir de ses informations contractuelles."
)

# =========================================================
# 4. CONNEXION À FASTAPI ET RÉCUPÉRATION DES FEATURES
# =========================================================

try:
    response = requests.get(f"{API_URL}/features", timeout=5)
    response.raise_for_status()
    api_data = response.json()

except requests.RequestException as exc:
    st.error(
        f"Impossible de contacter l'API FastAPI : {exc}\n\n"
        "Vérifiez que le backend est lancé avec :\n"
        "`uvicorn backend.main:app --reload`"
    )
    st.stop()


def extract_feature_names(data):
    """Extrait les noms des colonnes des formats API courants."""

    if isinstance(data, list):
        items = data

    elif isinstance(data, dict):
        # Format contenant directement une liste de colonnes.
        items = data.get(
            "features",
            data.get("feature_columns", data.get("columns")),
        )

        if items is None:
            raise ValueError(
                "La réponse de /features ne contient ni "
                "'features', ni 'feature_columns', ni 'columns'."
            )

    else:
        raise ValueError(
            "Le format de la réponse /features n'est pas reconnu."
        )

    if not isinstance(items, list):
        raise ValueError(
            "La liste des caractéristiques reçue n'est pas une liste."
        )

    names = []

    for item in items:
        if isinstance(item, str):
            names.append(item)

        elif isinstance(item, dict):
            # Formats possibles : {"name": "tenure"},
            # {"key": "tenure"} ou {"name": "tenure", ...}.
            name = item.get("name", item.get("key"))

            if not isinstance(name, str):
                raise ValueError(
                    f"Impossible d'extraire le nom de cette "
                    f"caractéristique : {item}"
                )

            names.append(name)

        else:
            raise ValueError(
                f"Type de caractéristique inattendu : {type(item).__name__}"
            )

    if not all(isinstance(name, str) for name in names):
        raise ValueError("Un nom de colonne n'est pas une chaîne.")

    return names


try:
    expected_features = extract_feature_names(api_data)
except ValueError as exc:
    st.error(f"Réponse invalide de GET /features : {exc}")
    st.json(api_data)
    st.stop()

# =========================================================
# 5. FORMULAIRE CLIENT
# =========================================================

st.subheader("Informations du client")

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        gender_label = st.selectbox(
            "Genre",
            ["Female", "Male"],
        )

        senior = st.selectbox(
            "Client senior",
            [0, 1],
            format_func=lambda x: "Oui" if x == 1 else "Non",
        )

        partner_label = st.selectbox(
            "A un partenaire ?",
            ["Yes", "No"],
        )

        dependents_label = st.selectbox(
            "Personnes à charge ?",
            ["Yes", "No"],
        )

        tenure = st.number_input(
            "Ancienneté (mois)",
            min_value=0,
            max_value=72,
            value=12,
            step=1,
        )

    with col2:
        phone_label = st.selectbox(
            "Service téléphonique",
            ["Yes", "No"],
        )

        multiple_lines = st.selectbox(
            "Lignes multiples",
            ["No", "Yes", "No phone service"],
        )

        internet = st.selectbox(
            "Service Internet",
            ["DSL", "Fiber optic", "No"],
        )

        contract = st.selectbox(
            "Type de contrat",
            ["Month-to-month", "One year", "Two year"],
        )

        payment = st.selectbox(
            "Méthode de paiement",
            [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
            ],
        )

    with col3:
        paperless_label = st.selectbox(
            "Facture électronique",
            ["Yes", "No"],
        )

        online_security = st.selectbox(
            "Sécurité en ligne",
            ["No", "Yes", "No internet service"],
        )

        online_backup = st.selectbox(
            "Sauvegarde en ligne",
            ["No", "Yes", "No internet service"],
        )

        device_protection = st.selectbox(
            "Protection de l'appareil",
            ["No", "Yes", "No internet service"],
        )

        tech_support = st.selectbox(
            "Support technique",
            ["No", "Yes", "No internet service"],
        )

    col4, col5 = st.columns(2)

    with col4:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"],
        )

        streaming_movies = st.selectbox(
            "Streaming de films",
            ["No", "Yes", "No internet service"],
        )

    with col5:
        monthly_charges = st.number_input(
            "Frais mensuels",
            min_value=0.0,
            value=70.5,
            step=1.0,
        )

        total_charges = st.number_input(
            "Frais totaux",
            min_value=0.0,
            value=846.0,
            step=10.0,
        )

    submitted = st.form_submit_button(
        "🔍 Prédire le risque de churn",
        type="primary",
        use_container_width=True,
    )

# =========================================================
# 6. MAPPING ET CONSTRUCTION DES FEATURES
# =========================================================

if submitted:

    features = {
        # Variables encodées
        "gender": GENDER_MAP[gender_label],
        "SeniorCitizen": int(senior),
        "Partner": BINARY_MAP[partner_label],
        "Dependents": BINARY_MAP[dependents_label],
        "PhoneService": BINARY_MAP[phone_label],
        "PaperlessBilling": BINARY_MAP[paperless_label],

        # Variables catégorielles restantes
        "MultipleLines": multiple_lines,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,

        # Variables numériques
        "tenure": int(tenure),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),

        # InternetService : One-Hot
        "InternetService_DSL": int(internet == "DSL"),
        "InternetService_Fiber optic": int(internet == "Fiber optic"),
        "InternetService_No": int(internet == "No"),

        # Contract : One-Hot
        "Contract_Month-to-month": int(contract == "Month-to-month"),
        "Contract_One year": int(contract == "One year"),
        "Contract_Two year": int(contract == "Two year"),

        # PaymentMethod : One-Hot
        "PaymentMethod_Bank transfer (automatic)": int(
            payment == "Bank transfer (automatic)"
        ),
        "PaymentMethod_Credit card (automatic)": int(
            payment == "Credit card (automatic)"
        ),
        "PaymentMethod_Electronic check": int(
            payment == "Electronic check"
        ),
        "PaymentMethod_Mailed check": int(payment == "Mailed check"),

        # DeviceProtection : One-Hot
        "DeviceProtection_No": int(device_protection == "No"),
        "DeviceProtection_No internet service": int(
            device_protection == "No internet service"
        ),
        "DeviceProtection_Yes": int(device_protection == "Yes"),

        # TechSupport : One-Hot
        "TechSupport_No": int(tech_support == "No"),
        "TechSupport_No internet service": int(
            tech_support == "No internet service"
        ),
        "TechSupport_Yes": int(tech_support == "Yes"),

        # StreamingTV : One-Hot
        "StreamingTV_No": int(streaming_tv == "No"),
        "StreamingTV_No internet service": int(
            streaming_tv == "No internet service"
        ),
        "StreamingTV_Yes": int(streaming_tv == "Yes"),

        # StreamingMovies : One-Hot
        "StreamingMovies_No": int(streaming_movies == "No"),
        "StreamingMovies_No internet service": int(
            streaming_movies == "No internet service"
        ),
        "StreamingMovies_Yes": int(streaming_movies == "Yes"),
    }

    # IMPORTANT :
    # Ces règles doivent correspondre à celles utilisées
    # pendant le feature engineering.
    features["TenureGroup"] = (
        "0-12" if tenure <= 12
        else "13-24" if tenure <= 24
        else "25-48" if tenure <= 48
        else "49-72"
    )

    features["HighMonthlyCharges"] = int(monthly_charges >= 70)

    # =====================================================
    # 7. VALIDATION DES COLONNES
    # =====================================================

    missing = [
        column
        for column in expected_features
        if column not in features
    ]

    unexpected = [
        column
        for column in features
        if column not in expected_features
    ]

    if missing or unexpected:
        st.error("Les caractéristiques ne correspondent pas à l'API.")

        if missing:
            st.write("**Colonnes manquantes :**", missing)

        if unexpected:
            st.write("**Colonnes inattendues :**", unexpected)

    else:
        # Respecter exactement l'ordre des caractéristiques attendu
        # par le backend.
        ordered_features = {
            column: features[column]
            for column in expected_features
        }

        # =================================================
        # 8. APPEL À FASTAPI
        # =================================================

        with st.spinner("Calcul de la prédiction..."):
            try:
                result = requests.post(
                    f"{API_URL}/predict",
                    json={"features": ordered_features},
                    timeout=30,
                )

                if result.ok:
                    prediction = result.json()

                    st.divider()
                    st.subheader("Résultat de la prédiction")
                    st.json(prediction)

                else:
                    st.error(
                        f"Erreur API : HTTP {result.status_code}"
                    )
                    st.code(result.text)

            except requests.RequestException as exc:
                st.error(
                    f"Erreur de communication avec FastAPI : {exc}"
                )
