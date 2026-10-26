# Telco Churn — API + tableau de bord d'évaluation

## Lancement (depuis la racine du projet, deux terminaux)

    uvicorn backend.main:app --reload        # API       -> http://127.0.0.1:8000/docs
    streamlit run frontend/app.py            # Interface -> http://localhost:8501

Autre URL d'API : `set CHURN_API_URL=http://hote:8000` (Windows) avant `streamlit run`.

## Structure

    backend/
      main.py      routes FastAPI
      service.py   chargement des modèles, feature engineering, métriques, SHAP
      schemas.py   validation des entrées
    frontend/app.py  interface Streamlit (7 pages)

## Ajouter un modèle à comparer
Dépose son `.joblib` (pipeline complet, mêmes colonnes d'entrée) dans
`machine_learning/models/` et redémarre l'API : il apparaît dans la barre latérale
et la page « Comparaison ».
