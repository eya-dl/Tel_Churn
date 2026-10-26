"""Schémas d'entrée de l'API.

Le client est décrit avec les colonnes *brutes* du fichier ``Churn.csv``
(valeurs lisibles : "Yes", "Fiber optic", "Month-to-month"...).
Le feature engineering (encodage binaire, one-hot, TenureGroup,
HighMonthlyCharges) est fait côté serveur, dans ``service.py``, afin que
l'API et l'entraînement ne puissent pas diverger.
"""
from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

Gender = Literal["Female", "Male"]
YesNo = Literal["Yes", "No"]
MultipleLines = Literal["Yes", "No", "No phone service"]
InternetService = Literal["DSL", "Fiber optic", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]
Contract = Literal["Month-to-month", "One year", "Two year"]
PaymentMethod = Literal[
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
]


class CustomerInput(BaseModel):
    """Un client, avec les mêmes colonnes que ``Churn.csv`` (hors ``Churn``)."""

    # Les colonnes en trop (ex. "Churn" dans un CSV brut) sont ignorées.
    model_config = ConfigDict(extra="ignore")

    customerID: str | None = None
    gender: Gender
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Ancienneté en mois")
    PhoneService: YesNo
    MultipleLines: MultipleLines
    InternetService: InternetService
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Contract
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethod
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(
        default=None,
        ge=0,
        description="Vide (null) pour un client à 0 mois d'ancienneté",
    )


class PredictRequest(BaseModel):
    customer: CustomerInput
    model: str | None = Field(default=None, description="Nom du modèle (fichier .joblib)")
    threshold: float = Field(default=0.5, gt=0, lt=1)
    explain: bool = True
    top_k: int = Field(default=10, ge=1, le=40)


class BatchRequest(BaseModel):
    customers: list[CustomerInput] = Field(min_length=1, max_length=10_000)
    model: str | None = None
    threshold: float = Field(default=0.5, gt=0, lt=1)


# Options affichées dans le formulaire (dérivées des Literal ci-dessus,
# donc toujours synchronisées avec la validation de l'API).
FORM_OPTIONS: dict[str, list] = {
    "gender": list(get_args(Gender)),
    "SeniorCitizen": [0, 1],
    "Partner": list(get_args(YesNo)),
    "Dependents": list(get_args(YesNo)),
    "PhoneService": list(get_args(YesNo)),
    "MultipleLines": list(get_args(MultipleLines)),
    "InternetService": list(get_args(InternetService)),
    "OnlineSecurity": list(get_args(InternetAddon)),
    "OnlineBackup": list(get_args(InternetAddon)),
    "DeviceProtection": list(get_args(InternetAddon)),
    "TechSupport": list(get_args(InternetAddon)),
    "StreamingTV": list(get_args(InternetAddon)),
    "StreamingMovies": list(get_args(InternetAddon)),
    "Contract": list(get_args(Contract)),
    "PaperlessBilling": list(get_args(YesNo)),
    "PaymentMethod": list(get_args(PaymentMethod)),
}
