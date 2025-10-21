"""
Utilidades para carga de datos, splits estratificados y métricas.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, matthews_corrcoef, accuracy_score,
    precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix
)


RANDOM_SEED = 42
DATA_PATH = "data/phishing_email.csv"
MAX_TEXT_LENGTH = 100_000


def load_data(path: str = DATA_PATH, filter_outliers: bool = True) -> pd.DataFrame:
    """
    Carga el dataset de phishing desde CSV y filtra outliers opcionales.

    Args:
        path: Ruta al archivo CSV.
        filter_outliers: Si True, descarta textos > MAX_TEXT_LENGTH.

    Returns:
        DataFrame con columnas 'text_combined' y 'label'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset no encontrado en '{path}'. "
            "Asegúrate de que el archivo exista antes de ejecutar."
        )

    df = pd.read_csv(path)

    # Validar columnas requeridas
    if "text_combined" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"El CSV debe contener las columnas 'text_combined' y 'label'. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    print(f"Dataset cargado: {len(df)} filas")

    if filter_outliers:
        df["text_len"] = df["text_combined"].fillna("").astype(str).apply(len)
        before = len(df)
        df = df[df["text_len"] <= MAX_TEXT_LENGTH].copy()
        after = len(df)
        if before > after:
            print(f"Filtrado de outliers: {before - after} filas descartadas (len > {MAX_TEXT_LENGTH})")
        df = df.drop(columns=["text_len"])

    return df


def stratified_split(df: pd.DataFrame, label_col: str = "label"):
    """
    Divide el dataset en train (70%), val (15%), test (15%) de forma estratificada.

    Args:
        df: DataFrame con columnas 'text_combined' y label_col.
        label_col: Nombre de la columna de etiquetas.

    Returns:
        Tupla (train_df, val_df, test_df).
    """
    # Split 70/30
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df[label_col], random_state=RANDOM_SEED
    )
    # Split 30 -> 15/15 (50% de 30% = 15%)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df[label_col], random_state=RANDOM_SEED
    )

    print(f"Split estratificado - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def compute_metrics(y_true, y_pred, y_proba=None, dataset_name: str = "Test"):
    """
    Calcula métricas completas: Accuracy, Precision, Recall, F1-score, MCC, AUC-ROC.

    Args:
        y_true: Etiquetas reales.
        y_pred: Predicciones del modelo.
        y_proba: Probabilidades predichas (necesarias para AUC-ROC). Opcional.
        dataset_name: Nombre del conjunto (para logging).

    Returns:
        Dict con métricas {'accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc_roc'}.
    """
    # Métricas básicas
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # AUC-ROC (requiere probabilidades)
    auc_roc = None
    if y_proba is not None:
        try:
            # Para clasificación binaria, usar probabilidad de clase positiva
            if y_proba.ndim == 2:
                y_proba_positive = y_proba[:, 1]
            else:
                y_proba_positive = y_proba
            auc_roc = roc_auc_score(y_true, y_proba_positive)
        except Exception as e:
            print(f"Advertencia: No se pudo calcular AUC-ROC: {e}")
            auc_roc = None

    # Imprimir resultados
    print(f"\n{'='*60}")
    print(f"Métricas en {dataset_name}")
    print(f"{'='*60}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")
    if auc_roc is not None:
        print(f"AUC-ROC:   {auc_roc:.4f}")
    else:
        print(f"AUC-ROC:   N/A")

    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"{'='*60}\n")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "auc_roc": auc_roc
    }
