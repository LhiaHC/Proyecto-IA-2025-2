"""
SCRIPT DE MUESTREO (VERSIÓN FINAL, ADAPTADA A TUS DATOS)

1. Lee tu archivo inicial: 'data/phishing_email.csv'.
2. Crea una muestra estratificada de 5000 filas.
3. Guarda la muestra como 'data/phishing_email_sample.csv'.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

# --- Parámetros para TU archivo de datos ---
YOUR_STARTING_FILE = "data/phishing_email.csv"
OUTPUT_SAMPLE_FILE = "data/phishing_email_sample.csv"
SAMPLE_SIZE = 5000
TARGET_COLUMN = "label"
# -------------------------------------------


def create_sample_from_your_data():
    print("--- Iniciando script de muestreo para 'phishing_email.csv' ---")

    print(f"Cargando tu archivo de datos: '{YOUR_STARTING_FILE}'")
    try:
        df_full = pd.read_csv(YOUR_STARTING_FILE)
    except FileNotFoundError:
        print(
            f"\nFATAL ERROR: No se encontró tu archivo de datos en '{YOUR_STARTING_FILE}'."
        )
        sys.exit(1)

    print(f"Éxito. Dataset cargado con {df_full.shape[0]} filas.")
    required_cols = ["text_combined", "label"]
    if not all(col in df_full.columns for col in required_cols):
        print(
            f"\nFATAL ERROR: El archivo no tiene las columnas esperadas: {required_cols}"
        )
        sys.exit(1)

    print(f"\nCreando una muestra estratificada de {SAMPLE_SIZE} filas...")
    df_sample, _ = train_test_split(
        df_full,
        train_size=SAMPLE_SIZE,
        stratify=df_full[TARGET_COLUMN],
        random_state=42,
    )

    print(f"Guardando la muestra en '{OUTPUT_SAMPLE_FILE}'...")
    os.makedirs("data", exist_ok=True)
    df_sample.to_csv(OUTPUT_SAMPLE_FILE, index=False)

    print("\nMuestra creada con éxito.")
    print("Ahora puedes ejecutar 'python src/preprocess_data.py'.")


if __name__ == "__main__":
    create_sample_from_your_data()
