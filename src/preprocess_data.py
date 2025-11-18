"""
SCRIPT DE PREPROCESAMIENTO (VERSIÓN FINAL, ADAPTADA A TUS DATOS)

1. Carga la muestra 'data/phishing_email_sample.csv'.
2. Aplica la función de limpieza 'preprocess_full' a la columna 'text_combined'.
3. Guarda el resultado final como 'data/processed/preprocessed_emails_sample.parquet'.
"""

import pandas as pd
import sys
import os
from pandarallel import pandarallel

# Importar solo la función de preprocesamiento necesaria
sys.path.insert(0, ".")
from preproc_01 import preprocess_full

# --- Configuración ---
INPUT_SAMPLE_FILE = "data/phishing_email_sample.csv"
FINAL_PROCESSED_FILE = "data/processed/preprocessed_emails_sample.parquet"
# --------------------


def main():
    print("--- Iniciando preprocesamiento de la MUESTRA ---")

    # 1. Cargar la MUESTRA de datos directamente. NO USAR load_data().
    print(f"Cargando la muestra desde '{INPUT_SAMPLE_FILE}'...")
    try:
        df = pd.read_csv(INPUT_SAMPLE_FILE)
    except FileNotFoundError:
        print(f"\nERROR: No se encontró el archivo de muestra '{INPUT_SAMPLE_FILE}'.")
        print("Por favor, ejecuta 'python src/create_sample.py' primero.")
        sys.exit(1)

    # 2. Aplicar el preprocesamiento de texto a la columna que ya existe.
    print("Aplicando preprocesamiento de texto a la columna 'text_combined'...")
    pandarallel.initialize(progress_bar=True)
    df["text_processed"] = df["text_combined"].parallel_apply(preprocess_full)

    # 3. Seleccionar y guardar las columnas finales.
    df_final = df[["text_processed", "label"]].copy()

    print(
        f"\nGuardando el archivo final listo para los experimentos en '{FINAL_PROCESSED_FILE}'..."
    )
    os.makedirs("data/processed", exist_ok=True)
    df_final.to_parquet(FINAL_PROCESSED_FILE, index=False)

    print("\nPreprocesamiento completado.")
    print("Ya puedes ejecutar 'experiment_runner.py' o './run_all_experiments.sh'.")


if __name__ == "__main__":
    main()
