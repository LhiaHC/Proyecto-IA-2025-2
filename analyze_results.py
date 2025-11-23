"""
Script para analizar todos los resultados de los experimentos
y generar un resumen consolidado.
"""

import pandas as pd
import os
import glob

# Directorio de resultados
REPORTS_DIR = "reports"

# Cargar todos los archivos CSV
all_results = []
for file in glob.glob(f"{REPORTS_DIR}/results_*.csv"):
    df = pd.read_parquet(file)
    all_results.append(df)

# Concatenar todos los resultados
results_df = pd.concat(all_results, ignore_index=True)

print("="*60)
print("RESUMEN DE TODOS LOS EXPERIMENTOS")
print("="*60)
print(f"\nTotal de experimentos: {len(results_df)}")
print(f"\nColumnas: {list(results_df.columns)}")

# Mostrar todos los resultados ordenados por F1
print("\n" + "="*60)
print("RESULTADOS ORDENADOS POR F1-SCORE")
print("="*60)
results_sorted = results_df.sort_values('cv_f1_mean', ascending=False)
print(results_sorted[['embedding', 'dimensionality', 'classifier',
                       'cv_accuracy_mean', 'cv_precision_mean',
                       'cv_recall_mean', 'cv_f1_mean', 'cv_fit_time_mean']].to_string())

# Mejores configuraciones por embedding
print("\n" + "="*60)
print("MEJOR CONFIGURACIÓN POR EMBEDDING")
print("="*60)
for emb in ['word2vec', 'fasttext', 'bert']:
    best = results_df[results_df['embedding'] == emb].sort_values('cv_f1_mean', ascending=False).iloc[0]
    print(f"\n{emb.upper()}:")
    print(f"  Mejor: {emb}-{int(best['dimensionality'])}-{best['classifier'].upper()}")
    print(f"  F1: {best['cv_f1_mean']:.4f} ± {best['cv_f1_std']:.4f}")
    print(f"  Accuracy: {best['cv_accuracy_mean']:.4f} ± {best['cv_accuracy_std']:.4f}")
    print(f"  Tiempo: {best['cv_fit_time_mean']:.2f}s ± {best['cv_fit_time_std']:.2f}s")

# Mejor configuración global
print("\n" + "="*60)
print("MEJOR CONFIGURACIÓN GLOBAL")
print("="*60)
best_overall = results_df.sort_values('cv_f1_mean', ascending=False).iloc[0]
print(f"Método: {best_overall['embedding'].upper()}-{int(best_overall['dimensionality'])}-{best_overall['classifier'].upper()}")
print(f"F1-Score: {best_overall['cv_f1_mean']:.4f} ± {best_overall['cv_f1_std']:.4f}")
print(f"Accuracy: {best_overall['cv_accuracy_mean']:.4f} ± {best_overall['cv_accuracy_std']:.4f}")
print(f"Precision: {best_overall['cv_precision_mean']:.4f} ± {best_overall['cv_precision_std']:.4f}")
print(f"Recall: {best_overall['cv_recall_mean']:.4f} ± {best_overall['cv_recall_std']:.4f}")
print(f"Tiempo: {best_overall['cv_fit_time_mean']:.2f}s ± {best_overall['cv_fit_time_std']:.2f}s")

# Comparación de embeddings (promedio)
print("\n" + "="*60)
print("COMPARACIÓN DE EMBEDDINGS (PROMEDIO)")
print("="*60)
embedding_comparison = results_df.groupby('embedding').agg({
    'cv_f1_mean': ['mean', 'max', 'min'],
    'cv_accuracy_mean': ['mean', 'max', 'min'],
    'cv_fit_time_mean': 'mean'
}).round(4)
print(embedding_comparison)

# Comparación de clasificadores (promedio)
print("\n" + "="*60)
print("COMPARACIÓN DE CLASIFICADORES (PROMEDIO)")
print("="*60)
classifier_comparison = results_df.groupby('classifier').agg({
    'cv_f1_mean': ['mean', 'max', 'min'],
    'cv_accuracy_mean': ['mean', 'max', 'min'],
    'cv_fit_time_mean': 'mean'
}).round(4)
print(classifier_comparison)

# Guardar resumen
summary_file = f"{REPORTS_DIR}/summary_all_experiments.csv"
results_sorted.to_csv(summary_file, index=False)
print(f"\n\nResumen guardado en: {summary_file}")
