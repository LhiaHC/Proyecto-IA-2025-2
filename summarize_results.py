import pandas as pd
import glob

results = []
for f in glob.glob("reports/results_*.csv"):
    df = pd.read_csv(f)
    results.append(df)

if results:
    all_df = pd.concat(results, ignore_index=True)
    all_df_sorted = all_df.sort_values('cv_f1_mean', ascending=False)

    print("="*70)
    print("RESUMEN DE RESULTADOS (ordenados por F1-Score)")
    print("="*70)
    print(all_df_sorted[['embedding', 'dimensionality', 'classifier',
                         'cv_accuracy_mean', 'cv_f1_mean', 'cv_fit_time_mean']].to_string(index=False))

    print("\n" + "="*70)
    print("TOP 5 MEJORES CONFIGURACIONES")
    print("="*70)
    for idx, row in all_df_sorted.head(5).iterrows():
        print(f"\n{idx+1}. {row['embedding'].upper()}-{int(row['dimensionality'])}-{row['classifier'].upper()}")
        print(f"   Accuracy: {row['cv_accuracy_mean']:.4f} ± {row['cv_accuracy_std']:.4f}")
        print(f"   F1-Score: {row['cv_f1_mean']:.4f} ± {row['cv_f1_std']:.4f}")
        print(f"   Precision: {row['cv_precision_mean']:.4f} ± {row['cv_precision_std']:.4f}")
        print(f"   Recall: {row['cv_recall_mean']:.4f} ± {row['cv_recall_std']:.4f}")
        print(f"   Tiempo: {row['cv_fit_time_mean']:.2f}s ± {row['cv_fit_time_std']:.2f}s")
else:
    print("No se encontraron resultados")
