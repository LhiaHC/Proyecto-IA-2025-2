"""
Script para generar análisis detallado de los 30 experimentos
"""
import pandas as pd
import glob

# Cargar todos los resultados
results = []
for f in glob.glob("reports/results_*.csv"):
    df = pd.read_csv(f)
    results.append(df)

all_df = pd.concat(results, ignore_index=True)
all_df_sorted = all_df.sort_values('cv_f1_mean', ascending=False)

print("="*80)
print("ANÁLISIS COMPLETO DE 30 EXPERIMENTOS")
print("="*80)

# Análisis por embedding
print("\n" + "="*80)
print("ANÁLISIS POR EMBEDDING")
print("="*80)

for emb in ['word2vec', 'fasttext', 'bert']:
    emb_data = all_df[all_df['embedding'] == emb].sort_values('cv_f1_mean', ascending=False)
    print(f"\n{emb.upper()}:")
    print(f"  Experimentos: {len(emb_data)}")
    print(f"  Mejor: {emb}-{int(emb_data.iloc[0]['dimensionality'])}-{emb_data.iloc[0]['classifier'].upper()}")
    print(f"    F1: {emb_data.iloc[0]['cv_f1_mean']:.4f} ± {emb_data.iloc[0]['cv_f1_std']:.4f}")
    print(f"    Accuracy: {emb_data.iloc[0]['cv_accuracy_mean']:.4f}")
    print(f"    Tiempo: {emb_data.iloc[0]['cv_fit_time_mean']:.2f}s")
    print(f"  Peor: {emb}-{int(emb_data.iloc[-1]['dimensionality'])}-{emb_data.iloc[-1]['classifier'].upper()}")
    print(f"    F1: {emb_data.iloc[-1]['cv_f1_mean']:.4f}")
    print(f"  Promedio F1: {emb_data['cv_f1_mean'].mean():.4f}")
    print(f"  Promedio Tiempo: {emb_data['cv_fit_time_mean'].mean():.2f}s")

# Análisis por clasificador
print("\n" + "="*80)
print("ANÁLISIS POR CLASIFICADOR")
print("="*80)

for clf in ['lr', 'svm', 'rf']:
    clf_data = all_df[all_df['classifier'] == clf].sort_values('cv_f1_mean', ascending=False)
    print(f"\n{clf.upper()}:")
    print(f"  Experimentos: {len(clf_data)}")
    print(f"  Mejor: {clf_data.iloc[0]['embedding'].upper()}-{int(clf_data.iloc[0]['dimensionality'])}-{clf.upper()}")
    print(f"    F1: {clf_data.iloc[0]['cv_f1_mean']:.4f}")
    print(f"  Promedio F1: {clf_data['cv_f1_mean'].mean():.4f}")
    print(f"  Promedio Tiempo: {clf_data['cv_fit_time_mean'].mean():.2f}s")

# Análisis por dimensionalidad
print("\n" + "="*80)
print("ANÁLISIS POR DIMENSIONALIDAD")
print("="*80)

for dim in [100, 200, 300, 768]:
    dim_data = all_df[all_df['dimensionality'] == dim]
    if len(dim_data) > 0:
        dim_sorted = dim_data.sort_values('cv_f1_mean', ascending=False)
        print(f"\nDIM {dim}:")
        print(f"  Experimentos: {len(dim_data)}")
        print(f"  Mejor: {dim_sorted.iloc[0]['embedding'].upper()}-{dim}-{dim_sorted.iloc[0]['classifier'].upper()}")
        print(f"    F1: {dim_sorted.iloc[0]['cv_f1_mean']:.4f}")
        print(f"  Promedio F1: {dim_data['cv_f1_mean'].mean():.4f}")

# Insights clave
print("\n" + "="*80)
print("INSIGHTS CLAVE")
print("="*80)

print("\n1. TOP 5 MODELOS:")
for i, row in all_df_sorted.head(5).iterrows():
    print(f"   {i+1}. {row['embedding'].upper()}-{int(row['dimensionality'])}-{row['classifier'].upper()}: F1={row['cv_f1_mean']:.4f}, Tiempo={row['cv_fit_time_mean']:.2f}s")

print("\n2. CLASIFICADORES EN TOP 10:")
top10 = all_df_sorted.head(10)
for clf in ['lr', 'svm', 'rf']:
    count = len(top10[top10['classifier'] == clf])
    print(f"   {clf.upper()}: {count} modelos")

print("\n3. EMBEDDINGS EN TOP 10:")
for emb in ['word2vec', 'fasttext', 'bert']:
    count = len(top10[top10['embedding'] == emb])
    print(f"   {emb.upper()}: {count} modelos")

print("\n4. TRADE-OFF VELOCIDAD/PERFORMANCE (TOP 5 MÁS RÁPIDOS CON F1 > 0.94):")
fast_good = all_df[all_df['cv_f1_mean'] > 0.94].sort_values('cv_fit_time_mean').head(5)
for i, row in fast_good.iterrows():
    print(f"   {row['embedding'].upper()}-{int(row['dimensionality'])}-{row['classifier'].upper()}: F1={row['cv_f1_mean']:.4f}, Tiempo={row['cv_fit_time_mean']:.2f}s")

print("\n5. ESTADÍSTICAS GENERALES:")
print(f"   Accuracy media: {all_df['cv_accuracy_mean'].mean():.4f}")
print(f"   F1 medio: {all_df['cv_f1_mean'].mean():.4f}")
print(f"   Tiempo medio: {all_df['cv_fit_time_mean'].mean():.2f}s")
print(f"   Modelos con F1 > 0.95: {len(all_df[all_df['cv_f1_mean'] > 0.95])}")
print(f"   Modelos con F1 > 0.94: {len(all_df[all_df['cv_f1_mean'] > 0.94])}")
print(f"   Modelos con F1 > 0.93: {len(all_df[all_df['cv_f1_mean'] > 0.93])}")
print(f"   Modelos con F1 > 0.92: {len(all_df[all_df['cv_f1_mean'] > 0.92])}")
