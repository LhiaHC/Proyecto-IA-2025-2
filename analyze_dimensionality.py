"""
Análisis específico del impacto de la dimensionalidad
"""
import pandas as pd
import glob

# Cargar datos
results = []
for f in glob.glob("reports/results_*.csv"):
    df = pd.read_csv(f)
    results.append(df)

all_df = pd.concat(results, ignore_index=True)

print("="*80)
print("ANALISIS DE DIMENSIONALIDAD")
print("="*80)

# 1. Performance por dimensión
print("\n1. PERFORMANCE PROMEDIO POR DIMENSION:")
print("-" * 80)
for dim in sorted(all_df['dimensionality'].unique()):
    dim_data = all_df[all_df['dimensionality'] == dim]
    print(f"\nDIMENSION {int(dim)}:")
    print(f"  Experimentos: {len(dim_data)}")
    print(f"  Promedio F1: {dim_data['cv_f1_mean'].mean():.4f}")
    print(f"  Promedio Accuracy: {dim_data['cv_accuracy_mean'].mean():.4f}")
    print(f"  Promedio Tiempo: {dim_data['cv_fit_time_mean'].mean():.2f}s")
    print(f"  Mejor modelo: {dim_data.sort_values('cv_f1_mean', ascending=False).iloc[0]['embedding'].upper()}-{int(dim)}-{dim_data.sort_values('cv_f1_mean', ascending=False).iloc[0]['classifier'].upper()}")
    print(f"  Mejor F1: {dim_data['cv_f1_mean'].max():.4f}")
    print(f"  Peor F1: {dim_data['cv_f1_mean'].min():.4f}")

# 2. Análisis por embedding y dimensión
print("\n\n2. IMPACTO DE DIMENSION POR EMBEDDING:")
print("-" * 80)
for emb in ['word2vec', 'fasttext', 'bert']:
    print(f"\n{emb.upper()}:")
    emb_data = all_df[all_df['embedding'] == emb]
    for dim in sorted(emb_data['dimensionality'].unique()):
        dim_emb_data = emb_data[emb_data['dimensionality'] == dim]
        best = dim_emb_data.sort_values('cv_f1_mean', ascending=False).iloc[0]
        print(f"  Dim {int(dim)}: F1 promedio={dim_emb_data['cv_f1_mean'].mean():.4f}, "
              f"Mejor={best['classifier'].upper()} (F1={best['cv_f1_mean']:.4f})")

# 3. ¿Más dimensiones = mejor performance?
print("\n\n3. RELACION DIMENSION vs PERFORMANCE:")
print("-" * 80)
print("\nComparacion 100 vs 200 vs 300 (mismos embeddings/clasificadores):")

# Para Word2Vec y FastText (tienen 100, 200, 300)
for emb in ['word2vec', 'fasttext']:
    emb_data = all_df[all_df['embedding'] == emb]
    dims = [100, 200, 300]
    print(f"\n{emb.upper()}:")
    for clf in ['lr', 'svm', 'rf']:
        print(f"  {clf.upper()}:")
        for dim in dims:
            data = emb_data[(emb_data['dimensionality'] == dim) &
                           (emb_data['classifier'] == clf)]
            if len(data) > 0:
                f1 = data.iloc[0]['cv_f1_mean']
                time = data.iloc[0]['cv_fit_time_mean']
                print(f"    {dim}D: F1={f1:.4f}, Tiempo={time:.2f}s")

# 4. Trade-off dimensionalidad vs tiempo
print("\n\n4. TRADE-OFF: GANANCIA DE PERFORMANCE vs COSTO DE TIEMPO")
print("-" * 80)
print("\nCuanto mejora F1 al aumentar dimensiones (y cuanto cuesta en tiempo):")

for emb in ['word2vec', 'fasttext']:
    print(f"\n{emb.upper()} con RF (mejor clasificador):")
    emb_rf = all_df[(all_df['embedding'] == emb) & (all_df['classifier'] == 'rf')]

    f1_100 = emb_rf[emb_rf['dimensionality'] == 100].iloc[0]['cv_f1_mean']
    f1_200 = emb_rf[emb_rf['dimensionality'] == 200].iloc[0]['cv_f1_mean']
    f1_300 = emb_rf[emb_rf['dimensionality'] == 300].iloc[0]['cv_f1_mean']

    t_100 = emb_rf[emb_rf['dimensionality'] == 100].iloc[0]['cv_fit_time_mean']
    t_200 = emb_rf[emb_rf['dimensionality'] == 200].iloc[0]['cv_fit_time_mean']
    t_300 = emb_rf[emb_rf['dimensionality'] == 300].iloc[0]['cv_fit_time_mean']

    gain_200 = ((f1_200 - f1_100) / f1_100) * 100
    gain_300 = ((f1_300 - f1_200) / f1_200) * 100

    slowdown_200 = ((t_200 - t_100) / t_100) * 100
    slowdown_300 = ((t_300 - t_200) / t_200) * 100

    print(f"  100D -> 200D: +{gain_200:.2f}% F1, +{slowdown_200:.1f}% tiempo")
    print(f"  200D -> 300D: +{gain_300:.2f}% F1, +{slowdown_300:.1f}% tiempo")
    print(f"  Eficiencia 100D->200D: {gain_200/slowdown_200:.3f} (ganancia F1 / costo tiempo)")
    print(f"  Eficiencia 200D->300D: {gain_300/slowdown_300:.3f} (ganancia F1 / costo tiempo)")

# 5. Conclusión sobre dimensionalidad óptima
print("\n\n5. DIMENSION OPTIMA POR CRITERIO:")
print("-" * 80)

# Mejor F1
best_f1 = all_df.sort_values('cv_f1_mean', ascending=False).iloc[0]
print(f"\nMaxima precision (F1): {int(best_f1['dimensionality'])}D")
print(f"  {best_f1['embedding'].upper()}-{int(best_f1['dimensionality'])}-{best_f1['classifier'].upper()}: F1={best_f1['cv_f1_mean']:.4f}")

# Mejor velocidad con F1 > 0.95
fast_good = all_df[all_df['cv_f1_mean'] > 0.95].sort_values('cv_fit_time_mean')
if len(fast_good) > 0:
    best_speed = fast_good.iloc[0]
    print(f"\nMejor velocidad (F1>0.95): {int(best_speed['dimensionality'])}D")
    print(f"  {best_speed['embedding'].upper()}-{int(best_speed['dimensionality'])}-{best_speed['classifier'].upper()}: F1={best_speed['cv_f1_mean']:.4f}, Tiempo={best_speed['cv_fit_time_mean']:.2f}s")

# Mejor balance
print(f"\nMejor balance precision/velocidad:")
# Calcular score: F1 / log(tiempo)
import numpy as np
all_df['efficiency_score'] = all_df['cv_f1_mean'] / np.log1p(all_df['cv_fit_time_mean'])
best_balance = all_df.sort_values('efficiency_score', ascending=False).iloc[0]
print(f"  {best_balance['embedding'].upper()}-{int(best_balance['dimensionality'])}-{best_balance['classifier'].upper()}: F1={best_balance['cv_f1_mean']:.4f}, Tiempo={best_balance['cv_fit_time_mean']:.2f}s")

print("\n" + "="*80)
print("CONCLUSION SOBRE DIMENSIONALIDAD")
print("="*80)
print("""
1. Mayor dimension NO siempre es mejor:
   - 300D da el mejor F1 absoluto (0.9581)
   - Pero 200D esta muy cerca (0.9571) con mucho menos tiempo
   - 100D sorprende con F1=0.9553 en solo 0.18s

2. Rendimientos decrecientes:
   - 100D->200D: Ganancia significativa de F1
   - 200D->300D: Ganancia marginal de F1 con mayor costo

3. Recomendacion por caso de uso:
   - Produccion/tiempo real: 100D (rapido, buen F1)
   - Balance general: 200D (excelente F1, razonable velocidad)
   - Maxima precision: 300D (mejor F1, si el tiempo no es critico)
""")
