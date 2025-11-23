"""
Script para verificar que los datos del README coincidan con los CSV
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
print("VERIFICACIÓN DE DATOS DEL README")
print("="*80)

# 1. Verificar mejor modelo
best = all_df_sorted.iloc[0]
print(f"\n1. MEJOR MODELO:")
print(f"   README dice: Word2Vec-300-RF, Acc=95.82%, F1=95.81%, Tiempo=0.88s")
print(f"   CSV dice: {best['embedding'].upper()}-{int(best['dimensionality'])}-{best['classifier'].upper()}")
print(f"            Acc={best['cv_accuracy_mean']:.4f}, F1={best['cv_f1_mean']:.4f}, Tiempo={best['cv_fit_time_mean']:.2f}s")
match = (best['embedding'] == 'word2vec' and
         int(best['dimensionality']) == 300 and
         best['classifier'] == 'rf' and
         abs(best['cv_accuracy_mean'] - 0.9582) < 0.0001 and
         abs(best['cv_f1_mean'] - 0.9581) < 0.0001)
print(f"   [OK] CORRECTO" if match else "   [ERROR] INCORRECTO")

# 2. Verificar mejores por embedding
print(f"\n2. MEJORES POR EMBEDDING:")
for emb in ['word2vec', 'fasttext', 'bert']:
    best_emb = all_df[all_df['embedding'] == emb].sort_values('cv_f1_mean', ascending=False).iloc[0]
    print(f"\n   {emb.upper()}:")
    print(f"   Mejor: {emb}-{int(best_emb['dimensionality'])}-{best_emb['classifier'].upper()}")
    print(f"   F1={best_emb['cv_f1_mean']:.4f}, Acc={best_emb['cv_accuracy_mean']:.4f}, Tiempo={best_emb['cv_fit_time_mean']:.2f}s")

# Verificar específicamente lo que dice el README
readme_claims = {
    'word2vec': {'dim': 300, 'clf': 'rf', 'f1': 0.9581, 'acc': 0.9582, 'time': 0.88},
    'fasttext': {'dim': 300, 'clf': 'rf', 'f1': 0.9469, 'acc': 0.9470, 'time': 0.89},
    'bert': {'dim': 768, 'clf': 'lr', 'f1': 0.9455, 'acc': 0.9456, 'time': 5.05}
}

for emb, expected in readme_claims.items():
    actual = all_df[(all_df['embedding'] == emb) &
                    (all_df['dimensionality'] == expected['dim']) &
                    (all_df['classifier'] == expected['clf'])].iloc[0]

    match = (abs(actual['cv_f1_mean'] - expected['f1']) < 0.0001 and
             abs(actual['cv_accuracy_mean'] - expected['acc']) < 0.0001)
    print(f"   {emb.upper()}-{expected['dim']}-{expected['clf'].upper()}: {'[OK]' if match else '[ERROR]'}")

# 3. Verificar TOP 10
print(f"\n3. TOP 10 RANKING:")
top10_readme = [
    ('word2vec', 300, 'rf'),
    ('word2vec', 200, 'rf'),
    ('word2vec', 100, 'rf'),
    ('fasttext', 300, 'rf'),
    ('fasttext', 200, 'rf'),
    ('bert', 768, 'lr'),
    ('fasttext', 100, 'rf'),
    ('bert', 768, 'svm'),
    ('word2vec', 200, 'svm'),
    ('bert', 300, 'lr')
]

top10_actual = [(row['embedding'], int(row['dimensionality']), row['classifier'])
                for _, row in all_df_sorted.head(10).iterrows()]

for i, (readme, actual) in enumerate(zip(top10_readme, top10_actual), 1):
    match = readme == actual
    status = "[OK]" if match else "[ERROR]"
    print(f"   #{i}: README={readme[0]}-{readme[1]}-{readme[2]} vs CSV={actual[0]}-{actual[1]}-{actual[2]} {status}")

# 4. Verificar promedios por embedding
print(f"\n4. PROMEDIOS POR EMBEDDING:")
for emb in ['word2vec', 'fasttext', 'bert']:
    emb_data = all_df[all_df['embedding'] == emb]
    avg_f1 = emb_data['cv_f1_mean'].mean()
    avg_time = emb_data['cv_fit_time_mean'].mean()
    print(f"   {emb.upper()}: Promedio F1={avg_f1:.4f}, Promedio Tiempo={avg_time:.2f}s")

readme_emb_avgs = {'word2vec': 0.9432, 'fasttext': 0.9331, 'bert': 0.9282}
for emb, expected in readme_emb_avgs.items():
    actual = all_df[all_df['embedding'] == emb]['cv_f1_mean'].mean()
    match = abs(actual - expected) < 0.0001
    print(f"   {emb.upper()} README vs CSV: {expected:.4f} vs {actual:.4f} {'[OK]' if match else '[ERROR]'}")

# 5. Verificar promedios por clasificador
print(f"\n5. PROMEDIOS POR CLASIFICADOR:")
for clf in ['lr', 'svm', 'rf']:
    clf_data = all_df[all_df['classifier'] == clf]
    avg_f1 = clf_data['cv_f1_mean'].mean()
    avg_time = clf_data['cv_fit_time_mean'].mean()
    print(f"   {clf.upper()}: Promedio F1={avg_f1:.4f}, Promedio Tiempo={avg_time:.2f}s")

readme_clf_avgs = {'lr': 0.9325, 'svm': 0.9337, 'rf': 0.9363}
for clf, expected in readme_clf_avgs.items():
    actual = all_df[all_df['classifier'] == clf]['cv_f1_mean'].mean()
    match = abs(actual - expected) < 0.0001
    print(f"   {clf.upper()} README vs CSV: {expected:.4f} vs {actual:.4f} {'[OK]' if match else '[ERROR]'}")

# 6. Verificar estadísticas generales
print(f"\n6. ESTADÍSTICAS GENERALES:")
min_acc = all_df['cv_accuracy_mean'].min()
max_acc = all_df['cv_accuracy_mean'].max()
models_f1_95 = len(all_df[all_df['cv_f1_mean'] > 0.95])
models_f1_94 = len(all_df[all_df['cv_f1_mean'] > 0.94])
models_f1_92 = len(all_df[all_df['cv_f1_mean'] > 0.92])

print(f"   Accuracy mínima: {min_acc:.4f} (README dice: 0.9030)")
print(f"   Accuracy máxima: {max_acc:.4f} (README dice: 0.9582)")
print(f"   Modelos F1>0.95: {models_f1_95} (README dice: 3)")
print(f"   Modelos F1>0.94: {models_f1_94} (README dice: 10)")
print(f"   Modelos F1>0.92: {models_f1_92} (README dice: 27)")

print(f"   Accuracy mínima: {'[OK]' if abs(min_acc - 0.9030) < 0.0001 else '[ERROR]'}")
print(f"   Accuracy máxima: {'[OK]' if abs(max_acc - 0.9582) < 0.0001 else '[ERROR]'}")
print(f"   Modelos F1>0.95: {'[OK]' if models_f1_95 == 3 else '[ERROR]'}")
print(f"   Modelos F1>0.94: {'[OK]' if models_f1_94 == 10 else '[ERROR]'}")
print(f"   Modelos F1>0.92: {'[OK]' if models_f1_92 == 27 else '[ERROR]'}")

print("\n" + "="*80)
print("RESUMEN DE VERIFICACION")
print("="*80)
print("Revisa los [ERROR] arriba para identificar discrepancias entre README y CSV")
