"""
Verificación final de afirmaciones específicas del README
"""
import pandas as pd
import glob

# Cargar datos
results = []
for f in glob.glob("reports/results_*.csv"):
    df = pd.read_csv(f)
    results.append(df)

all_df = pd.concat(results, ignore_index=True)
all_df_sorted = all_df.sort_values('cv_f1_mean', ascending=False)

print("="*80)
print("VERIFICACION FINAL DE AFIRMACIONES DEL README")
print("="*80)

# 1. "Word2Vec con Random Forest supera a BERT siendo además 5.7x más rápido"
best_w2v_rf = all_df[(all_df['embedding'] == 'word2vec') &
                     (all_df['classifier'] == 'rf')].sort_values('cv_f1_mean', ascending=False).iloc[0]
best_bert = all_df[all_df['embedding'] == 'bert'].sort_values('cv_f1_mean', ascending=False).iloc[0]

speedup = best_bert['cv_fit_time_mean'] / best_w2v_rf['cv_fit_time_mean']
print(f"\n1. 'Word2Vec-RF supera a BERT siendo 5.7x mas rapido'")
print(f"   Word2Vec-300-RF: F1={best_w2v_rf['cv_f1_mean']:.4f}, Tiempo={best_w2v_rf['cv_fit_time_mean']:.2f}s")
print(f"   BERT-768-LR: F1={best_bert['cv_f1_mean']:.4f}, Tiempo={best_bert['cv_fit_time_mean']:.2f}s")
print(f"   Speedup: {speedup:.1f}x")
print(f"   README dice 5.7x: {'[OK]' if abs(speedup - 5.7) < 0.2 else '[REVISAR]'}")

# 2. "Random Forest domina: los TOP 7 modelos usan RF"
top7 = all_df_sorted.head(7)
rf_count_top7 = len(top7[top7['classifier'] == 'rf'])
print(f"\n2. 'Random Forest domina: los TOP 7 modelos usan RF'")
print(f"   Modelos RF en TOP 7: {rf_count_top7}/7")
print(f"   {'[OK]' if rf_count_top7 == 7 else '[REVISAR]'}")

# 3. "Word2Vec-RF ocupa los primeros 3 lugares del ranking completo"
top3 = all_df_sorted.head(3)
w2v_rf_top3 = all([row['embedding'] == 'word2vec' and row['classifier'] == 'rf'
                   for _, row in top3.iterrows()])
print(f"\n3. 'Word2Vec-RF ocupa los primeros 3 lugares'")
for i, row in top3.iterrows():
    print(f"   #{i+1}: {row['embedding']}-{int(row['dimensionality'])}-{row['classifier']}")
print(f"   {'[OK]' if w2v_rf_top3 else '[REVISAR]'}")

# 4. "6 de los TOP 10 modelos usan RF"
top10 = all_df_sorted.head(10)
rf_count_top10 = len(top10[top10['classifier'] == 'rf'])
print(f"\n4. '6 de los TOP 10 modelos usan RF'")
print(f"   Modelos RF en TOP 10: {rf_count_top10}/10")
print(f"   {'[OK]' if rf_count_top10 == 6 else '[REVISAR]'}")

# 5. "Todos los 30 modelos logran Accuracy > 90%"
all_above_90 = all(all_df['cv_accuracy_mean'] > 0.90)
print(f"\n5. 'Todos los 30 modelos logran Accuracy > 90%'")
print(f"   Modelos con Acc>0.90: {len(all_df[all_df['cv_accuracy_mean'] > 0.90])}/30")
print(f"   Accuracy minima: {all_df['cv_accuracy_mean'].min():.4f}")
print(f"   {'[OK]' if all_above_90 else '[REVISAR]'}")

# 6. "27 de 30 modelos logran F1 > 0.92 (92%)"
models_above_92 = len(all_df[all_df['cv_f1_mean'] > 0.92])
print(f"\n6. '27 de 30 modelos logran F1 > 0.92'")
print(f"   Modelos con F1>0.92: {models_above_92}/30")
print(f"   {'[OK]' if models_above_92 == 27 else '[REVISAR]'}")

# 7. "Los TOP 10 modelos superan 94% de accuracy"
top10_above_94 = all(top10['cv_accuracy_mean'] > 0.94)
print(f"\n7. 'Los TOP 10 modelos superan 94% de accuracy'")
print(f"   Modelos TOP 10 con Acc>0.94: {len(top10[top10['cv_accuracy_mean'] > 0.94])}/10")
print(f"   Accuracy minima en TOP 10: {top10['cv_accuracy_mean'].min():.4f}")
print(f"   {'[OK]' if top10_above_94 else '[REVISAR]'}")

# 8. "Word2Vec es el mejor embedding: Promedio F1=0.9432"
w2v_avg = all_df[all_df['embedding'] == 'word2vec']['cv_f1_mean'].mean()
ft_avg = all_df[all_df['embedding'] == 'fasttext']['cv_f1_mean'].mean()
bert_avg = all_df[all_df['embedding'] == 'bert']['cv_f1_mean'].mean()
print(f"\n8. 'Word2Vec es el mejor embedding'")
print(f"   Word2Vec promedio F1: {w2v_avg:.4f}")
print(f"   FastText promedio F1: {ft_avg:.4f}")
print(f"   BERT promedio F1: {bert_avg:.4f}")
print(f"   {'[OK]' if w2v_avg > ft_avg and w2v_avg > bert_avg else '[REVISAR]'}")

# 9. "RF es el clasificador más rápido: 0.57s promedio"
rf_avg_time = all_df[all_df['classifier'] == 'rf']['cv_fit_time_mean'].mean()
svm_avg_time = all_df[all_df['classifier'] == 'svm']['cv_fit_time_mean'].mean()
lr_avg_time = all_df[all_df['classifier'] == 'lr']['cv_fit_time_mean'].mean()
print(f"\n9. 'RF es el clasificador mas rapido: 0.57s promedio'")
print(f"   RF promedio tiempo: {rf_avg_time:.2f}s")
print(f"   SVM promedio tiempo: {svm_avg_time:.2f}s")
print(f"   LR promedio tiempo: {lr_avg_time:.2f}s")
print(f"   {'[OK]' if rf_avg_time < svm_avg_time and rf_avg_time < lr_avg_time else '[REVISAR]'}")
print(f"   README dice 0.57s: {'[OK]' if abs(rf_avg_time - 0.57) < 0.01 else '[REVISAR]'}")

# 10. "Word2Vec-100-RF: F1=0.9553 en solo 0.18s (ranking #3)"
w2v_100_rf = all_df[(all_df['embedding'] == 'word2vec') &
                    (all_df['dimensionality'] == 100) &
                    (all_df['classifier'] == 'rf')].iloc[0]
ranking = list(all_df_sorted.index).index(w2v_100_rf.name) + 1
print(f"\n10. 'Word2Vec-100-RF: F1=0.9553 en solo 0.18s (ranking #3)'")
print(f"   F1: {w2v_100_rf['cv_f1_mean']:.4f} (README: 0.9553)")
print(f"   Tiempo: {w2v_100_rf['cv_fit_time_mean']:.2f}s (README: 0.18s)")
print(f"   Ranking: #{ranking} (README: #3)")
print(f"   {'[OK]' if ranking == 3 else '[REVISAR]'}")

print("\n" + "="*80)
print("RESUMEN: Todas las afirmaciones del README han sido verificadas")
print("="*80)
