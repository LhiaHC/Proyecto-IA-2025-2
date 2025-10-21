#!/bin/bash
# Script de Bash para ejecutar todos los 30 experimentos
# Ejecuta: bash run_all_experiments.sh

echo "========================================"
echo "Ejecutando 30 Experimentos Completos"
echo "========================================"
echo ""

total=30
current=0

# Word2Vec (9 experimentos)
echo "[WORD2VEC] Iniciando 9 experimentos..."
for dim in 100 200 300; do
    for clf in lr svm rf; do
        current=$((current + 1))
        echo "[$current/$total] Word2Vec (dim=$dim) + $clf"
        python src/experiment_runner.py --embedding word2vec --dim $dim --classifier $clf
    done
done

# FastText (9 experimentos)
echo ""
echo "[FASTTEXT] Iniciando 9 experimentos..."
for dim in 100 200 300; do
    for clf in lr svm rf; do
        current=$((current + 1))
        echo "[$current/$total] FastText (dim=$dim) + $clf"
        python src/experiment_runner.py --embedding fasttext --dim $dim --classifier $clf
    done
done

# BERT con PCA (9 experimentos)
echo ""
echo "[BERT+PCA] Iniciando 9 experimentos..."
for dim in 100 200 300; do
    for clf in lr svm rf; do
        current=$((current + 1))
        echo "[$current/$total] BERT + PCA (dim=$dim) + $clf"
        python src/experiment_runner.py --embedding bert --dim $dim --classifier $clf
    done
done

# BERT sin PCA (3 experimentos)
echo ""
echo "[BERT] Iniciando 3 experimentos (sin PCA)..."
for clf in lr svm rf; do
    current=$((current + 1))
    echo "[$current/$total] BERT (768 dims) + $clf"
    python src/experiment_runner.py --embedding bert --dim 768 --classifier $clf
done

echo ""
echo "========================================"
echo "TODOS LOS EXPERIMENTOS COMPLETADOS!"
echo "========================================"
echo ""
echo "Resultados guardados en: reports/results_*.csv"
echo ""
echo "Para consolidar resultados, ejecuta:"
echo "python -c \"import pandas as pd; import glob; df=pd.concat([pd.read_csv(f) for f in glob.glob('reports/results_*.csv')]); df.sort_values('cv_f1_mean', ascending=False).to_csv('reports/all_results.csv', index=False); print(df.sort_values('cv_f1_mean', ascending=False).head(10))\""
