# Script de PowerShell para ejecutar todos los 30 experimentos
# Ejecuta: .\run_all_experiments.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Ejecutando 30 Experimentos Completos" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$total = 30
$current = 0

# Word2Vec (9 experimentos)
Write-Host "[WORD2VEC] Iniciando 9 experimentos..." -ForegroundColor Yellow
foreach ($dim in 100,200,300) {
    foreach ($clf in "lr","svm","rf") {
        $current++
        Write-Host "[$current/$total] Word2Vec (dim=$dim) + $clf" -ForegroundColor Green
        python src/experiment_runner.py --embedding word2vec --dim $dim --classifier $clf
    }
}

# FastText (9 experimentos)
Write-Host ""
Write-Host "[FASTTEXT] Iniciando 9 experimentos..." -ForegroundColor Yellow
foreach ($dim in 100,200,300) {
    foreach ($clf in "lr","svm","rf") {
        $current++
        Write-Host "[$current/$total] FastText (dim=$dim) + $clf" -ForegroundColor Green
        python src/experiment_runner.py --embedding fasttext --dim $dim --classifier $clf
    }
}

# BERT con PCA (9 experimentos)
Write-Host ""
Write-Host "[BERT+PCA] Iniciando 9 experimentos..." -ForegroundColor Yellow
foreach ($dim in 100,200,300) {
    foreach ($clf in "lr","svm","rf") {
        $current++
        Write-Host "[$current/$total] BERT + PCA (dim=$dim) + $clf" -ForegroundColor Green
        python src/experiment_runner.py --embedding bert --dim $dim --classifier $clf
    }
}

# BERT sin PCA (3 experimentos)
Write-Host ""
Write-Host "[BERT] Iniciando 3 experimentos (sin PCA)..." -ForegroundColor Yellow
foreach ($clf in "lr","svm","rf") {
    $current++
    Write-Host "[$current/$total] BERT (768 dims) + $clf" -ForegroundColor Green
    python src/experiment_runner.py --embedding bert --dim 768 --classifier $clf
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TODOS LOS EXPERIMENTOS COMPLETADOS!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Resultados guardados en: reports/results_*.csv" -ForegroundColor Yellow
Write-Host ""
Write-Host "Para consolidar resultados, ejecuta:" -ForegroundColor Yellow
Write-Host "python -c \"import pandas as pd; import glob; df=pd.concat([pd.read_csv(f) for f in glob.glob('reports/results_*.csv')]); df.sort_values('cv_f1_mean', ascending=False).to_csv('reports/all_results.csv', index=False); print(df.sort_values('cv_f1_mean', ascending=False).head(10))\"" -ForegroundColor Cyan
