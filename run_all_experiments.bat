@echo off
REM Script para ejecutar todos los experimentos en Windows

echo ========================================
echo Ejecutando TODOS los experimentos
echo ========================================

REM Word2Vec experiments
echo.
echo ========== WORD2VEC EXPERIMENTS ==========
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding word2vec --dim 100 --classifier lr
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding word2vec --dim 100 --classifier svm
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding word2vec --dim 100 --classifier rf

iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding word2vec --dim 200 --classifier lr
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding word2vec --dim 200 --classifier svm
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding word2vec --dim 200 --classifier rf

iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding word2vec --dim 300 --classifier lr
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding word2vec --dim 300 --classifier svm
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding word2vec --dim 300 --classifier rf

REM FastText experiments
echo.
echo ========== FASTTEXT EXPERIMENTS ==========
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding fasttext --dim 100 --classifier lr
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding fasttext --dim 100 --classifier svm
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding fasttext --dim 100 --classifier rf

iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding fasttext --dim 200 --classifier lr
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding fasttext --dim 200 --classifier svm
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding fasttext --dim 200 --classifier rf

iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding fasttext --dim 300 --classifier lr
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding fasttext --dim 300 --classifier svm
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding fasttext --dim 300 --classifier rf

REM BERT experiments
echo.
echo ========== BERT EXPERIMENTS ==========
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding bert --dim 100 --classifier lr
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding bert --dim 100 --classifier svm
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding bert --dim 100 --classifier rf

iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding bert --dim 200 --classifier lr
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding bert --dim 200 --classifier svm
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding bert --dim 200 --classifier rf

iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding bert --dim 300 --classifier lr
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding bert --dim 300 --classifier svm
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding bert --dim 300 --classifier rf

iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding bert --dim 768 --classifier lr
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding bert --dim 768 --classifier svm
iaa_venv\Scripts\python.exe src/experiment_runner.py --embedding bert --dim 768 --classifier rf

echo.
echo ========================================
echo TODOS LOS EXPERIMENTOS COMPLETADOS
echo Resultados guardados en reports/
echo ========================================
