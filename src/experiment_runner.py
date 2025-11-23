"""
Script principal para ejecutar experimentos.
VERSIÓN FINAL Y ROBUSTA:
- Detecta automáticamente si hay una GPU disponible y usa `cuml`.
- Si no hay GPU, usa `sklearn` en la CPU de manera optimizada.
- Carga datos ya preprocesados desde un archivo Parquet.
- Guarda los embeddings generados en disco (caché) para evitar recalcularlos.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import sys
import time
import numpy as np
import pandas as pd
import multiprocessing
import torch

# --- DYNAMIC IMPORTS & GPU/CPU SETUP ---
# Intenta importar librerías de GPU. Si falla, usa la CPU.

# --- Imports de Sklearn como fallback y para métricas ---
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression as skLR
from sklearn.svm import SVC as skSVC
from sklearn.ensemble import RandomForestClassifier as skRF
from sklearn.decomposition import PCA as skPCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Imports de módulos locales ---
sys.path.insert(0, ".")
from embeddings import Word2VecEmbedder, FastTextEmbedder, BERTEmbedder

cuLR, cuLinearSVC, cuRF, cuPCA, cp = (None, None, None, None, None)

try:
    import cupy as cp
    from cuml.linear_model import LogisticRegression as cuLR
    from cuml.svm import LinearSVC as cuLinearSVC
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.decomposition import PCA as cuPCA

    if torch.cuda.is_available():
        use_gpu = True
        print("GPU y cuML encontrados. Usando algoritmos acelerados por GPU.")
    else:
        use_gpu = False
        print("cuML está instalado, pero PyTorch no detectó una GPU. Usando CPU.")

except ImportError:
    use_gpu = False
    print("cuML no encontrado. Usando scikit-learn en CPU.")

RANDOM_SEED = 42
CPU_CORES = multiprocessing.cpu_count()


def get_classifier(classifier_type: str):
    """Devuelve un clasificador de cuML (GPU) o sklearn (CPU) según la disponibilidad."""
    if use_gpu:
        if classifier_type == "lr":
            return cuLR(max_iter=1000)
        elif classifier_type == "svm":
            return cuLinearSVC()
        elif classifier_type == "rf":
            return cuRF(n_estimators=100)
    else:  # Fallback a CPU
        if classifier_type == "lr":
            return skLR(max_iter=2000, random_state=RANDOM_SEED, n_jobs=CPU_CORES)
        elif classifier_type == "svm":
            return skSVC(kernel="linear", random_state=RANDOM_SEED, probability=True)
        elif classifier_type == "rf":
            return skRF(n_estimators=100, random_state=RANDOM_SEED, n_jobs=CPU_CORES)
    raise ValueError(f"Clasificador no soportado: {classifier_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Experimentación con embeddings y clasificadores"
    )
    # ... (argument parsing sin cambios)
    parser.add_argument(
        "--embedding", required=True, choices=["word2vec", "fasttext", "bert"]
    )
    parser.add_argument("--dim", type=int, default=100, choices=[100, 200, 300, 768])
    parser.add_argument("--classifier", required=True, choices=["lr", "svm", "rf"])
    parser.add_argument("--k_folds", type=int, default=5)
    args = parser.parse_args()

    print(
        f"\n--- Experimento: {args.embedding.upper()} (dim={args.dim}) + {args.classifier.upper()} ---"
    )

    # --- Carga de datos preprocesados ---
    PROCESSED_FILE = "data/processed/preprocessed_emails_sample.parquet"
    try:
        df = pd.read_parquet(PROCESSED_FILE)
    except FileNotFoundError:
        print(
            "\nERROR: No se encontró el archivo preprocesado. Ejecuta 'preprocess_data.py' primero."
        )
        sys.exit(1)
    texts = df["text_processed"].tolist()
    labels = df["label"].values

    # --- Lógica de Caché de Embeddings para BERT ---
    X = None
    if args.embedding == "bert":
        EMBEDDING_CACHE_DIR = "data/embeddings"
        os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
        embedding_cache_file = f"{EMBEDDING_CACHE_DIR}/{args.embedding}_{args.dim}.npy"

        if os.path.exists(embedding_cache_file):
            print(f"Cargando embeddings cacheados desde '{embedding_cache_file}'...")
            X = np.load(embedding_cache_file)
        else:
            print("Generando embeddings BERT (se guardarán en caché)...")
            bert = BERTEmbedder()
            X_full_dim = bert.transform(texts)

            if args.dim != 768:
                print(f"Aplicando PCA para reducir a {args.dim} dimensiones...")
                if use_gpu:
                    pca = cuPCA(n_components=args.dim)
                    X_gpu = pca.fit_transform(cp.asarray(X_full_dim))
                    X = cp.asnumpy(X_gpu)
                else:  # PCA en CPU
                    pca = skPCA(n_components=args.dim, random_state=RANDOM_SEED)
                    X = pca.fit_transform(X_full_dim)
            else:
                X = X_full_dim

            print(f"Guardando embeddings en caché: '{embedding_cache_file}'")
            np.save(embedding_cache_file, X)

    # --- Cross-Validation ---
    fold_scores = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "fit_time": [],
    }
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=RANDOM_SEED)

    if args.embedding in ["word2vec", "fasttext"]:
        print(f"Iniciando CV con re-entrenamiento para {args.embedding.upper()}...")
        texts_array = np.array(texts, dtype=object)
        for fold, (train_idx, test_idx) in enumerate(skf.split(texts_array, labels)):
            print(f"--- Fold {fold + 1}/{args.k_folds} ---")
            train_texts, test_texts = texts_array[train_idx], texts_array[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            embedder_class = (
                Word2VecEmbedder if args.embedding == "word2vec" else FastTextEmbedder
            )
            embedder = embedder_class(vector_size=args.dim, workers=CPU_CORES)
            embedder.train(train_texts.tolist())
            X_train = embedder.transform(train_texts.tolist())
            X_test = embedder.transform(test_texts.tolist())

            classifier = get_classifier(args.classifier)

            if use_gpu:
                X_train, X_test, y_train = (
                    cp.asarray(X_train),
                    cp.asarray(X_test),
                    cp.asarray(y_train),
                )

            start_time = time.time()
            classifier.fit(X_train, y_train)
            fold_scores["fit_time"].append(time.time() - start_time)

            y_pred = classifier.predict(X_test)
            if use_gpu:
                y_pred = cp.asnumpy(y_pred)

            fold_scores["accuracy"].append(accuracy_score(y_test, y_pred))
            fold_scores["precision"].append(
                precision_score(y_test, y_pred, average="macro", zero_division=0)
            )
            fold_scores["recall"].append(
                recall_score(y_test, y_pred, average="macro", zero_division=0)
            )
            fold_scores["f1"].append(
                f1_score(y_test, y_pred, average="macro", zero_division=0)
            )

    else:  # BERT
        print("Iniciando CV para clasificador sobre embeddings BERT...")
        if use_gpu:
            X = cp.asarray(X)
        for fold, (train_idx, test_idx) in enumerate(
            skf.split(X if use_gpu else X, labels)
        ):
            print(f"--- Fold {fold + 1}/{args.k_folds} ---")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            classifier = get_classifier(args.classifier)

            if use_gpu:
                y_train = cp.asarray(y_train)

            start_time = time.time()
            classifier.fit(X_train, y_train)
            fold_scores["fit_time"].append(time.time() - start_time)

            y_pred = classifier.predict(X_test)
            if use_gpu:
                y_pred = cp.asnumpy(y_pred)

            fold_scores["accuracy"].append(accuracy_score(y_test, y_pred))
            fold_scores["precision"].append(
                precision_score(y_test, y_pred, average="macro", zero_division=0)
            )
            fold_scores["recall"].append(
                recall_score(y_test, y_pred, average="macro", zero_division=0)
            )
            fold_scores["f1"].append(
                f1_score(y_test, y_pred, average="macro", zero_division=0)
            )

    # --- Reporte de Resultados ---
    print(f"\n{'=' * 40}\nResultados Finales ({args.k_folds}-fold CV)\n{'=' * 40}")
    # ... (El resto del código para imprimir y guardar resultados no cambia)
    results = {}
    for metric, scores in fold_scores.items():
        results[metric] = {"mean": np.mean(scores), "std": np.std(scores)}
        print(
            f"{metric.upper():<10}: {results[metric]['mean']:.4f} ± {results[metric]['std']:.4f}"
        )

    result_summary = {
        "embedding": args.embedding,
        "dimensionality": args.dim,
        "pca_applied": "yes"
        if (args.embedding == "bert" and args.dim != 768)
        else "no",
        "classifier": args.classifier,
        "k_folds": args.k_folds,
        **{f"cv_{metric}_mean": values["mean"] for metric, values in results.items()},
        **{f"cv_{metric}_std": values["std"] for metric, values in results.items()},
    }
    results_df = pd.DataFrame([result_summary])
    os.makedirs("reports", exist_ok=True)
    output_file = f"reports/results_{args.embedding}_{args.dim}_{args.classifier}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResultados guardados en: {output_file}\n{'=' * 40}\n")


if __name__ == "__main__":
    main()
