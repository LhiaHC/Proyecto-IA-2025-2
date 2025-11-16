"""
Script principal para ejecutar experimentos completos según metodología propuesta.
Combina: Embeddings (Word2Vec, FastText, BERT) x Dimensionalidades x Clasificadores (SVM, RF, LR)
con validación cruzada k-fold=5.

Uso:
    python src/experiment_runner.py --embedding word2vec --dim 100 --classifier lr
    python src/experiment_runner.py --embedding fasttext --dim 200 --classifier svm
    python src/experiment_runner.py --embedding bert --classifier rf
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


# Imports de módulos locales
sys.path.insert(0, "src")
from utils_00 import (
    load_data,
    compute_metrics,
)
from preproc_01 import (
    preprocess_full,
)
from embeddings_03 import (
    Word2VecEmbedder,
    FastTextEmbedder,
    BERTEmbedder,
)


RANDOM_SEED = 42


def get_embedder(embedding_type: str, dim: int):
    """
    Factory para crear el embedder según el tipo y dimensionalidad.
    """
    if embedding_type == "word2vec":
        return Word2VecEmbedder(vector_size=dim, seed=RANDOM_SEED)
    elif embedding_type == "fasttext":
        return FastTextEmbedder(vector_size=dim, seed=RANDOM_SEED)
    elif embedding_type == "bert":
        if dim == 768:
            return BERTEmbedder(
                model_name="bert-base-uncased", pca_dim=None, random_state=RANDOM_SEED
            )
        else:
            return BERTEmbedder(
                model_name="bert-base-uncased", pca_dim=dim, random_state=RANDOM_SEED
            )
    else:
        raise ValueError(f"Tipo de embedding no soportado: {embedding_type}")


def get_classifier(classifier_type: str):
    """
    Factory para crear el clasificador.
    """
    if classifier_type == "lr":
        return LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, n_jobs=-1)
    elif classifier_type == "svm":
        return SVC(kernel="linear", random_state=RANDOM_SEED, probability=True)
    elif classifier_type == "rf":
        return RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
        )
    else:
        raise ValueError(f"Tipo de clasificador no soportado: {classifier_type}")


def run_cv_for_static_embedders(
    texts, labels, embedding_type, dim, classifier_type, k_folds
):
    """
    FIX: Ejecuta validación cruzada MANUALMENTE para Word2Vec y FastText.
    Esto previene data leakage al entrenar el embedder en cada fold por separado.
    """
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_SEED)

    fold_scores = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc_roc": [],
        "fit_time": [],
    }
    texts_array = np.array(texts, dtype=object)

    for fold, (train_index, test_index) in enumerate(skf.split(texts, labels)):
        print(f"--- Fold {fold + 1}/{k_folds} ---")

        train_texts, test_texts = texts_array[train_index], texts_array[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        # 1. Entrenar embedder SOLO en el split de entrenamiento
        embedder = get_embedder(embedding_type, dim)
        embedder.train(train_texts.tolist())

        # 2. Transformar ambos splits
        X_train = embedder.transform(train_texts.tolist())
        X_test = embedder.transform(test_texts.tolist())

        # 3. Entrenar y evaluar clasificador
        classifier = get_classifier(classifier_type)
        start_time = time.time()
        classifier.fit(X_train, train_labels)
        fit_time = time.time() - start_time

        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)[:, 1]

        # 4. Guardar métricas del fold
        fold_scores["accuracy"].append(accuracy_score(test_labels, y_pred))
        fold_scores["precision"].append(
            precision_score(test_labels, y_pred, average="macro", zero_division=0)
        )
        fold_scores["recall"].append(
            recall_score(test_labels, y_pred, average="macro", zero_division=0)
        )
        fold_scores["f1"].append(
            f1_score(test_labels, y_pred, average="macro", zero_division=0)
        )
        fold_scores["auc_roc"].append(roc_auc_score(test_labels, y_proba))
        fold_scores["fit_time"].append(fit_time)

    # 5. Calcular promedios y std dev
    results = {}
    for metric, scores in fold_scores.items():
        results[metric] = {"mean": np.mean(scores), "std": np.std(scores)}

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Experimentación con embeddings y clasificadores para detección de phishing"
    )
    # ... (argument parser code remains the same as original)
    parser.add_argument(
        "--embedding",
        type=str,
        required=True,
        choices=["word2vec", "fasttext", "bert"],
        help="Tipo de embedding: word2vec, fasttext, bert",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=100,
        choices=[100, 200, 300, 768],
        help="Dimensionalidad: 100/200/300 (W2V/FT/BERT+PCA), 768 (BERT sin PCA)",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        required=True,
        choices=["lr", "svm", "rf"],
        help="Clasificador: lr (Logistic Regression), svm (SVM), rf (Random Forest)",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Número de folds para validación cruzada (default: 5)",
    )
    args = parser.parse_args()

    # Configuración del experimento
    embedding_name = args.embedding.upper()
    classifier_name = {"lr": "LogisticRegression", "svm": "SVM", "rf": "RandomForest"}[
        args.classifier
    ]

    if args.embedding in ["word2vec", "fasttext"]:
        exp_title = f"{embedding_name} (dim={args.dim}) + {classifier_name}"
    else:
        if args.dim == 768:
            exp_title = f"{embedding_name} (bert-base, 768 dims) + {classifier_name}"
        else:
            exp_title = f"{embedding_name} (bert-base + PCA, {args.dim} dims) + {classifier_name}"

    print(f"\n{'=' * 70}")
    print(f"Experimento: {exp_title}")
    print(f"Validación cruzada: {args.k_folds}-fold")
    print(f"{'=' * 70}\n")

    # 1. Cargar y preprocesar datos
    df = load_data(filter_outliers=True)
    print("Aplicando preprocesamiento completo...")
    df["text_processed"] = df["text_combined"].apply(preprocess_full)
    texts = df["text_processed"].tolist()
    labels = df["label"].values

    # 2. Ejecutar validación cruzada
    if args.embedding in ["word2vec", "fasttext"]:
        print(
            f"\nIniciando validación cruzada MANUAL para {embedding_name} para prevenir data leakage..."
        )
        results = run_cv_for_static_embedders(
            texts, labels, args.embedding, args.dim, args.classifier, args.k_folds
        )

    else:  # BERT
        print(
            f"\nGenerando embeddings BERT y usando Pipeline para prevenir data leakage en PCA..."
        )
        # Generar embeddings base (768 dim)
        bert_embedder_base = BERTEmbedder(
            model_name="bert-base-uncased", pca_dim=None, random_state=RANDOM_SEED
        )
        X_bert_full = bert_embedder_base.transform(texts)

        # Crear pipeline para aplicar PCA y clasificador en cada fold
        classifier = get_classifier(args.classifier)
        steps = [("classifier", classifier)]
        if args.dim != 768:
            steps.insert(
                0, ("pca", PCA(n_components=args.dim, random_state=RANDOM_SEED))
            )

        pipeline = Pipeline(steps)

        scoring = {
            "accuracy": "accuracy",
            "precision": "precision_macro",
            "recall": "recall_macro",
            "f1": "f1_macro",
            "roc_auc": "roc_auc",
        }

        cv_results = cross_validate(
            pipeline,
            X_bert_full,
            labels,
            cv=args.k_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
        )

        results = {}
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            test_scores = cv_results[f"test_{metric}"]
            results[metric] = {"mean": np.mean(test_scores), "std": np.std(test_scores)}
        # Add fit time
        fit_time_scores = cv_results["fit_time"]
        results["fit_time"] = {
            "mean": np.mean(fit_time_scores),
            "std": np.std(fit_time_scores),
        }

    # 3. Resultados
    print(f"\n{'=' * 70}")
    print(f"Resultados de Validación Cruzada ({args.k_folds}-fold)")
    print(f"{'=' * 70}")

    for metric, values in results.items():
        metric_label = (
            metric.replace("_", "-").upper() if metric != "roc_auc" else "AUC-ROC"
        )
        print(f"{metric_label:12s}: {values['mean']:.4f} ± {values['std']:.4f}")

    print(f"{'=' * 70}\n")

    # 4. Guardar resultados
    result_summary = {
        "embedding": args.embedding,
        "dimensionality": args.dim,
        "pca_applied": "yes"
        if (args.embedding == "bert" and args.dim != 768)
        else "no",
        "classifier": args.classifier,
        "k_folds": args.k_folds,
        "cv_accuracy_mean": results["accuracy"]["mean"],
        "cv_accuracy_std": results["accuracy"]["std"],
        "cv_precision_mean": results["precision"]["mean"],
        "cv_precision_std": results["precision"]["std"],
        "cv_recall_mean": results["recall"]["mean"],
        "cv_recall_std": results["recall"]["std"],
        "cv_f1_mean": results["f1"]["mean"],
        "cv_f1_std": results["f1"]["std"],
        "cv_auc_roc_mean": results.get("auc_roc", results.get("roc_auc", {})).get(
            "mean", 0
        ),  # Handle both names
        "cv_auc_roc_std": results.get("auc_roc", results.get("roc_auc", {})).get(
            "std", 0
        ),
        "cv_fit_time_mean": results["fit_time"]["mean"],
        "cv_fit_time_std": results["fit_time"]["std"],
    }

    results_df = pd.DataFrame([result_summary])
    output_file = f"reports/results_{args.embedding}_{args.dim}_{args.classifier}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Resultados guardados en: {output_file}\n")


if __name__ == "__main__":
    main()
