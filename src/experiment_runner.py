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
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Imports de módulos locales
sys.path.insert(0, "src")
from utils_00 import load_data, compute_metrics
from preproc_01 import preprocess_full
from embeddings_03 import Word2VecEmbedder, FastTextEmbedder, BERTEmbedder


RANDOM_SEED = 42


def get_embedder(embedding_type: str, dim: int):
    """
    Factory para crear el embedder según el tipo y dimensionalidad.

    Args:
        embedding_type: 'word2vec', 'fasttext', o 'bert'.
        dim: Dimensionalidad (100, 200, 300 para W2V/FT; 100/200/300/768 para BERT).

    Returns:
        Instancia del embedder correspondiente.
    """
    if embedding_type == 'word2vec':
        return Word2VecEmbedder(vector_size=dim, seed=RANDOM_SEED)
    elif embedding_type == 'fasttext':
        return FastTextEmbedder(vector_size=dim, seed=RANDOM_SEED)
    elif embedding_type == 'bert':
        # Si dim es 768, usar BERT sin PCA. Si es 100/200/300, usar BERT + PCA
        if dim == 768:
            return BERTEmbedder(model_name='bert-base-uncased', pca_dim=None, random_state=RANDOM_SEED)
        else:
            return BERTEmbedder(model_name='bert-base-uncased', pca_dim=dim, random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Tipo de embedding no soportado: {embedding_type}")


def get_classifier(classifier_type: str):
    """
    Factory para crear el clasificador.

    Args:
        classifier_type: 'lr' (Logistic Regression), 'svm' (SVM), o 'rf' (Random Forest).

    Returns:
        Instancia del clasificador.
    """
    if classifier_type == 'lr':
        return LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, n_jobs=-1)
    elif classifier_type == 'svm':
        return SVC(kernel='linear', random_state=RANDOM_SEED, probability=True)
    elif classifier_type == 'rf':
        return RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    else:
        raise ValueError(f"Tipo de clasificador no soportado: {classifier_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Experimentación con embeddings y clasificadores para detección de phishing"
    )
    parser.add_argument(
        "--embedding",
        type=str,
        required=True,
        choices=['word2vec', 'fasttext', 'bert'],
        help="Tipo de embedding: word2vec, fasttext, bert"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=100,
        choices=[100, 200, 300, 768],
        help="Dimensionalidad: 100/200/300 (W2V/FT/BERT+PCA), 768 (BERT sin PCA)"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        required=True,
        choices=['lr', 'svm', 'rf'],
        help="Clasificador: lr (Logistic Regression), svm (SVM), rf (Random Forest)"
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Número de folds para validación cruzada (default: 5)"
    )
    args = parser.parse_args()

    # Configuración del experimento
    embedding_name = args.embedding.upper()
    classifier_name = {'lr': 'LogisticRegression', 'svm': 'SVM', 'rf': 'RandomForest'}[args.classifier]

    if args.embedding in ['word2vec', 'fasttext']:
        exp_title = f"{embedding_name} (dim={args.dim}) + {classifier_name}"
    else:
        # BERT con o sin PCA
        if args.dim == 768:
            exp_title = f"{embedding_name} (bert-base, 768 dims) + {classifier_name}"
        else:
            exp_title = f"{embedding_name} (bert-base + PCA, {args.dim} dims) + {classifier_name}"

    print(f"\n{'='*70}")
    print(f"Experimento: {exp_title}")
    print(f"Validación cruzada: {args.k_folds}-fold")
    print(f"{'='*70}\n")

    # 1. Cargar datos
    df = load_data(filter_outliers=True)

    # 2. Preprocesamiento completo
    print("Aplicando preprocesamiento completo (limpieza, cacografía, tokenización, lematización)...")
    df["text_processed"] = df["text_combined"].apply(preprocess_full)
    print(f"Ejemplo de texto preprocesado:\n{df['text_processed'].iloc[0][:200]}...\n")

    # 3. Generar embeddings
    texts = df["text_processed"].tolist()
    labels = df["label"].values

    embedder = get_embedder(args.embedding, args.dim)

    if args.embedding in ['word2vec', 'fasttext']:
        # Entrenar embedder con todo el corpus
        print(f"\nEntrenando {embedding_name}...")
        embedder.train(texts)
        print("Generando embeddings para todos los textos...")
        X = embedder.transform(texts)
    else:  # bert
        print(f"\nGenerando embeddings BERT para todos los textos...")
        print("(Esto puede tomar varios minutos dependiendo del tamaño del dataset)")
        # Para BERT, generar embeddings completos (768 dims)
        # Si PCA está activado, se aplicará dentro de cross_validate con cada fold
        # Por ahora, generamos embeddings base
        X = embedder.transform(texts, fit_pca=False if args.dim == 768 else True)

    print(f"Embeddings generados: {X.shape}")

    # 4. Validación cruzada estratificada con k-fold
    print(f"\nIniciando validación cruzada {args.k_folds}-fold...")
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=RANDOM_SEED)

    classifier = get_classifier(args.classifier)

    # Definir métricas para cross_validate
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro', zero_division=0),
        'recall': make_scorer(recall_score, average='macro', zero_division=0),
        'f1': make_scorer(f1_score, average='macro', zero_division=0),
        'auc_roc': make_scorer(roc_auc_score, needs_proba=True)
    }

    # Ejecutar validación cruzada
    cv_results = cross_validate(
        classifier,
        X,
        labels,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
        verbose=1
    )

    # 5. Resultados de validación cruzada
    print(f"\n{'='*70}")
    print(f"Resultados de Validación Cruzada ({args.k_folds}-fold)")
    print(f"{'='*70}")

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    results = {}

    for metric in metrics:
        test_scores = cv_results[f'test_{metric}']
        mean_score = np.mean(test_scores)
        std_score = np.std(test_scores)
        results[metric] = {'mean': mean_score, 'std': std_score}

        metric_label = metric.replace('_', '-').upper()
        print(f"{metric_label:12s}: {mean_score:.4f} ± {std_score:.4f}")

    print(f"{'='*70}\n")

    # 6. Entrenar modelo final en todo el dataset
    print("Entrenando modelo final en el dataset completo...")
    classifier.fit(X, labels)
    y_pred = classifier.predict(X)

    # Obtener probabilidades para AUC-ROC
    if hasattr(classifier, 'predict_proba'):
        y_proba = classifier.predict_proba(X)
    elif hasattr(classifier, 'decision_function'):
        y_proba = classifier.decision_function(X)
    else:
        y_proba = None

    # 7. Métricas en todo el dataset (para referencia)
    metrics_full = compute_metrics(labels, y_pred, y_proba, dataset_name="Full Dataset")

    # 8. Guardar resultados
    result_summary = {
        'embedding': args.embedding,
        'dimensionality': args.dim,  # Ahora BERT también tiene dim variable
        'pca_applied': 'yes' if (args.embedding == 'bert' and args.dim != 768) else 'no',
        'classifier': args.classifier,
        'k_folds': args.k_folds,
        'cv_accuracy_mean': results['accuracy']['mean'],
        'cv_accuracy_std': results['accuracy']['std'],
        'cv_precision_mean': results['precision']['mean'],
        'cv_precision_std': results['precision']['std'],
        'cv_recall_mean': results['recall']['mean'],
        'cv_recall_std': results['recall']['std'],
        'cv_f1_mean': results['f1']['mean'],
        'cv_f1_std': results['f1']['std'],
        'cv_auc_roc_mean': results['auc_roc']['mean'],
        'cv_auc_roc_std': results['auc_roc']['std'],
    }

    # Guardar en CSV
    results_df = pd.DataFrame([result_summary])
    output_file = f"reports/results_{args.embedding}_{args.dim}_{args.classifier}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResultados guardados en: {output_file}\n")


if __name__ == "__main__":
    main()
