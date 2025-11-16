"""
Generación de embeddings para correos electrónicos.
Implementa: Word2Vec, FastText y BERT (bert-base-uncased).
"""

import numpy as np
from typing import List, Optional
from gensim.models import Word2Vec, FastText
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
import torch


class Word2VecEmbedder:
    # ... (This class remains unchanged from your original code) ...
    """
    Generador de embeddings usando Word2Vec entrenado localmente.
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 10,
        seed: int = 42,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.seed = seed
        self.model = None

    def train(self, texts: List[str]):
        """Entrena el modelo Word2Vec con el corpus proporcionado."""
        sentences = [text.split() for text in texts]
        print(f"Entrenando Word2Vec (dim={self.vector_size})...")
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            seed=self.seed,
        )
        print(f"Word2Vec entrenado. Vocabulario: {len(self.model.wv)} palabras.")

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transforma textos a vectores embeddings (promedio de palabras)."""
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a train() primero.")
        embeddings = []
        for text in texts:
            words = text.split()
            word_vectors = [
                self.model.wv[word] for word in words if word in self.model.wv
            ]
            if word_vectors:
                embeddings.append(np.mean(word_vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.vector_size))
        return np.array(embeddings)


class FastTextEmbedder:
    # ... (This class remains unchanged from your original code) ...
    """
    Generador de embeddings usando FastText entrenado localmente.
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 10,
        seed: int = 42,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.seed = seed
        self.model = None

    def train(self, texts: List[str]):
        """Entrena el modelo FastText con el corpus proporcionado."""
        sentences = [text.split() for text in texts]
        print(f"Entrenando FastText (dim={self.vector_size})...")
        self.model = FastText(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            seed=self.seed,
        )
        print(f"FastText entrenado. Vocabulario: {len(self.model.wv)} palabras.")

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transforma textos a vectores embeddings (promedio de palabras)."""
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a train() primero.")
        embeddings = []
        for text in texts:
            words = text.split()
            word_vectors = [self.model.wv[word] for word in words if word]
            if word_vectors:
                embeddings.append(np.mean(word_vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.vector_size))
        return np.array(embeddings)


class BERTEmbedder:
    """
    Generador de embeddings usando BERT preentrenado (bert-base-uncased).
    Soporta reducción dimensional opcional con PCA de forma segura (sin data leakage).
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        batch_size: int = 16,
        device: str = None,
        pca_dim: Optional[int] = None,
        random_state: int = 42,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pca_dim = pca_dim
        self.random_state = random_state
        self.pca = None

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Cargando modelo BERT ({model_name}) en {self.device}...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        original_dim = self.model.config.hidden_size
        if pca_dim:
            print(
                f"BERT cargado. Dimensión original: {original_dim} → PCA a {pca_dim} dims"
            )
        else:
            print(f"BERT cargado. Dimensión de salida: {original_dim}")

    def _generate_base_embeddings(self, texts: List[str]) -> np.ndarray:
        """Helper para generar embeddings BERT de 768 dimensiones."""
        all_embeddings = []
        print(f"Generando embeddings BERT para {len(texts)} textos...")
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

        return np.vstack(all_embeddings)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Genera embeddings y AJUSTA el PCA en los datos proporcionados (datos de entrenamiento).
        """
        base_embeddings = self._generate_base_embeddings(texts)

        if self.pca_dim:
            print(
                f"Ajustando PCA para reducir de {base_embeddings.shape[1]} a {self.pca_dim} dims..."
            )
            self.pca = PCA(n_components=self.pca_dim, random_state=self.random_state)
            reduced_embeddings = self.pca.fit_transform(base_embeddings)
            variance_explained = np.sum(self.pca.explained_variance_ratio_)
            print(f"PCA ajustado. Varianza explicada: {variance_explained:.4f}")
            return reduced_embeddings
        else:
            return base_embeddings

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Genera embeddings y APLICA un PCA ya ajustado (para datos de validación/prueba).
        """
        base_embeddings = self._generate_base_embeddings(texts)

        if self.pca_dim:
            if self.pca is None:
                raise RuntimeError(
                    "PCA no ha sido ajustado. Llama a fit_transform() primero."
                )
            print(f"Aplicando PCA ajustado...")
            return self.pca.transform(base_embeddings)
        else:
            return base_embeddings
