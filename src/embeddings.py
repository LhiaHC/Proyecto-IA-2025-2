"""
Generación de embeddings para correos electrónicos.
Implementa: Word2Vec, FastText y BERT (bert-base-uncased).
VERSION OPTIMIZADA: Usa gensim.models.KeyedVectors.get_mean_vector para una transformación ultrarrápida.
"""

import numpy as np
from typing import List, Optional
from gensim.models import Word2Vec, FastText
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
import torch


class Word2VecEmbedder:
    """Generador de embeddings usando Word2Vec entrenado localmente."""

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 10,
        seed: int = 42,
    ):
        (
            self.vector_size,
            self.window,
            self.min_count,
            self.workers,
            self.epochs,
            self.seed,
        ) = vector_size, window, min_count, workers, epochs, seed
        self.model = None

    def train(self, texts: List[str]):
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
        """VERSION OPTIMIZADA Y CORREGIDA: Transforma textos a embeddings de forma robusta."""
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a train() primero.")
        print(
            f"Generando embeddings Word2Vec para {len(texts)} textos (método optimizado)..."
        )

        embeddings = []
        for text in texts:
            words = text.split()
            if not words:
                # Si el texto está vacío después del preprocesamiento, añade un vector de ceros.
                embeddings.append(np.zeros(self.vector_size, dtype=np.float32))
                continue

            # get_mean_vector es rápido, pero puede devolver None si todas las palabras son OOV.
            vector = self.model.wv.get_mean_vector(words, ignore_missing=True)
            if vector is not None:
                embeddings.append(vector)
            else:
                # Si todas las palabras eran desconocidas, añade un vector de ceros.
                embeddings.append(np.zeros(self.vector_size, dtype=np.float32))

        return np.array(embeddings)


class FastTextEmbedder:
    """Generador de embeddings usando FastText entrenado localmente."""

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 10,
        seed: int = 42,
    ):
        (
            self.vector_size,
            self.window,
            self.min_count,
            self.workers,
            self.epochs,
            self.seed,
        ) = vector_size, window, min_count, workers, epochs, seed
        self.model = None

    def train(self, texts: List[str]):
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
        """VERSION OPTIMIZADA Y CORREGIDA: Transforma textos a embeddings de forma robusta."""
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a train() primero.")
        print(
            f"Generando embeddings FastText para {len(texts)} textos (método optimizado)..."
        )

        embeddings = []
        for text in texts:
            words = text.split()
            if not words:
                embeddings.append(np.zeros(self.vector_size, dtype=np.float32))
                continue

            # FastText puede generar vectores para OOV, así que get_mean_vector no debería devolver None.
            # La comprobación 'if not words' sigue siendo crucial.
            vector = self.model.wv.get_mean_vector(words)
            embeddings.append(vector)

        return np.array(embeddings)


class BERTEmbedder:
    """Generador de embeddings usando BERT preentrenado."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        batch_size: int = 16,
        device: str = None,
        pca_dim: Optional[int] = None,
        random_state: int = 42,
    ):
        (
            self.model_name,
            self.max_length,
            self.batch_size,
            self.pca_dim,
            self.random_state,
        ) = model_name, max_length, batch_size, pca_dim, random_state
        self.pca, self.device = (
            None,
            torch.device(
                device if device else "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )

        print(f"Cargando modelo BERT ({model_name}) en {self.device}...")
        self.tokenizer, self.model = (
            BertTokenizer.from_pretrained(model_name),
            BertModel.from_pretrained(model_name),
        )
        self.model.to(self.device).eval()
        print(
            f"BERT cargado. Dim: {self.model.config.hidden_size}{f' → PCA a {pca_dim}' if pca_dim else ''}"
        )

    def _generate_base_embeddings(self, texts: List[str]) -> np.ndarray:
        all_embeddings = []
        print(f"Generando embeddings BERT para {len(texts)} textos...")
        torch.set_grad_enabled(False)
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids, attention_mask = (
                encoded["input_ids"].to(self.device),
                encoded["attention_mask"].to(self.device),
            )
            outputs = self.model(input_ids, attention_mask=attention_mask)
            all_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
        torch.set_grad_enabled(True)
        return np.vstack(all_embeddings)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        base_embeddings = self._generate_base_embeddings(texts)
        if not self.pca_dim:
            return base_embeddings
        print(f"Ajustando PCA para reducir a {self.pca_dim} dims...")
        self.pca = PCA(n_components=self.pca_dim, random_state=self.random_state)
        reduced_embeddings = self.pca.fit_transform(base_embeddings)
        print(
            f"PCA ajustado. Varianza explicada: {np.sum(self.pca.explained_variance_ratio_):.4f}"
        )
        return reduced_embeddings

    def transform(self, texts: List[str]) -> np.ndarray:
        base_embeddings = self._generate_base_embeddings(texts)
        if not self.pca_dim:
            return base_embeddings
        if self.pca is None:
            raise RuntimeError("PCA no ajustado. Llama a fit_transform() primero.")
        print("Aplicando PCA ajustado...")
        return self.pca.transform(base_embeddings)
